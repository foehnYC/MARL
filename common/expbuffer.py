import numpy as np


class ExpBuffer(object):
    """"ExpBuffer class to save and sample running experiences"""
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.episode_limit = args.episode_limit
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.state = np.zeros([self.buffer_size, self.episode_limit, self.state_shape])
        self.obs = np.zeros([self.buffer_size, self.episode_limit, self.n_agents, self.obs_shape])
        self.avail_actions = np.zeros([self.buffer_size, self.episode_limit, self.n_agents, self.n_actions])
        self.actions = np.zeros([self.buffer_size, self.episode_limit, self.n_agents])
        self.reward = np.zeros([self.buffer_size, self.episode_limit])
        self.terminal = np.ones([self.buffer_size, self.episode_limit])
        self.mask = np.ones([self.buffer_size, self.episode_limit])

        self.mini_batch_size = args.mini_batch_size
        self.iteration = args.n_envs

        self.idx = 0
        self.pointer = 0

    def insert(self, exp):
        """"
        insert one episode experience into buffer
        :param exp: (list) experience containing state, obs, avail_actions, actions, reward, and terminal information
        """
        state, obs, avail_actions, actions, reward, terminal = exp
        length = self.episode_limit - len(terminal)

        state = np.concatenate((state, np.zeros([length, self.state_shape])), axis=0)
        obs = np.concatenate((obs, np.zeros([length, self.n_agents, self.obs_shape])), axis=0)
        avail_actions_stuff = np.zeros([length, self.n_agents, self.n_actions])
        avail_actions_stuff[:, :, 0] = 1
        avail_actions = np.concatenate((avail_actions, avail_actions_stuff), axis=0)
        actions = np.concatenate((actions, np.zeros([length, self.n_agents])), axis=0)
        reward = np.concatenate((reward, np.zeros([length])), axis=0)
        terminal = np.concatenate((terminal, np.ones([length])), axis=0)

        self.state[self.idx] = state
        self.obs[self.idx] = obs
        self.avail_actions[self.idx] = avail_actions
        self.actions[self.idx] = actions
        self.reward[self.idx] = reward
        self.terminal[self.idx] = terminal
        self.mask[self.idx] = np.append(np.zeros([1]), terminal)[:-1]

        self.idx = (self.idx + 1) % self.buffer_size
        self.pointer = min(self.pointer + 1, self.buffer_size)

    def data_generator(self):
        """"
        Yielding episodic data for training
        """
        for _ in range(self.iteration):
            idx = np.random.randint(0, self.pointer, min(self.pointer, self.mini_batch_size)).squeeze()
            state_batch = self.state[idx]
            obs_batch = self.obs[idx]
            avail_actions_batch = self.avail_actions[idx]
            actions_batch = self.actions[idx]
            reward_batch = self.reward[idx]
            terminal_batch = self.terminal[idx]
            mask_batch = self.mask[idx]
            yield state_batch, obs_batch, avail_actions_batch, \
                  actions_batch, reward_batch, terminal_batch, mask_batch
