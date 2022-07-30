import os
import torch
import numpy as np
from algorithms import REGISTRY
from common.expbuffer import ExpBuffer
from tensorboardX import SummaryWriter


class Runner(object):
    """"Runner class to perform training, evaluating, and data collecting."""
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.result_path = os.path.join('./results/', args.map_name, args.alg)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.model_path = os.path.join('./models/', args.map_name, args.alg)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.writer = SummaryWriter(self.result_path)
        self.policy = REGISTRY[args.alg](self.args)
        self.buffer = ExpBuffer(self.args)

    def train(self):
        """"training process."""
        self.warmup()
        print('start training.')
        eval_times = 0
        for episode in range(self.args.n_episodes * self.args.q_steps // self.args.n_envs):
            if episode % (self.args.evaluate_cycle // self.args.n_envs) == 0:
                average_reward, average_win_rate = self.evaluate()
                self.writer.add_scalar('episodic reward', average_reward, eval_times)
                self.writer.add_scalar('win rate', average_win_rate, eval_times)
                eval_times += 1
                self.policy.save(self.model_path)
                print('{} in {} evaluation {} reward:{} winrate:{}'.format(
                    self.args.alg, self.args.map_name, eval_times, average_reward, average_win_rate))

            train_info = self.policy.update(self.buffer, episode)
            for key, value in train_info.items():
                self.writer.add_scalar(key, value, episode)
            self.episode_generator(True)
        self.env.close()
        print('end training.')

    def evaluate(self):
        """"evaluating process."""
        total_reward, total_win_num = 0, 0
        for _ in range(self.args.evaluate_epoch // self.args.n_envs):
            episode_reward, win_num = self.episode_generator(False)
            total_reward += episode_reward
            total_win_num += win_num
        average_reward = total_reward / self.args.evaluate_epoch
        average_win_rate = total_win_num / self.args.evaluate_epoch
        return average_reward, average_win_rate

    def warmup(self):
        """"stuff experience buffer with initialized policy."""
        for _ in range(self.args.mini_batch_size):
            self.episode_generator(True)

    @torch.no_grad()
    def episode_generator(self, explore):
        """"
        Generate one episode data.
        :param explore: whether to explore in choosing actions.
        """
        hidden = np.zeros([self.args.n_agents * self.args.n_envs, self.args.hidden_shape])
        win_flags = [False for _ in range(self.args.n_envs)]
        state_batch, obs_batch, avail_actions_batch, actions_batch, \
        reward_batch, terminal_batch = [], [], [], [], [], []
        self.env.reset()
        terminated = self.env.get_terminal_flag()
        while not terminated:
            state = self.env.get_state()
            obs = self.env.get_obs()
            avail_actions = self.env.get_avail_actions()
            actions, hidden = self.policy.act(obs, hidden, avail_actions, explore)
            # actions, hidden = self.policy.act_q(state, obs, hidden, avail_actions, explore)
            actions = actions.detach().cpu().numpy()
            hidden = hidden.detach().cpu().numpy()
            reward, terminal, infos = self.env.step(actions)
            terminated = self.env.get_terminal_flag()

            for idx, info in enumerate(infos):
                if 'battle_won' in info and info['battle_won']:
                    win_flags[idx] = True

            state_batch.append(state)
            obs_batch.append(obs)
            avail_actions_batch.append(avail_actions)
            actions_batch.append(actions)
            reward_batch.append(reward)
            terminal_batch.append(terminal)

        if explore:
            for idx in range(self.args.n_envs):
                state = np.array(state_batch)[:, idx]
                obs = np.array(obs_batch)[:, idx]
                avail_actions = np.array(avail_actions_batch)[:, idx]
                actions = np.array(actions_batch)[:, idx]
                reward = np.array(reward_batch)[:, idx]
                terminal = np.array(terminal_batch)[:, idx]
                exp = state, obs, avail_actions, actions, reward, terminal
                self.buffer.insert(exp)

        episode_reward = np.array(reward_batch).sum()
        win_num = np.array(win_flags).sum()
        return episode_reward, win_num
