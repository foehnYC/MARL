import torch
import numpy as np
from networks.basenet import RNN
from networks.qattennet import QattenNet


class Qatten(object):
    """"
    Qatten policy class.
    """
    def __init__(self, args):
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.hidden_shape = args.hidden_shape
        self.n_envs = args.n_envs
        self.n_heads = args.n_heads
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.episode_limit = args.episode_limit
        self.device = args.device

        self.eps = args.eps
        self.min_eps = args.min_eps
        self.delta_eps = args.delta_eps
        self.gamma = args.gamma
        self.grad_norm_clip = args.grad_norm_clip
        self.target_update_cycle = args.target_update_cycle

        self.eval_net = RNN(self.obs_shape, self.hidden_shape, self.n_actions, self.device)
        self.target_net = RNN(self.obs_shape, self.hidden_shape, self.n_actions, self.device)
        self.eval_mixnet = \
            QattenNet(self.state_shape, self.obs_shape, self.hidden_shape, self.n_agents, self.n_heads, self.device)
        self.target_mixnet = \
            QattenNet(self.state_shape, self.obs_shape, self.hidden_shape, self.n_agents, self.n_heads, self.device)
        self.parameters = list(self.eval_net.parameters()) + list(self.eval_mixnet.parameters())
        self.optimizer = torch.optim.RMSprop(self.parameters, lr=args.lr)

    def act(self, obs, hidden, avail_actions, explore):
        """"
        Compute actions and hidden state in one transition.
        :param obs: partical observation.
        :param hidden: last hidden state.
        :param avail_actions: available actions.
        :param explore: whether to randomly choose other actions for exploration.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        hidden = torch.tensor(hidden, dtype=torch.float32, device=self.device)
        avail_actions = torch.tensor(avail_actions, device=self.device)
        obs = obs.reshape(-1, self.obs_shape)
        q_eval, hidden_next = self.eval_net(obs, hidden)
        q_eval = q_eval.reshape(self.n_envs, self.n_agents, self.n_actions)
        q_eval[avail_actions == 0] = -9999
        actions = []
        for i in range(self.n_envs):
            for j in range(self.n_agents):
                if explore and np.random.uniform() < self.eps:
                    avail_actions_idx = torch.nonzero(avail_actions[i, j])
                    action_idx = torch.randint(0, len(avail_actions_idx), [1])
                    action = avail_actions_idx[action_idx]
                else:
                    action = torch.argmax(q_eval[i, j])
                actions.append(action)
        actions = torch.tensor(actions).reshape(self.n_envs, self.n_agents)
        self.eps = max(self.min_eps, self.eps - self.delta_eps)
        return actions, hidden_next

    def update(self, buffer, episode):
        """"
        Update network parameters using episodic data sampled from buffer.
        """
        if episode % (self.target_update_cycle // self.n_envs) == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.target_mixnet.load_state_dict(self.eval_mixnet.state_dict())
        train_info = dict()
        train_info['loss'] = 0
        data_batchs = buffer.data_generator()
        for data_batch in data_batchs:
            state_batch, obs_batch, avail_actions_batch, \
            actions_batch, reward_batch, terminal_batch, mask_batch = data_batch

            state = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
            obs = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            avail_actions = torch.tensor(avail_actions_batch, device=self.device)
            actions = torch.tensor(actions_batch, dtype=torch.int64, device=self.device)
            reward = torch.tensor(reward_batch, device=self.device)
            terminal = torch.tensor(terminal_batch, device=self.device)
            mask = torch.tensor(mask_batch, device=self.device)

            n_episode = obs.shape[0]
            hidden_eval = torch.zeros([n_episode * self.n_agents, self.hidden_shape], device=self.device)
            hidden_target = torch.zeros([n_episode * self.n_agents, self.hidden_shape], device=self.device)

            q_evals, q_targets = [], []
            for i in range(self.episode_limit):
                input = obs[:, i, :, :].reshape(-1, self.obs_shape)
                q_eval, hidden_eval = self.eval_net(input, hidden_eval)
                q_target, hidden_target = self.target_net(input, hidden_target)
                q_evals.append(q_eval.reshape(n_episode, self.n_agents, self.n_actions))
                q_targets.append(q_target.reshape(n_episode, self.n_agents, self.n_actions))
            q_evals = torch.stack(q_evals, dim=1)
            q_targets = torch.stack(q_targets, dim=1)
            q_targets[avail_actions == 0] = -9999
            q_eval = torch.gather(q_evals, dim=3, index=actions.unsqueeze(-1)).squeeze(-1)
            q_target = q_targets.max(dim=3)[0]

            q_eval_sum = self.eval_mixnet(state[:, :-1], obs[:, :-1], q_eval[:, :-1])
            q_target_sum = self.target_mixnet(state[:, 1:], obs[:, 1:], q_target[:, 1:])
            target = reward[:, :-1] + self.gamma * q_target_sum * (1 - terminal[:, 1:])
            td_error = (target.detach() - q_eval_sum) * (1 - mask[:, :-1])

            loss = (td_error ** 2).sum() / (1 - mask[:, :-1]).sum()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.grad_norm_clip)
            self.optimizer.step()

            train_info['loss'] += loss.item()
        train_info['loss'] /= self.n_envs
        return train_info

    def save(self, model_path):
        torch.save(self.target_net.state_dict(), model_path + '/qatten_q_params.pkl')
        torch.save(self.target_mixnet.state_dict(), model_path + '/qatten_mix_params.pkl')
