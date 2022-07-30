import torch
import numpy as np
from networks.basenet import RNN
from networks.qtrannet import QtranQnet, QtranVnet


class Qtran(object):
    """"
    Qtran policy class.
    """
    def __init__(self, args):
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.hidden_shape = args.hidden_shape
        self.n_envs = args.n_envs
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
        self.eval_qnet = QtranQnet(self.state_shape, self.hidden_shape, self.n_agents, self.device)
        self.target_qnet = QtranQnet(self.state_shape, self.hidden_shape, self.n_agents, self.device)
        self.eval_vnet = QtranVnet(self.state_shape, self.hidden_shape, self.n_agents, self.device)
        self.parameters = list(self.eval_net.parameters()) + \
                          list(self.eval_qnet.parameters()) + list(self.eval_vnet.parameters())
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
            self.target_qnet.load_state_dict(self.eval_qnet.state_dict())
        train_info = dict()
        train_info['loss_total'] = 0
        train_info['loss_td'] = 0
        train_info['loss_opt'] = 0
        train_info['loss_nopt'] = 0
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
            hidden_evals, hidden_targets = [], []
            for i in range(self.episode_limit):
                input = obs[:, i, :, :].reshape(-1, self.obs_shape)
                q_eval, hidden_eval = self.eval_net(input, hidden_eval)
                q_target, hidden_target = self.target_net(input, hidden_target)
                q_evals.append(q_eval.reshape(n_episode, self.n_agents, self.n_actions))
                q_targets.append(q_target.reshape(n_episode, self.n_agents, self.n_actions))
                hidden_evals.append(hidden_eval.reshape(n_episode, self.n_agents, self.hidden_shape))
                hidden_targets.append(hidden_target.reshape(n_episode, self.n_agents, self.hidden_shape))

            hidden_evals = torch.stack(hidden_evals, dim=1)
            hidden_targets = torch.stack(hidden_targets, dim=1)
            q_evals = torch.stack(q_evals, dim=1)
            q_evals_clone = q_evals.clone()
            q_targets = torch.stack(q_targets, dim=1)
            q_evals_act = torch.gather(q_evals_clone, dim=3, index=actions.unsqueeze(-1)).squeeze(-1)
            q_evals[avail_actions == 0] = -99999
            q_targets[avail_actions == 0] = -99999
            q_evals_opt, actions_opt = q_evals.max(dim=3)
            actions_opt_next = q_targets.max(dim=3)[1]

            q_evals_total = self.eval_qnet(state, actions, hidden_evals)
            q_targets_total = self.target_qnet(state, actions_opt_next, hidden_targets)
            y = reward[:, :-1] + self.gamma * q_targets_total[:, 1:] * (1 - terminal[:, 1:])
            td_error = (q_evals_total[:, :-1] - y.detach()) * (1 - mask[:, :-1])
            loss_td = (td_error ** 2).sum() / (1 - mask[:, :-1]).sum()

            q_evals_opt_sum = torch.sum(q_evals_opt, dim=2) * (1 - mask)
            q_evals_opt_hat = self.eval_qnet(state, actions_opt, hidden_evals) * (1 - mask)
            v = self.eval_vnet(state, hidden_evals) * (1 - mask)
            opt_error = q_evals_opt_sum - q_evals_opt_hat.detach() + v
            loss_opt = (opt_error ** 2).sum() / (1 - mask).sum()

            q_evals_act_sum = torch.sum(q_evals_act, dim=2) * (1 - mask)
            q_evals_act_hat = self.eval_qnet(state, actions, hidden_evals) * (1 - mask)
            nopt_error = q_evals_act_sum - q_evals_act_hat.detach() + v
            nopt_error = nopt_error.clamp(max=0)
            loss_nopt = (nopt_error ** 2).sum() / (1 - mask).sum()

            loss = loss_td + loss_opt + loss_nopt
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_norm_clip)
            self.optimizer.step()

            train_info['loss_td'] += loss_td.item()
            train_info['loss_opt'] += loss_opt.item()
            train_info['loss_nopt'] += loss_nopt.item()
            train_info['loss_total'] += loss.item()

        for key in train_info.keys():
            train_info[key] /= self.n_envs
        return train_info

    def save(self, model_path):
        torch.save(self.target_net.state_dict(), model_path + '/qtran_q_params.pkl')
        torch.save(self.target_qnet.state_dict(), model_path + '/qtran_Q_params.pkl')
        torch.save(self.eval_vnet.state_dict(), model_path + '/qtran_V_params.pkl')
