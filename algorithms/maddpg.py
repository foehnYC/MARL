import torch
import torch.nn.functional as F
import numpy as np
from networks.basenet import DNN, RNN


class Maddpg(object):
    """"
    Maddpg policy class.
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
        self.tau = args.tau
        self.grad_norm_clip = args.grad_norm_clip
        self.target_update_cycle = args.target_update_cycle

        self.critic = DNN(self.state_shape + self.obs_shape, self.hidden_shape, self.n_actions, self.device)
        self.target_critic = DNN(self.state_shape + self.obs_shape, self.hidden_shape, self.n_actions, self.device)
        self.actor = RNN(self.obs_shape, self.hidden_shape, self.n_actions, self.device)
        self.target_actor = RNN(self.obs_shape, self.hidden_shape, self.n_actions, self.device)

        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=1e-4)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=1e-4)

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
        actions_probs, hidden_next = self.actor(obs, hidden)
        actions_probs = actions_probs.reshape(self.n_envs, self.n_agents, self.n_actions)
        actions_probs[avail_actions == 0] = -9999
        actions = []
        for i in range(self.n_envs):
            for j in range(self.n_agents):
                if explore and np.random.uniform() < self.eps:
                    avail_actions_idx = torch.nonzero(avail_actions[i, j])
                    action_idx = torch.randint(0, len(avail_actions_idx), [1])
                    action = avail_actions_idx[action_idx]
                else:
                    action = torch.argmax(actions_probs[i, j])
                actions.append(action)
        actions = torch.tensor(actions).reshape(self.n_envs, self.n_agents)
        self.eps = max(self.min_eps, self.eps - self.delta_eps)
        return actions, hidden_next

    def update(self, buffer, episode):
        """"
        Update network parameters using episodic data sampled from buffer.
        """
        if episode % (self.target_update_cycle // self.n_envs) == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_actor.load_state_dict(self.actor.state_dict())
        train_info = dict()
        train_info['loss_critic'] = 0
        train_info['loss_actor'] = 0
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

            n_episode = state.shape[0]
            state = state.unsqueeze(dim=2).repeat(1, 1, self.n_agents, 1)
            hidden = torch.zeros([n_episode * self.n_agents, self.hidden_shape], device=self.device)
            hidden_target = torch.zeros([n_episode * self.n_agents, self.hidden_shape], device=self.device)

            q_evals, q_targets, actions_log_probs, actions_target = [], [], [], []
            for i in range(self.episode_limit):
                state_input = state[:, i, :, :].reshape(-1, self.state_shape)
                obs_input = obs[:, i, :, :].reshape(-1, self.obs_shape)
                avail_actions_input = avail_actions[:, i, :, :].reshape(-1, self.n_actions)

                critic_input = torch.cat([state_input, obs_input], dim=-1)
                q_eval = self.critic(critic_input)
                q_target = self.target_critic(critic_input)

                output, hidden = self.actor(obs_input, hidden)
                output_target, hidden_target = self.target_actor(obs_input, hidden_target)
                output[avail_actions_input == 0] = -9999
                output_target[avail_actions_input == 0] = -9999

                action_log_probs = F.log_softmax(output, dim=-1)
                action_target = torch.argmax(output_target, dim=-1)

                q_evals.append(q_eval.reshape(n_episode, self.n_agents, self.n_actions))
                q_targets.append(q_target.reshape(n_episode, self.n_agents, self.n_actions))
                actions_log_probs.append(action_log_probs.reshape(n_episode, self.n_agents, self.n_actions))
                actions_target.append(action_target.reshape(n_episode, self.n_agents))
            q_evals = torch.stack(q_evals, dim=1)
            q_eval = torch.gather(q_evals, dim=3, index=actions.unsqueeze(-1)).squeeze(-1)
            actions_log_probs = torch.stack(actions_log_probs, dim=1)
            action_log_probs = torch.gather(actions_log_probs, dim=3, index=actions.unsqueeze(-1)).squeeze(-1)
            actions_target = torch.stack(actions_target, dim=1)
            q_targets = torch.stack(q_targets, dim=1)
            q_target = torch.gather(q_targets, dim=3, index=actions_target.unsqueeze(-1)).squeeze(-1)
            q_target = torch.cat([q_target, torch.zeros([n_episode, 1, self.n_agents], device=self.device)], dim=1)
            q_target = q_target[:, 1:, :]

            reward_agent = reward.unsqueeze(dim=2).repeat(1, 1, self.n_agents)
            target = (reward_agent + self.gamma * q_target).detach()
            td_error = (target - q_eval).sum(dim=2) * (1 - mask)
            loss_critic = (td_error ** 2).sum() / (1 - mask).sum()
            train_info['loss_critic'] += loss_critic.item()

            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
            self.critic_optimizer.step()

            mask_agent = mask.unsqueeze(dim=-1).repeat(1, 1, self.n_agents)
            loss_actor = -(q_eval.detach() * action_log_probs * (1 - mask_agent)).sum() / (1 - mask).sum()
            train_info['loss_actor'] += loss_actor.item()

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
            self.actor_optimizer.step()

            # def soft_load(dst, src, tau):
            #     for dst_param, src_param in zip(dst.parameters(), src.parameters()):
            #         dst_param.copy_(tau * src_param.data + (1 - tau) * dst_param.data)
            # with torch.no_grad():
            #     soft_load(self.target_critic, self.critic, self.tau)

        for key in train_info.keys():
            train_info[key] /= self.n_envs
        return train_info

    def save(self, model_path):
        torch.save(self.target_actor.state_dict(), model_path + '/maddpg_actor_params.pkl')
        torch.save(self.target_critic.state_dict(), model_path + '/maddpg_critic_params.pkl')
