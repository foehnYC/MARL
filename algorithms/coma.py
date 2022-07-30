import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from networks.basenet import DNN, RNN


class Coma(object):
    """"
    Coma policy class.
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
        self.td_lambda = args.td_lambda
        self.grad_norm_clip = args.grad_norm_clip
        self.target_update_cycle = args.target_update_cycle

        self.critic_input_shape = self.state_shape + self.obs_shape
        self.critic = DNN(self.critic_input_shape, self.hidden_shape, self.n_actions, self.device)
        self.target_critic = DNN(self.critic_input_shape, self.hidden_shape, self.n_actions, self.device)
        self.actor = RNN(self.obs_shape, self.hidden_shape, self.n_actions, self.device)
        self.target_actor = RNN(self.obs_shape, self.hidden_shape, self.n_actions, self.device)

        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=1e-3)
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
                    dist = Categorical(logits=actions_probs[i, j])
                    action = dist.sample()
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
        train_info['grad_critic'] = 0
        train_info['grad_actor'] = 0
        data_batchs = buffer.data_generator()
        for data_batch in data_batchs:
            state_batch, obs_batch, avail_actions_batch,\
            actions_batch, reward_batch, terminal_batch, mask_batch = data_batch

            state = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
            obs = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            avail_actions = torch.tensor(avail_actions_batch, device=self.device)
            actions = torch.tensor(actions_batch, dtype=torch.int64, device=self.device)
            reward = torch.tensor(reward_batch, device=self.device)
            terminal = torch.tensor(terminal_batch, device=self.device)
            mask = torch.tensor(mask_batch, device=self.device)

            n_episode = obs.shape[0]
            state = state.unsqueeze(dim=2).repeat(1, 1, self.n_agents, 1)
            # other_actions = \
            #     torch.zeros([n_episode, self.episode_limit, self.n_agents, self.n_agents - 1], device=self.device)
            # for i in range(self.n_agents):
            #     other_actions[:, :, i, :] = actions[:, :, torch.arange(self.n_agents) != i]

            hidden = torch.zeros([n_episode * self.n_agents, self.hidden_shape], device=self.device)
            hidden_target = torch.zeros([n_episode * self.n_agents, self.hidden_shape], device=self.device)
            output_dists, output_dists_target = [], []
            for i in range(self.episode_limit):
                input = obs[:, i, :, :].reshape(-1, self.obs_shape)
                output_dist, hidden = self.actor(input, hidden)
                output_dist_target, hidden_target = self.target_actor(input, hidden_target)
                output_dists.append(output_dist.reshape(n_episode, self.n_agents, self.n_actions))
                output_dists_target.append(output_dist_target.reshape(n_episode, self.n_agents, self.n_actions))
            output_dists = torch.stack(output_dists, dim=1)
            output_dists_target = torch.stack(output_dists_target, dim=1)
            output_dists[avail_actions == 0] = -9999
            output_dists_target[avail_actions == 0] = -9999
            actions_probs = F.softmax(output_dists_target, dim=-1)
            log_actions_probs = F.log_softmax(output_dists, dim=-1)
            log_action_probs = torch.gather(log_actions_probs, dim=3, index=actions.unsqueeze(-1)).squeeze(-1)

            critic_input = torch.cat([state, obs], dim=-1)
            critic_input = critic_input.reshape([-1, self.critic_input_shape])
            q_vals = self.critic(critic_input)
            q_targets = self.target_critic(critic_input)
            q_vals = q_vals.reshape([n_episode, self.episode_limit, self.n_agents, self.n_actions])
            q_targets = q_targets.reshape([n_episode, self.episode_limit, self.n_agents, self.n_actions])
            q_evals = torch.gather(q_vals, dim=3, index=actions.unsqueeze(-1)).squeeze(-1)
            baseline = (actions_probs * q_targets).sum(dim=3)

            td_targets = torch.cat([baseline, torch.zeros([n_episode, 1, self.n_agents], device=self.device)], dim=1)
            td_targets = td_targets[:, 1:, :]

            reward_agent = reward.unsqueeze(dim=2).repeat(1, 1, self.n_agents)
            td_error = torch.sum(td_targets.detach() + reward_agent - q_evals, dim=2) * (1 - mask)
            loss_critic = (td_error ** 2).sum() / (1 - mask).sum()
            train_info['loss_critic'] += loss_critic.item()

            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            critic_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0) for p in self.critic.parameters()]), 2.0)
            train_info['grad_critic'] += critic_norm.item()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
            self.critic_optimizer.step()

            mask_agent = mask.unsqueeze(dim=2).repeat(1, 1, self.n_agents)
            advantage = (q_evals - baseline) * (1 - mask_agent)
            loss_actor = -(advantage.detach() * log_action_probs).sum() / (1 - mask).sum()
            train_info['loss_actor'] += loss_actor.item()

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            actor_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0) for p in self.actor.parameters()]), 2.0)
            train_info['grad_actor'] += actor_norm.item()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
            self.actor_optimizer.step()

        for key in train_info.keys():
            train_info[key] /= self.n_envs
        return train_info

    def save(self, model_path):
        torch.save(self.target_actor.state_dict(), model_path + '/coma_actor_params.pkl')
        torch.save(self.target_critic.state_dict(), model_path + '/coma_critic_params.pkl')
