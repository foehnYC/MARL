import torch
import torch.nn as nn
import torch.nn.functional as F


class QmixNet(nn.Module):
    """"
    Mixing Net in QMix.
    """
    def __init__(self, state_shape, hidden_shape, n_agents, device):
        # state = (n_episode, episode_limit, state_shape)
        # q_vals = (n_episode, episode_limit, n_agents)
        # q_total = (n_episode, episode_limit)
        super().__init__()

        self.state_shape = state_shape
        self.hidden_shape = hidden_shape
        self.n_agents = n_agents

        self.w1layer = nn.Linear(self.state_shape, self.n_agents * self.hidden_shape)
        self.b1layer = nn.Linear(self.state_shape, self.hidden_shape)
        self.w2layer = nn.Linear(self.state_shape, self.hidden_shape)
        self.b2layer = nn.Sequential(
            nn.Linear(self.state_shape, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, 1)
        )

        self.to(device)

    def forward(self, state, q_vals):
        n_episode = q_vals.shape[0]
        q_vals = q_vals.reshape(-1, 1, self.n_agents)
        state = state.reshape(-1, self.state_shape)
        w1 = torch.abs(self.w1layer(state)).reshape(-1, self.n_agents, self.hidden_shape)
        b1 = self.b1layer(state).reshape(-1, 1, self.hidden_shape)
        w2 = torch.abs(self.w2layer(state)).reshape(-1, self.hidden_shape, 1)
        b2 = self.b2layer(state).reshape(-1, 1, 1)

        x = F.elu(torch.bmm(q_vals, w1) + b1)
        q_tot = torch.bmm(x, w2) + b2
        q_tot = q_tot.reshape(n_episode, -1)
        return q_tot
