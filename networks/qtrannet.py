import torch
import torch.nn as nn
import torch.nn.functional as F


class QtranQnet(nn.Module):
    """"
    Q_joint net in Qtran.
    """
    def __init__(self, state_shape, hidden_shape, n_agents, device):
        # state = (n_episode, episode_limit, state_shape)
        # action = (n_episode, episode_limit, n_agents)
        # q_joint = (n_episode, episode_limit)
        super().__init__()

        self.state_shape = state_shape
        self.hidden_shape = hidden_shape
        self.n_agents = n_agents
        sapair_shape = self.state_shape + self.n_agents

        self.sa_encoder = nn.Sequential(
            nn.Linear(sapair_shape, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, self.hidden_shape // 2)
        )
        self.h_encoder = nn.Sequential(
            nn.Linear(self.hidden_shape * self.n_agents, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, self.hidden_shape // 2)
        )
        self.q_encoder = nn.Sequential(
            nn.Linear(self.hidden_shape, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, 1)
        )

        self.to(device)

    def forward(self, state, action, hidden):
        n_episode = state.shape[0]
        state = state.reshape(-1, self.state_shape)
        action = action.reshape(-1, self.n_agents)
        hidden = hidden.reshape(-1, self.hidden_shape * self.n_agents)
        sapair = torch.cat([state, action], dim=1)
        sa_hidden = self.sa_encoder(sapair)
        h_hidden = self.h_encoder(hidden)
        sah_hidden = torch.cat([sa_hidden, h_hidden], dim=1)
        q = self.q_encoder(sah_hidden)
        q = q.reshape(n_episode, -1)
        return q


class QtranVnet(nn.Module):
    """"
    V_joint net in Qtran.
    """
    def __init__(self, state_shape, hidden_shape, n_agents, device):
        # state = (n_episode, episode_limit, state_shape)
        # v_joint = (n_episode, episode_limit)
        super().__init__()

        self.state_shape = state_shape
        self.hidden_shape = hidden_shape
        self.n_agents = n_agents

        self.s_encoder = nn.Sequential(
            nn.Linear(self.state_shape, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, self.hidden_shape // 2)
        )
        self.h_encoder = nn.Sequential(
            nn.Linear(self.hidden_shape * self.n_agents, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, self.hidden_shape // 2)
        )
        self.v_encoder = nn.Sequential(
            nn.Linear(self.hidden_shape, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, 1)
        )

        self.to(device)

    def forward(self, state, hidden):
        n_episode = state.shape[0]
        state = state.reshape(-1, self.state_shape)
        hidden = hidden.reshape(-1, self.hidden_shape * self.n_agents)
        s_hidden = self.s_encoder(state)
        h_hidden = self.h_encoder(hidden)
        sh_hidden = torch.cat([s_hidden, h_hidden], dim=1)
        v = self.v_encoder(sh_hidden)
        v = v.reshape(n_episode, -1)
        return v
