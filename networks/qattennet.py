import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class QattenNet(nn.Module):
    """"
    Mixing net in Qatten.
    """
    def __init__(self, state_shape, obs_shape, hidden_shape, n_agents, n_heads, device):
        # Qi = (batch_size, n_agents)
        # state = (batch_size, state_shape)
        # obs = (batch_size, n_agents, obs_shape)
        # Qh = (batch_size, 1)
        super().__init__()

        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.hidden_shape = hidden_shape
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.d = numpy.sqrt(hidden_shape // 2)

        self.Qembed_list = nn.ModuleList()
        for i in range(n_heads):
            self.Qembed_list.append(nn.Sequential(
            nn.Linear(self.state_shape, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, self.hidden_shape // 2)
            ))
        self.Kembed_list = nn.ModuleList()
        for i in range(n_heads):
            self.Kembed_list.append(nn.Sequential(
                nn.Linear(self.obs_shape, self.hidden_shape),
                nn.ReLU(),
                nn.Linear(self.hidden_shape, self.hidden_shape // 2)
            ))

        self.wlayer = nn.Sequential(
            nn.Linear(self.state_shape, self.hidden_shape),
            nn.ReLU(),
            nn.Linear(self.hidden_shape, self.n_heads)
        )
        self.blayer = nn.Sequential(
            nn.Linear(self.state_shape, self.hidden_shape // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_shape // 2, 1)
        )

        self.to(device)

    def forward(self, state, obs, q_vals):
        n_episodes, episode_limit = state.shape[0], state.shape[1]
        batch_size = n_episodes * episode_limit
        state = state.reshape(-1, self.state_shape)
        obs = obs.reshape(-1, self.obs_shape)
        q_vals = q_vals.reshape(-1, self.n_agents)

        lam_list = []
        for i in range(self.n_heads):
            # eq = (batch_size, 1, hidden_shape / 2)
            eq = self.Qembed_list[i](state)
            # ek = (batch_size, n_agents, hidden_shape / 2)
            ek = self.Kembed_list[i](obs)
            eq = eq.reshape(batch_size, 1, -1)
            ek = ek.reshape(batch_size, self.n_agents, -1)
            ek = ek.permute(0, 2, 1)
            lam = torch.matmul(eq, ek) / self.d
            lam = F.softmax(lam, dim=-1)
            lam_list.append(lam.squeeze(1))
        # lam_list = (batch_size, n_heads, n_agents)
        lam_list = torch.stack(lam_list, dim=1)
        # q_vals = (batch_size, n_agents, 1)
        q_vals = q_vals.unsqueeze(dim=-1)
        # q_h = (batch_size, n_heads, 1)
        q_h = torch.matmul(lam_list, q_vals)
        w_h = self.wlayer(state)
        w_h = w_h.unsqueeze(dim=1)
        b_h = self.blayer(state)
        q_tot = torch.matmul(w_h, q_h).squeeze(-1) + b_h
        q_tot = q_tot.reshape(n_episodes, -1)
        return q_tot
