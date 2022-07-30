import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.init_param_()
        self.to(device)

    def forward(self, state):
        q = self.fc(state)
        return q

    def init_param_(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.init_param_()
        self.to(device)

    def forward(self, state, hidden):
        x = F.relu(self.fc1(state))
        h = self.gru(x, hidden)
        q = self.fc2(h)
        return q, h

    def init_param_(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
