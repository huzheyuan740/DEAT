import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, obs_one_dim, act_one_dim, hidden_layers_size):
        super(PolicyNetwork, self).__init__()

        # torch.set_default_dtype(torch.float)

        self.obs_one_dim = obs_one_dim
        self.act_one_dim = act_one_dim
        self.hidden_layers_size = hidden_layers_size

        self.hidden_layers = nn.ModuleList()
        size_in = self.obs_one_dim
        for size_out in self.hidden_layers_size:
            fully_connected_layer = nn.Linear(size_in, size_out)
            size_in = size_out
            self.hidden_layers.append(fully_connected_layer)

        self.output_layer = nn.Linear(size_in, self.act_one_dim)

    def forward(self, x):
        # torch.set_default_dtype(torch.float)
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


class CriticNetwork(nn.Module):
    def __init__(self, obs_one_dim, act_one_dim, hidden_layers_size, agent_num):
        super(CriticNetwork, self).__init__()

        torch.set_default_dtype(torch.float)

        self.obs_one_dim = obs_one_dim
        self.act_one_dim = act_one_dim
        self.hidden_layers_size = hidden_layers_size
        self.agent_num = agent_num

        self.obs_all_dim = self.obs_one_dim * self.agent_num
        self.act_all_dim = self.act_one_dim * self.agent_num

        self.hidden_layers = nn.ModuleList()
        size_in = self.obs_all_dim + self.act_all_dim
        for size_out in self.hidden_layers_size:
            fully_connected_layer = nn.Linear(size_in, size_out)
            size_in = size_out
            self.hidden_layers.append(fully_connected_layer)

        self.output_layer = nn.Linear(size_in, 1)

    def forward(self, obs, act):
        # torch.set_default_dtype(torch.float)
        x = torch.cat((obs, act), dim=1)
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x
