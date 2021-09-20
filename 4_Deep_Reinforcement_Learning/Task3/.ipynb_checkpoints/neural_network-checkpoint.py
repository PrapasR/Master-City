import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(9)  # Reproducibility

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# %% Constructing Actor Critic Neural Network Architectures

class ActorCriticNN(nn.Module):

    def __init__(self, input_size, n_layers, hidden_size, output_size):
        super().__init__()
        layers = []
        if output_size == 1:  # Critic NN
            for _ in range(n_layers):
                if len(layers) == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, output_size))

        else:  # Actor NN
            for _ in range(n_layers):
                if len(layers) == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, output_size))
            layers.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*layers)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(device)
        return self.model(obs)
