import torch
import torch.nn as nn
from .components import MHSA as TransformerEncoderLayer
from torch.nn.functional import batch_norm
from torch.nn import LSTM


class GraphTransformer(nn.Module):

    def __init__(self, model_config, node_num=360):

        super().__init__()
        self.attention_list = nn.ModuleList()
        self.readout = model_config['readout']
        self.node_num = node_num

        for _ in range(model_config["self_attention_layer"]):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=node_num, nhead=4, dim_feedforward=1024,
                                        batch_first=True)
            )


        final_dim = node_num

        if self.readout == "concat":

            self.dim_reduction = nn.Sequential(
                nn.Linear(node_num, 8),
                nn.LeakyReLU()
            )
            final_dim = 8 * node_num

        elif self.readout == "sum":
            self.norm = nn.BatchNorm1d(node_num)

        elif self.readout == "lstm":
            self.lstm = LSTM(input_size=node_num, hidden_size=node_num, num_layers=3, batch_first=True,
                             bidirectional=True)
            final_dim = 2 * node_num

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):

        bz, _, _, = x.shape

        for atten in self.attention_list:
            x = atten(x)

        if self.readout == "concat":
            x = self.dim_reduction(x)
            x = x.reshape((bz, -1))

        elif self.readout == "mean":
            x = torch.mean(x, dim=-1)
        elif self.readout == "max":
            x, _ = torch.max(x, dim=-1)
        elif self.readout == "sum":
            x = torch.sum(x, dim=-1)
            x = self.norm(x)
        elif self.readout == "lstm":

            hidden_state = nn.Parameter(torch.zeros(3 * 2, x.shape[0], self.node_num)).cuda()
            cell_state = nn.Parameter(torch.zeros(3 * 2, x.shape[0], self.node_num)).cuda()
            x, (_, _) = self.lstm(x, (hidden_state, cell_state))
            x = x[:, 0, :]

        return self.fc(x)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]