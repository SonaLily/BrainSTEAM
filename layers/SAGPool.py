
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch

import torch.nn as nn
import numpy as np

from typing import Union, Optional, Callable
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import softmax

import math


import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,
                 ratio=0.8,
                 Conv=GCNConv,
                 non_linearity=torch.tanh,
                 cus_drop_ratio= 0):

        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.dropout = torch.nn.Dropout(cus_drop_ratio)


    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze(-1)

        sc = self.dropout(score)

        perm = topk(sc, self.ratio, batch)
        x_ae = x[perm]
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, x_ae