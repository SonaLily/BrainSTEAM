import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import logging

from torch_geometric.utils import degree, to_dense_adj
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)
from layers import SAGPool

from nilearn import datasets
from nilearn import plotting
from nilearn.input_data import NiftiMapsMasker,NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure

from sklearn.model_selection import StratifiedKFold

from torch import Tensor
from torch.nn import Parameter, Linear,BatchNorm1d, ModuleList
from typing import Union, Tuple, Optional
from torch_geometric.typing import Size, OptTensor
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing, GATConv
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch import nn
import torch_geometric

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv,TransformerConv, TopKPooling,DynamicEdgeConv
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx

from copy import deepcopy
import pandas as pd

import networkx as nx
from networkx.convert_matrix import from_numpy_matrix

from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score


class DynamicGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        num_nodes = 39
        num_classes = 2

        self.mlp1 = Sequential(Linear(2 * num_nodes, hidden_channels), ReLU())
        self.mlp2 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
        self.mlp3 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())


        k = 10  # 5
        self.conv1 = DynamicEdgeConv(self.mlp1, k, aggr='max')
        self.conv2 = DynamicEdgeConv(self.mlp2, k, aggr='max')
        self.conv3 = DynamicEdgeConv(self.mlp3, k, aggr='max')

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, batch)
        x = self.bn1(x)
        x = F.relu(x)


        x = self.conv2(x, batch)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, batch)
        x = self.bn3(x)

        x = global_mean_pool(x, batch)


        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x

class DynamicGNNdec2(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            num_nodes = 39
            num_classes = 2
            self.nhid = hidden_channels
            self.pooling_ratio = 0.8  # args.pooling_ratio
            self.dropout_ratio = 0  # args.dropout_ratio
            self.cus_drop_ratio = 0  # args.cus_drop_ratio

            self.mlp1 = Sequential(Linear(2 * num_nodes, hidden_channels), ReLU())
            self.mlp2 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
            self.mlp3 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())


            k = 10  # 5
            self.conv1 = DynamicEdgeConv(self.mlp1, k, aggr='max')
            self.pool1 = SAGPool(self.nhid,
                                 ratio=self.pooling_ratio,
                                 cus_drop_ratio=self.cus_drop_ratio)
            self.conv2 = DynamicEdgeConv(self.mlp2, k, aggr='max')
            self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio,
                                 cus_drop_ratio=self.cus_drop_ratio)
            self.conv3 = DynamicEdgeConv(self.mlp3, k, aggr='max')
            self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio,
                                 cus_drop_ratio=self.cus_drop_ratio)

            self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn3 = torch.nn.BatchNorm1d(hidden_channels)


            self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
            self.lin3 = torch.nn.Linear(hidden_channels // 2, num_classes)


            self.conv4 = DynamicEdgeConv(self.mlp3, k, aggr='max')
            self.conv5 = DynamicEdgeConv(self.mlp2, k, aggr='max')
            self.conv6 = DynamicEdgeConv(self.mlp1, k, aggr='max')


            self.lin4 = torch.nn.Linear(self.nhid, self.nhid)
            self.lin5 = torch.nn.Linear(self.nhid, self.nhid // 2)
            self.lin6 = torch.nn.Linear(self.nhid // 2, 1)

        def forward(self, data):

            x, edge_index, batch = data.x, data.edge_index, data.batch


            degree_ground_truth = degree(edge_index[0], num_nodes=x.size(0)).to(x.device)


            x = self.conv1(x, batch)
            x = self.bn1(x)
            x = F.relu(x)

            res = x
            x, edge_index, _, batch, perm_1, x_ae1 = self.pool1(x, edge_index, None, batch)


            x_out = torch.zeros_like(res)
            x_out[perm_1] = x


            x_degree = F.relu(self.lin4(x_out))
            x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
            x_degree = F.relu(self.lin5(x_degree))
            x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
            x_degree = F.relu(self.lin6(x_degree))

            degree_ground_truth_1 = degree_ground_truth[perm_1]
            degree_predict_1 = x_degree[perm_1]

            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


            x = self.conv2(x, batch)
            x = self.bn2(x)
            x = F.relu(x)

            res_2 = x
            x, edge_index, _, batch, perm_2, x_ae2 = self.pool2(x, edge_index, None, batch)

            x_out = torch.zeros_like(res_2)
            x_out[perm_2] = x

            x_out_2 = torch.zeros_like(res)
            x_out_2[perm_1] = x_out



            x_degree_2 = F.relu(self.lin4(x_out_2))
            x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
            x_degree_2 = F.relu(self.lin5(x_degree_2))
            x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
            x_degree_2 = F.relu(self.lin6(x_degree_2))

            degree_ground_truth_2 = degree_ground_truth[perm_2]
            degree_predict_2 = x_degree[perm_2]

            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


            x = self.conv3(x, batch)
            x = self.bn3(x)
            x = F.relu(x)

            res_3 = x
            x, edge_index, _, batch, perm_3, x_ae3 = self.pool3(x, edge_index, None, batch)

            x_out = torch.zeros_like(res_3)
            x_out[perm_3] = x

            x_out_3 = torch.zeros_like(res_2)
            x_out_3[perm_2] = x_out



            x_degree_3 = F.relu(self.lin4(x_out_3))
            x_degree_3 = F.dropout(x_degree_3, p=self.dropout_ratio, training=self.training)
            x_degree_3 = F.relu(self.lin5(x_degree_3))
            x_degree_3 = F.dropout(x_degree_3, p=self.dropout_ratio, training=self.training)
            x_degree_3 = F.relu(self.lin6(x_degree_3))

            degree_ground_truth_3 = degree_ground_truth[perm_3]
            degree_predict_3 = x_degree[perm_3]

            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)



            x = F.relu(self.lin1(x))
            # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = F.relu(self.lin2(x))
            x = self.lin3(x)

            return x, degree_ground_truth_1, degree_predict_1, degree_ground_truth_2, degree_predict_2, degree_ground_truth_3, degree_predict_3



class DynamicGNNdec(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        num_nodes = 39

        num_classes = 2
        self.nhid = hidden_channels
        self.pooling_ratio = 0.8  # args.pooling_ratio
        self.dropout_ratio = 0  # args.dropout_ratio
        self.cus_drop_ratio = 0  # args.cus_drop_ratio

        self.mlp1 = Sequential(Linear(2 * num_nodes, hidden_channels), ReLU())
        self.mlp2 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
        self.mlp3 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
        self.mlp6 = Sequential(torch.nn.Linear(2 * hidden_channels, num_nodes), ReLU())


        k = 10  # 5
        self.conv1 = DynamicEdgeConv(self.mlp1, k, aggr='max')
        self.pool1 = SAGPool(self.nhid,
                             ratio=self.pooling_ratio,
                             cus_drop_ratio=self.cus_drop_ratio)
        self.conv2 = DynamicEdgeConv(self.mlp2, k, aggr='max')
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio,
                             cus_drop_ratio=self.cus_drop_ratio)
        self.conv3 = DynamicEdgeConv(self.mlp3, k, aggr='max')
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio,
                             cus_drop_ratio=self.cus_drop_ratio)


        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)


        self.lin1 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = torch.nn.Linear(hidden_channels // 2, num_classes)


        self.conv4 = DynamicEdgeConv(self.mlp3, k, aggr='max')
        self.conv5 = DynamicEdgeConv(self.mlp2, k, aggr='max')
        self.conv6 = DynamicEdgeConv(self.mlp6, k, aggr='max')


        self.lin4 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin5 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin6 = torch.nn.Linear(self.nhid // 2, 1)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch1=batch
        degree_ground_truth = degree(edge_index[0], num_nodes=x.size(0)).to(x.device)


        x = self.conv1(x, batch)
        x = self.bn1(x)
        x = F.relu(x)

        res = x
        x, edge_index, _, batch, perm_1, x_ae1 = self.pool1(x, edge_index, None, batch)


        x_out = torch.zeros_like(res)
        x_out[perm_1] = x

        x_decoder = torch.tanh(self.conv4(x_out, batch1))

        x_decoder = torch.tanh(self.conv5(x_decoder, batch1))
        x_decoder_1 = self.conv6(x_decoder, batch1)


        x_degree = F.relu(self.lin4(x_out))
        x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
        x_degree = F.relu(self.lin5(x_degree))
        x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
        x_degree = F.relu(self.lin6(x_degree))

        degree_ground_truth_1 = degree_ground_truth[perm_1]
        degree_predict_1 = x_degree[perm_1]

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


        x = self.conv2(x, batch)
        x = self.bn2(x)
        x = F.relu(x)

        res_2 = x
        x, edge_index, _, batch, perm_2, x_ae2 = self.pool2(x, edge_index, None, batch)

        x_out = torch.zeros_like(res_2)
        x_out[perm_2] = x

        x_out_2 = torch.zeros_like(res)
        x_out_2[perm_1] = x_out

        x_decoder = torch.tanh(self.conv4(x_out_2, batch1))
        x_decoder = torch.tanh(self.conv5(x_decoder, batch1))
        x_decoder_2 = self.conv6(x_decoder, batch1)

        x_degree_2 = F.relu(self.lin4(x_out_2))
        x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
        x_degree_2 = F.relu(self.lin5(x_degree_2))
        x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
        x_degree_2 = F.relu(self.lin6(x_degree_2))

        degree_ground_truth_2 = degree_ground_truth[perm_2]
        degree_predict_2 = x_degree[perm_2]

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


        x = self.conv3(x, batch)
        x = self.bn3(x)
        x = F.relu(x)

        res_3 = x
        x, edge_index, _, batch, perm_3, x_ae3 = self.pool3(x, edge_index, None, batch)

        x_out = torch.zeros_like(res_3)
        x_out[perm_3] = x

        x_out_3 = torch.zeros_like(res)
        x_out_3[perm_2] = x_out

        x_decoder = torch.tanh(self.conv4(x_out_3, batch1))
        x_decoder = torch.tanh(self.conv5(x_decoder, batch1))
        x_decoder_3 = self.conv6(x_decoder, batch1)

        x_degree_3 = F.relu(self.lin4(x_out_3))
        x_degree_3 = F.dropout(x_degree_3, p=self.dropout_ratio, training=self.training)
        x_degree_3 = F.relu(self.lin5(x_degree_3))
        x_degree_3 = F.dropout(x_degree_3, p=self.dropout_ratio, training=self.training)
        x_degree_3 = F.relu(self.lin6(x_degree_3))

        degree_ground_truth_3 = degree_ground_truth[perm_3]
        degree_predict_3 = x_degree[perm_3]

        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3


        x = F.relu(self.lin1(x))

        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = F.softmax(x, dim=1)

        return x, x_decoder_1, x_decoder_2,  x_decoder_3, degree_ground_truth_1, degree_predict_1, degree_ground_truth_2, degree_predict_2, degree_ground_truth_3, degree_predict_3
