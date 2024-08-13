import os
import torch
from torch import nn
import numpy as np
import util
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import logging

import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)

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


def dense_to_ind_val(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = (torch.isnan(adj) == 0).nonzero(as_tuple=True)
    edge_attr = adj[index]

    return torch.stack(index, dim=0), edge_attr

class BrainDatasetst(InMemoryDataset):
    def __init__(self, root,train_data_batch_dev,train_label_batch_dev, corr_file_path ='', transform=None, pre_transform=None, neighbors=10):
        self.neighbors = neighbors
        self.train_data_batch_dev = train_data_batch_dev
        self.corr_file_path = corr_file_path
        self.train_label_batch_dev = train_label_batch_dev
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def dense_to_ind_val(adj):
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)

        index = (torch.isnan(adj) == 0).nonzero(as_tuple=True)
        edge_attr = adj[index]

        return torch.stack(index, dim=0), edge_attr

    def process(self):

        graphs = []
        num_graphs = 0
        for subject in range(0,self.train_data_batch_dev.size(0)):
            tb = self.train_data_batch_dev[subject, 0, :, :, 0]
            tb = tb.unsqueeze(0)
            time_series = tb.cpu().detach().numpy()
            corr_measure = ConnectivityMeasure(kind='correlation')
            pcorr_measure = ConnectivityMeasure(kind='partial correlation')

            corr_matrices = corr_measure.fit_transform(time_series)
            pcorr_matrices = pcorr_measure.fit_transform(time_series)
            corr_matrices = np.squeeze(corr_matrices, axis = 0)
            pcorr_matrices = np.squeeze(pcorr_matrices, axis=0)
            label = self.train_label_batch_dev[subject].type(torch.LongTensor)


            pcorr_matrix_np = pcorr_matrices

            index = np.abs(pcorr_matrix_np).argsort(axis=1)

            num_graphs = subject + 1
            num_nodes = self.train_data_batch_dev.size(3)

            a = np.zeros((num_graphs, num_nodes, num_nodes))
            a[subject] = pcorr_matrix_np

            adj = torch.Tensor(a)
            edge_index2, edge_attr = dense_to_ind_val(adj[subject])
            adj = torch.sparse_coo_tensor(edge_index2, edge_attr, [num_nodes, num_nodes])
            adj = adj.to_dense()
            edge_attr = torch.unsqueeze(edge_attr, -1)
            pcorr_matrix_nx = from_numpy_matrix(pcorr_matrix_np)
            pcorr_matrix_data = from_networkx(pcorr_matrix_nx)

            corr_matrix_np = corr_matrices
            num_nodes = corr_matrix_np.shape[0]
            pcorr_matrix_data.x = torch.tensor(corr_matrix_np).float()
            pcorr_matrix_data.y = label.type(torch.LongTensor)
            pcorr_matrix_data.edge_attr = edge_attr
            graphs.append(pcorr_matrix_data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


