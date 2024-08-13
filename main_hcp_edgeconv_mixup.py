from pathlib import Path
import argparse
from timeit import repeat
import yaml
import torch
from models import GraphTransformer, DecTransformer,DynamicGNN
from train import BrainGNNTrain
from Brainedgeconv_mixupTrain import *
from datetime import datetime
from dataloader import init_dataloader
from dataloader2 import BrainDataset
import wandb

import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import logging
from util import logger

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


def main(args):
   
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

        dataset = BrainDataset('hcpBraindataset_pyg6')

        dataset = dataset.shuffle()

        train_share = int(len(dataset) * 0.8)

        train_dataset = dataset[:train_share]
        test_dataset = dataset[train_share:]

        batch_size = 32  # 64 -t58 32 -t64 -t57 62

        loss_name = 'loss'


        wandb.config = {
            "learning_rate": config['train']['lr'],
            "epochs": config['train']['epochs'],
            "batch_size": config['data']['batch_size'],
            "dataset": config['data']["dataset"],
            "model": config['model']['type']
        }


        now = datetime.now()

        date_time = now.strftime("%m-%d-%H-%M-%S")

        wandb.run.name = f"{date_time}_{config['data']['dataset']}_{config['model']['type']}"

        extractor_type = config['model']['extractor_type'] if 'extractor_type' in config['model'] else "none"
        embedding_size = config['model']['embedding_size'] if 'embedding_size' in config['model'] else "none"
        window_size = config['model']['window_size'] if 'window_size' in config['model'] else "none"

        if "graph_generation" in config['model'] and config['model']["graph_generation"]:
            model_name = f"{config['train']['method']}_{config['model']['graph_generation']}"
        else:
            model_name = f"{config['train']['method']}"

        save_folder_name = Path(config['train']['log_folder'])/Path(
            date_time +
            f"_{config['data']['dataset']}_{config['model']['type']}_{model_name}" 
            + f"_{extractor_type}_{loss_name}_{embedding_size}_{window_size}")


        learning_rate = 10 ** -4
        n_epochs = 5000 # 300 k=10
        batch_size = 32  # 64 -t58 32 -t64 -t57 62

        model = DynamicGNN(batch_size)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        criterion = torch.nn.NLLLoss()

        best_model, train_result, train_metrics = brainedge_mixuptrain(model, train_loader, criterion, optimizer, n_epochs)

        train_results_df, train_metrics = test(best_model, train_loader, criterion)
        test_results_df, test_metrics = test(best_model, test_loader, criterion)
        print(test_results_df)
        print(test_metrics)
        train_metrics_df=pd.DataFrame.from_dict([train_metrics])
        test_metrics_df=pd.DataFrame.from_dict([test_metrics])
        FinalTestAccuracy=test_metrics_df['accuracy']*100
        FinalTestAccuracy=int(FinalTestAccuracy)

        printcsv(FinalTestAccuracy, learning_rate, n_epochs, batch_size, train_metrics_df, test_metrics_df,
                 test_results_df,best_model)

        modelPath = '/home/sonalin/tech/sr2023/data/models/'
        extension = ".pt"

        FinalTestAccuracy = test_metrics_df['accuracy'] * 100
        FinalTestAccuracy = int(FinalTestAccuracy)
        FinalTestAccuracy_str=str(FinalTestAccuracy)
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        Model_file = modelPath + FinalTestAccuracy_str + '_2hcpMixupDynamic39k10' + timestr + extension


        torch.save(best_model.state_dict(), Model_file)
        torch.save(model.state_dict(),Model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',
                        default='/home/sonalin/tech/braindata/config/config_abide/',
                        type=str,
                        help='folders containing Configuration filename for training the model.')
    parser.add_argument('--config_filename', default='/home/sonalin/tech/gnnxf_al/setting/topk/mwd_dec_ortho_learnable_newpool_sftmx_topk_90_4_hierarchical_sum_5-4lr.yaml', type=str,
                        help='Configuration filename for training the model.')

    parser.add_argument('--repeat_time', default=5, type=int)
    parser.add_argument('--wandb', default="sonalily", type=str)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    config_dir=args.config_dir

    sl_data_tmp = os.listdir(config_dir)
    sl_data = []
    for i in sl_data_tmp:
        if i.endswith('.yaml'):
            sl_data.append(config_dir + i)

    sl_data_pos = 1
    for sl_data_i in sl_data:
        args.config_filename = sl_data_i

        yaml_name = Path(args.config_filename).name
        print(f'Processing config #{sl_data_pos} : {yaml_name}')
        dataset_name = yaml_name.split("_")[0]
        tags = [f"{dataset_name}_project"]
        other_tags = yaml_name.split(".")[0].split("_")
        tags.extend(other_tags)

        for i in range(args.repeat_time):
            wandb.login(key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            run = wandb.init(project="brain_hcp_edgeconv", entity=args.wandb, reinit=True,
                             group=yaml_name, tags=tags)
            main(args)
            run.finish()
        sl_data_pos += 1