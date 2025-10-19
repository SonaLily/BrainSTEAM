import torch
import numpy as np
import random
import scipy


def mixup_data(x, nodes, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_nodes, y_a, y_b, lam




def mixup_renn_data(x, log_nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = []
    for i, j in enumerate(index):
        mixed_nodes.append(torch.matrix_exp(lam * log_nodes[i] + (1 - lam) * log_nodes[j]))

    mixed_nodes = torch.stack(mixed_nodes)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_nodes, y_a, y_b, lam



def mixup_data_by_class(x, nodes, y, alpha=1.0, device='cuda'):

    mix_xs, mix_nodes, mix_ys = [], [], []

    for t_y in y.unique():
        idx = y == t_y

        t_mixed_x, t_mixed_nodes, _, _, _ = mixup_data(
            x[idx], nodes[idx], y[idx], alpha=alpha, device=device)
        mix_xs.append(t_mixed_x)
        mix_nodes.append(t_mixed_nodes)

        mix_ys.append(y[idx])

    return torch.cat(mix_xs, dim=0), torch.cat(mix_nodes, dim=0), torch.cat(mix_ys, dim=0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cal_step_connect(connectity, step):
    multi_step = connectity
    for _ in range(step):
        multi_step = np.dot(multi_step, connectity)
    multi_step[multi_step > 0] = 1
    return multi_step



def obtain_partition(dataloader, fc_threshold, step=2):
    pearsons = []
    for data_in, pearson, label in dataloader:
        pearsons.append(pearson)

    fc_data = torch.mean(torch.cat(pearsons), dim=0)

    fc_data[fc_data > fc_threshold] = 1
    fc_data[fc_data <= fc_threshold] = 0

    _, n = fc_data.shape

    final_partition = torch.zeros((n, (n-1)*n//2))

    connection = cal_step_connect(fc_data, step)
    temp = 0
    for i in range(connection.shape[0]):
        temp += i
        for j in range(i):
            if connection[i, j] > 0:
                final_partition[i, temp-i+j] = 1
                final_partition[j, temp-i+j] = 1

    connect_num = torch.sum(final_partition > 0)/n
    print(f'Final Partition {connect_num}')

    return final_partition.cuda().float(), connect_num

def dense_to_ind_val(adj):
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)

        index = (torch.isnan(adj) == 0).nonzero(as_tuple=True)
        edge_attr = adj[index]

        return torch.stack(index, dim=0), edge_attr

def mixup_data2(data, alpha=1.0, use_cuda=False):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)

        else:
            lam = 1
        batch_size = len(data.batch)

        y = data.y
        x = data.x
        edge_index = data.edge_index

        W1_rows, W1_columns = data.x.size()
        W1_edgeIndex_rows, W1_edgeIndex_columns = data.edge_index.size()

        mixed_x = torch.zeros([W1_rows, W1_columns])
        mixed_edge_index = torch.zeros([W1_edgeIndex_rows, W1_edgeIndex_columns], dtype=torch.long)

        y_index = torch.randperm(len(y))

        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            y_index = torch.randperm(len(y))
            x_index = torch.randperm(W1_columns)
            edge_index_index = torch.randperm(W1_edgeIndex_columns)

        for i in range(W1_rows):
            mixed_x[i] = lam * x[i] + (1 - lam) * x[i][x_index]
        for i in range(W1_edgeIndex_rows):
            mixed_edge_index[i] = lam * edge_index[i] + (1 - lam) * edge_index[i][edge_index_index]

        y_a, y_b = y, y[y_index]
        return mixed_x, mixed_edge_index, y_a, y_b, lam


