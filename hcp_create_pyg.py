
import os
import numpy as np


import torch
torch.manual_seed(42)

from nilearn import datasets

from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx


from networkx.convert_matrix import from_numpy_matrix



dataset_path = '/data/HCP_1200'
corr_matrices_dir = f'{dataset_path}/corr_matrices'
pcorr_matrices_dir = f'{dataset_path}/pcorr_matrices'
avg_pcorr_file = f'{dataset_path}/avg_pcorr.csv'
time_series_dir = f'{dataset_path}/time_series'

labels_file = f'{dataset_path}/labels_gender.csv'
labels = torch.from_numpy(np.loadtxt(labels_file, delimiter=','))
len(labels)
keys= labels[:,0] 
keys = [int(i) for i in keys]
keys = [str(i) for i in keys]

corr_path_list = sorted(os.listdir(corr_matrices_dir), key=lambda x: int(x[-8:-4]))
corr_matrix_path = os.path.join(corr_matrices_dir, corr_path_list[0])
corr_matrices = torch.from_numpy(np.loadtxt(corr_matrix_path, delimiter=','))

atlas = datasets.fetch_atlas_msdl()
atlas_filename = atlas.maps
atlas_labels = atlas.labels

corr_matrix_np = np.loadtxt(corr_matrix_path, delimiter=',')
pcorr_sublist = []
pcorr_path_list = sorted(os.listdir(pcorr_matrices_dir), key=lambda x: int(x[-8:-4]))

for i in range(0, len(pcorr_path_list)):

   subid=pcorr_path_list[i].split('_')[-1].split('.')[0]
   pcorr_sublist.append(subid)

main_list = list(set(pcorr_sublist) - set(keys))
main_list = list( set(keys) - set(pcorr_sublist) )

def dense_to_ind_val(adj):
  
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = (torch.isnan(adj)==0).nonzero(as_tuple=True)
    edge_attr = adj[index]

    return torch.stack(index, dim=0), edge_attr


class BrainDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, neighbors=10):
        self.neighbors = neighbors
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

        print('data before')

        dataset_path = '/data/HCP_1200'
        labels_file = f'{dataset_path}/labels_gender.csv'

        corr_matrices_dir = f'{dataset_path}/corr_matrices'
        pcorr_matrices_dir = f'{dataset_path}/pcorr_matrices'

        
        labels = torch.from_numpy(np.loadtxt(labels_file, delimiter=','))

        corr_path_list = sorted(os.listdir(corr_matrices_dir), key=lambda x: int(x[-10:-4]))
        pcorr_path_list = sorted(os.listdir(pcorr_matrices_dir), key=lambda x: int(x[-10:-4]))


        graphs = []
        
        for i in range(0, len(corr_path_list)):

 
            corr_matrix_path = os.path.join(corr_matrices_dir, corr_path_list[i])
            pcorr_matrix_path = os.path.join(pcorr_matrices_dir, pcorr_path_list[i])


            pcorr_matrix_np = np.loadtxt(pcorr_matrix_path, delimiter=',')


            n_rois = pcorr_matrix_np.shape[0]
            num_graphs = i + 1
            num_nodes = n_rois

            a = np.zeros((num_graphs, num_nodes, num_nodes))
            a[i] = pcorr_matrix_np

            adj = torch.Tensor(a)
            edge_index2, edge_attr = dense_to_ind_val(adj[i])
            adj = torch.sparse_coo_tensor(edge_index2, edge_attr, [num_nodes, num_nodes])
            edge_attr = torch.unsqueeze(edge_attr, -1)
            pcorr_matrix_nx = from_numpy_matrix(pcorr_matrix_np)
            pcorr_matrix_data = from_networkx(pcorr_matrix_nx)
            print('pcorr_matrix_data')
            print(pcorr_matrix_data)


            corr_matrix_np = np.loadtxt(corr_matrix_path, delimiter=',')
            pcorr_matrix_data.x = torch.tensor(corr_matrix_np).float()

            pcorr_matrix_data.y = labels[i].type(torch.LongTensor)
            pcorr_matrix_data.edge_attr = edge_attr

            graphs.append(pcorr_matrix_data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

dataset = BrainDataset('hcpBraindataset_pyg3')

dataset.data


