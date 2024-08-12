import numpy as np
import pandas as pd

from scipy import stats
from sklearn.model_selection import StratifiedKFold

import os

if __name__ == "__main__":

    ptfile = '/abide39/sub_labels.csv'
    pt_df = pd.read_csv(ptfile)

    labeldata = pt_df[['SUB_ID', 'DX_GROUP']].to_numpy()
    np.save('/abide39/labels.npy', labeldata)
    L = 130 #196 # 1000
    S = 0
    roi=39
    data = np.zeros((labeldata.shape[0],1,L,roi,1))
    label = np.zeros((labeldata.shape[0],))

    idx = 0
    data_all = None

    filename_full_path = '/abide39/time_series/'
    if(os.path.isdir(filename_full_path)):
        f_list_tmp = os.listdir(filename_full_path)
        for name_tmp_i in f_list_tmp:
            if name_tmp_i.endswith('.csv'):
                filename_full = filename_full_path + name_tmp_i

                data_df = pd.read_csv(filename_full)
                full_sequence = data_df.to_numpy()
                if full_sequence.shape[0] < S+L:
                    continue

                full_sequence = full_sequence[S:S + L,:]
                z_sequence = stats.zscore(full_sequence, axis=1)
                z_sequence2 = np.transpose(z_sequence)
        
                if data_all is None:
                   data_all = z_sequence2
                else:

                   if (z_sequence.shape[0]==130):
                     data_all = np.concatenate((data_all, z_sequence2), axis=1)

                     data[idx,0,:,:,0] = z_sequence

                     label[idx] = labeldata[idx,1]
                     idx = idx + 1
 

    n_regions = 39
    A = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i==j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data_all[i,:], data_all[j,:])[0][1]) # get value from corrcoef matrix
                A[j][i] = A[i][j]


    np.save('/abide39/adj_matrix.npy', A)



    data = data[:idx]
    label = label[:idx]
    print(data.shape)
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    fold = 1
    for train_idx, test_idx in skf.split(data, label):
        train_data = data[train_idx]
        train_label = label[train_idx]
        test_data = data[test_idx]
        test_label = label[test_idx] 

        filename = '/abide39/train_data_'+str(fold)+'.npy'
        np.save(filename,train_data)
        filename = '/abide39/train_label_'+str(fold)+'.npy'
        np.save(filename,train_label)
        filename = '/abide39/test_data_'+str(fold)+'.npy'
        np.save(filename,test_data)
        filename = '/abide39/test_label_'+str(fold)+'.npy'
        np.save(filename,test_label)
        fold = fold + 1
  
