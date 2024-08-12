
import os

import numpy as np
import pandas as pd

from nilearn import datasets

from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure


dataset_path ='/home/sonalin/tech/braindata/abide39'
corr_matrices_dir = f'{dataset_path}/corr_matrices'
pcorr_matrices_dir = f'{dataset_path}/pcorr_matrices'
avg_pcorr_file = f'{dataset_path}/avg_pcorr.csv'
time_series_dir = f'{dataset_path}/time_series'
phenotypic_file = f'{dataset_path}/Phenotypic_V1_0b_preprocessed1.csv'
sub_labels_file = f'{dataset_path}/sub_labels.csv'

labels_file = f'{dataset_path}/labels.csv'
labels_age_file = f'{dataset_path}/labels_age.csv'
labels_gender_file = f'{dataset_path}/labels_gender.csv'

atlas = datasets.fetch_atlas_msdl()
atlas_filename = atlas.maps
atlas_labels = atlas.labels

masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache')


labels = []
labels_age = []
labels_gender = []


phenotypic_df = pd.read_csv(phenotypic_file)

sublabel_df = pd.read_csv(sub_labels_file)
label_df= sublabel_df[['SUB_ID','DX_GROUP']]
labels_age_df =  phenotypic_df[['SUB_ID','AGE_AT_SCAN']]
labels_gender_df =  phenotypic_df[['SUB_ID','SEX']]

corr_measure = ConnectivityMeasure(kind='correlation')
pcorr_measure = ConnectivityMeasure(kind='partial correlation')


fPath = '/home/sonalin/tech/sr2023/data/asdall/ABIDE_pcp/cpac/filt_global/'
tmp_flist = os.listdir(fPath)


fmrdatadict = {}
id_list = []
site_list = []
time_series = []
i=0
for fi in tmp_flist:
    if os.path.isfile(fPath+fi):
        if fi.endswith('.gz'):
            site_list = fi.split('_')[0]
            sub_id = (((fi.split('_func_'))[0]).split('_')[-1]).lstrip('0')

            id_list.append(sub_id)

            fi= os.path.join(fPath, fi)
            print(fi)  

            ts = masker.fit_transform(fi)

            time_series.append(ts) 

            np.savetxt(f'{time_series_dir}/time_series_{sub_id}.csv', ts, delimiter=',')
         

            corr_matrices = corr_measure.fit_transform(time_series)
            pcorr_matrices = pcorr_measure.fit_transform(time_series)
            np.savetxt(f'{corr_matrices_dir}/corr_{sub_id}.csv', corr_matrices[i], delimiter=',')
            np.savetxt(f'{pcorr_matrices_dir}/pcorr_{sub_id}.csv', pcorr_matrices[i], delimiter=',')


            avg_pcorr_matrix = np.mean(pcorr_matrices, axis=0)
            np.savetxt(avg_pcorr_file, avg_pcorr_matrix, delimiter=',')

    
            i=i+1

np.save(f'{dataset_path}/abide39asd.npy',
                    {'timeseires': np.array(time_series), "label": np.array(labels), "corr": np.array(corr_matrices),
                     "pcorr": np.array(pcorr_matrices), 'site': np.array(site_list)})

np.save(f'{dataset_path}/abide39all.npy',
                    {'timeseires': np.array(time_series), "sublabel": np.array(label_df),"label_age": np.array(labels_age_df), "label_gender": np.array(labels_gender_df),"corr": np.array(corr_matrices),
                     "pcorr": np.array(pcorr_matrices), 'site': np.array(site_list)})
            
