import os

import numpy as np
import pandas as pd

from nilearn import datasets

from nilearn.input_data import NiftiMapsMasker,NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure

dataset_path = '/data/HCP_1200'
corr_matrices_dir = f'{dataset_path}/corr_matrices'
pcorr_matrices_dir = f'{dataset_path}/pcorr_matrices'
avg_pcorr_file = f'{dataset_path}/avg_pcorr.csv'
time_series_dir = f'{dataset_path}/time_series'
phenotypic_file = f'{dataset_path}/HCP_1200.csv'
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

labels_age_df =  phenotypic_df[['subject','age']]
labels_gender_df =  phenotypic_df[['subject','sex']]
gender_map = {'F' : 1, 'M' : 0}
labels_gender_df['sex'] = labels_gender_df['sex'].map(gender_map)

np.savetxt(labels_gender_file, labels_gender_df, delimiter=',')

corr_measure = ConnectivityMeasure(kind='correlation')
pcorr_measure = ConnectivityMeasure(kind='partial correlation')


fPath = '/data/HCP_1200/hcp_tc_npy_22/'
tmp_flist = os.listdir(fPath)


fmrdatadict = {}
subjectid_list = []
site_list = []
time_series = []
i=0
for fi in tmp_flist:

        if fi.endswith('.npy'):
            ts = np.load(fPath+fi);
            subject_id = fi.split('_')[0]

            subjectid_list.append(subject_id)

            time_series.append(ts.transpose())

            corr_matrices = corr_measure.fit_transform(time_series)
            pcorr_matrices = pcorr_measure.fit_transform(time_series)

            avg_pcorr_matrix = np.mean(pcorr_matrices, axis=0)
            np.savetxt(avg_pcorr_file, avg_pcorr_matrix, delimiter=',')



    
            i=i+1

np.save(f'{dataset_path}/hcp_gender.npy',
                    {'timeseires': np.array(time_series), "label_gender": np.array(labels_gender_df),"corr": np.array(corr_matrices),
                     "pcorr": np.array(pcorr_matrices), 'site': np.array(site_list)})
np.savetxt(labels_gender_file, labels_gender_df, delimiter=',')
            
