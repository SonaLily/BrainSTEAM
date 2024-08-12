import os

import numpy as np
import pandas as pd

from nilearn import datasets

from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

dataset_path = '/data/HCP_1200'
corr_matrices_dir = f'{dataset_path}/corr_matrices'
pcorr_matrices_dir = f'{dataset_path}/pcorr_matrices'
avg_pcorr_file = f'{dataset_path}/avg_pcorr.csv'
time_series_dir = f'{dataset_path}/time_series'
phenotypic_file = f'{dataset_path}/HCP_1200.csv'
sub_labels_file = f'{dataset_path}/sub_labels.csv'

labels_all_file = f'{dataset_path}/labels_all.csv'
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
labels_agegender_all_df =  phenotypic_df[['subject','sex','age']]
labels_age_df =  phenotypic_df[['subject','age']]
labels_gender_df =  phenotypic_df[['subject','sex']]
gender_map = {'F' : 1, 'M' : 0}
labels_gender_df['sex'] = labels_gender_df['sex'].map(gender_map)
labels_gender_m_df=pd.DataFrame()

corr_measure = ConnectivityMeasure(kind='correlation')
pcorr_measure = ConnectivityMeasure(kind='partial correlation')

fPath= '/data/HCP_1200/time_series/'
tmp_flist = os.listdir(fPath)


fmrdatadict = {}
subjectid_list = []
site_list = []
time_series = []
corr_matrices = []
pcorr_matrices = []
i=0
for fi in tmp_flist:

        if fi.endswith('.csv'):

            ts = np.loadtxt(fPath+fi, delimiter=',')

            subject_id = fi.split('_')[-1].split('.')[0]

            subjectid_list.append(subject_id)

            time_series.append(ts.transpose())


            corr_matrices = corr_measure.fit_transform(time_series)
            pcorr_matrices = pcorr_measure.fit_transform(time_series)
            np.savetxt(f'{corr_matrices_dir}/corr_{subject_id}.csv', corr_matrices[i], delimiter=',')
            np.savetxt(f'{pcorr_matrices_dir}/pcorr_{subject_id}.csv', pcorr_matrices[i], delimiter=',')



            i=i+1

subject_list = [int(i) for i in subjectid_list]
labels_gender_m_df = labels_gender_df[labels_gender_df['subject'].isin(subject_list)]

np.savetxt(labels_gender_file, labels_gender_m_df, delimiter=',')
np.savetxt(labels_age_file, labels_age_df, delimiter=',')
            
