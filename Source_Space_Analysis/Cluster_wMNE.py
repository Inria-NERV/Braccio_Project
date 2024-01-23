import numpy as np
import glob
import os.path as op
from os import listdir
import matplotlib.pyplot as plt
from scipy import signal,stats,fft,sparse
from statsmodels.stats.anova import AnovaRM

from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from collections import Counter

from scipy.stats import wilcoxon
import mne
from scipy.spatial.distance import cdist
from mne.datasets import somato
from mne.time_frequency import tfr_morlet, csd_tfr
from mne.beamformer import make_dics, apply_dics_tfr_epochs
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
from mne.datasets import fetch_fsaverage
import nibabel
import imageio
from scipy.spatial import distance
from difflib import SequenceMatcher
def perm_t_stat(X):
    mean = X.mean(0)
    t_values, pv = stats.wilcoxon(X)
    return t_values


# eng = matlab.engine.start_matlab()

def channel_generator(number_of_channel, Ground, Ref):
    if number_of_channel == 32:
        electrodes = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
        for i in range(len(electrodes)):
            if (electrodes[i] == Ground):
                index_gnd = i
            if (electrodes[i] == Ref):
                index_ref = i
        electrodes[index_gnd] = 'AFz'
        electrodes[index_ref] = 'FCz'

    if number_of_channel == 64:
        #electrodes = ['FP1','FP2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FT9','FT7','FC3','FC4','FT8','FT10','C5','C1','C2','C6','TP7','CP3','CPz','CP4','TP8','P5','P1','P2','P6','PO7','PO3','POz','PO4','PO8']
        electrodes = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2','Iz']
        for i in range(len(electrodes)):
            if (electrodes[i] == Ground):
                index_gnd = i
            if (electrodes[i] == Ref):
                index_ref = i
        electrodes[index_gnd] = 'FCz'
        electrodes[index_ref] = 'Fpz'

    return electrodes

# The files live in:
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
biosemi_montage_inter = mne.channels.make_standard_montage('standard_1020')

parc = "aparc"
fmin = 13.
fmax = 26.

labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)
#Actual_connections = ['unknown-lh', 'bankssts-lh', 'caudalanteriorcingulate-lh', 'caudalmiddlefrontal-lh', 'corpuscallosum-lh', 'cuneus-lh', 'entorhinal-lh', 'fusiform-lh', 'inferiorparietal-lh', 'inferiortemporal-lh', 'isthmuscingulate-lh', 'lateraloccipital-lh', 'lateralorbitofrontal-lh', 'lingual-lh', 'medialorbitofrontal-lh', 'middletemporal-lh', 'parahippocampal-lh', 'paracentral-lh', 'parsopercularis-lh', 'parsorbitalis-lh', 'parstriangularis-lh', 'pericalcarine-lh', 'postcentral-lh', 'posteriorcingulate-lh', 'precentral-lh', 'precuneus-lh', 'rostralanteriorcingulate-lh', 'rostralmiddlefrontal-lh', 'superiorfrontal-lh', 'superiorparietal-lh', 'superiortemporal-lh', 'supramarginal-lh', 'frontalpole-lh', 'temporalpole-lh', 'transversetemporal-lh', 'insula-lh', 'unknown-rh', 'bankssts-rh', 'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-rh', 'corpuscallosum-rh', 'cuneus-rh', 'entorhinal-rh', 'fusiform-rh', 'inferiorparietal-rh', 'inferiortemporal-rh', 'isthmuscingulate-rh', 'lateraloccipital-rh', 'lateralorbitofrontal-rh', 'lingual-rh', 'medialorbitofrontal-rh', 'middletemporal-rh', 'parahippocampal-rh', 'paracentral-rh', 'parsopercularis-rh', 'parsorbitalis-rh', 'parstriangularis-rh', 'pericalcarine-rh', 'postcentral-rh', 'posteriorcingulate-rh', 'precentral-rh', 'precuneus-rh', 'rostralanteriorcingulate-rh', 'rostralmiddlefrontal-rh', 'superiorfrontal-rh', 'superiorparietal-rh', 'superiortemporal-rh', 'supramarginal-rh', 'frontalpole-rh', 'temporalpole-rh', 'transversetemporal-rh', 'insula-rh']
Actual_connections = [
    'postcentral-lh', 'precentral-lh', 'postcentral-rh', 'precentral-rh',
    'inferiorparietal-lh', 'superiorparietal-lh', 'inferiorparietal-rh', 'superiorparietal-rh',
    'caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh', 'rostralmiddlefrontal-lh', 'rostralmiddlefrontal-rh',
    'superiorfrontal-lh', 'superiorfrontal-rh', 'frontalpole-lh', 'frontalpole-rh'
]

src_2 = mne.read_source_spaces(src)
Distance_centers =[]
Names_Labels = []
for i in range(len(labels_parc)):
    center_vertex = labels_parc[i].center_of_mass(subject,subjects_dir=subjects_dir)
    #hemi = labels_parc[i].hemi
    position = src_2[0]['rr'][center_vertex]
    Distance_centers.append(position)
    Names_Labels.append(labels_parc[i].name)

Index_name=[]
Position_reorder = []
for Nem in Actual_connections:
    for k in range(len(Names_Labels)):
        if Nem == Names_Labels[k]:
            Index_name.append(k)
            Position_reorder.append(Distance_centers[k])

print(Names_Labels)
print(Actual_connections)
indices_not_in_list2 = [index for index, element in enumerate(Actual_connections) if element not in Names_Labels]


Distance_centers = Position_reorder
# Define a threshold distance for adjacency
threshold_distance = 0.04  # Adjust this value as needed

# Compute the pairwise distances between points
distances = distance.cdist(np.array(Distance_centers), np.array(Distance_centers))

# Create an adjacency matrix
adjacency_matrix = sparse.csr_matrix((distances <= threshold_distance).astype(int))


electrodes = channel_generator(64, 'TP9', 'TP10')


# Create a dataframe of your data
import pandas as pd
fres = 1
fs = 500


n_observations = 15
pval =0.01 # arbitrary
df = n_observations - 1  # degrees of freedom for the test
thresh = stats.t.ppf(1 - pval / 2, df)

print(thresh)

ind = [i for (i, channel) in enumerate(biosemi_montage_inter.ch_names) if channel in electrodes]
biosemi_montage = biosemi_montage_inter.copy()
# Keep only the desired channels
biosemi_montage.ch_names = [biosemi_montage_inter.ch_names[x] for x in ind]

kept_channel_info = [biosemi_montage_inter.dig[x+3] for x in ind]
# Keep the first three rows as they are the fiducial points information
biosemi_montage.dig = biosemi_montage_inter.dig[0:3]+kept_channel_info
    #biosemi_montage = mne.channels.make_standard_montage('standard_1020')
n_channels = len(biosemi_montage.ch_names)
fake_info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=fs/2,
                            ch_types='eeg')

rng = np.random.RandomState(0)
data = rng.normal(size=(n_channels, 1)) * 1e-6
fake_evoked = mne.EvokedArray(data, fake_info)
fake_evoked.set_montage(biosemi_montage)

print(fake_info)

print(biosemi_montage.get_positions())

pos = np.stack([biosemi_montage.get_positions()['ch_pos'][ch] for ch in electrodes])



electrode_Cortex = ['FC3','FC1','FC5','FCz','FC6','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']

Index_electrode =[]
test = False
for i in range(len(electrodes)):
    for j in electrode_Cortex:
        if electrodes[i] == j:
            test = True
            print(j)
            break
        else:
            test = False
    if test == True:
        Index_electrode.append(1)
    else:
        Index_electrode.append(0)

Index_electrode = np.array(Index_electrode)


directory_path_Strat_1_cali = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/source_space_multitaper/Cali/1/*'
Strat_1_cali = sorted(glob.glob(directory_path_Strat_1_cali))
Dummy_stc = mne.read_source_estimate(Strat_1_cali[0])
#
# ## Loading
# directory_path_Strat_1_cali = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Cali/1/*'
# directory_path_Strat_2_cali = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Cali/2/*'
# directory_path_Strat_3_cali = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Cali/3/*'
#
#
# directory_path_Strat_1_drive1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Dr1/1/*'
# directory_path_Strat_2_drive1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Dr1/2/*'
# directory_path_Strat_3_drive1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Dr1/3/*'
#
#
# directory_path_Strat_1_drive2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Dr2/1/*'
# directory_path_Strat_2_drive2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Dr2/2/*'
# directory_path_Strat_3_drive2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_connectivity/Dr2/3/*'
#
#
# Strat_1_cali = sorted(glob.glob(directory_path_Strat_1_cali))
# Strat_2_cali = sorted(glob.glob(directory_path_Strat_2_cali))
# Strat_3_cali = sorted(glob.glob(directory_path_Strat_3_cali))
#
# Strat_1_drive1 = sorted(glob.glob(directory_path_Strat_1_drive1))
# Strat_2_drive1 = sorted(glob.glob(directory_path_Strat_2_drive1))
# Strat_3_drive1 = sorted(glob.glob(directory_path_Strat_3_drive1))
#
# Strat_1_drive2 = sorted(glob.glob(directory_path_Strat_1_drive2))
# Strat_2_drive2 = sorted(glob.glob(directory_path_Strat_2_drive2))
# Strat_3_drive2 = sorted(glob.glob(directory_path_Strat_3_drive2))
#
# Strat_1_cali_ld = []
# for i in range(len(Strat_1_cali)):
#     Strat_1_cali_ld.append(np.load(Strat_1_cali[i]))
# Strat_1_cali_mat = np.array(Strat_1_cali_ld)
# print(Strat_1_cali_mat.shape)
# Strat_2_cali_ld = []
# for i in range(len(Strat_2_cali)):
#     Strat_2_cali_ld.append(np.load(Strat_2_cali[i]))
# Strat_2_cali_mat = np.array(Strat_2_cali_ld)
# print(Strat_2_cali_mat.shape)
# Strat_3_cali_ld = []
# for i in range(len(Strat_3_cali)):
#     Strat_3_cali_ld.append(np.load(Strat_3_cali[i]))
# Strat_3_cali_mat = np.array(Strat_3_cali_ld)
# print(Strat_3_cali_mat.shape)
# Strat_1_drive1_ld = []
# for i in range(len(Strat_1_drive1)):
#     Strat_1_drive1_ld.append(np.load(Strat_1_drive1[i]))
# Strat_1_drive1_mat = np.array(Strat_1_drive1_ld)
# print(Strat_1_drive1_mat.shape)
# Strat_2_drive1_ld = []
# for i in range(len(Strat_2_drive1)):
#     Strat_2_drive1_ld.append(np.load(Strat_2_drive1[i]))
# Strat_2_drive1_mat = np.array(Strat_2_drive1_ld)
# print(Strat_2_drive1_mat.shape)
# Strat_3_drive1_ld = []
# for i in range(len(Strat_3_drive1)):
#     Strat_3_drive1_ld.append(np.load(Strat_3_drive1[i]))
#
# Strat_3_drive1_mat = np.array(Strat_3_drive1_ld)
# print(Strat_3_drive1_mat.shape)
#
# Strat_1_drive2_ld = []
# for i in range(len(Strat_1_drive2)):
#     Strat_1_drive2_ld.append(np.load(Strat_1_drive2[i]))
# Strat_1_drive2_mat = np.array(Strat_1_drive2_ld)
# print(Strat_1_drive2_mat.shape)
# Strat_2_drive2_ld = []
# for i in range(len(Strat_2_drive2)):
#     Strat_2_drive2_ld.append(np.load(Strat_2_drive2[i]))
# Strat_2_drive2_mat = np.array(Strat_2_drive2_ld)
# print(Strat_1_drive2_mat.shape)
# Strat_3_drive2_ld = []
# for i in range(len(Strat_3_drive2)):
#     Strat_3_drive2_ld.append(np.load(Strat_3_drive2[i]))
# Strat_3_drive2_mat = np.array(Strat_3_drive2_ld)
#

#del Actual_connections[indices_not_in_list2]

# X = Strat_1_drive2_mat
# print(X)
# T_obs_2, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=1024, tail=0,n_jobs =-1)#,adjacency=adjacency_matrix)
# #print(clusters_2)
# # print(T_obs_2.shape)
# T = np.zeros(T_obs_2.shape[0])
# List_Clusters_relevant = []
# for i in range(len(p_values_2)):
#     if p_values_2[i]<0.05:
#         List_Clusters_relevant.append(clusters_2[i])
# for k in range(len(List_Clusters_relevant)):
#     for cluster in range(len(List_Clusters_relevant[k])):
#         T[List_Clusters_relevant[k][cluster]]=1
# T_obs_2 = T*T_obs_2
# print(Actual_connections)
# print(T_obs_2)
# X = Strat_2_drive2_mat
# print(X.shape)
# T_obs_2, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=1024, tail=0,n_jobs =-1)#adjacency=adjacency_matrix)
# #print(clusters_2)
# # print(T_obs_2.shape)
# T = np.zeros(T_obs_2.shape[0])
# List_Clusters_relevant = []
# for i in range(len(p_values_2)):
#     if p_values_2[i]<0.05:
#         List_Clusters_relevant.append(clusters_2[i])
# for k in range(len(List_Clusters_relevant)):
#     for cluster in range(len(List_Clusters_relevant[k])):
#         T[List_Clusters_relevant[k][cluster]]=1
# T_obs_2 = T*T_obs_2
# print(Actual_connections)
# print(T_obs_2)
# X = Strat_3_drive2_mat
# print(X.shape)
# T_obs_2, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=1024, tail=0,n_jobs =-1)#,adjacency=adjacency_matrix)
# #print(clusters_2)
# # print(T_obs_2.shape)
# T = np.zeros(T_obs_2.shape[0])
# List_Clusters_relevant = []
# for i in range(len(p_values_2)):
#     if p_values_2[i]<0.05:
#         List_Clusters_relevant.append(clusters_2[i])
# for k in range(len(List_Clusters_relevant)):
#     for cluster in range(len(List_Clusters_relevant[k])):
#         T[List_Clusters_relevant[k][cluster]]=1
# T_obs_2 = T*T_obs_2
# print(Actual_connections)
# print(T_obs_2)
#
#

#
#
## Loading
directory_path_Strat_1_cali = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Cali/1/*'
directory_path_Strat_2_cali = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Cali/2/*'
directory_path_Strat_3_cali = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Cali/3/*'


directory_path_Strat_1_drive1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Dr1/1/*'
directory_path_Strat_2_drive1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Dr1/2/*'
directory_path_Strat_3_drive1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Dr1/3/*'


directory_path_Strat_1_drive2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Dr2/1/*'
directory_path_Strat_2_drive2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Dr2/2/*'
directory_path_Strat_3_drive2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/PSD/Source_Space_ComputedirectInInverse/Dr2/3/*'


Strat_1_cali = sorted(glob.glob(directory_path_Strat_1_cali))
Strat_2_cali = sorted(glob.glob(directory_path_Strat_2_cali))
Strat_3_cali = sorted(glob.glob(directory_path_Strat_3_cali))

Strat_1_drive1 = sorted(glob.glob(directory_path_Strat_1_drive1))
Strat_2_drive1 = sorted(glob.glob(directory_path_Strat_2_drive1))
Strat_3_drive1 = sorted(glob.glob(directory_path_Strat_3_drive1))

Strat_1_drive2 = sorted(glob.glob(directory_path_Strat_1_drive2))
Strat_2_drive2 = sorted(glob.glob(directory_path_Strat_2_drive2))
Strat_3_drive2 = sorted(glob.glob(directory_path_Strat_3_drive2))

Strat_1_cali_ld = []
for i in range(len(Strat_1_cali)):
    Strat_1_cali_ld.append(np.load(Strat_1_cali[i]))
Strat_1_cali_mat = np.array(Strat_1_cali_ld)
print(Strat_1_cali_mat.shape)
Strat_2_cali_ld = []
for i in range(len(Strat_2_cali)):
    Strat_2_cali_ld.append(np.load(Strat_2_cali[i]))
Strat_2_cali_mat = np.array(Strat_2_cali_ld)
print(Strat_2_cali_mat.shape)
Strat_3_cali_ld = []
for i in range(len(Strat_3_cali)):
    Strat_3_cali_ld.append(np.load(Strat_3_cali[i]))
Strat_3_cali_mat = np.array(Strat_3_cali_ld)
print(Strat_3_cali_mat.shape)
Strat_1_drive1_ld = []
for i in range(len(Strat_1_drive1)):
    Strat_1_drive1_ld.append(np.load(Strat_1_drive1[i]))
Strat_1_drive1_mat = np.array(Strat_1_drive1_ld)
print(Strat_1_drive1_mat.shape)
Strat_2_drive1_ld = []
for i in range(len(Strat_2_drive1)):
    Strat_2_drive1_ld.append(np.load(Strat_2_drive1[i]))
Strat_2_drive1_mat = np.array(Strat_2_drive1_ld)
print(Strat_2_drive1_mat.shape)
Strat_3_drive1_ld = []
for i in range(len(Strat_3_drive1)):
    Strat_3_drive1_ld.append(np.load(Strat_3_drive1[i]))

Strat_3_drive1_mat = np.array(Strat_3_drive1_ld)
print(Strat_3_drive1_mat.shape)

Strat_1_drive2_ld = []
for i in range(len(Strat_1_drive2)):
    Strat_1_drive2_ld.append(np.load(Strat_1_drive2[i]))
Strat_1_drive2_mat = np.array(Strat_1_drive2_ld)
print(Strat_1_drive2_mat.shape)
Strat_2_drive2_ld = []
for i in range(len(Strat_2_drive2)):
    Strat_2_drive2_ld.append(np.load(Strat_2_drive2[i]))
Strat_2_drive2_mat = np.array(Strat_2_drive2_ld)
print(Strat_1_drive2_mat.shape)
Strat_3_drive2_ld = []
for i in range(len(Strat_3_drive2)):
    Strat_3_drive2_ld.append(np.load(Strat_3_drive2[i]))
Strat_3_drive2_mat = np.array(Strat_3_drive2_ld)
print(Strat_3_drive2_mat.shape)


print(Strat_3_drive2_mat)

src_2 = mne.read_source_spaces(src)
src_dist = mne.add_source_space_distances(src_2,n_jobs = -1)
adjacency_matrix = mne.spatial_src_adjacency(src_dist,dist = 0.02)



#
X = Strat_1_drive2_mat
print(X.shape)
T_obs_2, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=1024, tail=0,n_jobs =-1,adjacency=adjacency_matrix)
#print(clusters_2)
# print(T_obs_2.shape)
T = np.zeros(T_obs_2.shape[0])
List_Clusters_relevant = []
for i in range(len(p_values_2)):
    if p_values_2[i]<0.05:
        List_Clusters_relevant.append(clusters_2[i])
for k in range(len(List_Clusters_relevant)):
    for cluster in range(len(List_Clusters_relevant[k])):
        T[List_Clusters_relevant[k][cluster]]=1
T_obs_2 = T*T_obs_2
print(Names_Labels)
print(T_obs_2)
#
#
clim = dict(kind='value', lims=[-15,0, 15])

DummyVerti = Dummy_stc.vertices
#
significant_stc = mne.SourceEstimate(T_obs_2,
                                     vertices=DummyVerti,
                                    tmin=0, tstep=1,
                                     subject=subject)
#
message = "DICS source power in the 12-30 Hz frequency band"
brain = significant_stc.plot(hemi="both", views="axial", subjects_dir=subjects_dir, subject=subject,surface="white",alpha = 0.9,
                             colormap='seismic', time_viewer=False, size=(1920,1080),background = 'white',clim = clim,colorbar=False,cortex = 'classic')
print(significant_stc.shape)
#brain.add_annotation("aparc_sub")
#fig = mne.viz.plot_alignment(fake_info, significant_stc, show_axes=True, trans = None,surfaces='white', subjects_dir=subjects_dir, subject=subject)
brain.save_image(filename = 'Strat_1_Cluster_dr2_beta_colorb.jpeg')

X = Strat_2_drive2_mat
print(X.shape)
T_obs_2, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=1024, tail=0,n_jobs =-1,adjacency=adjacency_matrix)
#print(clusters_2)
# print(T_obs_2.shape)
T = np.zeros(T_obs_2.shape[0])
List_Clusters_relevant = []
for i in range(len(p_values_2)):
    if p_values_2[i]<0.05:
        List_Clusters_relevant.append(clusters_2[i])
for k in range(len(List_Clusters_relevant)):
    for cluster in range(len(List_Clusters_relevant[k])):
        T[List_Clusters_relevant[k][cluster]]=1
T_obs_2 = T*T_obs_2
print(Names_Labels)
print(T_obs_2)

#
#

#
significant_stc = mne.SourceEstimate(T_obs_2,
                                     vertices=DummyVerti,
                                    tmin=0, tstep=1,
                                     subject=subject)

message = "DICS source power in the 12-30 Hz frequency band"
brain = significant_stc.plot(hemi="both", views="axial", subjects_dir=subjects_dir, subject=subject,surface="white",alpha = 0.9,
                             colormap='seismic', time_viewer=False, size=(1920,1080),background = 'white',clim = clim,colorbar=False,cortex = 'classic')
#brain.add_annotation("aparc_sub")
brain.save_image(filename = 'Strat_2_Cluster_dr2.jpeg')


X = Strat_3_drive2_mat
print(X.shape)
T_obs_2, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=1024, tail=0,n_jobs =-1,adjacency=adjacency_matrix)
#print(clusters_2)
# print(T_obs_2.shape)
T = np.zeros(T_obs_2.shape[0])
List_Clusters_relevant = []
for i in range(len(p_values_2)):
    if p_values_2[i]<1:
        List_Clusters_relevant.append(clusters_2[i])
for k in range(len(List_Clusters_relevant)):
    for cluster in range(len(List_Clusters_relevant[k])):
        T[List_Clusters_relevant[k][cluster]]=1
T_obs_2 = T*T_obs_2
print(Names_Labels)
print(T_obs_2)

# DummyVerti = Dummy_stc.vertices
#
significant_stc = mne.SourceEstimate(T_obs_2,
                                     vertices=DummyVerti,
                                    tmin=0, tstep=1,
                                     subject=subject)

message = "DICS source power in the 12-30 Hz frequency band"
brain = significant_stc.plot(hemi="both", views="axial", subjects_dir=subjects_dir, subject=subject,surface="white",alpha = 0.9,
                             colormap='seismic', time_viewer=False, size=(1920,1080),background = 'white',clim = clim,colorbar=False,cortex = 'classic')

#brain.add_annotation("aparc_sub")
brain.save_image(filename = 'Strat_3_Custer_dr2_beta.jpeg')