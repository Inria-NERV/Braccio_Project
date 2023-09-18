import numpy as np
import glob
from os import listdir
import matplotlib.pyplot as plt
from scipy import signal,stats,fft,sparse
from Topomap_separate import *
from statsmodels.stats.anova import AnovaRM
import matlab.engine
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from collections import Counter
import community
from scipy.stats import wilcoxon
import networkx as nx
from scipy.spatial.distance import cdist
import scipy


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

electrodes = channel_generator(64, 'TP9', 'TP10')


# Create a dataframe of your data
import pandas as pd
fres = 1
fs = 500

biosemi_montage_inter = mne.channels.make_standard_montage('standard_1020')


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

distances = cdist(pos, pos)
threshold = 0.04
adjacency = sparse.csr_matrix((distances <= threshold).astype(int))

print(adjacency.shape)
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



# Strat_1_MI = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/MI/Cali/Node_Strength_Strat1_MI.mat')
# Strat_2_MI = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/MI/Cali/Node_Strength_Strat2_MI.mat')
# Strat_3_MI = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/MI/Cali/Node_Strength_Strat3_MI.mat')
#
# Strat_1_Rest = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/Rest/Cali/Node_Strength_Strat1_Rest.mat')
# Strat_2_Rest = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/Rest/Cali/Node_Strength_Strat2_Rest.mat')
# Strat_3_Rest = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Matlab_Data/Rest/Cali/Node_Strength_Strat3_Rest.mat')
#
# print(Strat_1_MI.keys())
# print(Strat_2_MI.keys())
# print(Strat_3_MI.keys())
#
# print(Strat_1_Rest.keys())
# print(Strat_2_Rest.keys())
# print(Strat_3_Rest.keys())
#
# Strat_1_drive2_mat = (Strat_1_MI['Node_Aver_Bins_Coherence_Strat_1_MI']-Strat_1_Rest['Node_Aver_Bins_Coherence_Strat_1_Rest'])/Strat_1_Rest['Node_Aver_Bins_Coherence_Strat_1_Rest']
# Strat_2_drive2_mat = (Strat_2_MI['Node_Aver_Bins_Coherence_Strat_2_MI']-Strat_2_Rest['Node_Aver_Bins_Coherence_Strat_2_Rest'])/Strat_2_Rest['Node_Aver_Bins_Coherence_Strat_2_Rest']
# Strat_3_drive2_mat = (Strat_3_MI['Node_Aver_Bins_Coherence_Strat_3_MI']-Strat_3_Rest['Node_Aver_Bins_Coherence_Strat_3_Rest'])/Strat_3_Rest['Node_Aver_Bins_Coherence_Strat_3_Rest']
#

# CLUSTER_From_Test = []
#
# T_Cluseter = []
#
#
# Strat_1_DRIVE_CLUSTER_B1 = Strat_1_drive2_mat
# Strat_2_DRIVE_CLUSTER_B1 = Strat_2_drive2_mat
# Strat_3_DRIVE_CLUSTER_B1 = Strat_3_drive2_mat


Strat_1_MI = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch1/Matlab_Data/MI/Drive_2/Node_Strength_Strat1_MI.mat')
Strat_2_MI = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch1/Matlab_Data/MI/Drive_2/Node_Strength_Strat2_MI.mat')
Strat_3_MI = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch1/Matlab_Data/MI/Drive_2/Node_Strength_Strat3_MI.mat')

Strat_1_Rest = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch1/Matlab_Data/Rest/Drive_2/Node_Strength_Strat1_Rest.mat')
Strat_2_Rest = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch1/Matlab_Data/Rest/Drive_2/Node_Strength_Strat2_Rest.mat')
Strat_3_Rest = scipy.io.loadmat('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch1/Matlab_Data/Rest/Drive_2/Node_Strength_Strat3_Rest.mat')

print(Strat_1_MI.keys())
print(Strat_2_MI.keys())
print(Strat_3_MI.keys())

print(Strat_1_Rest.keys())
print(Strat_2_Rest.keys())
print(Strat_3_Rest.keys())

Strat_1_drive2_mat = (Strat_1_MI['Node_Aver_Bins_Coherence_Strat_1_MI']-Strat_1_Rest['Node_Aver_Bins_Coherence_Strat_1_Rest'])/Strat_1_Rest['Node_Aver_Bins_Coherence_Strat_1_Rest']
Strat_2_drive2_mat = (Strat_2_MI['Node_Aver_Bins_Coherence_Strat_2_MI']-Strat_2_Rest['Node_Aver_Bins_Coherence_Strat_2_Rest'])/Strat_2_Rest['Node_Aver_Bins_Coherence_Strat_2_Rest']
Strat_3_drive2_mat = (Strat_3_MI['Node_Aver_Bins_Coherence_Strat_3_MI']-Strat_3_Rest['Node_Aver_Bins_Coherence_Strat_3_Rest'])/Strat_3_Rest['Node_Aver_Bins_Coherence_Strat_3_Rest']


CLUSTER_From_Test = []

T_Cluseter = []


Strat_1_DRIVE_CLUSTER_B2 = Strat_1_drive2_mat
Strat_2_DRIVE_CLUSTER_B2 = Strat_2_drive2_mat
Strat_3_DRIVE_CLUSTER_B2 = Strat_3_drive2_mat

Strat_1_DRIVE_CLUSTER = Strat_1_DRIVE_CLUSTER_B2
Strat_2_DRIVE_CLUSTER = Strat_2_DRIVE_CLUSTER_B2
Strat_3_DRIVE_CLUSTER = Strat_3_DRIVE_CLUSTER_B2

#
# Strat_1_DRIVE_CLUSTER = Strat_1_drive1_mat
# Strat_2_DRIVE_CLUSTER = Strat_2_drive1_mat
# Strat_3_DRIVE_CLUSTER = Strat_3_drive1_mat

# Strat_1_DRIVE_CLUSTER = Strat_1_cali_mat
# Strat_2_DRIVE_CLUSTER = Strat_2_cali_mat
# Strat_3_DRIVE_CLUSTER = Strat_3_cali_mat


# n_observations = 15
# pval = 0.05  # arbitrary
# df = n_observations - 1  # degrees of freedom for the test
# thresh = pval/n_observations

n_observations = 11
pval = 0.05  # arbitrary
df = n_observations - 1  # degrees of freedom for the test
thresh = stats.t.ppf(1 - pval / 2, df)

print(thresh)


X = Strat_1_DRIVE_CLUSTER
T_obs_3, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=2000, tail=0,adjacency =adjacency  ,n_jobs =-1)
T = np.zeros(64)
List_Clusters_relevant = []
for i in range(len(p_values_2)):
    if p_values_2[i]<0.05:
        List_Clusters_relevant.append(clusters_2[i])
for k in range(len(List_Clusters_relevant)):
    for cluster in range(len(List_Clusters_relevant[k])):
        T[List_Clusters_relevant[k][cluster]]=1
T_obs_3 = T*T_obs_3

significant_coords = pos[(T_obs_3<-2).astype(bool)]
n_significant = significant_coords.shape[0]
# Compute the pairwise Euclidean distances between significant electrodes
distances = np.zeros((n_significant, n_significant))
for k in range(n_significant):
    for z in range(k+1, n_significant):
        distances[k,z] = np.linalg.norm(significant_coords[k,:] - significant_coords[z,:])
        distances[z,k] = distances[k,z]

# Find the maximum distance
if len(distances)>0:
    max_distance = np.mean(distances)
    CLUSTER_From_Test.append(np.sum(T))

    T_Cluseter.append(T)



fig, axs = plt.subplots(figsize = (14,14))
np.savetxt("Cluster_FC_St1_dr2_b1.csv", T_obs_3,
              delimiter = ",")
# topo_plot(np.array([T_obs_2,T_obs_2]).T,0,electrodes,fres,fs,'Wilcoxon',-4,4,axes =axs[0])

topo_plot(np.array([T_obs_3,T_obs_3]).T,0,electrodes,fres,fs,'Wilcoxon',-9,9)

# topo_plot(np.array([T_obs_4,T_obs_4]).T,0,electrodes,fres,fs,'Wilcoxon',-4.5,4.5,axes =axs[2])
plt.show()


X = Strat_2_DRIVE_CLUSTER
T_obs_3, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=2000, tail=0,adjacency =adjacency  ,n_jobs =-1)
T = np.zeros(64)
List_Clusters_relevant = []
for i in range(len(p_values_2)):
    if p_values_2[i]<0.05:
        List_Clusters_relevant.append(clusters_2[i])
for k in range(len(List_Clusters_relevant)):
    for cluster in range(len(List_Clusters_relevant[k])):
        T[List_Clusters_relevant[k][cluster]]=1
T_obs_3 = T*T_obs_3

significant_coords = pos[(T_obs_3<-2).astype(bool)]
n_significant = significant_coords.shape[0]
# Compute the pairwise Euclidean distances between significant electrodes
distances = np.zeros((n_significant, n_significant))
for k in range(n_significant):
    for z in range(k+1, n_significant):
        distances[k,z] = np.linalg.norm(significant_coords[k,:] - significant_coords[z,:])
        distances[z,k] = distances[k,z]

# Find the maximum distance
if len(distances)>0:
    max_distance = np.mean(distances)
    CLUSTER_From_Test.append(np.sum(T))

    T_Cluseter.append(T)


fig, axs = plt.subplots(figsize = (14,14))
np.savetxt("Cluster_FC_St2_dr2_b1.csv", T_obs_3,
              delimiter = ",")
# topo_plot(np.array([T_obs_2,T_obs_2]).T,0,electrodes,fres,fs,'Wilcoxon',-4,4,axes =axs[0])

topo_plot(np.array([T_obs_3,T_obs_3]).T,0,electrodes,fres,fs,'Wilcoxon',None,None)

# topo_plot(np.array([T_obs_4,T_obs_4]).T,0,electrodes,fres,fs,'Wilcoxon',-4.5,4.5,axes =axs[2])
plt.show()


X = Strat_3_DRIVE_CLUSTER
T_obs_3, clusters_2, p_values_2, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thresh, n_permutations=2000, tail=0,adjacency =adjacency  ,n_jobs =-1)
T = np.zeros(64)
List_Clusters_relevant = []
for i in range(len(p_values_2)):
    if p_values_2[i]<0.05:
        List_Clusters_relevant.append(clusters_2[i])
for k in range(len(List_Clusters_relevant)):
    for cluster in range(len(List_Clusters_relevant[k])):
        T[List_Clusters_relevant[k][cluster]]=1
T_obs_3 = T*T_obs_3

significant_coords = pos[(T_obs_3<-2).astype(bool)]
n_significant = significant_coords.shape[0]
# Compute the pairwise Euclidean distances between significant electrodes
distances = np.zeros((n_significant, n_significant))
for k in range(n_significant):
    for z in range(k+1, n_significant):
        distances[k,z] = np.linalg.norm(significant_coords[k,:] - significant_coords[z,:])
        distances[z,k] = distances[k,z]

# Find the maximum distance
if len(distances)>0:
    max_distance = np.mean(distances)
    CLUSTER_From_Test.append(np.sum(T))

    T_Cluseter.append(T)



fig, axs = plt.subplots(figsize = (14,14))

# topo_plot(np.array([T_obs_2,T_obs_2]).T,0,electrodes,fres,fs,'Wilcoxon',-4,4,axes =axs[0])
fig, axs = plt.subplots(figsize = (14,14))
np.savetxt("Cluster_FC_St3_dr2_b1.csv", T_obs_3,
              delimiter = ",")
topo_plot(np.array([T_obs_3,T_obs_3]).T,0,electrodes,fres,fs,'Wilcoxon',None,None)

# topo_plot(np.array([T_obs_4,T_obs_4]).T,0,electrodes,fres,fs,'Wilcoxon',-4.5,4.5,axes =axs[2])
plt.show()




Strat_1_drive2_mat = 100*Strat_1_drive2_mat.mean(0)
Strat_2_drive2_mat = 100*Strat_2_drive2_mat.mean(0)
Strat_3_drive2_mat = 100*Strat_3_drive2_mat.mean(0)



# St_1= np.array([Strat_1_drive1_mat,Strat_1_drive2_mat]).mean(0)
# St_2= np.array([Strat_2_drive1_mat,Strat_2_drive2_mat]).mean(0)
# St_3= np.array([Strat_3_drive1_mat,Strat_3_drive2_mat]).mean(0)


St_1= Strat_1_drive2_mat
St_2= Strat_2_drive2_mat
St_3= Strat_3_drive2_mat
# St_1= Strat_1_cali_mat
# St_2= Strat_2_cali_mat
# St_3= Strat_3_cali_mat


fig, axs = plt.subplots()
# j =0
# for i in range(2,5):
#     topo_plot(np.array([St_1[i,:],St_1[i,:]]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])
#     j+=1
#
# fig, axs = plt.subplots(1,3)
# j =0
# for i in range(2,5):
#     topo_plot(np.array([St_2[i,:],St_2[i,:]]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])
#     j+=1
#
# fig, axs = plt.subplots(1,3)
# j =0
# for i in range(2,5):
#     topo_plot(np.array([St_3[i,:],St_3[i,:]]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])
#     j+=1


topo_plot(np.array([St_1,St_1]).T,0,electrodes,fres,fs,'Wilcoxon',-6,6)


fig, axs = plt.subplots()

topo_plot(np.array([St_2,St_2]).T,0,electrodes,fres,fs,'Wilcoxon',-6,6)


fig, axs = plt.subplots()

topo_plot(np.array([St_3,St_3]).T,0,electrodes,fres,fs,'Wilcoxon',-6,6)




zbin = np.linspace(0.0, 1.5, 25)

ALPHA = []
BETA_Low = []
BETA_High = []

for l in range(len(zbin)):

#    fig, axs = plt.subplots(1,3)
    clusters_1 = []
    j =0


    psd = (((St_1)*(St_1>(zbin[l]))).T)
    print((St_1<(-zbin[l]))[:].shape)
    significant_coords = pos[(St_1<(-zbin[l]))[:]]
    n_significant = significant_coords.shape[0]
    # Compute the pairwise Euclidean distances between significant electrodes
    distances = np.zeros((n_significant, n_significant))
    for k in range(n_significant):
        for z in range(k+1, n_significant):
            distances[k,z] = np.linalg.norm(significant_coords[k,:] - significant_coords[z,:])
            distances[z,k] = distances[k,z]

    # Find the maximum distance
    if len(distances)>0:
        max_distance = np.max(distances)
        clusters_1.append(max_distance)
    else:
        clusters_1.append(0)

        #topo_plot(np.array([ST_1_Filter,ST_1_Filter]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])
        #topo_plot(np.array([psd[:,i],psd[:,i]]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])


    #plt.show()
    #
    #
    #fig, axs = plt.subplots(1,3)
    clusters_2 = []
    j=0


    psd = (((St_2)*(St_2<(-zbin[l]))).T)
    significant_coords = pos[(St_2<(-zbin[l]))[:]]
    n_significant = significant_coords.shape[0]
    # Compute the pairwise Euclidean distances between significant electrodes
    distances = np.zeros((n_significant, n_significant))
    for k in range(n_significant):
        for z in range(k+1, n_significant):
            distances[k,z] = np.linalg.norm(significant_coords[k,:] - significant_coords[z,:])
            distances[z,k] = distances[k,z]

    # Find the maximum distance
    if len(distances)>0:
        max_distance = np.max(distances)
        clusters_2.append(max_distance)
    else:
        clusters_2.append(0)
        # topo_plot(np.array([ST_2_Filter,ST_2_Filter]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])
        #topo_plot(np.array([psd[:,i],psd[:,i]]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])


    #plt.show()

    #
    clusters_3 = []
    #fig, axs = plt.subplots(1,3)
    j=0


    psd = (((St_3)*(St_3<(-zbin[l]))).T)
    significant_coords = pos[(St_3<(-zbin[l]))[:]]
    n_significant = significant_coords.shape[0]
    # Compute the pairwise Euclidean distances between significant electrodes
    distances = np.zeros((n_significant, n_significant))
    for k in range(n_significant):
        for z in range(k+1, n_significant):
            distances[k,z] = np.linalg.norm(significant_coords[k,:] - significant_coords[z,:])
            distances[z,k] = distances[k,z]

    # Find the maximum distance
    if len(distances)>0:
        max_distance = np.max(distances)
        clusters_3.append(max_distance)
    else:
        clusters_3.append(0)
        # topo_plot(np.array([ST_3_Filter,ST_3_Filter]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])
        #topo_plot(np.array([psd[:,i],psd[:,i]]).T,0,electrodes,fres,fs,'Wilcoxon',-20,20,axes = axs[j])


    #plt.show()

    print(clusters_1)
    print(clusters_2)
    print(clusters_3)

    BETA_Low.append([clusters_1,clusters_2,clusters_3])

beta_low = np.array(BETA_Low)
fig, axs = plt.subplots()
axs.set_title('Low Beta')
axs.plot(zbin,beta_low[:,1],'g-o',label = 'Strat 1')
axs.plot(zbin,beta_low[:,0],'b-o',label = 'Strat 2')
axs.plot(zbin,beta_low[:,2],'r-o',label = 'Strat 3')
axs.legend()
plt.legend(fontsize="20")
plt.xlabel("ERD threshold",fontsize = 20)
plt.ylabel("Largest Diameter",fontsize = 20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.show()
# fig, axs = plt.subplots(1,3)
#
# alpha = np.array(ALPHA)
# beta_low = np.array(BETA_Low)
# beta_high = np.array(BETA_High)
# axs[0].set_title('Alpha')
# axs[0].plot(zbin,alpha[:,0],'b-o',label = 'Strat 1')
# axs[0].plot(zbin,alpha[:,1],'g-o',label = 'Strat 2')
# axs[0].plot(zbin,alpha[:,2],'r-o',label = 'Strat 3')
# axs[0].set(xlabel='ERD threshold', ylabel='Largest Diameter')
# axs[0].legend()
# axs[1].set_title('Low Beta')
# axs[1].plot(zbin,beta_low[:,0],'b-o',label = 'Strat 1')
# axs[1].plot(zbin,beta_low[:,1],'g-o',label = 'Strat 2')
# axs[1].plot(zbin,beta_low[:,2],'r-o',label = 'Strat 3')
# axs[1].legend()
# axs[1].set(xlabel='ERD threshold', ylabel='Largest Diameter')
# axs[2].set_title('High Beta')
#
# axs[2].plot(zbin,beta_high[:,0],'b-o',label = 'Strat 1')
# axs[2].plot(zbin,beta_high[:,1],'g-o',label = 'Strat 2')
# axs[2].plot(zbin,beta_high[:,2],'r-o',label = 'Strat 3')
# axs[2].legend()
# axs[2].set(xlabel='ERD threshold', ylabel='Largest Diameter')

plt.show()
print(thresh)

print("HELLO")

print(CLUSTER_From_Test)




print(len(CLUSTER_From_Test))

fig, ax = plt.subplots()
import seaborn as sns
import matplotlib.pyplot as plt

a = [CLUSTER_From_Test[1],CLUSTER_From_Test[1]]
b = [CLUSTER_From_Test[0],CLUSTER_From_Test[0]]
c = [CLUSTER_From_Test[2],CLUSTER_From_Test[2]]

# combine data into a single dataframe
data = {'Strategy 1': a, 'Strategy 2': b, 'Strategy 3': c}
df = pd.DataFrame(data)
subcat_order = ['Strategy 1', 'Strategy 2', 'Strategy 3']
plotting_parameters = {
    'data':    df,
    'order':   subcat_order,
    'palette': ['#1f77b4', '#2ca02c', '#d62728'],
}


sns.barplot(data=df, palette=['#2ca02c','#1f77b4',  '#d62728'],estimator = 'mean',width = .5)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.ylabel("Number of electrodes in the cluster",fontsize = 20)
plt.title("Number of electrodes in the cluster found for each strategy",fontsize = 20)

plt.show()
