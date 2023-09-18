import numpy as np
import glob
from os import listdir
import matplotlib.pyplot as plt
from scipy import signal,stats,fft
from Topomap_separate import *
from statsmodels.stats.anova import AnovaRM
import matlab.engine
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from collections import Counter
import community
import networkx as nx
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
# electrode_Cortex = ['C1','FC1','Cz','FCz','CPz','CP3']
electrode_Cortex = ['C1']

Index_electrode =[]
test = False
for i in range(len(electrodes)):
    for j in electrode_Cortex:
        if electrodes[i] == j:
            Index_electrode.append(i)
            test = True
            print(j)
            break




Index_electrode = np.array(Index_electrode)

font_princ = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 15,
    }

font = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 15,
    }

# electrode = ['FC3','FC1','FC5','FCz','FC6','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
electrode = ['C1']
time_seres = []
for i in range(39):
    if i%10 !=0:
        time_seres.append('')
    else:
        time_seres.append(i/10)

freq_seres = []
frq_dep = 4
for i in range(0,358):
    if i%10!=0:
        freq_seres.append('')
    else:
        freq_seres.append(round(frq_dep))
    frq_dep = frq_dep + 0.1



directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = np.array(Strat_1_cali_ld)
print(Strat_1_cali_mat.shape)
Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = np.array(Strat_2_cali_ld)
print(Strat_2_cali_mat.shape)
Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = np.array(Strat_3_cali_ld)
print(Strat_3_cali_mat.shape)
Strategy_Fscore = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]
Strategy_Fscore_theta  = Strategy_Fscore
print(np.array(Strategy_Fscore).shape)

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/Cluster/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/Cluster/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/Cluster/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)
print(Strat_1)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = (Strat_1_cali_ld)

Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = (Strat_2_cali_ld)

Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = (Strat_3_cali_ld)


Strategy_Cluster = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/pval/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/pval/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/four_to_8/pval/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = (Strat_1_cali_ld)

Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = (Strat_2_cali_ld)

Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = (Strat_3_cali_ld)


Strategy_pval = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]
print(len(Strategy_Cluster))
print(len(Strategy_Cluster[0]))
print(len(Strategy_Cluster[0][0][0][0]))
Strategy_fscore_filtered = []
for k in range(len(Strategy_pval)):##Strat
    Electrode_filtered = []
    for j in range(len(Strategy_Cluster[k][0])):##Electrode
        A = np.zeros((Strategy_Fscore[k][0,j,:,:].shape[0],Strategy_Fscore[k][0,j,:,:].shape[1]))
        for z in range(len(Strategy_pval[k][0][j])):##Clusters
            #print(Strategy_pval[k][0,:][j])
            if Strategy_pval[k][0][j][z]<0.001:
                A[Strategy_Cluster[k][0][j][z][0],Strategy_Cluster[k][0][j][z][1]] = 1
                print("Hello")
        #print(A)
        B_Filter = A
        Electrode_filtered.append(B_Filter)
    Strategy_fscore_filtered.append(Electrode_filtered)

Strategy_fscore_filtered_theta = np.array(Strategy_fscore_filtered)







directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = np.array(Strat_1_cali_ld)
print(Strat_1_cali_mat.shape)
Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = np.array(Strat_2_cali_ld)
print(Strat_2_cali_mat.shape)
Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = np.array(Strat_3_cali_ld)
print(Strat_3_cali_mat.shape)
Strategy_Fscore = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]
Strategy_Fscore_alpha = Strategy_Fscore
print(np.array(Strategy_Fscore).shape)

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/Cluster/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/Cluster/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/Cluster/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)
print(Strat_1)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = (Strat_1_cali_ld)

Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = (Strat_2_cali_ld)

Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = (Strat_3_cali_ld)


Strategy_Cluster = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/pval/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/pval/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/alpha/pval/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = (Strat_1_cali_ld)

Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = (Strat_2_cali_ld)

Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = (Strat_3_cali_ld)


Strategy_pval = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]

Strategy_fscore_filtered = []
for k in range(len(Strategy_pval)):##Strat
    Electrode_filtered = []
    for j in range(len(Strategy_Cluster[k][0])):##Electrode
        A = np.zeros((Strategy_Fscore[k][0,j,:,:].shape[0],Strategy_Fscore[k][0,j,:,:].shape[1]))
        for z in range(len(Strategy_pval[k][0][j])):##Clusters
            #print(Strategy_pval[k][0,:][j])
            if Strategy_pval[k][0][j][z]<0.01:
                A[Strategy_Cluster[k][0][j][z][0],Strategy_Cluster[k][0][j][z][1]] = 1
                print("Hello")
        #print(A)
        B_Filter = A
        Electrode_filtered.append(B_Filter)
    Strategy_fscore_filtered.append(Electrode_filtered)

Strategy_fscore_filtered_alpha = np.array(Strategy_fscore_filtered)




directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = np.array(Strat_1_cali_ld)
print(Strat_1_cali_mat.shape)
Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = np.array(Strat_2_cali_ld)
print(Strat_2_cali_mat.shape)
Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = np.array(Strat_3_cali_ld)
print(Strat_3_cali_mat.shape)
Strategy_Fscore = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]
Strategy_Fscore_beta_low  = Strategy_Fscore
print(np.array(Strategy_Fscore).shape)

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/Cluster/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/Cluster/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/Cluster/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)
print(Strat_1)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = (Strat_1_cali_ld)

Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = (Strat_2_cali_ld)

Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = (Strat_3_cali_ld)


Strategy_Cluster = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/pval/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/pval/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/beta/pval/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = (Strat_1_cali_ld)

Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = (Strat_2_cali_ld)

Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = (Strat_3_cali_ld)


Strategy_pval = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]

Strategy_fscore_filtered = []
for k in range(len(Strategy_pval)):##Strat
    Electrode_filtered = []
    for j in range(len(Strategy_Cluster[k][0])):##Electrode
        A = np.zeros((Strategy_Fscore[k][0,j,:,:].shape[0],Strategy_Fscore[k][0,j,:,:].shape[1]))
        for z in range(len(Strategy_pval[k][0][j])):##Clusters
            #print(Strategy_pval[k][0,:][j])
            if Strategy_pval[k][0][j][z]<0.01:
                A[Strategy_Cluster[k][0][j][z][0],Strategy_Cluster[k][0][j][z][1]] = 1
                print("Hello")
        #print(A)
        B_Filter = A
        Electrode_filtered.append(B_Filter)
    Strategy_fscore_filtered.append(Electrode_filtered)

Strategy_fscore_filtered_low_beta = np.array(Strategy_fscore_filtered)


#
#
# directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/1/*'
# directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/2/*'
# directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/3/*'
#
#
# Strat_1 = glob.glob(directory_path_Strat_1)
# Strat_2 = glob.glob(directory_path_Strat_2)
# Strat_3 = glob.glob(directory_path_Strat_3)
#
# Strat_1_cali_ld=[]
# for i in range(len(Strat_1)):
#     Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
# Strat_1_cali_mat = np.array(Strat_1_cali_ld)
# print(Strat_1_cali_mat.shape)
# Strat_2_cali_ld = []
# for i in range(len(Strat_2)):
#     Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
# Strat_2_cali_mat = np.array(Strat_2_cali_ld)
# print(Strat_2_cali_mat.shape)
# Strat_3_cali_ld = []
# for i in range(len(Strat_3)):
#     Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
# Strat_3_cali_mat = np.array(Strat_3_cali_ld)
# print(Strat_3_cali_mat.shape)
# Strategy_Fscore = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]
# Strategy_Fscore_beta_high  = Strategy_Fscore
# print(np.array(Strategy_Fscore).shape)
#
# directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/Cluster/1/*'
# directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/Cluster/2/*'
# directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/Cluster/3/*'
#
#
# Strat_1 = glob.glob(directory_path_Strat_1)
# Strat_2 = glob.glob(directory_path_Strat_2)
# Strat_3 = glob.glob(directory_path_Strat_3)
# print(Strat_1)
#
# Strat_1_cali_ld=[]
# for i in range(len(Strat_1)):
#     Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
# Strat_1_cali_mat = np.array(Strat_1_cali_ld)
# print(Strat_1_cali_mat.shape)
# Strat_2_cali_ld = []
# for i in range(len(Strat_2)):
#     Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
# Strat_2_cali_mat = np.array(Strat_2_cali_ld)
# print(Strat_2_cali_mat.shape)
# Strat_3_cali_ld = []
# for i in range(len(Strat_3)):
#     Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
# Strat_3_cali_mat = np.array(Strat_3_cali_ld)
# print(Strat_3_cali_mat.shape)
#
# Strategy_Cluster = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]
#
# directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/pval/1/*'
# directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/pval/2/*'
# directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/high_beta/pval/3/*'
#
#
# Strat_1 = glob.glob(directory_path_Strat_1)
# Strat_2 = glob.glob(directory_path_Strat_2)
# Strat_3 = glob.glob(directory_path_Strat_3)
#
# Strat_1_cali_ld=[]
# for i in range(len(Strat_1)):
#     Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
# Strat_1_cali_mat = np.array(Strat_1_cali_ld)
# print(Strat_1_cali_mat.shape)
# Strat_2_cali_ld = []
# for i in range(len(Strat_2)):
#     Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
# Strat_2_cali_mat = np.array(Strat_2_cali_ld)
# print(Strat_2_cali_mat.shape)
# Strat_3_cali_ld = []
# for i in range(len(Strat_3)):
#     Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
# Strat_3_cali_mat = np.array(Strat_3_cali_ld)
# print(Strat_3_cali_mat.shape)
#
# Strategy_pval = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]
#
# Strategy_fscore_filtered = []
# for k in range(len(Strategy_pval)):##Strat
#     Electrode_filtered = []
#     for j in range(len(Strategy_Cluster[k][0,:])):##Electrode
#         A = np.zeros((Strategy_Fscore[k][0,j,:,:].shape[0],Strategy_Fscore[k][0,j,:,:].shape[1]))
#         for z in range(len(Strategy_pval[k][0,:][j])):##Clusters
#             #print(Strategy_pval[k][0,:][j])
#             if Strategy_pval[k][0,:][j][z]<0.001:
#                 A[Strategy_Cluster[k][0,:][j][z][0],Strategy_Cluster[k][0,:][j][z][1]] = 1
#                 print("Hello")
#         #print(A)
#         B_Filter = A
#         Electrode_filtered.append(B_Filter)
#     Strategy_fscore_filtered.append(Electrode_filtered)
#
# Strategy_fscore_filtered_high_beta = np.array(Strategy_fscore_filtered)


directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = np.array(Strat_1_cali_ld)
print(Strat_1_cali_mat.shape)
Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = np.array(Strat_2_cali_ld)
print(Strat_2_cali_mat.shape)
Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = np.array(Strat_3_cali_ld)
print(Strat_3_cali_mat.shape)
Strategy_Fscore = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]
Strategy_Fscore_gamma  = Strategy_Fscore
print(np.array(Strategy_Fscore).shape)

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/Cluster/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/Cluster/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/Cluster/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)
print(Strat_1)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = (Strat_1_cali_ld)

Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = (Strat_2_cali_ld)

Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = (Strat_3_cali_ld)


Strategy_Cluster = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/pval/1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/pval/2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/New/over_35/pval/3/*'


Strat_1 = glob.glob(directory_path_Strat_1)
Strat_2 = glob.glob(directory_path_Strat_2)
Strat_3 = glob.glob(directory_path_Strat_3)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat =(Strat_1_cali_ld)

Strat_2_cali_ld = []
for i in range(len(Strat_2)):
    Strat_2_cali_ld.append(np.load(Strat_2[i],allow_pickle=True))
Strat_2_cali_mat = (Strat_2_cali_ld)

Strat_3_cali_ld = []
for i in range(len(Strat_3)):
    Strat_3_cali_ld.append(np.load(Strat_3[i],allow_pickle=True))
Strat_3_cali_mat = (Strat_3_cali_ld)


Strategy_pval = [Strat_1_cali_mat, Strat_2_cali_mat, Strat_3_cali_mat]

Strategy_fscore_filtered = []
for k in range(len(Strategy_pval)):##Strat
    Electrode_filtered = []
    for j in range(len(Strategy_Cluster[k][0])):##Electrode
        A = np.zeros((Strategy_Fscore[k][0,j,:,:].shape[0],Strategy_Fscore[k][0,j,:,:].shape[1]))
        for z in range(len(Strategy_pval[k][0][j])):##Clusters
            #print(Strategy_pval[k][0,:][j])
            if Strategy_pval[k][0][j][z]<0.01:
                A[Strategy_Cluster[k][0][j][z][0],Strategy_Cluster[k][0][j][z][1]] = 1
                print("Hello")
        #print(A)
        B_Filter = A
        Electrode_filtered.append(B_Filter)
    Strategy_fscore_filtered.append(Electrode_filtered)

Strategy_fscore_filtered_gamma = np.array(Strategy_fscore_filtered)








print(Strategy_fscore_filtered_alpha.shape)
print(Strategy_fscore_filtered_low_beta.shape)
#print(Strategy_fscore_filtered_high_beta.shape)
Reassmble_score_filtered = np.concatenate((Strategy_fscore_filtered_theta,Strategy_fscore_filtered_alpha,Strategy_fscore_filtered_low_beta,Strategy_fscore_filtered_gamma),axis = 3)
Reassmble_score = np.concatenate((Strategy_Fscore_theta,Strategy_Fscore_alpha,Strategy_Fscore_beta_low,Strategy_Fscore_gamma),axis = 4)

print(Reassmble_score_filtered.shape)
print(Reassmble_score.shape)
directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/TimeFrequencyTest/Average/*'

Strat_1 = glob.glob(directory_path_Strat_1)

print(Strat_1)

Strat_1_cali_ld=[]
for i in range(len(Strat_1)):
    Strat_1_cali_ld.append(np.load(Strat_1[i],allow_pickle=True))
Strat_1_cali_mat = np.array(Strat_1_cali_ld)
print(Strat_1_cali_mat.shape)
Strat_1_cali_mat = 100*Strat_1_cali_mat.mean(1)
print(Reassmble_score.shape)
str_strat= ['3','1','2']
for i in (Index_electrode):


    VMAX = np.max([abs(Reassmble_score[0,0,i,:,:]).max(),abs(Reassmble_score[1,0,i,:,:]).max(),abs(Reassmble_score[2,0,i,:,:]).max()])
    print(VMAX)

    for k in range(3):
        strat_str = str_strat[k]
        fig,ax = plt.subplots(2,1)
        Disp_Strat_1 = (Reassmble_score[k,0,i,:,:])#*Reassmble_score_filtered[k,i,:,:]
        im1 = ax[0].imshow(Disp_Strat_1.T,cmap='jet',origin ='lower',aspect = 'auto',vmin = - 7,vmax = 7)
        contours = ax[0].contour(Reassmble_score_filtered[k,i,:,:].T, levels=[0.5], colors='black', linestyles='dotted',origin='lower')
        ax[0].set_xticks(range(39))
        ax[0].set_xticklabels(time_seres)
        ax[0].set_yticks(range(0,358))
        ax[0].set_yticklabels(freq_seres)
        ax[0].tick_params(axis='both', which='both', length=0)
        cbar1 = fig.colorbar(im1, ax=ax[0])
        cbar1.set_label('ERD/ERS - F-score', rotation=270,labelpad = 15)
        ax[0].set_xlabel(' Time (s)', fontdict=font)
        ax[0].set_ylabel('Frequency (Hz)', fontdict=font)

        ax[0].set_title( 'Strat '+ str(strat_str)+ ' Sensor F-Score ' + electrodes[i],fontdict = font_princ)


        Disp_Strat_1 = (Strat_1_cali_mat[k,i,:,40:400])
        im = ax[1].imshow(Disp_Strat_1.T,cmap='jet',origin ='lower',aspect = 'auto',vmin = - np.max(abs(Strat_1_cali_mat[k,i,:,40:400])),vmax = np.max(abs(Strat_1_cali_mat[k,i,:,40:400])))
        ax[1].set_xticks(range(39))
        ax[1].set_xticklabels(time_seres)
        ax[1].set_yticks(range(0,358))
        ax[1].set_yticklabels(freq_seres)
        ax[1].tick_params(axis='both', which='both', length=0)
        cbar = fig.colorbar(im, ax=ax[1])
        cbar.set_label('ERD/ERS', rotation=270,labelpad = 15)
        ax[1].set_xlabel(' Time (s)', fontdict=font)
        ax[1].set_ylabel('Frequency (Hz)', fontdict=font)
        ax[1].set_title( 'Strat '+ str(strat_str)+ ' Sensor ERD/ERS ' + electrodes[i],fontdict = font_princ)
        #strat_str +=1
        plt.subplots_adjust(bottom = 0.05,top = 0.92)

plt.show()
print(Reassmble_score_filtered)


print(Reassmble_score_filtered.shape)
file_FilterCluster= 'FilteredCluster.npy'

np.save(file_FilterCluster,Reassmble_score_filtered[:,:,:,:])
