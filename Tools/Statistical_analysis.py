import os
import time
import numpy as np
import mne
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.stats import permutation_cluster_test
from sklearn.metrics import r2_score
from mne.datasets import somato
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import neurokit2 as nk
from scipy import signal,stats,fft
from spectrum import arburg,arma2psd,pburg
import statsmodels.regression.linear_model as transform
from scipy.signal import spectrogram
from visbrain.utils import morlet, normalization, averaging
from scipy.stats import rankdata

def Compute_Rsquare_Map_Welch_optimization(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape
    #print(a)
    #print(b)
    Rsquare_tab = np.zeros([a[1]])

    for l in range(b[1]):
        concat_tab_MI = []
        concat_tab_Rest = []
        for i in range(a[0]):
            concat_tab_MI.append(Power_of_trials_1[i,l])
            concat_tab_Rest.append(Power_of_trials_2[i,l])
            #correlation_matrix = np.corrcoef(concat_tab_MI, concat_tab_Rest)
        Sum_q = sum(concat_tab_MI)
        Sum_r = sum(concat_tab_Rest)
        n1 = len(concat_tab_MI)
        n2 = len(concat_tab_Rest)
        sumsqu1 = sum(np.multiply(concat_tab_MI,concat_tab_MI))
        sumsqu2 = sum(np.multiply(concat_tab_Rest,concat_tab_Rest))

        G=((Sum_q+Sum_r)**2)/(n1+n2)

        #correlation_xy = correlation_matrix[0,1]
        #Rsquare_tab[k,l] = correlation_xy**2
        Rsquare_tab[l] = (Sum_q**2/n1+Sum_r**2/n2-G)/(sumsqu1+sumsqu2-G)

    return Rsquare_tab


def Compute_Rsquare_Map_Welch(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape
    #print(a)
    #print(b)
    Rsquare_tab = np.zeros([a[1],a[2]])
    for k in range(b[1]):
        for l in range(b[2]):
            concat_tab_MI = []
            concat_tab_Rest = []
            for i in range(a[0]):
                concat_tab_MI.append(Power_of_trials_1[i,k,l])
                concat_tab_Rest.append(Power_of_trials_2[i,k,l])
            #correlation_matrix = np.corrcoef(concat_tab_MI, concat_tab_Rest)
            Sum_q = sum(concat_tab_MI)
            Sum_r = sum(concat_tab_Rest)
            n1 = len(concat_tab_MI)
            n2 = len(concat_tab_Rest)
            sumsqu1 = sum(np.multiply(concat_tab_MI,concat_tab_MI))
            sumsqu2 = sum(np.multiply(concat_tab_Rest,concat_tab_Rest))

            G=((Sum_q+Sum_r)**2)/(n1+n2)

            #correlation_xy = correlation_matrix[0,1]
            #Rsquare_tab[k,l] = correlation_xy**2
            Rsquare_tab[k,l] = (Sum_q**2/n1+Sum_r**2/n2-G)/(sumsqu1+sumsqu2-G)

    return Rsquare_tab

def Compute_Wilcoxon_Map_optimization(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape

    Wsquare_tab = np.zeros([a[1]])
    Wpsqure_tab = np.zeros([a[1]])
    for k in range(b[1]):
        concat_tab_MI = []
        concat_tab_Rest = []
        for i in range(a[0]):
            concat_tab_MI.append(Power_of_trials_1[i,k])
            concat_tab_Rest.append(Power_of_trials_2[i,k])

        s,p = stats.ranksums(concat_tab_MI,concat_tab_Rest)
        Wsquare_tab[k] = s
        Wpsqure_tab[k] = p
    return Wsquare_tab,Wpsqure_tab

def Compute_Wilcoxon_Map(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape

    Wsquare_tab = np.zeros([a[1],a[2]])
    Wpsqure_tab = np.zeros([a[1],a[2]])
    for k in range(b[1]):
        for l in range(b[2]):
            concat_tab_MI = []
            concat_tab_Rest = []
            for i in range(a[0]):
                concat_tab_MI.append(Power_of_trials_1[i,k,l])
                concat_tab_Rest.append(Power_of_trials_2[i,k,l])

            s,p = stats.ranksums(concat_tab_MI,concat_tab_Rest)
            Wsquare_tab[k,l] = s
            Wpsqure_tab[k,l] = p
    return Wsquare_tab,Wpsqure_tab

def Compute_Rsquare_Map_Multiple_trials_welch(Power_of_trials_1,Power_of_trials_2,Power_of_trials_3,Power_of_trials_4,Power_of_trials_5,Power_of_trials_6,Power_of_trials_7,Power_of_trials_8,Power_of_trials_9,Power_of_trials_10,Power_of_trials_11,Power_of_trials_12,Power_of_trials_13,Power_of_trials_14,Power_of_trials_15,Power_of_trials_16):

    b = Power_of_trials_1.shape
    #print(b[2])
    Rsquare_tab = np.zeros([b[1],b[2]])
    for k in range(b[1]):
        for l in range(b[2]):
            concat_tab_MI = []
            concat_tab_Rest = []
            for i in range(b[0]):
                    concat_tab_MI.append(Power_of_trials_1[i,k,l])
                    concat_tab_MI.append(Power_of_trials_3[i,k,l])
                    concat_tab_MI.append(Power_of_trials_5[i,k,l])
                    concat_tab_MI.append(Power_of_trials_7[i,k,l])
                    concat_tab_MI.append(Power_of_trials_9[i,k,l])
                    concat_tab_MI.append(Power_of_trials_11[i,k,l])
                    concat_tab_MI.append(Power_of_trials_13[i,k,l])
                    concat_tab_MI.append(Power_of_trials_15[i,k,l])

                    concat_tab_Rest.append(Power_of_trials_2[i,k,l])
                    concat_tab_Rest.append(Power_of_trials_4[i,k,l])
                    concat_tab_Rest.append(Power_of_trials_6[i,k,l])
                    concat_tab_Rest.append(Power_of_trials_8[i,k,l])
                    concat_tab_Rest.append(Power_of_trials_10[i,k,l])
                    concat_tab_Rest.append(Power_of_trials_12[i,k,l])
                    concat_tab_Rest.append(Power_of_trials_14[i,k,l])
                    concat_tab_Rest.append(Power_of_trials_16[i,k,l])

            Sum_q = sum(concat_tab_MI)
            Sum_r = sum(concat_tab_Rest)
            n1 = len(concat_tab_MI)
            n2 = len(concat_tab_Rest)
            sumsqu1 = sum(np.multiply(concat_tab_MI,concat_tab_MI))
            sumsqu2 = sum(np.multiply(concat_tab_Rest,concat_tab_Rest))
            G=((Sum_q+Sum_r)**2)/(n1+n2)
            #correlation_xy = correlation_matrix[0,1]
            #Rsquare_tab[k,l] = correlation_xy**2
            Rsquare_tab[k,l] = (Sum_q**2/n1+Sum_r**2/n2-G)/(sumsqu1+sumsqu2-G)
            # correlation_matrix = np.corrcoef(concat_tab_MI, concat_tab_Rest)
            # correlation_xy = correlation_matrix[0,1]
            # Rsquare_tab[k,l] = correlation_xy**2

    return Rsquare_tab

def signed_Rsquare(Rsquare,Wsquare):
    inner = Wsquare/abs(Wsquare)
    return inner*Rsquare


# def Compute_Rsquare_Map(Power_of_trials_1,Power_of_trials_2):
#     a = Power_of_trials_1.shape
#     Rsquare_tab = np.zeros([a[1],a[2]])
#     for k in range(a[1]):
#         for l in range(a[2]):
#             concat_tab_MI = []
#             concat_tab_Rest = []
#             for i in range(a[0]):
#                 for j in range(a[3]):
#                     concat_tab_MI.append(Power_of_trials_1[i,k,l,j])
#                     concat_tab_Rest.append(Power_of_trials_2[i,k,l,j])
#             correlation_matrix = np.corrcoef(concat_tab_Rest, concat_tab_MI)
#             correlation_xy = correlation_matrix[0,1]
#             Rsquare_tab[k,l] = correlation_xy**2
#
#     return Rsquare_tab

def Compute_Signed_Rsquare(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape
    #print(a)
    #print(b)
    Rsquare_tab = np.zeros([a[1],a[2]])
    Wsquare_tab = np.zeros([a[1],a[2]])
    for k in range(b[1]):
        for l in range(b[2]):
            concat_tab_MI = []
            concat_tab_Rest = []
            for i in range(a[0]):
                concat_tab_MI.append(Power_of_trials_1[i,k,l])
                concat_tab_Rest.append(Power_of_trials_2[i,k,l])
            #correlation_matrix = np.corrcoef(concat_tab_MI, concat_tab_Rest)
            Sum_q = sum(concat_tab_MI)
            Sum_r = sum(concat_tab_Rest)
            n1 = len(concat_tab_MI)
            n2 = len(concat_tab_Rest)
            sumsqu1 = sum(np.multiply(concat_tab_MI,concat_tab_MI))
            sumsqu2 = sum(np.multiply(concat_tab_Rest,concat_tab_Rest))

            G=((Sum_q+Sum_r)**2)/(n1+n2)

            #correlation_xy = correlation_matrix[0,1]
            #Rsquare_tab[k,l] = correlation_xy**2
            s,p = stats.ranksums(concat_tab_MI,concat_tab_Rest)

            Wsquare_tab[k,l] = s
            Rsquare_tab[k,l] = ((Sum_q**2/n1+Sum_r**2/n2-G)/(sumsqu1+sumsqu2-G))
    Wsquare_tab = Wsquare_tab/abs(Wsquare_tab)
    return Wsquare_tab*Rsquare_tab


def Compute_Signed_Rsquare_optimization(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape
    #print(a)
    #print(b)
    Rsquare_tab = np.zeros([a[1]])
    Wsquare_tab = np.zeros([a[1]])
    for l in range(b[1]):
        concat_tab_MI = []
        concat_tab_Rest = []
        for i in range(a[0]):
            concat_tab_MI.append(Power_of_trials_1[i,l])
            concat_tab_Rest.append(Power_of_trials_2[i,l])
            #correlation_matrix = np.corrcoef(concat_tab_MI, concat_tab_Rest)
        Sum_q = sum(concat_tab_MI)
        Sum_r = sum(concat_tab_Rest)
        n1 = len(concat_tab_MI)
        n2 = len(concat_tab_Rest)
        sumsqu1 = sum(np.multiply(concat_tab_MI,concat_tab_MI))
        sumsqu2 = sum(np.multiply(concat_tab_Rest,concat_tab_Rest))

        G=((Sum_q+Sum_r)**2)/(n1+n2)
        s,p = stats.ranksums(concat_tab_MI,concat_tab_Rest)
        #correlation_xy = correlation_matrix[0,1]
        #Rsquare_tab[k,l] = correlation_xy**2
        Rsquare_tab[l] = np.sign(s)*((Sum_q**2/n1+Sum_r**2/n2-G)/(sumsqu1+sumsqu2-G))
        Wsquare_tab[l] = s
    Wsquare_tab = Wsquare_tab/abs(Wsquare_tab)
    return Wsquare_tab*Rsquare_tab



def Compute_T_Test_Map(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape

    Tsquare_tab = np.zeros([a[1],a[2]])
    Tpsquare_tab = np.zeros([a[1],a[2]])
    for k in range(b[1]):
        for l in range(b[2]):
            concat_tab_MI = []
            concat_tab_Rest = []
            for i in range(a[0]):
                concat_tab_MI.append(Power_of_trials_1[i,k,l])
                concat_tab_Rest.append(Power_of_trials_2[i,k,l])

            s,p = stats.ttest_ind(concat_tab_MI,concat_tab_Rest,equal_var = False,nan_policy = 'omit')
            Tsquare_tab[k,l] = s
            Tpsquare_tab[k,l] = p
    return Tsquare_tab,Tpsquare_tab
