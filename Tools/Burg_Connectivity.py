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
import time
from numpy.fft import fft
import multiprocessing
from spectrum.correlation import CORRELATION
from spectrum.covar import arcovar, arcovar_marple
import spectrum.yulewalker as yulewalker
from spectrum.psd import ParametricSpectrum

from multiprocessing import Pool

class MultithreadVariables:
    noverlap =  45
    N_FFT =  500
    f_max = 512
    n_per_seg =  125
    filter_order =  19


def compute_psd(Epoch_compute,j,tab,xStart,xEnd,filter_order,N_FFT):
    Block_spectrum = []
    for numBlock in tab:
        windowData = Epoch_compute[j, xStart[numBlock]:xEnd[numBlock]]
        windowData = signal.detrend(windowData, type='constant')

        AR, sigma2 = transform.burg(windowData, filter_order)
        PSD = arma2psd(-AR, NFFT=N_FFT, sides='centerdc')
        # Record the end time
        
        Block_spectrum.append(PSD)
    return np.mean(Block_spectrum, axis=0),Block_spectrum


def compute_psd_combined(Epoch_compute,j,eke,tab,xStart,xEnd,filter_order,N_FFT):
    Block_spectrum_yy = []
    Block_spectrum_xy = []
    for numBlock in tab:
        windowData = Epoch_compute[ j, xStart[numBlock]:xEnd[numBlock]]
        windowData_xx = signal.detrend(windowData, type='constant')

        windowData = Epoch_compute[ eke, xStart[numBlock]:xEnd[numBlock]]
        windowData_yy = signal.detrend(windowData, type='constant')

        CS = np.correlate(windowData_xx, windowData_yy, "same")
        AR, sigma2 = transform.burg(CS, filter_order)
        PSD = arma2psd(-AR, NFFT=N_FFT, sides='centerdc')
        Block_spectrum_xy.append(PSD)

        AR, sigma2 = transform.burg(windowData_yy, filter_order)
        PSD = arma2psd(-AR, NFFT=N_FFT, sides='centerdc')
        Block_spectrum_yy.append(PSD)
    return np.mean(Block_spectrum_xy, axis=0),np.mean(Block_spectrum_yy, axis=0),Block_spectrum_yy,Block_spectrum_xy


def spectral_coh_burg_calculation(Epoch_compute,noverlap,N_FFT,f_max, n_per_seg,freqs_left,filter_order):
    #burg = pburg(Epoch_compute,15,NFFT = nfft_test)
    print(N_FFT)
    print(f_max)
    print(n_per_seg)
    print(filter_order)
    print(noverlap)
    MultithreadVariables.noverlap = noverlap
    MultithreadVariables.N_FFT = N_FFT
    MultithreadVariables.f_max = f_max
    MultithreadVariables.n_per_seg = n_per_seg
    MultithreadVariables.filter_order = filter_order
    splitBranches = [Epoch_compute[i, :, :] for i in range(Epoch_compute.shape[0])]
    print(MultithreadVariables.noverlap)
    print(noverlap)
    num_processes = multiprocessing.cpu_count()  # Use the number of CPU cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(pool.map(spectral_speed_up, splitBranches))

    Full_Mat, Full_time = zip(*results)
    return np.array(Full_Mat),np.array(Full_time)




def spectral_speed_up(Epoch_compute):
    #burg = pburg(Epoch_compute,15,NFFT = nfft_test)
    noverlap = MultithreadVariables.noverlap
    N_FFT = MultithreadVariables.N_FFT
    f_max = MultithreadVariables.f_max
    n_per_seg = MultithreadVariables.n_per_seg
    filter_order = MultithreadVariables.filter_order
    a = Epoch_compute.shape
    print(a)
    M = a[1]  # M = trial length, in samples
    L = n_per_seg  # L = windowing size, in samples
    noverlap = noverlap  # size of overlapping segment btw windows, in samples
    print(noverlap)
    k = round((M-noverlap)/(L-noverlap))  # nb of windows in the trial


    # Arrays of starting and ending indices, for cutting the signal in overlapping chunks
    xStart = np.array(range(0, k*(L-noverlap), L-noverlap))
    xEnd = xStart.copy() + L

    fres = f_max/N_FFT

    tab = np.array(range(k))  # tab = indices of overlapping windows
    trialspectrum = np.zeros([a[0],a[0],N_FFT])
    Time_FC = np.zeros([a[0],a[0],len(tab),N_FFT])

    for j in range(a[0]):
        block_xx_ave,block_xx = compute_psd(Epoch_compute,j,tab,xStart,xEnd,filter_order,N_FFT)
        for eke in range(j+1,a[0]):
            block_xy_ave,block_yy_ave,block_yy,block_xy = compute_psd_combined(Epoch_compute,j,eke,tab,xStart,xEnd,filter_order,N_FFT)

            trialspectrum[j,eke] = abs(block_xy_ave)/np.sqrt(block_xx_ave*block_yy_ave)
            block_tot = abs(np.array(block_xy))/np.sqrt(np.array(block_xx)*np.array(block_yy))
            Time_FC[j,eke] = block_tot
            #print(np.angle(np.array(Block_complex).mean(0)))
    Full_Mat = trialspectrum[:,:,round(f_max/(2*fres)):round(f_max/fres)] + np.transpose(trialspectrum[:,:,round(f_max/(2*fres)):round(f_max/fres)],(1,0,2))
    Full_time = Time_FC[:,:,:,round(f_max/(2*fres)):round(f_max/fres)] + np.transpose(Time_FC[:,:,:,round(f_max/(2*fres)):round(f_max/fres)],(1,0,2,3))
    return Full_Mat,Full_time




#
# def spectral_coh_burg_calculation(Epoch_compute, noverlap, N_FFT, f_max, n_per_seg, freqs_left, filter_order):
#     a = Epoch_compute.shape
#     print(a)
#     M = a[2]
#     L = n_per_seg
#     k = int((M - noverlap) / (L - noverlap))
#
#     xStart = np.arange(0, k * (L - noverlap), L - noverlap)
#     xEnd = xStart + L
#
#     fres = f_max / N_FFT
#     tab = np.arange(k)
#
#     Spec_Coh = np.zeros([a[0], a[1], a[1], N_FFT])
#
#     with Pool(processes=4) as pool:  # Adjust the number of processes as needed
#         for i in range(a[0]):
#             for el in range(a[1]):
#                 Block_spectrum_xx = []
#                 for numBlock in tab:
#                     start = xStart[numBlock]
#                     end = xEnd[numBlock]
#                     epoch_data = Epoch_compute[i, el, start:end]
#                     args = (epoch_data, start, end, N_FFT, filter_order)
#                     PSD = pool.apply(compute_psd, (args,))
#                     Block_spectrum_xx.append(PSD)
#
#                 block_xx = np.mean(Block_spectrum_xx, axis=0)
#
#                 # for j in range(a[1]):
#                 #     Block_spectrum_xy = []
#                 #     Block_spectrum_yy = []
#                 #     for numBlock in tab:
#                 #         windowData_xx = Epoch_compute[i, el, xStart[numBlock]:xEnd[numBlock]]
#                 #         windowData_yy = Epoch_compute[i, j, xStart[numBlock]:xEnd[numBlock]]
#                 #         CS = np.correlate(windowData_xx, windowData_yy, "same")
#                 #
#                 #         args_xy = (CS, start, end, N_FFT, filter_order)
#                 #         PSD_xy = pool.apply(compute_psd, (args_xy,))
#                 #         Block_spectrum_xy.append(PSD_xy)
#                 #
#                 #         args_yy = (windowData_yy, start, end, N_FFT, filter_order)
#                 #         PSD_yy = pool.apply(compute_psd, (args_yy,))
#                 #         Block_spectrum_yy.append(PSD_yy)
#                 #
#                 #     block_xy = np.mean(Block_spectrum_xy, axis=0)
#                 #     block_yy = np.mean(Block_spectrum_yy, axis=0)
#                 #
#                 #     Spec_Coh[i, el, j] = abs(np.array(block_xy)) / np.sqrt(np.array(block_xx) * np.array(block_yy))
#
#     return Spec_Coh[:, :, :, int(f_max / (2 * fres)):int(f_max / fres)]
