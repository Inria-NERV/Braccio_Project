# Filename: dialog.py


"""Dialog-Style application."""
# Filename: dialog.py


"""Dialog-Style application."""
from mne.minimum_norm import make_inverse_operator, apply_inverse
from scipy import signal,stats,fft
from spectrum import arburg,arma2psd,pburg
import statsmodels.regression.linear_model as transform
from joblib import Parallel, delayed

import sys
import os
import time
import numpy as np
import mne
import matplotlib.pyplot as plt
import pyvista
import pandas as pd
import os
import scipy
import os.path as op
import numpy as np
import mne
from mne.datasets import somato
from mne.time_frequency import tfr_morlet, csd_tfr
from mne.beamformer import make_dics, apply_dics_tfr_epochs
from mne.time_frequency import csd_morlet,csd_multitaper
from mne.beamformer import make_dics, apply_dics_csd
from mne.datasets import fetch_fsaverage
from mne.inverse_sparse import mixed_norm
import nibabel
import imageio
from concurrent.futures import ThreadPoolExecutor


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
        electrodes[index_gnd] = 'Fpz'
        electrodes[index_ref] = 'FCz'

    return electrodes

def load_file(sample_data_folder,filename,car_bool):
    sample_Training_EDF = os.path.join(sample_data_folder, filename)
    raw_Training_EDF = mne.io.read_raw_edf(sample_Training_EDF, preload=True,verbose=False)
    if car_bool:
        raw_Training_EDF_CAR, ref_data = mne.set_eeg_reference(raw_Training_EDF, ref_channels='average')
        events_from_annot_1,event_id_1 = mne.events_from_annotations(raw_Training_EDF_CAR,event_id='auto')
        return raw_Training_EDF_CAR, events_from_annot_1,event_id_1
    else:
        #raw_Training_EDF_CAR, ref_data = mne.set_eeg_reference(raw_Training_EDF, ref_channels='average')
        events_from_annot_1,event_id_1 = mne.events_from_annotations(raw_Training_EDF,event_id='auto')
        return raw_Training_EDF, events_from_annot_1,event_id_1

def load_file_eeg(sample_data_folder,filename):
    sample_Training_EDF = os.path.join(sample_data_folder, filename)
    raw_Training_EEG = mne.io.read_raw_nihon(sample_Training_EDF, preload=True, verbose=False)
    events_from_annot_1,event_id_1 = mne.events_from_annotations(raw_Training_EEG,event_id='auto')
    return raw_Training_EEG, events_from_annot_1,event_id_1


def select_Event(event_name,RAW_data,events_from_annot,event_id,t_min,t_max,number_electrodes):

    epochs_training = mne.Epochs(RAW_data, events_from_annot, event_id,tmin=t_min, tmax=t_max,preload=True,event_repeated='merge',baseline = None,picks = np.arange(0,number_electrodes))

        #epochs_training = mne.Epochs(RAW_data, events_from_annot, event_id,tmin = t_min, tmax=t_max,preload=True,event_repeated='merge')
    return epochs_training[event_name]
class Statistical_variables:
    X_mat=[]
    R_mat = []
    Wsigned = []
    Rsigned = []
    electrodes = []
    power_right = []
    power_left = []
    power_right_Test = []
    power_left_Test = []
    power_right_Test_2 = []
    power_left_Test_2 = []
    freqs_left = []
    elec_2 = []
    timefreq_right = []
    timefreq_left = []
    time_left = []
    Average_rest = []
    fres = 1
    Average_baseline_MI = []
    Average_baseline_Rest = []
    CAR = False
    STD_baseline_MI = []
    STD_baseline_Rest = []
    split = 10
    Name_subject=[]
    raw_run_1_MI = []
    raw_run_2_MI = []
    raw_run_3_MI = []
    raw_run_4_MI = []
    raw_run_5_MI = []
    raw_run_11_MI = []

    raw_run_1_Rest = []
    raw_run_2_Rest = []
    raw_run_3_Rest = []
    raw_run_4_Rest = []
    raw_run_5_Rest = []
    raw_run_11_Rest = []


    Raw_Right =[]
    Raw_Left = []


class Signal_Rest:
    pass


def launch(path,filename_1,filename_2,filename_3,filename_4,filename_5,filename_6,filename_7,filename_8,cond1,cond2,tmin,tmax,nfft,noverlap,nper_seg,fs,filter_order,number_electrodes,car_bool,strat,name_sub,path_to_save):
    f_min_calc = 0
    f_max_calc = 512
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    # The files live in:
    subject = "fsaverage"
    trans = "fsaverage"  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    car_bool = True
    # Download fsaverage files

    # /!\ This script's overlap vs OpenViBE's shift
    # overlap = winlength - shift
    # shift = winlength - overlap
    # converting floats (0.161, 0.25 ...) to samples, we have to round to a neighbouring int
    # OV rounds down the shift, but here we use the overlap. So in order to match,
    # we "round up" the overlap.
    nper_segSamples = int(fs * nper_seg)
    shiftSamples = int(fs * (nper_seg - noverlap))
    noverlapSamples = nper_segSamples - shiftSamples

    freqs_left =np.arange(0,258)
    fres = fs/nfft
    # t_min = None
    # t_max = None
    # pick = None
    # proje = None
    # averag = 'mean'
    # windowing = 'hann'
    smoothing = False
    if strat == 1:
        tmin_baseline = -7.5
        tmax_baseline = -6.5
    else:
        tmin_baseline = -4
        tmax_baseline = -3

    tmin_baseline = -6.5
    tmax_baseline = -5.5

    raw_Training_EDF_1, events_from_annot_1, event_id_1 = load_file(path, filename_1, car_bool)
    raw_Testing_EDF_1, events_from_annot_1_test, event_id_1_test = load_file(path, filename_4, car_bool)
    raw_Testing_EDF_2, events_from_annot_1_test_2, event_id_1_test_2 = load_file(path, filename_7, car_bool)

    Epoch_compute_MI_1 = select_Event(cond1, raw_Training_EDF_1, events_from_annot_1, event_id_1, tmin, tmax,
                                      number_electrodes)
    Epoch_compute_MI_1_test = select_Event(cond1, raw_Testing_EDF_1, events_from_annot_1_test, event_id_1_test, tmin,
                                           tmax, number_electrodes)
    Epoch_compute_MI_1_test_2 = select_Event(cond1, raw_Testing_EDF_2, events_from_annot_1_test_2, event_id_1_test_2,
                                             tmin, tmax, number_electrodes)

    Epoch_compute_MI_1_ba = select_Event(cond1, raw_Training_EDF_1, events_from_annot_1, event_id_1, tmin_baseline,
                                         tmax_baseline, number_electrodes)
    Epoch_compute_MI_1_test_ba = select_Event(cond1, raw_Testing_EDF_1, events_from_annot_1_test, event_id_1_test,
                                              tmin_baseline, tmax_baseline, number_electrodes)
    Epoch_compute_MI_1_test_2_ba = select_Event(cond1, raw_Testing_EDF_2, events_from_annot_1_test_2, event_id_1_test_2,
                                                tmin_baseline, tmax_baseline, number_electrodes)

    Epoch_compute_Rest_1 = select_Event(cond2, raw_Training_EDF_1, events_from_annot_1, event_id_1, tmin, tmax,
                                        number_electrodes)
    Epoch_compute_Rest_1_test = select_Event(cond2, raw_Testing_EDF_1, events_from_annot_1_test, event_id_1_test, tmin,
                                             tmax, number_electrodes)
    Epoch_compute_Rest_1_test_2 = select_Event(cond2, raw_Testing_EDF_2, events_from_annot_1_test_2, event_id_1_test_2,
                                               tmin, tmax, number_electrodes)

    Statistical_variables.raw_run_1_MI = Epoch_compute_MI_1.get_data()[:, :, :]

    Statistical_variables.raw_run_1_Rest = Epoch_compute_Rest_1.get_data()[:, :, :]

    Signal_Rest = Epoch_compute_Rest_1.get_data()[:, :, :]
    Signal_MI = Epoch_compute_MI_1.get_data()[:, :, :]

    Signal_Rest_Test = Epoch_compute_Rest_1_test.get_data()[:, :, :]
    Signal_MI_Test = Epoch_compute_MI_1_test.get_data()[:, :, :]

    Signal_Rest_Test_2 = Epoch_compute_Rest_1_test_2.get_data()[:, :, :]
    Signal_MI_Test_2 = Epoch_compute_MI_1_test_2.get_data()[:, :, :]

    Signal_MI_ba = Epoch_compute_MI_1_ba.get_data()[:, :, :]
    Signal_MI_Test_ba = Epoch_compute_MI_1_test_ba.get_data()[:, :, :]
    Signal_MI_Test_2_ba = Epoch_compute_MI_1_test_2_ba.get_data()[:, :, :]

    if filename_2 != '':
        raw_Training_EDF_2, events_from_annot_2, event_id_2 = load_file(path, filename_2, car_bool)
        Epoch_compute_MI_2 = select_Event(cond1, raw_Training_EDF_2, events_from_annot_2, event_id_2, tmin, tmax,
                                          number_electrodes)
        Statistical_variables.raw_run_2_MI = Epoch_compute_MI_2.get_data()[:, :, :]
        Epoch_compute_Rest_2 = select_Event(cond2, raw_Training_EDF_2, events_from_annot_2, event_id_2, tmin, tmax,
                                            number_electrodes)
        Statistical_variables.raw_run_2_Rest = Epoch_compute_Rest_2.get_data()[:, :, :]
        Signal_Rest = np.append(Signal_Rest, Epoch_compute_Rest_2.get_data()[:, :, :], axis=0)
        Signal_MI = np.append(Signal_MI, Epoch_compute_MI_2.get_data()[:, :, :], axis=0)

        Epoch_compute_MI_2_ba = select_Event(cond1, raw_Training_EDF_2, events_from_annot_2, event_id_2, tmin_baseline,
                                             tmax_baseline, number_electrodes)
        Signal_MI_ba = np.append(Signal_MI_ba, Epoch_compute_MI_2_ba.get_data()[:, :, :], axis=0)

    if filename_3 != '':
        raw_Training_EDF_3, events_from_annot_3, event_id_3 = load_file(path, filename_3, car_bool)
        Epoch_compute_MI_3 = select_Event(cond1, raw_Training_EDF_3, events_from_annot_3, event_id_3, tmin, tmax,
                                          number_electrodes)
        Epoch_compute_Rest_3 = select_Event(cond2, raw_Training_EDF_3, events_from_annot_3, event_id_3, tmin, tmax,
                                            number_electrodes)
        Statistical_variables.raw_run_3_MI = Epoch_compute_MI_3.get_data()[:, :, :]
        Statistical_variables.raw_run_3_Rest = Epoch_compute_Rest_3.get_data()[:, :, :]
        Signal_Rest = np.append(Signal_Rest, Epoch_compute_Rest_3.get_data()[:, :, :], axis=0)
        Signal_MI = np.append(Signal_MI, Epoch_compute_MI_3.get_data()[:, :, :], axis=0)

        Epoch_compute_MI_3_ba = select_Event(cond1, raw_Training_EDF_3, events_from_annot_3, event_id_3, tmin_baseline,
                                             tmax_baseline, number_electrodes)
        Signal_MI_ba = np.append(Signal_MI_ba, Epoch_compute_MI_3_ba.get_data()[:, :, :], axis=0)

    # if filename_4 != '':
    #     raw_Testing_EDF_1_2, events_from_annot_2_2,event_id_2_2 = load_file(path,filename_4,car_bool)
    #     Epoch_compute_MI_1_2 = select_Event(cond1,raw_Testing_EDF_1_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
    #     Epoch_compute_Rest_1_2 = select_Event(cond2,raw_Testing_EDF_1_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
    #
    #     Signal_Rest_Test = np.append(Signal_Rest_Test,Epoch_compute_Rest_1_2.get_data()[:,:,:], axis=0)
    #     Signal_MI_Test = np.append(Signal_MI_Test,Epoch_compute_MI_1_2.get_data()[:,:,:], axis=0)
    #
    #     Epoch_compute_MI_1_2_ba = select_Event(cond1,raw_Testing_EDF_1_2,events_from_annot_2_2,event_id_2_2,tmin_baseline,tmax_baseline,number_electrodes)
    #     Signal_MI_Test_ba = np.append(Signal_MI_Test_ba,Epoch_compute_MI_1_2_ba.get_data()[:,:,:], axis=0)
    #

    if filename_5 != '':
        raw_Testing_EDF_1_2, events_from_annot_2_2, event_id_2_2 = load_file(path, filename_5, car_bool)
        Epoch_compute_MI_1_2 = select_Event(cond1, raw_Testing_EDF_1_2, events_from_annot_2_2, event_id_2_2, tmin, tmax,
                                            number_electrodes)
        Epoch_compute_Rest_1_2 = select_Event(cond2, raw_Testing_EDF_1_2, events_from_annot_2_2, event_id_2_2, tmin,
                                              tmax, number_electrodes)

        Signal_Rest_Test = np.append(Signal_Rest_Test, Epoch_compute_Rest_1_2.get_data()[:, :, :], axis=0)
        Signal_MI_Test = np.append(Signal_MI_Test, Epoch_compute_MI_1_2.get_data()[:, :, :], axis=0)

        Epoch_compute_MI_1_2_ba = select_Event(cond1, raw_Testing_EDF_1_2, events_from_annot_2_2, event_id_2_2,
                                               tmin_baseline, tmax_baseline, number_electrodes)
        Signal_MI_Test_ba = np.append(Signal_MI_Test_ba, Epoch_compute_MI_1_2_ba.get_data()[:, :, :], axis=0)

    if filename_6 != '':
        raw_Testing_EDF_1_3, events_from_annot_2_3, event_id_2_3 = load_file(path, filename_6, car_bool)
        Epoch_compute_MI_1_3 = select_Event(cond1, raw_Testing_EDF_1_3, events_from_annot_2_3, event_id_2_3, tmin, tmax,
                                            number_electrodes)
        Epoch_compute_Rest_1_3 = select_Event(cond2, raw_Testing_EDF_1_3, events_from_annot_2_3, event_id_2_3, tmin,
                                              tmax, number_electrodes)

        Signal_Rest_Test = np.append(Signal_Rest_Test, Epoch_compute_Rest_1_3.get_data()[:, :, :], axis=0)
        Signal_MI_Test = np.append(Signal_MI_Test, Epoch_compute_MI_1_3.get_data()[:, :, :], axis=0)

        Epoch_compute_MI_1_3_ba = select_Event(cond1, raw_Testing_EDF_1_3, events_from_annot_2_3, event_id_2_3,
                                               tmin_baseline, tmax_baseline, number_electrodes)
        Signal_MI_Test_ba = np.append(Signal_MI_Test_ba, Epoch_compute_MI_1_3_ba.get_data()[:, :, :], axis=0)

    # if filename_7 != '':
    #     raw_Testing_EDF_2_2, events_from_annot_2_2,event_id_2_2 = load_file(path,filename_7,car_bool)
    #     Epoch_compute_MI_2_2 = select_Event(cond1,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
    #     Epoch_compute_Rest_2_2 = select_Event(cond2,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
    #
    #     Signal_Rest_Test_2 = np.append(Signal_Rest_Test_2,Epoch_compute_Rest_2_2.get_data()[:,:,:], axis=0)
    #     Signal_MI_Test_2 = np.append(Signal_MI_Test_2,Epoch_compute_MI_2_2.get_data()[:,:,:], axis=0)
    #
    #     Epoch_compute_MI_2_2_ba = select_Event(cond1,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin_baseline,tmax_baseline,number_electrodes)
    #     Signal_MI_Test_2_ba =  np.append(Signal_MI_Test_2_ba,Epoch_compute_MI_2_2_ba.get_data()[:,:,:], axis=0)

    if filename_8 != '':
        raw_Testing_EDF_2_2, events_from_annot_2_2, event_id_2_2 = load_file(path, filename_8, car_bool)
        Epoch_compute_MI_2_2 = select_Event(cond1, raw_Testing_EDF_2_2, events_from_annot_2_2, event_id_2_2, tmin, tmax,
                                            number_electrodes)
        Epoch_compute_Rest_2_2 = select_Event(cond2, raw_Testing_EDF_2_2, events_from_annot_2_2, event_id_2_2, tmin,
                                              tmax, number_electrodes)

        Signal_Rest_Test_2 = np.append(Signal_Rest_Test_2, Epoch_compute_Rest_2_2.get_data()[:, :, :], axis=0)
        Signal_MI_Test_2 = np.append(Signal_MI_Test_2, Epoch_compute_MI_2_2.get_data()[:, :, :], axis=0)

        Epoch_compute_MI_2_2_ba = select_Event(cond1, raw_Testing_EDF_2_2, events_from_annot_2_2, event_id_2_2,
                                               tmin_baseline, tmax_baseline, number_electrodes)
        Signal_MI_Test_2_ba = np.append(Signal_MI_Test_2_ba, Epoch_compute_MI_2_2_ba.get_data()[:, :, :], axis=0)

    # Step 1: Create a numpy array representing your EEG/MEG data.
    # The array should have the shape (n_channels, n_samples), where each row corresponds to a channel.

    # Create an info structure
    sfreq = 500  # Sampling frequency

    electrodes = channel_generator(64, 'TP9', 'TP10')
    biosemi_montage_inter = mne.channels.make_standard_montage('standard_1020')

    ind = [i for (i, channel) in enumerate(biosemi_montage_inter.ch_names) if channel in electrodes]
    biosemi_montage = biosemi_montage_inter.copy()
    # Keep only the desired channels
    biosemi_montage.ch_names = [biosemi_montage_inter.ch_names[x] for x in ind]
    print(biosemi_montage.ch_names)
    kept_channel_info = [biosemi_montage_inter.dig[x + 3] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    biosemi_montage.dig = biosemi_montage_inter.dig[0:3] + kept_channel_info
    # biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    n_channels = len(biosemi_montage.ch_names)
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=500,
                                ch_types='eeg')
    pos = np.stack([biosemi_montage.get_positions()['ch_pos'][ch] for ch in electrodes])
    print(biosemi_montage)
    info.set_montage(biosemi_montage)

    # Create an events array
    n_events = Signal_MI.shape[0]  # Number of events
    event_id = {'event{}'.format(i): i for i in range(n_events)}  # Event IDs
    events = np.column_stack((np.arange(n_events), np.zeros(n_events, dtype=int), np.arange(n_events)))
    # # use the CSDs and the forward model to build the DICS beamformer
    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
    )
    #fwd = mne.convert_forward_solution(fwd, surf_ori=True)
    # Create the Epochs object
    Sig_MI_Reorder = np.zeros((Signal_MI.shape[0],Signal_MI.shape[1],Signal_MI.shape[2]))
    Sig_Rest_Reorder = np.zeros((Signal_Rest.shape[0], Signal_Rest.shape[1], Signal_Rest.shape[2]))

    Sig_MI_Reorder_Test = np.zeros((Signal_MI_Test.shape[0], Signal_MI_Test.shape[1], Signal_MI_Test.shape[2]))
    Sig_Rest_Reorder_Test = np.zeros((Signal_Rest_Test.shape[0], Signal_Rest_Test.shape[1], Signal_Rest_Test.shape[2]))

    Sig_MI_Reorder_Test_2 = np.zeros((Signal_MI_Test_2.shape[0], Signal_MI_Test_2.shape[1], Signal_MI_Test_2.shape[2]))
    Sig_Rest_Reorder_Test_2 = np.zeros((Signal_Rest_Test_2.shape[0], Signal_Rest_Test_2.shape[1], Signal_Rest_Test_2.shape[2]))

    Signal_MI_ba_reorder = np.zeros((Signal_MI_ba.shape[0],Signal_MI_ba.shape[1],Signal_MI_ba.shape[2]))
    Signal_MI_Test_ba_reorder = np.zeros((Signal_MI_Test_ba.shape[0], Signal_MI_Test_ba.shape[1], Signal_MI_Test_ba.shape[2]))
    Signal_MI_Test_2_ba_reorder = np.zeros((Signal_MI_Test_2_ba.shape[0], Signal_MI_Test_2_ba.shape[1], Signal_MI_Test_2_ba.shape[2]))
    for k in range(len(biosemi_montage.ch_names)):
        for l in range(len(electrodes)):
            if biosemi_montage.ch_names[k] == electrodes[l]:
                Sig_MI_Reorder[:,k,:] = Signal_MI[:,l,:]
                Sig_Rest_Reorder[:, k, :] = Signal_Rest[:, l, :]

                Sig_MI_Reorder_Test[:, k, :] = Signal_MI_Test[:, l, :]
                Sig_Rest_Reorder_Test[:, k, :] = Signal_Rest_Test[:, l, :]

                Sig_MI_Reorder_Test_2[:, k, :] = Signal_MI_Test_2[:, l, :]
                Sig_Rest_Reorder_Test_2[:, k, :] = Signal_Rest_Test_2[:, l, :]

                Signal_MI_ba_reorder[:,k,:] = Signal_MI_ba[:, l, :]
                Signal_MI_Test_ba_reorder[:,k,:] = Signal_MI_Test_ba[:, l, :]
                Signal_MI_Test_2_ba_reorder[:,k,:]= Signal_MI_Test_2_ba[:, l, :]



    tmin = 0.0  # Start time of each epoch

    n_events = Signal_MI.shape[0]  # Number of events
    event_id = {'event{}'.format(i): i for i in range(n_events)}  # Event IDs
    events = np.column_stack((np.arange(n_events), np.zeros(n_events, dtype=int), np.arange(n_events)))

    epoch_noise = mne.EpochsArray(Signal_MI_ba_reorder, info, events, tmin, event_id)

    mne.set_eeg_reference(epoch_noise, ref_channels='average', projection=True)
    n_events = Signal_MI_Test.shape[0]  # Number of events
    event_id = {'event{}'.format(i): i for i in range(n_events)}  # Event IDs
    events = np.column_stack((np.arange(n_events), np.zeros(n_events, dtype=int), np.arange(n_events)))

    epoch_noise_Test = mne.EpochsArray(Signal_MI_Test_ba_reorder, info, events, tmin, event_id)
    mne.set_eeg_reference(epoch_noise_Test, ref_channels='average', projection=True)
    n_events = Signal_MI_Test_2.shape[0]  # Number of events
    event_id = {'event{}'.format(i): i for i in range(n_events)}  # Event IDs
    events = np.column_stack((np.arange(n_events), np.zeros(n_events, dtype=int), np.arange(n_events)))

    epoch_noise_Test_2 = mne.EpochsArray(Signal_MI_Test_2_ba_reorder, info, events, tmin, event_id)

    mne.set_eeg_reference(epoch_noise_Test_2, ref_channels='average', projection=True)

    noise_cov = mne.compute_covariance(epoch_noise, method='auto', rank=None)
    noise_cov_Test = mne.compute_covariance(epoch_noise_Test, method='auto', rank=None)
    noise_cov_Test_2 = mne.compute_covariance(epoch_noise_Test_2, method='auto', rank=None)

    inverse_operator = make_inverse_operator(
        info, fwd, noise_cov, loose=0.2, depth=0.8
    )

    inverse_operator_Test = make_inverse_operator(
        info, fwd, noise_cov_Test, loose=0.2, depth=0.8
    )

    inverse_operator_Test_2 = make_inverse_operator(
        info, fwd, noise_cov_Test_2, loose=0.2, depth=0.8
    )
    method = "MNE"
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    PSD_MI_Cali_all = []
    PSD_MI_Test_all = []
    PSD_MI_Test_2_all = []

    PSD_Rest_Cali_all = []
    PSD_Rest_Test_all = []
    PSD_Rest_Test_2_all = []
    for i in range(Sig_MI_Reorder[:,:,:].shape[0]):
        raw_mi =mne.io.RawArray(Sig_MI_Reorder[i,:,:],info)
        raw_rest = mne.io.RawArray(Sig_Rest_Reorder[i, :, :], info)

        raw_mi_test = mne.io.RawArray(Sig_MI_Reorder_Test[i, :, :], info)
        raw_rest_test = mne.io.RawArray(Sig_Rest_Reorder_Test[i, :, :], info)
        if i<20:
            raw_mi_test_2 = mne.io.RawArray(Sig_MI_Reorder_Test_2[i, :, :], info)
            raw_rest_test_2 = mne.io.RawArray(Sig_Rest_Reorder_Test_2[i, :, :], info)
            PSD_MI_Test_2_ave = mne.minimum_norm.compute_source_psd(raw_mi_test_2, inverse_operator, lambda2=lambda2,
                                                                    method=method, tmin=0.0, tmax=None, fmin=13.0,
                                                                    fmax=26.0, n_fft=500, overlap=0.5, pca=True,
                                                                    bandwidth='hann', n_jobs=-1)
            PSD_Rest_Test_2_ave = mne.minimum_norm.compute_source_psd(raw_rest_test_2, inverse_operator,
                                                                      lambda2=lambda2, method=method, tmin=0.0,
                                                                      tmax=None, fmin=13.0, fmax=26.0, n_fft=500,
                                                                      overlap=0.5, pca=True, bandwidth='hann',
                                                                      n_jobs=-1)
            print(PSD_MI_Test_2_ave.shape)
            PSD_MI_Test_2_all.append(PSD_MI_Test_2_ave.data.mean(1))
            PSD_Rest_Test_2_all.append(PSD_Rest_Test_2_ave.data.mean(1))

        PSD_MI_Cali_ave = mne.minimum_norm.compute_source_psd(raw_mi, inverse_operator, lambda2=lambda2, method=method, tmin=0.0, tmax=None, fmin=13.0, fmax=26.0, n_fft=500, overlap=0.5, pca=True, bandwidth='hann', n_jobs = -1)
        PSD_Rest_Cali_ave = mne.minimum_norm.compute_source_psd(raw_rest, inverse_operator, lambda2=lambda2, method=method, tmin=0.0, tmax=None, fmin=13.0, fmax=26.0, n_fft=500, overlap=0.5, pca=True, bandwidth='hann', n_jobs = -1)
        PSD_MI_Cali_all.append(PSD_MI_Cali_ave.data.mean(1))
        PSD_Rest_Cali_all.append(PSD_Rest_Cali_ave.data.mean(1))
        PSD_MI_Test_ave = mne.minimum_norm.compute_source_psd(raw_mi_test, inverse_operator, lambda2=lambda2, method=method, tmin=0.0, tmax=None, fmin=13.0, fmax=26.0, n_fft=500, overlap=0.5, pca=True, bandwidth='hann', n_jobs = -1)
        PSD_Rest_Test_ave = mne.minimum_norm.compute_source_psd(raw_rest_test, inverse_operator, lambda2=lambda2, method=method, tmin=0.0, tmax=None, fmin=13.0, fmax=26.0, n_fft=500, overlap=0.5, pca=True, bandwidth='hann', n_jobs = -1)
        PSD_MI_Test_all.append(PSD_MI_Test_ave.data.mean(1))
        PSD_Rest_Test_all.append(PSD_Rest_Test_ave.data.mean(1))

        print(PSD_MI_Cali_ave.shape)


    np.save(path_to_save + name_sub + 'PSD_M_Cali_Diff_source.npy',(np.array(PSD_MI_Cali_all).mean(0)-np.array(PSD_Rest_Cali_all).mean(0))/np.array(PSD_Rest_Cali_all).mean(0))
    np.save(path_to_save + name_sub + 'PSD_M_Dr1_Diff_source.npy', (np.array(PSD_MI_Test_all).mean(0) - np.array(PSD_Rest_Test_all).mean(0)) / np.array(PSD_Rest_Test_all).mean(0))
    np.save(path_to_save + name_sub + 'PSD_M_Dr2_Diff_source.npy', (np.array(PSD_MI_Test_2_all).mean(0) - np.array(PSD_Rest_Test_2_all).mean(0)) / np.array(PSD_Rest_Test_2_all).mean(0))

tmin = 1
tmax = 4
nfft = 500
noverlap =0.089
nper_seg = 0.25
fs = 500
filter_order = 19
number_electrodes = 64
# 
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub18/ses-01/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub18/ses-01/EEG/'
#
# name_sub = '18_3_'
#
# filename_1='MI-[2022.09.28-15.31.27].ov.edf'
# filename_2='MI-[2022.09.28-15.42.05].ov.edf'
# filename_3='MI-[2022.09.28-15.54.42].ov.edf'
# filename_4='Test-[2022.09.28-16.11.51].edf'
# filename_5='Test-[2022.09.28-16.24.23].edf'
# filename_6='Test-[2022.09.28-16.37.00].edf'
# filename_7='Test-[2022.09.28-16.52.37].edf'
# filename_8='Test-[2022.09.28-17.05.05].edf'
#
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub18/ses-02/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub18/ses-02/EEG/'
#
# name_sub = '18_1_'
#
# filename_1 = 	'MI-[2022.10.05-14.12.17].ov.edf'
# filename_2 = 	'MI-[2022.10.05-14.22.13].ov.edf'
# filename_3 = 	'MI-[2022.10.05-14.32.33].ov.edf'
# filename_4 = 	'Test-[2022.10.05-14.49.04].edf'
# filename_5 = 	'Test-[2022.10.05-14.59.47].edf'
# filename_6 = 	'Test-[2022.10.05-15.10.30].edf'
# filename_7= 	'Test-[2022.10.05-15.25.03].edf'
# filename_8 = 	'Test-[2022.10.05-15.39.07].edf'
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub18/ses-03/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub18/ses-03/EEG/'
#
# name_sub = '18_2_'
#
# filename_1 = 	'MI-[2022.10.12-15.16.53].ov.edf'
# filename_2 = 	'MI-[2022.10.12-15.28.27].ov.edf'
# filename_3 = 	'MI-[2022.10.12-15.41.44].ov.edf'
# filename_4 = 	'Test-[2022.10.12-15.58.13].edf'
# filename_5 = 	'Test-[2022.10.12-16.10.14].edf'
# filename_6 = 	'Test-[2022.10.12-16.24.53].edf'
# filename_7= 	'Test-[2022.10.12-16.39.41].edf'
# filename_8 = 	'Test-[2022.10.12-16.52.38].edf'
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
#
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub19/ses-01/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub19/ses-01/EEG/'
#
# name_sub = '19_1_'
#
# filename_1='MI-[2022.09.29-15.34.21].ov.edf'
# filename_2='MI-[2022.09.29-15.46.27].ov.edf'
# filename_3='MI-[2022.09.29-15.57.19].ov.edf'
# filename_4='Test-[2022.09.29-16.19.01].edf'
# filename_5='Test-[2022.09.29-16.30.00].edf'
# filename_6='Test-[2022.09.29-16.43.25].edf'
# filename_7='Test-[2022.09.29-17.06.30].edf'
# filename_8='Test-[2022.09.29-17.16.47].edf'
#
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub19/ses-02/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub19/ses-02/EEG/'
#
# name_sub = '19_2_'
#
# filename_1='MI-[2022.10.07-15.38.00].ov.edf'
# filename_2='MI-[2022.10.07-15.52.13].ov.edf'
# filename_3='MI-[2022.10.07-16.05.21].ov.edf'
# filename_4='Test-[2022.10.07-16.24.26].edf'
# filename_5='Test-[2022.10.07-16.37.34].edf'
# filename_6='Test-[2022.10.07-16.56.06].edf'
# filename_7='Test-[2022.10.07-17.14.17].edf'
# filename_8='Test-[2022.10.07-17.27.38].edf'
#
#
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub19/ses-03/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub19/ses-03/EEG/'
#
# name_sub = '19_3_'
#
# filename_1='MI-[2022.10.13-14.33.11].ov.edf'
# filename_2='MI-[2022.10.13-14.44.18].ov.edf'
# filename_3='MI-[2022.10.13-14.58.33].ov.edf'
# filename_4='Test-[2022.10.13-15.17.42].edf'
# filename_5='Test-[2022.10.13-15.31.02].edf'
# filename_6='Test-[2022.10.13-15.44.35].edf'
# filename_7='Test-[2022.10.13-16.02.07].edf'
# filename_8='Test-[2022.10.13-16.14.11].edf'
#
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub20/ses-01/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub20/ses-01/EEG/'
#
# name_sub = '20_1_'
#
# filename_1='MI-[2022.10.14-15.52.25].ov.edf'
# filename_2='MI-[2022.10.14-16.07.28].ov.edf'
# filename_3='MI-[2022.10.14-16.17.56].ov.edf'
# filename_4='Test-[2022.10.14-16.37.53].edf'
# filename_5='Test-[2022.10.14-16.48.21].edf'
# filename_6='Test-[2022.10.14-16.59.22].edf'
# filename_7='Test-[2022.10.14-17.19.17].edf'
# filename_8='Test-[2022.10.14-17.29.43].edf'
#
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub20/ses-02/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub20/ses-02/EEG/'
#
# name_sub = '20_3_'
#
# filename_1='MI-[2022.10.21-15.24.45].ov.edf'
# filename_2='MI-[2022.10.21-15.36.42].ov.edf'
# filename_3='MI-[2022.10.21-15.49.25].ov.edf'
# filename_4='Test-[2022.10.21-16.05.05].edf'
# filename_5='Test-[2022.10.21-16.17.16].edf'
# filename_6='Test-[2022.10.21-16.30.18].edf'
# filename_7='Test-[2022.10.21-16.47.42].edf'
# filename_8='Test-[2022.10.21-17.00.32].edf'
#
#
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
#
# path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub20/ses-03/EEG/'
# path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub20/ses-03/EEG/'
#
# name_sub = '20_2_'
#
# filename_1='MI-[2022.10.24-16.00.28].ov.edf'
# filename_2='MI-[2022.10.24-16.12.28].ov.edf'
# filename_3='MI-[2022.10.24-16.25.44].ov.edf'
# filename_4='Test-[2022.10.24-16.39.35].edf'
# filename_5='Test-[2022.10.24-16.52.51].edf'
# filename_6='Test-[2022.10.24-17.10.01].edf'
# filename_7='Test-[2022.10.24-17.25.36].edf'
# filename_8='Test-[2022.10.24-17.39.33].edf'
#
#
#
# launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
#                                             'OVTK_GDF_Left', 'OVTK_GDF_Right',
#                                             tmin, tmax,
#                                             nfft, noverlap,
#                                             nper_seg, fs,
#                                             filter_order, number_electrodes,
#                                             False,'1',name_sub,path_to_save)
#
#
#

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub21/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub21/ses-01/EEG/'

name_sub = '21_3_'

filename_1='MI-[2022.11.29-14.40.15].ov.edf'
filename_2='MI-[2022.11.29-14.50.58].ov.edf'
filename_3='MI-[2022.11.29-15.03.17].ov.edf'
filename_4='Test-[2022.11.29-15.24.31].edf'
filename_5='Test-[2022.11.29-15.36.29].edf'
filename_6='Test-[2022.11.29-15.48.47].edf'
filename_7='Test-[2022.11.29-16.08.32].edf'
filename_8='Test-[2022.11.29-16.22.13].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)



path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub21/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub21/ses-02/EEG/'

name_sub = '21_2_'

filename_1='MI-[2022.12.06-14.24.26].ov.edf'
filename_2='MI-[2022.12.06-14.35.30].ov.edf'
filename_3='MI-[2022.12.06-14.47.44].ov.edf'
filename_4='Test-[2022.12.06-15.10.30].edf'
filename_5='Test-[2022.12.06-15.22.42].edf'
filename_6='Test-[2022.12.06-15.35.31].edf'
filename_7='Test-[2022.12.06-15.53.48].edf'
filename_8='Test-[2022.12.06-16.05.27].edf'




launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)



path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub21/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub21/ses-03/EEG/'

name_sub = '21_1_'

filename_1='Test-[2022.12.14-15.26.56].edf'
filename_2='Test-[2022.12.14-15.37.35].edf'
filename_3='Test-[2022.12.14-15.49.08].edf'
filename_4='Test-[2022.12.14-16.08.40].edf'
filename_5='Test-[2022.12.14-16.19.34].edf'
filename_6='Test-[2022.12.14-16.29.38].edf'
filename_7='Test-[2022.12.14-16.44.27].edf'
filename_8='Test-[2022.12.14-16.54.54].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)



path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub22/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub22/ses-01/EEG/'

name_sub = '22_3_'

filename_1='MI-[2022.11.28-15.24.01].ov.edf'
filename_2='MI-[2022.11.28-15.35.18].ov.edf'
filename_3='MI-[2022.11.28-15.47.35].ov.edf'
filename_4='Test-[2022.11.28-16.06.29].edf'
filename_5='Test-[2022.11.28-16.20.18].edf'
filename_6='Test-[2022.11.28-16.34.45].edf'
filename_7='Test-[2022.11.28-16.52.00].edf'
filename_8='Test-[2022.11.28-17.04.51].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)



path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub22/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub22/ses-02/EEG/'

name_sub = '22_1_'

filename_1='MI-[2022.12.05-15.16.19].ov.edf'
filename_2='MI-[2022.12.05-15.26.40].ov.edf'
filename_3='MI-[2022.12.05-15.37.51].ov.edf'
filename_4='Test-[2022.12.05-15.53.44].edf'
filename_5='Test-[2022.12.05-16.07.31].edf'
filename_6='Test-[2022.12.05-16.17.52].edf'
filename_7='Test-[2022.12.05-16.35.33].edf'
filename_8='Test-[2022.12.05-16.49.13].edf'




launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)



path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub22/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub22/ses-03/EEG/'

name_sub = '22_2_'

filename_1='MI-[2022.12.12-14.59.35].ov.edf'
filename_2='MI-[2022.12.12-15.10.29].ov.edf'
filename_3='MI-[2022.12.12-15.22.26].ov.edf'
filename_4='Test-[2022.12.12-15.43.52].edf'
filename_5='Test-[2022.12.12-15.58.06].edf'
filename_6='Test-[2022.12.12-16.11.11].edf'
filename_7='Test-[2022.12.12-16.25.26].edf'
filename_8='Test-[2022.12.12-16.37.35].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub23/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub23/ses-01/EEG/'

name_sub = '23_2_'

filename_1 = 	'MI-[2022.10.03-15.26.24].ov.edf'
filename_2 = 	'MI-[2022.10.03-15.40.24].ov.edf'
filename_3 = 	'MI-[2022.10.03-15.53.23].ov.edf'
filename_4 = 	'Test-[2022.10.03-16.11.41].edf'
filename_5 = 	'Test-[2022.10.03-16.24.19].edf'
filename_6 = 	'Test-[2022.10.03-16.37.12].edf'
filename_7= 	'Test-[2022.10.03-16.56.41].edf'
filename_8 = 	'Test-[2022.10.03-17.10.13].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub23/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub23/ses-02/EEG/'

name_sub = '23_3_'

filename_1 = 	'MI-[2022.10.10-15.02.37].ov.edf'
filename_2 = 	'MI-[2022.10.10-15.14.31].ov.edf'
filename_3 = 	'MI-[2022.10.10-15.27.23].ov.edf'
filename_4 = 	'Test-[2022.10.10-15.43.06].edf'
filename_5 = 	'Test-[2022.10.10-15.55.20].edf'
filename_6 = 	'Test-[2022.10.10-16.07.20].edf'
filename_7= 	'Test-[2022.10.10-16.25.12].edf'
filename_8 = 	'Test-[2022.10.10-16.38.33].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub23/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub23/ses-03/EEG/'

name_sub = '23_1_'

filename_1 = 	'MI-[2022.10.17-15.02.11].ov.edf'
filename_2 = 	'MI-[2022.10.17-15.12.11].ov.edf'
filename_3 = 	'MI-[2022.10.17-15.23.03].ov.edf'
filename_4 = 	'Test-[2022.10.17-15.37.37].edf'
filename_5 = 	'Test-[2022.10.17-15.48.03].edf'
filename_6 = 	'Test-[2022.10.17-15.58.41].edf'
filename_7= 	'Test-[2022.10.17-16.13.01].edf'
filename_8 = 	'Test-[2022.10.17-16.24.55].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub24/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub24/ses-01/EEG/'

name_sub = '24_1_'

filename_1 =   	'MI-[2022.11.23-14.47.45].ov.edf'
filename_2 = 	'MI-[2022.11.23-14.58.20].ov.edf'
filename_3 = 	'MI-[2022.11.23-15.10.45].ov.edf'
filename_4 = 	'Test-[2022.11.23-15.28.24].edf'
filename_5 = 	'Test-[2022.11.23-15.39.19].edf'
filename_6 = 	'Test-[2022.11.23-15.52.37].edf'
filename_7= 	'Test-[2022.11.23-16.08.03].edf'
filename_8 = 	'Test-[2022.11.23-16.19.58].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub24/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub24/ses-02/EEG/'

name_sub = '24_2_'

filename_1 = 	'MI-[2022.11.30-14.46.00].ov.edf'
filename_2 = 	'MI-[2022.11.30-14.57.05].ov.edf'
filename_3 = 	'MI-[2022.11.30-15.10.08].ov.edf'
filename_4 = 	'Test-[2022.11.30-15.25.25].edf'
filename_5 = 	'Test-[2022.11.30-15.37.38].edf'
filename_6 = 	'Test-[2022.11.30-15.50.30].edf'
filename_7= 	'Test-[2022.11.30-16.06.04].edf'
filename_8 = 	'Test-[2022.11.30-16.18.45].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub24/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub24/ses-03/EEG/'

name_sub = '24_3_'

filename_1 = 	'MI-[2022.12.07-15.29.53].ov.edf'
filename_2 = 	'MI-[2022.12.07-15.40.50].ov.edf'
filename_3 = 	'MI-[2022.12.07-15.53.59].ov.edf'
filename_4 = 	'Test-[2022.12.07-16.19.22].edf'
filename_5 = 	'Test-[2022.12.07-16.32.57].edf'
filename_6 = 	'Test-[2022.12.07-16.44.37].edf'
filename_7= 	'Test-[2022.12.07-16.59.18].edf'
filename_8 = 	'Test-[2022.12.07-17.11.34].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub25/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub25/ses-01/EEG/'

name_sub = '25_3_'

filename_1 = 	'MI-[2022.12.15-15.16.02].ov.edf'
filename_2 = 	'MI-[2022.12.15-15.26.34].ov.edf'
filename_3 = 	'MI-[2022.12.15-15.39.27].ov.edf'
filename_4 = 	'Test-[2022.12.15-15.55.05].edf'
filename_5 = 	'Test-[2022.12.15-16.06.33].edf'
filename_6 = 	'Test-[2022.12.15-16.19.15].edf'
filename_7= 	'Test-[2022.12.15-16.35.28].edf'
filename_8 = 	'Test-[2022.12.15-16.45.53].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub25/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub25/ses-02/EEG/'

name_sub = '25_2_'

filename_1 = 	'MI-[2022.12.21-15.29.29].ov.edf'
filename_2 = 	'MI-[2022.12.21-15.40.04].ov.edf'
filename_3 = 	'MI-[2022.12.21-15.52.28].ov.edf'
filename_4 = 	'Test-[2022.12.21-16.10.49].edf'
filename_5 = 	'Test-[2022.12.21-16.22.52].edf'
filename_6 = 	'Test-[2022.12.21-16.34.54].edf'
filename_7= 	'Test-[2022.12.21-16.47.53].edf'
filename_8 = 	'Test-[2022.12.21-17.00.52].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub25/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub25/ses-03/EEG/'

name_sub = '25_1_'

filename_1 = 	'MI-[2023.01.05-15.17.26].ov.edf'
filename_2 = 	'MI-[2023.01.05-15.29.04].ov.edf'
filename_3 = 	'MI-[2023.01.05-15.39.57].ov.edf'
filename_4 = 	'Test-[2023.01.05-16.08.31].edf'
filename_5 = 	'Test-[2023.01.05-16.19.16].edf'
filename_6 = 	'Test-[2023.01.05-16.31.06].edf'
filename_7= 	'Test-[2023.01.05-16.50.39].edf'
filename_8 = 	'Test-[2023.01.05-17.00.58].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub26/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub26/ses-01/EEG/'

name_sub = '26_2_'

filename_1 = 	'MI-[2022.12.16-15.29.57].ov.edf'
filename_2 = 	'MI-[2022.12.16-15.40.47].ov.edf'
filename_3 = 	'MI-[2022.12.16-15.52.04].ov.edf'
filename_4 = 	'Test-[2022.12.16-16.09.24].edf'
filename_5 = 	'Test-[2022.12.16-16.21.35].edf'
filename_6 = 	'Test-[2022.12.16-16.33.13].edf'
filename_7= 	'Test-[2022.12.16-16.49.15].edf'
filename_8 = 	'Test-[2022.12.16-17.00.32].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub26/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub26/ses-02/EEG/'

name_sub = '26_1_'

filename_1 = 	'MI-[2022.12.23-14.09.51].ov.edf'
filename_2 = 	'MI-[2022.12.23-14.20.08].ov.edf'
filename_3 = 	'MI-[2022.12.23-14.30.34].ov.edf'
filename_4 = 	'Test-[2022.12.23-14.46.24].edf'
filename_5 = 	'Test-[2022.12.23-14.57.23].edf'
filename_6 = 	'Test-[2022.12.23-15.08.06].edf'
filename_7= 	'Test-[2022.12.23-15.25.17].edf'
filename_8 = 	'Test-[2022.12.23-15.35.26].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub26/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub26/ses-03/EEG/'

name_sub = '26_3_'

filename_1 = 	'MI-[2022.12.30-15.24.08].ov.edf'
filename_2 = 	'MI-[2022.12.30-15.36.17].ov.edf'
filename_3 = 	'MI-[2022.12.30-15.52.32].ov.edf'
filename_4 = 	'Test-[2022.12.30-16.09.05].edf'
filename_5 = 	'Test-[2022.12.30-16.22.43].edf'
filename_6 = 	'Test-[2022.12.30-16.36.37].edf'
filename_7= 	'Test-[2022.12.30-16.57.23].edf'
filename_8 = 	'Test-[2022.12.30-17.10.34].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub27/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub27/ses-01/EEG/'

name_sub = '27_2_'
filename_1 =   	'MI-[2022.12.28-14.54.27].ov.edf'
filename_2 = 	'MI-[2022.12.28-15.05.19].ov.edf'
filename_3 = 	'MI-[2022.12.28-15.18.06].ov.edf'
filename_4 = 	'Test-[2022.12.28-15.34.29].edf'
filename_5 = 	'Test-[2022.12.28-15.47.07].edf'
filename_6 = 	'Test-[2022.12.28-15.59.41].edf'
filename_7= 	'Test-[2022.12.28-16.16.52].edf'
filename_8 = 	'Test-[2022.12.28-16.29.52].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub27/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub27/ses-02/EEG/'

name_sub = '27_1_'

filename_1 = 	'MI-[2023.01.06-15.16.11].ov.edf'
filename_2 = 	'Test-[2023.01.06-15.32.14].edf'
filename_3 = 	'Test-[2023.01.06-15.43.22].edf'
filename_4 = 	'Test-[2023.01.06-15.56.44].edf'
filename_5 = 	'Test-[2023.01.06-16.07.29].edf'
filename_6 = 	'Test-[2023.01.06-16.18.57].edf'
filename_7= 	'Test-[2023.01.06-16.33.10].edf'
filename_8 = 	'Test-[2023.01.06-16.43.56].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub27/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub27/ses-03/EEG/'

name_sub = '27_3_'

filename_1 = 	'MI-[2023.01.11-14.50.51].ov.edf'
filename_2 = 	'MI-[2023.01.11-15.01.29].ov.edf'
filename_3 = 	'MI-[2023.01.11-15.12.59].ov.edf'
filename_4 = 	'Test-[2023.01.11-15.28.41].edf'
filename_5 = 	'Test-[2023.01.11-15.39.37].edf'
filename_6 = 	'Test-[2023.01.11-15.51.32].edf'
filename_7= 	'Test-[2023.01.11-16.06.09].edf'
filename_8 = 	'Test-[2023.01.11-16.17.55].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub28/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub28/ses-01/EEG/'

name_sub = '28_1_'

filename_1 = 	'MI-[2022.12.27-14.39.56].ov.edf'
filename_2 = 	'MI-[2022.12.27-14.51.26].ov.edf'
filename_3 = 	'MI-[2022.12.27-15.02.48].ov.edf'
filename_4 = 	'Test-[2022.12.27-15.29.27].edf'
filename_5 = 	'Test-[2022.12.27-15.42.31].edf'
filename_6 = 	'Test-[2022.12.27-15.53.23].edf'
filename_7= 	'Test-[2022.12.27-16.11.18].edf'
filename_8 = 	'Test-[2022.12.27-16.21.26].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub28/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub28/ses-02/EEG/'

name_sub = '28_3_'

filename_1 = 	'MI-[2023.01.03-14.36.05].ov.edf'
filename_2 = 	'MI-[2023.01.03-14.46.54].ov.edf'
filename_3 = 	'MI-[2023.01.03-14.58.51].ov.edf'
filename_4 = 	'Test-[2023.01.03-15.13.57].edf'
filename_5 = 	'Test-[2023.01.03-15.26.20].edf'
filename_6 = 	'Test-[2023.01.03-15.38.09].edf'
filename_7= 	'Test-[2023.01.03-15.54.24].edf'
filename_8 = 	'Test-[2023.01.03-16.06.42].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub28/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub28/ses-03/EEG/'

name_sub = '28_2_'

filename_1 = 	'MI-[2023.01.17-14.33.59].ov.edf'
filename_2 = 	'MI-[2023.01.17-14.45.36].ov.edf'
filename_3 = 	'MI-[2023.01.17-14.57.52].ov.edf'
filename_4 = 	'Test-[2023.01.17-15.12.48].edf'
filename_5 = 	'Test-[2023.01.17-15.24.35].edf'
filename_6 = 	'Test-[2023.01.17-15.36.21].edf'
filename_7= 	'Test-[2023.01.17-15.51.36].edf'
filename_8 = 	'Test-[2023.01.17-16.03.14].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)



path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub29/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub29/ses-01/EEG/'

name_sub = '29_2_'

filename_1 = 	'MI-[2023.02.13-15.34.52].ov.edf'
filename_2 = 	'MI-[2023.02.13-15.46.59].ov.edf'
filename_3 = 	'MI-[2023.02.13-15.58.40].ov.edf'
filename_4 = 	'Test-[2023.02.13-16.16.05].edf'
filename_5 = 	'Test-[2023.02.13-16.28.30].edf'
filename_6 = 	'Test-[2023.02.13-16.39.43].edf'
filename_7= 	'Test-[2023.02.13-16.53.29].edf'
filename_8 = 	'Test-[2023.02.13-17.04.31].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub29/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub29/ses-02/EEG/'

name_sub = '29_3_'

filename_1 = 	'MI-[2023.02.21-15.15.13].ov.edf'
filename_2 = 	'MI-[2023.02.21-15.25.46].ov.edf'
filename_3 = 	'MI-[2023.02.21-15.37.28].ov.edf'
filename_4 = 	'Test-[2023.02.21-15.51.15].edf'
filename_5 = 	'Test-[2023.02.21-16.02.37].edf'
filename_6 = 	'Test-[2023.02.21-16.14.03].edf'
filename_7= 	'Test-[2023.02.21-16.27.49].edf'
filename_8 = 	'Test-[2023.02.21-16.39.14].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub29/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub29/ses-03/EEG/'

name_sub = '29_1_'

filename_1 = 	'MI-[2023.02.27-15.20.30].ov.edf'
filename_2 = 	'MI-[2023.02.27-15.30.08].ov.edf'
filename_3 = 	'MI-[2023.02.27-15.42.04].ov.edf'
filename_4 = 	'Test-[2023.02.27-15.56.17].edf'
filename_5 = 	'Test-[2023.02.27-16.06.27].edf'
filename_6 = 	'Test-[2023.02.27-16.17.15].edf'
filename_7= 	'Test-[2023.02.27-16.32.12].edf'
filename_8 = 	'Test-[2023.02.27-16.42.15].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)



path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub30/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub30/ses-01/EEG/'

name_sub = '30_2_'

filename_1 = 	'MI-[2023.02.15-14.21.52].ov.edf'
filename_2 = 	'MI-[2023.02.15-14.33.20].ov.edf'
filename_3 = 	'MI-[2023.02.15-14.45.57].ov.edf'
filename_4 = 	'Test-[2023.02.15-15.05.24].edf'
filename_5 = 	'Test-[2023.02.15-15.17.52].edf'
filename_6 = 	'Test-[2023.02.15-15.29.21].edf'
filename_7= 	'Test-[2023.02.15-15.44.02].edf'
filename_8 = 	'Test-[2023.02.15-15.57.22].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub30/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub30/ses-02/EEG/'

name_sub = '30_1_'

filename_1 = 	'MI-[2023.02.22-14.17.45].ov.edf'
filename_2 = 	'MI-[2023.02.22-14.29.42].ov.edf'
filename_3 = 	'MI-[2023.02.22-14.42.23].ov.edf'
filename_4 = 	'Test-[2023.02.22-14.57.14].edf'
filename_5 = 	'Test-[2023.02.22-15.07.43].edf'
filename_6 = 	'Test-[2023.02.22-15.18.57].edf'
filename_7= 	'Test-[2023.02.22-15.36.42].edf'
filename_8 = 	'Test-[2023.02.22-15.47.06].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub30/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub30/ses-03/EEG/'

name_sub = '30_3_'

filename_1 = 	'MI-[2023.03.01-14.10.48].ov.edf'
filename_2 = 	'MI-[2023.03.01-14.21.47].ov.edf'
filename_3 = 	'MI-[2023.03.01-14.32.41].ov.edf'
filename_4 = 	'Test-[2023.03.01-15.23.52].edf'
filename_5 = 	'Test-[2023.03.01-15.36.21].edf'
filename_6 = 	'Test-[2023.03.01-15.47.41].edf'
filename_7= 	'Test-[2023.03.01-15.59.46].edf'
filename_8 = 	'Test-[2023.03.01-16.10.20].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub31/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub31/ses-01/EEG/'

name_sub = '31_1_'

filename_1 = 	'MI-[2023.02.17-14.38.30].ov.edf'
filename_2 = 	'MI-[2023.02.17-14.49.11].ov.edf'
filename_3 = 	'MI-[2023.02.17-15.01.36].ov.edf'
filename_4 = 	'Test-[2023.02.17-15.34.16].edf'
filename_5 = 	'Test-[2023.02.17-15.44.35].edf'
filename_6 = 	'Test-[2023.02.17-15.56.21].edf'
filename_7= 	'Test-[2023.02.17-16.10.28].edf'
filename_8 = 	'Test-[2023.02.17-16.20.34].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub31/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub31/ses-02/EEG/'

name_sub = '31_3_'

filename_1 = 	'MI-[2023.02.24-14.27.56].ov.edf'
filename_2 = 	'MI-[2023.02.24-14.38.48].ov.edf'
filename_3 = 	'MI-[2023.02.24-14.50.12].ov.edf'
filename_4 = 	'Test-[2023.02.24-15.05.20].edf'
filename_5 = 	'Test-[2023.02.24-15.16.44].edf'
filename_6 = 	'Test-[2023.02.24-15.28.31].edf'
filename_7= 	'Test-[2023.02.24-15.45.42].edf'
filename_8 = 	'Test-[2023.02.24-15.56.49].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub31/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub31/ses-03/EEG/'

name_sub = '31_2_'

filename_1 = 	'MI-[2023.03.03-14.29.26].ov.edf'
filename_2 = 	'MI-[2023.03.03-14.39.35].ov.edf'
filename_3 = 	'MI-[2023.03.03-14.50.40].ov.edf'
filename_4 = 	'Test-[2023.03.03-15.05.01].edf'
filename_5 = 	'Test-[2023.03.03-15.15.41].edf'
filename_6 = 	'Test-[2023.03.03-15.26.40].edf'
filename_7= 	'Test-[2023.03.03-15.40.11].edf'
filename_8 = 	'Test-[2023.03.03-15.50.44].edf'

launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub32/ses-01/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub32/ses-01/EEG/'

name_sub = '32_3_'

filename_1 = 	'MI-[2023.03.06-15.48.14].ov.edf'
filename_2 = 	'MI-[2023.03.06-15.58.38].ov.edf'
filename_3 = 	'MI-[2023.03.06-16.09.29].ov.edf'
filename_4 = 	'Test-[2023.03.06-16.30.15].edf'
filename_5 = 	'Test-[2023.03.06-16.40.47].edf'
filename_6 = 	'Test-[2023.03.06-16.51.30].edf'
filename_7= 	'Test-[2023.03.06-17.04.51].edf'
filename_8 = 	'Test-[2023.03.06-17.15.49].edf'

launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)

path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub32/ses-02/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub32/ses-02/EEG/'

name_sub = '32_2_'

filename_1 = 	'MI-[2023.03.15-15.20.48].ov.edf'
filename_2 = 	'MI-[2023.03.15-15.31.21].ov.edf'
filename_3 = 	'MI-[2023.03.15-15.43.48].ov.edf'
filename_4 = 	'Test-[2023.03.15-15.58.17].edf'
filename_5 = 	'Test-[2023.03.15-16.10.43].edf'
filename_6 = 	'Test-[2023.03.15-16.22.59].edf'
filename_7= 	'Test-[2023.03.15-16.38.46].edf'
filename_8 = 	'Test-[2023.03.15-16.50.04].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)


path= os.path.dirname(os.path.abspath(__file__)) +'/Batch2'+'/Sub32/ses-03/EEG/'
path_to_save= os.path.dirname(os.path.abspath(__file__)) +'/Batch2_simplified'+'/Sub32/ses-03/EEG/'

name_sub = '32_1_'

filename_1 = 	'MI-[2023.03.20-15.19.06].ov.edf'
filename_2 = 	'MI-[2023.03.20-15.29.04].ov.edf'
filename_3 = 	'MI-[2023.03.20-15.39.33].ov.edf'
filename_4 = 	'Test-[2023.03.20-15.54.46].edf'
filename_5 = 	'Test-[2023.03.20-16.05.12].edf'
filename_6 = 	'Test-[2023.03.20-16.15.41].edf'
filename_7= 	'Test-[2023.03.20-16.29.10].edf'
filename_8 = 	'Test-[2023.03.20-16.39.53].edf'

launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub,path_to_save)
