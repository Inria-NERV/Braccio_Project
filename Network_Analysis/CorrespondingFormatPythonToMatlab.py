# Filename: dialog.py


"""Dialog-Style application."""

import sys
import os
import time
import numpy as np
import mne
from mne_connectivity import spectral_connectivity
from file_loading import *
import pandas as pd
import os
import matlab.engine
import scipy
eng = matlab.engine.start_matlab()

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


def launch(path,filename_1,filename_2,filename_3,filename_4,filename_5,filename_6,filename_7,filename_8,cond1,cond2,tmin,tmax,nfft,noverlap,nper_seg,fs,filter_order,number_electrodes,car_bool,strat,name_sub):
    f_min_calc = 0
    f_max_calc = 512
    car_bool = False

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

    raw_Training_EDF_1, events_from_annot_1,event_id_1 = load_file(path,filename_1,car_bool)
    raw_Testing_EDF_1, events_from_annot_1_test,event_id_1_test = load_file(path,filename_4,car_bool)
    raw_Testing_EDF_2, events_from_annot_1_test_2,event_id_1_test_2 = load_file(path,filename_7,car_bool)

    Epoch_compute_MI_1 = select_Event(cond1,raw_Training_EDF_1,events_from_annot_1,event_id_1,tmin,tmax,number_electrodes)
    Epoch_compute_MI_1_test = select_Event(cond1,raw_Testing_EDF_1,events_from_annot_1_test,event_id_1_test,tmin,tmax,number_electrodes)
    Epoch_compute_MI_1_test_2 = select_Event(cond1,raw_Testing_EDF_2,events_from_annot_1_test_2,event_id_1_test_2,tmin,tmax,number_electrodes)

    Epoch_compute_MI_1_ba = select_Event(cond1,raw_Training_EDF_1,events_from_annot_1,event_id_1,tmin_baseline,tmax_baseline,number_electrodes)
    Epoch_compute_MI_1_test_ba = select_Event(cond1,raw_Testing_EDF_1,events_from_annot_1_test,event_id_1_test,tmin_baseline,tmax_baseline,number_electrodes)
    Epoch_compute_MI_1_test_2_ba = select_Event(cond1,raw_Testing_EDF_2,events_from_annot_1_test_2,event_id_1_test_2,tmin_baseline,tmax_baseline,number_electrodes)



    Epoch_compute_Rest_1 = select_Event(cond2,raw_Training_EDF_1,events_from_annot_1,event_id_1,tmin,tmax,number_electrodes)
    Epoch_compute_Rest_1_test = select_Event(cond2,raw_Testing_EDF_1,events_from_annot_1_test,event_id_1_test,tmin,tmax,number_electrodes)
    Epoch_compute_Rest_1_test_2 = select_Event(cond2,raw_Testing_EDF_2,events_from_annot_1_test_2,event_id_1_test_2,tmin,tmax,number_electrodes)



    Statistical_variables.raw_run_1_MI = Epoch_compute_MI_1.get_data()[:,:,:]

    Statistical_variables.raw_run_1_Rest = Epoch_compute_Rest_1.get_data()[:,:,:]

    Signal_Rest = Epoch_compute_Rest_1.get_data()[:,:,:]
    Signal_MI = Epoch_compute_MI_1.get_data()[:,:,:]

    Signal_Rest_Test = Epoch_compute_Rest_1_test.get_data()[:,:,:]
    Signal_MI_Test = Epoch_compute_MI_1_test.get_data()[:,:,:]

    Signal_Rest_Test_2 = Epoch_compute_Rest_1_test_2.get_data()[:,:,:]
    Signal_MI_Test_2 = Epoch_compute_MI_1_test_2.get_data()[:,:,:]


    Signal_MI_ba = Epoch_compute_MI_1_ba.get_data()[:,:,:]
    Signal_MI_Test_ba = Epoch_compute_MI_1_test_ba.get_data()[:,:,:]
    Signal_MI_Test_2_ba = Epoch_compute_MI_1_test_2_ba.get_data()[:,:,:]

    if filename_2 != '':
        raw_Training_EDF_2, events_from_annot_2,event_id_2 = load_file(path,filename_2,car_bool)
        Epoch_compute_MI_2 = select_Event(cond1,raw_Training_EDF_2,events_from_annot_2,event_id_2,tmin,tmax,number_electrodes)
        Statistical_variables.raw_run_2_MI = Epoch_compute_MI_2.get_data()[:,:,:]
        Epoch_compute_Rest_2 = select_Event(cond2,raw_Training_EDF_2,events_from_annot_2,event_id_2,tmin,tmax,number_electrodes)
        Statistical_variables.raw_run_2_Rest = Epoch_compute_Rest_2.get_data()[:,:,:]
        Signal_Rest = np.append(Signal_Rest,Epoch_compute_Rest_2.get_data()[:,:,:], axis=0)
        Signal_MI = np.append(Signal_MI,Epoch_compute_MI_2.get_data()[:,:,:], axis=0)


        Epoch_compute_MI_2_ba = select_Event(cond1,raw_Training_EDF_2,events_from_annot_2,event_id_2,tmin_baseline,tmax_baseline,number_electrodes)
        Signal_MI_ba = np.append(Signal_MI_ba,Epoch_compute_MI_2_ba.get_data()[:,:,:], axis=0)


    if filename_3 != '':
        raw_Training_EDF_3, events_from_annot_3,event_id_3 = load_file(path,filename_3,car_bool)
        Epoch_compute_MI_3 = select_Event(cond1,raw_Training_EDF_3,events_from_annot_3,event_id_3,tmin,tmax,number_electrodes)
        Epoch_compute_Rest_3 = select_Event(cond2,raw_Training_EDF_3,events_from_annot_3,event_id_3,tmin,tmax,number_electrodes)
        Statistical_variables.raw_run_3_MI = Epoch_compute_MI_3.get_data()[:,:,:]
        Statistical_variables.raw_run_3_Rest = Epoch_compute_Rest_3.get_data()[:,:,:]
        Signal_Rest = np.append(Signal_Rest,Epoch_compute_Rest_3.get_data()[:,:,:], axis=0)
        Signal_MI = np.append(Signal_MI,Epoch_compute_MI_3.get_data()[:,:,:], axis=0)


        Epoch_compute_MI_3_ba = select_Event(cond1,raw_Training_EDF_3,events_from_annot_3,event_id_3,tmin_baseline,tmax_baseline,number_electrodes)
        Signal_MI_ba = np.append(Signal_MI_ba,Epoch_compute_MI_3_ba.get_data()[:,:,:], axis=0)


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
        raw_Testing_EDF_1_2, events_from_annot_2_2,event_id_2_2 = load_file(path,filename_5,car_bool)
        Epoch_compute_MI_1_2 = select_Event(cond1,raw_Testing_EDF_1_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
        Epoch_compute_Rest_1_2 = select_Event(cond2,raw_Testing_EDF_1_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)

        Signal_Rest_Test = np.append(Signal_Rest_Test,Epoch_compute_Rest_1_2.get_data()[:,:,:], axis=0)
        Signal_MI_Test = np.append(Signal_MI_Test,Epoch_compute_MI_1_2.get_data()[:,:,:], axis=0)

        Epoch_compute_MI_1_2_ba = select_Event(cond1,raw_Testing_EDF_1_2,events_from_annot_2_2,event_id_2_2,tmin_baseline,tmax_baseline,number_electrodes)
        Signal_MI_Test_ba = np.append(Signal_MI_Test_ba,Epoch_compute_MI_1_2_ba.get_data()[:,:,:], axis=0)

    if filename_6 != '':
        raw_Testing_EDF_1_3, events_from_annot_2_3,event_id_2_3 = load_file(path,filename_6,car_bool)
        Epoch_compute_MI_1_3 = select_Event(cond1,raw_Testing_EDF_1_3,events_from_annot_2_3,event_id_2_3,tmin,tmax,number_electrodes)
        Epoch_compute_Rest_1_3 = select_Event(cond2,raw_Testing_EDF_1_3,events_from_annot_2_3,event_id_2_3,tmin,tmax,number_electrodes)

        Signal_Rest_Test = np.append(Signal_Rest_Test,Epoch_compute_Rest_1_3.get_data()[:,:,:], axis=0)
        Signal_MI_Test = np.append(Signal_MI_Test,Epoch_compute_MI_1_3.get_data()[:,:,:], axis=0)

        Epoch_compute_MI_1_3_ba = select_Event(cond1,raw_Testing_EDF_1_3,events_from_annot_2_3,event_id_2_3,tmin_baseline,tmax_baseline,number_electrodes)
        Signal_MI_Test_ba = np.append(Signal_MI_Test_ba,Epoch_compute_MI_1_3_ba.get_data()[:,:,:], axis=0)


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
        raw_Testing_EDF_2_2, events_from_annot_2_2,event_id_2_2 = load_file(path,filename_8,car_bool)
        Epoch_compute_MI_2_2 = select_Event(cond1,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
        Epoch_compute_Rest_2_2 = select_Event(cond2,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)

        Signal_Rest_Test_2 = np.append(Signal_Rest_Test_2,Epoch_compute_Rest_2_2.get_data()[:,:,:], axis=0)
        Signal_MI_Test_2 = np.append(Signal_MI_Test_2,Epoch_compute_MI_2_2.get_data()[:,:,:], axis=0)

        Epoch_compute_MI_2_2_ba = select_Event(cond1,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin_baseline,tmax_baseline,number_electrodes)
        Signal_MI_Test_2_ba =  np.append(Signal_MI_Test_2_ba,Epoch_compute_MI_2_2_ba.get_data()[:,:,:], axis=0)

    Statistical_variables.Raw_Right = Signal_MI
    Statistical_variables.Raw_Left = Signal_Rest

    scipy.io.savemat(path + name_sub+'_Cali_MI.mat', {'mydata': Signal_MI})
    scipy.io.savemat(path + name_sub+'_Cali_Rest.mat', {'mydata': Signal_Rest})

    scipy.io.savemat(path + name_sub+'_Drive_1_MI.mat', {'mydata': Signal_MI_Test})
    scipy.io.savemat(path + name_sub+'_Drive_1_Rest.mat', {'mydata': Signal_Rest_Test})

    scipy.io.savemat(path + name_sub+'_Drive_2_MI.mat', {'mydata': Signal_MI_Test_2})
    scipy.io.savemat(path + name_sub+'_Drive_2_Rest.mat', {'mydata': Signal_Rest_Test_2})
#############################@

    # electrodes = channel_generator(number_electrodes, 'TP10', 'TP9')
    # Bands_inf = [4,8,13,30]
    # Bands_Sup = [7,12,29,40]
    # Coh_Trial_MI = []
    # Coh_Trial_Rest = []
    #
    # ImCoh_Trial_MI = []
    # ImCoh_Trial_Rest = []
    # for k in range(Signal_MI.shape[0]):
    #     Coh_Elec_1_MI = []
    #     Coh_Elec_1_Rest = []
    #     ImCoh_Elec_1_MI = []
    #     ImCoh_Elec_1_Rest = []
    #     for j in range(Signal_MI.shape[1]):
    #         Coh_Elec_2_MI = []
    #         Coh_Elec_2_Rest = []
    #         ImCoh_Elec_2_MI = []
    #         ImCoh_Elec_2_Rest = []
    #         eng.workspace['Signal_MI_per_trial_per_elec'] = eng.double(eng.cell2mat(Signal_MI[k,j,:].tolist()))
    #         eng.workspace['Signal_Rest_per_trial_per_elec'] = eng.double(eng.cell2mat(Signal_Rest[k,j,:].tolist()))
    #         #eng.eval('print(size(Signal_MI_per_trial_per_elec))')
    #         eng.workspace['Welch_MI_per_trial_per_elec'] =eng.eval('pwelch(Signal_MI_per_trial_per_elec,500,0.5,500)')
    #         print("Hello")
    #         eng.workspace['Welch_Rest_per_trial_per_elec'] =eng.eval('pwelch(Signal_Rest_per_trial_per_elec,500,0.5,500)')
    #         for l in range(Signal_MI.shape[1]):
    #             eng.workspace['Signal_MI_per_trial_per_elec_2'] = eng.double(eng.cell2mat(Signal_MI[k,l,:].tolist()))
    #             eng.workspace['Signal_Rest_per_trial_per_elec_2'] = eng.double(eng.cell2mat(Signal_Rest[k,l,:].tolist()))
    #             #eng.eval('print(size(Signal_MI_per_trial_per_elec))')
    #             eng.workspace['Welch_MI_per_trial_per_elec_2'] =eng.eval('pwelch(Signal_MI_per_trial_per_elec_2,500,0.5,500)')
    #             print("Hello")
    #             eng.workspace['Welch_Rest_per_trial_per_elec_2'] =eng.eval('pwelch(Signal_Rest_per_trial_per_elec_2,500,0.5,500)')
    #
    #
    #             if j!=l:
    #                 Coh_MI = eng.eval('abs(cpsd(Signal_MI_per_trial_per_elec,Signal_MI_per_trial_per_elec_2,500,0.5,500)./sqrt(Welch_MI_per_trial_per_elec.*Welch_MI_per_trial_per_elec_2))')
    #                 ImCoh_MI = eng.eval('abs(imag(cpsd(Signal_MI_per_trial_per_elec,Signal_MI_per_trial_per_elec_2,500,0.5,500))./sqrt(Welch_MI_per_trial_per_elec.*Welch_MI_per_trial_per_elec_2))')
    #                 print(len(Coh_MI))
    #                 Coh_Rest = eng.eval('abs(cpsd(Signal_Rest_per_trial_per_elec,Signal_Rest_per_trial_per_elec_2,500,0.5,500)./sqrt(Welch_Rest_per_trial_per_elec.*Welch_Rest_per_trial_per_elec_2))')
    #                 ImCoh_Rest = eng.eval('abs(imag(cpsd(Signal_Rest_per_trial_per_elec,Signal_Rest_per_trial_per_elec_2,500,0.5,500))./sqrt(Welch_Rest_per_trial_per_elec.*Welch_Rest_per_trial_per_elec_2))')
    #             else:
    #                 Coh_MI = np.zeros(251).tolist()
    #                 ImCoh_MI = np.zeros(251).tolist()
    #
    #                 Coh_Rest = np.zeros(251).tolist()
    #                 ImCoh_Rest = np.zeros(251).tolist()
    #
    #             Coh_Elec_2_MI.append(Coh_MI)
    #             ImCoh_Elec_2_MI.append(ImCoh_MI)
    #
    #             Coh_Elec_2_Rest.append(Coh_Rest)
    #             ImCoh_Elec_2_Rest.append(ImCoh_Rest)
    #             break
    #         Coh_Elec_1_MI.append(Coh_Elec_2_MI)
    #         ImCoh_Elec_1_MI.append(ImCoh_Elec_2_MI)
    #
    #         Coh_Elec_1_Rest.append(Coh_Elec_2_Rest)
    #         ImCoh_Elec_1_Rest.append(ImCoh_Elec_2_Rest)
    #         break
    #     Coh_Trial_MI.append(Coh_Elec_1_MI)
    #     Coh_Trial_Rest.append(Coh_Elec_1_Rest)
    #     ImCoh_Trial_MI.append(ImCoh_Elec_1_MI)
    #     ImCoh_Trial_Rest.append(ImCoh_Elec_1_Rest)
    #     break
    # print(Coh_Trial_MI)
    # Coh_Trial_MI_Cali = np.array(Coh_Trial_MI)
    # Coh_Trial_Rest_Cali = np.array(Coh_Trial_Rest)
    #
    # ImCoh_Trial_MI_Cali = np.array(ImCoh_Trial_MI)
    # ImCoh_Trial_Rest_Cali = np.array(ImCoh_Trial_Rest)
    #
    #
    # Coh_Trial_MI = []
    # Coh_Trial_Rest = []
    #
    # ImCoh_Trial_MI = []
    # ImCoh_Trial_Rest = []
    # for k in range(Signal_MI_Test.shape[0]):
    #     Coh_Elec_1_MI = []
    #     Coh_Elec_1_Rest = []
    #     ImCoh_Elec_1_MI = []
    #     ImCoh_Elec_1_Rest = []
    #     for j in range(Signal_MI_Test.shape[1]):
    #         Coh_Elec_2_MI = []
    #         Coh_Elec_2_Rest = []
    #         ImCoh_Elec_2_MI = []
    #         ImCoh_Elec_2_Rest = []
    #         eng.workspace['Signal_MI_per_trial_per_elec'] = eng.double(eng.cell2mat(Signal_MI_Test[k,j,:].tolist()))
    #         eng.workspace['Signal_Rest_per_trial_per_elec'] = eng.double(eng.cell2mat(Signal_Rest_Test[k,j,:].tolist()))
    #         #eng.eval('print(size(Signal_MI_per_trial_per_elec))')
    #         eng.workspace['Welch_MI_per_trial_per_elec'] =eng.eval('pwelch(Signal_MI_per_trial_per_elec,500,0.5,500)')
    #         print("Hello")
    #         eng.workspace['Welch_Rest_per_trial_per_elec'] =eng.eval('pwelch(Signal_Rest_per_trial_per_elec,500,0.5,500)')
    #         for l in range(Signal_MI.shape[1]):
    #             eng.workspace['Signal_MI_per_trial_per_elec_2'] = eng.double(eng.cell2mat(Signal_MI_Test[k,l,:].tolist()))
    #             eng.workspace['Signal_Rest_per_trial_per_elec_2'] = eng.double(eng.cell2mat(Signal_Rest_Test[k,l,:].tolist()))
    #             #eng.eval('print(size(Signal_MI_per_trial_per_elec))')
    #             eng.workspace['Welch_MI_per_trial_per_elec_2'] =eng.eval('pwelch(Signal_MI_per_trial_per_elec_2,500,0.5,500)')
    #             print("Hello")
    #             eng.workspace['Welch_Rest_per_trial_per_elec_2'] =eng.eval('pwelch(Signal_Rest_per_trial_per_elec_2,500,0.5,500)')
    #
    #
    #             if j!=l:
    #                 Coh_MI = eng.eval('abs(cpsd(Signal_MI_per_trial_per_elec,Signal_MI_per_trial_per_elec_2,500,0.5,500)./sqrt(Welch_MI_per_trial_per_elec.*Welch_MI_per_trial_per_elec_2))')
    #                 ImCoh_MI = eng.eval('abs(imag(cpsd(Signal_MI_per_trial_per_elec,Signal_MI_per_trial_per_elec_2,500,0.5,500))./sqrt(Welch_MI_per_trial_per_elec.*Welch_MI_per_trial_per_elec_2))')
    #
    #                 Coh_Rest = eng.eval('abs(cpsd(Signal_Rest_per_trial_per_elec,Signal_Rest_per_trial_per_elec_2,500,0.5,500)./sqrt(Welch_Rest_per_trial_per_elec.*Welch_Rest_per_trial_per_elec_2))')
    #                 ImCoh_Rest = eng.eval('abs(imag(cpsd(Signal_Rest_per_trial_per_elec,Signal_Rest_per_trial_per_elec_2,500,0.5,500))./sqrt(Welch_Rest_per_trial_per_elec.*Welch_Rest_per_trial_per_elec_2))')
    #             else:
    #                 Coh_MI = np.zeros(251).tolist()
    #                 ImCoh_MI = np.zeros(251).tolist()
    #
    #                 Coh_Rest = np.zeros(251).tolist()
    #                 ImCoh_Rest = np.zeros(251).tolist()
    #
    #             Coh_Elec_2_MI.append(Coh_MI)
    #             ImCoh_Elec_2_MI.append(ImCoh_MI)
    #
    #             Coh_Elec_2_Rest.append(Coh_Rest)
    #             ImCoh_Elec_2_Rest.append(ImCoh_Rest)
    #
    #         Coh_Elec_1_MI.append(Coh_Elec_2_MI)
    #         ImCoh_Elec_1_MI.append(ImCoh_Elec_2_MI)
    #
    #         Coh_Elec_1_Rest.append(Coh_Elec_2_Rest)
    #         ImCoh_Elec_1_Rest.append(ImCoh_Elec_2_Rest)
    #     Coh_Trial_MI.append(Coh_Elec_1_MI)
    #     Coh_Trial_Rest.append(Coh_Elec_1_Rest)
    #     ImCoh_Trial_MI.append(ImCoh_Elec_1_MI)
    #     ImCoh_Trial_Rest.append(ImCoh_Elec_1_Rest)
    #
    #
    # Coh_Trial_MI_Drive_1 = np.array(Coh_Trial_MI)
    # Coh_Trial_Rest_Drive_1 = np.array(Coh_Trial_Rest)
    #
    # ImCoh_Trial_MI_Drive_1 = np.array(ImCoh_Trial_MI)
    # ImCoh_Trial_Rest_Drive_1 = np.array(ImCoh_Trial_Rest)
    #
    #
    # Coh_Trial_MI = []
    # Coh_Trial_Rest = []
    #
    # ImCoh_Trial_MI = []
    # ImCoh_Trial_Rest = []
    # for k in range(Signal_MI_Test.shape[0]):
    #     Coh_Elec_1_MI = []
    #     Coh_Elec_1_Rest = []
    #     ImCoh_Elec_1_MI = []
    #     ImCoh_Elec_1_Rest = []
    #     for j in range(Signal_MI_Test.shape[1]):
    #         Coh_Elec_2_MI = []
    #         Coh_Elec_2_Rest = []
    #         ImCoh_Elec_2_MI = []
    #         ImCoh_Elec_2_Rest = []
    #         eng.workspace['Signal_MI_per_trial_per_elec'] = eng.double(eng.cell2mat(Signal_MI_Test_2[k,j,:].tolist()))
    #         eng.workspace['Signal_Rest_per_trial_per_elec'] = eng.double(eng.cell2mat(Signal_Rest_Test_2[k,j,:].tolist()))
    #         #eng.eval('print(size(Signal_MI_per_trial_per_elec))')
    #         eng.workspace['Welch_MI_per_trial_per_elec'] =eng.eval('pwelch(Signal_MI_per_trial_per_elec,500,0.5,500)')
    #         print("Hello")
    #         eng.workspace['Welch_Rest_per_trial_per_elec'] =eng.eval('pwelch(Signal_Rest_per_trial_per_elec,500,0.5,500)')
    #         for l in range(Signal_MI.shape[1]):
    #             eng.workspace['Signal_MI_per_trial_per_elec_2'] = eng.double(eng.cell2mat(Signal_MI_Test_2[k,l,:].tolist()))
    #             eng.workspace['Signal_Rest_per_trial_per_elec_2'] = eng.double(eng.cell2mat(Signal_Rest_Test_2[k,l,:].tolist()))
    #             #eng.eval('print(size(Signal_MI_per_trial_per_elec))')
    #             eng.workspace['Welch_MI_per_trial_per_elec_2'] =eng.eval('pwelch(Signal_MI_per_trial_per_elec_2,500,0.5,500)')
    #             print("Hello")
    #             eng.workspace['Welch_Rest_per_trial_per_elec_2'] =eng.eval('pwelch(Signal_Rest_per_trial_per_elec_2,500,0.5,500)')
    #
    #
    #             if j!=l:
    #
    #                 Coh_MI = eng.eval('abs(cpsd(Signal_MI_per_trial_per_elec,Signal_MI_per_trial_per_elec_2,500,0.5,500)./sqrt(Welch_MI_per_trial_per_elec.*Welch_MI_per_trial_per_elec_2))')
    #                 ImCoh_MI = eng.eval('abs(imag(cpsd(Signal_MI_per_trial_per_elec,Signal_MI_per_trial_per_elec_2,500,0.5,500))./sqrt(Welch_MI_per_trial_per_elec.*Welch_MI_per_trial_per_elec_2))')
    #                 print(len(Coh_MI))
    #                 Coh_Rest = eng.eval('abs(cpsd(Signal_Rest_per_trial_per_elec,Signal_Rest_per_trial_per_elec_2,500,0.5,500)./sqrt(Welch_Rest_per_trial_per_elec.*Welch_Rest_per_trial_per_elec_2))')
    #                 ImCoh_Rest = eng.eval('abs(imag(cpsd(Signal_Rest_per_trial_per_elec,Signal_Rest_per_trial_per_elec_2,500,0.5,500))./sqrt(Welch_Rest_per_trial_per_elec.*Welch_Rest_per_trial_per_elec_2))')
    #             else:
    #                 Coh_MI = np.zeros(251).tolist()
    #                 ImCoh_MI = np.zeros(251).tolist()
    #
    #                 Coh_Rest = np.zeros(251).tolist()
    #                 ImCoh_Rest = np.zeros(251).tolist()
    #
    #             Coh_Elec_2_MI.append(Coh_MI)
    #             ImCoh_Elec_2_MI.append(ImCoh_MI)
    #
    #             Coh_Elec_2_Rest.append(Coh_Rest)
    #             ImCoh_Elec_2_Rest.append(ImCoh_Rest)
    #             break
    #         Coh_Elec_1_MI.append(Coh_Elec_2_MI)
    #         ImCoh_Elec_1_MI.append(ImCoh_Elec_2_MI)
    #
    #         Coh_Elec_1_Rest.append(Coh_Elec_2_Rest)
    #         ImCoh_Elec_1_Rest.append(ImCoh_Elec_2_Rest)
    #         break
    #     Coh_Trial_MI.append(Coh_Elec_1_MI)
    #     Coh_Trial_Rest.append(Coh_Elec_1_Rest)
    #     ImCoh_Trial_MI.append(ImCoh_Elec_1_MI)
    #     ImCoh_Trial_Rest.append(ImCoh_Elec_1_Rest)
    #     break
    # print(Coh_Trial_MI)
    # Coh_Trial_MI_Drive_2 = np.array(Coh_Trial_MI)
    # Coh_Trial_Rest_Drive_2 = np.array(Coh_Trial_Rest)
    #
    # ImCoh_Trial_MI_Drive_2 = np.array(ImCoh_Trial_MI)
    # ImCoh_Trial_Rest_Drive_2 = np.array(ImCoh_Trial_Rest)
    #
    # file_PSD_5_MI= path + name_sub + '_MI_Cali_' + 'Coh' + 'Connectivity.npy'
    # file_PSD_5_test_MI= path + name_sub + '_MI_Drive_1_' + 'Coh' + 'Connectivity.npy'
    # file_PSD_1_test_2_MI= path + name_sub + '_MI_Drive_2_' + 'Coh' + 'Connectivity.npy'
    #
    # np.save(Coh_Trial_MI_Cali,file_PSD_5_MI)
    # np.save(Coh_Trial_MI_Drive_1,file_PSD_5_test_MI)
    # np.save(Coh_Trial_MI_Drive_2,file_PSD_1_test_2_MI)
    #
    # file_PSD_5_Rest= path + name_sub + '_Rest_Cali_' + 'Coh' + 'Connectivity.npy'
    # file_PSD_5_test_Rest= path + name_sub + '_Rest_Drive_1_' + 'Coh' + 'Connectivity.npy'
    # file_PSD_1_test_2_Rest= path + name_sub + '_Rest_Drive_2_' + 'Coh' + 'Connectivity.npy'
    #
    #
    # np.save(Coh_Trial_Rest_Cali,file_PSD_5_MI)
    # np.save(Coh_Trial_Rest_Drive_1,file_PSD_5_test_MI)
    # np.save(Coh_Trial_Rest_Drive_2,file_PSD_1_test_2_MI)
    #
    # file_PSD_5_MI= path + name_sub + '_MI_Cali_' + 'ImCoh' + 'Connectivity.npy'
    # file_PSD_5_test_MI= path + name_sub + '_MI_Drive_1_' + 'ImCoh' + 'Connectivity.npy'
    # file_PSD_1_test_2_MI= path + name_sub + '_MI_Drive_2_' + 'ImCoh' + 'Connectivity.npy'
    #
    # np.save(ImCoh_Trial_MI_Cali,file_PSD_5_MI)
    # np.save(ImCoh_Trial_MI_Drive_1,file_PSD_5_test_MI)
    # np.save(ImCoh_Trial_MI_Drive_2,file_PSD_1_test_2_MI)
    #
    # file_PSD_5_Rest= path + name_sub + '_Rest_Cali_' + 'ImCoh' + 'Connectivity.npy'
    # file_PSD_5_test_Rest= path + name_sub + '_Rest_Drive_1_' + 'ImCoh' + 'Connectivity.npy'
    # file_PSD_1_test_2_Rest= path + name_sub + '_Rest_Drive_2_' + 'ImCoh' + 'Connectivity.npy'
    #
    #
    # np.save(ImCoh_Trial_Rest_Cali,file_PSD_5_MI)
    # np.save(ImCoh_Trial_Rest_Drive_1,file_PSD_5_test_MI)
    # np.save(ImCoh_Trial_Rest_Drive_2,file_PSD_1_test_2_MI)
    # for i in range(len(Bands_inf)):
    #     Connectivity_Cali_MI = spectral_connectivity(Signal_MI, names=None, method=k, indices=None, sfreq=500, mode='multitaper', fmin=Bands_inf[i], fmax=Bands_Sup[i],faverage = True,n_jobs=-1)
    #     Connectivity_Cali_Rest = spectral_connectivity(Signal_Rest, names=None, method=k, indices=None, sfreq=500, mode='multitaper', fmin=Bands_inf[i], fmax=Bands_Sup[i],faverage = True,n_jobs=-1)
    #
    #     Connectivity_Test_MI = spectral_connectivity(Signal_MI_Test, names=None, method=k, indices=None, sfreq=500, mode='multitaper', fmin=Bands_inf[i], fmax=Bands_Sup[i],faverage = True,n_jobs=-1)
    #     Connectivity_Test_Rest = spectral_connectivity(Signal_Rest_Test, names=None, method=k, indices=None, sfreq=500, mode='multitaper', fmin=Bands_inf[i], fmax=Bands_Sup[i],faverage = True,n_jobs=-1)
    #
    #     Connectivity_Test_2_MI = spectral_connectivity(Signal_MI_Test_2, names=None, method=k, indices=None, sfreq=500, mode='multitaper', fmin=Bands_inf[i], fmax=Bands_Sup[i],faverage = True,n_jobs=-1)
    #     Connectivity_Test_2_Rest = spectral_connectivity(Signal_Rest_Test_2, names=None, method=k, indices=None, sfreq=500, mode='multitaper', fmin=Bands_inf[i], fmax=Bands_Sup[i],faverage = True,n_jobs=-1)
    #
    #     Cali_MI.append(Connectivity_Cali_MI.get_data(output='dense'))
    #     Cali_Rest.append(Connectivity_Cali_Rest.get_data(output='dense'))
    #
    #     Drive_1_MI.append(Connectivity_Test_MI.get_data(output='dense'))
    #     Drive_1_Rest.append(Connectivity_Test_Rest.get_data(output='dense'))
    #
    #     Drive_2_MI.append(Connectivity_Test_2_MI.get_data(output='dense'))
    #     Drive_2_Rest.append(Connectivity_Test_2_Rest.get_data(output='dense'))
    #
    #     print(Connectivity_Cali_MI.shape)
    #
    #     #np.save(file_PSD_ca,Power_Right_ba[:,:,:])
    #     #np.save(file_PSD_2_dr1,Power_Right_Test_ba[:,:,:])
    #     #np.save(file_PSD_2_dr2,Power_Right_Test_2_ba[:,:,:])
    #
    #     # file_PSD_1= path + name_sub + 'Cali_Trials_Difference_Bins_Diff.npy'
    #     # file_PSD_2 = path + name_sub + 'Cali_Trials_Bins_Diff.npy'
    #     file_PSD_5= path + name_sub + 'Cali_TimeFrequency.npy'
    #
    #     # file_PSD_1_test= path + name_sub + 'Drive1_Trials_Difference_Bins_Diff.npy'
    #     # file_PSD_2_test = path + name_sub + 'Drive1_Trials_Bins_Diff.npy'
    #     file_PSD_5_test= path + name_sub + 'Drive_1_TimeFrequency.npy'
    #
    #     file_PSD_1_test_2= path + name_sub + 'Drive_2_TimeFrequency.npy'
    #     file_PSD_2_test_2 = path + name_sub + 'Drive2_Trials_Bins_Diff.npy'
    #     file_PSD_5_test_2= path + name_sub + 'Drive2_Bins_Trials_Diff.npy'
    #
    #
    #     # TF_Cali_MI = Connectivity_Cali_MI.get_data(output='dense')
    #     # TF_Test_MI = Connectivity_Test_MI.get_data(output='dense')
    #     # TF_Test_2_MI = Connectivity_Test_2_MI.get_data(output='dense')
    #
    #     # print(TF_Cali_MI)
    #     method = k
    #     file_PSD_5_MI= path + name_sub + '_MI_Cali_' + method + 'Connectivity.npy'
    #     file_PSD_5_test_MI= path + name_sub + '_MI_Drive_1_' + method + 'Connectivity.npy'
    #     file_PSD_1_test_2_MI= path + name_sub + '_MI_Drive_2_' + method + 'Connectivity.npy'
    #
    #     np.save(file_PSD_5_MI,np.array(Cali_MI))
    #     np.save(file_PSD_5_test_MI,np.array(Drive_1_MI))
    #     np.save(file_PSD_1_test_2_MI,np.array(Drive_2_MI))
    #
    #     # TF_Cali_Rest = Connectivity_Cali_Rest.get_data(output='dense')
    #     # TF_Test_Rest = Connectivity_Test_Rest.get_data(output='dense')
    #     # TF_Test_2_Rest = Connectivity_Test_2_Rest.get_data(output='dense')
    #
    #     file_PSD_5_Rest= path + name_sub + '_Rest_Cali_' + method + 'Connectivity.npy'
    #     file_PSD_5_test_Rest= path + name_sub + '_Rest_Drive_1_' + method + 'Connectivity.npy'
    #     file_PSD_1_test_2_Rest= path + name_sub + '_Rest_Drive_2_' + method + 'Connectivity.npy'
    #
    #
    #
    #     np.save(file_PSD_5_Rest,np.array(Cali_Rest))
    #     np.save(file_PSD_5_test_Rest,np.array(Drive_1_Rest))
    #     np.save(file_PSD_1_test_2_Rest,np.array(Drive_2_Rest))




tmin = 0
tmax = 4
nfft = 500
noverlap =0.150
nper_seg = 0.25
fs = 500
filter_order = 19
number_electrodes = 64



path= os.path.dirname(os.path.abspath(__file__))+'/Sub18/ses-01/EEG/'

name_sub = '18_3_'

filename_1='MI-[2022.09.28-15.31.27].ov.edf'
filename_2='MI-[2022.09.28-15.42.05].ov.edf'
filename_3='MI-[2022.09.28-15.54.42].ov.edf'
filename_4='Test-[2022.09.28-16.11.51].edf'
filename_5='Test-[2022.09.28-16.24.23].edf'
filename_6='Test-[2022.09.28-16.37.00].edf'
filename_7='Test-[2022.09.28-16.52.37].edf'
filename_8='Test-[2022.09.28-17.05.05].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub18/ses-02/EEG/'

name_sub = '18_1_'

filename_1 = 	'MI-[2022.10.05-14.12.17].ov.edf'
filename_2 = 	'MI-[2022.10.05-14.22.13].ov.edf'
filename_3 = 	'MI-[2022.10.05-14.32.33].ov.edf'
filename_4 = 	'Test-[2022.10.05-14.49.04].edf'
filename_5 = 	'Test-[2022.10.05-14.59.47].edf'
filename_6 = 	'Test-[2022.10.05-15.10.30].edf'
filename_7= 	'Test-[2022.10.05-15.25.03].edf'
filename_8 = 	'Test-[2022.10.05-15.39.07].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub18/ses-03/EEG/'

name_sub = '18_2_'

filename_1 = 	'MI-[2022.10.12-15.16.53].ov.edf'
filename_2 = 	'MI-[2022.10.12-15.28.27].ov.edf'
filename_3 = 	'MI-[2022.10.12-15.41.44].ov.edf'
filename_4 = 	'Test-[2022.10.12-15.58.13].edf'
filename_5 = 	'Test-[2022.10.12-16.10.14].edf'
filename_6 = 	'Test-[2022.10.12-16.24.53].edf'
filename_7= 	'Test-[2022.10.12-16.39.41].edf'
filename_8 = 	'Test-[2022.10.12-16.52.38].edf'


launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)





path= os.path.dirname(os.path.abspath(__file__))+'/Sub19/ses-01/EEG/'

name_sub = '19_1_'

filename_1='MI-[2022.09.29-15.34.21].ov.edf'
filename_2='MI-[2022.09.29-15.46.27].ov.edf'
filename_3='MI-[2022.09.29-15.57.19].ov.edf'
filename_4='Test-[2022.09.29-16.19.01].edf'
filename_5='Test-[2022.09.29-16.30.00].edf'
filename_6='Test-[2022.09.29-16.43.25].edf'
filename_7='Test-[2022.09.29-17.06.30].edf'
filename_8='Test-[2022.09.29-17.16.47].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub19/ses-02/EEG/'

name_sub = '19_2_'

filename_1='MI-[2022.10.07-15.38.00].ov.edf'
filename_2='MI-[2022.10.07-15.52.13].ov.edf'
filename_3='MI-[2022.10.07-16.05.21].ov.edf'
filename_4='Test-[2022.10.07-16.24.26].edf'
filename_5='Test-[2022.10.07-16.37.34].edf'
filename_6='Test-[2022.10.07-16.56.06].edf'
filename_7='Test-[2022.10.07-17.14.17].edf'
filename_8='Test-[2022.10.07-17.27.38].edf'




launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub19/ses-03/EEG/'

name_sub = '19_3_'

filename_1='MI-[2022.10.13-14.33.11].ov.edf'
filename_2='MI-[2022.10.13-14.44.18].ov.edf'
filename_3='MI-[2022.10.13-14.58.33].ov.edf'
filename_4='Test-[2022.10.13-15.17.42].edf'
filename_5='Test-[2022.10.13-15.31.02].edf'
filename_6='Test-[2022.10.13-15.44.35].edf'
filename_7='Test-[2022.10.13-16.02.07].edf'
filename_8='Test-[2022.10.13-16.14.11].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub20/ses-01/EEG/'

name_sub = '20_1_'

filename_1='MI-[2022.10.14-15.52.25].ov.edf'
filename_2='MI-[2022.10.14-16.07.28].ov.edf'
filename_3='MI-[2022.10.14-16.17.56].ov.edf'
filename_4='Test-[2022.10.14-16.37.53].edf'
filename_5='Test-[2022.10.14-16.48.21].edf'
filename_6='Test-[2022.10.14-16.59.22].edf'
filename_7='Test-[2022.10.14-17.19.17].edf'
filename_8='Test-[2022.10.14-17.29.43].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub20/ses-02/EEG/'

name_sub = '20_3_'

filename_1='MI-[2022.10.21-15.24.45].ov.edf'
filename_2='MI-[2022.10.21-15.36.42].ov.edf'
filename_3='MI-[2022.10.21-15.49.25].ov.edf'
filename_4='Test-[2022.10.21-16.05.05].edf'
filename_5='Test-[2022.10.21-16.17.16].edf'
filename_6='Test-[2022.10.21-16.30.18].edf'
filename_7='Test-[2022.10.21-16.47.42].edf'
filename_8='Test-[2022.10.21-17.00.32].edf'




launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub20/ses-03/EEG/'

name_sub = '20_2_'

filename_1='MI-[2022.10.24-16.00.28].ov.edf'
filename_2='MI-[2022.10.24-16.12.28].ov.edf'
filename_3='MI-[2022.10.24-16.25.44].ov.edf'
filename_4='Test-[2022.10.24-16.39.35].edf'
filename_5='Test-[2022.10.24-16.52.51].edf'
filename_6='Test-[2022.10.24-17.10.01].edf'
filename_7='Test-[2022.10.24-17.25.36].edf'
filename_8='Test-[2022.10.24-17.39.33].edf'



launch(path, filename_1, filename_2,filename_3, filename_4, filename_5,filename_6, filename_7,filename_8,
                                            'OVTK_GDF_Left', 'OVTK_GDF_Right',
                                            tmin, tmax,
                                            nfft, noverlap,
                                            nper_seg, fs,
                                            filter_order, number_electrodes,
                                            False,'1',name_sub)




path= os.path.dirname(os.path.abspath(__file__))+'/Sub21/ses-01/EEG/'

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
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub21/ses-02/EEG/'

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
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub21/ses-03/EEG/'

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
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub22/ses-01/EEG/'

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
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub22/ses-02/EEG/'

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
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub22/ses-03/EEG/'

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
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub23/ses-01/EEG/'

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
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub23/ses-02/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub23/ses-03/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub24/ses-01/EEG/'

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
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub24/ses-02/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub24/ses-03/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub25/ses-01/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub25/ses-02/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub25/ses-03/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub26/ses-01/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub26/ses-02/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub26/ses-03/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub27/ses-01/EEG/'
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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub27/ses-02/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub27/ses-03/EEG/'

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
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub28/ses-01/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub28/ses-02/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub28/ses-03/EEG/'

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
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub29/ses-01/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub29/ses-02/EEG/'

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
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub29/ses-03/EEG/'

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
                                            False,'1',name_sub)



path= os.path.dirname(os.path.abspath(__file__))+'/Sub30/ses-01/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub30/ses-02/EEG/'

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
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub30/ses-03/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub31/ses-01/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub31/ses-02/EEG/'

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
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub31/ses-03/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub32/ses-01/EEG/'

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
                                            False,'1',name_sub)

path= os.path.dirname(os.path.abspath(__file__))+'/Sub32/ses-02/EEG/'

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
                                            False,'1',name_sub)


path= os.path.dirname(os.path.abspath(__file__))+'/Sub32/ses-03/EEG/'

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
                                            False,'1',name_sub)
eng.quit()
