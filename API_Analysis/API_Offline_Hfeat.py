# Filename: dialog.py


"""Dialog-Style application."""

import sys
import os
import time
import numpy as np
import mne
from mne_connectivity.viz import circular_layout, plot_connectivity_circle
from sklearn.covariance import ledoit_wolf
from Topomap_separate import *
from Spectral_Analysis import *
from Statistical_analysis import *
from file_loading import *
from Classification import *
from scipy.interpolate import interp1d
import pandas as pd
from Optimization_feature_functions import *
from itertools import combinations
from PyQt5.QtWidgets import QApplication,QMessageBox,QLabel,QHBoxLayout,QCheckBox
#from fc_pipeline import *
from PyQt5.QtWidgets import QDialog

from PyQt5.QtWidgets import QPushButton

from PyQt5.QtWidgets import QFormLayout

from PyQt5.QtWidgets import QLineEdit

from PyQt5.QtWidgets import QVBoxLayout


def btnstate(b):
    if b.isChecked() == True:
        Statistical_variables.CAR = True
    else:
        Statistical_variables.CAR = False


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
    print("---LAUNCH")
    print("CAR: " + str(car_bool))
    print("tmin: " + str(tmin))
    print("tmax: " + str(tmax))
    print("nfft: " + str(nfft))
    print("fs: " + str(fs))
    print("filter_order: " + str(filter_order))

    # /!\ This script's overlap vs OpenViBE's shift
    # overlap = winlength - shift
    # shift = winlength - overlap
    # converting floats (0.161, 0.25 ...) to samples, we have to round to a neighbouring int
    # OV rounds down the shift, but here we use the overlap. So in order to match,
    # we "round up" the overlap.
    nper_segSamples = int(fs * nper_seg)
    shiftSamples = int(fs * (nper_seg - noverlap))
    noverlapSamples = nper_segSamples - shiftSamples
    print("noverlap: " + str(noverlap) + " (" + str(noverlapSamples) + " samples)")
    print("nper_seg: " + str(nper_seg) + " (" + str(nper_segSamples) + " samples)")
    print("shift used for computation: " + str(shiftSamples) + " samples)")
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
    raw_Training_EDF_1, events_from_annot_1,event_id_1 = load_file(path,filename_1,car_bool)
    raw_Testing_EDF_1, events_from_annot_1_test,event_id_1_test = load_file(path,filename_4,car_bool)
    raw_Testing_EDF_2, events_from_annot_1_test_2,event_id_1_test_2 = load_file(path,filename_7,car_bool)


    raw_Training_EDF_1.plot(duration=10,start = 135,n_channels = 10,scalings="auto")
    plt.show()
    Epoch_compute_MI_1 = select_Event(cond1,raw_Training_EDF_1,events_from_annot_1,event_id_1,tmin,tmax,number_electrodes)
    Epoch_compute_MI_1_test = select_Event(cond1,raw_Testing_EDF_1,events_from_annot_1_test,event_id_1_test,tmin,tmax,number_electrodes)
    Epoch_compute_MI_1_test_2 = select_Event(cond1,raw_Testing_EDF_2,events_from_annot_1_test_2,event_id_1_test_2,tmin,tmax,number_electrodes)

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

    if filename_2 != '':
        raw_Training_EDF_2, events_from_annot_2,event_id_2 = load_file(path,filename_2,car_bool)
        Epoch_compute_MI_2 = select_Event(cond1,raw_Training_EDF_2,events_from_annot_2,event_id_2,tmin,tmax,number_electrodes)
        Statistical_variables.raw_run_2_MI = Epoch_compute_MI_2.get_data()[:,:,:]
        Epoch_compute_Rest_2 = select_Event(cond2,raw_Training_EDF_2,events_from_annot_2,event_id_2,tmin,tmax,number_electrodes)
        Statistical_variables.raw_run_2_Rest = Epoch_compute_Rest_2.get_data()[:,:,:]
        Signal_Rest = np.append(Signal_Rest,Epoch_compute_Rest_2.get_data()[:,:,:], axis=0)
        Signal_MI = np.append(Signal_MI,Epoch_compute_MI_2.get_data()[:,:,:], axis=0)





    if filename_3 != '':
        raw_Training_EDF_3, events_from_annot_3,event_id_3 = load_file(path,filename_3,car_bool)
        Epoch_compute_MI_3 = select_Event(cond1,raw_Training_EDF_3,events_from_annot_3,event_id_3,tmin,tmax,number_electrodes)
        Epoch_compute_Rest_3 = select_Event(cond2,raw_Training_EDF_3,events_from_annot_3,event_id_3,tmin,tmax,number_electrodes)
        Statistical_variables.raw_run_3_MI = Epoch_compute_MI_3.get_data()[:,:,:]
        Statistical_variables.raw_run_3_Rest = Epoch_compute_Rest_3.get_data()[:,:,:]
        Signal_Rest = np.append(Signal_Rest,Epoch_compute_Rest_3.get_data()[:,:,:], axis=0)
        Signal_MI = np.append(Signal_MI,Epoch_compute_MI_3.get_data()[:,:,:], axis=0)




    # if filename_4 != '':
    #     raw_Testing_EDF_1_2, events_from_annot_2_2,event_id_2_2 = load_file(path,filename_4,car_bool)
    #     Epoch_compute_MI_1_2 = select_Event(cond1,raw_Testing_EDF_1_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
    #     Epoch_compute_Rest_1_2 = select_Event(cond2,raw_Testing_EDF_1_2,events_from_annot_2,event_id_2,tmin,tmax,number_electrodes)
    #
    #     Signal_Rest_Test = np.append(Signal_Rest_Test,Epoch_compute_Rest_1_2.get_data()[:,:,:], axis=0)
    #     Signal_MI_Test = np.append(Signal_MI_Test,Epoch_compute_MI_1_2.get_data()[:,:,:], axis=0)



    if filename_5 != '':
        raw_Testing_EDF_1_2, events_from_annot_2_2,event_id_2_2 = load_file(path,filename_5,car_bool)
        Epoch_compute_MI_1_2 = select_Event(cond1,raw_Testing_EDF_1_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
        Epoch_compute_Rest_1_2 = select_Event(cond2,raw_Testing_EDF_1_2,events_from_annot_2,event_id_2,tmin,tmax,number_electrodes)

        Signal_Rest_Test = np.append(Signal_Rest_Test,Epoch_compute_Rest_1_2.get_data()[:,:,:], axis=0)
        Signal_MI_Test = np.append(Signal_MI_Test,Epoch_compute_MI_1_2.get_data()[:,:,:], axis=0)


    if filename_6 != '':
        raw_Testing_EDF_1_3, events_from_annot_2_3,event_id_2_3 = load_file(path,filename_6,car_bool)
        Epoch_compute_MI_1_3 = select_Event(cond1,raw_Testing_EDF_1_3,events_from_annot_2_3,event_id_2_3,tmin,tmax,number_electrodes)
        Epoch_compute_Rest_1_3 = select_Event(cond2,raw_Testing_EDF_1_3,events_from_annot_2_3,event_id_2_3,tmin,tmax,number_electrodes)

        Signal_Rest_Test = np.append(Signal_Rest_Test,Epoch_compute_Rest_1_3.get_data()[:,:,:], axis=0)
        Signal_MI_Test = np.append(Signal_MI_Test,Epoch_compute_MI_1_3.get_data()[:,:,:], axis=0)


    # if filename_7 != '':
    #     raw_Testing_EDF_2_2, events_from_annot_2_2,event_id_2_2 = load_file(path,filename_7,car_bool)
    #     Epoch_compute_MI_2_2 = select_Event(cond1,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
    #     Epoch_compute_Rest_2_2 = select_Event(cond2,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
    #
    #     Signal_Rest_Test_2 = np.append(Signal_Rest_Test_2,Epoch_compute_Rest_2_2.get_data()[:,:,:], axis=0)
    #     Signal_MI_Test_2 = np.append(Signal_MI_Test_2,Epoch_compute_MI_2_2.get_data()[:,:,:], axis=0)
    #

    if filename_8 != '':
        raw_Testing_EDF_2_2, events_from_annot_2_2,event_id_2_2 = load_file(path,filename_8,car_bool)
        Epoch_compute_MI_2_2 = select_Event(cond1,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)
        Epoch_compute_Rest_2_2 = select_Event(cond2,raw_Testing_EDF_2_2,events_from_annot_2_2,event_id_2_2,tmin,tmax,number_electrodes)

        Signal_Rest_Test_2 = np.append(Signal_Rest_Test_2,Epoch_compute_Rest_2_2.get_data()[:,:,:], axis=0)
        Signal_MI_Test_2 = np.append(Signal_MI_Test_2,Epoch_compute_MI_2_2.get_data()[:,:,:], axis=0)




    Statistical_variables.Raw_Right = Signal_MI
    Statistical_variables.Raw_Left = Signal_Rest


#############################@

    electrodes = channel_generator(number_electrodes, 'TP9', 'TP10')


    Power_Left_1,timefreq_left,time_left = \
        Power_burg_calculation(Signal_Rest,noverlapSamples,nfft,f_max_calc, nper_segSamples, smoothing,freqs_left,filter_order)
    Power_Right_1,timefreq_right,time_left= \
        Power_burg_calculation(Signal_MI,noverlapSamples,nfft,f_max_calc, nper_segSamples, smoothing,freqs_left,filter_order)

    Power_Left_Test_1,timefreq_left_test,time_left_test = \
        Power_burg_calculation(Signal_Rest_Test,noverlapSamples,nfft,f_max_calc, nper_segSamples, smoothing,freqs_left,filter_order)
    Power_Right_1_Test,timefreq_right_test,time_left= \
        Power_burg_calculation(Signal_MI_Test,noverlapSamples,nfft,f_max_calc, nper_segSamples, smoothing,freqs_left,filter_order)

    Power_Left_Test_2,timefreq_left_test_2,time_left_test_2 = \
        Power_burg_calculation(Signal_Rest_Test_2,noverlapSamples,nfft,f_max_calc, nper_segSamples, smoothing,freqs_left,filter_order)
    Power_Right_2_Test,timefreq_right_test_2,time_left_2= \
        Power_burg_calculation(Signal_MI_Test_2,noverlapSamples,nfft,f_max_calc, nper_segSamples, smoothing,freqs_left,filter_order)




    Power_Left_1 = np.nan_to_num(Power_Left_1)
    Power_Right_1 = np.nan_to_num(Power_Right_1)

    Power_Left_Test_1 = np.nan_to_num(Power_Left_Test_1)
    Power_Right_1_Test = np.nan_to_num(Power_Right_1_Test)

    Power_Left_Test_2 = np.nan_to_num(Power_Left_Test_2)
    Power_Right_2_Test = np.nan_to_num(Power_Right_2_Test)

    Rsigned = Compute_Rsquare_Map_Welch(Power_Right_1[:,:,:],Power_Left_1[:,:,:])
    Wsquare,Wpvalues = Compute_Wilcoxon_Map(Power_Right_1[:,:,:],Power_Left_1[:,:,:])

    Rsigned_test = Compute_Rsquare_Map_Welch(Power_Right_1_Test[:,:,:],Power_Left_Test_1[:,:,:])
    Wsquare_t,Wpvalues_t = Compute_Wilcoxon_Map(Power_Right_1_Test[:,:,:],Power_Left_Test_1[:,:,:])

    Rsigned_test_2 = Compute_Rsquare_Map_Welch(Power_Right_2_Test[:,:,:],Power_Left_Test_2[:,:,:])
    Wsquare_t_2,Wpvalues_2 = Compute_Wilcoxon_Map(Power_Right_2_Test[:,:,:],Power_Left_Test_2[:,:,:])

    Rsigned,Wsquare,Wpvalues,electrodes,Power_Left_1,Power_Right_1,timefreq_left,timefreq_right = Reorder_Rsquare(Rsigned,Wsquare,Wpvalues,electrodes,Power_Left_1,Power_Right_1,timefreq_left,timefreq_right)
    electrodes = channel_generator(number_electrodes, 'TP9', 'TP10')
    Rsigned_test,Wsquare_t,Wpvalues_t,electrodes,Power_Left_Test_1,Power_Right_1_Test,timefreq_left,timefreq_right = Reorder_Rsquare(Rsigned_test,Wsquare_t,Wpvalues_t,electrodes,Power_Left_Test_1,Power_Right_1_Test,timefreq_left,timefreq_right)
    electrodes = channel_generator(number_electrodes, 'TP9', 'TP10')
    Rsigned_test_2,Wsquare_t_2,Wpvalues_2,electrodes,Power_Left_Test_2,Power_Right_2_Test,timefreq_left,timefreq_right = Reorder_Rsquare(Rsigned_test_2,Wsquare_t_2,Wpvalues_2,electrodes,Power_Left_Test_2,Power_Right_2_Test,timefreq_left,timefreq_right)

    Statistical_variables.elec_2 = electrodes
    #plt.imshow(R_square)
    Statistical_variables.power_right = Power_Right_1[:,:,:]
    Statistical_variables.power_left = Power_Left_1[:,:,:]

    Statistical_variables.power_right_test = Power_Right_1_Test[:,:,:]
    Statistical_variables.power_left_test = Power_Left_Test_1[:,:,:]

    Statistical_variables.power_right_test_2 = Power_Right_2_Test[:,:,:]
    Statistical_variables.power_left_test_2 = Power_Left_Test_2[:,:,:]

    band_delta =range(1,4)
    band_theta =range(4,8)
    band_alpha =range(8,13)
    band_betalow=range(13,26)
    band_betahigh=range(26,36)
    band_gamma_low=range(36,70)
    band_gamma_high = range(70,101)
    Bands = [band_delta,band_theta,band_alpha,band_betalow,band_betahigh,band_gamma_low,band_gamma_high]


    # file_PSD_1= path + name_sub + 'Cali_Trials_Difference_Bins_Diff.npy'
    # file_PSD_2 = path + name_sub + 'Cali_Trials_Bins_Diff.npy'
    file_PSD_5= path + name_sub + 'Cali_Bins_Trials_Diff.npy'

    # file_PSD_1_test= path + name_sub + 'Drive1_Trials_Difference_Bins_Diff.npy'
    # file_PSD_2_test = path + name_sub + 'Drive1_Trials_Bins_Diff.npy'
    file_PSD_5_test= path + name_sub + 'Drive1_Bins_Trials_Diff.npy'

    file_PSD_1_test_2= path + name_sub + 'Drive2_Trials_Difference_Bins_Diff.npy'
    file_PSD_2_test_2 = path + name_sub + 'Drive2_Trials_Bins_Diff.npy'
    file_PSD_5_test_2= path + name_sub + 'Drive2_Bins_Trials_Diff.npy'


    Matrix_PSD_Diff_1 = (Statistical_variables.power_right.mean(0) - Statistical_variables.power_left.mean(0))/Statistical_variables.power_left.mean(0)

    print("Trials_Difference_Bins")
    print(np.array([Matrix_PSD_Diff_1[:,13:25].mean(1)]).shape)
    topo_plot(np.array([Matrix_PSD_Diff_1[:,13:25].mean(1)]).T,0,electrodes,fres,fs,'Wilcoxon')

    # np.save(file_PSD_1,Matrix_PSD_Diff_1)
    Mat_2 = []
    Mat_5 = []
    for ba in Bands :
        Matrix_PSD_Diff_2 = (Statistical_variables.power_right.mean(0)[:,ba].mean(1) - Statistical_variables.power_left.mean(0)[:,ba].mean(1))/Statistical_variables.power_left.mean(0)[:,ba].mean(1)
        Mat_2.append(Matrix_PSD_Diff_2)
        Matrix_PSD_Diff_5 = (Statistical_variables.power_right[:,:,ba].mean(2).mean(0) - Statistical_variables.power_left[:,:,ba].mean(2).mean(0))/Statistical_variables.power_left[:,:,ba].mean(2).mean(0)
        Mat_5.append(Matrix_PSD_Diff_5)

    # np.save(file_PSD_2,np.array(Mat_2))
    np.save(file_PSD_5,np.array(Mat_5))



    Matrix_PSD_Diff_1_test = (Statistical_variables.power_right_test.mean(0) - Statistical_variables.power_left_test.mean(0))/Statistical_variables.power_left_test.mean(0)

    print("Trials_Difference_Bins")
    print(np.array([Matrix_PSD_Diff_1_test[:,13:25].mean(1)]).shape)
    topo_plot(np.array([Matrix_PSD_Diff_1_test[:,13:25].mean(1)]).T,0,electrodes,fres,fs,'Wilcoxon')

    # np.save(file_PSD_1_test,Matrix_PSD_Diff_1_test)
    Mat_2_test = []
    Mat_5_test = []
    for ba in Bands :
        Matrix_PSD_Diff_2_test = (Statistical_variables.power_right_test.mean(0)[:,ba].mean(1) - Statistical_variables.power_left_test.mean(0)[:,ba].mean(1))/Statistical_variables.power_left_test.mean(0)[:,ba].mean(1)
        Mat_2_test.append(Matrix_PSD_Diff_2_test)
        Matrix_PSD_Diff_5_test = (Statistical_variables.power_right_test[:,:,ba].mean(2).mean(0) - Statistical_variables.power_left_test[:,:,ba].mean(2).mean(0))/Statistical_variables.power_left_test[:,:,ba].mean(2).mean(0)
        Mat_5_test.append(Matrix_PSD_Diff_5_test)

    # np.save(file_PSD_2_test,np.array(Mat_2_test))
    np.save(file_PSD_5_test,np.array(Mat_5_test))


    Matrix_PSD_Diff_1_test_2 = (Statistical_variables.power_right_test_2.mean(0) - Statistical_variables.power_left_test_2.mean(0))/Statistical_variables.power_left_test_2.mean(0)

    print("Trials_Difference_Bins")
    print(np.array([Matrix_PSD_Diff_1_test_2[:,13:25].mean(1)]).shape)
    topo_plot(np.array([Matrix_PSD_Diff_1_test_2[:,13:25].mean(1)]).T,0,electrodes,fres,fs,'Wilcoxon')

    # np.save(file_PSD_1_test_2,Matrix_PSD_Diff_1_test_2)
    Mat_2_test_2 = []
    Mat_5_test_2 = []
    for ba in Bands :
        Matrix_PSD_Diff_2_test_2 = (Statistical_variables.power_right_test_2.mean(0)[:,ba].mean(1) - Statistical_variables.power_left_test_2.mean(0)[:,ba].mean(1))/Statistical_variables.power_left_test_2.mean(0)[:,ba].mean(1)
        Mat_2_test_2.append(Matrix_PSD_Diff_2_test_2)
        Matrix_PSD_Diff_5_test_2 = (Statistical_variables.power_right_test_2[:,:,ba].mean(2).mean(0) - Statistical_variables.power_left_test_2[:,:,ba].mean(2).mean(0))/Statistical_variables.power_left_test_2[:,:,ba].mean(2).mean(0)
        Mat_5_test_2.append(Matrix_PSD_Diff_5_test_2)

    # np.save(file_PSD_2_test_2,np.array(Mat_2_test_2))
    np.save(file_PSD_5_test_2,np.array(Mat_5_test_2))





    plt.show()


class Dialog(QDialog):

    """Dialog."""
    msg_1 = QLabel('')
    msg_2 = QLabel('')
    msg_3 = QLabel('')
    msg_4 = QLabel('')
    msg_5 = QLabel('')
    msg_6 = QLabel('')
    msg_7 = QLabel('')
    def __init__(self, parent=None):

        """Initializer."""

        super().__init__(parent)

        self.setWindowTitle('Parameters')

        dlgLayout = QVBoxLayout()

        formLayout = QFormLayout()
        path = QLineEdit()
        path.setText('/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Sub18/ses-01/EEG/')
        formLayout.addRow('Path:', path)
        hbox_1 = QHBoxLayout()
        filename_1 = QLineEdit()
        filename_1.setText('MI-[2022.09.28-15.31.27].ov.edf')
        hbox_1.addWidget(filename_1)
        #formLayout.addRow('Name File Run 01', filename_1)
        hbox_2 = QHBoxLayout()
        filename_2 = QLineEdit()
        filename_2.setText('MI-[2022.09.28-15.42.05].ov.edf')
        hbox_2.addWidget(filename_2)
        #formLayout.addRow('Name File Run 02', filename_2)
        hbox_3 = QHBoxLayout()
        filename_3 = QLineEdit()
        filename_3.setText('MI-[2022.09.28-15.54.42].ov.edf')
        hbox_3.addWidget(filename_3)



        filename_4 = QLineEdit()
        filename_4.setText('Test-[2022.09.28-16.11.51].edf')
        hbox_1.addWidget(filename_4)
        #formLayout.addRow('Name File Run 01', filename_1)
        filename_5 = QLineEdit()
        filename_5.setText('Test-[2022.09.28-16.24.23].edf')
        hbox_2.addWidget(filename_5)
        #formLayout.addRow('Name File Run 02', filename_2)

        filename_6 = QLineEdit()
        filename_6.setText('Test-[2022.09.28-16.37.00].edf')
        hbox_3.addWidget(filename_6)
        #formLayout.addRow('Name File Run 03', filename_3)

        filename_7 = QLineEdit()
        filename_7.setText('Test-[2022.09.28-16.52.37].edf')
        hbox_1.addWidget(filename_7)
        #formLayout.addRow('Name File Run 01', filename_1)
        filename_8 = QLineEdit()
        filename_8.setText('Test-[2022.09.28-17.05.05].edf')
        hbox_2.addWidget(filename_8)
        #formLayout.addRow('Name File Run 02', filename_2)


        formLayout.addRow("Filename Train and Test 1: ",hbox_1)
        formLayout.addRow("Filename Train and Test 2: ",hbox_2)
        formLayout.addRow("Filename Train and Test 3: ",hbox_3)


        cond1 = QLineEdit()
        cond1.setText('OVTK_GDF_Left')
        formLayout.addRow('Motor Imagery', cond1)
        cond2 = QLineEdit()
        cond2.setText('OVTK_GDF_Right')
        formLayout.addRow('Resting State', cond2)

        tmin = QLineEdit()
        tmin.setText('1')
        formLayout.addRow('T min', tmin)
        tmax = QLineEdit()
        tmax.setText('4')
        formLayout.addRow('T max', tmax)

        number_electrodes = QLineEdit()
        number_electrodes.setText('64')
        formLayout.addRow('Number Of electrodes', number_electrodes)

        fs = QLineEdit()
        fs.setText('500')
        formLayout.addRow('Sampling Frequency', fs)
        f_min = QLineEdit()
        f_min.setText('0')
        formLayout.addRow('f min', f_min)
        f_max = QLineEdit()
        f_max.setText('40')
        formLayout.addRow('f max', f_max)

        nfft = QLineEdit()
        nfft.setText('500')
        formLayout.addRow('Nfft', nfft)
        nper_seg = QLineEdit()
        nper_seg.setText('0.25')
        formLayout.addRow('Window', nper_seg)
        noverlap = QLineEdit()
        noverlap.setText('0.089')
        formLayout.addRow('Overlap', noverlap)

        filter_order = QLineEdit()
        filter_order.setText('19')
        formLayout.addRow('Filter Order', filter_order)
        name_sub = QLineEdit()
        name_sub.setText('')
        formLayout.addRow('Name Subject', name_sub)
        dlgLayout.addLayout(formLayout)
        checkbox = QCheckBox('CAR')
        checkbox.setChecked(0)
        checkbox.clicked.connect(lambda:btnstate(checkbox))
        dlgLayout.addWidget(checkbox)

        strat = QLineEdit()
        strat.setText('1')
        formLayout.addRow('Strategy of Session', strat)

        btns = QPushButton("Perform tests")
        dlgLayout.addWidget(btns)
        btns.clicked.connect(lambda: launch(path.text(), filename_1.text(), filename_2.text(),
                                            filename_3.text(), filename_4.text(), filename_5.text(),filename_6.text(), filename_7.text(),filename_8.text(),
                                            cond1.text(), cond2.text(),
                                            float(tmin.text()), float(tmax.text()),
                                            int(nfft.text()), float(noverlap.text()),
                                            float(nper_seg.text()), int(fs.text()),
                                            int(filter_order.text()), int(number_electrodes.text()),
                                            Statistical_variables.CAR,int(strat.text()),name_sub.text()))


        self.setLayout(dlgLayout)



if __name__ == '__main__':

    app = QApplication(sys.argv)

    dlg = Dialog()

    dlg.show()

    sys.exit(app.exec_())
