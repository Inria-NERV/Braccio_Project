# Filename: dialog.py


"""Dialog-Style application."""

import sys
import os
import time
import numpy as np
import mne
from mne_connectivity.viz import circular_layout, plot_connectivity_circle
from sklearn.covariance import ledoit_wolf
from Visualization_Data import *
from Spectral_Analysis import *
from Statistical_analysis import *
from file_loading import *
from Classification import *
from scipy.interpolate import interp1d
import pandas as pd
from Optimization_feature_functions import *
from itertools import combinations
from sklearn.decomposition import KernelPCA,PCA
from PyQt5.QtWidgets import QApplication,QMessageBox,QLabel,QHBoxLayout,QCheckBox,QFileDialog

from PyQt5.QtWidgets import QDialog

from PyQt5.QtWidgets import QPushButton

from PyQt5.QtWidgets import QFormLayout

from PyQt5.QtWidgets import QLineEdit

from PyQt5.QtWidgets import QVBoxLayout

from mpl_toolkits.mplot3d import Axes3D
import nestle

def browseForRun1(self):
        directory = os.getcwd()
        Script, dummy = QFileDialog.getOpenFileName(self, "MI Train", str(directory))
        Statistical_variables.Path_run_1 = os.path.basename(Script)
        Statistical_variables.Path_all_Run = os.path.dirname(Script)+'/'
        print(Statistical_variables.Path_all_Run)
        print(Statistical_variables.Path_run_1)
        return

def browseForRun2(self):
        directory = os.getcwd()
        Script, dummy = QFileDialog.getOpenFileName(self, "Rest Train", str(directory))
        Statistical_variables.Path_run_2 = os.path.basename(Script)
        return

def browseForRun3(self):
        directory = os.getcwd()
        Script, dummy = QFileDialog.getOpenFileName(self, "Choice Taken 1", str(directory))
        Statistical_variables.Path_run_3 = os.path.basename(Script)
        return
def browseForRun4(self):
        directory = os.getcwd()
        Script, dummy = QFileDialog.getOpenFileName(self, "Feature Vec 1", str(directory))
        Statistical_variables.Path_run_4 = os.path.basename(Script)
        return
def browseForRun5(self):
        directory = os.getcwd()
        Script, dummy = QFileDialog.getOpenFileName(self, "Choice Taken 2", str(directory))
        Statistical_variables.Path_run_5 = os.path.basename(Script)
        return
def browseForRun6(self):
        directory = os.getcwd()
        Script, dummy = QFileDialog.getOpenFileName(self, "Feature Taken 2", str(directory))
        Statistical_variables.Path_run_6 = os.path.basename(Script)
        return
def browseForRun7(self):
        directory = os.getcwd()
        Script, dummy = QFileDialog.getOpenFileName(self, "Choice Taken 3", str(directory))
        Statistical_variables.Path_run_7 = os.path.basename(Script)
        return

def browseForRun8(self):
        directory = os.getcwd()
        Script, dummy = QFileDialog.getOpenFileName(self, "Feature Vec 3", str(directory))
        Statistical_variables.Path_run_8 = os.path.basename(Script)
        return

def rand_torus(ro, ri, npoints):
    """Generate points within a torus wth major radius `ro` and minor radius
    `ri` via rejection sampling"""
    out = np.empty((npoints, 3), dtype=np.float64)
    i = 0
    while i < npoints:
        # generate point within box
        x = np.random.uniform(-1., 1., size=3)
        x[0:2] *= ro + ri
        x[2] *= ri

        r = math.sqrt(x[0]**2 + x[1]**2) - ro
        if (r**2 + x[2]**2 < ri**2):
            out[i, :] = x
            i += 1

    return out


def plot_ellipsoid_3d(ell, ax,c):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = ell.ctr + np.dot(ell.axes,
                                                      [x[i,j],y[i,j],z[i,j]])

    #ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=c, alpha=0.1)
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=c, alpha=0.3)


class Statistical_variables:
    Path_run_1 = ''
    Path_run_2 = ''
    Path_run_3 = ''
    Path_run_4 = ''
    Path_run_5 = ''
    Path_run_6 = ''
    Path_run_7 = ''
    Path_run_8 = ''



def launch(path,filename_1,filename_2,filename_3,filename_4,filename_5,filename_6,filename_7,filename_8,Channel,freq1,freq2,freq3,freq4,freq5,freq6):

    Rsquare =[0.44, 0.49, 0.43]
    Wsquare =[5.23, 5.46, 5.2]



    electrodes = Channel.split(",")
    Frequency_select_list_1 = freq1.split(",")
    Frequencies_list_1 = list(map(int, Frequency_select_list_1))
    Frequencies_list_2 = []
    Frequencies_list_3 = []
    Frequencies_list_4 = []
    Frequencies_list_5 = []
    Frequencies_list_6 = []
    if freq2 != '':
        Frequency_select_list_2 = freq2.split(",")
        Frequencies_list_2 = list(map(int, Frequency_select_list_2))
    if freq3 != '':
        Frequency_select_list_3 = freq3.split(",")
        Frequencies_list_3 = list(map(int, Frequency_select_list_3))
    if freq4 != '':
        Frequency_select_list_4 = freq4.split(",")
        Frequencies_list_4 = list(map(int, Frequency_select_list_4))
    if freq5 != '':
        Frequency_select_list_5 = freq5.split(",")
        Frequencies_list_5 = list(map(int, Frequency_select_list_5))
    if freq6 != '':
        Frequency_select_list_6 = freq6.split(",")
        Frequencies_list_6 = list(map(int, Frequency_select_list_6))

    frequencies = [Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6]


    Path_Mi_Train = path+filename_1
    Path_Rest_Train = path+filename_2

    Path_Test_Choice_1 =path+filename_3
    Path_Test_Feature_1 = path+filename_4

    Path_Test_Choice_2 =path+filename_5
    Path_Test_Feature_2 = path+filename_6

    Path_Test_Choice_3 =path+filename_7
    Path_Test_Feature_3 = path+filename_8

    List_Electrode_asso_freq= []
    List_Index_Electrodes_freq = []
    for k in range(len(frequencies)):
        for j in range(len(frequencies[k])):
            List_Electrode_asso_freq.append(electrodes[k])

    ListFreq_asso_elec = [item for sublist in frequencies for item in sublist]
    print(ListFreq_asso_elec)
    numberFeature = len(ListFreq_asso_elec)
    file_cond_MI = load_csv_cond(Path_Mi_Train)
    file_cond_Rest = load_csv_cond(Path_Rest_Train)

    file_Test_Choice_1 = load_csv_cond(Path_Test_Choice_1)
    file_Test_Feature_1 = load_csv_cond(Path_Test_Feature_1)

    if filename_6 != '':
        file_Test_Choice_2 = load_csv_cond(Path_Test_Choice_2)
        file_Test_Feature_2 = load_csv_cond(Path_Test_Feature_2)

    if filename_7 != '':
        file_Test_Choice_3 = load_csv_cond(Path_Test_Choice_3)
        file_Test_Feature_3 = load_csv_cond(Path_Test_Feature_3)

    mat_MI_Train = file_cond_MI.to_numpy()
    mat_Rest_Train = file_cond_Rest.to_numpy()

    mat_Test_Choice_1 = file_Test_Choice_1.to_numpy()
    mat_Test_Feature_1 = file_Test_Feature_1.to_numpy()

    mat_Test_Choice = mat_Test_Choice_1
    mat_Test_Feature = mat_Test_Feature_1
    if filename_6 != '':
        mat_Test_Choice_2 = file_Test_Choice_2.to_numpy()
        mat_Test_Feature_2 = file_Test_Feature_2.to_numpy()

        mat_Test_Choice = np.concatenate((mat_Test_Choice_1,mat_Test_Choice_2),axis = 0)
        mat_Test_Feature = np.concatenate((mat_Test_Feature_1,mat_Test_Feature_2),axis=0)
    #
    if filename_7 != '':
        mat_Test_Choice_3 = file_Test_Choice_3.to_numpy()
        mat_Test_Feature_3 = file_Test_Feature_3.to_numpy()

        mat_Test_Choice = np.concatenate((mat_Test_Choice_1,mat_Test_Choice_2,mat_Test_Choice_3),axis = 0)
        mat_Test_Feature = np.concatenate((mat_Test_Feature_1,mat_Test_Feature_2,mat_Test_Feature_3),axis=0)
    # else:
    #     mat_Test_Choice = np.concatenate((mat_Test_Choice_1,mat_Test_Choice_2),axis = 0)
    #     mat_Test_Feature = np.concatenate((mat_Test_Feature_1,mat_Test_Feature_2),axis=0)




    Feature_Test_Selected = mat_Test_Feature[:,2:2+numberFeature]
    MI_Feature_Selected = mat_MI_Train[:,2:2+numberFeature]
    Rest_Feature_Selected = mat_Rest_Train[:,2:2+numberFeature]
    History_Choice = mat_Test_Feature[:,2+numberFeature]
    History_Choice_vec =[]
    for i in range(len(History_Choice)):
        if History_Choice[i].find('770') > 0:
            History_Choice_vec.append(1)
        else:
            History_Choice_vec.append(0)


    ChoiceMake_Inter = mat_Test_Choice[:,2:4]

    ChoiceTaken_Vec = ((ChoiceMake_Inter[:,1]-ChoiceMake_Inter[:,0])<0)

    Feature_MI_Test_l = []
    Feature_Rest_Test_l = []

    Feature_MI_Test_F = []
    Feature_Rest_Test_F = []
    for i in range(len(ChoiceTaken_Vec)):
        if ChoiceTaken_Vec[i] == True :
            if (History_Choice_vec[i] == 0):
                Feature_MI_Test_l.append(Feature_Test_Selected[i,:])
            else:
                Feature_MI_Test_F.append(Feature_Test_Selected[i,:])
        else:
            if (History_Choice_vec[i] == 0):
                Feature_Rest_Test_F.append(Feature_Test_Selected[i,:])
            else:
                Feature_Rest_Test_l.append(Feature_Test_Selected[i,:])



    Feature_MI_Test = np.array(Feature_MI_Test_l)
    Feature_Rest_Test = np.array(Feature_Rest_Test_l)

    Feature_MI_Test_F = np.array(Feature_MI_Test_F)
    Feature_Rest_Test_F = np.array(Feature_Rest_Test_F)




    transformer = PCA(n_components=3)

    transformer = PCA(n_components=3)
    n=3
    MI_transformed = transformer.fit_transform(MI_Feature_Selected).transpose()
    important_MI=[np.abs(transformer.components_[i]).argmax()for i in range(n)]
    important_Electrode_MI = [List_Electrode_asso_freq[important_MI[i]] for i in range(n)]
    important_Freq_MI = [ListFreq_asso_elec[important_MI[i]] for i in range(n)]
    print(important_Electrode_MI)
    print(important_Freq_MI)
    Rest_transformed = transformer.fit_transform(Rest_Feature_Selected).transpose()

    important_Rest=[np.abs(transformer.components_[i]).argmax()for i in range(n)]
    important_Electrode_Rest = [List_Electrode_asso_freq[important_Rest[i]] for i in range(n)]
    important_Freq_MI_Rest = [ListFreq_asso_elec[important_Rest[i]] for i in range(n)]

    print(important_Electrode_Rest)
    print(important_Freq_MI_Rest)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    imp = '\nImportance of features :'+important_Electrode_MI[0]+'('+str(important_Freq_MI[0])+'Hz)'+'>'+important_Electrode_MI[1]+'('+str(important_Freq_MI[1])+'Hz)'+'>'+important_Electrode_MI[2]+'('+str(important_Freq_MI[0])+'Hz)'
    ax.set_title('PCA Distribution of Power Spectrum Features between classes'+imp)

    ax.scatter(MI_transformed[0,:],MI_transformed[1,:],MI_transformed[2,:],c ='r')
    ax.scatter(Rest_transformed[0,:],Rest_transformed[1,:],Rest_transformed[2,:],c ='b')

    AvePoint_MI = MI_transformed.mean(axis = 1)
    AvePoint_Rest = Rest_transformed.mean(axis = 1)

    Distance_MI = np.sqrt((AvePoint_MI[0]-MI_transformed[0,:])**2+(AvePoint_MI[1]-MI_transformed[1,:])**2+(AvePoint_MI[2]-MI_transformed[2,:])**2)
    Distance_Rest = np.sqrt((AvePoint_Rest[0]-Rest_transformed[0,:])**2+(AvePoint_Rest[1]-Rest_transformed[1,:])**2+(AvePoint_Rest[2]-Rest_transformed[2,:])**2)
    Distance_MI_Ave = Distance_MI.mean()
    Distance_Rest_Ave = Distance_Rest.mean()


    npoints = 100
    # ell_gen = nestle.Ellipsoid([np.max(MI_transformed[0,:]), np.max(MI_transformed[1,:]), np.max(MI_transformed[2,:])], np.dot(MI_transformed, MI_transformed.transpose()))
    # # points = ell_gen.samples(npoints)
    # pointvol = ell_gen.vol / npoints

    ells = nestle.bounding_ellipsoids(MI_transformed.T)


    plot_ellipsoid_3d(ells, ax,'r')



    # ell_gen_r = nestle.Ellipsoid([np.max(Rest_transformed[0,:]), np.max(Rest_transformed[1,:]), np.max(Rest_transformed[2,:])], np.dot(Rest_transformed, Rest_transformed.transpose()))
    # # points = ell_gen.samples(npoints)
    # pointvol_r = ell_gen_r.vol / npoints

    ells_r = nestle.bounding_ellipsoids(Rest_transformed.T)


    plot_ellipsoid_3d(ells_r, ax,'b')
    plt.legend(['Motor Imagery feature distribution \n Average Distance : '+str(round(Distance_MI_Ave,2)),'Resting feature distribution\n Average Distance : '+str(round(Distance_Rest_Ave,2))])

    ax.set_xlabel("PCA 1")

    ax.set_ylabel("PCA 2")

    ax.set_zlabel("PCA 3")
    title = 'PCADistribTrain'
    #path = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Sub19/ses-01/Plots/' + title +'.png'
    #plt.savefig(path)
    plt.show()



    for i in range(MI_Feature_Selected.shape[1]):
    # Set up the plot
        lmi = np.array(MI_Feature_Selected[:,i],dtype=float)
        lrest = np.array(Rest_Feature_Selected[:,i],dtype=float)
        plt.figure()
        #ax = plt.subplot(X_train.shape[1], 2, i+1)

        # Draw the plot
        mu_mi, std_mi = norm.fit(lmi)
        mu_rest, std_rest = norm.fit(lrest)
        weights_mi = np.ones_like(lmi)/len(lmi)
        weights_rest = np.ones_like(lrest)/len(lrest)
        plt.hist(MI_Feature_Selected[:,i], bins = 30,
                 color = 'red', edgecolor = 'black',alpha=0.45,weights = weights_mi)
        plt.hist(Rest_Feature_Selected[:,i], bins = 30,
                 color = 'blue', edgecolor = 'black',alpha=0.45,weights = weights_rest)
        # ax.plot(X_test[index_list_succeed,i],y_success,'go')
        # ax.plot(X_test[index_list_fail,i],y_fail,'r+')
        x_MI_succ = []
        x_Rest_succ = []
        x_MI_Fail = []
        x_Rest_Fail = []
        # for k1 in range(len(MI_Success)):
        #     x_MI_succ.append(MI_Success[k1][i])
        # for k1 in range(len(MI_Fail)):
        #     x_MI_Fail.append(MI_Fail[k1][i])
        #
        # for k1 in range(len(Rest_Success)):
        #     x_Rest_succ.append(Rest_Success[k1][i])
        # for k1 in range(len(Rest_Fail)):
        #     x_Rest_Fail.append(Rest_Fail[k1][i])
        #
        # y_MI_Success = 2*np.ones(len(x_MI_succ))
        # y_Rest_Success = 2*np.ones(len(x_Rest_succ))
        if Feature_MI_Test_F.shape[0]!=0:
            y_MI_Fail = 0.2*np.ones(len(Feature_MI_Test_F[:,i]))
            plt.plot((Feature_MI_Test_F[:,i]),y_MI_Fail,'ro',alpha = 0,label='_nolegend_')
        if Feature_Rest_Test_F.shape[0]!=0:
            y_Rest_Fail = 0.1*np.ones(len(Feature_Rest_Test_F[:,i]))
            plt.plot((Feature_Rest_Test_F[:,i]),y_Rest_Fail,'bo',alpha =0,label='_nolegend_')

        if Feature_MI_Test.shape[0]!=0:
            y_MI = 0.2*np.ones(len(Feature_MI_Test[:,i]),)
            plt.plot((Feature_MI_Test[:,i]),y_MI,'ro',alpha = 0,label='_nolegend_')
        if Feature_Rest_Test.shape[0]!=0:
            y_Rest = 0.1*np.ones(len(Feature_Rest_Test[:,i]))
            plt.plot((Feature_Rest_Test[:,i]),y_Rest,'bo',alpha =0,label='_nolegend_')
        #





        #Distribution
        xmin, xmax = plt.xlim()
        ymin,ymax = plt.ylim()


        x_index = []
        x_rest_index =[]

        x_index_F = []
        x_rest_index_F =[]
        x = np.linspace(xmin, xmax, 100)
        if Feature_MI_Test.shape[0]!=0:
            for z in range(len(Feature_MI_Test[:,i])):
                for k in range(len(x)):
                    if ((Feature_MI_Test[:,i][z]-x[k])<0.1):
                        x_index.append(k)
                        break
        if Feature_Rest_Test.shape[0]!=0:
            for z in range(len(Feature_Rest_Test[:,i])):
                for k in range(len(x)):
                    if ((Feature_Rest_Test[:,i][z]-x[k])<0.1):
                        x_rest_index.append(k)
                        break

        if Feature_MI_Test_F.shape[0]!=0:
            for z in range(len(Feature_MI_Test_F[:,i])):
                for k in range(len(x)):
                    if ((Feature_MI_Test_F[:,i][z]-x[k])<0.1):
                        x_index_F.append(k)
                        break
        if Feature_Rest_Test_F.shape[0]!=0:
            for z in range(len(Feature_Rest_Test_F[:,i])):
                for k in range(len(x)):
                    if ((Feature_Rest_Test_F[:,i][z]-x[k])<0.1):
                        x_rest_index_F.append(k)
                        break
        #print(x_rest_index)
        p_mi = norm.pdf(x, mu_mi, std_mi)
        p_rest = norm.pdf(x, mu_rest, std_rest)

        if Feature_MI_Test.shape[0]!=0:
            plt.plot((Feature_MI_Test[:,i]),p_mi[x_index],'ro')
        if Feature_Rest_Test.shape[0]!=0:
            plt.plot((Feature_Rest_Test[:,i]),p_rest[x_rest_index],'bo')
        plt.plot(x, p_mi, 'r', linewidth=2)
        plt.plot(x, p_rest, 'b', linewidth=2)
        if Feature_MI_Test_F.shape[0]!=0:
            plt.plot((Feature_MI_Test_F[:,i]),p_rest[x_index_F],'ro',label='_nolegend_')
        if Feature_Rest_Test_F.shape[0]!=0:
            plt.plot((Feature_Rest_Test_F[:,i]),p_mi[x_rest_index_F],'bo',label='_nolegend_')




        # Title and labels
        title = 'Distribution for ' + List_Electrode_asso_freq[i] + ' at ' + str(ListFreq_asso_elec[i]) + 'Hz'+ '\n R^2 : ' + str(round(Rsquare[i],2))+ ' Wign : ' + str(round(Wsquare[i],2))
        #plt.text(2, -1, 'R^2 : ' + str(round(R[List_Index_Electrodes_freq[i],ListFreq_asso_elec[i]],4)), fontsize = 10)

        plt.title(title, size = 20)
        plt.xlabel('PowerSpectrum', size = 20)
        plt.ylabel('Occurence', size= 20)
        plt.legend(['Interpreted MI','Interpreted Rest','Distribution MI','Distribution Rest','MI','Rest'],loc='center right',fontsize=20)
        #path = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/Sub19/ses-01/Plots/' + title +'.png'
        #plt.savefig(path)
    # plt.figlegend(['Success MI','Success Rest','Failed MI','Failed Rest','Distribution MI','Distribution Rest','MI','Rest'], loc = 'lower center', ncol=5, labelspacing=0.)
    # plt.tight_layout()
    plt.show()


class Dialog(QDialog):

    """Dialog."""
    msg_1 = QLabel('')
    msg_2 = QLabel('')
    msg_3 = QLabel('')
    msg_4 = QLabel('')
    msg_5 = QLabel('')
    msg_6 = QLabel('')
    def __init__(self, parent=None):

        """Initializer."""

        super().__init__(parent)

        self.setWindowTitle('Parameters')

        dlgLayout = QVBoxLayout()

        formLayout = QFormLayout()


        btn_browseOV_Run_1 = QPushButton("MI")
        btn_browseOV_Run_1.clicked.connect(lambda: browseForRun1(self))
        dlgLayout.addWidget(btn_browseOV_Run_1)

        btn_browseOV_Run_2 = QPushButton("Rest")
        btn_browseOV_Run_2.clicked.connect(lambda: browseForRun2(self))
        dlgLayout.addWidget(btn_browseOV_Run_2)

        btn_browseOV_Run_3 = QPushButton("Choice 1")
        btn_browseOV_Run_3.clicked.connect(lambda: browseForRun3(self))
        dlgLayout.addWidget(btn_browseOV_Run_3)


        btn_browseOV_Run_4 = QPushButton("Feature 1")
        btn_browseOV_Run_4.clicked.connect(lambda: browseForRun4(self))
        dlgLayout.addWidget(btn_browseOV_Run_4)

        btn_browseOV_Run_5 = QPushButton("Choice 2")
        btn_browseOV_Run_5.clicked.connect(lambda: browseForRun5(self))
        dlgLayout.addWidget(btn_browseOV_Run_5)

        btn_browseOV_Run_6 = QPushButton("Feature 2")
        btn_browseOV_Run_6.clicked.connect(lambda: browseForRun6(self))
        dlgLayout.addWidget(btn_browseOV_Run_6)

        btn_browseOV_Run_7 = QPushButton("Choice 3")
        btn_browseOV_Run_7.clicked.connect(lambda: browseForRun7(self))
        dlgLayout.addWidget(btn_browseOV_Run_7)

        btn_browseOV_Run_8 = QPushButton("Feature 3")
        btn_browseOV_Run_8.clicked.connect(lambda: browseForRun8(self))
        dlgLayout.addWidget(btn_browseOV_Run_8)



        hbox = QHBoxLayout()
        channel_select = QLineEdit()
        Frequency_select_1 = QLineEdit()
        Frequency_select_2 = QLineEdit()
        Frequency_select_3 = QLineEdit()
        Frequency_select_4 = QLineEdit()
        Frequency_select_5 = QLineEdit()
        Frequency_select_6 = QLineEdit()
        Partitions = QLineEdit()
        hbox.addWidget(channel_select)
        hbox.addWidget(Frequency_select_1)
        hbox.addWidget(Frequency_select_2)
        hbox.addWidget(Frequency_select_3)
        hbox.addWidget(Frequency_select_4)
        hbox.addWidget(Frequency_select_5)
        hbox.addWidget(Frequency_select_6)

        formLayout.addRow("Channel to select and Frequency to select for classification: ",hbox)

        btns = QPushButton("Perform tests")
        dlgLayout.addWidget(btns)
        btns.clicked.connect(lambda: launch(Statistical_variables.Path_all_Run,  Statistical_variables.Path_run_1,  Statistical_variables.Path_run_2,Statistical_variables.Path_run_3,  Statistical_variables.Path_run_4, Statistical_variables.Path_run_5,Statistical_variables.Path_run_6,Statistical_variables.Path_run_7,Statistical_variables.Path_run_8,channel_select.text(),Frequency_select_1.text(),Frequency_select_2.text(),Frequency_select_3.text(),Frequency_select_4.text(),Frequency_select_5.text(),Frequency_select_6.text()))

        dlgLayout.addLayout(formLayout)
        self.setLayout(dlgLayout)






if __name__ == '__main__':

    app = QApplication(sys.argv)

    dlg = Dialog()

    dlg.show()

    sys.exit(app.exec_())
