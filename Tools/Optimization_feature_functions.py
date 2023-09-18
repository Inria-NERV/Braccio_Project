import os
import time
import numpy as np
import mne
from file_loading import *
from Visualization_Data import *
from Spectral_Analysis import *
from Statistical_analysis import *
# Import modules
import numpy as np

# Import sphere function as objective function
from pyswarms.utils.functions.single_obj import sphere as f

# Import backend modules
import pyswarms.backend as P
from pyswarms.backend.topology import Star
import matplotlib.pyplot as plt
import numpy as np
from tempfile import TemporaryFile


def R_square_parametrized(x,Epoched_data_cond_1,Epoched_data_cond_2,number_electrodes):
    #print(x.shape)
    model_order = x[:,0]
    overlap = x[:,1]
    noverlap =x[:,2]
    channel =x[:,3]
    frequency = x[:,4]
    nfft = 500
    f_min = 0
    f_max = 500
    t_min = None
    t_max = None
    pick = None
    proje = None
    fres = f_max/nfft
   # print(fres)
    #print(1/fres)
    averag = 'mean'
    windowing = 'hann'
    smoothing = False
    fs = 500
    nper_seg = overlap
    noverlap = noverlap
    electrodes = channel_generator(number_electrodes, 'TP9', 'TP10')
    Channel_cortex = ['FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4']
    Channel_cortex_index = []
    for i in range(len(electrodes)):
        for j in Channel_cortex:
            if (electrodes[i] == j):
                Channel_cortex_index.append(i)
    Matrix_complete = np.zeros([x.shape[0]])
    freqs_left = np.arange(0,500/2+1)
    for z in range(x.shape[0]):
        #print(z)
        #Power_Right_1 ,freqs_left= Power_calculation_welch_method(Epoched_data_cond_1.get_data()[:,Channel_cortex_index[round(channel[z])],:],f_min,f_max,t_min,t_max,nfft,round(noverlap[z]/1000*fs),round(nper_seg[z]/1000*fs),pick,proje,averag,windowing,smoothing)
        Power_Cond_1,timefreq_left,time_left = Power_burg_calculation_optimization(Epoched_data_cond_1[:,Channel_cortex_index[round(channel[z])],:],round(noverlap[z]/1000*fs),nfft,f_max, round(nper_seg[z]/1000*fs),smoothing,freqs_left,model_order[z])
        Power_Cond_2,timefreq_right,time_left= Power_burg_calculation_optimization(Epoched_data_cond_2[:,Channel_cortex_index[round(channel[z])],:],round(noverlap[z]/1000*fs),nfft,f_max, round(nper_seg[z]/1000*fs),smoothing,freqs_left,model_order[z])
        Rsquare = Compute_Rsquare_Map_Welch_optimization(Power_Cond_1[:,0:36],Power_Cond_2[:,0:36])
        Matrix_complete[z]= -Rsquare[round(frequency[z])]
        #if(swil[round(channel[z]),round(frequency[z])]<0):
        #    Matrix_complete[z]=pwil[round(channel[z]),round(frequency[z])]
        #else:
        #    Matrix_complete[z]=1
        #Matrix_complete[z]=Rsquare[round(channel[z]),round(frequency[z])]
    return Matrix_complete

def R_square_parametrized_chan_freq(x,modelorder,nwindow,n_overlap,electrode,Epoched_data_cond_1,Epoched_data_cond_2,number_electrodes):
    #print(x.shape)
    model_order = modelorder
    overlap = nwindow
    noverlap =n_overlap
    channel =x[:,0]
    frequency = x[:,1]
    nfft = 500
    f_min = 0
    f_max = 500
    t_min = None
    t_max = None
    pick = None
    proje = None
    fres = f_max/nfft
    #print(fres)
    #print(1/fres)
    averag = 'mean'
    windowing = 'hann'
    smoothing = False
    fs = 500
    nper_seg = overlap
    noverlap = noverlap
    electrodes = channel_generator(number_electrodes, 'TP9', 'TP10')
    Channel_cortex = ['FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4']
    Channel_cortex_index = []
    for i in electrode:
        electodes = electrodes.pop(i)
    for i in range(len(electrodes)):
        for j in Channel_cortex:
            if (electrodes[i] == j):
                Channel_cortex_index.append(i)
    Matrix_complete = np.zeros([x.shape[0]])
    freqs_left = np.arange(0,500/2+1)
    for z in range(x.shape[0]):
        #Power_Right_1 ,freqs_left= Power_calculation_welch_method(Epoched_data_cond_1.get_data()[:,Channel_cortex_index[round(channel[z])],:],f_min,f_max,t_min,t_max,nfft,round(noverlap/1000*fs),round(nper_seg/1000*fs),pick,proje,averag,windowing,smoothing)
        Power_Cond_1,timefreq_left,time_left = Power_burg_calculation_optimization(Epoched_data_cond_1[:,Channel_cortex_index[round(channel[z])],:],round(noverlap/1000*fs),nfft,f_max, round(nper_seg/1000*fs),smoothing,freqs_left,model_order)
        Power_Cond_2,timefreq_right,time_left= Power_burg_calculation_optimization(Epoched_data_cond_2[:,Channel_cortex_index[round(channel[z])],:],round(noverlap/1000*fs),nfft,f_max, round(nper_seg/1000*fs),smoothing,freqs_left,model_order)
        Rsquare = Compute_Rsquare_Map_Welch_optimization(Power_Cond_1[:,0:36],Power_Cond_2[:,0:36])

        #Rsquare = np.absolute(Rsquare)

        Matrix_complete[z]=-Rsquare[round(frequency[z])]
        #if(swil[round(channel[z]),round(frequency[z])]<0):
        #    Matrix_complete[z]=pwil[round(channel[z]),round(frequency[z])]
        #else:
        #    Matrix_complete[z]=1
        #Matrix_complete[z]=Rsquare[round(channel[z]),round(frequency[z])]
    return Matrix_complete

def Optimization_swarm(constraints, bounds, nparticles,options,dimension, iteration,Epoched_data_cond_1,Epoched_data_cond_2,number_electrodes):
    my_topology = Star() # The Topology Class
    my_swarm = P.create_swarm(n_particles=nparticles, dimensions=dimension, options=options,bounds = bounds,constraints = constraints) # The Swarm Class
    print('The following are the attributes of our swarm: {}'.format(my_swarm.__dict__.keys()))
    iterations = iteration # Set 100 iterations
    outfile = 'data.npy'
    #History_Pos = []
    for i in range(iterations):
        # Part 1: Update personal best
        my_swarm.current_cost = R_square_parametrized(my_swarm.position,Epoched_data_cond_1,Epoched_data_cond_2,number_electrodes) # Compute current cost
        my_swarm.pbest_cost = R_square_parametrized(my_swarm.pbest_pos,Epoched_data_cond_1,Epoched_data_cond_2,number_electrodes)  # Compute personal best pos
        my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store

        # Part 2: Update global best
        # Note that gbest computation is dependent on your topology
        if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
            my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

        # Let's print our output
        if i%2==0:
            print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))

        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        my_swarm.velocity = my_topology.compute_velocity(my_swarm)
        my_swarm.position = my_topology.compute_position(my_swarm,bounds)
        #History_Pos.append(np.array([my_swarm.position[:,3],my_swarm.position[:,4],my_swarm.current_cost]))
    print('The best cost found by our swarm is: {:.4f}'.format(my_swarm.best_cost))
    print('The best position found by our swarm is: {}'.format(my_swarm.best_pos))
    #np.save(outfile, History_Pos)
    return my_swarm.best_pos,my_swarm.best_cost



def Optimization_swarm_chan_freq(constraints, bounds,nparticles,options,dimension, iteration,modelorder,nwindow,n_overlap,electrode,Epoched_data_cond_1,Epoched_data_cond_2,number_electrodes):
    my_topology = Star() # The Topology Class
    my_swarm = P.create_swarm(n_particles=nparticles, dimensions=dimension, options=options,bounds = bounds,constraints = constraints) # The Swarm Class
    print('The following are the attributes of our swarm: {}'.format(my_swarm.__dict__.keys()))
    iterations = iteration # Set 100 iterations

    for i in range(iterations):
        # Part 1: Update personal best
        my_swarm.current_cost = R_square_parametrized_chan_freq(my_swarm.position,modelorder,nwindow,n_overlap,electrode,Epoched_data_cond_1,Epoched_data_cond_2,number_electrodes) # Compute current cost
        my_swarm.pbest_cost = R_square_parametrized_chan_freq(my_swarm.pbest_pos,modelorder,nwindow,n_overlap,electrode,Epoched_data_cond_1,Epoched_data_cond_2,number_electrodes)  # Compute personal best pos
        my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store

        # Part 2: Update global best
        # Note that gbest computation is dependent on your topology
        if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
            my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

        # Let's print our output
        if i%20==0:
            print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))

        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        my_swarm.velocity = my_topology.compute_velocity(my_swarm)
        my_swarm.position = my_topology.compute_position(my_swarm,bounds)

    print('The best cost found by our swarm is: {:.4f}'.format(my_swarm.best_cost))
    print('The best position found by our swarm is: {}'.format(my_swarm.best_pos))
    return my_swarm.best_pos,my_swarm.best_cost
