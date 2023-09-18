import numpy as np
import mne
import os.path as op
from os import listdir
import matplotlib.pyplot as plt
import mne
import matplotlib.cm as cm
from mne.datasets import fetch_fsaverage
plt.interactive(False)

arr = np.loadtxt("/Users/tristan.venot/Desktop/TravailThèse/openvibe-scripting/Cluster_FC_St3_cali_b1.csv",
                 delimiter=",", dtype=float)



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
electrodes_to_keep = ['FC1','C3','CP5','CP1','CP6','CP2','Cz','C4','FC6','FC2','FC3','C1','C5','CP3','CPz','CP4','C6','C2','FC4']
#electrodes_to_keep = ['C3']#,'CP5','CP1','CP6','CP2','Cz','C4','FC6','FC2','FC3','C1','C5','CP3','CPz','CP4','C6','C2','FC4']

mask_ar = np.ones(64)
for k in range(len(electrodes)):
   for j in range(len(electrodes_to_keep)):
        if electrodes[k]==electrodes_to_keep[j]:
            print(electrodes[k])
            mask_ar[k] = arr[k]
            print(arr[k])





biosemi_montage_inter = mne.channels.make_standard_montage('standard_1020')


ind = [i for (i, channel) in enumerate(biosemi_montage_inter.ch_names) if channel in electrodes]
biosemi_montage = biosemi_montage_inter.copy()
# Keep only the desired channels
biosemi_montage.ch_names = [biosemi_montage_inter.ch_names[x] for x in ind]
print(biosemi_montage.ch_names )
kept_channel_info = [biosemi_montage_inter.dig[x+3] for x in ind]
# Keep the first three rows as they are the fiducial points information
biosemi_montage.dig = biosemi_montage_inter.dig[0:3]+kept_channel_info
    #biosemi_montage = mne.channels.make_standard_montage('standard_1020')
n_channels = len(biosemi_montage.ch_names)
fake_info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=500/2,
                            ch_types='eeg')
pos = np.stack([biosemi_montage.get_positions()['ch_pos'][ch] for ch in electrodes])
print(biosemi_montage)
fake_info.set_montage(biosemi_montage)

t_values = np.random.randn(64)  # Example t-values

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
brain = mne.viz.Brain(
    "fsaverage", subjects_dir=subjects_dir, background="w",hemi='both',cortex=(0.7, 0.7, 0.7),alpha = 1
)

mask_ar_reag = np.ones(64)
for k in range(len(biosemi_montage.ch_names)):
    for l in range(len(electrodes)):
        if biosemi_montage.ch_names[k]==electrodes[l]:
            mask_ar_reag[k]=mask_ar[l]
# Add the sensors as spheres with sizes based on t-values
t_values = mask_ar_reag

# Create the colormap (Reds colormap)

cmap = plt.colormaps['YlOrRd']

# Normalize the values in your list to be within the range [0, 1]
norm = plt.Normalize(min(t_values), max(t_values))
normalized = norm(t_values)
modified = 3*(t_values/ max(t_values))**4.5
#modified = t_values-t_values
# Map the values to colors using the colormap
colors_elect = cmap(normalized)
print(colors_elect)

brain.add_sensors(fake_info, trans,eeg='original',colors_elect=colors_elect,tvalues=0.5+modified)
#for i in range(64):
#    brain.add_text(x=pos[i,0]+0.4,y = pos[i,1]+0.5,text = electrodes[i])
brain.add_head(dense =False,alpha=0.1)
# Show the brain plot
brain.show_view(distance=500,view='dorsal')
fig, ax = plt.subplots()
plt.imshow(t_values[:, np.newaxis],cmap='YlOrRd')
plt.colorbar()
plt.show()