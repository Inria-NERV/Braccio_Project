# importing the module
import numpy as np
import glob
from os import listdir
import matplotlib.pyplot as plt
from scipy import signal,stats,fft
import matlab.engine
import json
import pickle
from scipy.interpolate import interp1d
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator

def open_the_gaze(fname):
    with open(fname, 'rb') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]

    Ldict =[]
    L_gaze2d = []
    #time = []
    for i in lines:
        loaded = json.loads(i)
        #time.append(loaded.get('timestamp'))
        #print(loaded)
        if loaded is not None and 'data' in loaded:
            if bool(loaded.get('data')):
                Ldict.append([loaded.get("data").get("eyeleft").get("pupildiameter"),loaded.get("data").get("eyeright").get("pupildiameter")])
                L_gaze2d.append([1000*loaded.get("data").get("gaze2d")[0],1000*loaded.get("data").get("gaze2d")[1]])
            else:
                Ldict.append([None,None])
                L_gaze2d.append([None,None])
        #print(json.loads(i))
    #data_dict = json.loads(data_str)
    #print(time)
    # diff = []
    # for l in range(len(time)-1):
    #     diff.append(time[l+1]-time[l])
    # print(np.array(diff).mean(0))
    #print(Ldict[0].get("data").get("eyeleft").get("pupildiameter"))
    return Ldict,L_gaze2d


# reading the data from the file
fname = '/Users/tristan.venot/Desktop/Braccio_Protocol/20190215T081614Z/gazedata2.txt'

directory_path_Strat_1 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/GazeData_Analysis/Strat_1/*'
directory_path_Strat_2 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/GazeData_Analysis/Strat_2/*'
directory_path_Strat_3 = '/Users/tristan.venot/Desktop/Experimentation_dataset/Batch2/GazeData_Analysis/Strat_3/*'


Strat_1_path_gen = sorted(glob.glob(directory_path_Strat_1))
Strat_2_path_gen = sorted(glob.glob(directory_path_Strat_2))
Strat_3_path_gen = sorted(glob.glob(directory_path_Strat_3))

Strat_1_Sub = []
Strat_1_Sub_gaze = []
for i in range(len(Strat_1_path_gen)):
    path_To_extract = Strat_1_path_gen[i] + '/*'
    print(Strat_1_path_gen[i])
    Strat_1_path_specific_subject = sorted(glob.glob(path_To_extract))
    Subject_files = []
    Subject_files_gaze = []
    for k in range(len(Strat_1_path_specific_subject)):
         print(Strat_1_path_specific_subject[k])
         #open_the_gaze(Strat_1_path_specific_subject[k])
         pupil,gaz2 = open_the_gaze(Strat_1_path_specific_subject[k])
         Subject_files.append(pupil)
         Subject_files_gaze.append(gaz2)
    Strat_1_Sub.append(Subject_files)
    Strat_1_Sub_gaze.append(Subject_files_gaze)
#
#
Strat_2_Sub = []
Strat_2_Sub_gaze = []
for i in range(len(Strat_2_path_gen)):
    path_To_extract = Strat_2_path_gen[i] + '/*'
    print(Strat_2_path_gen[i])
    Strat_2_path_specific_subject = sorted(glob.glob(path_To_extract))
    Subject_files = []
    Subject_files_gaze = []
    for k in range(len(Strat_2_path_specific_subject)):
         print(Strat_2_path_specific_subject[k])
         #open_the_gaze(Strat_1_path_specific_subject[k])
         pupil,gaz2 = open_the_gaze(Strat_2_path_specific_subject[k])
         Subject_files.append(pupil)
         Subject_files_gaze.append(gaz2)
    Strat_2_Sub.append(Subject_files)
    Strat_2_Sub_gaze.append(Subject_files_gaze)

Strat_3_Sub = []
Strat_3_Sub_gaze = []
for i in range(len(Strat_3_path_gen)):
    path_To_extract = Strat_3_path_gen[i] + '/*'
    print(Strat_3_path_gen[i])
    Strat_3_path_specific_subject = sorted(glob.glob(path_To_extract))
    Subject_files = []
    Subject_files_gaze = []
    for k in range(len(Strat_3_path_specific_subject)):
         print(Strat_3_path_specific_subject[k])
         #open_the_gaze(Strat_1_path_specific_subject[k])
         pupil,gaz2 = open_the_gaze(Strat_3_path_specific_subject[k])
         Subject_files.append(pupil)
         Subject_files_gaze.append(gaz2)
    Strat_3_Sub.append(Subject_files)
    Strat_3_Sub_gaze.append(Subject_files_gaze)

official_start = round(95/0.02)
end_to_cut = round(10/0.02)
## Strat 1

Traj_Left_st_1 = []
Traj_Right_st_1 = []
Traj_Gaze_st_1 = []
Number_Blinks = []
for i in range(len(Strat_1_Sub)):
    Number_Blinks_per_sub = []
    Sub_l_t=[]
    Sub_r_t=[]
    Sub_g_t = []
    for k in range(len(Strat_1_Sub[i])):
        Sub_l = []
        Sub_r = []
        Sub_g = []
        cpt = 0
        Tester = True
        for z in range(official_start,len(Strat_1_Sub[i][k])-end_to_cut):
            #print(len(Strat_1_Sub[i][k]))
            if Strat_1_Sub[i][k][z][0] is None and Strat_1_Sub[i][k][z][1] is None:
                if tester == True:
                    cpt +=1
                tester = False
            if Strat_1_Sub[i][k][z][0] is not None:
                Sub_l.append(Strat_1_Sub[i][k][z][0])
                if Strat_1_Sub[i][k][z][1] is not None:
                    Sub_r.append(Strat_1_Sub[i][k][z][1])
                    tester = True
            # if Strat_1_Sub_gaze[i][k][z] is not None:
            #     print(Strat_1_Sub_gaze[i][k][z])
            #     Sub_g.append(Strat_1_Sub_gaze[i][k][z])
        Sub_l_t.append(Sub_l)
        Sub_r_t.append(Sub_r)
        Number_Blinks_per_sub.append(cpt)
    Number_Blinks.append(Number_Blinks_per_sub)
    Traj_Left_st_1.append(Sub_l_t)
    Traj_Right_st_1.append(Sub_r_t)



##Strat 2

Traj_Left_st_2 = []
Traj_Right_st_2 = []
Number_Blinks_2 = []
for i in range(len(Strat_2_Sub)):
    Number_Blinks_per_sub = []
    Sub_l_t=[]
    Sub_r_t=[]
    for k in range(len(Strat_2_Sub[i])):
        Sub_l = []
        Sub_r = []
        cpt = 0
        Tester = True
        for z in range(official_start,len(Strat_2_Sub[i][k])-end_to_cut):
            #print(len(Strat_1_Sub[i][k]))
            if Strat_2_Sub[i][k][z][0] is None and Strat_2_Sub[i][k][z][1] is None:
                if tester == True:
                    cpt +=1
                tester = False
            if Strat_2_Sub[i][k][z][0] is not None:
                Sub_l.append(Strat_2_Sub[i][k][z][0])
                if Strat_2_Sub[i][k][z][1] is not None:
                    Sub_r.append(Strat_2_Sub[i][k][z][1])
                    tester = True
        Sub_l_t.append(Sub_l)
        Sub_r_t.append(Sub_r)
        Number_Blinks_per_sub.append(cpt)
    Number_Blinks_2.append(Number_Blinks_per_sub)
    Traj_Left_st_2.append(Sub_l_t)
    Traj_Right_st_2.append(Sub_r_t)


## Strat 3

Traj_Left_st_3 = []
Traj_Right_st_3 = []
Number_Blinks_3 = []
for i in range(len(Strat_3_Sub)):
    Number_Blinks_per_sub = []
    Sub_l_t=[]
    Sub_r_t=[]
    for k in range(len(Strat_3_Sub[i])):
        Sub_l = []
        Sub_r = []
        cpt = 0
        Tester = True
        for z in range(official_start,len(Strat_3_Sub[i][k])-end_to_cut):
            #print(len(Strat_1_Sub[i][k]))
            if Strat_3_Sub[i][k][z][0] is None and Strat_3_Sub[i][k][z][1] is None:
                if tester == True:
                    cpt +=1
                tester = False
            if Strat_3_Sub[i][k][z][0] is not None:
                Sub_l.append(Strat_3_Sub[i][k][z][0])
                if Strat_3_Sub[i][k][z][1] is not None:
                    Sub_r.append(Strat_3_Sub[i][k][z][1])
                    tester = True
        Sub_l_t.append(Sub_l)
        Sub_r_t.append(Sub_r)
        Number_Blinks_per_sub.append(cpt)
    Number_Blinks_3.append(Number_Blinks_per_sub)
    Traj_Left_st_3.append(Sub_l_t)
    Traj_Right_st_3.append(Sub_r_t)







# Analysis

# Number of blinks

Blinks_Strat_1 = []
Blinks_Strat_2 = []
Blinks_Strat_3 = []
List_Blinks_comp = []
List_Blinks_comp_std = []
for k in range(len(Number_Blinks)): ## Number of Subjects
    Number_ave = np.array(Number_Blinks[k])
    Blinks_Strat_1.append(Number_ave.mean(0))
    Number_ave_2 = np.array(Number_Blinks_2[k])
    Blinks_Strat_2.append(Number_ave_2.mean(0))
    Number_ave_3 = np.array(Number_Blinks_3[k])
    Blinks_Strat_3.append(Number_ave_3.mean(0))
    List_Blinks_= np.array(Number_Blinks[k]+Number_Blinks_2[k]+Number_Blinks_3[k])
    List_Blinks_comp.append(List_Blinks_.mean(0))
    List_Blinks_comp_std.append(List_Blinks_.std(0))


S1_z = (np.array(Blinks_Strat_1)-np.array(List_Blinks_comp))/np.array(List_Blinks_comp_std)
S2_z = (np.array(Blinks_Strat_2)-np.array(List_Blinks_comp))/np.array(List_Blinks_comp_std)
S3_z = (np.array(Blinks_Strat_3)-np.array(List_Blinks_comp))/np.array(List_Blinks_comp_std)

test_1,pval1 = stats.ranksums(S1_z,S3_z)
test_2,pval2 = stats.ranksums(S1_z,S2_z)
test_3,pval3 = stats.ranksums(S2_z,S3_z)



print(pval1)
print(pval2)
print(pval3)


fig, ax = plt.subplots()
import seaborn as sns
import matplotlib.pyplot as plt

a = S2_z
b = S1_z
c = S3_z
# Get the length of the largest dataset
max_len = max(len(a), len(b), len(c))

# Pad the smaller datasets with NaN values to match the size of the largest dataset
a = np.pad(a, (0, max_len - len(a)), 'constant', constant_values=(np.nan,))
b = np.pad(b, (0, max_len - len(b)), 'constant', constant_values=(np.nan,))
c = np.pad(c, (0, max_len - len(c)), 'constant', constant_values=(np.nan,))


# combine data into a single dataframe
data = {'Strategy 1': a, 'Strategy 2': b, 'Strategy 3': c}
df = pd.DataFrame(data)
subcat_order = ['Strategy 1', 'Strategy 2', 'Strategy 3']
plotting_parameters = {
    'data':    df,
    'order':   subcat_order,
    'palette': ['#2ca02c', '#1f77b4', '#d62728'],
}


pairs = [('Strategy 1', 'Strategy 3'),
         ('Strategy 1', 'Strategy 2'),
         ('Strategy 2', 'Strategy 3')]


pval = [pval1,pval2,pval3]

formated = [f"p={p:.2e}" for p in pval]


sns.barplot(data=df, palette=['#2ca02c', '#1f77b4', '#d62728'],capsize=.4,errorbar="se",estimator = 'median')
# Add annotations
# annotator = Annotator(ax, pairs, **plotting_parameters)
# annotator.set_pvalues(pval)
# annotator.annotate()

plt.ylabel("Deviation to the average in the number of blinks ")
plt.title("Blinks Number for each strategy")

plt.show()


##### Fixation Analysis
## Strat 1
Distance_Strat_1=[]

for k in range(len(Strat_1_Sub_gaze)):
    Distance_sub=[]
    for j in range(len(Strat_1_Sub_gaze[k])):
        Distance_run=[]
        for z in range(official_start,len(Strat_1_Sub_gaze[k][j])-end_to_cut-1):
            if Strat_1_Sub_gaze[k][j][z][0] is not None and Strat_1_Sub_gaze[k][j][z+1][0] is not None:
                distance = np.sqrt((Strat_1_Sub_gaze[k][j][z+1][0]-Strat_1_Sub_gaze[k][j][z][0])**2 +(Strat_1_Sub_gaze[k][j][z+1][1]-Strat_1_Sub_gaze[k][j][z][1])**2)
                Distance_run.append(distance)
        Distance_sub.append(Distance_run)
    Distance_Strat_1.append(Distance_sub)

Velocity_Strat_1=[]
for k in range(len(Distance_Strat_1)):
    Velocity_sub=[]
    for j in range(len(Distance_Strat_1[k])):
        Velocity_run = np.array(Distance_Strat_1[k][j])/0.02
        #print(Velocity_run)
        Velocity_sub.append(Velocity_run)
    Velocity_Strat_1.append(Velocity_sub)

Acceleration_Strat_1=[]
for k in range(len(Velocity_Strat_1)):
    Acceleration_sub=[]
    for j in range(len(Velocity_Strat_1[k])):
        Acceleration_run = []
        for z in range(len(Velocity_Strat_1[k][j])-1):
            Acceleration_run.append((Velocity_Strat_1[k][j][z+1]-Velocity_Strat_1[k][j][z])/0.02)
        #print(Acceleration_run)
        Acceleration_sub.append(Acceleration_run)
    Acceleration_Strat_1.append(Acceleration_sub)

Fixation_Strat_1 = []
for k in range(len(Velocity_Strat_1)):
    Fixation_sub = []
    for j in range(len(Velocity_Strat_1[k])):
        cpt_fix = 0
        Cpt_fixation = 0
        AverVel = np.array(Velocity_Strat_1[k][j]).mean(0)
        AverAcc = np.array(Acceleration_Strat_1[k][j]).mean(0)
        for z in range(len(Velocity_Strat_1[k][j])-1):
            if (Velocity_Strat_1[k][j][z+1]<AverVel) and Acceleration_Strat_1[k][j][z]<AverAcc:
                cpt_fix +=1
            if cpt_fix >= 5:
                Cpt_fixation +=1
                cpt_fix = 0
        Fixation_sub.append(Cpt_fixation)
    Fixation_Strat_1.append(Fixation_sub)


## Strategy 2

Distance_Strat_2=[]

for k in range(len(Strat_2_Sub_gaze)):
    Distance_sub=[]
    for j in range(len(Strat_2_Sub_gaze[k])):
        Distance_run=[]
        for z in range(official_start,len(Strat_2_Sub_gaze[k][j])-end_to_cut-1):
            if Strat_2_Sub_gaze[k][j][z][0] is not None and Strat_2_Sub_gaze[k][j][z+1][0] is not None:
                distance = np.sqrt((Strat_2_Sub_gaze[k][j][z+1][0]-Strat_2_Sub_gaze[k][j][z][0])**2 +(Strat_2_Sub_gaze[k][j][z+1][1]-Strat_2_Sub_gaze[k][j][z][1])**2)
                Distance_run.append(distance)
        Distance_sub.append(Distance_run)
    Distance_Strat_2.append(Distance_sub)

Velocity_Strat_2=[]
for k in range(len(Distance_Strat_2)):
    Velocity_sub=[]
    for j in range(len(Distance_Strat_2[k])):
        Velocity_run = np.array(Distance_Strat_2[k][j])/0.02
        #print(Velocity_run)
        Velocity_sub.append(Velocity_run)
    Velocity_Strat_2.append(Velocity_sub)

Acceleration_Strat_2=[]
for k in range(len(Velocity_Strat_2)):
    Acceleration_sub=[]
    for j in range(len(Velocity_Strat_2[k])):
        Acceleration_run = []
        for z in range(len(Velocity_Strat_2[k][j])-1):
            Acceleration_run.append((Velocity_Strat_2[k][j][z+1]-Velocity_Strat_2[k][j][z])/0.02)
        #print(Acceleration_run)
        Acceleration_sub.append(Acceleration_run)
    Acceleration_Strat_2.append(Acceleration_sub)

Fixation_Strat_2 = []
for k in range(len(Velocity_Strat_2)):
    Fixation_sub = []
    for j in range(len(Velocity_Strat_2[k])):
        cpt_fix = 0
        Cpt_fixation = 0
        AverVel = np.array(Velocity_Strat_2[k][j]).mean(0)
        AverAcc = np.array(Acceleration_Strat_2[k][j]).mean(0)
        for z in range(len(Velocity_Strat_2[k][j])-1):
            if (Velocity_Strat_2[k][j][z+1]<AverVel) and Acceleration_Strat_2[k][j][z]<AverAcc:
                cpt_fix +=1
            if cpt_fix >= 5:
                Cpt_fixation +=1
                cpt_fix = 0
        Fixation_sub.append(Cpt_fixation)
    Fixation_Strat_2.append(Fixation_sub)

### Strategy 3

Distance_Strat_3=[]

for k in range(len(Strat_3_Sub_gaze)):
    Distance_sub=[]
    for j in range(len(Strat_3_Sub_gaze[k])):
        Distance_run=[]
        for z in range(official_start,len(Strat_3_Sub_gaze[k][j])-end_to_cut-1):
            if Strat_3_Sub_gaze[k][j][z][0] is not None and Strat_3_Sub_gaze[k][j][z+1][0] is not None:
                distance = np.sqrt((Strat_3_Sub_gaze[k][j][z+1][0]-Strat_3_Sub_gaze[k][j][z][0])**2 +(Strat_3_Sub_gaze[k][j][z+1][1]-Strat_3_Sub_gaze[k][j][z][1])**2)
                Distance_run.append(distance)
        Distance_sub.append(Distance_run)
    Distance_Strat_3.append(Distance_sub)

Velocity_Strat_3=[]
for k in range(len(Distance_Strat_3)):
    Velocity_sub=[]
    for j in range(len(Distance_Strat_3[k])):
        Velocity_run = np.array(Distance_Strat_3[k][j])/0.02
        #print(Velocity_run)
        Velocity_sub.append(Velocity_run)
    Velocity_Strat_3.append(Velocity_sub)

Acceleration_Strat_3=[]
for k in range(len(Velocity_Strat_3)):
    Acceleration_sub=[]
    for j in range(len(Velocity_Strat_3[k])):
        Acceleration_run = []
        for z in range(len(Velocity_Strat_3[k][j])-1):
            Acceleration_run.append((Velocity_Strat_3[k][j][z+1]-Velocity_Strat_3[k][j][z])/0.02)
        #print(Acceleration_run)
        Acceleration_sub.append(Acceleration_run)
    Acceleration_Strat_3.append(Acceleration_sub)

Fixation_Strat_3 = []
for k in range(len(Velocity_Strat_3)):
    Fixation_sub = []
    for j in range(len(Velocity_Strat_3[k])):
        cpt_fix = 0
        Cpt_fixation = 0
        AverVel = np.array(Velocity_Strat_3[k][j]).mean(0)
        AverAcc = np.array(Acceleration_Strat_3[k][j]).mean(0)
        for z in range(len(Velocity_Strat_3[k][j])-1):
            if (Velocity_Strat_3[k][j][z+1]<AverVel) and Acceleration_Strat_3[k][j][z]<AverAcc:
                cpt_fix +=1
            if cpt_fix >= 5:
                Cpt_fixation +=1
                cpt_fix = 0
        Fixation_sub.append(Cpt_fixation)
    Fixation_Strat_3.append(Fixation_sub)



Fix_Strat_1 = []
Fix_Strat_2 = []
Fix_Strat_3 = []
List_Fix_comp=[]
List_Fix_comp_std=[]
for k in range(len(Fixation_Strat_3)):
    Fix_ave_sub_1 = np.array(Fixation_Strat_1[k])
    print(Fix_ave_sub_1.shape)
    Fix_ave_sub_2 = np.array(Fixation_Strat_2[k])
    Fix_ave_sub_3 = np.array(Fixation_Strat_3[k])
    Fix_Strat_1.append(Fix_ave_sub_1.mean(0))
    Fix_Strat_2.append(Fix_ave_sub_2.mean(0))
    Fix_Strat_3.append(Fix_ave_sub_3.mean(0))
    List_Fix_= np.array(Fixation_Strat_1[k]+Fixation_Strat_2[k]+Fixation_Strat_3[k])
    print(List_Fix_.shape)
    List_Fix_comp.append(List_Fix_.mean(0))
    List_Fix_comp_std.append(List_Fix_.std(0))


S1_z = ((np.array(Fix_Strat_1)-np.array(List_Fix_comp))/np.array(List_Fix_comp_std))
S2_z = ((np.array(Fix_Strat_2)-np.array(List_Fix_comp))/np.array(List_Fix_comp_std))
S3_z = ((np.array(Fix_Strat_3)-np.array(List_Fix_comp))/np.array(List_Fix_comp_std))

test_1,pval1 = stats.ranksums(S1_z,S3_z)
test_2,pval2 = stats.ranksums(S1_z,S2_z)
test_3,pval3 = stats.ranksums(S2_z,S3_z)



print(pval1)
print(pval2)
print(pval3)


fig, ax = plt.subplots()
import seaborn as sns
import matplotlib.pyplot as plt

a = S2_z
b = S1_z
c = S3_z
# Get the length of the largest dataset
max_len = max(len(a), len(b), len(c))

# Pad the smaller datasets with NaN values to match the size of the largest dataset
a = np.pad(a, (0, max_len - len(a)), 'constant', constant_values=(np.nan,))
b = np.pad(b, (0, max_len - len(b)), 'constant', constant_values=(np.nan,))
c = np.pad(c, (0, max_len - len(c)), 'constant', constant_values=(np.nan,))


# combine data into a single dataframe
data = {'Strategy 1': a, 'Strategy 2': b, 'Strategy 3': c}
df = pd.DataFrame(data)
subcat_order = ['Strategy 1', 'Strategy 2', 'Strategy 3']
plotting_parameters = {
    'data':    df,
    'order':   subcat_order,
    'palette': ['#2ca02c', '#1f77b4', '#d62728'],
}


pairs = [('Strategy 1', 'Strategy 3'),
         ('Strategy 1', 'Strategy 2'),
         ('Strategy 2', 'Strategy 3')]


pval = [pval1,pval2,pval3]

formated = [f"p={p:.2e}" for p in pval]


sns.barplot(data=df, palette=['#2ca02c', '#1f77b4', '#d62728'],capsize=.4,errorbar="se",estimator = 'median')
# Add annotations
# annotator = Annotator(ax, pairs, **plotting_parameters)
# annotator.set_pvalues(pval)
# annotator.annotate()

plt.ylabel("Deviation to the average in number of fixations")
plt.title("Fixations Number for each strategy")

plt.show()



##Pupil Diameter analysis
# Make figure for processing

Sub_pupil_1 = []
for k in range(len(Traj_Left_st_1)):##n_subjects
    Run_pupil=[]
    for j in range(len(Traj_Left_st_1[k])):##n_runs
        if len(Traj_Left_st_1[k][j])>0:
            Average_Run = np.array(Traj_Left_st_1[k][j]).mean(0)
            Derivative_run = np.diff(Traj_Left_st_1[k][j]).mean(0)
            Max_run = np.max(Traj_Left_st_1[k][j])
            Min_run = np.min(Traj_Left_st_1[k][j])
            FFt,t,z = signal.stft(Traj_Left_st_1[k][j],fs=50)
            FFT_run_mean = FFt[0:5].mean(0)
            Run = [Average_Run,Max_run,Min_run,FFT_run_mean]
            Run_pupil.append(Run)
            print("Hello")
    Sub_pupil_1.append(Run_pupil)


Sub_pupil_2 = []
for k in range(len(Traj_Left_st_2)):##n_subjects
    Run_pupil=[]
    for j in range(len(Traj_Left_st_2[k])):##n_runs
        if len(Traj_Left_st_2[k][j])>0:
            Average_Run = np.array(Traj_Left_st_2[k][j]).mean(0)
            Derivative_run = np.diff(Traj_Left_st_2[k][j]).mean(0)
            Max_run = np.max(Traj_Left_st_2[k][j])
            Min_run = np.min(Traj_Left_st_2[k][j])
            FFt,t,z = signal.stft(Traj_Left_st_2[k][j],fs=50)
            FFT_run_mean = FFt[0:5].mean(0)
            Run = [Average_Run,Max_run,Min_run,FFT_run_mean]
            Run_pupil.append(Run)
            print("Hello2")
    Sub_pupil_2.append(Run_pupil)

Sub_pupil_3 = []
for k in range(len(Traj_Left_st_3)):##n_subjects
    Run_pupil=[]
    for j in range(len(Traj_Left_st_3[k])):##n_runs
        if len(Traj_Left_st_3[k][j])>0:
            Average_Run = np.array(Traj_Left_st_3[k][j]).mean(0)
            Derivative_run = np.diff(Traj_Left_st_3[k][j]).mean(0)
            Max_run = np.max(Traj_Left_st_3[k][j])
            Min_run = np.min(Traj_Left_st_3[k][j])
            FFt,t,z = signal.stft(Traj_Left_st_3[k][j],fs=50)
            FFT_run_mean = FFt[0:5].mean(0)
            Run = [Average_Run,Max_run,Min_run,FFT_run_mean]
            Run_pupil.append(Run)
            print("Hello3")
    Sub_pupil_3.append(Run_pupil)

Sub_Pupil1 = []
Sub_Pupil2 = []
Sub_Pupil3 = []
for k in range(len(Sub_pupil_3)):
    Sub_pupil_1_ave = np.array([np.array(Sub_pupil_1[k][0]).mean(0),np.array(Sub_pupil_1[k][1]).mean(0),np.array(Sub_pupil_1[k][2]).mean(0),np.array(Sub_pupil_1[k][3]).mean(0),np.array(Sub_pupil_1[k][4]).mean(0)])
    Sub_pupil_2_ave = np.array([np.array(Sub_pupil_2[k][0]).mean(0),np.array(Sub_pupil_2[k][1]).mean(0),np.array(Sub_pupil_2[k][2]).mean(0),np.array(Sub_pupil_2[k][3]).mean(0),np.array(Sub_pupil_2[k][4]).mean(0)])
    Sub_pupil_3_ave = np.array([np.array(Sub_pupil_3[k][0]).mean(0),np.array(Sub_pupil_3[k][1]).mean(0),np.array(Sub_pupil_3[k][2]).mean(0),np.array(Sub_pupil_3[k][3]).mean(0),np.array(Sub_pupil_3[k][4]).mean(0)])
    Sub_Pupil1.append(Sub_pupil_1_ave)
    Sub_Pupil2.append(Sub_pupil_2_ave)
    Sub_Pupil3.append(Sub_pupil_3_ave)
    print("HEY")

Name  = ['Mean', 'Mean Derivative', 'Max','Min','MFBA']
for dim in range(5):

    S1_z = np.array(Sub_Pupil1)[dim,:]
    S2_z = np.array(Sub_Pupil2)[dim,:]
    S3_z = np.array(Sub_Pupil3)[dim,:]

    test_1,pval1 = stats.ranksums(S1_z,S3_z)
    test_2,pval2 = stats.ranksums(S1_z,S2_z)
    test_3,pval3 = stats.ranksums(S2_z,S3_z)



    print(pval1)
    print(pval2)
    print(pval3)


    fig, ax = plt.subplots()
    import seaborn as sns
    import matplotlib.pyplot as plt

    a = S2_z
    b = S1_z
    c = S3_z
    # Get the length of the largest dataset
    max_len = max(len(a), len(b), len(c))

    # Pad the smaller datasets with NaN values to match the size of the largest dataset
    a = np.pad(a, (0, max_len - len(a)), 'constant', constant_values=(np.nan,))
    b = np.pad(b, (0, max_len - len(b)), 'constant', constant_values=(np.nan,))
    c = np.pad(c, (0, max_len - len(c)), 'constant', constant_values=(np.nan,))


    # combine data into a single dataframe
    data = {'Strategy 1': a, 'Strategy 2': b, 'Strategy 3': c}
    df = pd.DataFrame(data)
    subcat_order = ['Strategy 1', 'Strategy 2', 'Strategy 3']
    plotting_parameters = {
        'data':    df,
        'order':   subcat_order,
        'palette': ['#2ca02c', '#1f77b4', '#d62728'],
    }


    pairs = [('Strategy 1', 'Strategy 3'),
             ('Strategy 1', 'Strategy 2'),
             ('Strategy 2', 'Strategy 3')]


    pval = [pval3,pval2,pval1]

    formated = [f"p={p:.2e}" for p in pval]


    sns.boxplot(data=df, palette=['#2ca02c','#1f77b4',  '#d62728'])#,capsize=.4,errorbar="se",estimator = 'median')
    # Add annotations
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.set_pvalues(pval)
    annotator.annotate()

    Title = Name[dim]
    plt.ylabel(Title+" Left Pupil")
    plt.title(Title+" for each strategy")

    plt.show()




### Right pupil

Sub_pupil_1 = []
for k in range(len(Traj_Right_st_1)):##n_subjects
    Run_pupil=[]
    for j in range(len(Traj_Right_st_1[k])):##n_runs
        if len(Traj_Right_st_1[k][j])>0:
            Average_Run = np.array(Traj_Right_st_1[k][j]).mean(0)
            Derivative_run = np.diff(Traj_Right_st_1[k][j]).mean(0)
            Max_run = np.max(Traj_Right_st_1[k][j])
            Min_run = np.min(Traj_Right_st_1[k][j])
            FFt,t,z = signal.stft(Traj_Right_st_1[k][j],fs=50)
            FFT_run_mean = FFt[0:5].mean(0)
            Run = [Average_Run,Max_run,Min_run,FFT_run_mean]
            Run_pupil.append(Run)
            print("Hello")
    Sub_pupil_1.append(Run_pupil)


Sub_pupil_2 = []
for k in range(len(Traj_Right_st_2)):##n_subjects
    Run_pupil=[]
    for j in range(len(Traj_Left_st_2[k])):##n_runs
        if len(Traj_Left_st_2[k][j])>0:
            Average_Run = np.array(Traj_Right_st_2[k][j]).mean(0)
            Derivative_run = np.diff(Traj_Right_st_2[k][j]).mean(0)
            Max_run = np.max(Traj_Right_st_2[k][j])
            Min_run = np.min(Traj_Right_st_2[k][j])
            FFt,t,z = signal.stft(Traj_Right_st_2[k][j],fs=50)
            FFT_run_mean = FFt[0:5].mean(0)
            Run = [Average_Run,Max_run,Min_run,FFT_run_mean]
            Run_pupil.append(Run)
            print("Hello2")
    Sub_pupil_2.append(Run_pupil)

Sub_pupil_3 = []
for k in range(len(Traj_Right_st_3)):##n_subjects
    Run_pupil=[]
    for j in range(len(Traj_Right_st_3[k])):##n_runs
        if len(Traj_Left_st_3[k][j])>0:
            Average_Run = np.array(Traj_Right_st_3[k][j]).mean(0)
            Derivative_run = np.diff(Traj_Right_st_3[k][j]).mean(0)
            Max_run = np.max(Traj_Right_st_3[k][j])
            Min_run = np.min(Traj_Right_st_3[k][j])
            FFt,t,z = signal.stft(Traj_Right_st_3[k][j],fs=50)
            FFT_run_mean = FFt[0:5].mean(0)
            Run = [Average_Run,Max_run,Min_run,FFT_run_mean]
            Run_pupil.append(Run)
            print("Hello3")
    Sub_pupil_3.append(Run_pupil)

Sub_Pupil1 = []
Sub_Pupil2 = []
Sub_Pupil3 = []
for k in range(len(Sub_pupil_3)):
    Sub_pupil_1_ave = np.array([np.array(Sub_pupil_1[k][0]).mean(0),np.array(Sub_pupil_1[k][1]).mean(0),np.array(Sub_pupil_1[k][2]).mean(0),np.array(Sub_pupil_1[k][3]).mean(0),np.array(Sub_pupil_1[k][4]).mean(0)])
    Sub_pupil_2_ave = np.array([np.array(Sub_pupil_2[k][0]).mean(0),np.array(Sub_pupil_2[k][1]).mean(0),np.array(Sub_pupil_2[k][2]).mean(0),np.array(Sub_pupil_2[k][3]).mean(0),np.array(Sub_pupil_2[k][4]).mean(0)])
    Sub_pupil_3_ave = np.array([np.array(Sub_pupil_3[k][0]).mean(0),np.array(Sub_pupil_3[k][1]).mean(0),np.array(Sub_pupil_3[k][2]).mean(0),np.array(Sub_pupil_3[k][3]).mean(0),np.array(Sub_pupil_3[k][4]).mean(0)])
    Sub_Pupil1.append(Sub_pupil_1_ave)
    Sub_Pupil2.append(Sub_pupil_2_ave)
    Sub_Pupil3.append(Sub_pupil_3_ave)
    print("HEY")

Name  = ['Mean', 'Mean Derivative', 'Max','Min','MFBA']
for dim in range(5):

    S1_z = np.array(Sub_Pupil2)[dim,:]
    S2_z = np.array(Sub_Pupil3)[dim,:]
    S3_z = np.array(Sub_Pupil1)[dim,:]

    test_1,pval1 = stats.ranksums(S1_z,S3_z)
    test_2,pval2 = stats.ranksums(S1_z,S2_z)
    test_3,pval3 = stats.ranksums(S2_z,S3_z)



    print(pval1)
    print(pval2)
    print(pval3)


    fig, ax = plt.subplots()
    import seaborn as sns
    import matplotlib.pyplot as plt

    a = S1_z
    b = S2_z
    c = S3_z
    # Get the length of the largest dataset
    max_len = max(len(a), len(b), len(c))

    # Pad the smaller datasets with NaN values to match the size of the largest dataset
    a = np.pad(a, (0, max_len - len(a)), 'constant', constant_values=(np.nan,))
    b = np.pad(b, (0, max_len - len(b)), 'constant', constant_values=(np.nan,))
    c = np.pad(c, (0, max_len - len(c)), 'constant', constant_values=(np.nan,))


    # combine data into a single dataframe
    data = {'Strategy 1': a, 'Strategy 2': b, 'Strategy 3': c}
    df = pd.DataFrame(data)
    subcat_order = ['Strategy 1', 'Strategy 2', 'Strategy 3']
    plotting_parameters = {
        'data':    df,
        'order':   subcat_order,
        'palette': ['#2ca02c','#d62728',  '#1f77b4'],
    }


    pairs = [('Strategy 1', 'Strategy 3'),
             ('Strategy 1', 'Strategy 2'),
             ('Strategy 2', 'Strategy 3')]


    pval = [pval3,pval2,pval1]

    formated = [f"p={p:.2e}" for p in pval]


    sns.boxplot(data=df, palette=['#2ca02c','#d62728',  '#1f77b4'])#,capsize=.4,errorbar="se",estimator = 'median')
    # Add annotations
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.set_pvalues(pval)
    annotator.annotate()
    Title = Name[dim]
    plt.ylabel(Title+" Right Pupil")
    plt.title(Title+" for each strategy")

    plt.show()
