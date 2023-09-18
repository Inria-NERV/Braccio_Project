import os
import time
import numpy as np
import mne
from functools import reduce
import operator
from Visualization_Data import *
from Spectral_Analysis import *
from Statistical_analysis import *
from file_loading import *
from math import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
from scipy.stats import norm
import statistics
def plot_step_lda(X_lda,y):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,3),('^', 's'),('blue', 'red')):

        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Power spectrum MI/Rest projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()



def Feature_shape(power_right,power_left,Trials,Electrodes_selected,Frequencies,electrodes):
    #Statistical Analysis
    Power_extraction_MI  = power_right[:,Electrodes_selected,:]
    Power_extraction_MI =Power_extraction_MI[:,:,Frequencies]
    Power_extraction_rest  = power_left[:,Electrodes_selected,:]
    Power_extraction_rest = Power_extraction_rest[:,:,Frequencies]
    mean_rest = Power_extraction_rest.mean(axis = 0)
    mean_mi = Power_extraction_MI.mean(axis = 0)
    X = np.concatenate((Power_extraction_MI,Power_extraction_rest),axis=0)
    X_2 = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    y = np.concatenate((np.ones(Trials),2*np.ones(Trials)))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, shuffle=False)
    print(X_train.shape)

    lda = LDA()
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    #print("Confusion matrix")
    #print(cm)
    #print("Weights")
    #print(classifier.coef_)
    #print("score")
    #print(classifier.score(X_test,y_test))
    Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1] + cm[1,0]+cm[0,1])


    return X,y,mean_rest,mean_mi,Accuracy

def LDA_classification(X,y,mean_rest,mean_mi,Number_features):
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    np.set_printoptions(precision=4)

    mean_vectors = []
    for cl in range(1,3):
        mean_vectors.append(np.mean(X[y==cl], axis=0))
        print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))


    S_W = np.zeros((Number_features,Number_features))
    for cl,mv in zip(range(1,3), mean_vectors):
        class_sc_mat = np.zeros((Number_features,Number_features))                  # scatter matrix for every class
        for row in X[y == cl]:
            row, mv = row.reshape(Number_features,1), mv.reshape(Number_features,1) # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                             # sum class scatter matrices
    print('within-class Scatter Matrix:\n', S_W)

    overall_mean = np.mean(X, axis=0)

    S_B = np.zeros((Number_features,Number_features))
    for i,mean_vec in enumerate(mean_vectors):
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(Number_features,1) # make column vector
        overall_mean = overall_mean.reshape(Number_features,1) # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    print('between-class Scatter Matrix:\n', S_B)


    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(Number_features,1)
        print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
        print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))


    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues

    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])

    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))


    W = np.hstack((eig_pairs[0][1].reshape(Number_features,1), eig_pairs[1][1].reshape(Number_features,1)))
    print('Matrix W:\n', W.real)


    X_lda = X.dot(W)
    plot_step_lda(X_lda,y)
#assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."





#

def complete_classif(power_right,power_left,Trials,Electrodes_selected,Frequencies,electrodes,Number_features,split):
    Power_extraction_MI  = power_right[:,Electrodes_selected,:]
    Power_extraction_MI =Power_extraction_MI[:,:,Frequencies]
    Power_extraction_rest  = power_left[:,Electrodes_selected,:]
    Power_extraction_rest = Power_extraction_rest[:,:,Frequencies]
    mean_rest = Power_extraction_rest.mean(axis = 0)
    mean_mi = Power_extraction_MI.mean(axis = 0)
    X = np.concatenate((Power_extraction_MI,Power_extraction_rest),axis=0)
    X_2 = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    y = np.concatenate((np.ones(Trials),2*np.ones(Trials)))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)
    print(X_train.shape)
    kf = KFold(n_splits=split,shuffle = True)
    kf.get_n_splits(X_2)
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    Accuracy = 0
    specificity = 0
    sensitivity = 0
    itera = 0
    for train_index,test_index in kf.split(X_2):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_2[train_index], X_2[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lda = LDA()
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sensitivity =  sensitivity+cm[0,0]/(cm[0,0]+cm[0,1])


        specificity = specificity+cm[1,1]/(cm[1,0]+cm[1,1])

        #print("Confusion matrix")
        print(cm)
        #print("Weights")
        #print(classifier.coef_)
        print("score")
        print(classifier.score(X_test,y_test))
        Accuracy = Accuracy + classifier.score(X_test,y_test)
        itera = itera + 1
        #Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1] + cm[1,0]+cm[0,1])


        np.set_printoptions(precision=4)

        mean_vectors = []
        for cl in range(1,3):
            mean_vectors.append(np.mean(X[y==cl], axis=0))
            print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))


        S_W = np.zeros((Number_features,Number_features))
        for cl,mv in zip(range(1,3), mean_vectors):
            class_sc_mat = np.zeros((Number_features,Number_features))                  # scatter matrix for every class
            for row in X[y == cl]:
                row, mv = row.reshape(Number_features,1), mv.reshape(Number_features,1) # make column vectors
                class_sc_mat += (row-mv).dot((row-mv).T)
            S_W += class_sc_mat                             # sum class scatter matrices
        print('within-class Scatter Matrix:\n', S_W)

        overall_mean = np.mean(X, axis=0)

        S_B = np.zeros((Number_features,Number_features))
        for i,mean_vec in enumerate(mean_vectors):
            n = X[y==i+1,:].shape[0]
            mean_vec = mean_vec.reshape(Number_features,1) # make column vector
            overall_mean = overall_mean.reshape(Number_features,1) # make column vector
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        print('between-class Scatter Matrix:\n', S_B)


        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        for i in range(len(eig_vals)):
            eigvec_sc = eig_vecs[:,i].reshape(Number_features,1)
            print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
            print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))


        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        # Visually confirm that the list is correctly sorted by decreasing eigenvalues

        print('Eigenvalues in decreasing order:\n')
        for i in eig_pairs:
            print(i[0])

        print('Variance explained:\n')
        eigv_sum = sum(eig_vals)
        for i,j in enumerate(eig_pairs):
            print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

        if Number_features==1:
            W = eig_pairs[0][0]
            X_lda = W*X
            print(W)
        else:
            W = np.hstack((eig_pairs[0][1].reshape(Number_features,1), eig_pairs[1][1].reshape(Number_features,1)))
            print('Matrix W:\n', W.real)
            X_lda = X.dot(W)
            #plot_step_lda(X_lda,y)
    return Accuracy/itera,sensitivity/itera,specificity/itera

def Feature_shape_SimpleSplit(power_right,power_left,Trials,Electrodes_selected,Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6,electrodes):
    #Statistical Analysis
    ListFreq = [Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6]
    Power_extraction_MI  = np.reshape(power_right[:,Electrodes_selected[0],ListFreq[0]],-1)
    Power_extraction_rest  = np.reshape(power_left[:,Electrodes_selected[0],ListFreq[0]],-1)
    if len(Electrodes_selected)>1:
        for i in range(1,len(Electrodes_selected)):
            Power_extraction_MI  = np.concatenate((Power_extraction_MI,np.reshape(power_right[:,Electrodes_selected[i],ListFreq[i]],-1)),axis = 0)
            Power_extraction_rest= np.concatenate((Power_extraction_rest,np.reshape(power_left[:,Electrodes_selected[i],ListFreq[i]],-1)),axis = 0)

    print(Power_extraction_MI)
    print(Power_extraction_MI.shape)
    mean_rest = Power_extraction_rest.mean(axis = 0)
    mean_mi = Power_extraction_MI.mean(axis = 0)
    X = np.concatenate((Power_extraction_MI,Power_extraction_rest),axis=0)

    X_inter = X.reshape(-1,1)
    X = X_inter
    y = np.concatenate((np.ones(Power_extraction_MI.shape[0]),2*np.ones(Power_extraction_rest.shape[0])),axis=0)
    y_inter = y.reshape(-1,1)
    y = y_inter
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
    #X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)
    print(X_train.shape)
    print(y_train.shape)
    lda = LDA()
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix")
    print(cm)
    #print("Weights")
    #print(classifier.coef_)
    print("score")
    print(classifier.score(X_test,y_test))
    Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1] + cm[1,0]+cm[0,1])
    sensitivity =  cm[0,0]/(cm[0,0]+cm[0,1])


    specificity = cm[1,1]/(cm[1,0]+cm[1,1])

    return Accuracy,sensitivity,specificity


def Feature_shape_trainVstest(power_right,power_left,power_right_test,power_left_test,Trials,Electrodes_selected,Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6,electrodes):
    #Statistical Analysis
    ListFreq = [Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6]
    Power_extraction_MI  = np.reshape(power_right[:,Electrodes_selected[0],ListFreq[0]],-1)
    Power_extraction_rest  = np.reshape(power_left[:,Electrodes_selected[0],ListFreq[0]],-1)
    print(Power_extraction_MI.shape)
    if len(Electrodes_selected)>1:
        for i in range(1,len(Electrodes_selected)):
            Power_extraction_MI  = np.concatenate((Power_extraction_MI,np.reshape(power_right[:,Electrodes_selected[i],ListFreq[i]],-1)),axis = 0)
            Power_extraction_rest= np.concatenate((Power_extraction_rest,np.reshape(power_left[:,Electrodes_selected[i],ListFreq[i]],-1)),axis = 0)


    mean_rest = Power_extraction_rest.mean(axis = 0)
    mean_mi = Power_extraction_MI.mean(axis = 0)
    X = np.concatenate((Power_extraction_MI,Power_extraction_rest),axis=0)

    X_train = X.reshape(-1,1)
    y_train = np.concatenate((np.ones(Power_extraction_MI.shape[0]),2*np.ones(Power_extraction_rest.shape[0])))
    y_inter_train = y_train.reshape(-1,1)
    y_train = y_inter_train
    Power_extraction_MI_test  = np.reshape(power_right_test[:,Electrodes_selected[0],ListFreq[0]],-1)
    Power_extraction_rest_test  = np.reshape(power_left_test[:,Electrodes_selected[0],ListFreq[0]],-1)
    print(Power_extraction_MI_test.shape)
    if len(Electrodes_selected)>1:
        for i in range(1,len(Electrodes_selected)):
            Power_extraction_MI_test  = np.concatenate((Power_extraction_MI_test,np.reshape(power_right[:,Electrodes_selected[i],ListFreq[i]],-1)),axis = 0)
            Power_extraction_rest_test= np.concatenate((Power_extraction_rest_test,np.reshape(power_left[:,Electrodes_selected[i],ListFreq[i]],-1)),axis = 0)


    mean_rest_test = Power_extraction_rest_test.mean(axis = 0)
    mean_mi_test = Power_extraction_MI_test.mean(axis = 0)
    X_test = np.concatenate((Power_extraction_MI_test,Power_extraction_rest_test),axis=0)
    X_test_inter = X_test.reshape(-1,1)
    X_test = X_test_inter
    y_test = np.concatenate((np.ones(Power_extraction_MI_test.shape[0]),2*np.ones(Power_extraction_rest_test.shape[0])))
    y_test_inter = y_test.reshape(-1,1)
    y_test = y_test_inter
    print(X_test.shape)
    #X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)
    #print(X_train.shape)
    X, Xdisc, y, ydisc = train_test_split(X_train, y_train, test_size = 0.2, shuffle = False)
    lda = LDA()
    X_train = lda.fit_transform(X, y)
    X_test = lda.transform(X_test)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix")
    print(cm)
    #print("Weights")
    #print(classifier.coef_)
    print("score")
    print(classifier.score(X_test,y_test))
    Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1] + cm[1,0]+cm[0,1])
    sensitivity =  cm[0,0]/(cm[0,0]+cm[0,1])


    specificity = cm[1,1]/(cm[1,0]+cm[1,1])

    return Accuracy,sensitivity,specificity






##############################


def Feature_shape_SimpleSplit_SVM(power_right,power_left,Trials,Electrodes_selected,Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6,electrodes):
    #Statistical Analysis
    SizeFrequ =  0
    ListFreq = [Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6]
    for i in ListFreq:
        SizeFrequ += len(i)

    # Power_extraction_MI  = list(power_right[:,Electrodes_selected[0],ListFreq[0]])
    # Power_extraction_MI = [list(x) for x in Power_extraction_MI]
    # Power_extraction_MI = list(map(lambda el:[el], reduce(operator.concat, Power_extraction_MI)))
    # Power_extraction_rest  = list(power_left[:,Electrodes_selected[0],ListFreq[0]])
    # Power_extraction_rest = list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in Power_extraction_rest])))
    # if len(Electrodes_selected)>1:
    #     for i in range(1,len(Electrodes_selected)):
    #         Power_extraction_MI  = Power_extraction_MI + list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in list(power_right[:,Electrodes_selected[i],ListFreq[i]])])))
    #         Power_extraction_rest= Power_extraction_rest + list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in list(power_left[:,Electrodes_selected[i],ListFreq[i]])])))

    PowerMI_Electrodes=power_right[:,Electrodes_selected,:]
    PowerRest_Electrodes=power_left[:,Electrodes_selected,:]

    PowerMI_Inter = np.zeros((PowerMI_Electrodes.shape[0],SizeFrequ))
    PowerRest_Inter = np.zeros((PowerRest_Electrodes.shape[0],SizeFrequ))
    for j in range(PowerMI_Electrodes.shape[0]):
        inter_MI = []
        inter_Rest = []
        for i in range(len(Electrodes_selected)):
            inter_MI += list(PowerMI_Electrodes[j,i,ListFreq[i]])
            inter_Rest += list(PowerRest_Electrodes[j,i,ListFreq[i]])
        print(inter_MI)
        for k in range(len(inter_MI)):
            PowerMI_Inter[j,k] = inter_MI[k]
            PowerRest_Inter[j,k] = inter_Rest[k]

    Power_extraction_MI = PowerMI_Inter
    Power_extraction_rest = PowerRest_Inter

    print(Power_extraction_MI)


    X = np.concatenate((Power_extraction_MI,Power_extraction_rest),axis = 0)
    y = [1] * (Power_extraction_MI.shape[0]) + [2]*(Power_extraction_rest.shape[0])
    print(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
    kf = KFold(n_splits=10,shuffle = True)
    kf.get_n_splits(X)
    #X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)
    Accuracy = 0
    itera = 0
    specificity = 0
    sensitivity = 0
    y = np.array(y)
    for train_index,test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        print(y)
        y_train, y_test = y[train_index], y[test_index]
        #clf = LinearDiscriminantAnalysis()
        clf = svm.SVC(kernel = 'rbf')
        X_tra = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(y_pred)
        print(y_test)

        # cm = confusion_matrix(y_test, y_pred)
        # #Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1] + cm[1,0]+cm[0,1])
        # if len(cm.shape) > 1:
        #     if ((cm[0,0]+cm[1,0])>0):
        #         sensitivity = sensitivity + cm[0,0]/(cm[1,0]+cm[0,0])
        #
        #     if ((cm[1,0]+cm[1,1])>0):
        #         specificity = specificity + cm[1,1]/(cm[1,0]+cm[1,1])
        # else:
        #     sensitivity = sensitivity + cm[1]
        #     specificity = specificity + cm[1]
        # #print("Confusion matrix")
        # print(cm)
        # #print("Weights")
        # #print(classifier.coef_)
        # print("score")
        # print(sensitivity)
        # print(specificity)

        Accuracy = Accuracy + accuracy_score(y_test, y_pred)
        itera = itera + 1
        #Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1] + cm[1,0]+cm[0,1])










    # print(y_train)
    # #clf = svm.SVC(kernel = 'rbf')
    # clf = LinearDiscriminantAnalysis()
    # X_tra = clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print("Weights : ")
    # print(clf.coef_)
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion matrix")
    # print(cm)
    # #print("Weights")
    # #print(classifier.coef_)
    # print("score")
    # print(accuracy_score(y_test, y_pred))
    # # Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1] + cm[1,0]+cm[0,1])
    # # sensitivity =  cm[1,1]/(cm[0,1]+cm[1,1])
    # #
    # #
    # # specificity = cm[1,1]/(cm[1,0]+cm[1,1])

    return Accuracy/itera#,sensitivity/itera,specificity/itera


def Feature_shape_TrainVsTest_SVM(power_right,power_left,power_right_test,power_left_test,Trials,Electrodes_selected,Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6,electrodes,R,W):
    #Statistical Analysis
    SizeFrequ =  0
    ListFreq = [Frequencies_list_1,Frequencies_list_2,Frequencies_list_3,Frequencies_list_4,Frequencies_list_5,Frequencies_list_6]
    for i in ListFreq:
        SizeFrequ += len(i)
    # Power_extraction_MI  = list(power_right[:,Electrodes_selected[0],ListFreq[0]])
    # Power_extraction_MI = list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in Power_extraction_MI])))
    # Power_extraction_rest  = list(power_left[:,Electrodes_selected[0],ListFreq[0]])
    # Power_extraction_rest = list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in Power_extraction_rest])))
    # if len(Electrodes_selected)>1:
    #     for i in range(1,len(Electrodes_selected)):
    #         Power_extraction_MI  = Power_extraction_MI + list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in list(power_right[:,Electrodes_selected[i],ListFreq[i]])])))
    #         Power_extraction_rest= Power_extraction_rest + list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in list(power_left[:,Electrodes_selected[i],ListFreq[i]])])))
    #
    # print(Power_extraction_MI)
    #
    #
    # X_train = Power_extraction_MI + Power_extraction_rest
    # y_train = [1] * len(Power_extraction_MI) + [2]*len(Power_extraction_rest)
    #
    # Power_extraction_MI_test  = list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in list(power_right_test[:,Electrodes_selected[0],ListFreq[0]])])))
    # Power_extraction_rest_test  = list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in list(power_left_test[:,Electrodes_selected[0],ListFreq[0]])])))
    # if len(Electrodes_selected)>1:
    #     for i in range(1,len(Electrodes_selected)):
    #         Power_extraction_MI_test  = Power_extraction_MI_test + list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in list(power_right_test[:,Electrodes_selected[i],ListFreq[i]])])))
    #         Power_extraction_rest_test= Power_extraction_rest_test + list(map(lambda el:[el],reduce(operator.concat,[list(x) for x in list(power_left_test[:,Electrodes_selected[i],ListFreq[i]])])))
    #
    # print(Power_extraction_MI_test)
    List_Electrode_asso_freq= []
    List_Index_Electrodes_freq = []
    for k in range(len(ListFreq)):
        for j in range(len(ListFreq[k])):
            List_Electrode_asso_freq.append(electrodes[Electrodes_selected[k]])
            List_Index_Electrodes_freq.append(Electrodes_selected[k])
    ListFreq_asso_elec = [item for sublist in ListFreq for item in sublist]
    PowerMI_Electrodes=power_right[:,Electrodes_selected,:]
    PowerRest_Electrodes=power_left[:,Electrodes_selected,:]

    PowerMI_Inter = np.zeros((PowerMI_Electrodes.shape[0],SizeFrequ))
    PowerRest_Inter = np.zeros((PowerRest_Electrodes.shape[0],SizeFrequ))
    for j in range(PowerMI_Electrodes.shape[0]):
        inter_MI = []
        inter_Rest = []
        for i in range(len(Electrodes_selected)):
            inter_MI += list(PowerMI_Electrodes[j,i,ListFreq[i]])
            inter_Rest += list(PowerRest_Electrodes[j,i,ListFreq[i]])

        for k in range(len(inter_MI)):
            PowerMI_Inter[j,k] = inter_MI[k]
            PowerRest_Inter[j,k] = inter_Rest[k]


    Power_extraction_MI = PowerMI_Inter
    Power_extraction_rest = PowerRest_Inter

    print(Power_extraction_MI)


    X_train = np.concatenate((Power_extraction_MI,Power_extraction_rest),axis = 0)
    y_train = [1] * (Power_extraction_MI.shape[0]) + [2]*(Power_extraction_rest.shape[0])


    PowerMI_Electrodes=power_right_test[:,Electrodes_selected,:]
    PowerRest_Electrodes=power_left_test[:,Electrodes_selected,:]

    PowerMI_Inter = np.zeros((PowerMI_Electrodes.shape[0],SizeFrequ))
    PowerRest_Inter = np.zeros((PowerRest_Electrodes.shape[0],SizeFrequ))
    for j in range(PowerMI_Electrodes.shape[0]):
        inter_MI = []
        inter_Rest = []
        for i in range(len(Electrodes_selected)):
            inter_MI += list(PowerMI_Electrodes[j,i,ListFreq[i]])
            inter_Rest += list(PowerRest_Electrodes[j,i,ListFreq[i]])

        for k in range(len(inter_MI)):
            PowerMI_Inter[j,k] = inter_MI[k]
            PowerRest_Inter[j,k] = inter_Rest[k]

    Power_extraction_MI_test = PowerMI_Inter
    Power_extraction_rest_test = PowerRest_Inter

    print(Power_extraction_MI_test)


    X_test = np.concatenate((Power_extraction_MI_test,Power_extraction_rest_test),axis = 0)
    y_test = [1] * (Power_extraction_MI_test.shape[0]) + [2]*(Power_extraction_rest_test.shape[0])




    #
    # X_test = Power_extraction_MI_test + Power_extraction_rest_test
    # y_test = [1] * len(Power_extraction_MI_test) + [2]*len(Power_extraction_rest_test)



    #X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)


    clf = svm.SVC(kernel = 'rbf')
    #clf = LinearDiscriminantAnalysis()
    X_tra = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_Each =[]
    cm = confusion_matrix(y_test, y_pred)
    cm_each_pred =[]
    print("Confusion matrix")
    print(cm)


    for k in range(X_test.shape[0]):
        var = clf.predict(X_test[k,:].reshape(1, -1))
        y_pred_Each.append(abs(1-abs(y_test[k]-var)).tolist())
        #cm_each_pred.append(confusion_matrix(y_test[k], var))
    flat_list = [item for sublist in y_pred_Each for item in sublist]
    # print(clf.means_)
    # print(X_test)
    # print(flat_list)
    #print("Weights")
    #print(classifier.coef_)
    print("score")
    print(accuracy_score(y_test, y_pred))
    Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1] + cm[1,0]+cm[0,1])
    sensitivity =  cm[1,1]/(cm[0,1]+cm[1,1])


    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    index_list_fail = np.where(np.array(flat_list) == 0)[0]
    y_fail = np.ones(len(index_list_fail))
    index_list_succeed = np.where(np.array(flat_list) == 1)[0]
    y_success = 2*np.ones(len(index_list_succeed))

    MI_Success = []
    Rest_Success = []
    MI_Fail =[]
    Rest_Fail = []
    R_to_store =[]
    W_to_store = []
    for z in range(len(y_test)):
        if(y_test[z] == 1):
            if flat_list[z] == 0:
                MI_Fail.append(X_test[z,:])
            if flat_list[z] == 1:
                MI_Success.append(X_test[z,:])
        if(y_test[z] == 2):
            if flat_list[z] == 0:
                Rest_Fail.append(X_test[z,:])
            if flat_list[z] == 1:
                Rest_Success.append(X_test[z,:])





    for i in range(X_train.shape[1]):
    # Set up the plot
        plt.figure()
        #ax = plt.subplot(X_train.shape[1], 2, i+1)

        # Draw the plot
        mu_mi, std_mi = norm.fit((Power_extraction_MI[:,i]))
        mu_rest, std_rest = norm.fit((Power_extraction_rest[:,i]))

        weights_mi = np.ones_like(Power_extraction_MI[:,i])/len(Power_extraction_MI[:,i])
        weights_rest = np.ones_like(Power_extraction_rest[:,i])/len(Power_extraction_rest[:,i])
        plt.hist(Power_extraction_MI[:,i], bins = 30,
                 color = 'red', edgecolor = 'black',alpha=0.5,weights = weights_mi)
        plt.hist(Power_extraction_rest[:,i], bins = 30,
                 color = 'blue', edgecolor = 'black',alpha=0.5,weights = weights_rest)
        # ax.plot(X_test[index_list_succeed,i],y_success,'go')
        # ax.plot(X_test[index_list_fail,i],y_fail,'r+')
        x_MI_succ = []
        x_Rest_succ = []
        x_MI_Fail = []
        x_Rest_Fail = []
        for k1 in range(len(MI_Success)):
            x_MI_succ.append(MI_Success[k1][i])
        for k1 in range(len(MI_Fail)):
            x_MI_Fail.append(MI_Fail[k1][i])

        for k1 in range(len(Rest_Success)):
            x_Rest_succ.append(Rest_Success[k1][i])
        for k1 in range(len(Rest_Fail)):
            x_Rest_Fail.append(Rest_Fail[k1][i])

        y_MI_Success = 0.2*np.ones(len(x_MI_succ))
        y_Rest_Success = 0.2*np.ones(len(x_Rest_succ))
        y_MI_Fail = 0.1*np.ones(len(x_MI_Fail))
        y_Rest_Fail = 0.1*np.ones(len(x_Rest_Fail))




        # for z in range(len(x_MI_succ[:,i])):
        #     for k in range(len(x)):
        #         if ((Feature_MI_Test[:,i][z]-x[k])<0.1):
        #             x_index.append(k)
        #             break
        #
        # for z in range(len(Feature_Rest_Test[:,i])):
        #     for k in range(len(x)):
        #         if ((Feature_Rest_Test[:,i][z]-x[k])<0.1):
        #             x_rest_index.append(k)
        #             break
        #
        #
        # for z in range(len(Feature_MI_Test_F[:,i])):
        #     for k in range(len(x)):
        #         if ((Feature_MI_Test_F[:,i][z]-x[k])<0.1):
        #             x_index_F.append(k)
        #             break
        #
        # for z in range(len(Feature_Rest_Test_F[:,i])):
        #     for k in range(len(x)):
        #         if ((Feature_Rest_Test_F[:,i][z]-x[k])<0.1):
        #             x_rest_index_F.append(k)
        #             break


        plt.plot((x_MI_succ),y_MI_Success,'ro')
        plt.plot((x_Rest_succ),y_Rest_Success,'bo')
        plt.plot((x_MI_Fail),y_MI_Fail,'rD')
        plt.plot((x_Rest_Fail),y_Rest_Fail,'bD')

        #Distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p_mi = norm.pdf(x, mu_mi, std_mi)
        p_rest = norm.pdf(x, mu_rest, std_rest)
        plt.plot(x, p_mi, 'r', linewidth=2)
        plt.plot(x, p_rest, 'b', linewidth=2)

        # Title and labels
        title = 'Distribution for ' + List_Electrode_asso_freq[i] + ' at ' + str(ListFreq_asso_elec[i]) + 'Hz' + '\n R^2 : ' + str(round(R[List_Index_Electrodes_freq[i],ListFreq_asso_elec[i]],2))
        #plt.text(2, -1, 'R^2 : ' + str(round(R[List_Index_Electrodes_freq[i],ListFreq_asso_elec[i]],4)), fontsize = 10)

        plt.title(title, size = 15)
        plt.xlabel('PowerSpectrum', size = 15)
        plt.ylabel('Occurence', size= 15)
        plt.legend(['Success MI','Success Rest','Failed MI','Failed Rest','Distribution MI','Distribution Rest','MI','Rest'],loc='center right')
        R_to_store.append(round(R[List_Index_Electrodes_freq[i],ListFreq_asso_elec[i]],2))
        W_to_store.append(round(W[List_Index_Electrodes_freq[i],ListFreq_asso_elec[i]],2))
    #plt.figlegend(['Success MI','Success Rest','Failed MI','Failed Rest','Distribution MI','Distribution Rest','MI','Rest'], loc = 'lower center', ncol=5, labelspacing=0.)
    #plt.tight_layout()
    print(R_to_store)
    print(W_to_store)
    plt.show()


    return Accuracy,sensitivity,specificity
