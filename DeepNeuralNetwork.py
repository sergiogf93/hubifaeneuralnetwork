#!/bin/python
from optparse import OptionParser
import os

import numpy as np
import pandas as pd
import sklearn.utils
from bisect import bisect_right
from random import random,choice,seed
from numpy.lib.recfunctions import stack_arrays
import glob

from IPython.display import clear_output, display

from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
    
import math as m    

#****************************************
#Adding Options when runing the code. 
#****************************************
def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

def Randomizing(df):
    df = sklearn.utils.shuffle(df,random_state=123) #'123' is the random seed
    df = df.reset_index(drop=True)# drop=True does not allow the reset_index to insert a new column with the suffled index
    return df

def BuildDNN(N_input,width,depth):
    print "Building model with %d input nodes, %d inner nodes and %d inner layers" %(N_input, width, depth)
    model = Sequential()
    model.add(Dense(units=width, input_dim=N_input))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    for i in xrange(0, depth):
        model.add(Dense(int(width/(2*(i+1)))))
        model.add(Activation('relu'))
        # Dropout randomly sets a fraction of input units to 0 at each update during training time
        # which helps prevent overfitting.
        model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    return model

def weighted_selection(df,w,N):
    seed(123)
    w2 = w.values.copy()
    indices = np.array(range(len(df)))
    r = []
    sumw = 0
    n = 0
    prevPrint = -1
    while sumw < N:
        if len(indices) == 0:
            print "ERROR: not enough events to get weighted amount"
            return -1
        i = choice(indices)
        sumw += w2[i]
        r.append(i)
        if int(sumw) % 100 == 0 and prevPrint != int(sumw):
            clear_output(wait=True)
            prevPrint = int(sumw)
            print "Current sumw %d of %d" % (prevPrint,N),
        np.delete(indices,np.where(indices == i))
    return df.iloc[r]
        
def integrateHist(n, bins, x0=None, x1=None): #important
    s = 0.
    if x0 == None:
        x0 = bins[0]
    if x1 == None:
        x1 = bins[-1]
    for k in range(len(n)):
        if bins[k] >= x0 and bins[k] <= x1:
            s += m.fabs(n[k])*m.fabs(bins[k+1] - bins[k])
    return s
    
def weighted_avg(dataset, branch, selected_idx):
    return (dataset[branch].values[selected_idx] * dataset["weight"].values[selected_idx]).sum() / dataset["weight"].values[selected_idx].sum()

def getTcut(n_sig, n_bkg, bins): #important
    alpha = np.cumsum(n_sig/len(bins))
    beta = 1 - np.cumsum(n_bkg/len(bins))
    signal_to_noise_ratio = []
    for i in range(len(alpha)):
        if beta[i] > 0:
            signal_to_noise_ratio.append((1. - alpha[i])/m.sqrt(beta[i]))
        else:
            signal_to_noise_ratio.append(0)
    T_cut = bins[np.argmax(signal_to_noise_ratio)]
    return {'T_cut':T_cut,'alpha':alpha,'beta':beta,'signal_to_noise_ratio':signal_to_noise_ratio}

def saveModel(model,scaler,le,name):
    os.system('mkdir -p Models/' + name)
    model.save('Models/' + name + '/model.h5')
    joblib.dump(scaler, "Models/" + name + "/scaler.save")
    joblib.dump(le, "Models/" + name + "/le.save")

def loadModel(path_to_model):
    print "Loading model from " + path_to_model
    model = load_model(path_to_model + '/model.h5')
    scaler = joblib.load(path_to_model + '/scaler.save')
    le = joblib.load(path_to_model + '/le.save')
    return model, scaler, le

#######################################
parser = OptionParser()
parser.add_option('-s', '--SigPkl',
                  type='string',
                  help='Signal pickle path', 
                  dest = "sigPkl")
parser.add_option('-b', '--BkgPkl',
                  type='string',
                  help='Background pickle path',
                  dest = "bkgPkl")
parser.add_option('--SigName',type='string',help='Label for signal sample',dest='sigName',default="Znunu")
parser.add_option('--BkgName',type='string',help='Label for background sample',dest='bkgName',default='Diboson')
parser.add_option('-l', '--label',
                  type='string',
                  help='Label for the name in the plots',default="")
parser.add_option('-f', '--file-name',
                  dest = "file_name",
                  type='string',
                  help='File to write the Signal to Background ratio in', default="SigToBKg")
parser.add_option('-w', '--use-weights',
                  dest = "use_weights",
                  action="store_true",
                  help='Use weights when training', default=False)
parser.add_option("-x","--batch-size", dest="BatchSizeStart", help="Batch_size", type='int', default=1024)
parser.add_option("-e","--epochs", dest="epochs", help="Epochs", type='int', default=200)
parser.add_option("--patience", dest="patience", help="Patience", type='int', default=10000)
parser.add_option("--train-size", dest="train_size", help="train_size", type='float', default=0.7)

parser.add_option("--do-plots", action="store_true", dest="do_plots", help="do variable plots" , default=False)
parser.add_option("--lr", dest="LR", help="learning rate", type='float', default=0.001)
parser.add_option("--batch-size-end", dest="BatchSizeEnd", help="Batch_size_end", type='int', default=-1)
parser.add_option("--batch-size-step", dest="BatchSizeStep", help="Batch_size_step", type='int', default=-1)
parser.add_option("--load-model", dest="load_model", help="Path to folder with the model to load", type="string", default="")
# parser.add_option("--signal-isnt-one", action="store_false", dest="sigIsntOne", help="Use 1 as isSignal for Signal", default=False)

# parser.add_option("--plot_dir", dest="plot_dir", type='string', help='directory perfix to store plots', default="plots")
# parser.add_option("-f", "--files_dir", dest="files_dir", type='string', help='Files saved folder', default="Files")
# parser.add_option("-r", "--read", dest="Read", action="store_true", help="-r read full data sample, may need wait for long time", default=False)
# parser.add_option("-a", "--activation", dest="ACTIVATION", type='string', help="activation function", default='relu')
# parser.add_option("--do_batchnorm", dest="do_BN", action="store_false", help="Use batch normalizetion", default=True)
# parser.add_option("--dropout", dest='DropOut', type='float', help='Drop out', default=0.5)
# parser.add_option("-o", "--optimazer", dest="OPTIMAZER", help="optimazer used to minimaze loss function, See keras documentation", type='string', default='adam')
(options, args) = parser.parse_args()


def DeepNeuralNetwork(options):
    
    signalDataset = pd.read_pickle(options.sigPkl)
    bkgDataset = pd.read_pickle(options.bkgPkl)

    '''
    lim_inf = 400000 # if using this, discomment line 240 around the beginning of do-plots
    lim_sup = 800000
    signalDataset = signalDataset.loc[signalDataset['leading_jet_pt'] > lim_inf] # Only select rows with leading jet pT > lim_inf
    signalDataset = signalDataset.loc[signalDataset['leading_jet_pt'] < lim_sup] # Only select rows with leading jet pT < lim_sup
    bkgDataset = bkgDataset.loc[bkgDataset['leading_jet_pt'] > lim_inf] # Only select rows with leading jet pT > lim_inf
    bkgDataset = bkgDataset.loc[bkgDataset['leading_jet_pt'] < lim_sup] # Only select rows with leading jet pT < lim_sup
    
    # only for bkg vs bkg (comment sigPkl line in options)
    sig_and_bkg = pd.read_pickle(options.bkgPkl)
    sig_and_bkg = sig_and_bkg.sample(frac=1) #to shuffle the DataFrame
    rows, columns = sig_and_bkg.shape
    signalDataset = sig_and_bkg[:int(rows/2)]
    bkgDataset = sig_and_bkg[int(rows/2):]
    '''

    # ----------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------


    sum_sig = sum(signalDataset['weight'])
    sum_bkg = sum(bkgDataset['weight'])

    print "=================================================================="
    print "Weight in Signal {}".format(sum_sig)
    print "Weight in Background {}".format(sum_bkg)

    signalDataset.insert(0,'isSignal',1)
    bkgDataset.insert(0,'isSignal',0)

    print "=================================================================="
    sig_factor = 0.5 * (sum_sig + sum_bkg) / sum_sig
    bkg_factor = 0.5 * (sum_sig + sum_bkg) / sum_bkg
    signalDataset.insert(0,'weight_rescaled', sig_factor * signalDataset['weight'])
    bkgDataset.insert(0,'weight_rescaled', bkg_factor * bkgDataset['weight'])
    print "Signal rescale factor: {}".format(sig_factor)
    print "Background rescale factor: {}".format(bkg_factor)

    #==================================================================================

    # InputFeatures = ['j0_pt', 'j0_eta', 'j0_phi', 'j0_fch', 'j0_emfrac', 'j0_width', 'j0_sumpttrk', 'j0_n_tracks','met_tst_et', 'n_vx', 'DeltaPhi', 'HT', 'mass_eff']
    # InputFeatures = ['j0_pt', 'j0_eta', 'j0_phi', 'j0_width','met_tst_et', 'DeltaPhi', 'mass_eff']
    # InputFeatures = ['j0_width','met_tst_et', 'DeltaPhi', 'mass_eff']
    # InputFeatures = ['j0_width']
    # InputFeatures = ['j0_pt', 'j0_eta', 'j0_phi', 'j0_fch', 'j0_emfrac', 'j0_width', 'j0_sumpttrk', 'j0_n_tracks',
    #                      'met_tst_et', 'n_vx', 'DeltaPhi', 'j1_pt', 'j1_eta', 'j1_phi', 'j1_fch',
    #                      'j1_emfrac', 'j1_width', 'j1_sumpttrk', 'j1_n_tracks', 'j2_pt', 'j2_eta', 'j2_phi', 'j2_fch',
    #                      'j2_emfrac','j2_width', 'j2_sumpttrk', 'j2_n_tracks',  'j3_pt', 'j3_eta','j3_phi',
    #                      'j3_fch', 'j3_emfrac', 'j3_width', 'j3_sumpttrk','j3_n_tracks',  'invMass_j0_j1','invMass_j0_j2',
    #                      'invMass_j0_j3', 'invMass_j1_j2', 'invMass_j2_j3', 'DeltaR_j0_j1',  'DeltaR_j0_j2','DeltaR_j0_j3',
    #                      'DeltaR_j1_j2','DeltaR_j2_j3','min_invMass_jets','max_invMass_jets', 'min_DeltaR_jets',
    #                      'max_DeltaR_jets', 'DeltaEta_for_minDeltaR','DeltaPhi_for_minDeltaR', 'HT', 'mass_eff']
    # InputFeatures = ['n_el', 'n_mu', 'n_ph', 'met_tst_et', 'n_LargeJets',
    #                 'leading_jet_pt', 'leading_jet_eta', 'leading_jet_phi',
    #                 'leading_jet_m', 'leading_jet_tau21', 'leading_jet_D2',
    #                 'leading_jet_C2', 'leading_jet_nConstit']
    InputFeatures = ['met_tst_et','leading_jet_pt', 'leading_jet_eta', 'leading_jet_phi',
                    'leading_jet_m', 'leading_jet_tau21', 'leading_jet_D2',
                    'leading_jet_C2', 'leading_jet_nConstit']
    xLabel = ['Missing transverse energy (GeV)', 'Leading jet transverse momentum (GeV)', 'Leading jet rapidity', 'Leading jet phi',
              'Leading jet mass (GeV)','Leading jet tau21', 'Leading jet D2', 'Leading jet C2', 'Leading jet constituents']

    if 'DeltaPhi' in InputFeatures:
        signalDataset['DeltaPhi'] = np.fabs(signalDataset['DeltaPhi'])
        bkgDataset['DeltaPhi'] = np.fabs(bkgDataset['DeltaPhi'])
    if 'DeltaPhi_for_minDeltaR' in InputFeatures:
        signalDataset['DeltaPhi_for_minDeltaR'] = np.fabs(signalDataset['DeltaPhi_for_minDeltaR'])
        bkgDataset['DeltaPhi_for_minDeltaR'] = np.fabs(bkgDataset['DeltaPhi_for_minDeltaR'])

    #plots with same binning

    if options.do_plots:
        os.system('mkdir -p Distributions/' + options.label)
        for var in range(len(InputFeatures)):
            #adopt a common binning scheme for all channels
            bins = np.linspace(min(signalDataset[InputFeatures[var]]), max(signalDataset[InputFeatures[var]]) , 30)
            #bins = np.linspace(lim_inf, lim_sup, 30)
            if InputFeatures[var] == 'leading_jet_D2':
                bins = np.linspace(0, 4 , 30)
            
            if InputFeatures[var] == 'met_tst_et' or InputFeatures[var] == 'leading_jet_pt' or InputFeatures[var] == 'leading_jet_m':
                fig, ax = plt.subplots()
                plt.hist(signalDataset[InputFeatures[var]], weights=signalDataset['weight'], histtype='step', density=True, bins=bins, label=options.sigName, linewidth=2)
                plt.hist(bkgDataset[InputFeatures[var]], weights=bkgDataset['weight'], histtype='step', density=True, bins=bins, label=options.bkgName, linewidth=2)
            
                plt.xticks(ax.get_xticks(), ax.get_xticks()/1000)
                plt.xlim(0, max(signalDataset[InputFeatures[var]]))
                #plt.xlim(lim_inf, lim_sup)
            else:
                plt.hist(signalDataset[InputFeatures[var]], weights=signalDataset['weight'], histtype='step', density=True, bins=bins, label=options.sigName, linewidth=2)
                plt.hist(bkgDataset[InputFeatures[var]], weights=bkgDataset['weight'], histtype='step', density=True, bins=bins, label=options.bkgName, linewidth=2)
            
            plt.xlabel(xLabel[var])
            plt.yscale('log')
            if InputFeatures[var] == 'leading_jet_phi':
                plt.yscale('linear')
            plt.legend(loc='best')
            plt.savefig("Distributions/" + options.label + "/" + InputFeatures[var] + ".png")
            plt.savefig("Distributions/" + options.label + "/" + InputFeatures[var]  + ".eps")
            plt.show()
            plt.clf()
        
        #plots with different binning

        os.system('mkdir -p Distributions/' + options.label)
        bins = np.linspace(0, 1.5*10**6, 30)
        fig, ax = plt.subplots()
        plt.hist(signalDataset['met_tst_et'], weights=signalDataset['weight'], histtype='step', density=True, bins=bins, label=options.sigName, linewidth=2)
        plt.hist(bkgDataset['met_tst_et'], weights=bkgDataset['weight'], histtype='step', density=True, bins=bins, label=options.bkgName, linewidth=2)
        
        plt.xticks(ax.get_xticks(), ax.get_xticks()/1000)
        plt.xlim(0, 1.5*10**6)
        plt.xlabel('Missing transverse energy (GeV)')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig("Distributions/" + options.label + "/" + 'met_tst_et1' + ".png")
        plt.savefig("Distributions/" + options.label + "/" + 'met_tst_et1'  + ".eps")
        plt.show()
        plt.clf()

        bins = np.linspace(0, 2*10**5, 30)
        fig, ax = plt.subplots()
        plt.hist(signalDataset['leading_jet_m'], weights=signalDataset['weight'], histtype='step', density=True, bins=bins, label=options.sigName, linewidth=2)
        plt.hist(bkgDataset['leading_jet_m'], weights=bkgDataset['weight'], histtype='step', density=True, bins=bins, label=options.bkgName, linewidth=2)
        
        plt.xticks(ax.get_xticks(), ax.get_xticks()/1000)
        plt.xlim(0, 2*10**5)
        plt.xlabel('Leading jet mass (GeV)')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig("Distributions/" + options.label + "/" + 'leading_jet_m1' + ".png")
        plt.savefig("Distributions/" + options.label + "/" + 'leading_jet_m1'  + ".eps")
        plt.show()
        plt.clf()

    #==================================================================================

    signalList = [signalDataset] 
    backgroundList = [bkgDataset] 

    totalPD_sig = pd.concat(signalList,ignore_index=True)
    totalPD_bkg = pd.concat(backgroundList,ignore_index=True)

    print "=================================================================="
    print 'Signal events:',totalPD_sig.shape
    print 'Background events:',totalPD_bkg.shape

    totalPD_sig = Randomizing(totalPD_sig)
    totalPD_bkg = Randomizing(totalPD_bkg)

    signal_and_background = [totalPD_bkg,totalPD_sig]
    signal_and_background = pd.concat(signal_and_background,ignore_index=True, sort=False)   

    signal_and_background = Randomizing(signal_and_background)               
    print "Total events after shuffling: ",signal_and_background.shape
    print "=================================================================="

    # make the X array
    Xskim=signal_and_background[InputFeatures].values

    #make y vector
    y_tmpskim=signal_and_background['isSignal']

    # make the event weights vector
    wskim=signal_and_background['weight']
    wskim_rescaled=signal_and_background['weight_rescaled']

    if (options.load_model == ""):

        #Scale and format the inputs
        #scaler = StandardScaler()
        scaler = MinMaxScaler(feature_range = (0, 1))
        le = LabelEncoder()
        Xskim = scaler.fit_transform(Xskim)
        yskim = le.fit_transform(y_tmpskim)

        # Define the train, test sets
        # We want the wskim with and without rescaling
        # We use the same random_seed so Xskim, yskim are the same in both lines
        Xskim_train, Xskim_test, yskim_train, yskim_test, wskim_train, wskim_test = train_test_split(Xskim, yskim, wskim, train_size=options.train_size, test_size=(1-options.train_size),random_state=123)
        Xskim_train2, Xskim_test2, yskim_train2, yskim_test2, wskim_rescaled_train, wskim_rescaled_test = train_test_split(Xskim, yskim, wskim_rescaled, train_size=options.train_size, test_size=(1-options.train_size),random_state=123)

    #==================================================================================

        print "Train set shape: ",Xskim_train.shape
        print "Test set shape: ",Xskim_test.shape
        print "Total data shape: ",Xskim.shape
        print "=================================================================="


        #Define the number of variables, the nodes per layer and the number of hidden layers
        n_dim=Xskim_train.shape[1]
        # n_nodes = int((n_dim+1)/2)
        n_nodes = 50
        n_depth = 3 #usually 1

        #Build the model aka the neural network
        model=BuildDNN(n_dim,n_nodes,n_depth)

        #Compile the model while defining the loss function, the optimizer algorithm and the metrics to use
        adam = Adam(lr=options.LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='mean_squared_error',optimizer=adam,metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

        #Define the stopping condition and where the model will be stored. Use the above one in case you want EarlyStopping, the below one otherwise
        callbacks = [EarlyStopping(verbose=True, patience=options.patience, monitor='loss'), ModelCheckpoint('Models/' + options.label + 'model.h5', monitor='loss', verbose=True, save_best_only=True, mode='max')]
        # callbacks = ModelCheckpoint('Models/' + options.label + 'model.h5', monitor='loss', verbose=True, save_best_only=True, mode='max')

    #==================================================================================

        #We train the model
        EPOCHS = options.epochs
        print "Start training with a BATCH SIZE of {}".format(options.batch_size)

        if (options.use_weights):
            modelMetricsHistoryskim = model.fit(Xskim_train, yskim_train, sample_weight = wskim_rescaled_train.values,epochs=EPOCHS,batch_size=options.batch_size,validation_split=0.0,callbacks=callbacks, verbose=0)
        else:
            modelMetricsHistoryskim = model.fit(Xskim_train, yskim_train, epochs=EPOCHS,batch_size=options.batch_size,validation_split=0.0,callbacks=callbacks, verbose=0)

        perf = model.evaluate(Xskim_test, yskim_test, batch_size=options.batch_size)
        saveModel(model,scaler,le,options.label)

    else: # Load model
        model, scaler, le = loadModel(options.load_model)
        Xskim = scaler.transform(Xskim)
        yskim = le.transform(y_tmpskim)

        Xskim_train, Xskim_test, yskim_train, yskim_test, wskim_train, wskim_test = train_test_split(Xskim, yskim, wskim, train_size=options.train_size, test_size=(1-options.train_size),random_state=123)
        Xskim_train2, Xskim_test2, yskim_train2, yskim_test2, wskim_rescaled_train, wskim_rescaled_test = train_test_split(Xskim, yskim, wskim_rescaled, train_size=options.train_size, test_size=(1-options.train_size),random_state=123)

        perf = model.evaluate(Xskim_test, yskim_test, batch_size=options.batch_size)

    #==================================================================================

    doROCCurve(options, model, Xskim_test, yskim_test)

    S_to_B = doDiscriminant(options,model,Xskim_test,yskim_test,wskim_test,Xskim_train,yskim_train,wskim_train)

    ###########################################################################

    print "=================================================================="
    print "END"

    return {'S_to_B':S_to_B,'perf':perf}



def doROCCurve(options, model, Xskim_test, yskim_test):
    # Generates output predictions for the input samples.
    yhatskim_test = model.predict(Xskim_test, batch_size=options.batch_size)
    # Get 'Receiver operating characteristic' (ROC)
    FPR, TPR, thresholds = roc_curve(yskim_test, yhatskim_test)

    # Compute Area Under the Curve (AUC) from prediction scores
    roc_auc_ZnunuWtaunuhighwith0  = auc(FPR, TPR)
    print "ROC AUC: %0.5f" % roc_auc_ZnunuWtaunuhighwith0

    plt.plot(FPR, TPR, color='darkorange',  lw=2, label='Full curve (area = %0.4f)' % roc_auc_ZnunuWtaunuhighwith0)
    plt.plot([0, 0], [1, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curves for Signal vs Background')
    #plt.plot([0.038], [0.45], marker='*', color='red',markersize=5, label="Cut-based",linestyle="None")
    #plt.plot([0.038, 0.038], [0,1], color='red', lw=1, linestyle='--') # same background rejection point
    plt.legend(loc="lower right")

    os.system('mkdir -p Plots/' + options.label)
    plt.savefig("Plots/" + options.label + "/ROC_Curve.png")
    plt.savefig("Plots/" + options.label + "/ROC_Curve.eps")
    plt.clf()

    print "================================================="
    print "ROC Curve plot stored in Plots/" + options.label + "/ROC_Curve.png" 

    return roc_auc_ZnunuWtaunuhighwith0


def doDiscriminant(options,model,Xskim_test,yskim_test,wskim_test,Xskim_train,yskim_train,wskim_train):
    n_bins = 100 # Granularity to compute the T_cut
    n_bins_plt = 100 # Binning to use in the plot

    ############## Compute Discriminant in test subsample

    signal_test = Xskim_test[yskim_test == 1]
    bkg_test = Xskim_test[yskim_test == 0]
    signal_wskim_test = wskim_test[yskim_test == 1]
    bkg_wskim_test = wskim_test[yskim_test == 0]
    signal_wskim_test_reshaped = np.array(signal_wskim_test).reshape(signal_wskim_test.shape[0],1)
    bkg_wskim_test_reshaped = np.array(bkg_wskim_test).reshape(bkg_wskim_test.shape[0],1)

    yhat_test_sig = model.predict(signal_test, batch_size=options.batch_size)
    yhat_test_bkg = model.predict(bkg_test, batch_size=options.batch_size)

    ############## Compute Discriminant in train subsample

    signal_train = Xskim_train[yskim_train == 1]
    bkg_train = Xskim_train[yskim_train == 0]
    signal_wskim_train = wskim_train[yskim_train == 1]
    bkg_wskim_train = wskim_train[yskim_train == 0]
    signal_wskim_train_reshaped = np.array(signal_wskim_train).reshape(signal_wskim_train.shape[0],1)
    bkg_wskim_train_reshaped = np.array(bkg_wskim_train).reshape(bkg_wskim_train.shape[0],1)

    yhat_train_sig = model.predict(signal_train, batch_size=options.batch_size)
    yhat_train_bkg = model.predict(bkg_train, batch_size=options.batch_size)

    ################ Compute T_Cut using the TEST subsample

    n_sig, bins = np.histogram(yhat_test_sig, weights=signal_wskim_test_reshaped, bins = n_bins, density = True, range=(0,1))
    n_bkg, bins = np.histogram(yhat_test_bkg, weights=bkg_wskim_test_reshaped, bins = n_bins, density = True, range=(0,1))

    cut_info = getTcut(n_sig, n_bkg, bins)
    T_cut = cut_info['T_cut']

    ################# Compute Signal and Background using the TEST subsamples

    S = sum(signal_wskim_test_reshaped[yhat_test_sig > T_cut])
    B = sum(bkg_wskim_test_reshaped[yhat_test_bkg > T_cut])

    ################# Make the plots

    plt.hist(yhat_train_bkg,  weights=bkg_wskim_train, histtype='step', density=True, label=options.bkgName + " (train)", linestyle="--", linewidth=0, range=(0,1),bins=n_bins_plt,fill=True, edgecolor=None, facecolor='r', alpha=0.5)
    plt.hist(yhat_train_sig, weights=signal_wskim_train, histtype='step', density=True, label=options.sigName + " (train)", linestyle="--", linewidth=0, range=(0,1),bins=n_bins_plt,fill=True, edgecolor=None, facecolor='g', alpha=0.5)
    plt.hist(yhat_test_bkg,  weights=bkg_wskim_test, histtype='step', density=True, label=options.bkgName + " (test)", edgecolor='darkred', linewidth=2, range=(0,1),bins=n_bins_plt, facecolor=None)
    plt.hist(yhat_test_sig, weights=signal_wskim_test, histtype='step', density=True, label=options.sigName + " (test)", edgecolor='darkgreen', linewidth=2, range=(0,1),bins=n_bins_plt, facecolor=None)
    plt.axvline(x=T_cut,color='black',label=r'$T_{cut}$' + ' = {:.2f}'.format(T_cut), linestyle='-.')
    plt.plot(bins[:-1], 1 - cut_info['alpha'], label=r"$1 - \alpha$")
    plt.plot(bins[:-1], cut_info['beta'], label=r"$\beta$")
    plt.plot(bins[:-1], cut_info['signal_to_noise_ratio'], label=r"$(1 - \alpha)/\sqrt{\beta}$")
    plt.xlim(0.,1)
    plt.yscale('log')
    plt.legend(loc='best')
    plt.xlabel('DNN score')
    S_to_B = -1
    S_to_sqrtB = -1
    if B > 0:
        S_to_B = S/B
        S_to_sqrtB = S/m.sqrt(B)
    plt.title(S_to_sqrtB)

    os.system('mkdir -p Discriminant/' + options.label)
    plt.savefig("Discriminant/" + options.label + "/Discriminant.png")
    plt.savefig("Discriminant/" + options.label + "/Discriminant.eps")
    plt.clf()

    print "================================================="
    print "Discriminant plot stored in Discriminant/" + options.label + "/Discriminant.png" 

    #########################################
    
    # S_to_B = -1
    # S_to_sqrtB = -1
    # if B > 0:
    #     S_to_B = S/B
    #     S_to_sqrtB = S/m.sqrt(B)

    print "================================================="
    print "The S/B is {}".format(S_to_B)
    print "The S/sqrt(B) is {}".format(S_to_sqrtB)
    print "S is {} and B is {}".format(S,B)

    return S_to_B



def main(options):

    os.system('mkdir -p Plots')
    os.system('mkdir -p Models')
    os.system('mkdir -p Datasets')
    os.system('mkdir -p Discriminant')
    os.system('mkdir -p Distributions')
    

    f = open(options.file_name + ".dat","a+") # Append to file

    if (options.label == ""):
        options.label = options.file_name

    S_to_Bs = []
    loss = []
    acc = []
    print("Hello")

    bs_range = [options.BatchSizeStart]
    if options.BatchSizeEnd != -1 and options.BatchSizeStep != -1:
        bs_range = range(options.BatchSizeStart,options.BatchSizeEnd,options.BatchSizeStep)
        bs_range.append(options.BatchSizeEnd)

    # bs_range = range(10,5000,50)
    # bs_range += range(5000,10000,1000)
    # bs_range.append(10000)

    label = options.label
    if (options.label == ""):
        label = options.file_name

    for bs in bs_range:
        options.batch_size = bs
        options.label = label + "_BS{}".format(bs)
        out = DeepNeuralNetwork(options)
        S_to_B = out['S_to_B']
        perf = out['perf']
        loss.append(perf[0])
        acc.append(perf[1])
        S_to_Bs.append(S_to_B)
        f.write("{}\t{}\t{}\t{}\n".format(bs,S_to_B,perf[0],perf[1]))

    f.close()

if __name__ == '__main__':
    main(options)

