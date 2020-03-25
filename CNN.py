#!/bin/python

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


from optparse import OptionParser
import numpy as np
import os
import random
import math as m
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

#######################################
parser = OptionParser()

parser.add_option('-p', '--path',type='string',help='Path to folder where the folder images are', dest = "parentFolder", default="/nfs/pic.es/user/s/sgonzalez/scratch2/DL/BernatJetImages/")
parser.add_option('-s', '--signalImages',type='string',help='Signal path to images folder', dest = "sigFolder", default="imgFiles_Znunu_sig") #actually you should put either invH for sig and Znunu for bkg
parser.add_option('-b', '--bkgImages',type='string',help='Bkg path to images folder', dest = "bkgFolder", default="imgFiles_Znunu_bkg")
parser.add_option("--train-size", dest="train_size", help="train_size", type='float', default=0.7)
parser.add_option("-x","--batch-size", dest="batch_size", help="Batch_size", type='int', default=128)
parser.add_option("-e","--epochs", dest="epochs", help="Epochs", type='int', default=12)
parser.add_option("-m","--maxFiles", dest="max_files", help="Maximum number of files you want to use", type='int', default=5000)
parser.add_option('-l', '--label', type='string', help='Label for the name in the plots',default="")
parser.add_option('-c', '--channels', type='string', help='Comma separated channel names', action='callback', callback=get_comma_separated_args)
parser.add_option("--do-normalise-per-image", action="store_true", dest="norm_per_image", help="Normalise each image instad of all images at the same time" , default=False)

(options, args) = parser.parse_args()



def doCNN(options):

    print("=============================================================================================================================")
    print("Reading signal data from: {}".format(options.sigFolder))
    print("Reading bkg data from: {}".format(options.bkgFolder))
    print("Channels used are {}".format(options.channels))
    print("=============================================================================================================================")

    (x_train, y_train, w_train), (x_test, y_test, w_test) = load_data(options.parentFolder + options.sigFolder, options.parentFolder + options.bkgFolder, options.train_size, options.max_files, options.channels, options.norm_per_image)
    print("========================================================================")
    print("x_train shape: {}".format(x_train.shape))
    print("x_test shape: {}".format(x_test.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("y_test shape: {}".format(y_test.shape))
    print("========================================================================")

    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    
    if K.image_data_format() == 'channels_first':
        input_shape = (len(options.channels), img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, len(options.channels))

    if not options.norm_per_image:
        scale_x(x_train, x_test)

    model = buildCNN(input_shape)
    model.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    print("Start model fit")
    model.fit(x_train, y_train, sample_weight = w_train, batch_size=options.batch_size, epochs=options.epochs, verbose=1, validation_data=(x_test,y_test))

    print("========================================================================")
    print("Start model evaluation")
    score = model.evaluate(x_test, y_test)
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))

    doROCCurve(model, x_test, y_test, options.batch_size)
    doDiscriminantCNN(options, model, x_train, y_train, x_test, y_test, options.batch_size)

def doROCCurve(model, x_test, y_test, batch_size):
    yhat_test = model.predict(x_test, batch_size=batch_size)
    fpr, tpr, thresholds = roc_curve(y_test, yhat_test)

    roc_auc = auc(fpr, tpr)
    print("ROC AUC: {}".format(roc_auc))

    plt.plot(fpr, tpr, color='darkorange',  lw=2, label='Full curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 0], [1, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curves for Signal vs Background')
    plt.legend(loc="lower right")
    plt.savefig("CNN/Plots/ROC_AUC" + options.label + ".png")
    plt.savefig("CNN/Plots/ROC_AUC" + options.label + ".eps")
    plt.clf()

    return roc_auc

def doDiscriminantCNN(options, model, x_train, y_train, x_test, y_test, batch_size):
    n_bins = 100 # Granularity to compute the T_cut
    n_bins_plt = 100 # Binning to use in the plot

    ############## Compute Discriminant in test subsample

    signalx_test = x_test[y_test == 1]
    bkgx_test = x_test[y_test == 0]
    yhat_test_sig = model.predict(signalx_test, batch_size=options.batch_size)
    yhat_test_bkg = model.predict(bkgx_test, batch_size=options.batch_size)

    ############## Compute Discriminant in train subsample

    signalx_train = x_train[y_train == 1]
    bkgx_train = x_train[y_train == 0]
    yhat_train_sig = model.predict(signalx_train, batch_size=options.batch_size)
    yhat_train_bkg = model.predict(bkgx_train, batch_size=options.batch_size)

    ################ Compute T_Cut using the TEST subsample

    n_sig, bins = np.histogram(yhat_test_sig, bins = n_bins, density = True, range=(0,1))
    n_bkg, bins = np.histogram(yhat_test_bkg, bins = n_bins, density = True, range=(0,1))

    cut_info = getTcut(n_sig, n_bkg, bins)
    T_cut = cut_info['T_cut']

    ################# Compute Signal and Background using the TEST subsamples

    ################# Make the plots

    plt.hist(yhat_test_bkg, histtype='step', density=True, label="Background (test)", edgecolor='darkred', linewidth=2, range=(0,1),bins=n_bins_plt, facecolor=None)
    plt.hist(yhat_test_sig, histtype='step', density=True, label="Signal (test)", edgecolor='darkgreen', linewidth=2, range=(0,1),bins=n_bins_plt, facecolor=None)
    plt.hist(yhat_train_bkg, histtype='step', density=True, label="Background (train)", linestyle="--", linewidth=0, range=(0,1),bins=n_bins_plt, fill=True, edgecolor=None, facecolor='r', alpha=0.5)
    plt.hist(yhat_train_sig, histtype='step', density=True, label="Signal (train)", linestyle="--", linewidth=0, range=(0,1),bins=n_bins_plt, fill=True, edgecolor=None, facecolor='g', alpha=0.5)
    plt.axvline(x=T_cut,color='black',label=r'$T_{cut}$' + ' = {:.2f}'.format(T_cut), linestyle='-.')
    plt.plot(bins[:-1], 1 - cut_info['alpha'], label=r"$1 - \alpha$")
    plt.plot(bins[:-1], cut_info['beta'], label=r"$\beta$")
    plt.plot(bins[:-1], cut_info['signal_to_noise_ratio'], label=r"$(1 - \alpha)/\sqrt{\beta}$")
    plt.xlim(0.,1)
    plt.yscale('log')
    plt.legend(loc='best')
    plt.xlabel('CNN score')
    plt.savefig("CNN/Discriminant/Discriminant" + options.label + ".png")
    plt.savefig("CNN/Discriminant/Discriminant" + options.label + ".eps")
    plt.clf()

def buildCNN(input_shape):
    model = Sequential()
    print("input shape is {}".format(input_shape))
    model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def scale_x(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    max_val = np.amax(np.concatenate((x_train, x_test)))
    return (x_train / max_val, x_test / max_val)


def load_data(sigFolder, bkgFolder, train_size, max_files, channels, norm_per_image):
    N_sig, N_bkg = len(os.listdir(sigFolder + "/pt")), len(os.listdir(bkgFolder + "/pt"))
    (sig_train, sig_y_train, sig_w_train) , (sig_test, sig_y_test, sig_w_test) = load_folder(sigFolder, train_size, channels, 1, min(N_sig,N_bkg,max_files), norm_per_image)
    (bkg_train, bkg_y_train, bkg_w_train) , (bkg_test, bkg_y_test, bkg_w_test) = load_folder(bkgFolder, train_size, channels, 0, min(N_sig,N_bkg,max_files), norm_per_image)
    (x_train, y_train, w_train) = shuffle_images(sig_train, bkg_train, sig_y_train, bkg_y_train, sig_w_train, bkg_w_train)
    print("sig_train shape {}".format(sig_train.shape))
    print("sig_test shape {}".format(sig_test.shape))
    print("bkg_train shape {}".format(bkg_train.shape))
    print("bkg_test shape {}".format(bkg_test.shape))
    (x_test, y_test, w_test) = shuffle_images(sig_test, bkg_test, sig_y_test, bkg_y_test, sig_w_test, bkg_w_test)
    return (x_train, y_train, w_train), (x_test, y_test, w_test)

def shuffle_images(sig, bkg, sig_y, bkg_y, sig_w, bkg_w):
    x = np.concatenate((sig, bkg))
    y = np.concatenate((sig_y, bkg_y))
    w = np.concatenate((sig_w, bkg_w))
    z = list(zip(x,y,w))
    random.shuffle(z)
    x, y, w = zip(*z)
    return (np.array(x), np.array(y), np.array(w))
    

def load_folder(folder, train_size, channels, yLabel, N_max, norm_per_image): #in case you want to use glob see CNN_copy2.py
    file_indices = os.listdir(folder + "/pt")
    file_indices = [file.replace('img_pt_','').replace('.txt','') for file in file_indices]
    random.shuffle(file_indices)
    file_indices = file_indices[0:N_max]
    N = float(len(file_indices))
    train, test, w_train, w_test = [], [], [], []
    counter = 0
    for index in file_indices:
        images = []
        for channel in channels:
            image = np.loadtxt(folder + "/" + channel + "/img_" + channel + "_" + index + ".txt")
            if norm_per_image:
                image.astype('float32')
                image = image / np.amax(image)
            images.append(image)
        images = np.stack(images, axis=2)
        
        weight = float(np.loadtxt(folder + "/weight/weight_" + index + ".txt"))
        # weight = 1

        if (len(train)/N < train_size):
            train.append(images)
            w_train.append(weight)
        else:
            test.append(images)
            w_test.append(weight)
        counter += 1
        if counter%1000 == 0:
            print("{}/{}".format(counter, N))
    print("There are {} files in {}. {} were used for the training set".format(N, folder + "/pt", len(train)))
    train = np.array(train)
    test = np.array(test)
    y_train = yLabel * np.ones(len(train))
    y_test = yLabel * np.ones(len(test))
    w_train = np.array(w_train)
    w_test = np.array(w_test)
    return (train, y_train, w_train) , (test, y_test, w_test)

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

def main(options):
    if options.channels == None: 
        options.channels=['pt']

    doCNN(options)


if __name__ == '__main__':
    main(options)
