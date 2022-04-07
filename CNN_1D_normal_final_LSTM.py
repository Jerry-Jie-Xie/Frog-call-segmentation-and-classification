# -*- coding: utf-8 -*-
"""
Created on Fri May  3 07:16:45 2019

@author: Jie
"""

from __future__ import print_function
import numpy as np
import os
#import soundfile as sf
import matplotlib.pyplot as plt
#import math
import pandas as pd
#from operator import itemgetter
#from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling1D, Conv1D, CuDNNLSTM
#from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras
import keras.backend as K
from itertools import product
import errno
from keras.layers.normalization import BatchNormalization
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#===========================#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#===========================#
np.random.seed(1337)
#------------------------------------------------------------------------------#
def build_CNN_model(shape_a, shape_b):

    #===========================#
    # build 1D CNN
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=32, strides=2, input_shape=(shape_a, shape_b)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.4))
    
    model.add(Conv1D(filters=32, kernel_size=16, strides=2))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.4))

    model.add(Conv1D(filters=64, kernel_size=8, strides=2))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.4))
    
    model.add(CuDNNLSTM(128, return_sequences=True))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()

    return model

                                                                        
#------------------------------------------------------------------------------#
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
#------------------------------------------------------------------------------#
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
#------------------------------------------------------------------------------#    
num_classes = 24
batch_size = 64
epochs = 200
fs = 44100
#===========================#
label_type_array = ['weak', 'mid', 'strong' ]
#label_type_array = [ 'strong' ]

nType = len(label_type_array)
for iType in range(nType):
    
    label_type = label_type_array[iType]

    baseFolder = r'D:\Project\Segmentaion_revise1'
    baseFolder = baseFolder.replace('\\', '/')
    #train_percent = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    train_percent = np.array([0.8])
    #===========================#
    nPerc = len(train_percent)
    for iPerc in range(nPerc):
        select_percent = train_percent[iPerc]
        
        #win_size_array = np.array([0.02, 0.05, 0.1, 0.2]) # seconds
        win_size_array = np.array([0.5]) # seconds
        
        nWin = len(win_size_array)
        for iWin in range(nWin):
            select_win_size = win_size_array[iWin]        
            win_over_array = np.array([0.2, 0.5, 0.8])
            #win_over_array = np.array([0.8])
            
            nOver = len(win_over_array)
            for iOver in range(nOver):
                select_win_over = win_over_array[iOver]        
                
                #===========================#
                train_path = baseFolder + '/raw_data_sliding/feat_all_percent_' + str(select_percent) + \
                '_winsize_' + str(int(select_win_size*fs)) + '_winover_' + str(select_win_over) + \
                '/training_feat_sliding_window.csv'
                
                test_path = baseFolder + '/raw_data_sliding/feat_all_percent_' + str(select_percent) + \
                '_winsize_' + str(int(select_win_size*fs)) + '_winover_' + str(select_win_over) + \
                '/testing_feat_sliding_window.csv'
                
                #=====# read csv files=======#
                trainData = pd.read_csv(train_path, low_memory=False, header=None)
                testData = pd.read_csv(test_path, low_memory=False, header=None)
            
                trainData = trainData.values
                testData = testData.values
                #===========================#
                # change data to feature and label
                n_sample, n_feat = trainData.shape
    
                trainFeature = trainData[:,0:n_feat-3]
                                            
                # split train into training and validation
                training_data, validation_data = train_test_split(trainData, test_size=0.2)
                
                # strong, mid, weak
                #===========================#                            
                training_feat = training_data[:,0:n_feat-3]               
                validation_feat = validation_data[:,0:n_feat-3]
                testFeature = testData[:,0:n_feat-3]
                
                if label_type == 'weak':
                    trainLabel = np.ravel(trainData[:,-3:-2])
                    training_label_org = np.ravel(training_data[:,-3:-2])
                    validation_label_org = np.ravel(validation_data[:,-3:-2])
                    testLabel = np.ravel(testData[:,-3:-2])
                elif label_type == 'mid':
                    trainLabel = np.ravel(trainData[:,-2:-1])
                    training_label_org = np.ravel(training_data[:,-2:-1])
                    validation_label_org = np.ravel(validation_data[:,-2:-1])
                    testLabel = np.ravel(testData[:,-2:-1])            
                elif label_type == 'strong':
                    trainLabel = np.ravel(trainData[:,-1:])
                    training_label_org = np.ravel(training_data[:,-1:])
                    validation_label_org = np.ravel(validation_data[:,-1:])
                    testLabel = np.ravel(testData[:,-1:])
                
                #===========================#
                std_scaler = StandardScaler(copy=False).fit(training_feat)
                training_feat = std_scaler.transform(training_feat)
                validation_feat = std_scaler.transform(validation_feat)
                testFeature = std_scaler.transform(testFeature)

                #training_feat = MinMaxScaler().fit_transform(training_feat)
                #validation_feat = MinMaxScaler().fit_transform(validation_feat)            
                #testFeature = MinMaxScaler().fit_transform(testFeature)
                
                #===========================#
                # reshape train data
                shape_a = int(select_win_size*fs)
                shape_b = 1
                
                # reshape training data
                training_feat_final = training_feat.reshape(training_feat.shape[0], shape_a, shape_b).astype('float32')            
                training_label = keras.utils.to_categorical(np.ravel(training_label_org), num_classes)    
                       
                # reshape validation data
                validation_feat_final = validation_feat.reshape(validation_feat.shape[0], shape_a, shape_b).astype('float32')            
                validation_label = keras.utils.to_categorical(np.ravel(validation_label_org), num_classes)    
         
                # reshape test data
                testFeature_final = testFeature.reshape(testFeature.shape[0], shape_a, shape_b).astype('float32')            
                test_label = keras.utils.to_categorical(np.ravel(testLabel), num_classes)    
                                   
                #===========================#
                # build 1D CNN
                model = build_CNN_model(shape_a, shape_b)
                                                                        
                # initiate RMSprop optimizer
                #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
                opt = keras.optimizers.Adam(lr=0.0001)
                              
                #w_array = np.ones((num_classes,num_classes))
                ##w_array[np.diag_indices(num_classes)] = weight
                #w_array[0,0] = 0.5
                #loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)                
                ## Let's train the model using RMSprop
                #model.compile(loss=loss,
                #  optimizer=opt,
                #  metrics=['accuracy'])
                
                model.compile(loss='categorical_crossentropy',
                              optimizer=opt,
                              metrics=['accuracy'])
                
                '''
                saves the model weights after each epoch if the validation loss decreased
                '''
                #checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
                # checkpoint
                filepath="./tmp/weights.best.org.left.hdf5"
                
                mc = ModelCheckpoint(filepath, monitor='val_acc', 
                                             verbose=1, save_best_only=True, mode='max')
                
                es = EarlyStopping(monitor='val_loss', 
                           mode='min', 
                           verbose=1, 
                           patience=20)
                                
                #print('Not using data augmentation.')
                callbacks_list = [mc, es]
                
                
                # Fit the model
                history = model.fit(training_feat_final, training_label,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(validation_feat_final, validation_label),
                          callbacks=callbacks_list,
                          verbose=0)                
                                
                ###############################################################################################
            
                # load the best model and do the classification
                print('start loading best weights')
                # build 1D CNN
                model = build_CNN_model(shape_a, shape_b)

                # load the model
                print("Created model and loaded weights from file")
                model.load_weights(filepath)
            
                # Compile model (required to make predictions)
                model.compile(loss='categorical_crossentropy', 
                              optimizer=opt, 
                              metrics=['accuracy'])
                
                predict_label_test = model.predict_classes(testFeature_final)        
                out_label_prob = model.predict_proba(testFeature_final)
    
                
                temp_cm = confusion_matrix(testLabel, predict_label_test)
                
                accuracy_result = accuracy_score(testLabel, predict_label_test)
                fscore_result = f1_score(testLabel, predict_label_test, average='weighted')    
                temp_f1_score_class = f1_score(testLabel, predict_label_test, average=None)
                kapppa_score = cohen_kappa_score(testLabel, predict_label_test)
                
                
                # sliding window based evaluation vs syllable based evaluation                
                save_folder_final = baseFolder + '/python_LSTM/' + label_type + '_normal_final/result_all_percent_' + str(select_percent) + '_winsize_' + str(int(select_win_size*fs)) + '_winover_' + str(select_win_over)
                make_sure_path_exists(save_folder_final)
                
                np.savetxt(save_folder_final + '/accuracy_fscore_kappa.csv', [accuracy_result, fscore_result, kapppa_score], delimiter=',');
                np.savetxt(save_folder_final + '/temp_f1_score_class.csv', temp_f1_score_class, delimiter=',');
                np.savetxt(save_folder_final + '/confusion_matrix.csv', temp_cm, delimiter=',');
                np.savetxt(save_folder_final + '/predict_label.csv', predict_label_test, delimiter=',')
                
                # plot the accuracy
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.title('Mode Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig(save_folder_final + '/Mode_accuracy.png')                
                plt.show()
                                
                # plot the loss
                plt.figure()
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Mode loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig(save_folder_final + '/Mode_loss.png')
                plt.show()
                                
                
                
                
                
  
