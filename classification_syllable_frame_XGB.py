# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:05:23 2020

@author: arnou
"""

import os, csv, errno
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost.sklearn import XGBClassifier
#import xgboost as xgb
#from sklearn.model_selection import TimeSeriesSplit

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise    
         
def storFile(data,fileName):
    data = list(map(lambda x:[x],data))
    with open(fileName,'w',newline ='') as f:
        mywrite = csv.writer(f)
        for i in data:
            mywrite.writerow(i)

# Classification using syllable-based features
fs = 44100
tune_parmater = 1

#segmentation_type = ['manual', 'energy', 'auto_energy', 'spectral', 'renyi']
segmentation_type = ['renyi'] 
nSeg = len(segmentation_type)
for iSeg in range(nSeg):
        
    select_segment = segmentation_type[iSeg]
    
    percent_array = [0.8]
    nPer = len(percent_array)
    for iPer in range(nPer):
        
        select_percent = percent_array[iPer]
        
        win_time_array = np.array([ 0.02, 0.05, 0.1, 0.2 ])* fs
        #win_time_array = np.array([ 0.01, 0.005 ])* fs
        
        nTime = len(win_time_array)
        for iTime in range(nTime):
            
            select_win_time = win_time_array[iTime]
            
            win_step_array = [0.2, 0.5, 0.8]
            nStep = len(win_step_array)
            for iStep in range(nStep):
                
                select_win_step = win_step_array[iStep]
                
                #print(select_segment)    
                #print(select_percent)
                print([select_win_time, select_win_step])
                
                # combine csv data of train data
                train_path = 'D:/Project/Segmentaion_revise1/feat_frame/' + select_segment + '/training_' + str(select_percent) + \
                 '/win_len_' + str(int(select_win_time)) + '_win_over_' + str(select_win_step)
                     
                train_list = os.listdir(train_path)   
                nTrain = len(train_list)
                train_data_list = []
                for iTrain in np.arange(0, nTrain, 2):
          
                    #print(iTrain)
                    train_csv_path = train_path + '/' + train_list[iTrain]                           
                    trainDataFrame = pd.read_csv(train_csv_path, low_memory=False, header=None)                   
                    trainData = trainDataFrame.values # change DataFrame to matrix
                    
                    train_data_list.append(trainData)
                        
                train_data_mat = np.concatenate(train_data_list)
                train_feat = train_data_mat[:,0:-1]
                train_feat[np.isnan(train_feat)] = 0
                train_label = np.ravel(train_data_mat[:,-1:])
                    
                # combine csv data of test data
                test_path = 'D:/Project/Segmentaion_revise1/feat_frame/' + select_segment + '/testing_' + str(select_percent) + \
                 '/win_len_' + str(int(select_win_time)) + '_win_over_' + str(select_win_step)
                
                test_list = os.listdir(test_path)   
                nTest = len(test_list)
                test_data_list = []
                for iTest in np.arange(0, nTest, 2):
         
                    #print(iTest)
                    test_csv_path = test_path + '/' + test_list[iTest]                           
                    testDataFrame = pd.read_csv(test_csv_path, low_memory=False, header=None)
                    testData = testDataFrame.values
                    
                    test_data_list.append(testData)
                
                testData_mat = np.concatenate(test_data_list)
                test_feature = testData_mat[:,0:-1]
                test_feature[np.isnan(test_feature)] = 0
                test_label = np.ravel(testData_mat[:,-1:])
                
                # obtain label location
                test_loc_list = []
                for jTest in np.arange(1, nTest, 2):
                    
                    #print(jTest)
                    test_csv_path = test_path + '/' + test_list[jTest]                           
                    testDataFrame = pd.read_csv(test_csv_path, low_memory=False, header=None)
                    testData = testDataFrame.values
                    
                    test_loc_list.append(testData)
                    
                testLoc_mat = np.concatenate(test_loc_list)

                # normalization
                scaler = preprocessing.StandardScaler().fit(train_feat)
                train_feat = scaler.transform(train_feat)             
                test_feature = scaler.transform(test_feature)    
                
                # split train into training and validatiaon                
                paramGrid1 = {'learning_rate': [0.01, 0.02, 0.03, 0.06, 0.1, 0.2, 0.3],
                         'gamma': [0.1, 0.2, 0.5, 1, 1.5, 2, 10]  
                        }               
                model1 = XGBClassifier(learning_rate=0.1, 
                                      n_estimators=500, 
                                      gamma=0, 
                                      objective='multi:softmax', 
                                      num_class=24,
                                      )                              
                gsearch1 = GridSearchCV(model1, paramGrid1, verbose=1, n_jobs=4, iid=False, cv=5)              
                cv_res1 = gsearch1.fit(train_feat, train_label)
                best_param1 = cv_res1.best_params_
                
                best_lr = best_param1.get('gamma')
                bset_gamma = best_param1.get('learning_rate')
                
                print('First step done')
                
                paramGrid2 = {'max_depth': [2,4,6,8,10],
                        'min_child_weight': [2,4,6,8,10]
                        }
                model2 = XGBClassifier(learning_rate=best_lr, 
                                      n_estimators=500, 
                                      gamma=bset_gamma, 
                                      objective='multi:softmax', 
                                      max_depth=4,
                                      min_child_weight=6,
                                      num_class=24,
                                      )       
                gsearch2 = GridSearchCV(model2, paramGrid2, verbose=1, n_jobs=4, iid=False, cv=5)  
                cv_res2 = gsearch2.fit(train_feat, train_label)
                best_param2 = cv_res2.best_params_
                
                best_depth = best_param2.get('max_depth')
                best_weight = best_param2.get('min_child_weight')
                
                print('Second step done')
                
                #-----------------#
                #dtrain = xgb.DMatrix(train_feat,train_label)
                #dtest  = xgb.DMatrix(test_feature)                
                #cv_res= xgb.cv(param_1,dtrain,num_boost_round=2000,early_stopping_rounds=10,nfold=5,show_stdv=True)
                #-----------------#
                #cv_res.shape[0]为最佳迭代次数
                #bst = xgb.train(v,dtrain,num_boost_round=cv_res.shape[0])
                

                out_label_xgb = cv_res2.best_estimator_.predict(test_feature)
                
                out_label_rf = out_label_xgb
                
                #out_label_rf = np.asarray(out_label_rf, 'float32')
                #testLabel = np.asarray(testLabel, 'float32')
                       
                accuracy_value = accuracy_score(test_label, out_label_rf)
                balance_accuracy_value = balanced_accuracy_score(test_label, out_label_rf)
                macro_f1_score = f1_score(test_label, out_label_rf, average='macro')                
                weight_macro_f1_score = f1_score(test_label, out_label_rf, average='weighted')

                cm_data = confusion_matrix(test_label, out_label_rf)

                comb_evaluation = np.array([accuracy_value, balance_accuracy_value, macro_f1_score, 
                                            weight_macro_f1_score])  
    
                comb_evaluation_list = comb_evaluation.tolist()  
                
                #print(comb_evaluation)

                # combine location and label                
                out_info = np.hstack([testLoc_mat, test_label.reshape([len(test_label), 1]), 
                                      out_label_rf.reshape([len(out_label_rf), 1])])
                
                out_info_list = out_info.tolist()  

                # save path
                save_folder = './out_label_frame_xgb_500/' +  select_segment + '/training_' + str(select_percent) + \
                             '/win_len_' + str(int(select_win_time)) + '_win_over_' + str(select_win_step)
                                                          
                make_sure_path_exists(save_folder)
                save_path =save_folder + '/outlabel.csv'

                # save label information for evaluation
                with open(save_path,'w',newline='') as out:
                    csv_out=csv.writer(out)
                    csv_out.writerow(['start', 'stop', 'ground_truth', 'predict'])
                    for row in out_info_list:
                        csv_out.writerow(row)   
                
                # save performance and optimize paramters in random forest
                save_path_performance = save_folder + '/performance.csv'
                storFile(comb_evaluation, save_path_performance)
                
                print(comb_evaluation)
                
                #comb_evaluation = 0











