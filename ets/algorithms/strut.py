import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from pyts.transformation import WEASEL
from pyts.multivariate.transformation import WEASELMUSE
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
from sktime.datasets import load_from_arff_to_dataframe
from sktime.datatypes._panel._convert import (from_3d_numpy_to_nested,from_nested_to_3d_numpy,)
from scipy.io import arff               
from ets.algorithms.utils import accuracy, harmonic_mean
from ets.algorithms.utils import topy
from progressbar import ProgressBar, AnimatedMarker, Bar, AdaptiveETA, Percentage, ProgressBar, SimpleProgress
import math
from xcm_class import XCMModel
class STRUT():

        def __init__(self, timestamps, ts_length, variate, optimize, tsc_method, n_splits, class_imbalance, dataset):
            self.timestamps = timestamps
            self.tsc_method = tsc_method #MINIROCKET, MINIROCKET_FAV, WEASEL, WEASEL_FAV
            self.optimize = optimize
            self.variate = variate
            self.n_splits = n_splits
            self.ts_length = ts_length
            self.class_imbalance = class_imbalance
            self.dataset = dataset

        def train_test_prefix(self, X_training, X_test, Y_training, Y_test):

            training_time, test_time = 0.0, 0.0

            if self.tsc_method == "MINIROCKET" or self.tsc_method == "MINIROCKET_FAV" : # or self.tsc_method =="XCM":

                if not (self.tsc_method == "XCM"):
                    if(self.variate > 1):  # if multivariate

                        transformation = MiniRocketMultivariate() 
                    
                    else: # if univariate 
                    
                        transformation = MiniRocket()     
                else: ######## UPDATES HERE
                    if(self.variate > 1):  # if multivariate
                        #self.xcm.k= self.variate
                        transformation = XCMModel()
                    else:
                        transformation = XCMModel()   
                    


                transformation.fit(X_training)

                if self.class_imbalance:
                    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), class_weight = 'balanced')
                else:
                    
                    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))

            elif self.tsc_method == "WEASEL" or self.tsc_method == "WEASEL_FAV":
            
                if(self.variate > 1  ): # if multivariate
                    
                    transformation = WEASELMUSE(word_size=3, n_bins=2, window_sizes =  [ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                                            chi2_threshold=15, sparse=False, norm_mean = False, norm_std = False, drop_sum=False)
                    

                else:
                    transformation = WEASEL(word_size=3, n_bins = 2, window_sizes =  [ 0.4, 0.5,  0.6, 0.7, 0.8, 0.9], norm_mean = False, norm_std = False)
                
                transformation.fit(X_training, Y_training)

                if self.class_imbalance:
                    classifier = LogisticRegression(max_iter = 10000, class_weight = 'balanced')
                else:
                    classifier = LogisticRegression(max_iter = 10000)
            
            else:
                print("Unsupported method")
                return

            # -- transform training ------------------------------------------------

            time_a = time.perf_counter()
            X_training_transform = transformation.transform(X_training)
            time_b = time.perf_counter()
            training_time += time_b - time_a

            # -- transform test ----------------------------------------------------

            time_a = time.perf_counter()
            X_test_transform = transformation.transform(X_test)
            time_b = time.perf_counter()
            test_time += time_b - time_a

            # -- training ----------------------------------------------------------

            time_a = time.perf_counter()
            classifier.fit(X_training_transform, Y_training)
            time_b = time.perf_counter()
            training_time += time_b - time_a

            # -- test --------------------------------------------------------------

            time_a = time.perf_counter()
            Y_pred = classifier.predict(X_test_transform)
            time_b = time.perf_counter()
            test_time += time_b - time_a
            
            return (Y_pred, accuracy_score(Y_test, Y_pred), f1_score(Y_test, Y_pred, average = 'weighted'), training_time, test_time)

        def trunc_data(self, data, tp):
            data = from_nested_to_3d_numpy(data)[:,:,:tp]
            return  from_3d_numpy_to_nested(data)

        def nan_handler(self, df, tsc_method):
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            df = df.reset_index(drop =True)
            if tsc_method == 'MINIROCKET' or tsc_method == 'MINIROCKET_FAV':
                X_3d = from_nested_to_3d_numpy(df)
                size, dim, tpoint = X_3d.shape
                for i in range(size): # for each row
                    for j in range(dim): # for each dimension
                        prev = 0
                        next = 0
                        for k in range(tpoint): 
                            if math.isnan(X_3d[i][j][k]): 
                                for l in range(k+1, tpoint): # for each NaN
                                    if not math.isnan(X_3d[i][j][l]): # find the next not NaN
                                        next = X_3d[i][j][l]
                                        break
                                average = (next + prev) / 2
                                prev = average
                                X_3d[i][j][k] = average 
                            else:
                                prev = X_3d[i][j][k]
                return from_3d_numpy_to_nested(X_3d)
            else:
                for index_r, row in df.iterrows():  # for each row
                    prev = 0
                    next = 0
                    row = row.reset_index(drop=True)
                    for index_c, item in row.iteritems():  # for each NaN
                        if math.isnan(item):
                            for index_a, real in row[index_c:].iteritems():  # find the next not NaN
                                if not math.isnan(real):
                                    next = real
                                    break
                            average = (next + prev) / 2
                            prev = average
                            df.iloc[int(index_r), index_c+1] = average
                        else:
                            prev = item
            return df
        
        def xcm_strut(self, train_data, test_data): 
            if isinstance(train_data, str):  # ARFF files provided as input (no CV)
                X_TRAIN, Y_TRAIN = load_from_arff_to_dataframe(train_data)
                X_TEST, Y_TEST = load_from_arff_to_dataframe(test_data)
            elif isinstance(train_data, tuple):  # Perform CV
                X_TRAIN, Y_TRAIN = train_data
                X_TRAIN = self.nan_handler(X_TRAIN, self.tsc_method)
                X_TEST, Y_TEST = test_data
                X_TEST = self.nan_handler(X_TEST, self.tsc_method)
            else:  # Single CSV file provided as input (no CV)
                X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
                    train_data, test_data, test_size=0.2, stratify=test_data, random_state=0
                )
                X_TRAIN = self.nan_handler(X_TRAIN, self.tsc_method)
                X_TEST = self.nan_handler(X_TEST, self.tsc_method)

            label_encoder = LabelEncoder()
            Y_TRAIN = label_encoder.fit_transform(Y_TRAIN)
            Y_TEST = label_encoder.transform(Y_TEST)

            # Stratified Shuffle Split for cross-validation
            kfold = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=0)

            training_time = np.zeros(self.n_splits)
            test_time = np.zeros(self.n_splits)
            accuracies = []
            f1_scores = []
            earlinesses = np.arange(1, self.ts_length + 1) / self.ts_length

            for fold, (train_index, test_index) in enumerate(kfold.split(X_TRAIN, Y_TRAIN)):
                X_train_cv, X_test_cv = X_TRAIN.iloc[train_index], X_TRAIN.iloc[test_index]
                Y_train_cv, Y_test_cv = Y_TRAIN[train_index], Y_TRAIN[test_index]

                fold_accuracies = []
                fold_f1_scores = []

                for trunc_length in range(9, self.ts_length + 1):  # Minimum length for XCM
                    X_train_trunc = self.trunc_data(X_train_cv, trunc_length)
                    X_test_trunc = self.trunc_data(X_test_cv, trunc_length)

                    # Train and test using XCM
                    result = self.train_test_xcm(X_train_trunc, X_test_trunc, Y_train_cv, Y_test_cv)

                    fold_accuracies.append(result[1])  # Append accuracy
                    fold_f1_scores.append(result[2])  # Append F1-score
                    training_time[fold] += result[3]
                    test_time[fold] += result[4]

                accuracies.append(fold_accuracies)
                f1_scores.append(fold_f1_scores)

            # Aggregate metrics
            accuracies = np.array(accuracies)
            f1_scores = np.array(f1_scores)
            accuracies_mean, accuracies_std = accuracies.mean(axis=0), accuracies.std(axis=0)
            f1_scores_mean, f1_scores_std = f1_scores.mean(axis=0), f1_scores.std(axis=0)

            harmonic_means = 2 * accuracies_mean * (1 - earlinesses) / (accuracies_mean + (1 - earlinesses))
            harmonic_means_mean, harmonic_means_std = harmonic_means.mean(), harmonic_means.std()

            # Identify optimal truncation point based on selected metric
            if self.optimize == 0:  # Optimize for accuracy
                best_index = np.argmax(accuracies_mean)
            elif self.optimize == 1:  # Optimize for F1-score
                best_index = np.argmax(f1_scores_mean)
            elif self.optimize == 2:  # Optimize for harmonic mean
                best_index = np.argmax(harmonic_means)

            # Train and evaluate the final model
            best_trunc_length = best_index + 9
            X_train_best = self.trunc_data(X_TRAIN, best_trunc_length)
            X_test_best = self.trunc_data(X_TEST, best_trunc_length)
            final_result = self.train_test_xcm(X_train_best, X_test_best, Y_TRAIN, Y_TEST)

            # Save metrics and plots
            self.save_results("xcm", accuracies, f1_scores, harmonic_means, accuracies_mean, f1_scores_mean, harmonic_means_mean)

            return final_result


        def minirocket_strut(self, train_data, test_data): # -> None:
            
            if isinstance(train_data, str): # arrf files provided as input (no cv)
                X_TRAIN, Y_TRAIN = load_from_arff_to_dataframe(train_data)
                X_TEST, Y_TEST = load_from_arff_to_dataframe(test_data)
                X = X_TRAIN
                Y = Y_TRAIN
            elif isinstance(train_data, tuple): # perform cv 
                X = train_data[0]
                X = self.nan_handler(X, self.tsc_method)
                Y = train_data[1]
                X_TRAIN = X
                Y_TRAIN = Y
                X_TEST = test_data[0]
                X_TEST = self.nan_handler(X_TEST, self.tsc_method)
                Y_TEST = test_data[1]
            else: # single csv file provided as input (no cv)
                X = train_data
                X = self.nan_handler(X, self.tsc_method)
                Y = test_data
                X_TRAIN , X_TEST, Y_TRAIN,  Y_TEST  = train_test_split(X, Y, test_size=0.2,  stratify = Y, random_state=0)
                X = X_TRAIN 
                Y = Y_TRAIN

            label_encoder = LabelEncoder()
            Y = label_encoder.fit_transform(Y)

            kfold = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=0) 

            start = 9 #  Minirocket can't process input time series with lengths less than 9

            training_time = np.zeros(self.n_splits)
            test_time = np.zeros(self.n_splits)
            accuracies = np.zeros((self.n_splits,self.ts_length+1-start))
            f1_scores = np.zeros((self.n_splits,self.ts_length+1-start))
            earlinesses = np.array([float(x)/self.ts_length for x in range(start, self.ts_length+1)])

            fold = 0

            for train_index, test_index in kfold.split(X,Y):
    
                X_training, X_testing = X.iloc[train_index], X.iloc[test_index]
                Y_training, Y_testing = Y[train_index], Y[test_index]
                
                pbar = ProgressBar(widgets=['Split: ' + str(fold+1)+ " | ", SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
                    ' ', AdaptiveETA()])
                
                for i in pbar(range(start,self.ts_length+1)): 

                    #truncate nested dataframe
                    X_training_trunc = self.trunc_data(X_training,i) 
                    X_testing_trunc = self.trunc_data(X_testing,i) 
                    
                    result = self.train_test_prefix(X_training_trunc, X_testing_trunc, Y_training, Y_testing)
                    #returns (predictions, accuracy, f1 score, training time, test time)
                    
                    accuracies[fold][i-start] = result[1]
                    f1_scores[fold][i-start] =  result[2]
                    training_time[fold] += result[3]
                    test_time[fold] += result[4]
                    
                if self.optimize == 0: # optimize accuracy
                    best_accuracy = accuracies[fold].max()
                    best_accuracy_timepoint = np.argmax(accuracies[fold])
                    X_training_trunc = self.trunc_data(X_TRAIN,best_accuracy_timepoint+start) 
                    X_testing_trunc = self.trunc_data(X_TEST,best_accuracy_timepoint+start) 
                    res = self.train_test_prefix(X_training_trunc, X_testing_trunc, Y_TRAIN, Y_TEST)
                    harmonic_mean =  (2 * (1 - earlinesses[best_accuracy_timepoint]) * best_accuracy) / ((1 - earlinesses[best_accuracy_timepoint]) + best_accuracy)
                if self.optimize == 1: # optimize f1 score
                    best_f1_score = f1_scores[fold].max()
                    best_f1_score_timepoint = np.argmax(f1_scores[fold])
                    X_training_trunc = self.trunc_data(X_TRAIN,best_f1_score_timepoint+start) 
                    X_testing_trunc = self.trunc_data(X_TEST,best_f1_score_timepoint+start) 
                    res = self.train_test_prefix(X_training_trunc, X_testing_trunc, Y_TRAIN, Y_TEST)
                    harmonic_mean =  (2 * (1 - earlinesses[best_f1_score_timepoint]) * best_f1_score) / ((1 - earlinesses[best_f1_score_timepoint]) + best_f1_score)
                if self.optimize == 2: # optimize harnmonic mean
                    harmonic_means =  (2 * (1 - earlinesses) * accuracies[fold]) / ((1 - earlinesses) + accuracies[fold])
                    best_harmonic_mean  = harmonic_means.max()
                    best_harmonic_mean_timepoint = np.argmax(harmonic_means)
                    X_training_trunc = self.trunc_data(X_TRAIN,best_harmonic_mean_timepoint+start) 
                    X_testing_trunc = self.trunc_data(X_TEST,best_harmonic_mean_timepoint+start) 
                    res = self.train_test_prefix(X_training_trunc, X_testing_trunc, Y_TRAIN, Y_TEST)
                    
                fold += 1

            # calculate mean and std of metric scores per time point
            # ci =  confidence intervals

            accuracies_mean = accuracies.mean(axis=0)
            accuracies_std = accuracies.std(axis=0)
            accuracies_ci = accuracies_std

            f1_scores_mean = f1_scores.mean(axis=0)
            f1_scores_std = f1_scores.std(axis=0)
            f1_scores_ci = f1_scores_std 

            harmonic_means = np.divide(np.multiply(2 * (1 - earlinesses), accuracies), np.add((1 - earlinesses), accuracies))
            #harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)

            harmonic_means_mean = harmonic_means.mean(axis=0)
            harmonic_means_std = harmonic_means.std(axis=0)
            harmonic_means_ci = harmonic_means_std # 0.1 * harmonic_means_std / harmonic_means_mean

            best_accuracy = accuracies_mean.max()
            best_accuracy_timepoint = np.argmax(accuracies_mean)

            best_f1_score = f1_scores_mean.max()
            best_f1_score_timepoint = np.argmax(f1_scores_mean)

            best_harmonic_mean = harmonic_means_mean.max()
            best_harmonic_mean_timepoint = np.argmax(harmonic_means_mean)
            
            os.makedirs('results', exist_ok=True) # create a directory to store results
            os.makedirs('results/'+self.dataset+'_metric_scores', exist_ok=True) # create a subdirectory to store metric scores
            os.makedirs('results/'+self.dataset+'_plots', exist_ok=True) # create a subdirectory to store olots

            # save metric scores in a 2d ndarray where each each row corresponds to a split 
            # and each column to a truncation time point starting from start = 9

            with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_minirocket_strut_accuracy.txt','a') as f: #what if the user wants to delete the existing file before running again the experiment?
                np.savetxt(f, accuracies,  delimiter=',')

            with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_minirocket_strut_f1_score.txt','a') as f: 
                np.savetxt(f, f1_scores, delimiter=',')

            with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_minirocket_strut_harmonic_mean.txt','a') as f: 
                np.savetxt(f, harmonic_means, delimiter=',')

            # Create and store metric score plots
            timepoints = np.arange(start,self.ts_length+1) 

            #plot accuracy
            full_time_accuracy = np.repeat(accuracies_mean[-1], timepoints.shape[0]) # full tsc accuracy for comparison
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.plot(timepoints, full_time_accuracy , '--', color = 'blue', label = "Minirocket Full-Time")
            ax.plot(timepoints, accuracies_mean, color = 'blue', label = "Minirocket STRUT") 
            ax.fill_between(timepoints,(accuracies_mean - accuracies_ci), (accuracies_mean + accuracies_ci), color='blue', alpha=0.4)
            ax.plot([best_accuracy_timepoint+start], [best_accuracy], 'D', markersize = 10,  label = 'Best Accuracy')
            plt.xlabel("Truncation Timepoint")
            plt.ylabel("Accuracy")
            plt.ylim(0.0,1.1)
            plt.legend() 
            plt.savefig('results/'+self.dataset+'_plots/minirocket_strut_accuracy.pdf', format='pdf', dpi=320, bbox_inches='tight')  

            #plot f1 score
            full_time_f1 = np.repeat(f1_scores_mean[-1], timepoints.shape[0]) # full tsc f1 score for comparison
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.plot(timepoints, full_time_f1 , '--', color = 'blue', label = "Minirocket Full-Time")
            ax.plot(timepoints, f1_scores_mean, color = 'blue', label = "Minirocket STRUT" ) 
            ax.fill_between(timepoints,(f1_scores_mean - f1_scores_ci), (f1_scores_mean + f1_scores_ci), color='blue', alpha=0.4)
            ax.plot([best_f1_score_timepoint+start], [best_f1_score], 'D', markersize = 10, label = 'Best F1-score')
            plt.xlabel("Truncation Time-point")
            plt.ylabel("F1-score")
            plt.ylim(0.0,1.1)
            plt.legend() 
            plt.savefig('results/'+self.dataset+'_plots/minirocket_strut_f1_score.pdf', format='pdf', dpi=320, bbox_inches='tight')  
            
            #plot harmonic mean
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.plot(timepoints , harmonic_means_mean , color='blue', label = "Minirocket STRUT" ) 
            ax.plot([best_harmonic_mean_timepoint+start], [best_harmonic_mean], 'D', markersize = 10,  label = 'Best Harmonic Mean')
            ax.fill_between(timepoints,(harmonic_means_mean - harmonic_means_ci), (harmonic_means_mean + harmonic_means_ci), color='blue', alpha=0.4)
            plt.ylabel("Harmonic Mean")
            plt.xlabel("Truncation Time-point")
            plt.ylim(0.0,1.1)
            plt.legend() 
            plt.savefig('results/'+self.dataset+'_plots/minirocket_strut_harmonic_mean.pdf', format='pdf', dpi=320, bbox_inches='tight')  

            #Test for best earliness
            
            Y_TRAIN = label_encoder.fit_transform(Y_TRAIN)
            Y_TEST = label_encoder.fit_transform(Y_TEST)

            if self.optimize == 0: # accuracy
                X_training_trunc = self.trunc_data(X_TRAIN,best_accuracy_timepoint+start) 
                X_testing_trunc = self.trunc_data(X_TEST,best_accuracy_timepoint+start) 
                result = self.train_test_prefix(X_training_trunc, X_testing_trunc, Y_TRAIN, Y_TEST)
                preds = [(best_accuracy_timepoint+start,result[0][x]) for x in range(result[0].size)]
                earliness = earlinesses[best_accuracy_timepoint]
            elif self.optimize == 1: #f1 score
                X_training_trunc = self.trunc_data(X_TRAIN,best_f1_score_timepoint+start) 
                X_testing_trunc = self.trunc_data(X_TEST,best_f1_score_timepoint+start)
                result = self.train_test_prefix(X_training_trunc, X_testing_trunc, Y_TRAIN, Y_TEST)
                preds = [(best_f1_score_timepoint+start,result[0][x]) for x in range(result[0].size)]
                earliness = earlinesses[best_f1_score_timepoint]
            elif self.optimize == 2: #harmonic mean
                X_training_trunc = self.trunc_data(X_TRAIN,best_harmonic_mean_timepoint+start) 
                X_testing_trunc = self.trunc_data(X_TEST,best_harmonic_mean_timepoint+start)
                result = self.train_test_prefix(X_training_trunc, X_testing_trunc, Y_TRAIN, Y_TEST)
                preds = [(best_harmonic_mean_timepoint+start,result[0][x]) for x in range(result[0].size)]
                earliness = earlinesses[best_harmonic_mean_timepoint]
            
            testing_time = result[4]

            return preds, training_time.sum(), testing_time, earliness           

        def data_reform(self, X): # reform input data from nested dataframe to ndarray for Weasel
            
            if (self.variate > 1):

                a = list()
                for i in range(X.values.size): #for each sample
                    
                    b = list()
                    for j in range(X.values[i][0].size): #for each dim
                        b.append(list(X.values[i][0][j]))
                    a.append(b)

                X = np.asarray(a)

            else:
                X = X.values

            
            return X

        def weasel_strut(self, train_data, test_data):
             
            if isinstance(train_data, str): # arrf files provided as input (no cv)     
                train_data = pd.DataFrame(arff.loadarff(train_data)[0])
                test_data = pd.DataFrame(arff.loadarff(test_data)[0])
                data = train_data.append(test_data) 
                X = data.iloc[:,:-1] 
                X =  self.data_reform(X)
                Y = data.iloc[: ,-1].values.ravel() 
                X = np.nan_to_num(X)
                label_encoder = LabelEncoder()
                Y = label_encoder.fit_transform(Y)
                X_TRAIN = self.data_reform(train_data.iloc[:,:-1])
                Y_TRAIN = train_data.iloc[: ,-1].values.ravel()
                X_TEST  = self.data_reform(test_data.iloc[:,:-1])
                Y_TEST  = test_data.iloc[: ,-1].values.ravel()
                Y_TRAIN = label_encoder.fit_transform(Y_TRAIN)
                Y_TEST = label_encoder.fit_transform(Y_TEST)
            elif isinstance(train_data, tuple): # perform cv
                X = train_data[0]
                X = np.nan_to_num(X) 
                Y = train_data[1]
                label_encoder = LabelEncoder()
                Y = label_encoder.fit_transform(Y)
                X_TRAIN = X
                Y_TRAIN = Y
                X_TEST = test_data[0]
                X_TEST = np.nan_to_num(X_TEST)
                Y_TEST = test_data[1]
                Y_TEST = label_encoder.fit_transform(Y_TEST)
            else: # single csv file provided as input (no cv) 
                X = train_data
                Y = test_data
                Y = Y.ravel()
                X = np.nan_to_num(X)
                label_encoder = LabelEncoder()
                Y = label_encoder.fit_transform(Y)
                X_TRAIN , X_TEST, Y_TRAIN,  Y_TEST  = train_test_split(X, Y, test_size=0.2,  stratify = Y, random_state=0)
                Y_TRAIN = label_encoder.fit_transform(Y_TRAIN)
                Y_TEST = label_encoder.fit_transform(Y_TEST)
            

            kfold = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=0)  

            start = 11 #  Weasel can't process input time series with lengths less than 11
            
            training_time = np.zeros(self.n_splits)
            test_time = np.zeros(self.n_splits)
            accuracies = np.zeros((self.n_splits,self.ts_length+1-start))
            f1_scores = np.zeros((self.n_splits,self.ts_length+1-start))
            earlinesses = np.array([float(x)/self.ts_length for x in range(start, self.ts_length+1)])


            fold = 0
           
            for train_index, test_index in kfold.split(X,Y):
            
                X_training, X_testing = X[train_index], X[test_index]
                Y_training, Y_testing = Y[train_index], Y[test_index]

                pbar = ProgressBar(widgets=['Split: ' + str(fold+1)+ " | ", SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
                    ' ', AdaptiveETA()])
                for i in pbar(range(start,self.ts_length+1)): 
                    
                    if self.variate > 1:
                        result = self.train_test_prefix(X_training[:,:,:i], X_testing[:,:,:i], Y_training, Y_testing)
                    else:
                        result = self.train_test_prefix(X_training[:,:i], X_testing[:,:i], Y_training, Y_testing)
                    
                    #returns (predictions, accuracy, f1 score, training time, test time)

                    accuracies[fold][i-start] = result[1]
                    f1_scores[fold][i-start] =  result[2]
                    training_time[fold] += result[3]
                    test_time[fold] += result[4]

                if self.optimize == 0:
                    best_accuracy = accuracies[fold].max()
                    best_accuracy_timepoint = np.argmax(accuracies[fold])
                    if self.variate > 1:
                        res = self.train_test_prefix(X_TRAIN[:,:,:best_accuracy_timepoint+start], X_TEST[:,:,:best_accuracy_timepoint+start], Y_TRAIN, Y_TEST)
                    else:
                        res = self.train_test_prefix(X_TRAIN[:,:best_accuracy_timepoint+start], X_TEST[:,:best_accuracy_timepoint+start], Y_TRAIN, Y_TEST)
                    harmonic_mean =  (2 * (1 - earlinesses[best_accuracy_timepoint]) * best_accuracy) / ((1 - earlinesses[best_accuracy_timepoint]) + best_accuracy)
                if self.optimize == 1:
                    best_f1_score = f1_scores[fold].max()
                    best_f1_score_timepoint = np.argmax(f1_scores[fold])
                    if self.variate > 1:
                        res = self.train_test_prefix(X_TRAIN[:,:,:best_f1_score_timepoint+start], X_TEST[:,:,:best_f1_score_timepoint+start], Y_TRAIN, Y_TEST)
                    else:
                        res = self.train_test_prefix(X_TRAIN[:,:best_f1_score_timepoint+start], X_TEST[:,:best_f1_score_timepoint+start], Y_TRAIN, Y_TEST)
                    harmonic_mean =  (2 * (1 - earlinesses[best_f1_score_timepoint]) * best_f1_score) / ((1 - earlinesses[best_f1_score_timepoint]) + best_f1_score)
                if self.optimize == 2:
                    harmonic_means =  (2 * (1 - earlinesses) * accuracies[fold]) / ((1 - earlinesses) + accuracies[fold])
                    best_harmonic_mean  = harmonic_means.max()
                    best_harmonic_mean_timepoint = np.argmax(harmonic_means)
                    if self.variate > 1:
                        res = self.train_test_prefix(X_TRAIN[:,:,:best_harmonic_mean_timepoint+start], X_TEST[:,:,:best_harmonic_mean_timepoint+start], Y_TRAIN, Y_TEST)
                    else:
                        res = self.train_test_prefix(X_TRAIN[:,:best_harmonic_mean_timepoint+start], X_TEST[:,:best_harmonic_mean_timepoint+start], Y_TRAIN, Y_TEST)
                     
                fold += 1


            # calculate mean and std of metric scores per time point
            # ci =  confidence intervals

            accuracies_mean = accuracies.mean(axis=0)
            accuracies_std = accuracies.std(axis=0)
            accuracies_ci = accuracies_std

            f1_scores_mean = f1_scores.mean(axis=0)
            f1_scores_std = f1_scores.std(axis=0)
            f1_scores_ci = f1_scores_std

            harmonic_means = np.divide(np.multiply(2 * (1 - earlinesses), accuracies), np.add((1 - earlinesses), accuracies))
            
            #harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)

            harmonic_means_mean = harmonic_means.mean(axis=0)
            harmonic_means_std = harmonic_means.std(axis=0)
            harmonic_means_ci = harmonic_means_std

            best_accuracy = accuracies_mean.max()
            best_accuracy_timepoint = np.argmax(accuracies_mean)

            best_f1_score = f1_scores_mean.max()
            best_f1_score_timepoint = np.argmax(f1_scores_mean)

            best_harmonic_mean = harmonic_means_mean.max()
            best_harmonic_mean_timepoint = np.argmax(harmonic_means_mean)

            os.makedirs('results', exist_ok=True) # create a directory to store results
            os.makedirs('results/'+self.dataset+'_metric_scores', exist_ok=True) # create a subdirectory to store metric scores
            os.makedirs('results/'+self.dataset+'_plots', exist_ok=True) # create a subdirectory to store plots

            # save metric scores in a 2d ndarray where each each row corresponds to a split 
            # and each column to a truncation time point starting from start = 11

            with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_weasel_strut_accuracy.txt','a') as f: #what if the user wants to delete the existing file before running again the experiment?
                np.savetxt(f, accuracies, delimiter=',')

            with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_weasel_strut_f1_score.txt','a') as f: 
                np.savetxt(f, f1_scores, delimiter=',')

            with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_weasel_strut_harmonic_mean.txt','a') as f: 
                np.savetxt(f, harmonic_means, delimiter=',')

            # Create and store metric score plots
            timepoints = np.arange(start,self.ts_length+1) 
            
            #plot accuracy
            full_time_accuracy = np.repeat(accuracies_mean[-1], timepoints.shape[0] )
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80) # full tsc accuracy for comparison
            ax.plot(timepoints, full_time_accuracy , '--', color = 'blue', label = "Weasel Full-Time") 
            ax.plot(timepoints, accuracies_mean, color = 'blue', label = "Weasel STRUT") 
            ax.fill_between(timepoints,(accuracies_mean - accuracies_ci), (accuracies_mean + accuracies_ci), color='blue', alpha=0.4)
            ax.plot([best_accuracy_timepoint+start], [best_accuracy], 'D', markersize = 10,  label = 'Best Accuracy')
            plt.xlabel("Truncation Timepoint")
            plt.ylabel("Accuracy")
            plt.ylim(0.0,1.1)
            plt.legend() 
            plt.savefig('results/'+self.dataset+'_plots/'+self.dataset+'_weasel_strut_accuracy.pdf', format='pdf', dpi=320, bbox_inches='tight')  
            
            #plot f1 score
            full_time_f1 = np.repeat(f1_scores_mean[-1], timepoints.shape[0] ) # full tsc f1 score for comparison
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.plot(timepoints, full_time_f1 , '--', color = 'blue', label = "Weasel Full-Time")
            ax.plot(timepoints, f1_scores_mean, color = 'blue', label = "Weasel STRUT" ) 
            ax.fill_between(timepoints,(f1_scores_mean - f1_scores_ci), (f1_scores_mean + f1_scores_ci), color='blue', alpha=0.4)
            ax.plot([best_f1_score_timepoint+start], [best_f1_score], 'D', markersize = 10, label = 'Best F1-score')
            plt.xlabel("Truncation Time-point")
            plt.ylabel("F1-score")
            plt.ylim(0.0,1.1)
            plt.legend() 
            plt.savefig('results/'+self.dataset+'_plots/'+self.dataset+'_weasel_strut_f1_score.pdf', format='pdf', dpi=320, bbox_inches='tight')  
            
            #plot harmonic mean
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.plot(timepoints , harmonic_means_mean , color='blue', label = "Weasel STRUT" ) 
            ax.plot([best_harmonic_mean_timepoint+start], [best_harmonic_mean], 'D', markersize = 10,  label = 'Best Harmonic Mean')
            ax.fill_between(timepoints,(harmonic_means_mean - harmonic_means_ci), (harmonic_means_mean + harmonic_means_ci), color='blue', alpha=0.4)
            plt.ylabel("Harmonic Mean")
            plt.xlabel("Truncation Time-point")
            plt.ylim(0.0,1.1)
            plt.legend() 
            plt.savefig('results/'+self.dataset+'_plots/'+self.dataset+'_weasel_strut_harmonic_mean.pdf', format='pdf', dpi=320, bbox_inches='tight')  

            #Test for best earliness

            if self.optimize == 0: #accuracy
                if self.variate > 1:
                    result = self.train_test_prefix(X_TRAIN[:,:,:best_accuracy_timepoint+start], X_TEST[:,:,:best_accuracy_timepoint+start], Y_TRAIN, Y_TEST)
                else:
                    result = self.train_test_prefix(X_TRAIN[:,:best_accuracy_timepoint+start], X_TEST[:,:best_accuracy_timepoint+start], Y_TRAIN, Y_TEST)
                preds = [(best_accuracy_timepoint+start,result[0][x]) for x in range(result[0].size)]
                earliness = earlinesses[best_accuracy_timepoint]
            elif self.optimize == 1: #f1 score
                if self.variate > 1:
                    result = self.train_test_prefix(X_TRAIN[:,:,:best_f1_score_timepoint+start], X_TEST[:,:,:best_f1_score_timepoint+start], Y_TRAIN, Y_TEST)
                else:
                    result = self.train_test_prefix(X_TRAIN[:,:best_f1_score_timepoint+start], X_TEST[:,:best_f1_score_timepoint+start], Y_TRAIN, Y_TEST)
                preds = [(best_f1_score_timepoint+start,result[0][x]) for x in range(result[0].size)]
                earliness = earlinesses[best_f1_score_timepoint]
            elif self.optimize == 2: #harmonic mean
                if self.variate > 1:
                    result = self.train_test_prefix(X_TRAIN[:,:,:best_harmonic_mean_timepoint+start], X_TEST[:,:,:best_harmonic_mean_timepoint+start], Y_TRAIN, Y_TEST)
                else:
                    result = self.train_test_prefix(X_TRAIN[:,:best_harmonic_mean_timepoint+start], X_TEST[:,:best_harmonic_mean_timepoint+start], Y_TRAIN, Y_TEST)
                preds = [(best_harmonic_mean_timepoint+start,result[0][x]) for x in range(result[0].size)]
                earliness = earlinesses[best_harmonic_mean_timepoint]
            
            testing_time = result[4]
            
            return preds, training_time.sum(), testing_time, earliness

        def minirocket_strut_fav(self, train_data, test_data): 
                
            if isinstance(train_data, str): # arrf files provided as input (no cv)
                X_TRAIN, Y_TRAIN = load_from_arff_to_dataframe(train_data)
                X_TEST, Y_TEST = load_from_arff_to_dataframe(test_data)
                X = X_TRAIN.append(X_TEST)
                X = nan_handler(X)
                Y = np.append(Y_TRAIN, Y_TEST)
            elif isinstance(train_data, tuple): # perform cv
                X = train_data[0]
                X = self.nan_handler(X, self.tsc_method)
                Y = train_data[1]
                X_TRAIN = X
                Y_TRAIN = Y
                X_TEST = test_data[0]
                X_TEST = self.nan_handler(X_TEST, self.tsc_method)
                Y_TEST = test_data[1]
            else:  # single csv file provided as input (no cv) 
                X = train_data
                X = self.nan_handler(X, self.tsc_method)
                Y = test_data
                X_TRAIN , X_TEST, Y_TRAIN,  Y_TEST  = train_test_split(X, Y, test_size=0.2,  stratify = Y, random_state=0)

            label_encoder = LabelEncoder() 
            Y = label_encoder.fit_transform(Y)

            kfold = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=0)

            start = 9 #  Minirocket can't process input time series with lengths less than 9

            training_time = 0.0
            best_timepoints = []

            # save metric scores in a 2d ndarray where each each row corresponds to a split 
            # and each column to a evaluated time point 
            # the first row of stores the evluated time points

            os.makedirs('results', exist_ok=True) # create a directory to store results
            os.makedirs('results/'+self.dataset+'_metric_scores', exist_ok=True) # create a subdirectory to store metric scores
            os.makedirs('results/'+self.dataset+'_plots', exist_ok=True) # create a subdirectory to store olots

            for train_index, test_index in kfold.split(X,Y):
            
                X_training, X_testing = X.iloc[train_index], X.iloc[test_index]
                Y_training, Y_testing = Y[train_index], Y[test_index]

                high = self.ts_length
                mid = int(high/2)
                low = 9 

                accuracies = {}
                f1_scores = {}
                harmonic_means = {}
                evaluated_timepoints = [] # timepoints for which self.train_test_prefix is called 
                                            # a.k.a. low, mid, and high
                bisect_timepoints = [] # timepoints based on which the next bisection will happen
                                       # i.e. timepoints based on which the low, mid, and high will be updated
                
                while(True): 

                    if low not in evaluated_timepoints:
                        low_result = self.train_test_prefix(self.trunc_data(X_training,low), self.trunc_data(X_testing,low), Y_training, Y_testing)
                        low_harmonic_mean = (2 * (1 - low/self.ts_length) * low_result[1]) / ((1 - low/self.ts_length) + low_result[1])
                        training_time += low_result[3]
                    if mid not in evaluated_timepoints:
                        mid_result = self.train_test_prefix(self.trunc_data(X_training,mid), self.trunc_data(X_testing,mid), Y_training, Y_testing)
                        mid_harmonic_mean = (2 * (1 - mid/self.ts_length) * mid_result[1]) / ((1 - mid/self.ts_length) + mid_result[1])
                        training_time += mid_result[3]
                    if high not in evaluated_timepoints:
                        high_result = self.train_test_prefix(self.trunc_data(X_training,high), self.trunc_data(X_testing,high), Y_training, Y_testing)
                        high_harmonic_mean = (2 * (1 - high/self.ts_length) * high_result[1]) / ((1 - high/self.ts_length) + high_result[1])
                        training_time += high_result[3]
                    # train_test_prefix returns (predictions, accuracy, f1 score, training time, test time)

                    for x in zip([low, mid, high],[low_result, mid_result, high_result],[low_harmonic_mean, mid_harmonic_mean, high_harmonic_mean]):
                        if x[0] not in evaluated_timepoints:
                            evaluated_timepoints.append(x[0])
                            accuracies[x[0]] = x[1][1]
                            f1_scores[x[0]] = x[1][2]
                            harmonic_means[x[0]] = x[2]
                        
                    
                    if self.optimize < 2: # 0 = accuracy, 1 = f1 score 
                        
                        if mid - low < 2: # adjacent timepoints => can't bisect further
                            if low_result[self.optimize + 1] >= mid_result[self.optimize + 1]:
                                bisect_timepoints.append((low, low_result[self.optimize + 1]))
                                break
                            else:
                                bisect_timepoints.append((mid, mid_result[self.optimize + 1]))
                                break 
                        elif high - mid < 2:
                            if mid_result[self.optimize + 1] >= high_result[self.optimize + 1]:
                                bisect_timepoints.append((mid, mid_result[self.optimize + 1]))
                                break
                            else:
                                bisect_timepoints.append((high, high_result[self.optimize + 1]))
                                break 
                    
                        if high_result[self.optimize + 1] > mid_result[self.optimize + 1] and high_result[self.optimize + 1] > low_result[self.optimize + 1]:
                            bisect_timepoints.append((high, high_result[self.optimize + 1]))
                            low = mid 
                            mid += int((high-low)/2)
                        elif low_result[self.optimize + 1] >= high_result[self.optimize + 1] and low_result[self.optimize + 1] >= mid_result[self.optimize + 1]:
                            bisect_timepoints.append((low, low_result[self.optimize + 1]))
                            high = mid 
                            mid = int((high-low)/2) + low
                        elif mid_result[self.optimize + 1] >= high_result[self.optimize + 1] and mid_result[self.optimize + 1] > low_result[self.optimize + 1]:
                            bisect_timepoints.append((mid, mid_result[1]))
                            low += int((mid-low)/2) 
                            high = int((high-mid)/2) + mid


                    else: # harmonic mean
                        
                        if mid - low < 2: # adjacent timepoints => can't bisect further
                            if low_harmonic_mean >= mid_harmonic_mean:
                                bisect_timepoints.append((low, low_harmonic_mean))
                                break
                            else:
                                bisect_timepoints.append((mid, mid_harmonic_mean))
                                break 
                        elif high - mid < 2:
                            if mid_harmonic_mean >= high_harmonic_mean:
                                bisect_timepoints.append((mid, mid_harmonic_mean))
                                break
                            else:
                                bisect_timepoints.append((high, high_harmonic_mean))
                                break 
                        
                        if high_harmonic_mean > mid_harmonic_mean and high_harmonic_mean > low_harmonic_mean:
                            bisect_timepoints.append((high, high_harmonic_mean))
                            low = mid 
                            mid += int((high-low)/2)
                        elif low_harmonic_mean >= high_harmonic_mean and low_harmonic_mean >= mid_harmonic_mean:
                            bisect_timepoints.append((low, low_harmonic_mean))
                            high = mid 
                            mid = int((high-low)/2) + low
                        elif mid_harmonic_mean >= high_harmonic_mean and mid_harmonic_mean > low_harmonic_mean:
                            bisect_timepoints.append((mid, mid_result[1]))
                            low += int((mid-low)/2)
                            high = int((high-mid)/2) + mid

                # sort bisect_timepoints by timepoint and then find max metric score
                bisect_timepoints.sort(key=lambda item:item[0])
                best_timepoints.append(max(bisect_timepoints,key=lambda item:item[1])[0])

                # sort evaluated timepoints in ascending order 
                evaluated_timepoints.sort() 
                evaluated_timepoints = np.array(evaluated_timepoints)
                
                # sort metric score dicts by timepoint (key) in ascending order and convert to ndarray
                accuracies = np.array([x[1] for x in sorted(accuracies.items())])
                f1_scores = np.array([x[1] for x in sorted(f1_scores.items())])
                harmonic_means = np.array([x[1] for x in sorted(harmonic_means.items())])

                with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_minirocket_fav_accuracy.txt','a') as f: #what if the user wants to delete the existing file before running again the experiment?
                    np.savetxt(f, (evaluated_timepoints, accuracies),  delimiter=',')

                with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_minirocket_fav_f1_score.txt','a') as f: 
                    np.savetxt(f, (evaluated_timepoints,f1_scores),  delimiter=',')

                with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_minirocket_fav_harmonic_mean.txt','a') as f: 
                    np.savetxt(f, (evaluated_timepoints,harmonic_means),  delimiter=',')

                #plot accuracy
                full_time_accuracy = np.repeat(accuracies[-1], evaluated_timepoints.shape[0]) # full tsc accuracy for comparison
                fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
                ax.plot(evaluated_timepoints, full_time_accuracy , '--', color = 'blue', label = "Minirocket Full-Time")
                ax.plot(evaluated_timepoints, accuracies, color = 'blue', label = "Minirocket STRUT-FAV") 
                plt.xlabel("Truncation Timepoint")
                plt.ylabel("Accuracy")
                plt.ylim(0.0,1.1)
                plt.legend() 
                plt.savefig('results/'+self.dataset+'_plots/minirocket_strut_fav_accuracy.pdf', format='pdf', dpi=320, bbox_inches='tight')  
                

                #plot f1-score
                full_time_f1_score = np.repeat(f1_scores[-1], evaluated_timepoints.shape[0]) # full tsc f1 score for comparison
                fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
                ax.plot(evaluated_timepoints, full_time_f1_score , '--', color = 'blue', label = "Minirocket Full-Time")
                ax.plot(evaluated_timepoints, f1_scores, color = 'blue', label = "Minirocket STRUT-FAV") 
                plt.xlabel("Truncation Timepoint")
                plt.ylabel("F1-score")
                plt.ylim(0.0,1.1)
                plt.legend() 
                plt.savefig('results/'+self.dataset+'_plots/minirocket_strut_fav_f1_score.pdf', format='pdf', dpi=320, bbox_inches='tight') 

                #plot harmonic mean
                fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
                ax.plot(evaluated_timepoints, harmonic_means, color = 'blue', label = "Minirocket STRUT-FAV") 
                plt.xlabel("Truncation Timepoint")
                plt.ylabel("Harmonic Mean")
                plt.ylim(0.0,1.1)
                plt.legend() 
                plt.savefig('results/'+self.dataset+'_plots/minirocket_strut_fav_harmonic_mean.pdf', format='pdf', dpi=320, bbox_inches='tight') 
                
                
            #Test for best earliness

            best_timepoint = int(np.array([x for x in best_timepoints]).mean())
            Y_TRAIN = label_encoder.fit_transform(Y_TRAIN)
            Y_TEST = label_encoder.fit_transform(Y_TEST)
            result = self.train_test_prefix(self.trunc_data(X_TRAIN,best_timepoint), self.trunc_data(X_TEST,best_timepoint), Y_TRAIN, Y_TEST)
            preds = [(best_timepoint,result[0][x]) for x in range(result[0].size)]
            testing_time = result[4]
            
            return preds, training_time, testing_time, float(best_timepoint)/self.ts_length

        def weasel_strut_fav(self, train_data, test_data):

            if isinstance(train_data, str):  # arrf files provided as input (no cv)    
                train_data = pd.DataFrame(arff.loadarff(train_data)[0])
                test_data = pd.DataFrame(arff.loadarff(test_data)[0])
                data = train_data.append(test_data) 
                X = data.iloc[:,:-1] 
                X =  self.data_reform(X)
                Y = data.iloc[: ,-1].values.ravel() 
                X = np.nan_to_num(X)
                label_encoder = LabelEncoder()
                Y = label_encoder.fit_transform(Y)
                X_TRAIN = self.data_reform(train_data.iloc[:,:-1])
                Y_TRAIN = train_data.iloc[: ,-1].values.ravel()
                X_TEST  = self.data_reform(test_data.iloc[:,:-1])
                Y_TEST  = test_data.iloc[: ,-1].values.ravel()
                Y_TRAIN = label_encoder.fit_transform(Y_TRAIN)
                Y_TEST = label_encoder.fit_transform(Y_TEST)
            elif isinstance(train_data, tuple): # perform cv 
                X = train_data[0]
                X = np.nan_to_num(X) 
                Y = train_data[1]
                label_encoder = LabelEncoder()
                Y = label_encoder.fit_transform(Y)
                X_TRAIN = X
                Y_TRAIN = Y
                X_TEST = test_data[0]
                X_TEST = np.nan_to_num(X_TEST)
                Y_TEST = test_data[1]
                Y_TEST = label_encoder.fit_transform(Y_TEST)
            else: # single csv file provided as input (no cv) 
                X = train_data
                Y = test_data
                Y = Y.ravel()
                X = np.nan_to_num(X)
                label_encoder = LabelEncoder()
                Y = label_encoder.fit_transform(Y)
                X_TRAIN , X_TEST, Y_TRAIN,  Y_TEST  = train_test_split(X, Y, test_size=0.2,  stratify = Y, random_state=0)
                Y_TRAIN = label_encoder.fit_transform(Y_TRAIN)
                Y_TEST = label_encoder.fit_transform(Y_TEST)
            
            kfold = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=0)  
            start = 11 # Weasel can't process input time series with lengths less than 11


            training_time = 0.0
            best_timepoints = []

            # save metric scores in a 2d ndarray where each each row corresponds to a split 
            # and each column to a evaluated time point 
            # the first row of stores the evluated time points

            os.makedirs('results', exist_ok=True) # create a directory to store results
            os.makedirs('results/'+self.dataset+'_metric_scores', exist_ok=True) # create a subdirectory to store metric scores
            os.makedirs('results/'+self.dataset+'_plots', exist_ok=True) # create a subdirectory to store olots

            for train_index, test_index in kfold.split(X,Y):
            
                X_training, X_testing = X[train_index], X[test_index]
                Y_training, Y_testing = Y[train_index], Y[test_index]

                high = self.ts_length
                mid = int(high/2)
                low = 9 

                accuracies = {}
                f1_scores = {}
                harmonic_means = {}
                evaluated_timepoints = [] # timepoints for which self.train_test_prefix is called 
                                            # a.k.a. low, mid, and high
                bisect_timepoints = [] # timepoints based on which the next bisection will happen
                                       # i.e. timepoints based on which the low, mid, and high will be updated
                
                while(True): 

                    if low not in evaluated_timepoints:
                        if self.variate > 1:
                            low_result = self.train_test_prefix(X_training[:,:,:low], X_testing[:,:,:low], Y_training, Y_testing)
                        else:
                            low_result = self.train_test_prefix(X_training[:,:low], X_testing[:,:low], Y_training, Y_testing)
                        low_harmonic_mean = (2 * (1 - low/self.ts_length) * low_result[1]) / ((1 - low/self.ts_length) + low_result[1])
                        training_time += low_result[3]
                    if mid not in evaluated_timepoints:
                        if self.variate > 1:
                            mid_result = self.train_test_prefix(X_training[:,:,:mid], X_testing[:,:,:mid], Y_training, Y_testing)
                        else:
                            mid_result = self.train_test_prefix(X_training[:,:mid], X_testing[:,:mid], Y_training, Y_testing)
                        mid_harmonic_mean = (2 * (1 - mid/self.ts_length) * mid_result[1]) / ((1 - mid/self.ts_length) + mid_result[1])
                        training_time += mid_result[3]
                    if high not in evaluated_timepoints:
                        if self.variate > 1:
                            high_result = self.train_test_prefix(X_training[:,:,:high], X_testing[:,:,:high], Y_training, Y_testing)
                        else:
                            high_result = self.train_test_prefix(X_training[:,:high], X_testing[:,:high], Y_training, Y_testing)
                        high_harmonic_mean = (2 * (1 - high/self.ts_length) * high_result[1]) / ((1 - high/self.ts_length) + high_result[1])
                        training_time += high_result[3]
                    # train_test_prefix returns (predictions, accuracy, f1 score, training time, test time)

                    for x in zip([low, mid, high],[low_result, mid_result, high_result],[low_harmonic_mean, mid_harmonic_mean, high_harmonic_mean]):
                        if x[0] not in evaluated_timepoints:
                            evaluated_timepoints.append(x[0])
                            accuracies[x[0]] = x[1][1]
                            f1_scores[x[0]] = x[1][2]
                            harmonic_means[x[0]] = x[2]
                        
                    
                    if self.optimize < 2: # 0 = accuracy, 1 = f1 score 
                        
                        if mid - low < 2: # adjacent timepoints => can't bisect further
                            if low_result[self.optimize + 1] >= mid_result[self.optimize + 1]:
                                bisect_timepoints.append((low, low_result[self.optimize + 1]))
                                break
                            else:
                                bisect_timepoints.append((mid, mid_result[self.optimize + 1]))
                                break 
                        elif high - mid < 2:
                            if mid_result[self.optimize + 1] >= high_result[self.optimize + 1]:
                                bisect_timepoints.append((mid, mid_result[self.optimize + 1]))
                                break
                            else:
                                bisect_timepoints.append((high, high_result[self.optimize + 1]))
                                break 
                    
                        if high_result[self.optimize + 1] > mid_result[self.optimize + 1] and high_result[self.optimize + 1] > low_result[self.optimize + 1]:
                            bisect_timepoints.append((high, high_result[self.optimize + 1]))
                            low = mid 
                            mid += int((high-low)/2)
                        elif low_result[self.optimize + 1] >= high_result[self.optimize + 1] and low_result[self.optimize + 1] >= mid_result[self.optimize + 1]:
                            bisect_timepoints.append((low, low_result[self.optimize + 1]))
                            high = mid 
                            mid = int((high-low)/2) + low
                        elif mid_result[self.optimize + 1] >= high_result[self.optimize + 1] and mid_result[self.optimize + 1] > low_result[self.optimize + 1]:
                            bisect_timepoints.append((mid, mid_result[1]))
                            low += int((mid-low)/2) 
                            high = int((high-mid)/2) + mid


                    else: # harmonic mean
                        
                        if mid - low < 2: # adjacent timepoints => can't bisect further
                            if low_harmonic_mean >= mid_harmonic_mean:
                                bisect_timepoints.append((low, low_harmonic_mean))
                                break
                            else:
                                bisect_timepoints.append((mid, mid_harmonic_mean))
                                break 
                        elif high - mid < 2:
                            if mid_harmonic_mean >= high_harmonic_mean:
                                bisect_timepoints.append((mid, mid_harmonic_mean))
                                break
                            else:
                                bisect_timepoints.append((high, high_harmonic_mean))
                                break 
                        
                        if high_harmonic_mean > mid_harmonic_mean and high_harmonic_mean > low_harmonic_mean:
                            bisect_timepoints.append((high, high_harmonic_mean))
                            low = mid 
                            mid += int((high-low)/2)
                        elif low_harmonic_mean >= high_harmonic_mean and low_harmonic_mean >= mid_harmonic_mean:
                            bisect_timepoints.append((low, low_harmonic_mean))
                            high = mid 
                            mid = int((high-low)/2) + low
                        elif mid_harmonic_mean >= high_harmonic_mean and mid_harmonic_mean > low_harmonic_mean:
                            bisect_timepoints.append((mid, mid_result[1]))
                            low += int((mid-low)/2)
                            high = int((high-mid)/2) + mid

                # sort  bisect_timepoints by timepoint and then find max metric score
                bisect_timepoints.sort(key=lambda item:item[0])
                best_timepoints.append(max(bisect_timepoints,key=lambda item:item[1])[0])

                # sort evaluated timepoints in ascending order 
                evaluated_timepoints.sort() 
                evaluated_timepoints = np.array(evaluated_timepoints)
                

                # sort metric score dicts by timepoint (key) in ascending order and convert to ndarray
                accuracies = np.array([x[1] for x in sorted(accuracies.items())])
                f1_scores = np.array([x[1] for x in sorted(f1_scores.items())])
                harmonic_means = np.array([x[1] for x in sorted(harmonic_means.items())])

                with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_weasel_fav_accuracy.txt','a') as f: #what if the user wants to delete the existing file before running again the experiment?
                    np.savetxt(f, (evaluated_timepoints, accuracies),  delimiter=',')

                with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_weasel_fav_f1_score.txt','a') as f: 
                    np.savetxt(f, (evaluated_timepoints,f1_scores),  delimiter=',')

                with open('results/'+self.dataset+'_metric_scores/'+self.dataset+'_weasel_fav_harmonic_mean.txt','a') as f: 
                    np.savetxt(f, (evaluated_timepoints,harmonic_means),  delimiter=',')

                #plot accuracy
                full_time_accuracy = np.repeat(accuracies[-1], evaluated_timepoints.shape[0]) # full tsc accuracy for comparison
                fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
                ax.plot(evaluated_timepoints, full_time_accuracy , '--', color = 'blue', label = "Weasel Full-Time")
                ax.plot(evaluated_timepoints, accuracies, color = 'blue', label = "Weasel STRUT-FAV") 
                plt.xlabel("Truncation Timepoint")
                plt.ylabel("Accuracy")
                plt.ylim(0.0,1.1)
                plt.legend() 
                plt.savefig('results/'+self.dataset+'_plots/weasel_strut_fav_accuracy.pdf', format='pdf', dpi=320, bbox_inches='tight')  

                #plot f1-score
                full_time_f1_score = np.repeat(f1_scores[-1], evaluated_timepoints.shape[0]) # full tsc f1 score for comparison
                fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
                ax.plot(evaluated_timepoints, full_time_f1_score , '--', color = 'blue', label = "Weasel Full-Time")
                ax.plot(evaluated_timepoints, accuracies, color = 'blue', label = "Weasel STRUT-FAV") 
                plt.xlabel("Truncation Timepoint")
                plt.ylabel("F1-score")
                plt.ylim(0.0,1.1)
                plt.legend() 
                plt.savefig('results/'+self.dataset+'_plots/weasel_strut_fav_f1_score.pdf', format='pdf', dpi=320, bbox_inches='tight') 


                #plot harmonic mean
                fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
                ax.plot(evaluated_timepoints, harmonic_means, color = 'blue', label = "Weasel STRUT-FAV") 
                plt.xlabel("Truncation Timepoint")
                plt.ylabel("Harmonic Mean")
                plt.ylim(0.0,1.1)
                plt.legend() 
                plt.savefig('results/'+self.dataset+'_plots/weasel_strut_fav_harmonic_mean.pdf', format='pdf', dpi=320, bbox_inches='tight') 

                
            #Test for best earliness

            best_timepoint = int(np.array([x for x in best_timepoints]).mean())
            Y_TRAIN = label_encoder.fit_transform(Y_TRAIN)
            Y_TEST = label_encoder.fit_transform(Y_TEST)
            if self.variate > 1:
                result = self.train_test_prefix(X_TRAIN[:,:,:best_timepoint], X_TEST[:,:,:best_timepoint], Y_TRAIN, Y_TEST)
            else:
                result = self.train_test_prefix(X_TRAIN[:,:best_timepoint], X_TEST[:,:best_timepoint], Y_TRAIN, Y_TEST)
            preds = [(best_timepoint,result[0][x]) for x in range(result[0].size)]
            testing_time = result[4]


            return preds, training_time, testing_time, float(best_timepoint)/self.ts_length