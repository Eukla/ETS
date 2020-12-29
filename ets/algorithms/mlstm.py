from typing import Sequence, Tuple
import os
import keras
import pandas as pd
from ets.algorithms.utils import topy

from ets.algorithms.MLSTM.mlstm_impl import run

from ets.algorithms.early_classifier import EarlyClassifier
from ets.algorithms.utils import accuracy,harmonic_mean


class MLSTM():

    def __init__(self, timestamps, earliness):
        self.timestamps = timestamps
        self.predict_model = keras.Model
        if earliness is None:
            self.earliness = [0.4,0.5,0.6]
        else:
            self.earliness = earliness

    def true_predict(self, train_d, test_d, train_l, test_l):
        res = {}
        earl = self.earliness
        h_max= 0
        best_earl = 0
        timesteps = test_d[0].shape[1]
        if not os.path.exists("./ets/algorithms/MLSTM/data"):
            os.mkdir("./ets/algorithms/MLSTM/data")
        if not os.path.exists("./ets/algorithms/MLSTM/weights"):
            os.mkdir("./ets/algorithms/MLSTM/weights")
        for earliness in earl:
            sizes = int(len(self.timestamps)*earliness)
            new_d = []
            for data in train_d:
                temp = data.iloc[:,0:self.timestamps[sizes]+1]
                new_d.append(temp)
            new_t =[]
            for data in test_d:
                temp = data.iloc[:,0:self.timestamps[sizes]+1]
                new_t.append(temp)
            topy(new_d, train_l, new_t, test_l,timesteps)
            result = run()
            results = []
            labels = sorted(train_l.unique())

            for item in result[0]:
                results.append((item[0],labels[item[1]]))

            acc = accuracy(results,test_l.to_list())
            harmonic_means = harmonic_mean(acc,sizes/len(self.timestamps))
            if h_max<harmonic_means:
                h_max=harmonic_means
                best_earl = earliness
            res[earliness] = [results,result[1],result[2],acc]
        return res[best_earl][0],res[best_earl][1],res[best_earl][2], best_earl
