import os

import keras
from sklearn.model_selection import StratifiedKFold

from ets.algorithms.MLSTM.mlstm_impl import run
from ets.algorithms.utils import accuracy, harmonic_mean
from ets.algorithms.utils import topy


class MLSTM():

    def __init__(self, timestamps, earliness, folds):
        self.timestamps = timestamps
        self.folds = folds
        self.predict_model = keras.Model
        if earliness is None:
            self.earliness = [0.4, 0.5, 0.6]
        else:
            self.earliness = earliness

    def true_predict(self, train_d, test_d, train_l):
        res = {}
        earl = self.earliness
        timesteps = test_d[0].shape[1]
        if not os.path.exists("./ets/algorithms/MLSTM/data"):
            os.mkdir("./ets/algorithms/MLSTM/data")
        if not os.path.exists("./ets/algorithms/MLSTM/weights"):
            os.mkdir("./ets/algorithms/MLSTM/weights")

        train_time = 0
        test_time = 0
        indices = zip(StratifiedKFold(5).split(train_d[0], train_l), range(1, self.folds + 1))
        variables = len(train_d)
        for ((train_indices, test_indices), i) in indices:
            h_max = 0
            res[i] = None  # For each fold
            fold_train_data = [train_d[i].iloc[train_indices].reset_index(drop=True) for i in
                               range(0, variables)]
            fold_test_data = [train_d[i].iloc[test_indices].reset_index(drop=True) for i in
                              range(0, variables)]
            fold_train_labels = train_l[train_indices].reset_index(drop=True)
            fold_test_labels = train_l[test_indices].reset_index(drop=True)
            for earliness in earl:
                sizes = int(len(self.timestamps) * earliness)
                new_d = []
                for data in fold_train_data:
                    temp = data.iloc[:, 0:self.timestamps[sizes]]
                    new_d.append(temp)
                new_t = []
                for data in fold_test_data:
                    temp = data.iloc[:, 0:self.timestamps[sizes]]
                    new_t.append(temp)
                topy(new_d, fold_train_labels, new_t, timesteps)
                result = run(earliness)  # 1)prediction 2) train time 3) test time 4) LSTM cell
                results = []
                labels = sorted(fold_train_labels.unique())
                for item in result[0]:
                    results.append((item[0], labels[item[1]]))
                acc = accuracy(results, fold_test_labels.to_list())
                harmonic_means = harmonic_mean(acc, sizes / len(self.timestamps))
                if h_max < harmonic_means:
                    h_max = harmonic_means
                    best_earl = earliness
                    best_cell = result[3]
                    res[i] = (h_max, best_earl, best_cell)
                train_time += result[1]
                train_time += result[2]
        """We need to pick the best earliness and cells for the final prediction"""
        counts = {}
        best_folds = {}
        for earliness in earl:  # initallizing
            counts[earliness] = 0
            best_folds[earliness] = []

        for i in range(1, self.folds + 1):  # find the most common picked earliness among folds
            counts[res[i][1]] += 1
            best_folds[res[i][1]].append(i)
        count = 0
        best_earl = 0
        for earliness in earl:
            if count < counts[earliness]:
                count = counts[earliness]
                best_earl = earliness
        best_score = 0  # find the best LSTM cells
        best_cell = 0
        for item in best_folds[best_earl]:
            if best_score < res[item][0]:
                best_score = res[item][0]
                best_cell = res[item][2]
        print(best_earl, best_cell)
        sizes = int(len(self.timestamps) * best_earl)
        new_d = []
        for data in train_d:
            temp = data.iloc[:, 0:self.timestamps[sizes]]
            new_d.append(temp)
        new_t = []
        for data in test_d:
            temp = data.iloc[:, 0:self.timestamps[sizes]]
            new_t.append(temp)
        topy(new_d, train_l, new_t, timesteps)
        result = run(best_earl, best_cell)
        results = []
        labels = sorted(train_l.unique())
        for item in result[0]:
            results.append((item[0], labels[item[1]]))
        train_time += result[1]
        test_time += result[2]
        print(train_time, test_time)
        return results, train_time, test_time, best_earl, best_cell
