import logging
import subprocess
import os
import coloredlogs
import numpy as np
import pandas as pd
from multiprocessing import Pool
import threading
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from ets.algorithms.utils import DataSetCreation_EDSC
from typing import Tuple, List, Sequence, Dict, Optional
from ets.algorithms.early_classifier import EarlyClassifier
import multiprocessing as mp
from subprocess import call, check_output

# Configure the logger (change level to DEBUG for more information)
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger,
                    fmt='%(asctime)s - %(hostname)s - %(name)s[%(process)d] - [%(levelname)s]: %(message)s')


class EDSC_C(EarlyClassifier):
    "Uses the ECTS_C ready implementation"

    def predict(self, test_data: pd.DataFrame) -> Sequence[Tuple[int, int]]:
        pass

    def __init__(self, timestamps):
        self.train_d = pd.DataFrame()
        self.test = pd.DataFrame()
        self.time_stamps = timestamps
        self.labels = pd.Series()

    def train(self, train_data: pd.DataFrame, labels: Sequence[int]) -> None:
        self.train_d = train_data
        self.labels = labels

    def predict2(self, test_data: pd.DataFrame, labels: pd.Series, numbers: pd.Series, types:int):
        self.test_d = test_data
        dimension = len(self.time_stamps)
        rowtraining = self.train_d.shape[0]
        rowtesting = self.test_d.shape[0]
        labels_c = None
        if(types == 1):
                 labels_c = sorted(self.labels.unique())
        elif(types == 0):
                 labels_c = sorted(self.labels.unique(), reverse = True)
        DataSetCreation_EDSC(dimension, rowtraining, rowtesting, labels_c, numbers)
        predictions = []
        self.train_d.insert(loc=0, value=self.labels, column="Class")
        self.test_d.insert(loc=0, value=labels, column="Class")
        self.train_d.to_csv("C_files/edsc/Data/train", sep=" ", header=False, index_label=False, index=False)
        self.test_d.to_csv("C_files/edsc/Data/test", sep=" ", header=False, index_label=False, index=False)
        os.system("g++ C_files/edsc/ByInstanceDP.cpp C_files/edsc/quickSort.cpp C_files/edsc/Euclidean.cpp -o edsc")
        bin_output = check_output(["./edsc"])
        output = bin_output.decode("utf-8")
        truncated_output = output.split("\n")
        found = False
        train=0
        test = 0
        for item in truncated_output:
            if "" == item and found:
                found=False
            if found:
                res = item.split(" ")
                if(int(res[1]) == -4):
                    occurrences = self.labels.value_counts()
                    predictions.append((self.time_stamps[-1], occurrences.idxmax()))
                else:
                    predictions.append((int(res[0]), int(res[1])))

            if "finish2" in item:
                res = item.split(" ")
                test = res[0]
                break
            if "finish" in item and not found:
                res = item.split(" ")
                train = res[0]
                found = True
        return predictions,train,test
