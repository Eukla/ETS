import subprocess
from typing import Sequence, Optional

import pandas as pd

from ets.algorithms.early_classifier import EarlyClassifier


class TEASER(EarlyClassifier):
    """
        The TEASER algorithm.

        Publications:

        TEASER: early and accurate time series classification(2020)
    """

    def train(self, train_data: pd.DataFrame, labels: Sequence[int]) -> None:
        pass

    def __init__(self, timestamps, S: int):
        """
        Creates the Teaser object

        :param timestamps: The list of timestamps for classification
        :param S: The total number of slave-master classifier pairs
        :param bins: Number of bins for WEASEL
        """
        self.timestamps = timestamps
        self.S = S

        self.dataset: Optional[pd.DataFrame] = None
        self.labels: Sequence[int]

    def predict(self, test_data):
        bin_output = subprocess.check_output(["java", "-jar", "Java/sfa.main.jar", str(self.S), "./train", "./test"])
        output = bin_output.decode("utf-8")
        train = 0
        truncated_output = output.split("\n")
        new = truncated_output.copy()
        counter = 0
        for item in truncated_output:
            if item == '-1 -1':
                new.pop(counter)
                break
            elif "TimeTrain" in item:
                out = item.split(" ")
                new.pop(counter)
                counter -= 1
                train = out[1]
            else:
                new.pop(counter)
                counter -= 1
            counter += 1
        final_list = []
        final_c = 0
        test = 0
        for item in new:
            if final_c == 0:
                found = item.find("Class")
                if found == -1:
                    final_list.append(item)
                else:
                    final_c = 1
            elif "TimeTest" in item:
                out = item.split(" ")
                test = out[1]
                break
        predictions = []
        for item in final_list:
            final = item.split(" ")
            predictions.append([float(final[0]), float(final[1])])
        return predictions, train, test
