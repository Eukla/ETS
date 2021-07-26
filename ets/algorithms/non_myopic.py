from ets.algorithms.early_classifier import EarlyClassifier
from tslearn.early_classification import NonMyopicEarlyClassifier
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

import pandas as pd
import numpy as np
from typing import Sequence, Tuple


class Trigger(EarlyClassifier):
    """
    The algorithm from Dachraoui et al. 2015

    Publications:

    Early classification of time series as a non myopic sequential decision making problem(2015)

    Code:
    https://tslearn.readthedocs.io/en/stable/user_guide/early.html#examples-involving-early-classification-estimators
    """

    def __init__(self, n_clusters: int, cost_time_parameter: float, lamb: float, random_state: bool):
        self.n_clusters = n_clusters
        self.cost_time_parameter = cost_time_parameter
        self.lamb = lamb
        self.random_state = random_state
        self.classifier = NonMyopicEarlyClassifier

    def train(self, train_data: pd.DataFrame, labels: Sequence[int]) -> None:
        self.classifier = NonMyopicEarlyClassifier(n_clusters=self.n_clusters,
                                                   cost_time_parameter=self.cost_time_parameter, lamb=self.lamb,
                                                   random_state=self.random_state)
        np.random.seed(0)
        train_data = train_data.values.tolist()
        train_data = to_time_series_dataset(train_data)
        labels = labels.values
        self.classifier.fit(train_data, labels)

    def predict(self, test_data: pd.DataFrame) -> Sequence[Tuple[int, int]]:
        np.random.seed(0)
        test_data = test_data.values.tolist()
        test_data = to_time_series_dataset(test_data)
        pred, times = self.classifier.predict_class_and_earliness(test_data)
        results = []
        for preds, earl in zip(pred, times):
            results.append((earl, preds))
        return results
