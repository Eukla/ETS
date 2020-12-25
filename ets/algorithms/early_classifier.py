import abc
import pandas as pd
from typing import Tuple, Sequence


class EarlyClassifier(metaclass=abc.ABCMeta):
    """
    EarlyClassifier is an abstract class that should be extended
    by any algorithm for early time-series classification.
    """

    @abc.abstractmethod
    def train(self, train_data: pd.DataFrame, labels: Sequence[int]) -> None:
        """
        Trains the classifier.

        :param train_data: training time-series data
        :param labels: training time-series labels
        """

    @abc.abstractmethod
    def predict(self, test_data: pd.DataFrame) -> Sequence[Tuple[int, int]]:
        """
        Predict the class of the given time-series as early as possible.

        :param test_data: time-series to predict
        :return: a sequence of early predictions for the given time-series
        """
