import logging
import subprocess
from typing import List, Tuple, Sequence

import coloredlogs
import pandas as pd

# Configure the logger (change level to DEBUG for more information)
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger,
                    fmt='%(asctime)s - %(hostname)s - %(name)s[%(process)d] - [%(levelname)s]: %(message)s')


class ECEC():
    """
    The ECEC algorithm.

    Publications:

    An Effective Confidence-Based Early Classification of Time Series (2019)
    """

    def __init__(self, timestamps: Sequence[int]):

        self.timestamps = timestamps

    def predict(self, test_data: pd.DataFrame) -> List[Tuple[int, int]]:
        predictions = []
        bin_output = subprocess.check_output(["java", "-jar", "Java/ecec_test.main.jar", "./train", "./test"])
        output = bin_output.decode("utf-8")
        truncated_output = output.split("\n")
        new = truncated_output.copy()
        counter = 0
        for item in truncated_output:
            if item == '-1 -1':
                new.pop(counter)
                break
            else:
                new.pop(counter)
                counter -= 1
            counter += 1
        predictions = []
        for item in new:
            if item == '\n':
                continue
            if "TimeTrain" in item:
                final = item.split(" ")
                train = final[1]
                continue
            if "TimeTest" in item:
                final = item.split(" ")
                test = final[1]
                break
            if item == '':
                continue
            final = item.split(" ")
            predictions.append([float(final[1]), float(final[0])])
        return predictions, train, test
