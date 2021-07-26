import math
import os
import sys
from cmath import nan
from typing import Sequence, Tuple, Optional
from os import path
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn import preprocessing


def arff_parser(file_name):
    """

        Program that reads arff files and turns it into dataframes readble for our framework

        :param file_name: Name of the .arff file
        :param type_file: 0 for train file and 1 for test
    """

    files = os.listdir("./data/UCR_UEA")
    for file in files:
        if ".py" in file:
            continue
        if file in file_name:
            # read the data
            raw_data = loadarff(file_name)
            df = pd.DataFrame(raw_data[0])
            # encode the labels
            labels = df.iloc[:, -1]
            encoder = preprocessing.LabelEncoder()
            new = pd.Series(encoder.fit_transform(labels), name="Class")
            df.drop(df.columns[[-1]], axis=1, inplace=True)
            df['Class'] = new
            new_df = None
            # prepare the new dataframe where each (number_of_variable) rows will represent a time-series
            columns = None
            for index, row in df.iterrows():
                label = [row[-1]]
                if len(row) > 2:
                    attributes = [row[:-1]]
                    variables = 1
                else:
                    attributes = row[0]
                    variables = len(attributes)
                    attributes = np.asarray(attributes)
                if columns is None:
                    columns = ['Class'] + list(range(0, len(attributes[0])))
                for attribute in attributes:
                    attribute = label + list(attribute)
                    if new_df is None:
                        new_df = pd.DataFrame(np.reshape(attribute, (1, len(attribute))), columns=columns)
                    else:
                        # data = np.reshape(attribute, (1, len(attribute)))
                        data = pd.Series(attribute, index=columns, name=str(index))
                        new_df = new_df.append(data)
            new_df = nan_handler(new_df)
            return new_df, variables
    return None,None


def nan_handler(df: pd.DataFrame):
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    df = df.reset_index(drop =True)
    for index_r, row in df.iterrows():  # for each row
        prev = 0
        next = 0
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

def earliness(predictions: Sequence[Tuple[int, int]], ts_length: int) -> Optional[float]:
    """
    Computes the earliness.

    :param predictions: a sequence of prediction tuples of the form (timestamp, class)
    :param ts_length: the length of the time-series
    :return: the mean timestamp in which predictions are made
    """

    # If the time-series length is zero or no predictions are provided,
    # the earliness can not be defined.
    if ts_length == 0 or not predictions:
        return None

    return sum([(t) / (ts_length + 1) for t, _ in predictions]) / len(predictions)


def DataSetCreation_EDSC(dimension, rowtraining, rowtesting, classes, numbers):
    """Code created in order to run the ECTS and EDSC static code. It created the DatasetInformation.h dynamically"""
    f = open("C_files/edsc/DataSetInformation.h", "w+")
    f.write("#include <string>\n\n")
    f.write("const int DIMENSION = {};\n".format(dimension))
    f.write("const int ROWTRAINING = {};\n".format(rowtraining))
    f.write("const int ROWTESTING = {};\n".format(rowtesting))
    f.write(
        """const char* trainingFileName="C_files/edsc/Data/train";\nconst char* testingFileName="C_files/edsc/Data/test";\nconst char* resultFileName="C_files/edsc/Data/result.txt";\nconst char* path ="C_files/edsc";\n""")
    f.write("const int  NofClasses = {};\n".format(len(classes)))
    f.write("const int Classes[] = {")
    f.write(str(int(classes[0])))
    for i in classes[1:]:
        f.write(",{}".format(int(i)))
    f.write("};\n")
    f.write("const int ClassIndexes[] = {")
    sum = 0
    f.write("{}".format(sum))
    for index, values in list(numbers.items())[:-1]:
        sum += int(values)
        f.write(",{}".format(sum))
    f.write("};\n")
    f.write("const int ClassNumber[] = {")
    f.write("{}".format(list(numbers.items())[0][1]))
    for index, value in list(numbers.items())[1:]:
        f.write(",{}".format(int(value)))
    f.write("};\n")
    f.close()


def topy(train, train_labels, test, timesteps):
    """
    Preprocessign data for the MLSTM
    """
    training = []
    variables = len(train)
    classes = train_labels.value_counts()
    classes = classes.count()
    for index in range(train[0].shape[0]):
        feature_list = []
        for feature in train:
            feature_list.append(feature.iloc[index].tolist())
        training.append(feature_list)
    testing = []
    for index in range(test[0].shape[0]):
        feature_list = []
        for feature in test:
            feature_list.append(feature.iloc[index].tolist())
        testing.append(feature_list)
    train_labels = train_labels.tolist()
    path = "./ets/algorithms/MLSTM/data/current_dataset"
    try:
        os.mkdir(path, 0o755)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)
    np.save('./ets/algorithms/MLSTM/data/current_dataset/X_test.npy', testing)
    np.save('./ets/algorithms/MLSTM/data/current_dataset/y_train.npy', train_labels)
    np.save('./ets/algorithms/MLSTM/data/current_dataset/X_train.npy', training)
    f2 = open("./ets/algorithms/MLSTM/utils/constants.txt", "w")
    f2.write("TRAIN_FILES=./ets/algorithms/MLSTM/data/current_dataset/\n")
    f2.write("TEST_FILES=./ets/algorithms/MLSTM/data/current_dataset/\n")
    f2.write("MAX_NB_VARIABLES=" + str(variables) + "\n")
    f2.write("MAX_TIMESTEPS_LIST=" + str(timesteps) + "\n")
    f2.write("NB_CLASSES_LIST=" + str(classes) + "\n")
    f2.close()


def temp_accuracy(predictions: Sequence[int], ground_truth_labels: Sequence[int]) -> Optional[float]:
    """
        Computes the accuracy.

        :param predictions: a sequence of prediction tuples of the form (timestamp, class)
        :param ground_truth_labels: a sequence of ground truth labels
        :return: the percentage of correctly classified instances
        """

    # If no predictions or ground truth is provided,
    # the accuracy can not be defined.
    if not ground_truth_labels or not list(predictions):
        return None

    correct = sum([1 for (prediction, y) in zip(predictions, ground_truth_labels) if prediction == y])
    return correct / len(ground_truth_labels)


def accuracy(predictions: Sequence[Tuple[int, int]], ground_truth_labels: Sequence[int]) -> Optional[float]:
    """
    Computes the accuracy.

    :param predictions: a sequence of prediction tuples of the form (timestamp, class)
    :param ground_truth_labels: a sequence of ground truth labels
    :return: the percentage of correctly classified instances
    """

    # If no predictions or ground truth is provided,
    # the accuracy can not be defined.
    if not ground_truth_labels or not predictions:
        return None

    correct = sum([1 for ((_, prediction), y) in zip(predictions, ground_truth_labels) if prediction == y])
    return correct / len(ground_truth_labels)


def harmonic_mean(acc: float, earl: float):
    """
    Computes the harmonic mean as illustrated by Patrick Schäfer et al. 2020
    "TEASER: early and accurate time series classification"

    :param acc: The accuracy of the prediction
    :param earl: The earliness of the prediction
    """
    return (2 * (1 - earl) * acc) / ((1 - earl) + acc)


def counts(target_class: int,
           predictions: Sequence[Tuple[int, int]],
           ground_truth_labels: Sequence[int]) -> Tuple[int, int, int, int]:
    """
    Counts the correct and erroneous predictions according to a given target class.

    :param target_class: a class of interest
    :param predictions: a sequence of prediction tuples of the form (timestamp, class)
    :param ground_truth_labels: a sequence of ground truth labels
    :return: a tuple holding the true positives, true negatives, false positives, false negatives.
    """

    tp, tn, fp, fn = 0, 0, 0, 0
    for ((_, prediction), y) in zip(predictions, ground_truth_labels):
        if prediction == target_class and y == target_class:
            tp += 1
        elif prediction != target_class and y != target_class:
            tn += 1
        elif prediction == target_class and y != target_class:
            fp += 1
        else:
            fn += 1

    return tp, tn, fp, fn


def precision(tp: int, fp: int) -> Optional[float]:
    """
    Computes the precision.

    :param tp: true positives
    :param fp: false positives
    :return: a precision value, or None if counts are zero
    """
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0


def recall(tp: int, fn: int) -> Optional[float]:
    """
    Computes the recall.

    :param tp: true positives
    :param fn: false negatives
    :return: a recall value, or None if counts are zero
    """
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0


def f_measure(tp: int, fp: int, fn: int, beta: int = 1) -> Optional[float]:
    """
    Computes the F-measure.

    :param tp: true positives
    :param fp: false positives
    :param fn: false negatives
    :param beta: beta parameter (default is 1)
    :return: f-beta measure, or None if precision or recall is zero
    """

    pr = precision(tp, fp)
    re = recall(tp, fn)
    if pr is None or re is None or (pr == 0 and re == 0):
        return 0
    else:
        return ((1 + beta ** 2) * pr * re) / (beta ** 2 * pr + re)


#
# Transformation functions
#


def df_merge(df_seq: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of data frames by computing their average.

    :param df_seq: a sequence of data frames
    :return: a data frame holding the average of the given data frames
    """
    df_sum = pd.DataFrame()
    for df in df_seq:
        df_sum = df.add(df_sum, fill_value=0)
    return df_sum.div(len(df_seq))


def df_dimensionality_reduction(df: pd.DataFrame, dimensions: int) -> pd.DataFrame:
    """
    Dimensionality reduction on a data frame using the Piecewise Aggregate Approximation (PAA).

    :param df: a data frame to reduce
    :param dimensions: number of dimensions to retain
    :return: a reduced data frame
    """

    columns = df.shape[1]

    # If the df columns are already less than the desired dimensions, we should not change anything
    if columns <= dimensions:
        return df

    # Compute segment size
    segment_size = round(columns / dimensions)

    reduced_data = pd.DataFrame()
    for _, row in df.iterrows():

        pos = 0
        aggregated_series = []

        for d in range(0, dimensions):

            # For the final dimension, find the average of the remaining timestamps
            if d == dimensions - 1:
                avg = sum(row[pos:columns]) / (columns - pos)
                aggregated_series.append(avg)

            # Or else calculate the average of segment size
            else:
                avg = sum(row[pos:pos + segment_size]) / segment_size
                aggregated_series.append(avg)
                pos = pos + segment_size

        reduced_data = reduced_data.append(pd.Series(aggregated_series), ignore_index=True)

    return reduced_data
