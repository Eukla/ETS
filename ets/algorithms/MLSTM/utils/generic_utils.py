import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pylab as plt

mpl.style.use('seaborn-paper')


def load_dataset_at(index, fold_index=None, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    f = open("./ets/algorithms/MLSTM/utils/constants.txt", "r")
    lines = f.read()
    lines = lines.split("\n")
    train = None
    test = None
    for line in lines:
        print(line)
        if "TRAIN" in line:
            line = line.split("=")
            train = line[1]
        if "TEST" in line:
            line = line.split("=")
            test = line[1]
    f.close()
    if verbose: print("Loading train / test dataset : ", train, test)
    if fold_index is None:
        x_train_path = train + "X_train.npy"
        y_train_path = train+ "y_train.npy"
        x_test_path = test + "X_test.npy"
    else:
        x_train_path = train + "X_train_%d.npy" % fold_index
        y_train_path = train + "y_train_%d.npy" % fold_index
        x_test_path = test + "X_test_%d.npy" % fold_index
    import os

    path = os.getcwd()

    print(os.path.isdir("./ets/algorithms/MLSTM/data/current_dataset"),x_train_path)
    if os.path.exists(x_train_path):

        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
    elif os.path.exists(x_train_path[1:]):
        X_train = np.load(x_train_path[1:])
        y_train = np.load(y_train_path[1:])
        X_test = np.load(x_test_path[1:])
    else:
        raise FileNotFoundError('File %s not found!' % (train))

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished processing train dataset..")


    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, is_timeseries


def calculate_dataset_metrics(X_train):
    max_nb_variables = X_train.shape[1]
    max_timesteps = X_train.shape[-1]

    return max_timesteps, max_nb_variables


def cutoff_choice(dataset_id, sequence_length):
    f = open("./ets/algorithms/MLSTM/utils/constants.txt", "r")
    lines = f.readlines()
    MAX_NB_VARIABLES = 0
    for line in lines:
        if "VARIABLES" in line:
            line = line.split("=")
            MAX_NB_VARIABLES = int(line[1])
    f.close()
    print("Original sequence length was :", sequence_length, "New sequence Length will be : ",
          MAX_NB_VARIABLES)
    choice = input('Options : \n'
                   '`pre` - cut the sequence from the beginning\n'
                   '`post`- cut the sequence from the end\n'
                   '`anything else` - stop execution\n'
                   'To automate choice: add flag `cutoff` = choice as above\n'
                   'Choice = ')

    choice = str(choice).lower()
    return choice


def cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length):
    f = open("./ets/algorithms/MLSTM/utils/constants.txt", "r")
    lines = f.readlines()
    MAX_NB_VARIABLES = 0
    for line in lines:
        if "VARIABLES" in line:
            line = line.split("=")
            MAX_NB_VARIABLES = int(line[1])
    f.close()
    assert MAX_NB_VARIABLES< sequence_length, "If sequence is to be cut, max sequence" \
                                                                   "length must be less than original sequence length."
    cutoff = sequence_length - MAX_NB_VARIABLES[dataset_id]
    if choice == 'pre':
        if X_train is not None:
            X_train = X_train[:, :, cutoff:]
        if X_test is not None:
            X_test = X_test[:, :, cutoff:]
    else:
        if X_train is not None:
            X_train = X_train[:, :, :-cutoff]
        if X_test is not None:
            X_test = X_test[:, :, :-cutoff]
    print("New sequence length :", MAX_NB_VARIABLES[dataset_id])
    return X_train, X_test


if __name__ == "__main__":
    pass