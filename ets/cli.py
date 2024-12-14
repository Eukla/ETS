import logging
import sys
import os
import time
import math
from collections import Counter
from datetime import timedelta
from typing import Set, List, Tuple, Optional
import click
import coloredlogs
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle as pkl
from scipy.io import arff as scipy_arff
import arff
import ets.algorithms.utils as utils
from ets.algorithms.early_classifier import EarlyClassifier
from ets.algorithms.ecec import ECEC
from ets.algorithms.non_myopic import Trigger
from ets.algorithms.ects import ECTS
from ets.algorithms.edsc_c import EDSC_C
from ets.algorithms.mlstm import MLSTM
from ets.algorithms.teaser import TEASER
from ets.algorithms.strut import STRUT
from ets.algorithms.calimera import CALIMERA
from sktime.datasets import load_from_arff_to_dataframe
from sklearn.preprocessing import LabelEncoder

# Configure the logger (change level to DEBUG for more information)
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger,
                    fmt='%(asctime)s - %(hostname)s - %(name)s[%(process)d] - [%(levelname)s]: %(message)s')

delim_1 = " "
delim_2 = " "


class Config(object):
    def __init__(self):
        self.cv_data: Optional[List[pd.DataFrame]] = None
        self.cv_labels: Optional[pd.DataFrame] = None
        self.train_data: Optional[List[pd.DataFrame]] = None
        self.train_labels: Optional[pd.DataFrame] = None
        self.test_data: Optional[List[pd.DataFrame]] = None
        self.test_labels: Optional[pd.DataFrame] = None
        self.classes: Optional[Set[int]] = None
        self.num_classes: Optional[int] = None
        self.ts_length: Optional[int] = None
        self.variate: Optional[int] = None
        self.strategy: Optional[str] = None
        self.timestamps: Optional[List[int]] = None
        self.folds: Optional[int] = None
        self.target_class: Optional[int] = None
        self.output: Optional[click.File] = None
        self.java: Optional[bool] = None
        self.file: Optional[click.File] = None
        self.splits: Optional[dict] = None
        self.make_cv: Optional[bool] = None
        self.strut: Optional[bool] = None
        self.pyts_csv: Optional[bool] = None
        self.train_file: Optional[str] = None
        self.test_file: Optional[str] = None

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('-i', '--input-cv-file', type=click.Path(exists=True, dir_okay=False),
              help='Input CSV data file for cross-validation.')
@click.option('-t', '--train-file', type=click.Path(exists=True, dir_okay=False),
              help='Train CSV data file.')
@click.option('-e', '--test-file', type=click.Path(exists=True, dir_okay=False),
              help='Test CSV data file.')
@click.option('-s', '--separator', type=click.STRING, default=',', show_default=True,
              help='Separator of the data files.')
@click.option('-d', '--class-idx', type=click.IntRange(min=0),
              help='Class column index of the data files.')
@click.option('-h', '--class-header', type=click.STRING,
              help='Class column header of the data files.')
@click.option('-z', '--zero-replacement', type=click.FLOAT, default=1e-10, show_default=True,
              help='Zero values replacement.')
@click.option('-r', '--reduction', type=click.IntRange(min=1), default=1, show_default=True,
              help='Dimensionality reduction.')
@click.option('-p', '--percentage', type=click.FloatRange(min=0, max=1), multiple=True,
              help='Time-series percentage to be used for early prediction (multiple values can be given).')
@click.option('-v', '--variate', type=click.IntRange(min=1), default=1, show_default=True,
              help='Number of series (attributes) per example.')
@click.option('-g', '--strategy', type=click.Choice(['merge', 'vote', 'normal'], case_sensitive=False),
              help='Multi-variate training strategy.')
@click.option('-f', '--folds', type=click.IntRange(min=2), default=5, show_default=True,
              help='Number of folds for cross-validation.')
@click.option('-c', '--target-class', type=click.INT,
              help='Target class for computing counts. -1 stands for f1 for each class')
@click.option('--java', is_flag=True,
              help='Algorithm implementation in java')
@click.option('--cplus', is_flag=True,
              help='Algorithm implementation in C++')
@click.option('--splits', type=click.Path(exists=True, dir_okay=False), help='Provided fold-indices file'
              )
@click.option('--make-cv', is_flag=True,
              help='If the dataset is divided and cross-validation is wanted'
              )
@click.option('-o', '--output', type=click.File(mode='w'), default='-', required=False,
              help='Results file (if not provided results shall be printed in the standard output).')
@click.option('--trunc', is_flag=True,
              help='Use STRUT approach to find the best time-point to perform ETSC.')
@click.option('--pyts-csv', is_flag=True,
              help='Use pyts format for STRUT Weasel\'s input, when the dataset comes in csv format.')
@pass_config
def cli(config: Config,
        input_cv_file: click.Path,
        train_file: click.Path,
        test_file: click.Path,
        separator: str,
        class_idx: int,
        class_header: str,
        zero_replacement: float,
        reduction: int,
        percentage: List[float],
        variate: int,
        strategy: click.Choice,
        folds: int,
        target_class: int,
        java: bool,
        cplus: bool,
        splits: click.Path,
        output: click.File,
        make_cv: bool,
        trunc: bool,
        pyts_csv: bool) -> None:
        
    """
    Library of Early Time-Series Classification algorithms.
    """

    # Store generic parameters
    config.variate = variate
    config.strategy = strategy
    config.folds = folds
    config.target_class = target_class
    config.output = output
    config.java = java
    config.cplus = cplus
    config.splits = None
    config.make_cv = make_cv
    config.trunc = trunc
    config.pyts_csv = pyts_csv
    # Check if class header or class index is specified.
    if class_header is not None:
        class_column = class_header
    elif class_idx is not None:
        class_column = class_idx
    else:
        logger.error('Either class column header or class column index should be given.')

    file = None
    df = None

    if config.trunc: #In case of STRUT
        
        file = input_cv_file if input_cv_file is not None else train_file
        try:
            if input_cv_file is not None:

                data = pd.read_csv(input_cv_file, header = None)
                
                if config.pyts_csv: 
                    
                    #Parse data for Weasel STRUT and Weasel STRUT FAV when a csv file is provided as input

                    Y = data.iloc[::variate,class_column]
                    data = data.drop(data.columns[class_column], axis=1).values
                    data = np.asarray([data[start::variate,:] for start in range(variate)]).astype('float64')
                    X = np.transpose(data, (1,0,2))

                    config.ts_length = X.shape[2]
                    config.dataset = file.split("/")[-1].split(".")[0]

                    label_encoder = LabelEncoder()
                    Y = label_encoder.fit_transform(Y)
                    
                    if config.make_cv:
                        config.cv_data = X
                        config.cv_labels = pd.Series(Y)
                    else:
                        _, _, Y_TRAIN,  Y_TEST  = train_test_split(X, Y, test_size=0.2,  stratify = Y, random_state=0)
                        config.train_file = X
                        config.test_file = Y
                        config.train_labels = pd.Series(Y_TRAIN)
                        config.test_labels = pd.Series(Y_TEST)
                    

                else:
                    # Parse data for minirocket when a csv file is provided as input
                    for start in range(variate):
                        dim_data = data.iloc[start::variate,:]
                        X = dim_data.drop(dim_data.columns[class_column], axis=1)
                        y = dim_data.iloc[:,class_column]
                        train_X , test_X, train_y,  test_y  = train_test_split(X, y, test_size=0.2,  stratify = y, random_state=0)
                        arff.dump('Dimension'+str(start+1)+'_TRAIN.arff', train_X.values, names=train_X.columns)
                        arff.dump('Dimension'+str(start+1)+'_TEST.arff', test_X.values, names=test_X.columns)

                    config.ts_length = X.shape[1]

                    train_data = [pd.DataFrame(scipy_arff.loadarff('Dimension'+str(i+1)+'_TRAIN.arff')[0]) for i in range(variate)]
                    test_data = [pd.DataFrame(scipy_arff.loadarff('Dimension'+str(i+1)+'_TEST.arff')[0]) for i in range(variate)]

                    reformed_train_data = pd.DataFrame(columns=['dim'+str(i+1) for i in range(variate)])
                    reformed_test_data = pd.DataFrame(columns=['dim'+str(i+1) for i in range(variate)])
                    
                    #construct nested dataframe
                    size = train_data[0].shape[0] #min([x.shape[0] for x in train_data])
                    for i in range(size):
                        reformed_train_data = reformed_train_data.append({'dim'+str(j+1):train_data[j].iloc[i] for j in range(variate)}, ignore_index=True)
                    

                    size = test_data[0].shape[0] #min([x.shape[0] for x in test_data])
                    for i in range(size):
                        reformed_test_data = reformed_test_data.append({'dim'+str(j+1):test_data[j].iloc[i] for j in range(variate)}, ignore_index=True)

                    X = reformed_train_data.append(reformed_test_data)
                    Y = y
                    config.dataset = file.split("/")[-1].split(".")[0]

                    label_encoder = LabelEncoder()
                    Y = label_encoder.fit_transform(Y)
                    
                    if config.make_cv:
                        config.cv_data = X
                        config.cv_labels = pd.Series(Y)
                    else:
                        Y = pd.Series(Y)
                        _, _, Y_TRAIN,  Y_TEST  = train_test_split(X, Y, test_size=0.2,  stratify = Y, random_state=0)
                        config.train_file = X
                        config.test_file = Y
                        config.train_labels = Y_TRAIN
                        config.test_labels = Y_TEST
                    

                    for i in range(variate):
                        os.remove('Dimension'+str(i+1)+'_TRAIN.arff')
                        os.remove('Dimension'+str(i+1)+'_TEST.arff')

            else: # train and test files are seperate .arff files
                if config.make_cv:
                    X_TRAIN, Y_TRAIN = load_from_arff_to_dataframe(train_file)
                    X_TEST, Y_TEST = load_from_arff_to_dataframe(test_file)
                    config.cv_data = X_TRAIN.append(X_TEST)
                    label_encoder = LabelEncoder()
                    config.cv_labels = pd.Series(label_encoder.fit_transform(np.append(Y_TRAIN, Y_TEST)))
                    config.ts_length = config.cv_data.iloc[0,0].size
                    config.dataset = file.split("/")[-1].split(".")[0].split("_")[0]
                    config.train_file = train_file
                    config.test_file  = test_file 
                else:
                    X_TRAIN, Y_TRAIN = load_from_arff_to_dataframe(train_file)
                    X_TEST, Y_TEST = load_from_arff_to_dataframe(test_file)
                    config.train_file = train_file
                    config.test_file  = test_file  
                    label_encoder = LabelEncoder()
                    config.train_labels = pd.Series(label_encoder.fit_transform(load_from_arff_to_dataframe(train_file)[1]))
                    ts_data , config.test_labels = load_from_arff_to_dataframe(test_file)
                    config.test_labels = label_encoder.fit_transform(config.test_labels)
                    config.ts_length = ts_data.iloc[0,0].size
                    config.dataset = file.split("/")[-1].split(".")[0].split("_")[0]

        except pd.errors.ParserError:
            logger.error("Cannot parse file '" + str(file) + "'.")
            sys.exit(-1)

    if config.make_cv and config.trunc == False:
        config.file = train_file
        file = train_file

        try:
            if ".arff" in file:
                train, variate = utils.arff_parser(file)
                config.variate = variate
                # Open the train file and load the time-series data
            else:
                train = pd.read_csv(file, sep=separator, header=None if class_header is None else 0, engine='python')
            print(train.shape)
            file = test_file
            if ".arff" in file:
                test, _ = utils.arff_parser(file)
            else:
                # Open the test file and load the time-series data
                test = pd.read_csv(file, sep=separator, header=None if class_header is None else 0, engine='python')
            print(test.shape)
            df = pd.concat([train, test]).reset_index(drop=True)
            input_cv_file = True
        except pd.errors.ParserError:
            logger.error("Cannot parse file '" + str(file) + "'.")
            sys.exit(-1)
    if config.trunc == False:
        if input_cv_file is not None:
            config.file = input_cv_file
            logger.warning("Found input CSV file for cross-validation. Ignoring options '-t' and '-e'.")
            try:
                if splits is not None:
                    with open(str(splits), "rb") as file:
                        config.splits = pkl.load(file)
                # arff file support
                if not config.make_cv:
                    if ".arff" in config.file:
                        df, variate = utils.arff_parser(file)
                        config.variate = variate
                    # Open the file and load the time-series data
                    else:
                        df = pd.read_csv(input_cv_file, sep=separator, header=None if class_header is None else 0,
                                        engine='python')
                    logger.info("CSV file '" + str(input_cv_file) + "' for CV has dimensions " + str(df.shape) + ".")
                    logger.debug('\n{}'.format(df))
                # Obtain the time-series (replace 0s in order to avoid floating exceptions)
                data = df.drop([class_column], axis=1).replace(0, zero_replacement).T.reset_index(drop=True).T
                # Check if the data frame contains multi-variate time-series examples
                if variate > 1:
                    config.cv_data = list()
                    logger.info('Found ' + str(variate) + ' time-series per example.')
                    for start in range(variate):
                        config.cv_data.append(data.iloc[start::variate].reset_index(drop=True))
                else:
                    config.cv_data = [data]

                # Obtain the labels and compute unique classes
                config.cv_labels = df[class_column].iloc[::variate].reset_index(drop=True)
                config.classes = set(config.cv_labels.unique())
                config.num_classes = len(config.classes)
                logger.info('Found ' + str(config.num_classes) + ' classes: ' + str(config.classes))
                if config.target_class and config.target_class != -1 and config.target_class not in config.classes:
                    logger.error("Target class '" + str(target_class) + "' does not exist in found classes.")
                    sys.exit(1)

                # Store time-series length and define the timestamps used for early prediction
                config.ts_length = data.shape[1]
                if percentage:
                    percentage += (1,)
                    config.timestamps = sorted(list(set([int(p * (config.ts_length - 1)) for p in percentage])))
                    logger.info('Found timestamps ' + str(config.timestamps) + '.')
                else:
                    config.timestamps = range(0, config.ts_length)
                    logger.info(
                        'No percentages found, using all timestamps in range [0,' + str(config.ts_length - 1) + '].')

                if reduction != 1:
                    logger.info("Dimensionality reduction from " + str(config.ts_length) + " to " + str(reduction) + "...")
                    config.cv_data = [utils.df_dimensionality_reduction(df, reduction) for df in config.cv_data]

            except pd.errors.ParserError:
                logger.error("Cannot parse file '" + str(input_cv_file) + "'.")
                sys.exit(-1)

        elif (train_file is not None) and (test_file is not None):
            if splits is not None:
                logger.info("Ignoring the fold indices file provided.")
            try:
                config.file = train_file
                file = train_file

                if ".arff" in file:
                    df, variate = utils.arff_parser(file)
                    config.variate = variate
                # Open the train file and load the time-series data
                else:
                    df = pd.read_csv(file, sep=separator, header=None if class_header is None else 0, engine='python')
                logger.info("CSV file '" + str(file) + "' has dimensions " + str(df.shape) + ".")
                logger.debug('\n{}'.format(df))
                if not config.java:
                    df = (df.sort_values(by=[0])).reset_index(drop=True)
                # Obtain the time-series (replace 0s in order to avoid floating exceptions)
                data = df.drop([class_column], axis=1).replace(0, zero_replacement).T.reset_index(drop=True).T

                # Check if the data frame contains multi-variate time-series examples
                if variate > 1:
                    config.train_data = list()
                    logger.info('Found ' + str(variate) + ' time-series per example.')
                    for start in range(variate):
                        config.train_data.append(data.iloc[start::variate].reset_index(drop=True))
                else:
                    config.train_data = [data]
                # Obtain the train labels and compute unique classes
                config.train_labels = df[class_column].iloc[::variate].reset_index(drop=True)
                config.classes = set(config.train_labels.unique())
                config.num_classes = len(config.classes)
                logger.info('Found ' + str(config.num_classes) + ' classes: ' + str(config.classes))
                if config.target_class and config.target_class not in config.classes and config.target_class != -1:
                    logger.error("Target class '" + str(target_class) + "' does not exist in found classes.")
                    sys.exit(1)

                file = test_file
                if ".arff" in file:
                    df, _ = utils.arff_parser(file)
                else:
                    # Open the test file and load the time-series data
                    df = pd.read_csv(file, sep=separator, header=None if class_header is None else 0, engine='python')
                logger.info("CSV file '" + str(file) + "' has dimensions " + str(df.shape) + ".")
                logger.debug('\n{}'.format(df))
                # Obtain the time-series (replace 0s in order to avoid floating exceptions)
                data = df.drop([class_column], axis=1).replace(0, zero_replacement).T.reset_index(drop=True).T
                # Check if the data frame contains multi-variate time-series examples
                if variate > 1:
                    config.test_data = list()
                    for start in range(variate):
                        config.test_data.append(data.iloc[start::variate].reset_index(drop=True))
                else:
                    config.test_data = [data]

                # Obtain the test labels
                config.test_labels = df[class_column].iloc[::variate].reset_index(drop=True)
                test_classes = set(config.test_labels.unique())

                if config.classes != test_classes:
                    logger.error('Train classes ' + str(config.classes)
                                + ' do not match the test classes ' + str(test_classes) + '.')
                    sys.exit(-1)

                # Store time-series length and define the timestamps used for early prediction
                config.ts_length = data.shape[1]
                if percentage:
                    percentage += (1,)
                    config.timestamps = sorted(list(set([int(p * (config.ts_length - 1)) for p in percentage])))
                    logger.info('Found timestamps ' + str(config.timestamps) + '.')
                else:
                    config.timestamps = range(0, config.ts_length)
                    logger.info(
                        'No percentages found, using all timestamps in range [0,' + str(config.ts_length - 1) + '].')

                if reduction != 1:
                    logger.info("Dimensionality reduction from " + str(config.ts_length) + " to " + str(reduction) + "...")
                    config.train_data = [utils.df_dimensionality_reduction(df, reduction) for df in config.train_data]
                    config.test_data = [utils.df_dimensionality_reduction(df, reduction) for df in config.test_data]

            except pd.errors.ParserError:
                logger.error("Cannot parse file '" + str(file) + "'.")
                sys.exit(-1)
        else:
            logger.error("No input data file provided. "
                        "Use options '-i' / '--input-cv-file' or '-t' / --train-file and 'e' / '--test-file'")
            sys.exit(-1)


@cli.command()
@click.option('-s', '--class-no', type=click.IntRange(min=1), default=20, show_default=True,
              help='Number of classifiers')
@click.option('-n', '--normalize', is_flag=True,
              help='Normalized version of the method')
@pass_config
def teaser(config: Config, class_no: int, normalize: bool) -> None:
    """
    Run 'TEASER' algorithm.
    """
    if (normalize):
        logger.info("Running teaserZ ...")
    else:
        logger.info("Running teaser ...")
    classifier = TEASER(config.timestamps, class_no, normalize)
    if config.cv_data is not None:
        cv(config, classifier)
    else:
        train_and_test(config, classifier)


@cli.command()
@click.option('-u', '--support', type=click.FloatRange(min=0, max=1), default=0.0, show_default=True,
              help='Support threshold.')
@click.option('--relaxed/--no-relaxed', default=False,
              help='Run relaxed ECTS.')
@pass_config
def ects(config: Config, support: float, relaxed: bool) -> None:
    """
     Run 'ECTS' algorithm.
    """

    logger.info("Running ECTS ...")
    classifier = ECTS(config.timestamps, support, relaxed)
    if config.cv_data is not None:
        cv(config, classifier)
    else:
        train_and_test(config, classifier)


@cli.command()
@pass_config
def edsccplus(config: Config) -> None:
    """
     Run 'EDSC' algorithm.
    """
    classifier = EDSC_C(config.timestamps)
    logger.info("Running EDSC with CHE...")
    if config.cv_data is not None:
        cv(config, classifier)
    else:
        train_and_test(config, classifier)


@cli.command()
@click.option('-e', '--earliness', type=click.FloatRange(min=0, max=1), default=0, show_default=True,
              help='Size of prefix')
@click.option('-f', '--folds', type=click.IntRange(min=1), default=1, show_default=True,
              help='Fold for earliness check')
@pass_config
def mlstm(config: Config, earliness, folds) -> None:
    """
    Run 'MLSTM' algorithm.
    """
    logger.info("Running MLSTM ...")
    if earliness == 0:
        classifier = MLSTM(config.timestamps, None, folds)
    else:
        classifier = MLSTM(config.timestamps, [earliness], folds)
    if config.cv_data is not None:
        cv(config, classifier)
    else:
        train_and_test(config, classifier)


@cli.command()
@pass_config
def ecec(config: Config) -> None:
    """
    Run 'ECEC' algorithm.
    """
    logger.info("Running ECEC ...")
    classifier = ECEC(config.timestamps)
    if config.cv_data is not None:
        cv(config, classifier)
    else:
        train_and_test(config, classifier)


@cli.command()
@click.option('-c', '--clusters', type=click.IntRange(min=0), default=3, show_default=True,
              help='Number of clusters')
@click.option('-t', '--cost-time', type=click.FloatRange(min=0, max=1), default=0.001, show_default=True,
              help='Cost time parameter')
@click.option('-l', '--lamb', type=click.FloatRange(min=0), default=100, show_default=True,
              help='Size of prefix')
@click.option('-r', '--random-state', is_flag=True,
              help='Random state')
@pass_config
def economy_k(config: Config, clusters, cost_time, lamb, random_state) -> None:
    """
    Run 'ECONOMY-k' algorithm.
    """
    logger.info("Running ECONOMY-k ...")
    classifier = Trigger(clusters, cost_time, lamb, random_state)
    if config.cv_data is not None:
        cv(config, classifier)
    else:
        train_and_test(config, classifier)

@cli.command()
@click.option('-m', '--method', type=click.Choice(['MINIROCKET', 'WEASEL', 'MINIROCKET_FAV', 'WEASEL_FAV'], case_sensitive=False) , default=3, show_default=True,
              help='Method variant to perform ETSC')
@click.option('-p', '--optimize', type=click.IntRange(min=0, max=2), default=0, show_default=True,
              help='Metric to optimize: 0 - accuracy, 1 - F1-score, 2 - harmonic mean') 
@click.option('-s', '--splits', type=click.IntRange(min=2), default=2, show_default=True,
              help='Number of splits')
@click.option('-i', '--class_imbalance', is_flag=True,
              help='Class imbalance')
@pass_config
def strut(config: Config, method, optimize, splits, class_imbalance) -> None: #maybe add method
    """
    Run 'STRUT' algorithm.
    """
    logger.info("Running "+ str(method).replace('_', ' ') + " ...")
    classifier = STRUT(config.timestamps, config.ts_length, config.variate, optimize = optimize, tsc_method = method, n_splits = splits, class_imbalance = class_imbalance,  dataset = config.dataset)
    #remove result files if exist
    if os.path.exists('results/'+config.dataset+'_metric_scores/'+config.dataset+'_'+method.lower()+'_accuracy.txt'):
        os.remove('results/'+config.dataset+'_metric_scores/'+config.dataset+'_'+method.lower()+'_accuracy.txt')
    if os.path.exists('results/'+config.dataset+'_metric_scores/'+config.dataset+'_'+method.lower()+'_f1_score.txt'):
        os.remove('results/'+config.dataset+'_metric_scores/'+config.dataset+'_'+method.lower()+'_f1_score.txt')
    if os.path.exists('results/'+config.dataset+'_metric_scores/'+config.dataset+'_'+method.lower()+'_harmonic_mean.txt'):
        os.remove('results/'+config.dataset+'_metric_scores/'+config.dataset+'_'+method.lower()+'_harmonic_mean.txt')
    if config.cv_data is not None:
        cv(config, classifier)
    else:
        train_and_test(config, classifier)

@cli.command()
@click.option('-dp', '--delaypenalty', type=click.FLOAT, default=1.0, show_default=True,
              help='alpha parameter in the time delay function')
@pass_config
def calimera(config: Config, delaypenalty: float) -> None:
    """
    Run 'CALIMERA' algorithm.
    """
    logger.info("Running CALIMERA ...")
    classifier = CALIMERA(config.timestamps, delaypenalty)
    if config.cv_data is not None:
        cv(config, classifier)
    else:
        train_and_test(config, classifier)


def determine_voting_result(votes):
    req_votes = math.ceil(len(votes)/2)
    winner = []
    for ii in range(len(votes[0])):
        temp_votes = []
        temp_classes = []
        for j in range(len(votes)):
            temp_votes.append(votes[j][ii])
            temp_classes.append(votes[j][ii][1])
        g = sorted(temp_votes)
        c = Counter(temp_classes).keys()
        if len(c) == 1: # if all variates have the same label then take the middle one
            winner.append(g[req_votes-1])
        else: # else find the most early and common
            cnt = 0
            found = 0
            pl = {key: 0 for key in c}
            while cnt<len(votes):
                pl[g[cnt][1]]+=1
                if pl[g[cnt][1]]>=req_votes:
                    winner.append(g[cnt])
                    break
                cnt=cnt+1
            # In case we have been so far inconclusive, consider the latest [interval,class label]
            if req_votes not in pl.values():
                winner.append(g[-1])
    return(winner)


def cv(config: Config, classifier: EarlyClassifier) -> None:
    sum_accuracy, sum_earliness, sum_precision, sum_recall, sum_f1 = 0, 0, 0, 0, 0
    all_predictions: List[Tuple[int, int]] = list()
    all_labels: List[int] = list()
    if config.splits:
        ind = []
        for key in config.splits.keys():
            ind.append((config.splits[key][0], config.splits[key][1]))
        indices = zip(ind, range(1, config.folds + 1))
    else:
        print("Folds : {}".format(config.folds))
        if config.trunc:
            if (classifier.tsc_method == 'WEASEL_FAV' or classifier.tsc_method == 'WEASEL') and  config.pyts_csv == False:
                #convert nested dataframe to pyts compatible format 
                    train_data = pd.DataFrame(scipy_arff.loadarff(config.train_file)[0])
                    test_data = pd.DataFrame(scipy_arff.loadarff(config.test_file)[0])
                    data = train_data.append(test_data) 
                    X = data.iloc[:,:-1]
                    X =  classifier.data_reform(X)
                    Y = data.iloc[: ,-1].values.ravel() 
                    config.cv_data = np.nan_to_num(X)
                    label_encoder = LabelEncoder()
                    config.cv_labels = pd.Series(label_encoder.fit_transform(Y))
            
            indices = zip(StratifiedKFold(config.folds).split(config.cv_data, config.cv_labels),
                      range(1, config.folds + 1))

        else:
            indices = zip(StratifiedKFold(config.folds).split(config.cv_data[0], config.cv_labels),
                      range(1, config.folds + 1))
    count = 0

    for ((train_indices, test_indices), i) in indices:
        predictions = []
        count += 1
        click.echo('== Fold ' + str(i), file=config.output)
        if config.trunc == False and (config.variate == 1 or config.strategy == 'merge' or config.strategy == 'normal'):

            """ Merge is a method that turns a multivariate time-series to a univariate """
            if config.variate > 1 and config.strategy == 'merge':
                logger.info("Merging multivariate time-series ...")
                config.cv_data = [utils.df_merge(config.cv_data)]
            """ Normal is used for algorithms that support multivariate time-series """
            if config.variate > 1 and config.strategy == 'normal':
                fold_train_data = [config.cv_data[i].iloc[train_indices].reset_index(drop=True) for i in
                                   range(0, config.variate)]
                fold_test_data = [config.cv_data[i].iloc[test_indices].reset_index(drop=True) for i in
                                  range(0, config.variate)]
                fold_train_labels = config.cv_labels[train_indices].reset_index(drop=True)

            else:
                fold_train_data = config.cv_data[0].iloc[train_indices].reset_index(drop=True)
                fold_train_labels = config.cv_labels[train_indices].reset_index(drop=True)
                fold_test_data = config.cv_data[0].iloc[test_indices].reset_index(drop=True)

            """In case we call algorithms implemented in Java (TEASER, ECTS)"""
            if config.java is True:

                temp = pd.concat([fold_train_labels, fold_train_data], axis=1, sort=False)
                temp.to_csv('train', index=False, header=False, sep=delim_1)

                temp2 = pd.concat([config.cv_labels[test_indices].reset_index(drop=True), fold_test_data], axis=1,
                                  sort=False)
                temp2.to_csv('test', index=False, header=False, sep=delim_2)
                res = classifier.predict(pd.DataFrame())
                predictions = res[0]
                click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                           file=config.output)
                click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                           file=config.output)


            elif config.cplus is True:
                """In case the method is implemented in C++ (EDSC)"""

                fold_test_labels = config.cv_labels[test_indices].reset_index(drop=True)

                classifier.train(fold_train_data, fold_train_labels)
                a = fold_train_labels.value_counts()
                a = a.sort_index(ascending=False)

                # The EDSC method returns the tuple (predictions, train time, test time)
                res = classifier.predict2(test_data=fold_test_data, labels=fold_train_labels, numbers=a, types=0)
                predictions = res[0]
                click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                           file=config.output)
                click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                           file=config.output)


            elif config.strategy == "normal":
                if isinstance(fold_train_data, pd.DataFrame):
                    fold_train_data = [fold_train_data]
                    fold_test_data = [fold_test_data]
                # Train the MLSTM
                result = classifier.true_predict(fold_train_data, fold_test_data, fold_train_labels)
                predictions = result[0]
                click.echo('Total training time := {}'.format(timedelta(seconds=result[1])), file=config.output)
                click.echo('Total testing time := {}'.format(timedelta(seconds=result[2])), file=config.output)
                click.echo('Best earl:={}'.format(result[3]), file=config.output)
                click.echo('Best cells:={}'.format(result[4]), file=config.output)
  

            else:
                """ For ECTS and CALIMERA methods """
                # Train the classifier
                start = time.time()
                classifier.train(fold_train_data, fold_train_labels)
                click.echo('Total training time := {}'.format(timedelta(seconds=time.time() - start)),
                           file=config.output)

                # Make predictions
                start = time.time()
                predictions = classifier.predict(fold_test_data)
                click.echo('Total testing time := {}'.format(timedelta(seconds=time.time() - start)),
                           file=config.output)
        
        elif config.trunc:
            if classifier.tsc_method == 'MINIROCKET':
                predictions, training_time, test_time, earliness = classifier.minirocket_strut((config.cv_data.iloc[train_indices], config.cv_labels.iloc[train_indices]),(config.cv_data.iloc[test_indices], config.cv_labels.iloc[test_indices]))
            elif classifier.tsc_method == 'WEASEL':
                predictions, training_time, test_time, earliness = classifier.weasel_strut((config.cv_data[train_indices], config.cv_labels[train_indices]),(config.cv_data[test_indices], config.cv_labels[test_indices]))
            elif classifier.tsc_method == 'MINIROCKET_FAV':
                predictions, training_time, test_time, earliness = classifier.minirocket_strut_fav((config.cv_data.iloc[train_indices], config.cv_labels.iloc[train_indices]),(config.cv_data.iloc[test_indices], config.cv_labels.iloc[test_indices]))
            elif classifier.tsc_method == 'WEASEL_FAV':
                predictions, training_time, test_time, earliness = classifier.weasel_strut_fav((config.cv_data[train_indices], config.cv_labels[train_indices]),(config.cv_data[test_indices], config.cv_labels[test_indices]))
            else:
                print("Unsupported method")
                return  
            click.echo('Total training time := {}'.format(timedelta(seconds=training_time)), file=config.output)
            click.echo('Total testing time := {}'.format(timedelta(seconds=test_time)), file=config.output)
            click.echo('Best earl:={}'.format(earliness), file=config.output)

        else:

            """In case of a multivariate cv dataset is passed on one of the univariate based approaches"""
            votes = []
            for ii in range(config.variate):

                fold_train_data = config.cv_data[ii].iloc[train_indices].reset_index(drop=True)
                fold_train_labels = config.cv_labels[train_indices].reset_index(drop=True)
                fold_test_data = config.cv_data[ii].iloc[test_indices].reset_index(drop=True)

                if config.java is True:
                    """ For the java approaches"""
                    temp = pd.concat([fold_train_labels, fold_train_data], axis=1, sort=False)
                    temp.to_csv('train', index=False, header=False, sep=delim_1)

                    temp2 = pd.concat([config.cv_labels[test_indices].reset_index(drop=True), fold_test_data], axis=1,
                                      sort=False)
                    temp2.to_csv('test', index=False, header=False, sep=delim_2)
                    res = classifier.predict(pd.DataFrame())  # The java methods return the tuple (predictions,
                    # train time, test time)

                    click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                               file=config.output)
                    click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                               file=config.output)
                    votes.append(res[0])

                elif config.cplus is True:

                    fold_test_labels = config.cv_labels[test_indices].reset_index(drop=True)
                    classifier.train(fold_train_data, fold_train_labels)
                    a = fold_train_labels.value_counts()
                    a = a.sort_index(ascending=False)

                    # The EDSC method returns the tuple (predictions, train time, test time)
                    res = classifier.predict2(test_data=fold_test_data, labels=fold_test_labels, numbers=a, types=0)

                    click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                               file=config.output)
                    click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                               file=config.output)
                    votes.append(res[0])
                else:
                    # Train the classifier
                    start = time.time()
                    classifier.train(fold_train_data, fold_train_labels)
                    click.echo('Total training time := {}'.format(timedelta(seconds=time.time() - start)),
                               file=config.output)

                    # Make predictions
                    start = time.time()
                    votes.append(classifier.predict(fold_test_data))
                    click.echo('Total testing time := {}'.format(timedelta(seconds=time.time() - start)),
                               file=config.output)

            # Make predictions from the votes of each test example
            # click.echo('votes: '+str(votes))
            # for ii in range(len(votes[0])):
            #     max_timestamp = max(map(lambda x: x[ii][0], votes))
            #     most_predicted = Counter(map(lambda x: x[ii][1], votes)).most_common(1)[0][0]
            #     predictions.append((max_timestamp, most_predicted))
            winners = determine_voting_result(votes)
            for i in range(len(votes[0])):
                predictions.append((winners[i][0],winners[i][1]))
        all_predictions.extend(predictions)
        if config.trunc:
            all_labels.extend(config.cv_labels[test_indices])
            accuracy = utils.accuracy(predictions, config.cv_labels[test_indices].tolist())
            #all_labels.extend(config.cv_labels.reindex(test_indices))
            #accuracy = utils.accuracy(predictions, config.cv_labels.reindex(test_indices).tolist())
            #config.cv_labels.loc[config.cv_labels.index.intersection(test_indices)]

        else:
            all_labels.extend(config.cv_labels[test_indices])
            accuracy = utils.accuracy(predictions, config.cv_labels[test_indices].tolist())
        
        # Calculate accuracy and earliness
        
        sum_accuracy += accuracy
        earliness = utils.earliness(predictions, config.ts_length - 1)
        sum_earliness += earliness
        click.echo('Accuracy: ' + str(round(accuracy, 4)) + ' Earliness: ' + str(round(earliness * 100, 4)) + '%',
                   file=config.output)
        # Calculate counts, precision, recall and f1-score if a target class is provided
        if config.target_class == -1:
            if config.trunc:
                #items = config.cv_labels.reindex(train_indices).unique() #[1:] to ignore nan as class
                items = config.cv_labels[train_indices].unique()
            else:
                items = config.cv_labels[train_indices].unique() #[1:] to ignore nan as class
            for item in items:
                click.echo('For the class: ' + str(item), file=config.output)
                if config.trunc:
                    #tp, tn, fp, fn = utils.counts(item, predictions, config.cv_labels.reindex(test_indices).tolist())
                    tp, tn, fp, fn = utils.counts(item, predictions, config.cv_labels[test_indices].tolist()) 
                else:
                   tp, tn, fp, fn = utils.counts(item, predictions, config.cv_labels[test_indices].tolist()) 
                click.echo('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn),
                           file=config.output)
                precision = utils.precision(tp, fp)
                click.echo('Precision: ' + str(round(precision, 4)), file=config.output)
                recall = utils.recall(tp, fn)
                click.echo('Recall: ' + str(round(recall, 4)), file=config.output)
                f1 = utils.f_measure(tp, fp, fn)
                click.echo('F1-score: ' + str(round(f1, 4)) + "\n", file=config.output)
        elif config.target_class:
            if config.trunc:
                #tp, tn, fp, fn = utils.counts(config.target_class, predictions, config.cv_labels.reindex(test_indices).tolist())
                tp, tn, fp, fn = utils.counts(config.target_class, predictions, config.cv_labels[test_indices].tolist())
            else:
                tp, tn, fp, fn = utils.counts(config.target_class, predictions, config.cv_labels[test_indices].tolist())
            click.echo('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn), file=config.output)
            precision = utils.precision(tp, fp)
            sum_precision += precision
            click.echo('Precision: ' + str(round(precision, 4)), file=config.output)
            recall = utils.recall(tp, fn)
            sum_recall += recall
            click.echo('Recall: ' + str(round(recall, 4)), file=config.output)
            f1 = utils.f_measure(tp, fp, fn)
            sum_f1 += f1
            click.echo('F1-score: ' + str(round(f1, 4)), file=config.output)
        click.echo('Predictions' + str(predictions), file=config.output)
    click.echo('== Macro-average', file=config.output)
    macro_accuracy = sum_accuracy / config.folds
    macro_earliness = sum_earliness / config.folds
    click.echo('Accuracy: ' + str(round(macro_accuracy, 4)) +
               ' Earliness: ' + str(round(macro_earliness * 100, 4)) + '%',
               file=config.output)

    if config.target_class and config.target_class != -1:
        macro_precision = sum_precision / config.folds
        macro_recall = sum_recall / config.folds
        macro_f1 = sum_f1 / config.folds
        click.echo('Precision: ' + str(round(macro_precision, 4)), file=config.output)
        click.echo('Recall: ' + str(round(macro_recall, 4)), file=config.output)
        click.echo('F1-score: ' + str(round(macro_f1, 4)), file=config.output)

    click.echo('== Micro-average:', file=config.output)
    micro_accuracy = utils.accuracy(all_predictions, all_labels)
    micro_earliness = utils.earliness(all_predictions, config.ts_length - 1)
    click.echo('Accuracy: ' + str(round(micro_accuracy, 4)) +
               ' Earliness: ' + str(round(micro_earliness * 100, 4)) + '%',
               file=config.output)

    # Calculate counts, precision, recall and f1-score if a target class is provided
    if config.target_class and config.target_class != -1:
        tp, tn, fp, fn = utils.counts(config.target_class, all_predictions, all_labels)
        click.echo('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn), file=config.output)
        precision = utils.precision(tp, fp)
        click.echo('Precision: ' + str(round(precision, 4)), file=config.output)
        recall = utils.recall(tp, fn)
        click.echo('Recall: ' + str(round(recall, 4)), file=config.output)
        f1 = utils.f_measure(tp, fp, fn)
        click.echo('F1-score: ' + str(round(f1, 4)), file=config.output)


def train_and_test(config: Config, classifier: EarlyClassifier) -> None:
    predictions = []

    if config.variate == 1 or config.strategy == 'merge' or config.strategy == 'normal' or config.trunc == True:
        predictions = []
        if config.variate > 1 and config.strategy != "normal" and config.trunc == False:
            logger.info("Merging multivariate time-series ...")
            config.train_data = [utils.df_merge(config.train_data)]
            config.test_data = [utils.df_merge(config.test_data)]

        if config.java is True:
            config.train_labels = config.train_labels.astype(int)
            temp = pd.concat([config.train_labels.reset_index(drop=True), config.train_data[0].reset_index(drop=True)],
                             axis=1, sort=False)
            temp.to_csv('train', index=False, header=False, sep=delim_1)
            temp2 = pd.concat([config.test_labels.reset_index(drop=True), config.test_data[0].reset_index(drop=True)],
                              axis=1, sort=False)
            temp2.to_csv('test', index=False, header=False, sep=delim_2)
            res = classifier.predict(pd.DataFrame())
            predictions = res[0]

            click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                       file=config.output)
            click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                       file=config.output)

        elif config.cplus is True:

            a = config.train_labels.value_counts()
            a = a.sort_index()
            classifier.train(config.train_data[0], config.train_labels)

            res = classifier.predict2(test_data=config.test_data[0], labels=config.test_labels, numbers=a, types=1)
            predictions = res[0]
            click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                       file=config.output)
            click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                       file=config.output)

        elif config.strategy == 'normal':

            result = classifier.true_predict(config.train_data, config.test_data, config.train_labels)
            predictions = result[0]
            click.echo('Total training time := {}'.format(timedelta(seconds=result[1])), file=config.output)
            click.echo('Total testing time := {}'.format(timedelta(seconds=result[2])), file=config.output)
            click.echo('Best earl:={}'.format(result[3]), file=config.output)
            click.echo('Best cells:={}'.format(result[4]), file=config.output)

        elif config.trunc == True:
            if classifier.tsc_method == 'MINIROCKET':
                predictions, training_time, test_time, earliness = classifier.minirocket_strut(config.train_file, config.test_file)
            elif classifier.tsc_method == 'WEASEL':
                predictions, training_time, test_time, earliness = classifier.weasel_strut(config.train_file, config.test_file)
            elif classifier.tsc_method == 'MINIROCKET_FAV':
                predictions, training_time, test_time, earliness = classifier.minirocket_strut_fav(config.train_file, config.test_file)
            elif classifier.tsc_method == 'WEASEL_FAV':
                predictions, training_time, test_time, earliness = classifier.weasel_strut_fav(config.train_file, config.test_file)
            else:
                print("Unsupported method")
                return 
            click.echo('Total training time := {}'.format(timedelta(seconds=training_time)), file=config.output)
            click.echo('Total testing time := {}'.format(timedelta(seconds=test_time)), file=config.output)
            click.echo('Best earl:={}'.format(earliness), file=config.output)
            
        else:
            # Train the classifier
            start = time.time()
            trip = classifier.train(config.train_data[0], config.train_labels)
            click.echo('Total training time := {}'.format(timedelta(seconds=time.time() - start)), file=config.output)

            # Make predictions
            start = time.time()
            predictions = classifier.predict(config.test_data[0])
            click.echo('Total testing time := {}'.format(timedelta(seconds=time.time() - start)), file=config.output)

    else:
        logger.info("Voting over the multivariate time-series attributes ...")

        votes = []
        for i in range(config.variate):
            if config.java is True:
                temp = pd.concat([config.train_labels, config.train_data[i]], axis=1, sort=False)
                temp.to_csv('train', index=False, header=False, sep=delim_1)
                temp2 = pd.concat([config.test_labels, config.test_data[i]], axis=1, sort=False)

                temp2.to_csv('test', index=False, header=False, sep=delim_2)
                res = classifier.predict(pd.DataFrame())

                votes.append(res[0])
                click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                           file=config.output)
                click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                           file=config.output)

            elif config.cplus is True:
                a = config.train_labels.value_counts()
                a = a.sort_index()

                classifier.train(config.train_data[i], config.train_labels)
                res = classifier.predict2(test_data=config.test_data[i], labels=config.test_labels, numbers=a, types=1)
                votes.append(res[0])
                click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                           file=config.output)
                click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                           file=config.output)
            else:
                # Train the classifier
                start = time.time()
                trip = classifier.train(config.train_data[i], config.train_labels)
                click.echo('Total training time := {}'.format(timedelta(seconds=time.time() - start)),
                           file=config.output)

                # Make predictions
                start = time.time()
                votes.append(classifier.predict(config.test_data[i]))
                click.echo('Total testing time := {}'.format(timedelta(seconds=time.time() - start)),
                           file=config.output)

        # Make predictions from the votes of each test example
        # click.echo('votes: '+str(votes))
        # for i in range(len(votes[0])):
        #     max_timestamp = max(map(lambda x: x[i][0], votes))
        #     most_predicted = Counter(map(lambda x: x[i][1], votes)).most_common(1)[0][0]
        #     predictions.append((max_timestamp, most_predicted))
        winners = determine_voting_result(votes)
        for i in range(len(votes[0])):
            predictions.append((winners[i][0],winners[i][1]))

    accuracy = utils.accuracy(predictions, config.test_labels.tolist())
    earliness = utils.earliness(predictions, config.ts_length - 1)
    harmonic = utils.harmonic_mean(accuracy, earliness)
    click.echo('Accuracy: ' + str(round(accuracy, 4)) + ' Earliness: ' + str(round(earliness * 100, 4)) + '%',
               file=config.output)
    click.echo('Harmonic mean: ' + str(round(harmonic, 4)),
               file=config.output)

    # Calculate counts, precision, recall and f1-score if a target class is provided
    if config.target_class == -1:
        items = config.train_labels.unique()
        for item in items:
            click.echo('For the class: ' + str(item), file=config.output)
            config.target_class = item
            tp, tn, fp, fn = utils.counts(config.target_class, predictions, config.test_labels)
            click.echo('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn), file=config.output)
            precision = utils.precision(tp, fp)
            click.echo('Precision: ' + str(round(precision, 4)), file=config.output)
            recall = utils.recall(tp, fn)
            click.echo('Recall: ' + str(round(recall, 4)), file=config.output)
            f1 = utils.f_measure(tp, fp, fn)
            click.echo('F1-score: ' + str(round(f1, 4)) + "\n", file=config.output)
    elif config.target_class:
        tp, tn, fp, fn = utils.counts(config.target_class, predictions, config.test_labels)
        click.echo('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn), file=config.output)
        precision = utils.precision(tp, fp)
        click.echo('Precision: ' + str(round(precision, 4)), file=config.output)
        recall = utils.recall(tp, fn)
        click.echo('Recall: ' + str(round(recall, 4)), file=config.output)
        f1 = utils.f_measure(tp, fp, fn)
        click.echo('F1-score: ' + str(round(f1, 4)), file=config.output)
    click.echo('Predictions' + str(predictions), file=config.output)


if __name__ == '__main__':
    cli()
