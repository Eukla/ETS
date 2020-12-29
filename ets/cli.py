import logging
import sys
import time
from collections import Counter
from datetime import timedelta
from typing import Set, List, Tuple, Optional

import click
import coloredlogs
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import ets.algorithms.utils as utils
from ets.algorithms.early_classifier import EarlyClassifier
from ets.algorithms.ecec import ECEC
from ets.algorithms.ects import ECTS
from ets.algorithms.edsc_c import EDSC_C
from ets.algorithms.mlstm import MLSTM
from ets.algorithms.teaser import TEASER

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
@click.option('-o', '--output', type=click.File(mode='w'), default='-', required=False,
              help='Results file (if not provided results shall be printed in the standard output).')
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
        output: click.File) -> None:
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
    # Check if class header or class index is specified.
    if class_header is not None:
        class_column = class_header
    elif class_idx is not None:
        class_column = class_idx
    else:
        logger.error('Either class column header or class column index should be given.')
        sys.exit(-1)

    file = None
    if input_cv_file is not None:
        config.file = input_cv_file
        logger.warning("Found input CSV file for cross-validation. Ignoring options '-t' and '-e'.")
        try:
            # Open the file and load the time-series data
            df = pd.read_csv(input_cv_file, sep=separator, header=None if class_header is None else 0, engine='python')
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
            if config.target_class and config.target_class not in config.classes:
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
        try:
            config.file = train_file
            file = train_file
            # Open the train file and load the time-series data
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
            config.train_labels = df[class_column]
            config.classes = set(config.train_labels.unique())
            config.num_classes = len(config.classes)
            logger.info('Found ' + str(config.num_classes) + ' classes: ' + str(config.classes))
            if config.target_class and config.target_class not in config.classes and config.target_class != -1:
                logger.error("Target class '" + str(target_class) + "' does not exist in found classes.")
                sys.exit(1)

            file = test_file

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
            config.test_labels = df[class_column]
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
@pass_config
def teaser(config: Config, class_no: int) -> None:
    """
    Run 'TEASER' algorithm.
    """
    logger.info("Running teaser ...")
    classifier = TEASER(config.timestamps, class_no)
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
@pass_config
def mlstm(config: Config, earliness) -> None:
    """
    Run 'MLSTM' algorithm.
    """
    logger.info("Running MLSTM ...")
    if earliness == 0:
        classifier = MLSTM(config.timestamps, None)
    else:
        classifier = MLSTM(config.timestamps, [earliness])
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


def cv(config: Config, classifier: EarlyClassifier) -> None:
    sum_accuracy, sum_earliness, sum_precision, sum_recall, sum_f1 = 0, 0, 0, 0, 0
    predictions = []
    all_predictions: List[Tuple[int, int]] = list()
    all_labels: List[int] = list()
    my_dict = {}
    indices = zip(StratifiedKFold(config.folds).split(config.cv_data[0], config.cv_labels),
                  range(1, config.folds + 1))

    for ((train_indices, test_indices), i) in indices:

        click.echo('== Fold ' + str(i), file=config.output)
        if config.variate == 1 or config.strategy == 'merge' or config.strategy == 'normal':

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
                fold_test_labels = config.cv_labels[test_indices].reset_index(drop=True)

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
                predictions = classifier.predict(pd.DataFrame())
            elif config.cplus is True:
                fold_test_labels = config.cv_labels[test_indices].reset_index(drop=True)

                classifier.train(fold_train_data, fold_train_labels)
                a = fold_train_labels.value_counts()
                a = a.sort_index(ascending=False)
                res = classifier.predict2(test_data=fold_test_data, labels=fold_train_labels, numbers=a,types=0)
                predictions = res[0]
                click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                           file=config.output)
                click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                           file=config.output)
            else:
                # Train the MLSTM
                start = time.time()
                result = classifier.true_predict(fold_train_data, fold_test_data, fold_train_labels,
                                                 fold_test_labels)
                predictions = result[0]
                click.echo('Total training time := {}'.format(timedelta(seconds=result[1])), file=config.output)
                click.echo('Total testing time := {}'.format(timedelta(seconds=result[2])), file=config.output)
                click.echo('Best earl:={}'.format(result[3]), file=config.output)
                # predictions = classifier.predict(fold_test_data)
                # click.echo('Total testing time := {}'.format(timedelta(seconds=time.time() - start)),
                #           file=config.output)
        else:
            votes = []
            for ii in range(config.variate):
                fold_train_data = config.cv_data[ii].iloc[train_indices].reset_index(drop=True)
                fold_train_labels = config.cv_labels[train_indices].reset_index(drop=True)
                fold_test_data = config.cv_data[ii].iloc[test_indices].reset_index(drop=True)
                if config.java is True:
                    temp = pd.concat([fold_train_labels, fold_train_data], axis=1, sort=False)
                    temp.to_csv('train', index=False, header=False, sep=delim_1)
                    temp2 = pd.concat([config.cv_labels[test_indices].reset_index(drop=True), fold_test_data], axis=1,
                                      sort=False)
                    temp2.to_csv('test', index=False, header=False, sep=delim_2)
                    res = classifier.predict(pd.DataFrame())
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
                    res = classifier.predict2(test_data=fold_test_data, labels=fold_test_labels, numbers=a,types=0)
                    votes.append(res[0])
                    click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                               file=config.output)
                    click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                               file=config.output)
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
            for ii in range(len(votes[0])):
                max_timestamp = max(map(lambda x: x[ii][0], votes))
                most_predicted = Counter(map(lambda x: x[ii][1], votes)).most_common(1)[0][0]
                predictions.append((max_timestamp, most_predicted))

        all_predictions.extend(predictions)
        all_labels.extend(config.cv_labels[test_indices])

        # Calculate accuracy and earliness
        accuracy = utils.accuracy(predictions, config.cv_labels[test_indices].tolist())
        # my_dict = utils.results_organize(predictions, config.cv_labels[test_indices].tolist(), my_dict, test_indices)
        sum_accuracy += accuracy
        earliness = utils.earliness(predictions, config.ts_length - 1)
        sum_earliness += earliness
        click.echo('Accuracy: ' + str(round(accuracy, 4)) + ' Earliness: ' + str(round(earliness * 100, 4)) + '%',
                   file=config.output)

        # Calculate counts, precision, recall and f1-score if a target class is provided
        if config.target_class:
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
    # utils.results_smart_print(my_dict, config.output)
    click.echo('== Macro-average', file=config.output)
    macro_accuracy = sum_accuracy / config.folds
    macro_earliness = sum_earliness / config.folds
    click.echo('Accuracy: ' + str(round(macro_accuracy, 4)) +
               ' Earliness: ' + str(round(macro_earliness * 100, 4)) + '%',
               file=config.output)

    if config.target_class:
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
            click.echo('F1-score: ' + str(round(f1, 4)), file=config.output)
    elif config.target_class:
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

    if config.variate == 1 or config.strategy == 'merge' or config.strategy == 'normal':
        if config.variate > 1:
            logger.info("Merging multivariate time-series ...")
            config.train_data = [utils.df_merge(config.train_data)]
            config.test_data = [utils.df_merge(config.test_data)]
        if config.java is True:
            temp = pd.concat([config.train_labels, config.train_data[0]], axis=1, sort=False)
            temp.to_csv('train', index=False, header=False, sep=delim_1)
            temp2 = pd.concat([config.test_labels, config.test_data[0]], axis=1, sort=False)
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
            res = classifier.predict2(test_data=config.test_data[0], labels=config.test_labels, numbers=a,types=1)
            predictions = res[0]
            click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                       file=config.output)
            click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                       file=config.output)
        elif config.strategy == 'normal':
            start = time.time()
            # click.echo('Total training time := {}'.format(timedelta(seconds=time.time() - start)),
            #           file=config.output)

            # Make predictions
            start = time.time()
            result = classifier.true_predict(config.train_data, config.test_data, config.train_labels,
                                             config.test_labels)
            predictions = result[0]
            click.echo('Total training time := {}'.format(timedelta(seconds=result[1])), file=config.output)
            click.echo('Total testing time := {}'.format(timedelta(seconds=result[2])), file=config.output)
            click.echo('Best earl:={}'.format(result[3]), file=config.output)
        else:
            # Train the classifier
            start = time.time()
            trip = classifier.train(config.train_data[0], config.train_labels)
            click.echo('Total training time := {}'.format(timedelta(seconds=time.time() - start)), file=config.output)

            # Make predictions
            start = time.time()
            predictions = classifier.predict(config.test_data[0])
            # classifier.graphs(config.test_data[0],trip)
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
                res = classifier.predict2(test_data=config.test_data[0], labels=config.test_labels, numbers=a,types=1)
                votes.append(res[0])
                click.echo('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                        file=config.output)
                click.echo('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                        file=config.output)
            elif config.strategy == 'normal':
                start = time.time()
                # click.echo('Total training time := {}'.format(timedelta(seconds=time.time() - start)),
                #           file=config.output)

                # Make predictions
                start = time.time()
                result = classifier.true_predict(config.train_data, config.test_data, config.train_labels,
                                                config.test_labels)
                votes.append(result[0])
                click.echo('Total training time := {}'.format(timedelta(seconds=result[1])), file=config.output)
                click.echo('Total training time := {}'.format(timedelta(seconds=result[2])), file=config.output)
                click.echo('Best earl:={}'.format(result[3]), file=config.output)
            else:
                # Train the classifier
                start = time.time()
                trip = classifier.train(config.train_data[i], config.train_labels)
                click.echo('Total training time := {}'.format(timedelta(seconds=time.time() - start)), file=config.output)

                # Make predictions
                start = time.time()
                votes.append(classifier.predict(config.test_data[i]))
                # classifier.graphs(config.test_data[0],trip)
                click.echo('Total testing time := {}'.format(timedelta(seconds=time.time() - start)), file=config.output)

        # Make predictions from the votes of each test example
        for i in range(len(votes[0])):
            max_timestamp = max(map(lambda x: x[i][0], votes))
            most_predicted = Counter(map(lambda x: x[i][1], votes)).most_common(1)[0][0]
            predictions.append((max_timestamp, most_predicted))

    # Calculate accuracy and earliness
    # acc = utils.temp_accuracy(predictions, config.test_labels.tolist())
    # print(acc)
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
            click.echo('Predictions' + str(predictions), file=config.output)
    elif config.target_class:
        tp, tn, fp, fn = utils.counts(config.target_class, predictions, config.test_labels)
        click.echo('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn), file=config.output)
        precision = utils.precision(tp, fp)
        click.echo('Precision: ' + str(round(precision, 4)), file=config.output)
        recall = utils.recall(tp, fn)
        click.echo('Recall: ' + str(round(recall, 4)), file=config.output)
        f1 = utils.f_measure(tp, fp, fn)
        click.echo('F1-score: ' + str(round(f1, 4)), file=config.output)


if __name__ == '__main__':
    cli()
