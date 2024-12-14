import sys
sys.path.append('/mnt/d/maaz_work/ETSC-master (3)/ETSC-master')  # Ensure this is the path to the ETS-master 2 directory

import logging
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin

import click
from ets.algorithms.utils import accuracy, earliness
from ets.algorithms.strut import STRUT
from ets.algorithms.xcm import xcm  # Assuming xcm.py is in the correct path


# Manually implement KerasClassifier
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, **kwargs):
        self.build_fn = build_fn
        self.model = None
        self.kwargs = kwargs

    def fit(self, X, y, **fit_args):
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, **fit_args)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Example of usage:
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Configure the logger
logger = logging.getLogger(__name__)

# Configuration for the CLI
class Config:
    def __init__(self):
        self.file = None
        self.splits = None
        self.make_cv = None
        self.strut = None
        self.pyts_csv = None
        self.train_file = None
        self.test_file = None
        self.dataset = None
        self.cv_data = None
        self.cv_labels = None
        self.ts_length = None
        self.trunc = False

pass_config = click.make_pass_decorator(Config, ensure=True)

# CLI to handle command-line arguments and execution
@click.command()
@click.option('-i', '--input_train_file', type=click.Path(), required=True, help='Input train file.')
@click.option('-t', '--input_test_file', type=click.Path(), required=True, help='Input test file.')
@click.option('-m', '--method', type=click.Choice(['MINIROCKET', 'WEASEL', 'MINIROCKET_FAV', 'WEASEL_FAV', 'XCM', 'XCM_STACKING'], case_sensitive=False), default='XCM', show_default=True, help='Method variant to perform ETSC')
@click.option('-p', '--optimize', type=click.IntRange(min=0, max=2), default=0, show_default=True, help='Metric to optimize: 0 - accuracy, 1 - F1-score, 2 - harmonic mean')
@click.option('-s', '--splits', type=click.IntRange(min=2), default=5, show_default=True, help='Number of splits')
@click.option('--trunc', is_flag=True, help='Use STRUT approach to find the best time-point to perform ETSC.')
@pass_config
def cli(config: Config, input_train_file, input_test_file, method, optimize, splits, trunc):
    """
    Command line tool for Early Time-Series Classification using various methods including XCM and XCM stacking.
    """
    logger.info(f"Running {method} method ...")
    
    # Load train and test data, assuming space is the delimiter. Change this to comma or another delimiter as necessary
    train_data = pd.read_csv(input_train_file, header=None, delim_whitespace=True)
    test_data = pd.read_csv(input_test_file, header=None, delim_whitespace=True)

    # Separate features and labels correctly
    X_train = train_data.iloc[:, 1:].values  # All columns except the first are features
    y_train = train_data.iloc[:, 0].values   # The first column is the label

    X_test = test_data.iloc[:, 1:].values    # All columns except the first are features
    y_test = test_data.iloc[:, 0].values     # The first column is the label

    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # Handle unseen labels in the test set
    y_test = pd.Series(y_test).apply(lambda label: label if label in label_encoder.classes_ else None).values
    y_test = label_encoder.transform(y_test[pd.notna(y_test)])  # Only transform seen labels

    # Check the unique labels and number of classes
    n_classes = len(np.unique(y_train))
    print(f"Unique y_train labels: {np.unique(y_train)}")
    print(f"Unique y_test labels: {np.unique(y_test)}")
    print(f"Number of classes (n_classes): {n_classes}")

    # Encode labels into one-hot format
    y_train_onehot = to_categorical(y_train, num_classes=n_classes)
    y_test_onehot = to_categorical(y_test, num_classes=n_classes)
    
    # Create classifier based on method
    classifier = STRUT(config)

    if method == 'XCM':
        logger.info("Training and evaluating XCM model...")
        predictions, training_time, test_time, earliness = classifier.xcm_strut((X_train, y_train_onehot), (X_test, y_test_onehot))
    elif method == 'XCM_STACKING':
        logger.info("Training and evaluating XCM Stacking ensemble...")
        predictions, training_time, test_time, earliness = classifier.xcm_stacking_strut((X_train, y_train), (X_test, y_test))

    accuracy_score = accuracy(predictions, y_test)
    logger.info(f"Accuracy: {accuracy_score}")
    logger.info(f"Training time: {training_time}s, Testing time: {test_time}s, Earliness: {earliness}")

# Define the STRUT class that supports multiple methods, including XCM and XCM stacking
class STRUT:
    def __init__(self, config):
        self.config = config
    
    def xcm_strut(self, train_data, test_data):
        X_train, y_train_onehot = train_data
        X_test, y_test_onehot = test_data

        # XCM model settings
        n_timesteps = X_train.shape[1]
        n_features = 1  # Set to 1 because we now use 2D input (timesteps, 1 feature per sample)
        n_classes = y_train_onehot.shape[1]  # One-hot encoding sets the number of classes

        # Reshape data for XCM
        X_train = X_train.reshape((X_train.shape[0], n_timesteps, n_features, 1))
        X_test = X_test.reshape((X_test.shape[0], n_timesteps, n_features, 1))

        # Build and compile XCM model
        model = xcm(input_shape=(n_timesteps, n_features), n_class=n_classes, window_size=0.1)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        start_time = time.time()
        model.fit(X_train, y_train_onehot, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
        training_time = time.time() - start_time

        # Test the model
        start_time = time.time()
        y_pred = np.argmax(model.predict(X_test), axis=1)
        test_time = time.time() - start_time

        return y_pred, training_time, test_time, None  # Earliness not yet implemented

    def xcm_stacking_strut(self, train_data, test_data):
        X_train, y_train = train_data
        X_test, y_test = test_data

        # XCM model settings
        n_timesteps = X_train.shape[1]
        n_features = 1
        n_classes = len(np.unique(y_train))

        X_train_reshaped = X_train.reshape((X_train.shape[0], n_timesteps, n_features, 1))
        X_test_reshaped = X_test.reshape((X_test.shape[0], n_timesteps, n_features, 1))

        # Base learners
        base_learners = [
            ('xcm', KerasClassifier(build_fn=lambda: xcm((n_timesteps, n_features), n_classes, 0.1), epochs=50, verbose=0)),
            ('decision_tree', DecisionTreeClassifier()),
            ('knn', KNeighborsClassifier())
        ]

        # Stacking ensemble with LogisticRegression as the meta-learner
        meta_learner = LogisticRegression()
        stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

        # Train the stacking model
        stacking_model.fit(X_train_reshaped, y_train)

        # Predict using the stacking ensemble
        y_pred = stacking_model.predict(X_test_reshaped)
        return y_pred, None, None, None  # Adjust for training/test time and earliness if needed

if __name__ == "__main__":
    cli()
