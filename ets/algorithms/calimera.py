import numpy as np
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.calibration import CalibratedClassifierCV

from ets.algorithms.early_classifier import EarlyClassifier


class CALIMERA(EarlyClassifier):
    def __init__(self, timestamps, delaypenalty):
        self.delaypenalty = delaypenalty

    def _generate_timestamps(max_timestamp):
        NUM_TIMESTAMPS = 20
        num_intervals_between_timestamps = min(NUM_TIMESTAMPS, max_timestamp)
        step = max_timestamp / num_intervals_between_timestamps
        timestamps = np.arange(step, max_timestamp+0.5*step, step).astype(np.int32)
        timestamps[-1] = max_timestamp
        return timestamps

    def _learn_feature_extractors(X, timestamps):
        extractors = []
        for timestamp in timestamps:
            if timestamp < 9:
                extractors.append(lambda x: x.reshape(x.shape[0], x.shape[2]))
            else:
                X_sub = X[:, :timestamp]
                X_sub = X_sub.reshape(X_sub.shape[0], 1, -1)
                extractors.append(MiniRocket().fit(X_sub).transform)
        return extractors

    def _learn_classifiers(X, feature_extractors, ys, timestamps):
        T = timestamps.shape[0]
        classifiers = [None for t in range(T)]

        for t in range(T):
            timestamp = timestamps[t]
            X_sub = X[:, :timestamp]
            X_sub = X_sub.reshape(X_sub.shape[0], 1, -1)
            features = feature_extractors[t](X_sub)
            features = np.asarray(features)
            features = features.reshape(features.shape[0], -1)
            classifier = WeakClassifier()
            classifier.fit(features, ys)
            classifiers[t] = classifier

        return classifiers

    def _generate_data_for_training_stopping_module(classifiers):
        predictors = []
        costs = []
        for classifier in classifiers:
            costs.append(classifier.costs_for_training_stopping_module)
            predictors.append([
                _scores_to_predictors(s) 
                for s in classifier.predictors_for_training_stopping_module
            ])
        return np.asarray(predictors), np.asarray(costs)

    def train(self, train_data, labels):
        X_train = train_data.to_numpy()
        timestamps = CALIMERA._generate_timestamps(max_timestamp=X_train.shape[1])

        self.feature_extractors = CALIMERA._learn_feature_extractors(X_train, timestamps)
        self.classifiers = CALIMERA._learn_classifiers(
            X_train,self.feature_extractors, labels, timestamps
        )
        predictors, costs = CALIMERA._generate_data_for_training_stopping_module(self.classifiers)
        self.stopping_module = StoppingModule()
        self.stopping_module.fit(
            predictors,
            costs,
            timestamps,
            self.delaypenalty,
            KernelRidgeRegressionWrapper
        )
        self.timestamps = timestamps

    def predict(self, test_data):
        X = test_data.to_numpy()
        predictions = test(
            X, self.timestamps, self.feature_extractors, self.classifiers, self.stopping_module
        )
        return predictions


class KernelRidgeRegressionWrapper:
    def __init__(self):
        self.model = KernelRidge(kernel="rbf")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class WeakClassifier:
    def normalize_X(self, X):
        return (X - self.feature_means) / self.feature_norms

    def fit(self, X, y):
        RC_ALPHAS = np.logspace(-3, 3, 10)

        self.feature_means = np.mean(X, axis=0)
        self.feature_norms = np.linalg.norm(X, axis=0)
        self.feature_norms[self.feature_norms == 0] = 1.0

        uncalibrated_clf = RidgeClassifierCV(
            alphas=RC_ALPHAS, store_cv_values=True, scoring='accuracy')
        normalized_X = self.normalize_X(X)
        uncalibrated_clf.fit(normalized_X, y)

        chosen_alpha_index = np.where(RC_ALPHAS == uncalibrated_clf.alpha_)
        X_scores = uncalibrated_clf.cv_values_[:, :, chosen_alpha_index]
        X_scores = X_scores.reshape((X_scores.shape[0], X_scores.shape[1]))
        X_scores = X_scores + uncalibrated_clf.intercept_

        # transform to uncalibrated probabilities
        exped_X_scores = np.exp(X_scores)
        if len(uncalibrated_clf.classes_) == 2:
            X_probab = exped_X_scores / (exped_X_scores + np.exp(-X_scores))
        else:
            X_probab = exped_X_scores / np.sum(exped_X_scores, axis=1)[:,None]

        # walkaround to use L1O validation data in an sklearn calibrator and save some time
        mockup_clf = MockupClassifierForPassingValidationDataToSklearnCalibrator(
            X_probab, uncalibrated_clf.classes_
        )
        calibrated_clf = CalibratedClassifierCV(mockup_clf, method="sigmoid", cv="prefit")
        mockup_X = np.zeros((y.shape[0], 1))
        calibrated_clf.fit(mockup_X, y)

        # generate data for stopping module training
        self.costs_for_training_stopping_module = 1.0 - np.max(
            calibrated_clf.predict_proba(mockup_X), axis=1
        )
        self.predictors_for_training_stopping_module = X_scores

        # actual classification can be performed with uncalibrated clfs
        self.clf = uncalibrated_clf 

    def predict(self, X):
        return self.clf.predict(self.normalize_X(X))

    def get_scores(self, X):
        return np.atleast_2d(self.clf.decision_function(self.normalize_X(X)))

    def get_labels(self):
        return self.clf.classes_


class MockupClassifierForPassingValidationDataToSklearnCalibrator:
    def __init__(self, mockup_scores, classes):
        self.mockup_scores = mockup_scores
        self.classes_ = classes

    def fit(self):
        pass

    def decision_function(self, X):
        return self.mockup_scores


class StoppingModule:
    def fit(self, predictors, original_costs, timestamps, alpha, REGRESSOR_WAIT):
        costs = np.copy(original_costs)

        T = timestamps.shape[0]
        n = predictors.shape[1]

        for t in range(timestamps.shape[0]):
            costs[t, :] += alpha * (timestamps[t] / timestamps[-1])

        self.halters = [None for t in range(T-1)]

        for t in range(T-2, -1, -1):
            X = predictors[t, :].squeeze()
            X = X.reshape(X.shape[0], -1)
            y = costs[t+1, :] - costs[t, :]

            model = REGRESSOR_WAIT().fit(X, y)

            self.halters[t] = model
            predicted_cost_difference = model.predict(X)
            for j in range(n):
                if predicted_cost_difference[j] < 0:
                    costs[t, j] = costs[t+1, j]

    def should_stop(self, predictors, t):
        predicted_cost_difference = self.halters[t].predict([predictors])
        return predicted_cost_difference > 0


def _scores_to_predictors(scores):
    if len(scores) == 1:
        return scores
    highest_score = np.max(scores)
    second_highest_score = np.partition(scores, -2)[-2]
    score_diff_stolen_from_teaser = highest_score - second_highest_score
    predictors = np.zeros(scores.shape[0]+2)
    predictors[:-2] = scores
    predictors[-2] = score_diff_stolen_from_teaser
    predictors[-1] = highest_score
    return predictors


def test(X, timestamps, feature_extractors, classifiers, stopping_module):
    n = X.shape[0]
    predictions = []
    for j in range(n):
        for t in range(timestamps.shape[0]):
            X_sub = X[j, :timestamps[t]]
            X_sub = X_sub.reshape(1, 1, X_sub.shape[0])
            features = np.asarray(feature_extractors[t](X_sub))
            scores = classifiers[t].get_scores(features.reshape(1, -1))[0]
            predictors = _scores_to_predictors(scores)
            should_stop = (t==(timestamps.shape[0]-1) or stopping_module.should_stop(predictors, t))
            if should_stop:
                predicted_label = classifiers[t].predict(features.reshape(1, -1))[0]
                predictions.append((timestamps[t], predicted_label))
                break
    return predictions
