import pytest
import ets.algorithms.utils as utils

CLASS_ONE = 1
CLASS_FIVE = 5

#
# Earliness
#


def test_earliness_zero_length():
    # Earliness should be None if time-series length is zero.
    assert utils.earliness([], 0) is None
    assert utils.earliness([(1, CLASS_ONE), (4, CLASS_ONE)], 0) is None


def test_earliness_empty_predictions():
    # Earliness should be None if no predictions are provided.
    assert utils.earliness([], 5) is None


def test_earliness_length_5():
    # Test earliness on time-series of length 5.
    assert utils.earliness([(1, CLASS_ONE)], 5) == 0.2
    assert utils.earliness([(1, CLASS_ONE), (1, CLASS_ONE)], 5) == 0.2
    assert utils.earliness([(1, CLASS_ONE), (4, CLASS_ONE)], 5) == 0.5


def test_earliness_length_100():
    # Test earliness on time-series of length 100.
    assert utils.earliness([(1, CLASS_ONE)], 100) == 0.01
    assert utils.earliness([(1, CLASS_ONE), (10, CLASS_ONE)], 100) == 0.055
    assert utils.earliness([(20, CLASS_ONE), (50, CLASS_ONE)], 100) == 0.35


#
# Accuracy
#

def test_accuracy_empty_predictions():
    # Accuracy should be None if no predictions are provided.
    assert utils.accuracy([], []) is None
    assert utils.accuracy([], [CLASS_ONE, CLASS_FIVE]) is None


def test_accuracy_empty_ground_truth():
    # Accuracy should be None if no ground truth is provided.
    assert utils.accuracy([(1, CLASS_ONE)], []) is None
    assert utils.accuracy([(1, CLASS_ONE), (25, CLASS_FIVE)], []) is None


def test_accuracy_binary_classification():
    # Test accuracy on binary classification.
    assert utils.accuracy([(1, CLASS_ONE), (1, CLASS_FIVE)], [CLASS_ONE, CLASS_FIVE]) == 1
    assert utils.accuracy([(1, CLASS_ONE), (1, CLASS_ONE)], [CLASS_ONE, CLASS_FIVE]) == 0.5


#
# Counts
#

def test_counts_empty_predictions():
    # All counts should be 0 if no predictions are provided.
    assert utils.counts(CLASS_ONE, [], []) == (0, 0, 0, 0)
    assert utils.counts(CLASS_ONE, [], [CLASS_ONE, CLASS_ONE]) == (0, 0, 0, 0)


def test_counts_empty_ground_truth():
    # All counts should be 0 if no ground truth is provided.
    assert utils.counts(CLASS_ONE, [(1, CLASS_ONE), (25, CLASS_FIVE)], []) == (0, 0, 0, 0)


def test_counts_binary_classification():
    # Test counts on binary classification
    assert utils.counts(CLASS_ONE, [(1, CLASS_ONE), (1, CLASS_FIVE)], [CLASS_ONE, CLASS_FIVE]) == (1, 1, 0, 0)
    assert utils.counts(CLASS_ONE, [(1, CLASS_ONE), (1, CLASS_ONE)], [CLASS_ONE, CLASS_FIVE]) == (1, 0, 1, 0)
    assert utils.counts(CLASS_ONE, [(1, CLASS_ONE), (1, CLASS_ONE)], [CLASS_FIVE, CLASS_FIVE]) == (0, 0, 2, 0)
    assert utils.counts(CLASS_ONE, [(1, CLASS_FIVE), (1, CLASS_FIVE)], [CLASS_ONE, CLASS_ONE]) == (0, 0, 0, 2)


#
# Precision
#

def test_precision_zero_counts():
    # Precision should be None if tp and fp are zero.
    assert utils.precision(0, 0) is None


def test_precision_binary_classification():
    # Test precision for non-zero counts.
    assert utils.precision(10, 40) == 0.2


#
# Recall
#

def test_recall_zero_counts():
    # Recall should be None if tp and fn are zero.
    assert utils.recall(0, 0) is None


def test_recall_binary_classification():
    # Test recall for non-zero counts.
    assert utils.recall(10, 40) == 0.2


#
# F-measure
#

def test_f_measure_zero_counts():
    # F-measure should be None if tp, fp and fn are zero.
    assert utils.f_measure(0, 0, 0) is None


def test_f_measure_no_precision():
    # F-measure should be None if precision is None.
    assert utils.f_measure(0, 0, 20) is None


def test_f_measure_no_recall():
    # F-measure should be None if precision is None.
    assert utils.f_measure(0, 20, 0) is None


def test_f_measure_zero():
    # F-measure should be None if precision and recall are zero.
    assert utils.f_measure(0, 10, 20) is None


def test_f_measure_binary_classification():
    # Test f-measure for non-zero counts.
    assert utils.f_measure(10, 40, 40) == pytest.approx(0.2)
