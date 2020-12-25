import numpy as np
import pandas as pd
import ets.algorithms.utils as utils

#
# DF merge
#


def test_df_merge():
    merged = utils.df_merge([pd.DataFrame()])
    assert merged.empty


def test_df_merge1():
    merged = utils.df_merge([pd.DataFrame(), pd.DataFrame()])
    assert merged.empty


def test_df_merge2():
    merged = utils.df_merge([pd.DataFrame(np.ones([2, 2])), pd.DataFrame(np.ones([2, 2]))])
    assert not merged.empty
    assert np.array_equal(merged, pd.DataFrame(np.ones([2, 2])))


def test_df_merge3():
    merged = utils.df_merge([pd.DataFrame([[2, 2], [2, 2]]), pd.DataFrame(np.ones([2, 2]))])
    assert not merged.empty
    assert np.array_equal(merged, pd.DataFrame([[1.5, 1.5], [1.5, 1.5]]))
