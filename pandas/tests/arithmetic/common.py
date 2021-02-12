
'\nAssertion helpers for arithmetic tests.\n'
import numpy as np
import pytest
from pandas import DataFrame, Index, Series, array as pd_array
import pandas._testing as tm
from pandas.core.arrays import PandasArray

def assert_invalid_addsub_type(left, right, msg=None):
    '\n    Helper to assert that left and right can be neither added nor subtracted.\n\n    Parameters\n    ----------\n    left : object\n    right : object\n    msg : str or None, default None\n    '
    with pytest.raises(TypeError, match=msg):
        (left + right)
    with pytest.raises(TypeError, match=msg):
        (right + left)
    with pytest.raises(TypeError, match=msg):
        (left - right)
    with pytest.raises(TypeError, match=msg):
        (right - left)

def get_upcast_box(box, vector):
    '\n    Given two box-types, find the one that takes priority\n    '
    if ((box is DataFrame) or isinstance(vector, DataFrame)):
        return DataFrame
    if ((box is Series) or isinstance(vector, Series)):
        return Series
    if ((box is Index) or isinstance(vector, Index)):
        return Index
    return box

def assert_invalid_comparison(left, right, box):
    '\n    Assert that comparison operations with mismatched types behave correctly.\n\n    Parameters\n    ----------\n    left : np.ndarray, ExtensionArray, Index, or Series\n    right : object\n    box : {pd.DataFrame, pd.Series, pd.Index, pd.array, tm.to_array}\n    '
    xbox = (box if (box not in [Index, pd_array]) else np.array)

    def xbox2(x):
        if isinstance(x, PandasArray):
            return x._ndarray
        return x
    result = xbox2((left == right))
    expected = xbox(np.zeros(result.shape, dtype=np.bool_))
    tm.assert_equal(result, expected)
    result = xbox2((right == left))
    tm.assert_equal(result, expected)
    result = xbox2((left != right))
    tm.assert_equal(result, (~ expected))
    result = xbox2((right != left))
    tm.assert_equal(result, (~ expected))
    msg = '|'.join(['Invalid comparison between', 'Cannot compare type', 'not supported between', 'invalid type promotion', "The DTypes <class 'numpy.dtype\\[datetime64\\]'> and <class 'numpy.dtype\\[int64\\]'> do not have a common DType. For example they cannot be stored in a single array unless the dtype is `object`."])
    with pytest.raises(TypeError, match=msg):
        (left < right)
    with pytest.raises(TypeError, match=msg):
        (left <= right)
    with pytest.raises(TypeError, match=msg):
        (left > right)
    with pytest.raises(TypeError, match=msg):
        (left >= right)
    with pytest.raises(TypeError, match=msg):
        (right < left)
    with pytest.raises(TypeError, match=msg):
        (right <= left)
    with pytest.raises(TypeError, match=msg):
        (right > left)
    with pytest.raises(TypeError, match=msg):
        (right >= left)
