
import operator
import numpy as np
import pytest
from pandas import DataFrame, Series
import pandas._testing as tm

class TestMatmul():

    def test_matmul(self):
        a = Series(np.random.randn(4), index=['p', 'q', 'r', 's'])
        b = DataFrame(np.random.randn(3, 4), index=['1', '2', '3'], columns=['p', 'q', 'r', 's']).T
        result = operator.matmul(a, b)
        expected = Series(np.dot(a.values, b.values), index=['1', '2', '3'])
        tm.assert_series_equal(result, expected)
        result = operator.matmul(b.T, a)
        expected = Series(np.dot(b.T.values, a.T.values), index=['1', '2', '3'])
        tm.assert_series_equal(result, expected)
        result = operator.matmul(a, a)
        expected = np.dot(a.values, a.values)
        tm.assert_almost_equal(result, expected)
        result = operator.matmul(a.values, a)
        expected = np.dot(a.values, a.values)
        tm.assert_almost_equal(result, expected)
        result = operator.matmul(a.values.tolist(), a)
        expected = np.dot(a.values, a.values)
        tm.assert_almost_equal(result, expected)
        result = operator.matmul(b.T.values, a)
        expected = np.dot(b.T.values, a.values)
        tm.assert_almost_equal(result, expected)
        result = operator.matmul(b.T.values.tolist(), a)
        expected = np.dot(b.T.values, a.values)
        tm.assert_almost_equal(result, expected)
        a['p'] = int(a.p)
        result = operator.matmul(b.T, a)
        expected = Series(np.dot(b.T.values, a.T.values), index=['1', '2', '3'])
        tm.assert_series_equal(result, expected)
        a = a.astype(int)
        result = operator.matmul(b.T, a)
        expected = Series(np.dot(b.T.values, a.T.values), index=['1', '2', '3'])
        tm.assert_series_equal(result, expected)
        msg = 'Dot product shape mismatch, \\(4,\\) vs \\(3,\\)'
        with pytest.raises(Exception, match=msg):
            a.dot(a.values[:3])
        msg = 'matrices are not aligned'
        with pytest.raises(ValueError, match=msg):
            a.dot(b.T)
