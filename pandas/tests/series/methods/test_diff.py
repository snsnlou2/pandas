
import numpy as np
import pytest
from pandas import Series, TimedeltaIndex, date_range
import pandas._testing as tm

class TestSeriesDiff():

    def test_diff_np(self):
        pytest.skip('skipping due to Series no longer being an ndarray')
        s = Series(np.arange(5))
        r = np.diff(s)
        tm.assert_series_equal(Series([np.nan, 0, 0, 0, np.nan]), r)

    def test_diff_int(self):
        a = 10000000000000000
        b = (a + 1)
        s = Series([a, b])
        result = s.diff()
        assert (result[1] == 1)

    def test_diff_tz(self):
        ts = tm.makeTimeSeries(name='ts')
        ts.diff()
        result = ts.diff((- 1))
        expected = (ts - ts.shift((- 1)))
        tm.assert_series_equal(result, expected)
        result = ts.diff(0)
        expected = (ts - ts)
        tm.assert_series_equal(result, expected)
        s = Series(date_range('20130102', periods=5))
        result = s.diff()
        expected = (s - s.shift(1))
        tm.assert_series_equal(result, expected)
        result = (result - result.shift(1))
        expected = expected.diff()
        tm.assert_series_equal(result, expected)
        s = Series(date_range('2000-01-01 09:00:00', periods=5, tz='US/Eastern'), name='foo')
        result = s.diff()
        expected = Series(TimedeltaIndex((['NaT'] + (['1 days'] * 4))), name='foo')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('input,output,diff', [([False, True, True, False, False], [np.nan, True, False, True, False], 1)])
    def test_diff_bool(self, input, output, diff):
        s = Series(input)
        result = s.diff()
        expected = Series(output)
        tm.assert_series_equal(result, expected)

    def test_diff_object_dtype(self):
        s = Series([False, True, 5.0, np.nan, True, False])
        result = s.diff()
        expected = (s - s.shift(1))
        tm.assert_series_equal(result, expected)
