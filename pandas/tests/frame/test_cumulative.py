
'\nTests for DataFrame cumulative operations\n\nSee also\n--------\ntests.series.test_cumulative\n'
import numpy as np
from pandas import DataFrame, Series
import pandas._testing as tm

class TestDataFrameCumulativeOps():

    def test_cumsum_corner(self):
        dm = DataFrame(np.arange(20).reshape(4, 5), index=range(4), columns=range(5))
        result = dm.cumsum()

    def test_cumsum(self, datetime_frame):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan
        cumsum = datetime_frame.cumsum()
        expected = datetime_frame.apply(Series.cumsum)
        tm.assert_frame_equal(cumsum, expected)
        cumsum = datetime_frame.cumsum(axis=1)
        expected = datetime_frame.apply(Series.cumsum, axis=1)
        tm.assert_frame_equal(cumsum, expected)
        df = DataFrame({'A': np.arange(20)}, index=np.arange(20))
        df.cumsum()
        cumsum_xs = datetime_frame.cumsum(axis=1)
        assert (np.shape(cumsum_xs) == np.shape(datetime_frame))

    def test_cumprod(self, datetime_frame):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan
        cumprod = datetime_frame.cumprod()
        expected = datetime_frame.apply(Series.cumprod)
        tm.assert_frame_equal(cumprod, expected)
        cumprod = datetime_frame.cumprod(axis=1)
        expected = datetime_frame.apply(Series.cumprod, axis=1)
        tm.assert_frame_equal(cumprod, expected)
        cumprod_xs = datetime_frame.cumprod(axis=1)
        assert (np.shape(cumprod_xs) == np.shape(datetime_frame))
        df = datetime_frame.fillna(0).astype(int)
        df.cumprod(0)
        df.cumprod(1)
        df = datetime_frame.fillna(0).astype(np.int32)
        df.cumprod(0)
        df.cumprod(1)

    def test_cummin(self, datetime_frame):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan
        cummin = datetime_frame.cummin()
        expected = datetime_frame.apply(Series.cummin)
        tm.assert_frame_equal(cummin, expected)
        cummin = datetime_frame.cummin(axis=1)
        expected = datetime_frame.apply(Series.cummin, axis=1)
        tm.assert_frame_equal(cummin, expected)
        df = DataFrame({'A': np.arange(20)}, index=np.arange(20))
        df.cummin()
        cummin_xs = datetime_frame.cummin(axis=1)
        assert (np.shape(cummin_xs) == np.shape(datetime_frame))

    def test_cummax(self, datetime_frame):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan
        cummax = datetime_frame.cummax()
        expected = datetime_frame.apply(Series.cummax)
        tm.assert_frame_equal(cummax, expected)
        cummax = datetime_frame.cummax(axis=1)
        expected = datetime_frame.apply(Series.cummax, axis=1)
        tm.assert_frame_equal(cummax, expected)
        df = DataFrame({'A': np.arange(20)}, index=np.arange(20))
        df.cummax()
        cummax_xs = datetime_frame.cummax(axis=1)
        assert (np.shape(cummax_xs) == np.shape(datetime_frame))

    def test_cumulative_ops_preserve_dtypes(self):
        df = DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3.0], 'C': [True, False, False]})
        result = df.cumsum()
        expected = DataFrame({'A': Series([1, 3, 6], dtype=np.int64), 'B': Series([1, 3, 6], dtype=np.float64), 'C': df['C'].cumsum()})
        tm.assert_frame_equal(result, expected)
