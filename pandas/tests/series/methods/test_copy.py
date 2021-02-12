
import numpy as np
import pytest
from pandas import Series, Timestamp
import pandas._testing as tm

class TestCopy():

    @pytest.mark.parametrize('deep', [None, False, True])
    def test_copy(self, deep):
        ser = Series(np.arange(10), dtype='float64')
        if (deep is None):
            ser2 = ser.copy()
        else:
            ser2 = ser.copy(deep=deep)
        ser2[::2] = np.NaN
        if ((deep is None) or (deep is True)):
            assert np.isnan(ser2[0])
            assert (not np.isnan(ser[0]))
        else:
            assert np.isnan(ser2[0])
            assert np.isnan(ser[0])

    @pytest.mark.parametrize('deep', [None, False, True])
    def test_copy_tzaware(self, deep):
        expected = Series([Timestamp('2012/01/01', tz='UTC')])
        expected2 = Series([Timestamp('1999/01/01', tz='UTC')])
        ser = Series([Timestamp('2012/01/01', tz='UTC')])
        if (deep is None):
            ser2 = ser.copy()
        else:
            ser2 = ser.copy(deep=deep)
        ser2[0] = Timestamp('1999/01/01', tz='UTC')
        if ((deep is None) or (deep is True)):
            tm.assert_series_equal(ser2, expected2)
            tm.assert_series_equal(ser, expected)
        else:
            tm.assert_series_equal(ser2, expected2)
            tm.assert_series_equal(ser, expected2)

    def test_copy_name(self, datetime_series):
        result = datetime_series.copy()
        assert (result.name == datetime_series.name)

    def test_copy_index_name_checking(self, datetime_series):
        datetime_series.index.name = None
        assert (datetime_series.index.name is None)
        assert (datetime_series is datetime_series)
        cp = datetime_series.copy()
        cp.index.name = 'foo'
        assert (datetime_series.index.name is None)
