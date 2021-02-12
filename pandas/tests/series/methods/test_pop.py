
from pandas import Series
import pandas._testing as tm

def test_pop():
    ser = Series([0, 4, 0], index=['A', 'B', 'C'], name=4)
    result = ser.pop('B')
    assert (result == 4)
    expected = Series([0, 0], index=['A', 'C'], name=4)
    tm.assert_series_equal(ser, expected)
