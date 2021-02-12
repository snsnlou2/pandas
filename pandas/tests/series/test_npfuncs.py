
'\nTests for np.foo applied to Series, not necessarily ufuncs.\n'
import numpy as np
from pandas import Series

class TestPtp():

    def test_ptp(self):
        N = 1000
        arr = np.random.randn(N)
        ser = Series(arr)
        assert (np.ptp(ser) == np.ptp(arr))

def test_numpy_unique(datetime_series):
    np.unique(datetime_series)
