
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import delta_to_nanoseconds
from pandas import Timedelta, offsets

@pytest.mark.parametrize('obj,expected', [(np.timedelta64(14, 'D'), (((14 * 24) * 3600) * 1000000000.0)), (Timedelta(minutes=(- 7)), (((- 7) * 60) * 1000000000.0)), (Timedelta(minutes=(- 7)).to_pytimedelta(), (((- 7) * 60) * 1000000000.0)), (offsets.Nano(125), 125), (1, 1), (np.int64(2), 2), (np.int32(3), 3)])
def test_delta_to_nanoseconds(obj, expected):
    result = delta_to_nanoseconds(obj)
    assert (result == expected)

def test_delta_to_nanoseconds_error():
    obj = np.array([123456789], dtype='m8[ns]')
    with pytest.raises(TypeError, match="<class 'numpy.ndarray'>"):
        delta_to_nanoseconds(obj)

def test_huge_nanoseconds_overflow():
    assert (delta_to_nanoseconds(Timedelta(10000000000.0)) == 10000000000.0)
    assert (delta_to_nanoseconds(Timedelta(nanoseconds=10000000000.0)) == 10000000000.0)
