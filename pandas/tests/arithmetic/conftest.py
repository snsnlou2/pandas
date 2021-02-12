
import numpy as np
import pytest
import pandas as pd
from pandas import Float64Index, Int64Index, RangeIndex, UInt64Index
import pandas._testing as tm

def id_func(x):
    if isinstance(x, tuple):
        assert (len(x) == 2)
        return ((x[0].__name__ + '-') + str(x[1]))
    else:
        return x.__name__

@pytest.fixture(params=[1, np.array(1, dtype=np.int64)])
def one(request):
    "\n    Several variants of integer value 1. The zero-dim integer array\n    behaves like an integer.\n\n    This fixture can be used to check that datetimelike indexes handle\n    addition and subtraction of integers and zero-dimensional arrays\n    of integers.\n\n    Examples\n    --------\n    >>> dti = pd.date_range('2016-01-01', periods=2, freq='H')\n    >>> dti\n    DatetimeIndex(['2016-01-01 00:00:00', '2016-01-01 01:00:00'],\n    dtype='datetime64[ns]', freq='H')\n    >>> dti + one\n    DatetimeIndex(['2016-01-01 01:00:00', '2016-01-01 02:00:00'],\n    dtype='datetime64[ns]', freq='H')\n    "
    return request.param
zeros = [box_cls(([0] * 5), dtype=dtype) for box_cls in [pd.Index, np.array, pd.array] for dtype in [np.int64, np.uint64, np.float64]]
zeros.extend([box_cls(([(- 0.0)] * 5), dtype=np.float64) for box_cls in [pd.Index, np.array]])
zeros.extend([np.array(0, dtype=dtype) for dtype in [np.int64, np.uint64, np.float64]])
zeros.extend([np.array((- 0.0), dtype=np.float64)])
zeros.extend([0, 0.0, (- 0.0)])

@pytest.fixture(params=zeros)
def zero(request):
    "\n    Several types of scalar zeros and length 5 vectors of zeros.\n\n    This fixture can be used to check that numeric-dtype indexes handle\n    division by any zero numeric-dtype.\n\n    Uses vector of length 5 for broadcasting with `numeric_idx` fixture,\n    which creates numeric-dtype vectors also of length 5.\n\n    Examples\n    --------\n    >>> arr = RangeIndex(5)\n    >>> arr / zeros\n    Float64Index([nan, inf, inf, inf, inf], dtype='float64')\n    "
    return request.param

@pytest.fixture(params=[Float64Index(np.arange(5, dtype='float64')), Int64Index(np.arange(5, dtype='int64')), UInt64Index(np.arange(5, dtype='uint64')), RangeIndex(5)], ids=(lambda x: type(x).__name__))
def numeric_idx(request):
    '\n    Several types of numeric-dtypes Index objects\n    '
    return request.param

@pytest.fixture(params=[pd.Timedelta('5m4s').to_pytimedelta(), pd.Timedelta('5m4s'), pd.Timedelta('5m4s').to_timedelta64()], ids=(lambda x: type(x).__name__))
def scalar_td(request):
    '\n    Several variants of Timedelta scalars representing 5 minutes and 4 seconds\n    '
    return request.param

@pytest.fixture(params=[pd.offsets.Day(3), pd.offsets.Hour(72), pd.Timedelta(days=3).to_pytimedelta(), pd.Timedelta('72:00:00'), np.timedelta64(3, 'D'), np.timedelta64(72, 'h')], ids=(lambda x: type(x).__name__))
def three_days(request):
    '\n    Several timedelta-like and DateOffset objects that each represent\n    a 3-day timedelta\n    '
    return request.param

@pytest.fixture(params=[pd.offsets.Hour(2), pd.offsets.Minute(120), pd.Timedelta(hours=2).to_pytimedelta(), pd.Timedelta(seconds=(2 * 3600)), np.timedelta64(2, 'h'), np.timedelta64(120, 'm')], ids=(lambda x: type(x).__name__))
def two_hours(request):
    '\n    Several timedelta-like and DateOffset objects that each represent\n    a 2-hour timedelta\n    '
    return request.param
_common_mismatch = [pd.offsets.YearBegin(2), pd.offsets.MonthBegin(1), pd.offsets.Minute()]

@pytest.fixture(params=([pd.Timedelta(minutes=30).to_pytimedelta(), np.timedelta64(30, 's'), pd.Timedelta(seconds=30)] + _common_mismatch))
def not_hourly(request):
    '\n    Several timedelta-like and DateOffset instances that are _not_\n    compatible with Hourly frequencies.\n    '
    return request.param

@pytest.fixture(params=([np.timedelta64(4, 'h'), pd.Timedelta(hours=23).to_pytimedelta(), pd.Timedelta('23:00:00')] + _common_mismatch))
def not_daily(request):
    '\n    Several timedelta-like and DateOffset instances that are _not_\n    compatible with Daily frequencies.\n    '
    return request.param

@pytest.fixture(params=([np.timedelta64(365, 'D'), pd.Timedelta(days=365).to_pytimedelta(), pd.Timedelta(days=365)] + _common_mismatch))
def mismatched_freq(request):
    '\n    Several timedelta-like and DateOffset instances that are _not_\n    compatible with Monthly or Annual frequencies.\n    '
    return request.param

@pytest.fixture(params=[pd.Index, pd.Series, pd.DataFrame, pd.array], ids=id_func)
def box_with_array(request):
    '\n    Fixture to test behavior for Index, Series, DataFrame, and pandas Array\n    classes\n    '
    return request.param

@pytest.fixture(params=[pd.Index, pd.Series, tm.to_array, np.array, list], ids=id_func)
def box_1d_array(request):
    '\n    Fixture to test behavior for Index, Series, tm.to_array, numpy Array and list\n    classes\n    '
    return request.param
box_with_array2 = box_with_array
