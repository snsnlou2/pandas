
'\nThese test the method maybe_promote from core/dtypes/cast.py\n'
import datetime
import numpy as np
import pytest
from pandas._libs.tslibs import NaT
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import is_complex_dtype, is_datetime64_dtype, is_datetime_or_timedelta_dtype, is_float_dtype, is_integer_dtype, is_object_dtype, is_scalar, is_timedelta64_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
import pandas as pd

@pytest.fixture(params=[bool, 'uint8', 'int32', 'uint64', 'float32', 'float64', 'complex64', 'complex128', 'M8[ns]', 'm8[ns]', str, bytes, object])
def any_numpy_dtype_reduced(request):
    "\n    Parameterized fixture for numpy dtypes, reduced from any_numpy_dtype.\n\n    * bool\n    * 'int32'\n    * 'uint64'\n    * 'float32'\n    * 'float64'\n    * 'complex64'\n    * 'complex128'\n    * 'M8[ns]'\n    * 'M8[ns]'\n    * str\n    * bytes\n    * object\n    "
    return request.param

def _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar=None):
    '\n    Auxiliary function to unify testing of scalar/array promotion.\n\n    Parameters\n    ----------\n    dtype : dtype\n        The value to pass on as the first argument to maybe_promote.\n    fill_value : scalar\n        The value to pass on as the second argument to maybe_promote as\n        a scalar.\n    expected_dtype : dtype\n        The expected dtype returned by maybe_promote (by design this is the\n        same regardless of whether fill_value was passed as a scalar or in an\n        array!).\n    exp_val_for_scalar : scalar\n        The expected value for the (potentially upcast) fill_value returned by\n        maybe_promote.\n    '
    assert is_scalar(fill_value)
    (result_dtype, result_fill_value) = maybe_promote(dtype, fill_value)
    expected_fill_value = exp_val_for_scalar
    assert (result_dtype == expected_dtype)
    _assert_match(result_fill_value, expected_fill_value)

def _assert_match(result_fill_value, expected_fill_value):
    res_type = type(result_fill_value)
    ex_type = type(expected_fill_value)
    if hasattr(result_fill_value, 'dtype'):
        assert (result_fill_value.dtype.kind == expected_fill_value.dtype.kind)
        assert (result_fill_value.dtype.itemsize == expected_fill_value.dtype.itemsize)
    else:
        assert ((res_type == ex_type) or (res_type.__name__ == ex_type.__name__))
    match_value = (result_fill_value == expected_fill_value)
    if (match_value is pd.NA):
        match_value = False
    match_missing = (isna(result_fill_value) and isna(expected_fill_value))
    assert (match_value or match_missing)

@pytest.mark.parametrize('dtype, fill_value, expected_dtype', [('int8', 1, 'int8'), ('int8', (np.iinfo('int8').max + 1), 'int16'), ('int8', (np.iinfo('int16').max + 1), 'int32'), ('int8', (np.iinfo('int32').max + 1), 'int64'), ('int8', (np.iinfo('int64').max + 1), 'object'), ('int8', (- 1), 'int8'), ('int8', (np.iinfo('int8').min - 1), 'int16'), ('int8', (np.iinfo('int16').min - 1), 'int32'), ('int8', (np.iinfo('int32').min - 1), 'int64'), ('int8', (np.iinfo('int64').min - 1), 'object'), ('uint8', 1, 'uint8'), ('uint8', (np.iinfo('int8').max + 1), 'uint8'), ('uint8', (np.iinfo('uint8').max + 1), 'uint16'), ('uint8', (np.iinfo('int16').max + 1), 'uint16'), ('uint8', (np.iinfo('uint16').max + 1), 'uint32'), ('uint8', (np.iinfo('int32').max + 1), 'uint32'), ('uint8', (np.iinfo('uint32').max + 1), 'uint64'), ('uint8', (np.iinfo('int64').max + 1), 'uint64'), ('uint8', (np.iinfo('uint64').max + 1), 'object'), ('uint8', (- 1), 'int16'), ('uint8', (np.iinfo('int8').min - 1), 'int16'), ('uint8', (np.iinfo('int16').min - 1), 'int32'), ('uint8', (np.iinfo('int32').min - 1), 'int64'), ('uint8', (np.iinfo('int64').min - 1), 'object'), ('int16', 1, 'int16'), ('int16', (np.iinfo('int8').max + 1), 'int16'), ('int16', (np.iinfo('int16').max + 1), 'int32'), ('int16', (np.iinfo('int32').max + 1), 'int64'), ('int16', (np.iinfo('int64').max + 1), 'object'), ('int16', (- 1), 'int16'), ('int16', (np.iinfo('int8').min - 1), 'int16'), ('int16', (np.iinfo('int16').min - 1), 'int32'), ('int16', (np.iinfo('int32').min - 1), 'int64'), ('int16', (np.iinfo('int64').min - 1), 'object'), ('uint16', 1, 'uint16'), ('uint16', (np.iinfo('int8').max + 1), 'uint16'), ('uint16', (np.iinfo('uint8').max + 1), 'uint16'), ('uint16', (np.iinfo('int16').max + 1), 'uint16'), ('uint16', (np.iinfo('uint16').max + 1), 'uint32'), ('uint16', (np.iinfo('int32').max + 1), 'uint32'), ('uint16', (np.iinfo('uint32').max + 1), 'uint64'), ('uint16', (np.iinfo('int64').max + 1), 'uint64'), ('uint16', (np.iinfo('uint64').max + 1), 'object'), ('uint16', (- 1), 'int32'), ('uint16', (np.iinfo('int8').min - 1), 'int32'), ('uint16', (np.iinfo('int16').min - 1), 'int32'), ('uint16', (np.iinfo('int32').min - 1), 'int64'), ('uint16', (np.iinfo('int64').min - 1), 'object'), ('int32', 1, 'int32'), ('int32', (np.iinfo('int8').max + 1), 'int32'), ('int32', (np.iinfo('int16').max + 1), 'int32'), ('int32', (np.iinfo('int32').max + 1), 'int64'), ('int32', (np.iinfo('int64').max + 1), 'object'), ('int32', (- 1), 'int32'), ('int32', (np.iinfo('int8').min - 1), 'int32'), ('int32', (np.iinfo('int16').min - 1), 'int32'), ('int32', (np.iinfo('int32').min - 1), 'int64'), ('int32', (np.iinfo('int64').min - 1), 'object'), ('uint32', 1, 'uint32'), ('uint32', (np.iinfo('int8').max + 1), 'uint32'), ('uint32', (np.iinfo('uint8').max + 1), 'uint32'), ('uint32', (np.iinfo('int16').max + 1), 'uint32'), ('uint32', (np.iinfo('uint16').max + 1), 'uint32'), ('uint32', (np.iinfo('int32').max + 1), 'uint32'), ('uint32', (np.iinfo('uint32').max + 1), 'uint64'), ('uint32', (np.iinfo('int64').max + 1), 'uint64'), ('uint32', (np.iinfo('uint64').max + 1), 'object'), ('uint32', (- 1), 'int64'), ('uint32', (np.iinfo('int8').min - 1), 'int64'), ('uint32', (np.iinfo('int16').min - 1), 'int64'), ('uint32', (np.iinfo('int32').min - 1), 'int64'), ('uint32', (np.iinfo('int64').min - 1), 'object'), ('int64', 1, 'int64'), ('int64', (np.iinfo('int8').max + 1), 'int64'), ('int64', (np.iinfo('int16').max + 1), 'int64'), ('int64', (np.iinfo('int32').max + 1), 'int64'), ('int64', (np.iinfo('int64').max + 1), 'object'), ('int64', (- 1), 'int64'), ('int64', (np.iinfo('int8').min - 1), 'int64'), ('int64', (np.iinfo('int16').min - 1), 'int64'), ('int64', (np.iinfo('int32').min - 1), 'int64'), ('int64', (np.iinfo('int64').min - 1), 'object'), ('uint64', 1, 'uint64'), ('uint64', (np.iinfo('int8').max + 1), 'uint64'), ('uint64', (np.iinfo('uint8').max + 1), 'uint64'), ('uint64', (np.iinfo('int16').max + 1), 'uint64'), ('uint64', (np.iinfo('uint16').max + 1), 'uint64'), ('uint64', (np.iinfo('int32').max + 1), 'uint64'), ('uint64', (np.iinfo('uint32').max + 1), 'uint64'), ('uint64', (np.iinfo('int64').max + 1), 'uint64'), ('uint64', (np.iinfo('uint64').max + 1), 'object'), ('uint64', (- 1), 'object'), ('uint64', (np.iinfo('int8').min - 1), 'object'), ('uint64', (np.iinfo('int16').min - 1), 'object'), ('uint64', (np.iinfo('int32').min - 1), 'object'), ('uint64', (np.iinfo('int64').min - 1), 'object')])
def test_maybe_promote_int_with_int(dtype, fill_value, expected_dtype):
    dtype = np.dtype(dtype)
    expected_dtype = np.dtype(expected_dtype)
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_int_with_float(any_int_dtype, float_dtype):
    dtype = np.dtype(any_int_dtype)
    fill_dtype = np.dtype(float_dtype)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    expected_dtype = np.float64
    exp_val_for_scalar = np.float64(fill_value)
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_float_with_int(float_dtype, any_int_dtype):
    dtype = np.dtype(float_dtype)
    fill_dtype = np.dtype(any_int_dtype)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    expected_dtype = dtype
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

@pytest.mark.parametrize('dtype, fill_value, expected_dtype', [('float32', 1, 'float32'), ('float32', (np.finfo('float32').max * 1.1), 'float64'), ('float64', 1, 'float64'), ('float64', (np.finfo('float32').max * 1.1), 'float64'), ('complex64', 1, 'complex64'), ('complex64', (np.finfo('float32').max * 1.1), 'complex128'), ('complex128', 1, 'complex128'), ('complex128', (np.finfo('float32').max * 1.1), 'complex128'), ('float32', (1 + 1j), 'complex64'), ('float32', (np.finfo('float32').max * (1.1 + 1j)), 'complex128'), ('float64', (1 + 1j), 'complex128'), ('float64', (np.finfo('float32').max * (1.1 + 1j)), 'complex128'), ('complex64', (1 + 1j), 'complex64'), ('complex64', (np.finfo('float32').max * (1.1 + 1j)), 'complex128'), ('complex128', (1 + 1j), 'complex128'), ('complex128', (np.finfo('float32').max * (1.1 + 1j)), 'complex128')])
def test_maybe_promote_float_with_float(dtype, fill_value, expected_dtype):
    dtype = np.dtype(dtype)
    expected_dtype = np.dtype(expected_dtype)
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_bool_with_any(any_numpy_dtype_reduced):
    dtype = np.dtype(bool)
    fill_dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    expected_dtype = (np.dtype(object) if (fill_dtype != bool) else fill_dtype)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_any_with_bool(any_numpy_dtype_reduced):
    dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = True
    expected_dtype = (np.dtype(object) if (dtype != bool) else dtype)
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_bytes_with_any(bytes_dtype, any_numpy_dtype_reduced):
    dtype = np.dtype(bytes_dtype)
    fill_dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    expected_dtype = np.dtype(np.object_)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_any_with_bytes(any_numpy_dtype_reduced, bytes_dtype):
    dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = b'abc'
    expected_dtype = np.dtype(np.object_)
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_datetime64_with_any(datetime64_dtype, any_numpy_dtype_reduced):
    dtype = np.dtype(datetime64_dtype)
    fill_dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    if is_datetime64_dtype(fill_dtype):
        expected_dtype = dtype
        exp_val_for_scalar = pd.Timestamp(fill_value).to_datetime64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

@pytest.mark.parametrize('fill_value', [pd.Timestamp('now'), np.datetime64('now'), datetime.datetime.now(), datetime.date.today()], ids=['pd.Timestamp', 'np.datetime64', 'datetime.datetime', 'datetime.date'])
def test_maybe_promote_any_with_datetime64(any_numpy_dtype_reduced, datetime64_dtype, fill_value):
    dtype = np.dtype(any_numpy_dtype_reduced)
    if is_datetime64_dtype(dtype):
        expected_dtype = dtype
        exp_val_for_scalar = pd.Timestamp(fill_value).to_datetime64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_datetimetz_with_any_numpy_dtype(tz_aware_fixture, any_numpy_dtype_reduced):
    dtype = DatetimeTZDtype(tz=tz_aware_fixture)
    fill_dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_datetimetz_with_datetimetz(tz_aware_fixture, tz_aware_fixture2):
    dtype = DatetimeTZDtype(tz=tz_aware_fixture)
    fill_dtype = DatetimeTZDtype(tz=tz_aware_fixture2)
    fill_value = pd.Series([(10 ** 9)], dtype=fill_dtype)[0]
    exp_val_for_scalar = fill_value
    if (dtype.tz == fill_dtype.tz):
        expected_dtype = dtype
    else:
        expected_dtype = np.dtype(object)
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

@pytest.mark.parametrize('fill_value', [None, np.nan, NaT])
def test_maybe_promote_datetimetz_with_na(tz_aware_fixture, fill_value):
    dtype = DatetimeTZDtype(tz=tz_aware_fixture)
    expected_dtype = dtype
    exp_val_for_scalar = NaT
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

@pytest.mark.parametrize('fill_value', [pd.Timestamp('now'), np.datetime64('now'), datetime.datetime.now(), datetime.date.today()], ids=['pd.Timestamp', 'np.datetime64', 'datetime.datetime', 'datetime.date'])
def test_maybe_promote_any_numpy_dtype_with_datetimetz(any_numpy_dtype_reduced, tz_aware_fixture, fill_value):
    dtype = np.dtype(any_numpy_dtype_reduced)
    fill_dtype = DatetimeTZDtype(tz=tz_aware_fixture)
    fill_value = pd.Series([fill_value], dtype=fill_dtype)[0]
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_timedelta64_with_any(timedelta64_dtype, any_numpy_dtype_reduced):
    dtype = np.dtype(timedelta64_dtype)
    fill_dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    if is_timedelta64_dtype(fill_dtype):
        expected_dtype = dtype
        exp_val_for_scalar = pd.Timedelta(fill_value).to_timedelta64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

@pytest.mark.parametrize('fill_value', [pd.Timedelta(days=1), np.timedelta64(24, 'h'), datetime.timedelta(1)], ids=['pd.Timedelta', 'np.timedelta64', 'datetime.timedelta'])
def test_maybe_promote_any_with_timedelta64(any_numpy_dtype_reduced, timedelta64_dtype, fill_value):
    dtype = np.dtype(any_numpy_dtype_reduced)
    if is_timedelta64_dtype(dtype):
        expected_dtype = dtype
        exp_val_for_scalar = pd.Timedelta(fill_value).to_timedelta64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_string_with_any(string_dtype, any_numpy_dtype_reduced):
    dtype = np.dtype(string_dtype)
    fill_dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_any_with_string(any_numpy_dtype_reduced, string_dtype):
    dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = 'abc'
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_object_with_any(object_dtype, any_numpy_dtype_reduced):
    dtype = np.dtype(object_dtype)
    fill_dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_any_with_object(any_numpy_dtype_reduced, object_dtype):
    dtype = np.dtype(any_numpy_dtype_reduced)
    fill_value = pd.DateOffset(1)
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

def test_maybe_promote_any_numpy_dtype_with_na(any_numpy_dtype_reduced, nulls_fixture):
    fill_value = nulls_fixture
    dtype = np.dtype(any_numpy_dtype_reduced)
    if (is_integer_dtype(dtype) and (fill_value is not NaT)):
        expected_dtype = np.float64
        exp_val_for_scalar = np.nan
    elif (is_object_dtype(dtype) and (fill_value is NaT)):
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    elif is_datetime_or_timedelta_dtype(dtype):
        expected_dtype = dtype
        exp_val_for_scalar = dtype.type('NaT', 'ns')
    elif (fill_value is NaT):
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = NaT
    elif (is_float_dtype(dtype) or is_complex_dtype(dtype)):
        expected_dtype = dtype
        exp_val_for_scalar = np.nan
    else:
        expected_dtype = np.dtype(object)
        if (fill_value is pd.NA):
            exp_val_for_scalar = pd.NA
        else:
            exp_val_for_scalar = np.nan
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)

@pytest.mark.parametrize('dim', [0, 2, 3])
def test_maybe_promote_dimensions(any_numpy_dtype_reduced, dim):
    dtype = np.dtype(any_numpy_dtype_reduced)
    fill_array = np.array(1, dtype=dtype)
    for _ in range(dim):
        fill_array = np.expand_dims(fill_array, 0)
    if (dtype != object):
        with pytest.raises(ValueError, match='fill_value must be a scalar'):
            maybe_promote(dtype, np.array([1], dtype=dtype))
        with pytest.raises(ValueError, match='fill_value must be a scalar'):
            maybe_promote(dtype, fill_array)
    else:
        (expected_dtype, expected_missing_value) = maybe_promote(dtype, np.array([1], dtype=dtype))
        (result_dtype, result_missing_value) = maybe_promote(dtype, fill_array)
        assert (result_dtype == expected_dtype)
        _assert_match(result_missing_value, expected_missing_value)
