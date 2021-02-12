
'\nRoutines for casting.\n'
from contextlib import suppress
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Sized, Tuple, Type, Union, cast
import warnings
import numpy as np
from pandas._libs import lib, missing as libmissing, tslib
from pandas._libs.tslibs import NaT, OutOfBoundsDatetime, Period, Timedelta, Timestamp, conversion, iNaT, ints_to_pydatetime
from pandas._libs.tslibs.timezones import tz_compare
from pandas._typing import AnyArrayLike, ArrayLike, Dtype, DtypeObj, Scalar
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.common import DT64NS_DTYPE, POSSIBLY_CAST_DTYPES, TD64NS_DTYPE, ensure_int8, ensure_int16, ensure_int32, ensure_int64, ensure_object, ensure_str, is_bool, is_bool_dtype, is_categorical_dtype, is_complex, is_complex_dtype, is_datetime64_dtype, is_datetime64_ns_dtype, is_datetime64tz_dtype, is_dtype_equal, is_extension_array_dtype, is_float, is_float_dtype, is_integer, is_integer_dtype, is_numeric_dtype, is_object_dtype, is_scalar, is_sparse, is_string_dtype, is_timedelta64_dtype, is_timedelta64_ns_dtype, is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, ExtensionDtype, IntervalDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCExtensionArray, ABCIndex, ABCSeries
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import is_valid_nat_for_dtype, isna, notna
if TYPE_CHECKING:
    from pandas import Series
    from pandas.core.arrays import DatetimeArray, ExtensionArray
_int8_max = np.iinfo(np.int8).max
_int16_max = np.iinfo(np.int16).max
_int32_max = np.iinfo(np.int32).max
_int64_max = np.iinfo(np.int64).max

def maybe_convert_platform(values):
    ' try to do platform conversion, allow ndarray or list here '
    if isinstance(values, (list, tuple, range)):
        values = construct_1d_object_array_from_listlike(values)
    if (getattr(values, 'dtype', None) == np.object_):
        if hasattr(values, '_values'):
            values = values._values
        values = lib.maybe_convert_objects(values)
    return values

def is_nested_object(obj):
    '\n    return a boolean if we have a nested object, e.g. a Series with 1 or\n    more Series elements\n\n    This may not be necessarily be performant.\n\n    '
    if (isinstance(obj, ABCSeries) and is_object_dtype(obj.dtype)):
        if any((isinstance(v, ABCSeries) for v in obj._values)):
            return True
    return False

def maybe_box_datetimelike(value, dtype=None):
    '\n    Cast scalar to Timestamp or Timedelta if scalar is datetime-like\n    and dtype is not object.\n\n    Parameters\n    ----------\n    value : scalar\n    dtype : Dtype, optional\n\n    Returns\n    -------\n    scalar\n    '
    if (dtype == object):
        pass
    elif isinstance(value, (np.datetime64, datetime)):
        value = Timestamp(value)
    elif isinstance(value, (np.timedelta64, timedelta)):
        value = Timedelta(value)
    return value

def maybe_unbox_datetimelike(value, dtype):
    '\n    Convert a Timedelta or Timestamp to timedelta64 or datetime64 for setting\n    into a numpy array.  Failing to unbox would risk dropping nanoseconds.\n\n    Notes\n    -----\n    Caller is responsible for checking dtype.kind in ["m", "M"]\n    '
    if is_valid_nat_for_dtype(value, dtype):
        value = dtype.type('NaT', 'ns')
    elif isinstance(value, Timestamp):
        if (value.tz is None):
            value = value.to_datetime64()
    elif isinstance(value, Timedelta):
        value = value.to_timedelta64()
    _disallow_mismatched_datetimelike(value, dtype)
    return value

def _disallow_mismatched_datetimelike(value, dtype):
    '\n    numpy allows np.array(dt64values, dtype="timedelta64[ns]") and\n    vice-versa, but we do not want to allow this, so we need to\n    check explicitly\n    '
    vdtype = getattr(value, 'dtype', None)
    if (vdtype is None):
        return
    elif (((vdtype.kind == 'm') and (dtype.kind == 'M')) or ((vdtype.kind == 'M') and (dtype.kind == 'm'))):
        raise TypeError(f'Cannot cast {repr(value)} to {dtype}')

def maybe_downcast_to_dtype(result, dtype):
    '\n    try to cast to the specified dtype (e.g. convert back to bool/int\n    or could be an astype of float64->float32\n    '
    do_round = False
    if is_scalar(result):
        return result
    elif isinstance(result, ABCDataFrame):
        return result
    if isinstance(dtype, str):
        if (dtype == 'infer'):
            inferred_type = lib.infer_dtype(ensure_object(result), skipna=False)
            if (inferred_type == 'boolean'):
                dtype = 'bool'
            elif (inferred_type == 'integer'):
                dtype = 'int64'
            elif (inferred_type == 'datetime64'):
                dtype = 'datetime64[ns]'
            elif (inferred_type == 'timedelta64'):
                dtype = 'timedelta64[ns]'
            elif (inferred_type == 'floating'):
                dtype = 'int64'
                if issubclass(result.dtype.type, np.number):
                    do_round = True
            else:
                dtype = 'object'
        dtype = np.dtype(dtype)
    elif (dtype.type is Period):
        from pandas.core.arrays import PeriodArray
        with suppress(TypeError):
            return PeriodArray(result, freq=dtype.freq)
    converted = maybe_downcast_numeric(result, dtype, do_round)
    if (converted is not result):
        return converted
    if ((dtype.kind in ['M', 'm']) and (result.dtype.kind in ['i', 'f'])):
        if isinstance(dtype, DatetimeTZDtype):
            i8values = result.astype('i8', copy=False)
            cls = dtype.construct_array_type()
            result = cls._simple_new(i8values, dtype=dtype)
        else:
            result = result.astype(dtype)
    return result

def maybe_downcast_numeric(result, dtype, do_round=False):
    '\n    Subset of maybe_downcast_to_dtype restricted to numeric dtypes.\n\n    Parameters\n    ----------\n    result : ndarray or ExtensionArray\n    dtype : np.dtype or ExtensionDtype\n    do_round : bool\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n    '
    if (not isinstance(dtype, np.dtype)):
        return result

    def trans(x):
        if do_round:
            return x.round()
        return x
    if (dtype.kind == result.dtype.kind):
        if ((result.dtype.itemsize <= dtype.itemsize) and result.size):
            return result
    if (is_bool_dtype(dtype) or is_integer_dtype(dtype)):
        if (not result.size):
            return trans(result).astype(dtype)
        r = result.ravel()
        arr = np.array([r[0]])
        if isna(arr).any():
            return result
        elif (not isinstance(r[0], (np.integer, np.floating, int, float, bool))):
            return result
        if (issubclass(result.dtype.type, (np.object_, np.number)) and notna(result).all()):
            new_result = trans(result).astype(dtype)
            if ((new_result.dtype.kind == 'O') or (result.dtype.kind == 'O')):
                if (new_result == result).all():
                    return new_result
            elif np.allclose(new_result, result, rtol=0):
                return new_result
    elif (issubclass(dtype.type, np.floating) and (not is_bool_dtype(result.dtype)) and (not is_string_dtype(result.dtype))):
        return result.astype(dtype)
    return result

def maybe_cast_result(result, obj, numeric_only=False, how=''):
    '\n    Try casting result to a different type if appropriate\n\n    Parameters\n    ----------\n    result : array-like\n        Result to cast.\n    obj : Series\n        Input Series from which result was calculated.\n    numeric_only : bool, default False\n        Whether to cast only numerics or datetimes as well.\n    how : str, default ""\n        How the result was computed.\n\n    Returns\n    -------\n    result : array-like\n        result maybe casted to the dtype.\n    '
    dtype = obj.dtype
    dtype = maybe_cast_result_dtype(dtype, how)
    assert (not is_scalar(result))
    if (is_extension_array_dtype(dtype) and (not is_categorical_dtype(dtype)) and (dtype.kind != 'M')):
        cls = dtype.construct_array_type()
        result = maybe_cast_to_extension_array(cls, result, dtype=dtype)
    elif ((numeric_only and is_numeric_dtype(dtype)) or (not numeric_only)):
        result = maybe_downcast_to_dtype(result, dtype)
    return result

def maybe_cast_result_dtype(dtype, how):
    '\n    Get the desired dtype of a result based on the\n    input dtype and how it was computed.\n\n    Parameters\n    ----------\n    dtype : DtypeObj\n        Input dtype.\n    how : str\n        How the result was computed.\n\n    Returns\n    -------\n    DtypeObj\n        The desired dtype of the result.\n    '
    from pandas.core.arrays.boolean import BooleanDtype
    from pandas.core.arrays.floating import Float64Dtype
    from pandas.core.arrays.integer import Int64Dtype, _IntegerDtype
    if (how in ['add', 'cumsum', 'sum', 'prod']):
        if (dtype == np.dtype(bool)):
            return np.dtype(np.int64)
        elif isinstance(dtype, (BooleanDtype, _IntegerDtype)):
            return Int64Dtype()
    elif ((how in ['mean', 'median', 'var']) and isinstance(dtype, (BooleanDtype, _IntegerDtype))):
        return Float64Dtype()
    return dtype

def maybe_cast_to_extension_array(cls, obj, dtype=None):
    '\n    Call to `_from_sequence` that returns the object unchanged on Exception.\n\n    Parameters\n    ----------\n    cls : class, subclass of ExtensionArray\n    obj : arraylike\n        Values to pass to cls._from_sequence\n    dtype : ExtensionDtype, optional\n\n    Returns\n    -------\n    ExtensionArray or obj\n    '
    from pandas.core.arrays.string_ import StringArray
    from pandas.core.arrays.string_arrow import ArrowStringArray
    assert isinstance(cls, type), f'must pass a type: {cls}'
    assertion_msg = f'must pass a subclass of ExtensionArray: {cls}'
    assert issubclass(cls, ABCExtensionArray), assertion_msg
    if (issubclass(cls, (StringArray, ArrowStringArray)) and (lib.infer_dtype(obj) != 'string')):
        return obj
    try:
        result = cls._from_sequence(obj, dtype=dtype)
    except Exception:
        result = obj
    return result

def maybe_upcast_putmask(result, mask):
    '\n    A safe version of putmask that potentially upcasts the result.\n\n    The result is replaced with the first N elements of other,\n    where N is the number of True values in mask.\n    If the length of other is shorter than N, other will be repeated.\n\n    Parameters\n    ----------\n    result : ndarray\n        The destination array. This will be mutated in-place if no upcasting is\n        necessary.\n    mask : boolean ndarray\n\n    Returns\n    -------\n    result : ndarray\n\n    Examples\n    --------\n    >>> arr = np.arange(1, 6)\n    >>> mask = np.array([False, True, False, True, True])\n    >>> result = maybe_upcast_putmask(arr, mask)\n    >>> result\n    array([ 1., nan,  3., nan, nan])\n    '
    if (not isinstance(result, np.ndarray)):
        raise ValueError('The result input must be a ndarray.')
    if mask.any():
        (new_dtype, _) = maybe_promote(result.dtype, np.nan)
        if (new_dtype != result.dtype):
            result = result.astype(new_dtype, copy=True)
        np.place(result, mask, np.nan)
    return result

def maybe_promote(dtype, fill_value=np.nan):
    '\n    Find the minimal dtype that can hold both the given dtype and fill_value.\n\n    Parameters\n    ----------\n    dtype : np.dtype or ExtensionDtype\n    fill_value : scalar, default np.nan\n\n    Returns\n    -------\n    dtype\n        Upcasted from dtype argument if necessary.\n    fill_value\n        Upcasted from fill_value argument if necessary.\n\n    Raises\n    ------\n    ValueError\n        If fill_value is a non-scalar and dtype is not object.\n    '
    if ((not is_scalar(fill_value)) and (not is_object_dtype(dtype))):
        raise ValueError('fill_value must be a scalar')
    if isinstance(fill_value, np.ndarray):
        if issubclass(fill_value.dtype.type, (np.datetime64, np.timedelta64)):
            fill_value = fill_value.dtype.type('NaT', 'ns')
        else:
            if (fill_value.dtype == np.object_):
                dtype = np.dtype(np.object_)
            fill_value = np.nan
        if ((dtype == np.object_) or (dtype.kind in ['U', 'S'])):
            fill_value = np.nan
            dtype = np.dtype(np.object_)
    if issubclass(dtype.type, np.datetime64):
        if (isinstance(fill_value, datetime) and (fill_value.tzinfo is not None)):
            dtype = np.dtype(np.object_)
        elif (is_integer(fill_value) or (is_float(fill_value) and (not isna(fill_value)))):
            dtype = np.dtype(np.object_)
        elif is_valid_nat_for_dtype(fill_value, dtype):
            fill_value = np.datetime64('NaT', 'ns')
        else:
            try:
                fill_value = Timestamp(fill_value).to_datetime64()
            except (TypeError, ValueError):
                dtype = np.dtype(np.object_)
    elif issubclass(dtype.type, np.timedelta64):
        if (is_integer(fill_value) or (is_float(fill_value) and (not np.isnan(fill_value))) or isinstance(fill_value, str)):
            dtype = np.dtype(np.object_)
        elif is_valid_nat_for_dtype(fill_value, dtype):
            fill_value = np.timedelta64('NaT', 'ns')
        else:
            try:
                fv = Timedelta(fill_value)
            except ValueError:
                dtype = np.dtype(np.object_)
            else:
                if (fv is NaT):
                    fill_value = np.timedelta64('NaT', 'ns')
                else:
                    fill_value = fv.to_timedelta64()
    elif is_datetime64tz_dtype(dtype):
        if isna(fill_value):
            fill_value = NaT
        elif (not isinstance(fill_value, datetime)):
            dtype = np.dtype(np.object_)
        elif (fill_value.tzinfo is None):
            dtype = np.dtype(np.object_)
        elif (not tz_compare(fill_value.tzinfo, dtype.tz)):
            dtype = np.dtype(np.object_)
    elif (is_extension_array_dtype(dtype) and isna(fill_value)):
        fill_value = dtype.na_value
    elif is_float(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, np.integer):
            dtype = np.dtype(np.float64)
        elif (dtype.kind == 'f'):
            mst = np.min_scalar_type(fill_value)
            if (mst > dtype):
                dtype = mst
        elif (dtype.kind == 'c'):
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)
    elif is_bool(fill_value):
        if (not issubclass(dtype.type, np.bool_)):
            dtype = np.dtype(np.object_)
    elif is_integer(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, np.integer):
            if (not np.can_cast(fill_value, dtype)):
                mst = np.min_scalar_type(fill_value)
                dtype = np.promote_types(dtype, mst)
                if (dtype.kind == 'f'):
                    dtype = np.dtype(np.object_)
    elif is_complex(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, (np.integer, np.floating)):
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)
        elif (dtype.kind == 'c'):
            mst = np.min_scalar_type(fill_value)
            if (mst > dtype):
                dtype = mst
    elif ((fill_value is None) or (fill_value is libmissing.NA)):
        if (is_float_dtype(dtype) or is_complex_dtype(dtype)):
            fill_value = np.nan
        elif is_integer_dtype(dtype):
            dtype = np.float64
            fill_value = np.nan
        else:
            dtype = np.dtype(np.object_)
            if (fill_value is not libmissing.NA):
                fill_value = np.nan
    else:
        dtype = np.dtype(np.object_)
    if is_extension_array_dtype(dtype):
        pass
    elif issubclass(np.dtype(dtype).type, (bytes, str)):
        dtype = np.dtype(np.object_)
    fill_value = _ensure_dtype_type(fill_value, dtype)
    return (dtype, fill_value)

def _ensure_dtype_type(value, dtype):
    '\n    Ensure that the given value is an instance of the given dtype.\n\n    e.g. if out dtype is np.complex64_, we should have an instance of that\n    as opposed to a python complex object.\n\n    Parameters\n    ----------\n    value : object\n    dtype : np.dtype or ExtensionDtype\n\n    Returns\n    -------\n    object\n    '
    if is_extension_array_dtype(dtype):
        return value
    elif (dtype == np.object_):
        return value
    elif isna(value):
        return value
    return dtype.type(value)

def infer_dtype_from(val, pandas_dtype=False):
    '\n    Interpret the dtype from a scalar or array.\n\n    Parameters\n    ----------\n    val : object\n    pandas_dtype : bool, default False\n        whether to infer dtype including pandas extension types.\n        If False, scalar/array belongs to pandas extension types is inferred as\n        object\n    '
    if (not is_list_like(val)):
        return infer_dtype_from_scalar(val, pandas_dtype=pandas_dtype)
    return infer_dtype_from_array(val, pandas_dtype=pandas_dtype)

def infer_dtype_from_scalar(val, pandas_dtype=False):
    '\n    Interpret the dtype from a scalar.\n\n    Parameters\n    ----------\n    pandas_dtype : bool, default False\n        whether to infer dtype including pandas extension types.\n        If False, scalar belongs to pandas extension types is inferred as\n        object\n    '
    dtype: DtypeObj = np.dtype(object)
    if isinstance(val, np.ndarray):
        msg = 'invalid ndarray passed to infer_dtype_from_scalar'
        if (val.ndim != 0):
            raise ValueError(msg)
        dtype = val.dtype
        val = lib.item_from_zerodim(val)
    elif isinstance(val, str):
        dtype = np.dtype(object)
    elif isinstance(val, (np.datetime64, datetime)):
        try:
            val = Timestamp(val)
        except OutOfBoundsDatetime:
            return (np.dtype(object), val)
        if ((val is NaT) or (val.tz is None)):
            dtype = np.dtype('M8[ns]')
        elif pandas_dtype:
            dtype = DatetimeTZDtype(unit='ns', tz=val.tz)
        else:
            return (np.dtype(object), val)
        val = val.value
    elif isinstance(val, (np.timedelta64, timedelta)):
        val = Timedelta(val).value
        dtype = np.dtype('m8[ns]')
    elif is_bool(val):
        dtype = np.dtype(np.bool_)
    elif is_integer(val):
        if isinstance(val, np.integer):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.int64)
        try:
            np.array(val, dtype=dtype)
        except OverflowError:
            dtype = np.array(val).dtype
    elif is_float(val):
        if isinstance(val, np.floating):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.float64)
    elif is_complex(val):
        dtype = np.dtype(np.complex_)
    elif pandas_dtype:
        if lib.is_period(val):
            dtype = PeriodDtype(freq=val.freq)
        elif lib.is_interval(val):
            subtype = infer_dtype_from_scalar(val.left, pandas_dtype=True)[0]
            dtype = IntervalDtype(subtype=subtype)
    return (dtype, val)

def dict_compat(d):
    '\n    Convert datetimelike-keyed dicts to a Timestamp-keyed dict.\n\n    Parameters\n    ----------\n    d: dict-like object\n\n    Returns\n    -------\n    dict\n\n    '
    return {maybe_box_datetimelike(key): value for (key, value) in d.items()}

def infer_dtype_from_array(arr, pandas_dtype=False):
    "\n    Infer the dtype from an array.\n\n    Parameters\n    ----------\n    arr : array\n    pandas_dtype : bool, default False\n        whether to infer dtype including pandas extension types.\n        If False, array belongs to pandas extension types\n        is inferred as object\n\n    Returns\n    -------\n    tuple (numpy-compat/pandas-compat dtype, array)\n\n    Notes\n    -----\n    if pandas_dtype=False. these infer to numpy dtypes\n    exactly with the exception that mixed / object dtypes\n    are not coerced by stringifying or conversion\n\n    if pandas_dtype=True. datetime64tz-aware/categorical\n    types will retain there character.\n\n    Examples\n    --------\n    >>> np.asarray([1, '1'])\n    array(['1', '1'], dtype='<U21')\n\n    >>> infer_dtype_from_array([1, '1'])\n    (dtype('O'), [1, '1'])\n    "
    if isinstance(arr, np.ndarray):
        return (arr.dtype, arr)
    if (not is_list_like(arr)):
        raise TypeError("'arr' must be list-like")
    if (pandas_dtype and is_extension_array_dtype(arr)):
        return (arr.dtype, arr)
    elif isinstance(arr, ABCSeries):
        return (arr.dtype, np.asarray(arr))
    inferred = lib.infer_dtype(arr, skipna=False)
    if (inferred in ['string', 'bytes', 'mixed', 'mixed-integer']):
        return (np.dtype(np.object_), arr)
    arr = np.asarray(arr)
    return (arr.dtype, arr)

def maybe_infer_dtype_type(element):
    '\n    Try to infer an object\'s dtype, for use in arithmetic ops.\n\n    Uses `element.dtype` if that\'s available.\n    Objects implementing the iterator protocol are cast to a NumPy array,\n    and from there the array\'s type is used.\n\n    Parameters\n    ----------\n    element : object\n        Possibly has a `.dtype` attribute, and possibly the iterator\n        protocol.\n\n    Returns\n    -------\n    tipo : type\n\n    Examples\n    --------\n    >>> from collections import namedtuple\n    >>> Foo = namedtuple("Foo", "dtype")\n    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))\n    dtype(\'int64\')\n    '
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def maybe_upcast(values, fill_value=np.nan, copy=False):
    '\n    Provide explicit type promotion and coercion.\n\n    Parameters\n    ----------\n    values : np.ndarray\n        The array that we may want to upcast.\n    fill_value : what we want to fill with\n    copy : bool, default True\n        If True always make a copy even if no upcast is required.\n\n    Returns\n    -------\n    values: np.ndarray\n        the original array, possibly upcast\n    fill_value:\n        the fill value, possibly upcast\n    '
    (new_dtype, fill_value) = maybe_promote(values.dtype, fill_value)
    values = values.astype(new_dtype, copy=copy)
    return (values, fill_value)

def invalidate_string_dtypes(dtype_set):
    '\n    Change string like dtypes to object for\n    ``DataFrame.select_dtypes()``.\n    '
    non_string_dtypes = (dtype_set - {np.dtype('S').type, np.dtype('<U').type})
    if (non_string_dtypes != dtype_set):
        raise TypeError("string dtypes are not allowed, use 'object' instead")

def coerce_indexer_dtype(indexer, categories):
    ' coerce the indexer input array to the smallest dtype possible '
    length = len(categories)
    if (length < _int8_max):
        return ensure_int8(indexer)
    elif (length < _int16_max):
        return ensure_int16(indexer)
    elif (length < _int32_max):
        return ensure_int32(indexer)
    return ensure_int64(indexer)

def astype_dt64_to_dt64tz(values, dtype, copy, via_utc=False):
    from pandas.core.construction import ensure_wrapped_if_datetimelike
    values = ensure_wrapped_if_datetimelike(values)
    values = cast('DatetimeArray', values)
    aware = isinstance(dtype, DatetimeTZDtype)
    if via_utc:
        assert ((values.tz is None) and aware)
        dtype = cast(DatetimeTZDtype, dtype)
        if copy:
            values = values.copy()
        return values.tz_localize('UTC').tz_convert(dtype.tz)
    else:
        if ((values.tz is None) and aware):
            dtype = cast(DatetimeTZDtype, dtype)
            return values.tz_localize(dtype.tz)
        elif aware:
            dtype = cast(DatetimeTZDtype, dtype)
            result = values.tz_convert(dtype.tz)
            if copy:
                result = result.copy()
            return result
        elif ((values.tz is not None) and (not aware)):
            result = values.tz_convert('UTC').tz_localize(None)
            if copy:
                result = result.copy()
            return result
        raise NotImplementedError('dtype_equal case should be handled elsewhere')

def astype_td64_unit_conversion(values, dtype, copy):
    '\n    By pandas convention, converting to non-nano timedelta64\n    returns an int64-dtyped array with ints representing multiples\n    of the desired timedelta unit.  This is essentially division.\n\n    Parameters\n    ----------\n    values : np.ndarray[timedelta64[ns]]\n    dtype : np.dtype\n        timedelta64 with unit not-necessarily nano\n    copy : bool\n\n    Returns\n    -------\n    np.ndarray\n    '
    if is_dtype_equal(values.dtype, dtype):
        if copy:
            return values.copy()
        return values
    result = values.astype(dtype, copy=False)
    result = result.astype(np.float64)
    mask = isna(values)
    np.putmask(result, mask, np.nan)
    return result

def astype_nansafe(arr, dtype, copy=True, skipna=False):
    "\n    Cast the elements of an array to a given dtype a nan-safe manner.\n\n    Parameters\n    ----------\n    arr : ndarray\n    dtype : np.dtype or ExtensionDtype\n    copy : bool, default True\n        If False, a view will be attempted but may fail, if\n        e.g. the item sizes don't align.\n    skipna: bool, default False\n        Whether or not we should skip NaN when casting as a string-type.\n\n    Raises\n    ------\n    ValueError\n        The dtype was a datetime64/timedelta64 dtype, but it had no unit.\n    "
    if (arr.ndim > 1):
        flags = arr.flags
        flat = arr.ravel('K')
        result = astype_nansafe(flat, dtype, copy=copy, skipna=skipna)
        order = ('F' if flags.f_contiguous else 'C')
        return result.reshape(arr.shape, order=order)
    arr = np.atleast_1d(arr)
    if isinstance(dtype, ExtensionDtype):
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)
    elif (not isinstance(dtype, np.dtype)):
        raise ValueError('dtype must be np.dtype or ExtensionDtype')
    if ((arr.dtype.kind in ['m', 'M']) and (issubclass(dtype.type, str) or (dtype == object))):
        from pandas.core.construction import ensure_wrapped_if_datetimelike
        arr = ensure_wrapped_if_datetimelike(arr)
        return arr.astype(dtype, copy=copy)
    if issubclass(dtype.type, str):
        return lib.ensure_string_array(arr, skipna=skipna, convert_na_value=False)
    elif is_datetime64_dtype(arr):
        if (dtype == np.int64):
            warnings.warn(f'casting {arr.dtype} values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.', FutureWarning, stacklevel=7)
            if isna(arr).any():
                raise ValueError('Cannot convert NaT values to integer')
            return arr.view(dtype)
        if (dtype.kind == 'M'):
            return arr.astype(dtype)
        raise TypeError(f'cannot astype a datetimelike from [{arr.dtype}] to [{dtype}]')
    elif is_timedelta64_dtype(arr):
        if (dtype == np.int64):
            warnings.warn(f'casting {arr.dtype} values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.', FutureWarning, stacklevel=7)
            if isna(arr).any():
                raise ValueError('Cannot convert NaT values to integer')
            return arr.view(dtype)
        elif (dtype.kind == 'm'):
            return astype_td64_unit_conversion(arr, dtype, copy=copy)
        raise TypeError(f'cannot astype a timedelta from [{arr.dtype}] to [{dtype}]')
    elif (np.issubdtype(arr.dtype, np.floating) and np.issubdtype(dtype, np.integer)):
        if (not np.isfinite(arr).all()):
            raise ValueError('Cannot convert non-finite values (NA or inf) to integer')
    elif is_object_dtype(arr):
        if np.issubdtype(dtype.type, np.integer):
            return lib.astype_intsafe(arr, dtype)
        elif is_datetime64_dtype(dtype):
            from pandas import to_datetime
            return astype_nansafe(to_datetime(arr).values, dtype, copy=copy)
        elif is_timedelta64_dtype(dtype):
            from pandas import to_timedelta
            return astype_nansafe(to_timedelta(arr)._values, dtype, copy=copy)
    if (dtype.name in ('datetime64', 'timedelta64')):
        msg = f"The '{dtype.name}' dtype has no unit. Please pass in '{dtype.name}[ns]' instead."
        raise ValueError(msg)
    if (copy or is_object_dtype(arr.dtype) or is_object_dtype(dtype)):
        return arr.astype(dtype, copy=True)
    return arr.astype(dtype, copy=copy)

def soft_convert_objects(values, datetime=True, numeric=True, timedelta=True, copy=True):
    '\n    Try to coerce datetime, timedelta, and numeric object-dtype columns\n    to inferred dtype.\n\n    Parameters\n    ----------\n    values : np.ndarray[object]\n    datetime : bool, default True\n    numeric: bool, default True\n    timedelta : bool, default True\n    copy : bool, default True\n\n    Returns\n    -------\n    np.ndarray\n    '
    validate_bool_kwarg(datetime, 'datetime')
    validate_bool_kwarg(numeric, 'numeric')
    validate_bool_kwarg(timedelta, 'timedelta')
    validate_bool_kwarg(copy, 'copy')
    conversion_count = sum((datetime, numeric, timedelta))
    if (conversion_count == 0):
        raise ValueError('At least one of datetime, numeric or timedelta must be True.')
    if (datetime or timedelta):
        try:
            values = lib.maybe_convert_objects(values, convert_datetime=datetime, convert_timedelta=timedelta)
        except OutOfBoundsDatetime:
            return values
    if (numeric and is_object_dtype(values.dtype)):
        converted = lib.maybe_convert_numeric(values, set(), coerce_numeric=True)
        values = (converted if (not isna(converted).all()) else values)
        values = (values.copy() if copy else values)
    return values

def convert_dtypes(input_array, convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True):
    '\n    Convert objects to best possible type, and optionally,\n    to types supporting ``pd.NA``.\n\n    Parameters\n    ----------\n    input_array : ExtensionArray, Index, Series or np.ndarray\n    convert_string : bool, default True\n        Whether object dtypes should be converted to ``StringDtype()``.\n    convert_integer : bool, default True\n        Whether, if possible, conversion can be done to integer extension types.\n    convert_boolean : bool, defaults True\n        Whether object dtypes should be converted to ``BooleanDtypes()``.\n    convert_floating : bool, defaults True\n        Whether, if possible, conversion can be done to floating extension types.\n        If `convert_integer` is also True, preference will be give to integer\n        dtypes if the floats can be faithfully casted to integers.\n\n    Returns\n    -------\n    dtype\n        new dtype\n    '
    is_extension = is_extension_array_dtype(input_array.dtype)
    if ((convert_string or convert_integer or convert_boolean or convert_floating) and (not is_extension)):
        try:
            inferred_dtype = lib.infer_dtype(input_array)
        except ValueError:
            inferred_dtype = input_array.dtype
        if ((not convert_string) and is_string_dtype(inferred_dtype)):
            inferred_dtype = input_array.dtype
        if convert_integer:
            target_int_dtype = 'Int64'
            if is_integer_dtype(input_array.dtype):
                from pandas.core.arrays.integer import INT_STR_TO_DTYPE
                inferred_dtype = INT_STR_TO_DTYPE.get(input_array.dtype.name, target_int_dtype)
            if ((not is_integer_dtype(input_array.dtype)) and is_numeric_dtype(input_array.dtype)):
                inferred_dtype = target_int_dtype
        elif is_integer_dtype(inferred_dtype):
            inferred_dtype = input_array.dtype
        if convert_floating:
            if ((not is_integer_dtype(input_array.dtype)) and is_numeric_dtype(input_array.dtype)):
                from pandas.core.arrays.floating import FLOAT_STR_TO_DTYPE
                inferred_float_dtype = FLOAT_STR_TO_DTYPE.get(input_array.dtype.name, 'Float64')
                if convert_integer:
                    arr = input_array[notna(input_array)]
                    if (arr.astype(int) == arr).all():
                        inferred_dtype = 'Int64'
                    else:
                        inferred_dtype = inferred_float_dtype
                else:
                    inferred_dtype = inferred_float_dtype
        elif is_float_dtype(inferred_dtype):
            inferred_dtype = input_array.dtype
        if convert_boolean:
            if is_bool_dtype(input_array.dtype):
                inferred_dtype = 'boolean'
        elif (isinstance(inferred_dtype, str) and (inferred_dtype == 'boolean')):
            inferred_dtype = input_array.dtype
    else:
        inferred_dtype = input_array.dtype
    return inferred_dtype

def maybe_castable(arr):
    assert isinstance(arr, np.ndarray)
    kind = arr.dtype.kind
    if (kind == 'M'):
        return is_datetime64_ns_dtype(arr.dtype)
    elif (kind == 'm'):
        return is_timedelta64_ns_dtype(arr.dtype)
    return (arr.dtype.name not in POSSIBLY_CAST_DTYPES)

def maybe_infer_to_datetimelike(value, convert_dates=False):
    "\n    we might have a array (or single object) that is datetime like,\n    and no dtype is passed don't change the value unless we find a\n    datetime/timedelta set\n\n    this is pretty strict in that a datetime/timedelta is REQUIRED\n    in addition to possible nulls/string likes\n\n    Parameters\n    ----------\n    value : np.array / Series / Index / list-like\n    convert_dates : bool, default False\n       if True try really hard to convert dates (such as datetime.date), other\n       leave inferred dtype 'date' alone\n\n    "
    if isinstance(value, (ABCIndex, ABCExtensionArray)):
        if (not is_object_dtype(value.dtype)):
            raise ValueError('array-like value must be object-dtype')
    v = value
    if (not is_list_like(v)):
        v = [v]
    v = np.array(v, copy=False)
    if (not is_object_dtype(v)):
        return value
    shape = v.shape
    if (v.ndim != 1):
        v = v.ravel()
    if (not len(v)):
        return value

    def try_datetime(v):
        try:
            v = tslib.array_to_datetime(v, require_iso8601=True, errors='raise')[0]
        except ValueError:
            from pandas import DatetimeIndex
            try:
                (values, tz) = conversion.datetime_to_datetime64(v)
            except (ValueError, TypeError):
                pass
            else:
                return DatetimeIndex(values).tz_localize('UTC').tz_convert(tz=tz)
        except TypeError:
            pass
        return v.reshape(shape)

    def try_timedelta(v):
        from pandas import to_timedelta
        try:
            td_values = to_timedelta(v)
        except ValueError:
            return v.reshape(shape)
        else:
            return np.asarray(td_values).reshape(shape)
    inferred_type = lib.infer_datetimelike_array(ensure_object(v))
    if ((inferred_type == 'date') and convert_dates):
        value = try_datetime(v)
    elif (inferred_type == 'datetime'):
        value = try_datetime(v)
    elif (inferred_type == 'timedelta'):
        value = try_timedelta(v)
    elif (inferred_type == 'nat'):
        if isna(v).all():
            value = try_datetime(v)
        else:
            value = try_timedelta(v)
            if (lib.infer_dtype(value, skipna=False) in ['mixed']):
                value = try_datetime(v)
    return value

def maybe_cast_to_datetime(value, dtype):
    '\n    try to cast the array/value to a datetimelike dtype, converting float\n    nan to iNaT\n    '
    from pandas.core.tools.datetimes import to_datetime
    from pandas.core.tools.timedeltas import to_timedelta
    if (not is_list_like(value)):
        raise TypeError('value must be listlike')
    if (dtype is not None):
        is_datetime64 = is_datetime64_dtype(dtype)
        is_datetime64tz = is_datetime64tz_dtype(dtype)
        is_timedelta64 = is_timedelta64_dtype(dtype)
        if (is_datetime64 or is_datetime64tz or is_timedelta64):
            msg = f"The '{dtype.name}' dtype has no unit. Please pass in '{dtype.name}[ns]' instead."
            if is_datetime64:
                dtype = getattr(dtype, 'subtype', dtype)
                if (not is_dtype_equal(dtype, DT64NS_DTYPE)):
                    if (dtype <= np.dtype('M8[ns]')):
                        if (dtype.name == 'datetime64'):
                            raise ValueError(msg)
                        dtype = DT64NS_DTYPE
                    else:
                        raise TypeError(f'cannot convert datetimelike to dtype [{dtype}]')
            elif (is_timedelta64 and (not is_dtype_equal(dtype, TD64NS_DTYPE))):
                if (dtype <= np.dtype('m8[ns]')):
                    if (dtype.name == 'timedelta64'):
                        raise ValueError(msg)
                    dtype = TD64NS_DTYPE
                else:
                    raise TypeError(f'cannot convert timedeltalike to dtype [{dtype}]')
            if (not is_sparse(value)):
                value = np.array(value, copy=False)
                if (value.ndim == 0):
                    value = iNaT
                elif (np.prod(value.shape) or (not is_dtype_equal(value.dtype, dtype))):
                    _disallow_mismatched_datetimelike(value, dtype)
                    try:
                        if is_datetime64:
                            value = to_datetime(value, errors='raise')
                            if (value.tz is not None):
                                value = value.tz_localize(None)
                            value = value._values
                        elif is_datetime64tz:
                            is_dt_string = is_string_dtype(value.dtype)
                            value = to_datetime(value, errors='raise').array
                            if is_dt_string:
                                value = value.tz_localize(dtype.tz)
                            else:
                                value = value.tz_localize('UTC').tz_convert(dtype.tz)
                        elif is_timedelta64:
                            value = to_timedelta(value, errors='raise')._values
                    except OutOfBoundsDatetime:
                        raise
                    except (ValueError, TypeError):
                        pass
        elif (is_datetime64_dtype(getattr(value, 'dtype', None)) and (not is_datetime64_dtype(dtype))):
            if is_object_dtype(dtype):
                if (value.dtype != DT64NS_DTYPE):
                    value = value.astype(DT64NS_DTYPE)
                ints = np.asarray(value).view('i8')
                return ints_to_pydatetime(ints)
            raise TypeError(f'Cannot cast datetime64 to {dtype}')
    else:
        is_array = isinstance(value, np.ndarray)
        if (is_array and (value.dtype.kind in ['M', 'm'])):
            value = sanitize_to_nanoseconds(value)
        elif (not (is_array and (not (issubclass(value.dtype.type, np.integer) or (value.dtype == np.object_))))):
            value = maybe_infer_to_datetimelike(value)
    return value

def sanitize_to_nanoseconds(values):
    '\n    Safely convert non-nanosecond datetime64 or timedelta64 values to nanosecond.\n    '
    dtype = values.dtype
    if ((dtype.kind == 'M') and (dtype != DT64NS_DTYPE)):
        values = conversion.ensure_datetime64ns(values)
    elif ((dtype.kind == 'm') and (dtype != TD64NS_DTYPE)):
        values = conversion.ensure_timedelta64ns(values)
    return values

def find_common_type(types):
    '\n    Find a common data type among the given dtypes.\n\n    Parameters\n    ----------\n    types : list of dtypes\n\n    Returns\n    -------\n    pandas extension or numpy dtype\n\n    See Also\n    --------\n    numpy.find_common_type\n\n    '
    if (len(types) == 0):
        raise ValueError('no types given')
    first = types[0]
    if all((is_dtype_equal(first, t) for t in types[1:])):
        return first
    types = list(dict.fromkeys(types).keys())
    if any((isinstance(t, ExtensionDtype) for t in types)):
        for t in types:
            if isinstance(t, ExtensionDtype):
                res = t._get_common_dtype(types)
                if (res is not None):
                    return res
        return np.dtype('object')
    if all((is_datetime64_dtype(t) for t in types)):
        return np.dtype('datetime64[ns]')
    if all((is_timedelta64_dtype(t) for t in types)):
        return np.dtype('timedelta64[ns]')
    has_bools = any((is_bool_dtype(t) for t in types))
    if has_bools:
        for t in types:
            if (is_integer_dtype(t) or is_float_dtype(t) or is_complex_dtype(t)):
                return np.dtype('object')
    return np.find_common_type(types, [])

def construct_2d_arraylike_from_scalar(value, length, width, dtype, copy):
    if (dtype.kind in ['m', 'M']):
        value = maybe_unbox_datetimelike(value, dtype)
    try:
        arr = np.array(value, dtype=dtype, copy=copy)
    except (ValueError, TypeError) as err:
        raise TypeError(f'DataFrame constructor called with incompatible data and dtype: {err}') from err
    if (arr.ndim != 0):
        raise ValueError('DataFrame constructor not properly called!')
    shape = (length, width)
    return np.full(shape, arr)

def construct_1d_arraylike_from_scalar(value, length, dtype):
    '\n    create a np.ndarray / pandas type of specified shape and dtype\n    filled with values\n\n    Parameters\n    ----------\n    value : scalar value\n    length : int\n    dtype : pandas_dtype or np.dtype\n\n    Returns\n    -------\n    np.ndarray / pandas type of length, filled with value\n\n    '
    if (dtype is None):
        try:
            (dtype, value) = infer_dtype_from_scalar(value, pandas_dtype=True)
        except OutOfBoundsDatetime:
            dtype = np.dtype(object)
    if is_extension_array_dtype(dtype):
        cls = dtype.construct_array_type()
        subarr = cls._from_sequence(([value] * length), dtype=dtype)
    else:
        if (length and is_integer_dtype(dtype) and isna(value)):
            dtype = np.dtype('float64')
        elif (isinstance(dtype, np.dtype) and (dtype.kind in ('U', 'S'))):
            dtype = np.dtype('object')
            if (not isna(value)):
                value = ensure_str(value)
        elif (dtype.kind in ['M', 'm']):
            value = maybe_unbox_datetimelike(value, dtype)
        subarr = np.empty(length, dtype=dtype)
        subarr.fill(value)
    return subarr

def construct_1d_object_array_from_listlike(values):
    '\n    Transform any list-like object in a 1-dimensional numpy array of object\n    dtype.\n\n    Parameters\n    ----------\n    values : any iterable which has a len()\n\n    Raises\n    ------\n    TypeError\n        * If `values` does not have a len()\n\n    Returns\n    -------\n    1-dimensional numpy array of dtype object\n    '
    result = np.empty(len(values), dtype='object')
    result[:] = values
    return result

def construct_1d_ndarray_preserving_na(values, dtype=None, copy=False):
    "\n    Construct a new ndarray, coercing `values` to `dtype`, preserving NA.\n\n    Parameters\n    ----------\n    values : Sequence\n    dtype : numpy.dtype, optional\n    copy : bool, default False\n        Note that copies may still be made with ``copy=False`` if casting\n        is required.\n\n    Returns\n    -------\n    arr : ndarray[dtype]\n\n    Examples\n    --------\n    >>> np.array([1.0, 2.0, None], dtype='str')\n    array(['1.0', '2.0', 'None'], dtype='<U4')\n\n    >>> construct_1d_ndarray_preserving_na([1.0, 2.0, None], dtype=np.dtype('str'))\n    array(['1.0', '2.0', None], dtype=object)\n    "
    if ((dtype is not None) and (dtype.kind == 'U')):
        subarr = lib.ensure_string_array(values, convert_na_value=False, copy=copy)
    else:
        if (dtype is not None):
            _disallow_mismatched_datetimelike(values, dtype)
        subarr = np.array(values, dtype=dtype, copy=copy)
    return subarr

def maybe_cast_to_integer_array(arr, dtype, copy=False):
    '\n    Takes any dtype and returns the casted version, raising for when data is\n    incompatible with integer/unsigned integer dtypes.\n\n    .. versionadded:: 0.24.0\n\n    Parameters\n    ----------\n    arr : array-like\n        The array to cast.\n    dtype : str, np.dtype\n        The integer dtype to cast the array to.\n    copy: bool, default False\n        Whether to make a copy of the array before returning.\n\n    Returns\n    -------\n    ndarray\n        Array of integer or unsigned integer dtype.\n\n    Raises\n    ------\n    OverflowError : the dtype is incompatible with the data\n    ValueError : loss of precision has occurred during casting\n\n    Examples\n    --------\n    If you try to coerce negative values to unsigned integers, it raises:\n\n    >>> pd.Series([-1], dtype="uint64")\n    Traceback (most recent call last):\n        ...\n    OverflowError: Trying to coerce negative values to unsigned integers\n\n    Also, if you try to coerce float values to integers, it raises:\n\n    >>> pd.Series([1, 2, 3.5], dtype="int64")\n    Traceback (most recent call last):\n        ...\n    ValueError: Trying to coerce float values to integers\n    '
    assert is_integer_dtype(dtype)
    try:
        if (not hasattr(arr, 'astype')):
            casted = np.array(arr, dtype=dtype, copy=copy)
        else:
            casted = arr.astype(dtype, copy=copy)
    except OverflowError as err:
        raise OverflowError(f'The elements provided in the data cannot all be casted to the dtype {dtype}') from err
    if np.array_equal(arr, casted):
        return casted
    arr = np.asarray(arr)
    if (is_unsigned_integer_dtype(dtype) and (arr < 0).any()):
        raise OverflowError('Trying to coerce negative values to unsigned integers')
    if (is_float_dtype(arr) or is_object_dtype(arr)):
        raise ValueError('Trying to coerce float values to integers')

def convert_scalar_for_putitemlike(scalar, dtype):
    '\n    Convert datetimelike scalar if we are setting into a datetime64\n    or timedelta64 ndarray.\n\n    Parameters\n    ----------\n    scalar : scalar\n    dtype : np.dtype\n\n    Returns\n    -------\n    scalar\n    '
    if (dtype.kind in ['m', 'M']):
        scalar = maybe_box_datetimelike(scalar, dtype)
        return maybe_unbox_datetimelike(scalar, dtype)
    else:
        validate_numeric_casting(dtype, scalar)
    return scalar

def validate_numeric_casting(dtype, value):
    '\n    Check that we can losslessly insert the given value into an array\n    with the given dtype.\n\n    Parameters\n    ----------\n    dtype : np.dtype\n    value : scalar\n\n    Raises\n    ------\n    ValueError\n    '
    if issubclass(dtype.type, (np.integer, np.bool_)):
        if (is_float(value) and np.isnan(value)):
            raise ValueError('Cannot assign nan to integer series')
    if (issubclass(dtype.type, (np.integer, np.floating, complex)) and (not issubclass(dtype.type, np.bool_))):
        if is_bool(value):
            raise ValueError('Cannot assign bool to float/integer series')
