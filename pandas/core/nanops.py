
import functools
import itertools
import operator
from typing import Any, Optional, Tuple, Union, cast
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import NaT, Timedelta, iNaT, lib
from pandas._typing import ArrayLike, Dtype, DtypeObj, F, Scalar
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import get_dtype, is_any_int_dtype, is_bool_dtype, is_complex, is_datetime64_any_dtype, is_float, is_float_dtype, is_integer, is_integer_dtype, is_numeric_dtype, is_object_dtype, is_scalar, is_timedelta64_dtype, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna
from pandas.core.construction import extract_array
bn = import_optional_dependency('bottleneck', raise_on_missing=False, on_version='warn')
_BOTTLENECK_INSTALLED = (bn is not None)
_USE_BOTTLENECK = False

def set_use_bottleneck(v=True):
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v
set_use_bottleneck(get_option('compute.use_bottleneck'))

class disallow():

    def __init__(self, *dtypes):
        super().__init__()
        self.dtypes = tuple((pandas_dtype(dtype).type for dtype in dtypes))

    def check(self, obj):
        return (hasattr(obj, 'dtype') and issubclass(obj.dtype.type, self.dtypes))

    def __call__(self, f):

        @functools.wraps(f)
        def _f(*args, **kwargs):
            obj_iter = itertools.chain(args, kwargs.values())
            if any((self.check(obj) for obj in obj_iter)):
                f_name = f.__name__.replace('nan', '')
                raise TypeError(f"reduction operation '{f_name}' not allowed for this dtype")
            try:
                with np.errstate(invalid='ignore'):
                    return f(*args, **kwargs)
            except ValueError as e:
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise
        return cast(F, _f)

class bottleneck_switch():

    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self, alt):
        bn_name = (self.name or alt.__name__)
        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):
            bn_func = None

        @functools.wraps(alt)
        def f(values: np.ndarray, *, axis: Optional[int]=None, skipna: bool=True, **kwds):
            if (len(self.kwargs) > 0):
                for (k, v) in self.kwargs.items():
                    if (k not in kwds):
                        kwds[k] = v
            if ((values.size == 0) and (kwds.get('min_count') is None)):
                return _na_for_min_count(values, axis)
            if (_USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name)):
                if (kwds.get('mask', None) is None):
                    kwds.pop('mask', None)
                    result = bn_func(values, axis=axis, **kwds)
                    if _has_infs(result):
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                result = alt(values, axis=axis, skipna=skipna, **kwds)
            return result
        return cast(F, f)

def _bn_ok_dtype(dtype, name):
    if ((not is_object_dtype(dtype)) and (not needs_i8_conversion(dtype))):
        if (name in ['nansum', 'nanprod']):
            return False
        return True
    return False

def _has_infs(result):
    if isinstance(result, np.ndarray):
        if (result.dtype == 'f8'):
            return lib.has_infs_f8(result.ravel('K'))
        elif (result.dtype == 'f4'):
            return lib.has_infs_f4(result.ravel('K'))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        return False

def _get_fill_value(dtype, fill_value=None, fill_value_typ=None):
    ' return the correct fill value for the dtype of the values '
    if (fill_value is not None):
        return fill_value
    if _na_ok_dtype(dtype):
        if (fill_value_typ is None):
            return np.nan
        elif (fill_value_typ == '+inf'):
            return np.inf
        else:
            return (- np.inf)
    elif (fill_value_typ is None):
        return iNaT
    elif (fill_value_typ == '+inf'):
        return np.iinfo(np.int64).max
    else:
        return iNaT

def _maybe_get_mask(values, skipna, mask):
    '\n    Compute a mask if and only if necessary.\n\n    This function will compute a mask iff it is necessary. Otherwise,\n    return the provided mask (potentially None) when a mask does not need to be\n    computed.\n\n    A mask is never necessary if the values array is of boolean or integer\n    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable\n    dtype that is interpretable as either boolean or integer data (eg,\n    timedelta64), a mask must be provided.\n\n    If the skipna parameter is False, a new mask will not be computed.\n\n    The mask is computed using isna() by default. Setting invert=True selects\n    notna() as the masking function.\n\n    Parameters\n    ----------\n    values : ndarray\n        input array to potentially compute mask for\n    skipna : bool\n        boolean for whether NaNs should be skipped\n    mask : Optional[ndarray]\n        nan-mask if known\n\n    Returns\n    -------\n    Optional[np.ndarray]\n    '
    if (mask is None):
        if (is_bool_dtype(values.dtype) or is_integer_dtype(values.dtype)):
            return None
        if (skipna or needs_i8_conversion(values.dtype)):
            mask = isna(values)
    return mask

def _get_values(values, skipna, fill_value=None, fill_value_typ=None, mask=None):
    "\n    Utility to get the values view, mask, dtype, dtype_max, and fill_value.\n\n    If both mask and fill_value/fill_value_typ are not None and skipna is True,\n    the values array will be copied.\n\n    For input arrays of boolean or integer dtypes, copies will only occur if a\n    precomputed mask, a fill_value/fill_value_typ, and skipna=True are\n    provided.\n\n    Parameters\n    ----------\n    values : ndarray\n        input array to potentially compute mask for\n    skipna : bool\n        boolean for whether NaNs should be skipped\n    fill_value : Any\n        value to fill NaNs with\n    fill_value_typ : str\n        Set to '+inf' or '-inf' to handle dtype-specific infinities\n    mask : Optional[np.ndarray]\n        nan-mask if known\n\n    Returns\n    -------\n    values : ndarray\n        Potential copy of input value array\n    mask : Optional[ndarray[bool]]\n        Mask for values, if deemed necessary to compute\n    dtype : np.dtype\n        dtype for values\n    dtype_max : np.dtype\n        platform independent dtype\n    fill_value : Any\n        fill value used\n    "
    assert is_scalar(fill_value)
    values = extract_array(values, extract_numpy=True)
    mask = _maybe_get_mask(values, skipna, mask)
    dtype = values.dtype
    datetimelike = False
    if needs_i8_conversion(values.dtype):
        values = np.asarray(values.view('i8'))
        datetimelike = True
    dtype_ok = _na_ok_dtype(dtype)
    fill_value = _get_fill_value(dtype, fill_value=fill_value, fill_value_typ=fill_value_typ)
    if (skipna and (mask is not None) and (fill_value is not None)):
        if mask.any():
            if (dtype_ok or datetimelike):
                values = values.copy()
                np.putmask(values, mask, fill_value)
            else:
                values = np.where((~ mask), values, fill_value)
    dtype_max = dtype
    if (is_integer_dtype(dtype) or is_bool_dtype(dtype)):
        dtype_max = np.dtype(np.int64)
    elif is_float_dtype(dtype):
        dtype_max = np.dtype(np.float64)
    return (values, mask, dtype, dtype_max, fill_value)

def _na_ok_dtype(dtype):
    if needs_i8_conversion(dtype):
        return False
    return (not issubclass(dtype.type, np.integer))

def _wrap_results(result, dtype, fill_value=None):
    ' wrap our results if needed '
    if (result is NaT):
        pass
    elif is_datetime64_any_dtype(dtype):
        if (fill_value is None):
            fill_value = iNaT
        if (not isinstance(result, np.ndarray)):
            assert (not isna(fill_value)), 'Expected non-null fill_value'
            if (result == fill_value):
                result = np.nan
            if isna(result):
                result = np.datetime64('NaT', 'ns')
            else:
                result = np.int64(result).view('datetime64[ns]')
        else:
            result = result.astype(dtype)
    elif is_timedelta64_dtype(dtype):
        if (not isinstance(result, np.ndarray)):
            if (result == fill_value):
                result = np.nan
            if (np.fabs(result) > np.iinfo(np.int64).max):
                raise ValueError('overflow in timedelta operation')
            result = Timedelta(result, unit='ns')
        else:
            result = result.astype('m8[ns]').view(dtype)
    return result

def _datetimelike_compat(func):
    '\n    If we have datetime64 or timedelta64 values, ensure we have a correct\n    mask before calling the wrapped function, then cast back afterwards.\n    '

    @functools.wraps(func)
    def new_func(values: np.ndarray, *, axis: Optional[int]=None, skipna: bool=True, mask: Optional[np.ndarray]=None, **kwargs):
        orig_values = values
        datetimelike = (values.dtype.kind in ['m', 'M'])
        if (datetimelike and (mask is None)):
            mask = isna(values)
        result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        if datetimelike:
            result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)
            if (not skipna):
                result = _mask_datetimelike_result(result, axis, mask, orig_values)
        return result
    return cast(F, new_func)

def _na_for_min_count(values, axis):
    '\n    Return the missing value for `values`.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int or None\n        axis for the reduction, required if values.ndim > 1.\n\n    Returns\n    -------\n    result : scalar or ndarray\n        For 1-D values, returns a scalar of the correct missing type.\n        For 2-D values, returns a 1-D array where each element is missing.\n    '
    if is_numeric_dtype(values):
        values = values.astype('float64')
    fill_value = na_value_for_dtype(values.dtype)
    if (fill_value is NaT):
        fill_value = values.dtype.type('NaT', 'ns')
    if (values.ndim == 1):
        return fill_value
    elif (axis is None):
        return fill_value
    else:
        result_shape = (values.shape[:axis] + values.shape[(axis + 1):])
        result = np.full(result_shape, fill_value, dtype=values.dtype)
        return result

def nanany(values, *, axis=None, skipna=True, mask=None):
    '\n    Check if any elements along an axis evaluate to True.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis : int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : bool\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, 2])\n    >>> nanops.nanany(s)\n    True\n\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([np.nan])\n    >>> nanops.nanany(s)\n    False\n    '
    (values, _, _, _, _) = _get_values(values, skipna, fill_value=False, mask=mask)
    return values.any(axis)

def nanall(values, *, axis=None, skipna=True, mask=None):
    '\n    Check if all elements along an axis evaluate to True.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : bool\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, 2, np.nan])\n    >>> nanops.nanall(s)\n    True\n\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, 0])\n    >>> nanops.nanall(s)\n    False\n    '
    (values, _, _, _, _) = _get_values(values, skipna, fill_value=True, mask=mask)
    return values.all(axis)

@disallow('M8')
@_datetimelike_compat
def nansum(values, *, axis=None, skipna=True, min_count=0, mask=None):
    '\n    Sum the elements along an axis ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray[dtype]\n    axis: int, optional\n    skipna : bool, default True\n    min_count: int, default 0\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : dtype\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, 2, np.nan])\n    >>> nanops.nansum(s)\n    3.0\n    '
    (values, mask, dtype, dtype_max, _) = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = dtype_max
    if is_float_dtype(dtype):
        dtype_sum = dtype
    elif is_timedelta64_dtype(dtype):
        dtype_sum = np.float64
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)
    return the_sum

def _mask_datetimelike_result(result, axis, mask, orig_values):
    if isinstance(result, np.ndarray):
        result = result.astype('i8').view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        result[axis_mask] = iNaT
    elif mask.any():
        result = NaT
    return result

@disallow(PeriodDtype)
@bottleneck_switch()
@_datetimelike_compat
def nanmean(values, *, axis=None, skipna=True, mask=None):
    '\n    Compute the mean of the element along an axis ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    float\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, 2, np.nan])\n    >>> nanops.nanmean(s)\n    1.5\n    '
    (values, mask, dtype, dtype_max, _) = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = dtype_max
    dtype_count = np.float64
    if (dtype.kind in ['m', 'M']):
        dtype_sum = np.float64
    elif is_integer_dtype(dtype):
        dtype_sum = np.float64
    elif is_float_dtype(dtype):
        dtype_sum = dtype
        dtype_count = dtype
    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    the_sum = _ensure_numeric(values.sum(axis, dtype=dtype_sum))
    if ((axis is not None) and getattr(the_sum, 'ndim', False)):
        count = cast(np.ndarray, count)
        with np.errstate(all='ignore'):
            the_mean = (the_sum / count)
        ct_mask = (count == 0)
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        the_mean = ((the_sum / count) if (count > 0) else np.nan)
    return the_mean

@bottleneck_switch()
def nanmedian(values, *, axis=None, skipna=True, mask=None):
    '\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, np.nan, 2, 2])\n    >>> nanops.nanmedian(s)\n    2.0\n    '

    def get_median(x):
        mask = notna(x)
        if ((not skipna) and (not mask.all())):
            return np.nan
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            res = np.nanmedian(x[mask])
        return res
    (values, mask, dtype, _, _) = _get_values(values, skipna, mask=mask)
    if (not is_float_dtype(values.dtype)):
        try:
            values = values.astype('f8')
        except ValueError as err:
            raise TypeError(str(err)) from err
        if (mask is not None):
            values[mask] = np.nan
    if (axis is None):
        values = values.ravel('K')
    notempty = values.size
    if (values.ndim > 1):
        if notempty:
            if (not skipna):
                res = np.apply_along_axis(get_median, axis, values)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'All-NaN slice encountered')
                    res = np.nanmedian(values, axis)
        else:
            res = get_empty_reduction_result(values.shape, axis, np.float_, np.nan)
    else:
        res = (get_median(values) if notempty else np.nan)
    return _wrap_results(res, dtype)

def get_empty_reduction_result(shape, axis, dtype, fill_value):
    '\n    The result from a reduction on an empty ndarray.\n\n    Parameters\n    ----------\n    shape : Tuple[int]\n    axis : int\n    dtype : np.dtype\n    fill_value : Any\n\n    Returns\n    -------\n    np.ndarray\n    '
    shp = np.array(shape)
    dims = np.arange(len(shape))
    ret = np.empty(shp[(dims != axis)], dtype=dtype)
    ret.fill(fill_value)
    return ret

def _get_counts_nanvar(value_counts, mask, axis, ddof, dtype=float):
    '\n    Get the count of non-null values along an axis, accounting\n    for degrees of freedom.\n\n    Parameters\n    ----------\n    values_shape : Tuple[int]\n        shape tuple from values ndarray, used if mask is None\n    mask : Optional[ndarray[bool]]\n        locations in values that should be considered missing\n    axis : Optional[int]\n        axis to count along\n    ddof : int\n        degrees of freedom\n    dtype : type, optional\n        type to use for count\n\n    Returns\n    -------\n    count : scalar or array\n    d : scalar or array\n    '
    dtype = get_dtype(dtype)
    count = _get_counts(value_counts, mask, axis, dtype=dtype)
    d = (count - dtype.type(ddof))
    if is_scalar(count):
        if (count <= ddof):
            count = np.nan
            d = np.nan
    else:
        mask2: np.ndarray = (count <= ddof)
        if mask2.any():
            np.putmask(d, mask2, np.nan)
            np.putmask(count, mask2, np.nan)
    return (count, d)

@bottleneck_switch(ddof=1)
def nanstd(values, *, axis=None, skipna=True, ddof=1, mask=None):
    '\n    Compute the standard deviation along given axis while ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    ddof : int, default 1\n        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,\n        where N represents the number of elements.\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, np.nan, 2, 3])\n    >>> nanops.nanstd(s)\n    1.0\n    '
    if (values.dtype == 'M8[ns]'):
        values = values.view('m8[ns]')
    orig_dtype = values.dtype
    (values, mask, _, _, _) = _get_values(values, skipna, mask=mask)
    result = np.sqrt(nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask))
    return _wrap_results(result, orig_dtype)

@disallow('M8', 'm8')
@bottleneck_switch(ddof=1)
def nanvar(values, *, axis=None, skipna=True, ddof=1, mask=None):
    '\n    Compute the variance along given axis while ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    ddof : int, default 1\n        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,\n        where N represents the number of elements.\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, np.nan, 2, 3])\n    >>> nanops.nanvar(s)\n    1.0\n    '
    values = extract_array(values, extract_numpy=True)
    dtype = values.dtype
    mask = _maybe_get_mask(values, skipna, mask)
    if is_any_int_dtype(dtype):
        values = values.astype('f8')
        if (mask is not None):
            values[mask] = np.nan
    if is_float_dtype(values.dtype):
        (count, d) = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        (count, d) = _get_counts_nanvar(values.shape, mask, axis, ddof)
    if (skipna and (mask is not None)):
        values = values.copy()
        np.putmask(values, mask, 0)
    avg = (_ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count)
    if (axis is not None):
        avg = np.expand_dims(avg, axis)
    sqr = _ensure_numeric(((avg - values) ** 2))
    if (mask is not None):
        np.putmask(sqr, mask, 0)
    result = (sqr.sum(axis=axis, dtype=np.float64) / d)
    if is_float_dtype(dtype):
        result = result.astype(dtype)
    return result

@disallow('M8', 'm8')
def nansem(values, *, axis=None, skipna=True, ddof=1, mask=None):
    '\n    Compute the standard error in the mean along given axis while ignoring NaNs\n\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    ddof : int, default 1\n        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,\n        where N represents the number of elements.\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float64\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, np.nan, 2, 3])\n    >>> nanops.nansem(s)\n     0.5773502691896258\n    '
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)
    mask = _maybe_get_mask(values, skipna, mask)
    if (not is_float_dtype(values.dtype)):
        values = values.astype('f8')
    (count, _) = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof)
    return (np.sqrt(var) / np.sqrt(count))

def _nanminmax(meth, fill_value_typ):

    @bottleneck_switch(name=('nan' + meth))
    @_datetimelike_compat
    def reduction(values: np.ndarray, *, axis: Optional[int]=None, skipna: bool=True, mask: Optional[np.ndarray]=None) -> Dtype:
        (values, mask, dtype, dtype_max, fill_value) = _get_values(values, skipna, fill_value_typ=fill_value_typ, mask=mask)
        if (((axis is not None) and (values.shape[axis] == 0)) or (values.size == 0)):
            try:
                result = getattr(values, meth)(axis, dtype=dtype_max)
                result.fill(np.nan)
            except (AttributeError, TypeError, ValueError):
                result = np.nan
        else:
            result = getattr(values, meth)(axis)
        result = _maybe_null_out(result, axis, mask, values.shape)
        return result
    return reduction
nanmin = _nanminmax('min', fill_value_typ='+inf')
nanmax = _nanminmax('max', fill_value_typ='-inf')

@disallow('O')
def nanargmax(values, *, axis=None, skipna=True, mask=None):
    '\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : int or ndarray[int]\n        The index/indices  of max value in specified axis or -1 in the NA case\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> arr = np.array([1, 2, 3, np.nan, 4])\n    >>> nanops.nanargmax(arr)\n    4\n\n    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)\n    >>> arr[2:, 2] = np.nan\n    >>> arr\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4.,  5.],\n           [ 6.,  7., nan],\n           [ 9., 10., nan]])\n    >>> nanops.nanargmax(arr, axis=1)\n    array([2, 2, 1, 1], dtype=int64)\n    '
    (values, mask, _, _, _) = _get_values(values, True, fill_value_typ='-inf', mask=mask)
    result = values.argmax(axis)
    result = _maybe_arg_null_out(result, axis, mask, skipna)
    return result

@disallow('O')
def nanargmin(values, *, axis=None, skipna=True, mask=None):
    '\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : int or ndarray[int]\n        The index/indices of min value in specified axis or -1 in the NA case\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> arr = np.array([1, 2, 3, np.nan, 4])\n    >>> nanops.nanargmin(arr)\n    0\n\n    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)\n    >>> arr[2:, 0] = np.nan\n    >>> arr\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4.,  5.],\n           [nan,  7.,  8.],\n           [nan, 10., 11.]])\n    >>> nanops.nanargmin(arr, axis=1)\n    array([0, 0, 1, 1], dtype=int64)\n    '
    (values, mask, _, _, _) = _get_values(values, True, fill_value_typ='+inf', mask=mask)
    result = values.argmin(axis)
    result = _maybe_arg_null_out(result, axis, mask, skipna)
    return result

@disallow('M8', 'm8')
def nanskew(values, *, axis=None, skipna=True, mask=None):
    '\n    Compute the sample skewness.\n\n    The statistic computed here is the adjusted Fisher-Pearson standardized\n    moment coefficient G1. The algorithm computes this coefficient directly\n    from the second and third central moment.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float64\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, np.nan, 1, 2])\n    >>> nanops.nanskew(s)\n    1.7320508075688787\n    '
    values = extract_array(values, extract_numpy=True)
    mask = _maybe_get_mask(values, skipna, mask)
    if (not is_float_dtype(values.dtype)):
        values = values.astype('f8')
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)
    if (skipna and (mask is not None)):
        values = values.copy()
        np.putmask(values, mask, 0)
    mean = (values.sum(axis, dtype=np.float64) / count)
    if (axis is not None):
        mean = np.expand_dims(mean, axis)
    adjusted = (values - mean)
    if (skipna and (mask is not None)):
        np.putmask(adjusted, mask, 0)
    adjusted2 = (adjusted ** 2)
    adjusted3 = (adjusted2 * adjusted)
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m3 = adjusted3.sum(axis, dtype=np.float64)
    m2 = _zero_out_fperr(m2)
    m3 = _zero_out_fperr(m3)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = (((count * ((count - 1) ** 0.5)) / (count - 2)) * (m3 / (m2 ** 1.5)))
    dtype = values.dtype
    if is_float_dtype(dtype):
        result = result.astype(dtype)
    if isinstance(result, np.ndarray):
        result = np.where((m2 == 0), 0, result)
        result[(count < 3)] = np.nan
        return result
    else:
        result = (0 if (m2 == 0) else result)
        if (count < 3):
            return np.nan
        return result

@disallow('M8', 'm8')
def nankurt(values, *, axis=None, skipna=True, mask=None):
    '\n    Compute the sample excess kurtosis\n\n    The statistic computed here is the adjusted Fisher-Pearson standardized\n    moment coefficient G2, computed directly from the second and fourth\n    central moment.\n\n    Parameters\n    ----------\n    values : ndarray\n    axis: int, optional\n    skipna : bool, default True\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    result : float64\n        Unless input is a float array, in which case use the same\n        precision as the input array.\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, np.nan, 1, 3, 2])\n    >>> nanops.nankurt(s)\n    -1.2892561983471076\n    '
    values = extract_array(values, extract_numpy=True)
    mask = _maybe_get_mask(values, skipna, mask)
    if (not is_float_dtype(values.dtype)):
        values = values.astype('f8')
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)
    if (skipna and (mask is not None)):
        values = values.copy()
        np.putmask(values, mask, 0)
    mean = (values.sum(axis, dtype=np.float64) / count)
    if (axis is not None):
        mean = np.expand_dims(mean, axis)
    adjusted = (values - mean)
    if (skipna and (mask is not None)):
        np.putmask(adjusted, mask, 0)
    adjusted2 = (adjusted ** 2)
    adjusted4 = (adjusted2 ** 2)
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        adj = ((3 * ((count - 1) ** 2)) / ((count - 2) * (count - 3)))
        numerator = (((count * (count + 1)) * (count - 1)) * m4)
        denominator = (((count - 2) * (count - 3)) * (m2 ** 2))
    numerator = _zero_out_fperr(numerator)
    denominator = _zero_out_fperr(denominator)
    if (not isinstance(denominator, np.ndarray)):
        if (count < 4):
            return np.nan
        if (denominator == 0):
            return 0
    with np.errstate(invalid='ignore', divide='ignore'):
        result = ((numerator / denominator) - adj)
    dtype = values.dtype
    if is_float_dtype(dtype):
        result = result.astype(dtype)
    if isinstance(result, np.ndarray):
        result = np.where((denominator == 0), 0, result)
        result[(count < 4)] = np.nan
    return result

@disallow('M8', 'm8')
def nanprod(values, *, axis=None, skipna=True, min_count=0, mask=None):
    '\n    Parameters\n    ----------\n    values : ndarray[dtype]\n    axis: int, optional\n    skipna : bool, default True\n    min_count: int, default 0\n    mask : ndarray[bool], optional\n        nan-mask if known\n\n    Returns\n    -------\n    Dtype\n        The product of all elements on a given axis. ( NaNs are treated as 1)\n\n    Examples\n    --------\n    >>> import pandas.core.nanops as nanops\n    >>> s = pd.Series([1, 2, 3, np.nan])\n    >>> nanops.nanprod(s)\n    6.0\n    '
    mask = _maybe_get_mask(values, skipna, mask)
    if (skipna and (mask is not None)):
        values = values.copy()
        values[mask] = 1
    result = values.prod(axis)
    return _maybe_null_out(result, axis, mask, values.shape, min_count=min_count)

def _maybe_arg_null_out(result, axis, mask, skipna):
    if (mask is None):
        return result
    if ((axis is None) or (not getattr(result, 'ndim', False))):
        if skipna:
            if mask.all():
                result = (- 1)
        elif mask.any():
            result = (- 1)
    else:
        if skipna:
            na_mask = mask.all(axis)
        else:
            na_mask = mask.any(axis)
        if na_mask.any():
            result[na_mask] = (- 1)
    return result

def _get_counts(values_shape, mask, axis, dtype=float):
    '\n    Get the count of non-null values along an axis\n\n    Parameters\n    ----------\n    values_shape : tuple of int\n        shape tuple from values ndarray, used if mask is None\n    mask : Optional[ndarray[bool]]\n        locations in values that should be considered missing\n    axis : Optional[int]\n        axis to count along\n    dtype : type, optional\n        type to use for count\n\n    Returns\n    -------\n    count : scalar or array\n    '
    dtype = get_dtype(dtype)
    if (axis is None):
        if (mask is not None):
            n = (mask.size - mask.sum())
        else:
            n = np.prod(values_shape)
        return dtype.type(n)
    if (mask is not None):
        count = (mask.shape[axis] - mask.sum(axis))
    else:
        count = values_shape[axis]
    if is_scalar(count):
        return dtype.type(count)
    try:
        return count.astype(dtype)
    except AttributeError:
        return np.array(count, dtype=dtype)

def _maybe_null_out(result, axis, mask, shape, min_count=1):
    '\n    Returns\n    -------\n    Dtype\n        The product of all elements on a given axis. ( NaNs are treated as 1)\n    '
    if ((mask is not None) and (axis is not None) and getattr(result, 'ndim', False)):
        null_mask = (((mask.shape[axis] - mask.sum(axis)) - min_count) < 0)
        if np.any(null_mask):
            if is_numeric_dtype(result):
                if np.iscomplexobj(result):
                    result = result.astype('c16')
                else:
                    result = result.astype('f8')
                result[null_mask] = np.nan
            else:
                result[null_mask] = None
    elif (result is not NaT):
        if check_below_min_count(shape, mask, min_count):
            result = np.nan
    return result

def check_below_min_count(shape, mask, min_count):
    '\n    Check for the `min_count` keyword. Returns True if below `min_count` (when\n    missing value should be returned from the reduction).\n\n    Parameters\n    ----------\n    shape : tuple\n        The shape of the values (`values.shape`).\n    mask : ndarray or None\n        Boolean numpy array (typically of same shape as `shape`) or None.\n    min_count : int\n        Keyword passed through from sum/prod call.\n\n    Returns\n    -------\n    bool\n    '
    if (min_count > 0):
        if (mask is None):
            non_nulls = np.prod(shape)
        else:
            non_nulls = (mask.size - mask.sum())
        if (non_nulls < min_count):
            return True
    return False

def _zero_out_fperr(arg):
    if isinstance(arg, np.ndarray):
        with np.errstate(invalid='ignore'):
            return np.where((np.abs(arg) < 1e-14), 0, arg)
    else:
        return (arg.dtype.type(0) if (np.abs(arg) < 1e-14) else arg)

@disallow('M8', 'm8')
def nancorr(a, b, *, method='pearson', min_periods=None):
    '\n    a, b: ndarrays\n    '
    if (len(a) != len(b)):
        raise AssertionError('Operands to nancorr must have same size')
    if (min_periods is None):
        min_periods = 1
    valid = (notna(a) & notna(b))
    if (not valid.all()):
        a = a[valid]
        b = b[valid]
    if (len(a) < min_periods):
        return np.nan
    f = get_corr_func(method)
    return f(a, b)

def get_corr_func(method):
    if (method == 'kendall'):
        from scipy.stats import kendalltau

        def func(a, b):
            return kendalltau(a, b)[0]
        return func
    elif (method == 'spearman'):
        from scipy.stats import spearmanr

        def func(a, b):
            return spearmanr(a, b)[0]
        return func
    elif (method == 'pearson'):

        def func(a, b):
            return np.corrcoef(a, b)[(0, 1)]
        return func
    elif callable(method):
        return method
    raise ValueError(f"Unknown method '{method}', expected one of 'kendall', 'spearman', 'pearson', or callable")

@disallow('M8', 'm8')
def nancov(a, b, *, min_periods=None, ddof=1):
    if (len(a) != len(b)):
        raise AssertionError('Operands to nancov must have same size')
    if (min_periods is None):
        min_periods = 1
    valid = (notna(a) & notna(b))
    if (not valid.all()):
        a = a[valid]
        b = b[valid]
    if (len(a) < min_periods):
        return np.nan
    return np.cov(a, b, ddof=ddof)[(0, 1)]

def _ensure_numeric(x):
    if isinstance(x, np.ndarray):
        if (is_integer_dtype(x) or is_bool_dtype(x)):
            x = x.astype(np.float64)
        elif is_object_dtype(x):
            try:
                x = x.astype(np.complex128)
            except (TypeError, ValueError):
                try:
                    x = x.astype(np.float64)
                except ValueError as err:
                    raise TypeError(f'Could not convert {x} to numeric') from err
            else:
                if (not np.any(np.imag(x))):
                    x = x.real
    elif (not (is_float(x) or is_integer(x) or is_complex(x))):
        try:
            x = float(x)
        except ValueError:
            try:
                x = complex(x)
            except ValueError as err:
                raise TypeError(f'Could not convert {x} to numeric') from err
    return x

def make_nancomp(op):

    def f(x, y):
        xmask = isna(x)
        ymask = isna(y)
        mask = (xmask | ymask)
        with np.errstate(all='ignore'):
            result = op(x, y)
        if mask.any():
            if is_bool_dtype(result):
                result = result.astype('O')
            np.putmask(result, mask, np.nan)
        return result
    return f
nangt = make_nancomp(operator.gt)
nange = make_nancomp(operator.ge)
nanlt = make_nancomp(operator.lt)
nanle = make_nancomp(operator.le)
naneq = make_nancomp(operator.eq)
nanne = make_nancomp(operator.ne)

def _nanpercentile_1d(values, mask, q, na_value, interpolation):
    '\n    Wrapper for np.percentile that skips missing values, specialized to\n    1-dimensional case.\n\n    Parameters\n    ----------\n    values : array over which to find quantiles\n    mask : ndarray[bool]\n        locations in values that should be considered missing\n    q : scalar or array of quantile indices to find\n    na_value : scalar\n        value to return for empty or all-null values\n    interpolation : str\n\n    Returns\n    -------\n    quantiles : scalar or array\n    '
    values = values[(~ mask)]
    if (len(values) == 0):
        if lib.is_scalar(q):
            return na_value
        else:
            return np.array(([na_value] * len(q)), dtype=values.dtype)
    return np.percentile(values, q, interpolation=interpolation)

def nanpercentile(values, q, *, axis, na_value, mask, ndim, interpolation):
    '\n    Wrapper for np.percentile that skips missing values.\n\n    Parameters\n    ----------\n    values : array over which to find quantiles\n    q : scalar or array of quantile indices to find\n    axis : {0, 1}\n    na_value : scalar\n        value to return for empty or all-null values\n    mask : ndarray[bool]\n        locations in values that should be considered missing\n    ndim : {1, 2}\n    interpolation : str\n\n    Returns\n    -------\n    quantiles : scalar or array\n    '
    if (values.dtype.kind in ['m', 'M']):
        result = nanpercentile(values.view('i8'), q=q, axis=axis, na_value=na_value.view('i8'), mask=mask, ndim=ndim, interpolation=interpolation)
        return result.astype(values.dtype)
    if ((not lib.is_scalar(mask)) and mask.any()):
        if (ndim == 1):
            return _nanpercentile_1d(values, mask, q, na_value, interpolation=interpolation)
        else:
            if (mask.ndim < values.ndim):
                mask = mask.reshape(values.shape)
            if (axis == 0):
                values = values.T
                mask = mask.T
            result = [_nanpercentile_1d(val, m, q, na_value, interpolation=interpolation) for (val, m) in zip(list(values), list(mask))]
            result = np.array(result, dtype=values.dtype, copy=False).T
            return result
    else:
        return np.percentile(values, q, axis=axis, interpolation=interpolation)

def na_accum_func(values, accum_func, *, skipna):
    '\n    Cumulative function with skipna support.\n\n    Parameters\n    ----------\n    values : np.ndarray or ExtensionArray\n    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}\n    skipna : bool\n\n    Returns\n    -------\n    np.ndarray or ExtensionArray\n    '
    (mask_a, mask_b) = {np.cumprod: (1.0, np.nan), np.maximum.accumulate: ((- np.inf), np.nan), np.cumsum: (0.0, np.nan), np.minimum.accumulate: (np.inf, np.nan)}[accum_func]
    if (values.dtype.kind in ['m', 'M']):
        orig_dtype = values.dtype
        mask = isna(values)
        if (accum_func == np.minimum.accumulate):
            y = values.view('i8')
            y[mask] = np.iinfo(np.int64).max
            changed = True
        else:
            y = values
            changed = False
        result = accum_func(y.view('i8'), axis=0)
        if skipna:
            result[mask] = iNaT
        elif (accum_func == np.minimum.accumulate):
            nz = (~ np.asarray(mask)).nonzero()[0]
            if len(nz):
                result[:nz[0]] = iNaT
        if changed:
            y[mask] = iNaT
        if isinstance(values, np.ndarray):
            result = result.view(orig_dtype)
        else:
            result = type(values)._simple_new(result, dtype=orig_dtype)
    elif (skipna and (not issubclass(values.dtype.type, (np.integer, np.bool_)))):
        vals = values.copy()
        mask = isna(vals)
        vals[mask] = mask_a
        result = accum_func(vals, axis=0)
        result[mask] = mask_b
    else:
        result = accum_func(values, axis=0)
    return result
