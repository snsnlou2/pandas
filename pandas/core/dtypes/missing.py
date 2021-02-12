
'\nmissing types & inference\n'
from functools import partial
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import NaT, Period, iNaT
from pandas._typing import ArrayLike, DtypeObj
from pandas.core.dtypes.common import DT64NS_DTYPE, TD64NS_DTYPE, ensure_object, is_bool_dtype, is_categorical_dtype, is_complex_dtype, is_datetimelike_v_numeric, is_dtype_equal, is_extension_array_dtype, is_float_dtype, is_integer_dtype, is_object_dtype, is_scalar, is_string_dtype, is_string_like_dtype, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCExtensionArray, ABCIndex, ABCMultiIndex, ABCSeries
from pandas.core.dtypes.inference import is_list_like
isposinf_scalar = libmissing.isposinf_scalar
isneginf_scalar = libmissing.isneginf_scalar
nan_checker = np.isnan
INF_AS_NA = False

def isna(obj):
    '\n    Detect missing values for an array-like object.\n\n    This function takes a scalar or array-like object and indicates\n    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``\n    in object arrays, ``NaT`` in datetimelike).\n\n    Parameters\n    ----------\n    obj : scalar or array-like\n        Object to check for null or missing values.\n\n    Returns\n    -------\n    bool or array-like of bool\n        For scalar input, returns a scalar boolean.\n        For array input, returns an array of boolean indicating whether each\n        corresponding element is missing.\n\n    See Also\n    --------\n    notna : Boolean inverse of pandas.isna.\n    Series.isna : Detect missing values in a Series.\n    DataFrame.isna : Detect missing values in a DataFrame.\n    Index.isna : Detect missing values in an Index.\n\n    Examples\n    --------\n    Scalar arguments (including strings) result in a scalar boolean.\n\n    >>> pd.isna(\'dog\')\n    False\n\n    >>> pd.isna(pd.NA)\n    True\n\n    >>> pd.isna(np.nan)\n    True\n\n    ndarrays result in an ndarray of booleans.\n\n    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])\n    >>> array\n    array([[ 1., nan,  3.],\n           [ 4.,  5., nan]])\n    >>> pd.isna(array)\n    array([[False,  True, False],\n           [False, False,  True]])\n\n    For indexes, an ndarray of booleans is returned.\n\n    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,\n    ...                           "2017-07-08"])\n    >>> index\n    DatetimeIndex([\'2017-07-05\', \'2017-07-06\', \'NaT\', \'2017-07-08\'],\n                  dtype=\'datetime64[ns]\', freq=None)\n    >>> pd.isna(index)\n    array([False, False,  True, False])\n\n    For Series and DataFrame, the same type is returned, containing booleans.\n\n    >>> df = pd.DataFrame([[\'ant\', \'bee\', \'cat\'], [\'dog\', None, \'fly\']])\n    >>> df\n         0     1    2\n    0  ant   bee  cat\n    1  dog  None  fly\n    >>> pd.isna(df)\n           0      1      2\n    0  False  False  False\n    1  False   True  False\n\n    >>> pd.isna(df[1])\n    0    False\n    1     True\n    Name: 1, dtype: bool\n    '
    return _isna(obj)
isnull = isna

def _isna(obj, inf_as_na=False):
    '\n    Detect missing values, treating None, NaN or NA as null. Infinite\n    values will also be treated as null if inf_as_na is True.\n\n    Parameters\n    ----------\n    obj: ndarray or object value\n        Input array or scalar value.\n    inf_as_na: bool\n        Whether to treat infinity as null.\n\n    Returns\n    -------\n    boolean ndarray or boolean\n    '
    if is_scalar(obj):
        if inf_as_na:
            return libmissing.checknull_old(obj)
        else:
            return libmissing.checknull(obj)
    elif isinstance(obj, ABCMultiIndex):
        raise NotImplementedError('isna is not defined for MultiIndex')
    elif isinstance(obj, type):
        return False
    elif isinstance(obj, (ABCSeries, np.ndarray, ABCIndex, ABCExtensionArray)):
        return _isna_ndarraylike(obj, inf_as_na=inf_as_na)
    elif isinstance(obj, ABCDataFrame):
        return obj.isna()
    elif isinstance(obj, list):
        return _isna_ndarraylike(np.asarray(obj, dtype=object), inf_as_na=inf_as_na)
    elif hasattr(obj, '__array__'):
        return _isna_ndarraylike(np.asarray(obj), inf_as_na=inf_as_na)
    else:
        return False

def _use_inf_as_na(key):
    '\n    Option change callback for na/inf behaviour.\n\n    Choose which replacement for numpy.isnan / -numpy.isfinite is used.\n\n    Parameters\n    ----------\n    flag: bool\n        True means treat None, NaN, INF, -INF as null (old way),\n        False means None and NaN are null, but INF, -INF are not null\n        (new way).\n\n    Notes\n    -----\n    This approach to setting global module values is discussed and\n    approved here:\n\n    * https://stackoverflow.com/questions/4859217/\n      programmatically-creating-variables-in-python/4859312#4859312\n    '
    inf_as_na = get_option(key)
    globals()['_isna'] = partial(_isna, inf_as_na=inf_as_na)
    if inf_as_na:
        globals()['nan_checker'] = (lambda x: (~ np.isfinite(x)))
        globals()['INF_AS_NA'] = True
    else:
        globals()['nan_checker'] = np.isnan
        globals()['INF_AS_NA'] = False

def _isna_ndarraylike(obj, inf_as_na=False):
    '\n    Return an array indicating which values of the input array are NaN / NA.\n\n    Parameters\n    ----------\n    obj: array-like\n        The input array whose elements are to be checked.\n    inf_as_na: bool\n        Whether or not to treat infinite values as NA.\n\n    Returns\n    -------\n    array-like\n        Array of boolean values denoting the NA status of each element.\n    '
    values = getattr(obj, '_values', obj)
    dtype = values.dtype
    if is_extension_array_dtype(dtype):
        if (inf_as_na and is_categorical_dtype(dtype)):
            result = libmissing.isnaobj_old(values.to_numpy())
        else:
            result = values.isna()
    elif is_string_dtype(dtype):
        result = _isna_string_dtype(values, dtype, inf_as_na=inf_as_na)
    elif needs_i8_conversion(dtype):
        result = (values.view('i8') == iNaT)
    elif inf_as_na:
        result = (~ np.isfinite(values))
    else:
        result = np.isnan(values)
    if isinstance(obj, ABCSeries):
        result = obj._constructor(result, index=obj.index, name=obj.name, copy=False)
    return result

def _isna_string_dtype(values, dtype, inf_as_na):
    shape = values.shape
    if is_string_like_dtype(dtype):
        result = np.zeros(values.shape, dtype=bool)
    else:
        result = np.empty(shape, dtype=bool)
        if inf_as_na:
            vec = libmissing.isnaobj_old(values.ravel())
        else:
            vec = libmissing.isnaobj(values.ravel())
        result[...] = vec.reshape(shape)
    return result

def notna(obj):
    '\n    Detect non-missing values for an array-like object.\n\n    This function takes a scalar or array-like object and indicates\n    whether values are valid (not missing, which is ``NaN`` in numeric\n    arrays, ``None`` or ``NaN`` in object arrays, ``NaT`` in datetimelike).\n\n    Parameters\n    ----------\n    obj : array-like or object value\n        Object to check for *not* null or *non*-missing values.\n\n    Returns\n    -------\n    bool or array-like of bool\n        For scalar input, returns a scalar boolean.\n        For array input, returns an array of boolean indicating whether each\n        corresponding element is valid.\n\n    See Also\n    --------\n    isna : Boolean inverse of pandas.notna.\n    Series.notna : Detect valid values in a Series.\n    DataFrame.notna : Detect valid values in a DataFrame.\n    Index.notna : Detect valid values in an Index.\n\n    Examples\n    --------\n    Scalar arguments (including strings) result in a scalar boolean.\n\n    >>> pd.notna(\'dog\')\n    True\n\n    >>> pd.notna(pd.NA)\n    False\n\n    >>> pd.notna(np.nan)\n    False\n\n    ndarrays result in an ndarray of booleans.\n\n    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])\n    >>> array\n    array([[ 1., nan,  3.],\n           [ 4.,  5., nan]])\n    >>> pd.notna(array)\n    array([[ True, False,  True],\n           [ True,  True, False]])\n\n    For indexes, an ndarray of booleans is returned.\n\n    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,\n    ...                          "2017-07-08"])\n    >>> index\n    DatetimeIndex([\'2017-07-05\', \'2017-07-06\', \'NaT\', \'2017-07-08\'],\n                  dtype=\'datetime64[ns]\', freq=None)\n    >>> pd.notna(index)\n    array([ True,  True, False,  True])\n\n    For Series and DataFrame, the same type is returned, containing booleans.\n\n    >>> df = pd.DataFrame([[\'ant\', \'bee\', \'cat\'], [\'dog\', None, \'fly\']])\n    >>> df\n         0     1    2\n    0  ant   bee  cat\n    1  dog  None  fly\n    >>> pd.notna(df)\n          0      1     2\n    0  True   True  True\n    1  True  False  True\n\n    >>> pd.notna(df[1])\n    0     True\n    1    False\n    Name: 1, dtype: bool\n    '
    res = isna(obj)
    if is_scalar(res):
        return (not res)
    return (~ res)
notnull = notna

def isna_compat(arr, fill_value=np.nan):
    '\n    Parameters\n    ----------\n    arr: a numpy array\n    fill_value: fill value, default to np.nan\n\n    Returns\n    -------\n    True if we can fill using this fill_value\n    '
    dtype = arr.dtype
    if isna(fill_value):
        return (not (is_bool_dtype(dtype) or is_integer_dtype(dtype)))
    return True

def array_equivalent(left, right, strict_nan=False, dtype_equal=False):
    '\n    True if two arrays, left and right, have equal non-NaN elements, and NaNs\n    in corresponding locations.  False otherwise. It is assumed that left and\n    right are NumPy arrays of the same dtype. The behavior of this function\n    (particularly with respect to NaNs) is not defined if the dtypes are\n    different.\n\n    Parameters\n    ----------\n    left, right : ndarrays\n    strict_nan : bool, default False\n        If True, consider NaN and None to be different.\n    dtype_equal : bool, default False\n        Whether `left` and `right` are known to have the same dtype\n        according to `is_dtype_equal`. Some methods like `BlockManager.equals`.\n        require that the dtypes match. Setting this to ``True`` can improve\n        performance, but will give different results for arrays that are\n        equal but different dtypes.\n\n    Returns\n    -------\n    b : bool\n        Returns True if the arrays are equivalent.\n\n    Examples\n    --------\n    >>> array_equivalent(\n    ...     np.array([1, 2, np.nan]),\n    ...     np.array([1, 2, np.nan]))\n    True\n    >>> array_equivalent(\n    ...     np.array([1, np.nan, 2]),\n    ...     np.array([1, 2, np.nan]))\n    False\n    '
    (left, right) = (np.asarray(left), np.asarray(right))
    if (left.shape != right.shape):
        return False
    if dtype_equal:
        if (is_float_dtype(left.dtype) or is_complex_dtype(left.dtype)):
            return _array_equivalent_float(left, right)
        elif is_datetimelike_v_numeric(left.dtype, right.dtype):
            return False
        elif needs_i8_conversion(left.dtype):
            return _array_equivalent_datetimelike(left, right)
        elif is_string_dtype(left.dtype):
            return _array_equivalent_object(left, right, strict_nan)
        else:
            return np.array_equal(left, right)
    if (is_string_dtype(left.dtype) or is_string_dtype(right.dtype)):
        return _array_equivalent_object(left, right, strict_nan)
    if (is_float_dtype(left.dtype) or is_complex_dtype(left.dtype)):
        if (not (np.prod(left.shape) and np.prod(right.shape))):
            return True
        return ((left == right) | (isna(left) & isna(right))).all()
    elif is_datetimelike_v_numeric(left, right):
        return False
    elif (needs_i8_conversion(left.dtype) or needs_i8_conversion(right.dtype)):
        if (not is_dtype_equal(left.dtype, right.dtype)):
            return False
        left = left.view('i8')
        right = right.view('i8')
    if ((left.dtype.type is np.void) or (right.dtype.type is np.void)):
        if (left.dtype != right.dtype):
            return False
    return np.array_equal(left, right)

def _array_equivalent_float(left, right):
    return ((left == right) | (np.isnan(left) & np.isnan(right))).all()

def _array_equivalent_datetimelike(left, right):
    return np.array_equal(left.view('i8'), right.view('i8'))

def _array_equivalent_object(left, right, strict_nan):
    if (not strict_nan):
        return lib.array_equivalent_object(ensure_object(left.ravel()), ensure_object(right.ravel()))
    for (left_value, right_value) in zip(left, right):
        if ((left_value is NaT) and (right_value is not NaT)):
            return False
        elif ((left_value is libmissing.NA) and (right_value is not libmissing.NA)):
            return False
        elif (isinstance(left_value, float) and np.isnan(left_value)):
            if ((not isinstance(right_value, float)) or (not np.isnan(right_value))):
                return False
        else:
            try:
                if np.any(np.asarray((left_value != right_value))):
                    return False
            except TypeError as err:
                if ('Cannot compare tz-naive' in str(err)):
                    return False
                elif ('boolean value of NA is ambiguous' in str(err)):
                    return False
                raise
    return True

def array_equals(left, right):
    '\n    ExtensionArray-compatible implementation of array_equivalent.\n    '
    if (not is_dtype_equal(left.dtype, right.dtype)):
        return False
    elif isinstance(left, ABCExtensionArray):
        return left.equals(right)
    else:
        return array_equivalent(left, right, dtype_equal=True)

def infer_fill_value(val):
    '\n    infer the fill value for the nan/NaT from the provided\n    scalar/ndarray/list-like if we are a NaT, return the correct dtyped\n    element to provide proper block construction\n    '
    if (not is_list_like(val)):
        val = [val]
    val = np.array(val, copy=False)
    if needs_i8_conversion(val.dtype):
        return np.array('NaT', dtype=val.dtype)
    elif is_object_dtype(val.dtype):
        dtype = lib.infer_dtype(ensure_object(val), skipna=False)
        if (dtype in ['datetime', 'datetime64']):
            return np.array('NaT', dtype=DT64NS_DTYPE)
        elif (dtype in ['timedelta', 'timedelta64']):
            return np.array('NaT', dtype=TD64NS_DTYPE)
    return np.nan

def maybe_fill(arr, fill_value=np.nan):
    '\n    if we have a compatible fill_value and arr dtype, then fill\n    '
    if isna_compat(arr, fill_value):
        arr.fill(fill_value)
    return arr

def na_value_for_dtype(dtype, compat=True):
    "\n    Return a dtype compat na value\n\n    Parameters\n    ----------\n    dtype : string / dtype\n    compat : bool, default True\n\n    Returns\n    -------\n    np.dtype or a pandas dtype\n\n    Examples\n    --------\n    >>> na_value_for_dtype(np.dtype('int64'))\n    0\n    >>> na_value_for_dtype(np.dtype('int64'), compat=False)\n    nan\n    >>> na_value_for_dtype(np.dtype('float64'))\n    nan\n    >>> na_value_for_dtype(np.dtype('bool'))\n    False\n    >>> na_value_for_dtype(np.dtype('datetime64[ns]'))\n    NaT\n    "
    dtype = pandas_dtype(dtype)
    if is_extension_array_dtype(dtype):
        return dtype.na_value
    if needs_i8_conversion(dtype):
        return NaT
    elif is_float_dtype(dtype):
        return np.nan
    elif is_integer_dtype(dtype):
        if compat:
            return 0
        return np.nan
    elif is_bool_dtype(dtype):
        if compat:
            return False
        return np.nan
    return np.nan

def remove_na_arraylike(arr):
    '\n    Return array-like containing only true/non-NaN values, possibly empty.\n    '
    if is_extension_array_dtype(arr):
        return arr[notna(arr)]
    else:
        return arr[notna(np.asarray(arr))]

def is_valid_nat_for_dtype(obj, dtype):
    '\n    isna check that excludes incompatible dtypes\n\n    Parameters\n    ----------\n    obj : object\n    dtype : np.datetime64, np.timedelta64, DatetimeTZDtype, or PeriodDtype\n\n    Returns\n    -------\n    bool\n    '
    if ((not lib.is_scalar(obj)) or (not isna(obj))):
        return False
    if (dtype.kind == 'M'):
        return (not isinstance(obj, np.timedelta64))
    if (dtype.kind == 'm'):
        return (not isinstance(obj, np.datetime64))
    if (dtype.kind in ['i', 'u', 'f', 'c']):
        return ((obj is not NaT) and (not isinstance(obj, (np.datetime64, np.timedelta64))))
    return (not isinstance(obj, (np.datetime64, np.timedelta64)))

def isna_all(arr):
    '\n    Optimized equivalent to isna(arr).all()\n    '
    total_len = len(arr)
    chunk_len = max((total_len // 40), 1000)
    dtype = arr.dtype
    if (dtype.kind == 'f'):
        checker = nan_checker
    elif ((dtype.kind in ['m', 'M']) or (dtype.type is Period)):
        checker = (lambda x: (np.asarray(x.view('i8')) == iNaT))
    else:
        checker = (lambda x: _isna_ndarraylike(x, inf_as_na=INF_AS_NA))
    for i in range(0, total_len, chunk_len):
        if (not checker(arr[i:(i + chunk_len)]).all()):
            return False
    return True
