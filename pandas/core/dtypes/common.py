
'\nCommon type operations.\n'
from typing import Any, Callable, Union
import warnings
import numpy as np
from pandas._libs import Interval, Period, algos
from pandas._libs.tslibs import conversion
from pandas._typing import ArrayLike, DtypeObj, Optional
from pandas.core.dtypes.base import registry
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, ExtensionDtype, IntervalDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCCategorical, ABCIndex
from pandas.core.dtypes.inference import is_array_like, is_bool, is_complex, is_dataclass, is_decimal, is_dict_like, is_file_like, is_float, is_hashable, is_integer, is_interval, is_iterator, is_list_like, is_named_tuple, is_nested_list_like, is_number, is_re, is_re_compilable, is_scalar, is_sequence
POSSIBLY_CAST_DTYPES = {np.dtype(t).name for t in ['O', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']}
DT64NS_DTYPE = conversion.DT64NS_DTYPE
TD64NS_DTYPE = conversion.TD64NS_DTYPE
INT64_DTYPE = np.dtype(np.int64)
_is_scipy_sparse = None
ensure_float64 = algos.ensure_float64
ensure_float32 = algos.ensure_float32

def ensure_float(arr):
    '\n    Ensure that an array object has a float dtype if possible.\n\n    Parameters\n    ----------\n    arr : array-like\n        The array whose data type we want to enforce as float.\n\n    Returns\n    -------\n    float_arr : The original array cast to the float dtype if\n                possible. Otherwise, the original array is returned.\n    '
    if is_extension_array_dtype(arr.dtype):
        if is_float_dtype(arr.dtype):
            arr = arr.to_numpy(dtype=arr.dtype.numpy_dtype, na_value=np.nan)
        else:
            arr = arr.to_numpy(dtype='float64', na_value=np.nan)
    elif issubclass(arr.dtype.type, (np.integer, np.bool_)):
        arr = arr.astype(float)
    return arr
ensure_uint64 = algos.ensure_uint64
ensure_int64 = algos.ensure_int64
ensure_int32 = algos.ensure_int32
ensure_int16 = algos.ensure_int16
ensure_int8 = algos.ensure_int8
ensure_platform_int = algos.ensure_platform_int
ensure_object = algos.ensure_object

def ensure_str(value):
    '\n    Ensure that bytes and non-strings get converted into ``str`` objects.\n    '
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    elif (not isinstance(value, str)):
        value = str(value)
    return value

def ensure_int_or_float(arr, copy=False):
    "\n    Ensure that an dtype array of some integer dtype\n    has an int64 dtype if possible.\n    If it's not possible, potentially because of overflow,\n    convert the array to float64 instead.\n\n    Parameters\n    ----------\n    arr : array-like\n          The array whose data type we want to enforce.\n    copy: bool\n          Whether to copy the original array or reuse\n          it in place, if possible.\n\n    Returns\n    -------\n    out_arr : The input array cast as int64 if\n              possible without overflow.\n              Otherwise the input array cast to float64.\n\n    Notes\n    -----\n    If the array is explicitly of type uint64 the type\n    will remain unchanged.\n    "
    try:
        return arr.astype('int64', copy=copy, casting='safe')
    except TypeError:
        pass
    try:
        return arr.astype('uint64', copy=copy, casting='safe')
    except TypeError:
        if is_extension_array_dtype(arr.dtype):
            return arr.to_numpy(dtype='float64', na_value=np.nan)
        return arr.astype('float64', copy=copy)

def ensure_python_int(value):
    "\n    Ensure that a value is a python int.\n\n    Parameters\n    ----------\n    value: int or numpy.integer\n\n    Returns\n    -------\n    int\n\n    Raises\n    ------\n    TypeError: if the value isn't an int or can't be converted to one.\n    "
    if (not is_scalar(value)):
        raise TypeError(f'Value needs to be a scalar value, was type {type(value).__name__}')
    try:
        new_value = int(value)
        assert (new_value == value)
    except (TypeError, ValueError, AssertionError) as err:
        raise TypeError(f'Wrong type {type(value)} for value {value}') from err
    return new_value

def classes(*klasses):
    ' evaluate if the tipo is a subclass of the klasses '
    return (lambda tipo: issubclass(tipo, klasses))

def classes_and_not_datetimelike(*klasses):
    '\n    evaluate if the tipo is a subclass of the klasses\n    and not a datetimelike\n    '
    return (lambda tipo: (issubclass(tipo, klasses) and (not issubclass(tipo, (np.datetime64, np.timedelta64)))))

def is_object_dtype(arr_or_dtype):
    '\n    Check whether an array-like or dtype is of the object dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like or dtype is of the object dtype.\n\n    Examples\n    --------\n    >>> is_object_dtype(object)\n    True\n    >>> is_object_dtype(int)\n    False\n    >>> is_object_dtype(np.array([], dtype=object))\n    True\n    >>> is_object_dtype(np.array([], dtype=int))\n    False\n    >>> is_object_dtype([1, 2, 3])\n    False\n    '
    return _is_dtype_type(arr_or_dtype, classes(np.object_))

def is_sparse(arr):
    '\n    Check whether an array-like is a 1-D pandas sparse array.\n\n    Check that the one-dimensional array-like is a pandas sparse array.\n    Returns True if it is a pandas sparse array, not another type of\n    sparse array.\n\n    Parameters\n    ----------\n    arr : array-like\n        Array-like to check.\n\n    Returns\n    -------\n    bool\n        Whether or not the array-like is a pandas sparse array.\n\n    Examples\n    --------\n    Returns `True` if the parameter is a 1-D pandas sparse array.\n\n    >>> is_sparse(pd.arrays.SparseArray([0, 0, 1, 0]))\n    True\n    >>> is_sparse(pd.Series(pd.arrays.SparseArray([0, 0, 1, 0])))\n    True\n\n    Returns `False` if the parameter is not sparse.\n\n    >>> is_sparse(np.array([0, 0, 1, 0]))\n    False\n    >>> is_sparse(pd.Series([0, 1, 0, 0]))\n    False\n\n    Returns `False` if the parameter is not a pandas sparse array.\n\n    >>> from scipy.sparse import bsr_matrix\n    >>> is_sparse(bsr_matrix([0, 1, 0, 0]))\n    False\n\n    Returns `False` if the parameter has more than one dimension.\n    '
    from pandas.core.arrays.sparse import SparseDtype
    dtype = getattr(arr, 'dtype', arr)
    return isinstance(dtype, SparseDtype)

def is_scipy_sparse(arr):
    '\n    Check whether an array-like is a scipy.sparse.spmatrix instance.\n\n    Parameters\n    ----------\n    arr : array-like\n        The array-like to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like is a scipy.sparse.spmatrix instance.\n\n    Notes\n    -----\n    If scipy is not installed, this function will always return False.\n\n    Examples\n    --------\n    >>> from scipy.sparse import bsr_matrix\n    >>> is_scipy_sparse(bsr_matrix([1, 2, 3]))\n    True\n    >>> is_scipy_sparse(pd.arrays.SparseArray([1, 2, 3]))\n    False\n    '
    global _is_scipy_sparse
    if (_is_scipy_sparse is None):
        try:
            from scipy.sparse import issparse as _is_scipy_sparse
        except ImportError:
            _is_scipy_sparse = (lambda _: False)
    assert (_is_scipy_sparse is not None)
    return _is_scipy_sparse(arr)

def is_categorical(arr):
    '\n    Check whether an array-like is a Categorical instance.\n\n    Parameters\n    ----------\n    arr : array-like\n        The array-like to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like is of a Categorical instance.\n\n    Examples\n    --------\n    >>> is_categorical([1, 2, 3])\n    False\n\n    Categoricals, Series Categoricals, and CategoricalIndex will return True.\n\n    >>> cat = pd.Categorical([1, 2, 3])\n    >>> is_categorical(cat)\n    True\n    >>> is_categorical(pd.Series(cat))\n    True\n    >>> is_categorical(pd.CategoricalIndex([1, 2, 3]))\n    True\n    '
    warnings.warn('is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead', FutureWarning, stacklevel=2)
    return (isinstance(arr, ABCCategorical) or is_categorical_dtype(arr))

def is_datetime64_dtype(arr_or_dtype):
    '\n    Check whether an array-like or dtype is of the datetime64 dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like or dtype is of the datetime64 dtype.\n\n    Examples\n    --------\n    >>> is_datetime64_dtype(object)\n    False\n    >>> is_datetime64_dtype(np.datetime64)\n    True\n    >>> is_datetime64_dtype(np.array([], dtype=int))\n    False\n    >>> is_datetime64_dtype(np.array([], dtype=np.datetime64))\n    True\n    >>> is_datetime64_dtype([1, 2, 3])\n    False\n    '
    if isinstance(arr_or_dtype, np.dtype):
        return (arr_or_dtype.kind == 'M')
    return _is_dtype_type(arr_or_dtype, classes(np.datetime64))

def is_datetime64tz_dtype(arr_or_dtype):
    '\n    Check whether an array-like or dtype is of a DatetimeTZDtype dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like or dtype is of a DatetimeTZDtype dtype.\n\n    Examples\n    --------\n    >>> is_datetime64tz_dtype(object)\n    False\n    >>> is_datetime64tz_dtype([1, 2, 3])\n    False\n    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))  # tz-naive\n    False\n    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))\n    True\n\n    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")\n    >>> s = pd.Series([], dtype=dtype)\n    >>> is_datetime64tz_dtype(dtype)\n    True\n    >>> is_datetime64tz_dtype(s)\n    True\n    '
    if isinstance(arr_or_dtype, ExtensionDtype):
        return (arr_or_dtype.kind == 'M')
    if (arr_or_dtype is None):
        return False
    return DatetimeTZDtype.is_dtype(arr_or_dtype)

def is_timedelta64_dtype(arr_or_dtype):
    '\n    Check whether an array-like or dtype is of the timedelta64 dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like or dtype is of the timedelta64 dtype.\n\n    Examples\n    --------\n    >>> is_timedelta64_dtype(object)\n    False\n    >>> is_timedelta64_dtype(np.timedelta64)\n    True\n    >>> is_timedelta64_dtype([1, 2, 3])\n    False\n    >>> is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))\n    True\n    >>> is_timedelta64_dtype(\'0 days\')\n    False\n    '
    if isinstance(arr_or_dtype, np.dtype):
        return (arr_or_dtype.kind == 'm')
    return _is_dtype_type(arr_or_dtype, classes(np.timedelta64))

def is_period_dtype(arr_or_dtype):
    '\n    Check whether an array-like or dtype is of the Period dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like or dtype is of the Period dtype.\n\n    Examples\n    --------\n    >>> is_period_dtype(object)\n    False\n    >>> is_period_dtype(PeriodDtype(freq="D"))\n    True\n    >>> is_period_dtype([1, 2, 3])\n    False\n    >>> is_period_dtype(pd.Period("2017-01-01"))\n    False\n    >>> is_period_dtype(pd.PeriodIndex([], freq="A"))\n    True\n    '
    if isinstance(arr_or_dtype, ExtensionDtype):
        return (arr_or_dtype.type is Period)
    if (arr_or_dtype is None):
        return False
    return PeriodDtype.is_dtype(arr_or_dtype)

def is_interval_dtype(arr_or_dtype):
    '\n    Check whether an array-like or dtype is of the Interval dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like or dtype is of the Interval dtype.\n\n    Examples\n    --------\n    >>> is_interval_dtype(object)\n    False\n    >>> is_interval_dtype(IntervalDtype())\n    True\n    >>> is_interval_dtype([1, 2, 3])\n    False\n    >>>\n    >>> interval = pd.Interval(1, 2, closed="right")\n    >>> is_interval_dtype(interval)\n    False\n    >>> is_interval_dtype(pd.IntervalIndex([interval]))\n    True\n    '
    if isinstance(arr_or_dtype, ExtensionDtype):
        return (arr_or_dtype.type is Interval)
    if (arr_or_dtype is None):
        return False
    return IntervalDtype.is_dtype(arr_or_dtype)

def is_categorical_dtype(arr_or_dtype):
    '\n    Check whether an array-like or dtype is of the Categorical dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like or dtype is of the Categorical dtype.\n\n    Examples\n    --------\n    >>> is_categorical_dtype(object)\n    False\n    >>> is_categorical_dtype(CategoricalDtype())\n    True\n    >>> is_categorical_dtype([1, 2, 3])\n    False\n    >>> is_categorical_dtype(pd.Categorical([1, 2, 3]))\n    True\n    >>> is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))\n    True\n    '
    if isinstance(arr_or_dtype, ExtensionDtype):
        return (arr_or_dtype.name == 'category')
    if (arr_or_dtype is None):
        return False
    return CategoricalDtype.is_dtype(arr_or_dtype)

def is_string_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of the string dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of the string dtype.\n\n    Examples\n    --------\n    >>> is_string_dtype(str)\n    True\n    >>> is_string_dtype(object)\n    True\n    >>> is_string_dtype(int)\n    False\n    >>>\n    >>> is_string_dtype(np.array(['a', 'b']))\n    True\n    >>> is_string_dtype(pd.Series([1, 2]))\n    False\n    "

    def condition(dtype) -> bool:
        return ((dtype.kind in ('O', 'S', 'U')) and (not is_excluded_dtype(dtype)))

    def is_excluded_dtype(dtype) -> bool:
        '\n        These have kind = "O" but aren\'t string dtypes so need to be explicitly excluded\n        '
        is_excluded_checks = (is_period_dtype, is_interval_dtype, is_categorical_dtype)
        return any((is_excluded(dtype) for is_excluded in is_excluded_checks))
    return _is_dtype(arr_or_dtype, condition)

def is_dtype_equal(source, target):
    '\n    Check if two dtypes are equal.\n\n    Parameters\n    ----------\n    source : The first dtype to compare\n    target : The second dtype to compare\n\n    Returns\n    -------\n    boolean\n        Whether or not the two dtypes are equal.\n\n    Examples\n    --------\n    >>> is_dtype_equal(int, float)\n    False\n    >>> is_dtype_equal("int", int)\n    True\n    >>> is_dtype_equal(object, "category")\n    False\n    >>> is_dtype_equal(CategoricalDtype(), "category")\n    True\n    >>> is_dtype_equal(DatetimeTZDtype(tz="UTC"), "datetime64")\n    False\n    '
    if isinstance(target, str):
        if (not isinstance(source, str)):
            try:
                src = get_dtype(source)
                if isinstance(src, ExtensionDtype):
                    return (src == target)
            except (TypeError, AttributeError):
                return False
    elif isinstance(source, str):
        return is_dtype_equal(target, source)
    try:
        source = get_dtype(source)
        target = get_dtype(target)
        return (source == target)
    except (TypeError, AttributeError):
        return False

def is_any_int_dtype(arr_or_dtype):
    '\n    Check whether the provided array or dtype is of an integer dtype.\n\n    In this function, timedelta64 instances are also considered "any-integer"\n    type objects and will return True.\n\n    This function is internal and should not be exposed in the public API.\n\n    .. versionchanged:: 0.24.0\n\n       The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered\n       as integer by this function.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of an integer dtype.\n\n    Examples\n    --------\n    >>> is_any_int_dtype(str)\n    False\n    >>> is_any_int_dtype(int)\n    True\n    >>> is_any_int_dtype(float)\n    False\n    >>> is_any_int_dtype(np.uint64)\n    True\n    >>> is_any_int_dtype(np.datetime64)\n    False\n    >>> is_any_int_dtype(np.timedelta64)\n    True\n    >>> is_any_int_dtype(np.array([\'a\', \'b\']))\n    False\n    >>> is_any_int_dtype(pd.Series([1, 2]))\n    True\n    >>> is_any_int_dtype(np.array([], dtype=np.timedelta64))\n    True\n    >>> is_any_int_dtype(pd.Index([1, 2.]))  # float\n    False\n    '
    return _is_dtype_type(arr_or_dtype, classes(np.integer, np.timedelta64))

def is_integer_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of an integer dtype.\n\n    Unlike in `in_any_int_dtype`, timedelta64 instances will return False.\n\n    .. versionchanged:: 0.24.0\n\n       The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered\n       as integer by this function.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of an integer dtype and\n        not an instance of timedelta64.\n\n    Examples\n    --------\n    >>> is_integer_dtype(str)\n    False\n    >>> is_integer_dtype(int)\n    True\n    >>> is_integer_dtype(float)\n    False\n    >>> is_integer_dtype(np.uint64)\n    True\n    >>> is_integer_dtype('int8')\n    True\n    >>> is_integer_dtype('Int8')\n    True\n    >>> is_integer_dtype(pd.Int8Dtype)\n    True\n    >>> is_integer_dtype(np.datetime64)\n    False\n    >>> is_integer_dtype(np.timedelta64)\n    False\n    >>> is_integer_dtype(np.array(['a', 'b']))\n    False\n    >>> is_integer_dtype(pd.Series([1, 2]))\n    True\n    >>> is_integer_dtype(np.array([], dtype=np.timedelta64))\n    False\n    >>> is_integer_dtype(pd.Index([1, 2.]))  # float\n    False\n    "
    return _is_dtype_type(arr_or_dtype, classes_and_not_datetimelike(np.integer))

def is_signed_integer_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of a signed integer dtype.\n\n    Unlike in `in_any_int_dtype`, timedelta64 instances will return False.\n\n    .. versionchanged:: 0.24.0\n\n       The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered\n       as integer by this function.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of a signed integer dtype\n        and not an instance of timedelta64.\n\n    Examples\n    --------\n    >>> is_signed_integer_dtype(str)\n    False\n    >>> is_signed_integer_dtype(int)\n    True\n    >>> is_signed_integer_dtype(float)\n    False\n    >>> is_signed_integer_dtype(np.uint64)  # unsigned\n    False\n    >>> is_signed_integer_dtype('int8')\n    True\n    >>> is_signed_integer_dtype('Int8')\n    True\n    >>> is_signed_integer_dtype(pd.Int8Dtype)\n    True\n    >>> is_signed_integer_dtype(np.datetime64)\n    False\n    >>> is_signed_integer_dtype(np.timedelta64)\n    False\n    >>> is_signed_integer_dtype(np.array(['a', 'b']))\n    False\n    >>> is_signed_integer_dtype(pd.Series([1, 2]))\n    True\n    >>> is_signed_integer_dtype(np.array([], dtype=np.timedelta64))\n    False\n    >>> is_signed_integer_dtype(pd.Index([1, 2.]))  # float\n    False\n    >>> is_signed_integer_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned\n    False\n    "
    return _is_dtype_type(arr_or_dtype, classes_and_not_datetimelike(np.signedinteger))

def is_unsigned_integer_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of an unsigned integer dtype.\n\n    .. versionchanged:: 0.24.0\n\n       The nullable Integer dtypes (e.g. pandas.UInt64Dtype) are also\n       considered as integer by this function.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of an unsigned integer dtype.\n\n    Examples\n    --------\n    >>> is_unsigned_integer_dtype(str)\n    False\n    >>> is_unsigned_integer_dtype(int)  # signed\n    False\n    >>> is_unsigned_integer_dtype(float)\n    False\n    >>> is_unsigned_integer_dtype(np.uint64)\n    True\n    >>> is_unsigned_integer_dtype('uint8')\n    True\n    >>> is_unsigned_integer_dtype('UInt8')\n    True\n    >>> is_unsigned_integer_dtype(pd.UInt8Dtype)\n    True\n    >>> is_unsigned_integer_dtype(np.array(['a', 'b']))\n    False\n    >>> is_unsigned_integer_dtype(pd.Series([1, 2]))  # signed\n    False\n    >>> is_unsigned_integer_dtype(pd.Index([1, 2.]))  # float\n    False\n    >>> is_unsigned_integer_dtype(np.array([1, 2], dtype=np.uint32))\n    True\n    "
    return _is_dtype_type(arr_or_dtype, classes_and_not_datetimelike(np.unsignedinteger))

def is_int64_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of the int64 dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of the int64 dtype.\n\n    Notes\n    -----\n    Depending on system architecture, the return value of `is_int64_dtype(\n    int)` will be True if the OS uses 64-bit integers and False if the OS\n    uses 32-bit integers.\n\n    Examples\n    --------\n    >>> is_int64_dtype(str)\n    False\n    >>> is_int64_dtype(np.int32)\n    False\n    >>> is_int64_dtype(np.int64)\n    True\n    >>> is_int64_dtype('int8')\n    False\n    >>> is_int64_dtype('Int8')\n    False\n    >>> is_int64_dtype(pd.Int64Dtype)\n    True\n    >>> is_int64_dtype(float)\n    False\n    >>> is_int64_dtype(np.uint64)  # unsigned\n    False\n    >>> is_int64_dtype(np.array(['a', 'b']))\n    False\n    >>> is_int64_dtype(np.array([1, 2], dtype=np.int64))\n    True\n    >>> is_int64_dtype(pd.Index([1, 2.]))  # float\n    False\n    >>> is_int64_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned\n    False\n    "
    return _is_dtype_type(arr_or_dtype, classes(np.int64))

def is_datetime64_any_dtype(arr_or_dtype):
    '\n    Check whether the provided array or dtype is of the datetime64 dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    bool\n        Whether or not the array or dtype is of the datetime64 dtype.\n\n    Examples\n    --------\n    >>> is_datetime64_any_dtype(str)\n    False\n    >>> is_datetime64_any_dtype(int)\n    False\n    >>> is_datetime64_any_dtype(np.datetime64)  # can be tz-naive\n    True\n    >>> is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))\n    True\n    >>> is_datetime64_any_dtype(np.array([\'a\', \'b\']))\n    False\n    >>> is_datetime64_any_dtype(np.array([1, 2]))\n    False\n    >>> is_datetime64_any_dtype(np.array([], dtype="datetime64[ns]"))\n    True\n    >>> is_datetime64_any_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))\n    True\n    '
    if isinstance(arr_or_dtype, (np.dtype, ExtensionDtype)):
        return (arr_or_dtype.kind == 'M')
    if (arr_or_dtype is None):
        return False
    return (is_datetime64_dtype(arr_or_dtype) or is_datetime64tz_dtype(arr_or_dtype))

def is_datetime64_ns_dtype(arr_or_dtype):
    '\n    Check whether the provided array or dtype is of the datetime64[ns] dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    bool\n        Whether or not the array or dtype is of the datetime64[ns] dtype.\n\n    Examples\n    --------\n    >>> is_datetime64_ns_dtype(str)\n    False\n    >>> is_datetime64_ns_dtype(int)\n    False\n    >>> is_datetime64_ns_dtype(np.datetime64)  # no unit\n    False\n    >>> is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))\n    True\n    >>> is_datetime64_ns_dtype(np.array([\'a\', \'b\']))\n    False\n    >>> is_datetime64_ns_dtype(np.array([1, 2]))\n    False\n    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64"))  # no unit\n    False\n    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))  # wrong unit\n    False\n    >>> is_datetime64_ns_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))\n    True\n    '
    if (arr_or_dtype is None):
        return False
    try:
        tipo = get_dtype(arr_or_dtype)
    except TypeError:
        if is_datetime64tz_dtype(arr_or_dtype):
            tipo = get_dtype(arr_or_dtype.dtype)
        else:
            return False
    return ((tipo == DT64NS_DTYPE) or (getattr(tipo, 'base', None) == DT64NS_DTYPE))

def is_timedelta64_ns_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of the timedelta64[ns] dtype.\n\n    This is a very specific dtype, so generic ones like `np.timedelta64`\n    will return False if passed into this function.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of the timedelta64[ns] dtype.\n\n    Examples\n    --------\n    >>> is_timedelta64_ns_dtype(np.dtype('m8[ns]'))\n    True\n    >>> is_timedelta64_ns_dtype(np.dtype('m8[ps]'))  # Wrong frequency\n    False\n    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype='m8[ns]'))\n    True\n    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))\n    False\n    "
    return _is_dtype(arr_or_dtype, (lambda dtype: (dtype == TD64NS_DTYPE)))

def is_datetime_or_timedelta_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of\n    a timedelta64 or datetime64 dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of a timedelta64,\n        or datetime64 dtype.\n\n    Examples\n    --------\n    >>> is_datetime_or_timedelta_dtype(str)\n    False\n    >>> is_datetime_or_timedelta_dtype(int)\n    False\n    >>> is_datetime_or_timedelta_dtype(np.datetime64)\n    True\n    >>> is_datetime_or_timedelta_dtype(np.timedelta64)\n    True\n    >>> is_datetime_or_timedelta_dtype(np.array(['a', 'b']))\n    False\n    >>> is_datetime_or_timedelta_dtype(pd.Series([1, 2]))\n    False\n    >>> is_datetime_or_timedelta_dtype(np.array([], dtype=np.timedelta64))\n    True\n    >>> is_datetime_or_timedelta_dtype(np.array([], dtype=np.datetime64))\n    True\n    "
    return _is_dtype_type(arr_or_dtype, classes(np.datetime64, np.timedelta64))

def is_numeric_v_string_like(a, b):
    '\n    Check if we are comparing a string-like object to a numeric ndarray.\n    NumPy doesn\'t like to compare such objects, especially numeric arrays\n    and scalar string-likes.\n\n    Parameters\n    ----------\n    a : array-like, scalar\n        The first object to check.\n    b : array-like, scalar\n        The second object to check.\n\n    Returns\n    -------\n    boolean\n        Whether we return a comparing a string-like object to a numeric array.\n\n    Examples\n    --------\n    >>> is_numeric_v_string_like(1, 1)\n    False\n    >>> is_numeric_v_string_like("foo", "foo")\n    False\n    >>> is_numeric_v_string_like(1, "foo")  # non-array numeric\n    False\n    >>> is_numeric_v_string_like(np.array([1]), "foo")\n    True\n    >>> is_numeric_v_string_like("foo", np.array([1]))  # symmetric check\n    True\n    >>> is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))\n    True\n    >>> is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))\n    True\n    >>> is_numeric_v_string_like(np.array([1]), np.array([2]))\n    False\n    >>> is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))\n    False\n    '
    is_a_array = isinstance(a, np.ndarray)
    is_b_array = isinstance(b, np.ndarray)
    is_a_numeric_array = (is_a_array and is_numeric_dtype(a))
    is_b_numeric_array = (is_b_array and is_numeric_dtype(b))
    is_a_string_array = (is_a_array and is_string_like_dtype(a))
    is_b_string_array = (is_b_array and is_string_like_dtype(b))
    is_a_scalar_string_like = ((not is_a_array) and isinstance(a, str))
    is_b_scalar_string_like = ((not is_b_array) and isinstance(b, str))
    return ((is_a_numeric_array and is_b_scalar_string_like) or (is_b_numeric_array and is_a_scalar_string_like) or (is_a_numeric_array and is_b_string_array) or (is_b_numeric_array and is_a_string_array))

def is_datetimelike_v_numeric(a, b):
    '\n    Check if we are comparing a datetime-like object to a numeric object.\n    By "numeric," we mean an object that is either of an int or float dtype.\n\n    Parameters\n    ----------\n    a : array-like, scalar\n        The first object to check.\n    b : array-like, scalar\n        The second object to check.\n\n    Returns\n    -------\n    boolean\n        Whether we return a comparing a datetime-like to a numeric object.\n\n    Examples\n    --------\n    >>> from datetime import datetime\n    >>> dt = np.datetime64(datetime(2017, 1, 1))\n    >>>\n    >>> is_datetimelike_v_numeric(1, 1)\n    False\n    >>> is_datetimelike_v_numeric(dt, dt)\n    False\n    >>> is_datetimelike_v_numeric(1, dt)\n    True\n    >>> is_datetimelike_v_numeric(dt, 1)  # symmetric check\n    True\n    >>> is_datetimelike_v_numeric(np.array([dt]), 1)\n    True\n    >>> is_datetimelike_v_numeric(np.array([1]), dt)\n    True\n    >>> is_datetimelike_v_numeric(np.array([dt]), np.array([1]))\n    True\n    >>> is_datetimelike_v_numeric(np.array([1]), np.array([2]))\n    False\n    >>> is_datetimelike_v_numeric(np.array([dt]), np.array([dt]))\n    False\n    '
    if (not hasattr(a, 'dtype')):
        a = np.asarray(a)
    if (not hasattr(b, 'dtype')):
        b = np.asarray(b)

    def is_numeric(x):
        '\n        Check if an object has a numeric dtype (i.e. integer or float).\n        '
        return (is_integer_dtype(x) or is_float_dtype(x))
    return ((needs_i8_conversion(a) and is_numeric(b)) or (needs_i8_conversion(b) and is_numeric(a)))

def needs_i8_conversion(arr_or_dtype):
    '\n    Check whether the array or dtype should be converted to int64.\n\n    An array-like or dtype "needs" such a conversion if the array-like\n    or dtype is of a datetime-like dtype\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype should be converted to int64.\n\n    Examples\n    --------\n    >>> needs_i8_conversion(str)\n    False\n    >>> needs_i8_conversion(np.int64)\n    False\n    >>> needs_i8_conversion(np.datetime64)\n    True\n    >>> needs_i8_conversion(np.array([\'a\', \'b\']))\n    False\n    >>> needs_i8_conversion(pd.Series([1, 2]))\n    False\n    >>> needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))\n    True\n    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))\n    True\n    '
    if (arr_or_dtype is None):
        return False
    if isinstance(arr_or_dtype, (np.dtype, ExtensionDtype)):
        dtype = arr_or_dtype
        return ((dtype.kind in ['m', 'M']) or (dtype.type is Period))
    return (is_datetime_or_timedelta_dtype(arr_or_dtype) or is_datetime64tz_dtype(arr_or_dtype) or is_period_dtype(arr_or_dtype))

def is_numeric_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of a numeric dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of a numeric dtype.\n\n    Examples\n    --------\n    >>> is_numeric_dtype(str)\n    False\n    >>> is_numeric_dtype(int)\n    True\n    >>> is_numeric_dtype(float)\n    True\n    >>> is_numeric_dtype(np.uint64)\n    True\n    >>> is_numeric_dtype(np.datetime64)\n    False\n    >>> is_numeric_dtype(np.timedelta64)\n    False\n    >>> is_numeric_dtype(np.array(['a', 'b']))\n    False\n    >>> is_numeric_dtype(pd.Series([1, 2]))\n    True\n    >>> is_numeric_dtype(pd.Index([1, 2.]))\n    True\n    >>> is_numeric_dtype(np.array([], dtype=np.timedelta64))\n    False\n    "
    return _is_dtype_type(arr_or_dtype, classes_and_not_datetimelike(np.number, np.bool_))

def is_string_like_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of a string-like dtype.\n\n    Unlike `is_string_dtype`, the object dtype is excluded because it\n    is a mixed dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of the string dtype.\n\n    Examples\n    --------\n    >>> is_string_like_dtype(str)\n    True\n    >>> is_string_like_dtype(object)\n    False\n    >>> is_string_like_dtype(np.array(['a', 'b']))\n    True\n    >>> is_string_like_dtype(pd.Series([1, 2]))\n    False\n    "
    return _is_dtype(arr_or_dtype, (lambda dtype: (dtype.kind in ('S', 'U'))))

def is_float_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of a float dtype.\n\n    This function is internal and should not be exposed in the public API.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of a float dtype.\n\n    Examples\n    --------\n    >>> is_float_dtype(str)\n    False\n    >>> is_float_dtype(int)\n    False\n    >>> is_float_dtype(float)\n    True\n    >>> is_float_dtype(np.array(['a', 'b']))\n    False\n    >>> is_float_dtype(pd.Series([1, 2]))\n    False\n    >>> is_float_dtype(pd.Index([1, 2.]))\n    True\n    "
    return _is_dtype_type(arr_or_dtype, classes(np.floating))

def is_bool_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of a boolean dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of a boolean dtype.\n\n    Notes\n    -----\n    An ExtensionArray is considered boolean when the ``_is_boolean``\n    attribute is set to True.\n\n    Examples\n    --------\n    >>> is_bool_dtype(str)\n    False\n    >>> is_bool_dtype(int)\n    False\n    >>> is_bool_dtype(bool)\n    True\n    >>> is_bool_dtype(np.bool_)\n    True\n    >>> is_bool_dtype(np.array(['a', 'b']))\n    False\n    >>> is_bool_dtype(pd.Series([1, 2]))\n    False\n    >>> is_bool_dtype(np.array([True, False]))\n    True\n    >>> is_bool_dtype(pd.Categorical([True, False]))\n    True\n    >>> is_bool_dtype(pd.arrays.SparseArray([True, False]))\n    True\n    "
    if (arr_or_dtype is None):
        return False
    try:
        dtype = get_dtype(arr_or_dtype)
    except TypeError:
        return False
    if isinstance(arr_or_dtype, CategoricalDtype):
        arr_or_dtype = arr_or_dtype.categories
    if isinstance(arr_or_dtype, ABCIndex):
        return (arr_or_dtype.is_object and (arr_or_dtype.inferred_type == 'boolean'))
    elif is_extension_array_dtype(arr_or_dtype):
        return getattr(dtype, '_is_boolean', False)
    return issubclass(dtype.type, np.bool_)

def is_extension_type(arr):
    '\n    Check whether an array-like is of a pandas extension class instance.\n\n    .. deprecated:: 1.0.0\n        Use ``is_extension_array_dtype`` instead.\n\n    Extension classes include categoricals, pandas sparse objects (i.e.\n    classes represented within the pandas library and not ones external\n    to it like scipy sparse matrices), and datetime-like arrays.\n\n    Parameters\n    ----------\n    arr : array-like\n        The array-like to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array-like is of a pandas extension class instance.\n\n    Examples\n    --------\n    >>> is_extension_type([1, 2, 3])\n    False\n    >>> is_extension_type(np.array([1, 2, 3]))\n    False\n    >>>\n    >>> cat = pd.Categorical([1, 2, 3])\n    >>>\n    >>> is_extension_type(cat)\n    True\n    >>> is_extension_type(pd.Series(cat))\n    True\n    >>> is_extension_type(pd.arrays.SparseArray([1, 2, 3]))\n    True\n    >>> from scipy.sparse import bsr_matrix\n    >>> is_extension_type(bsr_matrix([1, 2, 3]))\n    False\n    >>> is_extension_type(pd.DatetimeIndex([1, 2, 3]))\n    False\n    >>> is_extension_type(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))\n    True\n    >>>\n    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")\n    >>> s = pd.Series([], dtype=dtype)\n    >>> is_extension_type(s)\n    True\n    '
    warnings.warn("'is_extension_type' is deprecated and will be removed in a future version.  Use 'is_extension_array_dtype' instead.", FutureWarning, stacklevel=2)
    if is_categorical_dtype(arr):
        return True
    elif is_sparse(arr):
        return True
    elif is_datetime64tz_dtype(arr):
        return True
    return False

def is_extension_array_dtype(arr_or_dtype):
    "\n    Check if an object is a pandas extension array type.\n\n    See the :ref:`Use Guide <extending.extension-types>` for more.\n\n    Parameters\n    ----------\n    arr_or_dtype : object\n        For array-like input, the ``.dtype`` attribute will\n        be extracted.\n\n    Returns\n    -------\n    bool\n        Whether the `arr_or_dtype` is an extension array type.\n\n    Notes\n    -----\n    This checks whether an object implements the pandas extension\n    array interface. In pandas, this includes:\n\n    * Categorical\n    * Sparse\n    * Interval\n    * Period\n    * DatetimeArray\n    * TimedeltaArray\n\n    Third-party libraries may implement arrays or types satisfying\n    this interface as well.\n\n    Examples\n    --------\n    >>> from pandas.api.types import is_extension_array_dtype\n    >>> arr = pd.Categorical(['a', 'b'])\n    >>> is_extension_array_dtype(arr)\n    True\n    >>> is_extension_array_dtype(arr.dtype)\n    True\n\n    >>> arr = np.array(['a', 'b'])\n    >>> is_extension_array_dtype(arr.dtype)\n    False\n    "
    dtype = getattr(arr_or_dtype, 'dtype', arr_or_dtype)
    return (isinstance(dtype, ExtensionDtype) or (registry.find(dtype) is not None))

def is_complex_dtype(arr_or_dtype):
    "\n    Check whether the provided array or dtype is of a complex dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array or dtype to check.\n\n    Returns\n    -------\n    boolean\n        Whether or not the array or dtype is of a complex dtype.\n\n    Examples\n    --------\n    >>> is_complex_dtype(str)\n    False\n    >>> is_complex_dtype(int)\n    False\n    >>> is_complex_dtype(np.complex_)\n    True\n    >>> is_complex_dtype(np.array(['a', 'b']))\n    False\n    >>> is_complex_dtype(pd.Series([1, 2]))\n    False\n    >>> is_complex_dtype(np.array([1 + 1j, 5]))\n    True\n    "
    return _is_dtype_type(arr_or_dtype, classes(np.complexfloating))

def _is_dtype(arr_or_dtype, condition):
    '\n    Return a boolean if the condition is satisfied for the arr_or_dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like, str, np.dtype, or ExtensionArrayType\n        The array-like or dtype object whose dtype we want to extract.\n    condition : callable[Union[np.dtype, ExtensionDtype]]\n\n    Returns\n    -------\n    bool\n\n    '
    if (arr_or_dtype is None):
        return False
    try:
        dtype = get_dtype(arr_or_dtype)
    except (TypeError, ValueError, UnicodeEncodeError):
        return False
    return condition(dtype)

def get_dtype(arr_or_dtype):
    '\n    Get the dtype instance associated with an array\n    or dtype object.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype object whose dtype we want to extract.\n\n    Returns\n    -------\n    obj_dtype : The extract dtype instance from the\n                passed in array or dtype object.\n\n    Raises\n    ------\n    TypeError : The passed in object is None.\n    '
    if (arr_or_dtype is None):
        raise TypeError('Cannot deduce dtype from null object')
    elif isinstance(arr_or_dtype, np.dtype):
        return arr_or_dtype
    elif isinstance(arr_or_dtype, type):
        return np.dtype(arr_or_dtype)
    elif hasattr(arr_or_dtype, 'dtype'):
        arr_or_dtype = arr_or_dtype.dtype
    return pandas_dtype(arr_or_dtype)

def _is_dtype_type(arr_or_dtype, condition):
    '\n    Return a boolean if the condition is satisfied for the arr_or_dtype.\n\n    Parameters\n    ----------\n    arr_or_dtype : array-like\n        The array-like or dtype object whose dtype we want to extract.\n    condition : callable[Union[np.dtype, ExtensionDtypeType]]\n\n    Returns\n    -------\n    bool : if the condition is satisfied for the arr_or_dtype\n    '
    if (arr_or_dtype is None):
        return condition(type(None))
    if isinstance(arr_or_dtype, np.dtype):
        return condition(arr_or_dtype.type)
    elif isinstance(arr_or_dtype, type):
        if issubclass(arr_or_dtype, ExtensionDtype):
            arr_or_dtype = arr_or_dtype.type
        return condition(np.dtype(arr_or_dtype).type)
    if hasattr(arr_or_dtype, 'dtype'):
        arr_or_dtype = arr_or_dtype.dtype
    elif is_list_like(arr_or_dtype):
        return condition(type(None))
    try:
        tipo = pandas_dtype(arr_or_dtype).type
    except (TypeError, ValueError, UnicodeEncodeError):
        if is_scalar(arr_or_dtype):
            return condition(type(None))
        return False
    return condition(tipo)

def infer_dtype_from_object(dtype):
    '\n    Get a numpy dtype.type-style object for a dtype object.\n\n    This methods also includes handling of the datetime64[ns] and\n    datetime64[ns, TZ] objects.\n\n    If no dtype can be found, we return ``object``.\n\n    Parameters\n    ----------\n    dtype : dtype, type\n        The dtype object whose numpy dtype.type-style\n        object we want to extract.\n\n    Returns\n    -------\n    dtype_object : The extracted numpy dtype.type-style object.\n    '
    if (isinstance(dtype, type) and issubclass(dtype, np.generic)):
        return dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        try:
            _validate_date_like_dtype(dtype)
        except TypeError:
            pass
        return dtype.type
    try:
        dtype = pandas_dtype(dtype)
    except TypeError:
        pass
    if is_extension_array_dtype(dtype):
        return dtype.type
    elif isinstance(dtype, str):
        if (dtype in ['datetimetz', 'datetime64tz']):
            return DatetimeTZDtype.type
        elif (dtype in ['period']):
            raise NotImplementedError
        if ((dtype == 'datetime') or (dtype == 'timedelta')):
            dtype += '64'
        try:
            return infer_dtype_from_object(getattr(np, dtype))
        except (AttributeError, TypeError):
            pass
    return infer_dtype_from_object(np.dtype(dtype))

def _validate_date_like_dtype(dtype):
    '\n    Check whether the dtype is a date-like dtype. Raises an error if invalid.\n\n    Parameters\n    ----------\n    dtype : dtype, type\n        The dtype to check.\n\n    Raises\n    ------\n    TypeError : The dtype could not be casted to a date-like dtype.\n    ValueError : The dtype is an illegal date-like dtype (e.g. the\n                 frequency provided is too specific)\n    '
    try:
        typ = np.datetime_data(dtype)[0]
    except ValueError as e:
        raise TypeError(e) from e
    if ((typ != 'generic') and (typ != 'ns')):
        raise ValueError(f'{repr(dtype.name)} is too specific of a frequency, try passing {repr(dtype.type.__name__)}')

def validate_all_hashable(*args, error_name=None):
    '\n    Return None if all args are hashable, else raise a TypeError.\n\n    Parameters\n    ----------\n    *args\n        Arguments to validate.\n    error_name : str, optional\n        The name to use if error\n\n    Raises\n    ------\n    TypeError : If an argument is not hashable\n\n    Returns\n    -------\n    None\n    '
    if (not all((is_hashable(arg) for arg in args))):
        if error_name:
            raise TypeError(f'{error_name} must be a hashable type')
        else:
            raise TypeError('All elements must be hashable')

def pandas_dtype(dtype):
    '\n    Convert input into a pandas only dtype object or a numpy dtype object.\n\n    Parameters\n    ----------\n    dtype : object to be converted\n\n    Returns\n    -------\n    np.dtype or a pandas dtype\n\n    Raises\n    ------\n    TypeError if not a dtype\n    '
    if isinstance(dtype, np.ndarray):
        return dtype.dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype
    result = registry.find(dtype)
    if (result is not None):
        return result
    try:
        npdtype = np.dtype(dtype)
    except SyntaxError as err:
        raise TypeError(f"data type '{dtype}' not understood") from err
    if (is_hashable(dtype) and (dtype in [object, np.object_, 'object', 'O'])):
        return npdtype
    elif (npdtype.kind == 'O'):
        raise TypeError(f"dtype '{dtype}' not understood")
    return npdtype
