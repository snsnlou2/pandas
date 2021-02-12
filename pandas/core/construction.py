
'\nConstructor functions intended to be shared by pd.array, Series.__init__,\nand Index.__new__.\n\nThese should not depend on core.internals.\n'
from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union, cast
import numpy as np
import numpy.ma as ma
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency, OutOfBoundsDatetime
from pandas._typing import AnyArrayLike, ArrayLike, Dtype, DtypeObj
from pandas.core.dtypes.base import ExtensionDtype, registry
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar, construct_1d_ndarray_preserving_na, construct_1d_object_array_from_listlike, maybe_cast_to_datetime, maybe_cast_to_integer_array, maybe_castable, maybe_convert_platform, maybe_upcast
from pandas.core.dtypes.common import is_datetime64_ns_dtype, is_extension_array_dtype, is_float_dtype, is_integer_dtype, is_list_like, is_object_dtype, is_sparse, is_string_dtype, is_timedelta64_ns_dtype
from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCPandasArray, ABCSeries
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
if TYPE_CHECKING:
    from pandas import ExtensionArray, Index, Series

def array(data, dtype=None, copy=True):
    '\n    Create an array.\n\n    .. versionadded:: 0.24.0\n\n    Parameters\n    ----------\n    data : Sequence of objects\n        The scalars inside `data` should be instances of the\n        scalar type for `dtype`. It\'s expected that `data`\n        represents a 1-dimensional array of data.\n\n        When `data` is an Index or Series, the underlying array\n        will be extracted from `data`.\n\n    dtype : str, np.dtype, or ExtensionDtype, optional\n        The dtype to use for the array. This may be a NumPy\n        dtype or an extension type registered with pandas using\n        :meth:`pandas.api.extensions.register_extension_dtype`.\n\n        If not specified, there are two possibilities:\n\n        1. When `data` is a :class:`Series`, :class:`Index`, or\n           :class:`ExtensionArray`, the `dtype` will be taken\n           from the data.\n        2. Otherwise, pandas will attempt to infer the `dtype`\n           from the data.\n\n        Note that when `data` is a NumPy array, ``data.dtype`` is\n        *not* used for inferring the array type. This is because\n        NumPy cannot represent all the types of data that can be\n        held in extension arrays.\n\n        Currently, pandas will infer an extension dtype for sequences of\n\n        ============================== =====================================\n        Scalar Type                    Array Type\n        ============================== =====================================\n        :class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`\n        :class:`pandas.Period`         :class:`pandas.arrays.PeriodArray`\n        :class:`datetime.datetime`     :class:`pandas.arrays.DatetimeArray`\n        :class:`datetime.timedelta`    :class:`pandas.arrays.TimedeltaArray`\n        :class:`int`                   :class:`pandas.arrays.IntegerArray`\n        :class:`float`                 :class:`pandas.arrays.FloatingArray`\n        :class:`str`                   :class:`pandas.arrays.StringArray`\n        :class:`bool`                  :class:`pandas.arrays.BooleanArray`\n        ============================== =====================================\n\n        For all other cases, NumPy\'s usual inference rules will be used.\n\n        .. versionchanged:: 1.0.0\n\n           Pandas infers nullable-integer dtype for integer data,\n           string dtype for string data, and nullable-boolean dtype\n           for boolean data.\n\n        .. versionchanged:: 1.2.0\n\n            Pandas now also infers nullable-floating dtype for float-like\n            input data\n\n    copy : bool, default True\n        Whether to copy the data, even if not necessary. Depending\n        on the type of `data`, creating the new array may require\n        copying data, even if ``copy=False``.\n\n    Returns\n    -------\n    ExtensionArray\n        The newly created array.\n\n    Raises\n    ------\n    ValueError\n        When `data` is not 1-dimensional.\n\n    See Also\n    --------\n    numpy.array : Construct a NumPy array.\n    Series : Construct a pandas Series.\n    Index : Construct a pandas Index.\n    arrays.PandasArray : ExtensionArray wrapping a NumPy array.\n    Series.array : Extract the array stored within a Series.\n\n    Notes\n    -----\n    Omitting the `dtype` argument means pandas will attempt to infer the\n    best array type from the values in the data. As new array types are\n    added by pandas and 3rd party libraries, the "best" array type may\n    change. We recommend specifying `dtype` to ensure that\n\n    1. the correct array type for the data is returned\n    2. the returned array type doesn\'t change as new extension types\n       are added by pandas and third-party libraries\n\n    Additionally, if the underlying memory representation of the returned\n    array matters, we recommend specifying the `dtype` as a concrete object\n    rather than a string alias or allowing it to be inferred. For example,\n    a future version of pandas or a 3rd-party library may include a\n    dedicated ExtensionArray for string data. In this event, the following\n    would no longer return a :class:`arrays.PandasArray` backed by a NumPy\n    array.\n\n    >>> pd.array([\'a\', \'b\'], dtype=str)\n    <PandasArray>\n    [\'a\', \'b\']\n    Length: 2, dtype: str32\n\n    This would instead return the new ExtensionArray dedicated for string\n    data. If you really need the new array to be backed by a  NumPy array,\n    specify that in the dtype.\n\n    >>> pd.array([\'a\', \'b\'], dtype=np.dtype("<U1"))\n    <PandasArray>\n    [\'a\', \'b\']\n    Length: 2, dtype: str32\n\n    Finally, Pandas has arrays that mostly overlap with NumPy\n\n      * :class:`arrays.DatetimeArray`\n      * :class:`arrays.TimedeltaArray`\n\n    When data with a ``datetime64[ns]`` or ``timedelta64[ns]`` dtype is\n    passed, pandas will always return a ``DatetimeArray`` or ``TimedeltaArray``\n    rather than a ``PandasArray``. This is for symmetry with the case of\n    timezone-aware data, which NumPy does not natively support.\n\n    >>> pd.array([\'2015\', \'2016\'], dtype=\'datetime64[ns]\')\n    <DatetimeArray>\n    [\'2015-01-01 00:00:00\', \'2016-01-01 00:00:00\']\n    Length: 2, dtype: datetime64[ns]\n\n    >>> pd.array(["1H", "2H"], dtype=\'timedelta64[ns]\')\n    <TimedeltaArray>\n    [\'0 days 01:00:00\', \'0 days 02:00:00\']\n    Length: 2, dtype: timedelta64[ns]\n\n    Examples\n    --------\n    If a dtype is not specified, pandas will infer the best dtype from the values.\n    See the description of `dtype` for the types pandas infers for.\n\n    >>> pd.array([1, 2])\n    <IntegerArray>\n    [1, 2]\n    Length: 2, dtype: Int64\n\n    >>> pd.array([1, 2, np.nan])\n    <IntegerArray>\n    [1, 2, <NA>]\n    Length: 3, dtype: Int64\n\n    >>> pd.array([1.1, 2.2])\n    <FloatingArray>\n    [1.1, 2.2]\n    Length: 2, dtype: Float64\n\n    >>> pd.array(["a", None, "c"])\n    <StringArray>\n    [\'a\', <NA>, \'c\']\n    Length: 3, dtype: string\n\n    >>> pd.array([pd.Period(\'2000\', freq="D"), pd.Period("2000", freq="D")])\n    <PeriodArray>\n    [\'2000-01-01\', \'2000-01-01\']\n    Length: 2, dtype: period[D]\n\n    You can use the string alias for `dtype`\n\n    >>> pd.array([\'a\', \'b\', \'a\'], dtype=\'category\')\n    [\'a\', \'b\', \'a\']\n    Categories (2, object): [\'a\', \'b\']\n\n    Or specify the actual dtype\n\n    >>> pd.array([\'a\', \'b\', \'a\'],\n    ...          dtype=pd.CategoricalDtype([\'a\', \'b\', \'c\'], ordered=True))\n    [\'a\', \'b\', \'a\']\n    Categories (3, object): [\'a\' < \'b\' < \'c\']\n\n    If pandas does not infer a dedicated extension type a\n    :class:`arrays.PandasArray` is returned.\n\n    >>> pd.array([1 + 1j, 3 + 2j])\n    <PandasArray>\n    [(1+1j), (3+2j)]\n    Length: 2, dtype: complex128\n\n    As mentioned in the "Notes" section, new extension types may be added\n    in the future (by pandas or 3rd party libraries), causing the return\n    value to no longer be a :class:`arrays.PandasArray`. Specify the `dtype`\n    as a NumPy dtype if you need to ensure there\'s no future change in\n    behavior.\n\n    >>> pd.array([1, 2], dtype=np.dtype("int32"))\n    <PandasArray>\n    [1, 2]\n    Length: 2, dtype: int32\n\n    `data` must be 1-dimensional. A ValueError is raised when the input\n    has the wrong dimensionality.\n\n    >>> pd.array(1)\n    Traceback (most recent call last):\n      ...\n    ValueError: Cannot pass scalar \'1\' to \'pandas.array\'.\n    '
    from pandas.core.arrays import BooleanArray, DatetimeArray, FloatingArray, IntegerArray, IntervalArray, PandasArray, StringArray, TimedeltaArray, period_array
    if lib.is_scalar(data):
        msg = f"Cannot pass scalar '{data}' to 'pandas.array'."
        raise ValueError(msg)
    if ((dtype is None) and isinstance(data, (ABCSeries, ABCIndex, ABCExtensionArray))):
        dtype = data.dtype
    data = extract_array(data, extract_numpy=True)
    if isinstance(dtype, str):
        dtype = (registry.find(dtype) or dtype)
    if is_extension_array_dtype(dtype):
        cls = cast(ExtensionDtype, dtype).construct_array_type()
        return cls._from_sequence(data, dtype=dtype, copy=copy)
    if (dtype is None):
        inferred_dtype = lib.infer_dtype(data, skipna=True)
        if (inferred_dtype == 'period'):
            try:
                return period_array(data, copy=copy)
            except IncompatibleFrequency:
                pass
        elif (inferred_dtype == 'interval'):
            try:
                return IntervalArray(data, copy=copy)
            except ValueError:
                pass
        elif inferred_dtype.startswith('datetime'):
            try:
                return DatetimeArray._from_sequence(data, copy=copy)
            except ValueError:
                pass
        elif inferred_dtype.startswith('timedelta'):
            return TimedeltaArray._from_sequence(data, copy=copy)
        elif (inferred_dtype == 'string'):
            return StringArray._from_sequence(data, copy=copy)
        elif (inferred_dtype == 'integer'):
            return IntegerArray._from_sequence(data, copy=copy)
        elif (inferred_dtype in ('floating', 'mixed-integer-float')):
            return FloatingArray._from_sequence(data, copy=copy)
        elif (inferred_dtype == 'boolean'):
            return BooleanArray._from_sequence(data, copy=copy)
    if is_datetime64_ns_dtype(dtype):
        return DatetimeArray._from_sequence(data, dtype=dtype, copy=copy)
    elif is_timedelta64_ns_dtype(dtype):
        return TimedeltaArray._from_sequence(data, dtype=dtype, copy=copy)
    result = PandasArray._from_sequence(data, dtype=dtype, copy=copy)
    return result

def extract_array(obj, extract_numpy=False):
    "\n    Extract the ndarray or ExtensionArray from a Series or Index.\n\n    For all other types, `obj` is just returned as is.\n\n    Parameters\n    ----------\n    obj : object\n        For Series / Index, the underlying ExtensionArray is unboxed.\n        For Numpy-backed ExtensionArrays, the ndarray is extracted.\n\n    extract_numpy : bool, default False\n        Whether to extract the ndarray from a PandasArray\n\n    Returns\n    -------\n    arr : object\n\n    Examples\n    --------\n    >>> extract_array(pd.Series(['a', 'b', 'c'], dtype='category'))\n    ['a', 'b', 'c']\n    Categories (3, object): ['a', 'b', 'c']\n\n    Other objects like lists, arrays, and DataFrames are just passed through.\n\n    >>> extract_array([1, 2, 3])\n    [1, 2, 3]\n\n    For an ndarray-backed Series / Index a PandasArray is returned.\n\n    >>> extract_array(pd.Series([1, 2, 3]))\n    <PandasArray>\n    [1, 2, 3]\n    Length: 3, dtype: int64\n\n    To extract all the way down to the ndarray, pass ``extract_numpy=True``.\n\n    >>> extract_array(pd.Series([1, 2, 3]), extract_numpy=True)\n    array([1, 2, 3])\n    "
    if isinstance(obj, (ABCIndex, ABCSeries)):
        obj = obj.array
    if (extract_numpy and isinstance(obj, ABCPandasArray)):
        obj = obj.to_numpy()
    return obj

def ensure_wrapped_if_datetimelike(arr):
    '\n    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.\n    '
    if isinstance(arr, np.ndarray):
        if (arr.dtype.kind == 'M'):
            from pandas.core.arrays import DatetimeArray
            return DatetimeArray._from_sequence(arr)
        elif (arr.dtype.kind == 'm'):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._from_sequence(arr)
    return arr

def sanitize_masked_array(data):
    '\n    Convert numpy MaskedArray to ensure mask is softened.\n    '
    mask = ma.getmaskarray(data)
    if mask.any():
        (data, fill_value) = maybe_upcast(data, copy=True)
        data.soften_mask()
        data[mask] = fill_value
    else:
        data = data.copy()
    return data

def sanitize_array(data, index, dtype=None, copy=False, raise_cast_failure=False):
    '\n    Sanitize input data to an ndarray or ExtensionArray, copy if specified,\n    coerce to the dtype if specified.\n    '
    if isinstance(data, ma.MaskedArray):
        data = sanitize_masked_array(data)
    data = extract_array(data, extract_numpy=True)
    if (isinstance(data, np.ndarray) and (data.ndim == 0)):
        if (dtype is None):
            dtype = data.dtype
        data = lib.item_from_zerodim(data)
    if isinstance(data, np.ndarray):
        if ((dtype is not None) and is_float_dtype(data.dtype) and is_integer_dtype(dtype)):
            try:
                subarr = _try_cast(data, dtype, copy, True)
            except ValueError:
                subarr = np.array(data, copy=copy)
        else:
            subarr = _try_cast(data, dtype, copy, raise_cast_failure)
    elif isinstance(data, ABCExtensionArray):
        subarr = data
        if (dtype is not None):
            subarr = subarr.astype(dtype, copy=copy)
        elif copy:
            subarr = subarr.copy()
        return subarr
    elif (isinstance(data, (list, tuple, abc.Set, abc.ValuesView)) and (len(data) > 0)):
        if isinstance(data, set):
            raise TypeError('Set type is unordered')
        data = list(data)
        if (dtype is not None):
            subarr = _try_cast(data, dtype, copy, raise_cast_failure)
        else:
            subarr = maybe_convert_platform(data)
            subarr = maybe_cast_to_datetime(subarr, dtype)
    elif isinstance(data, range):
        arr = np.arange(data.start, data.stop, data.step, dtype='int64')
        subarr = _try_cast(arr, dtype, copy, raise_cast_failure)
    elif (not is_list_like(data)):
        if (index is None):
            raise ValueError('index must be specified when data is not list-like')
        subarr = construct_1d_arraylike_from_scalar(data, len(index), dtype)
    else:
        subarr = _try_cast(data, dtype, copy, raise_cast_failure)
    subarr = _sanitize_ndim(subarr, data, dtype, index)
    if (not (is_extension_array_dtype(subarr.dtype) or is_extension_array_dtype(dtype))):
        subarr = _sanitize_str_dtypes(subarr, data, dtype, copy)
        is_object_or_str_dtype = (is_object_dtype(dtype) or is_string_dtype(dtype))
        if (is_object_dtype(subarr.dtype) and (not is_object_or_str_dtype)):
            inferred = lib.infer_dtype(subarr, skipna=False)
            if (inferred in {'interval', 'period'}):
                subarr = array(subarr)
    return subarr

def _sanitize_ndim(result, data, dtype, index):
    '\n    Ensure we have a 1-dimensional result array.\n    '
    if (getattr(result, 'ndim', 0) == 0):
        raise ValueError('result should be arraylike with ndim > 0')
    elif (result.ndim == 1):
        result = _maybe_repeat(result, index)
    elif (result.ndim > 1):
        if isinstance(data, np.ndarray):
            raise ValueError('Data must be 1-dimensional')
        else:
            result = com.asarray_tuplesafe(data, dtype=dtype)
    return result

def _sanitize_str_dtypes(result, data, dtype, copy):
    '\n    Ensure we have a dtype that is supported by pandas.\n    '
    if issubclass(result.dtype.type, str):
        if (not lib.is_scalar(data)):
            if (not np.all(isna(data))):
                data = np.array(data, dtype=dtype, copy=False)
            result = np.array(data, dtype=object, copy=copy)
    return result

def _maybe_repeat(arr, index):
    '\n    If we have a length-1 array and an index describing how long we expect\n    the result to be, repeat the array.\n    '
    if (index is not None):
        if (1 == len(arr) != len(index)):
            arr = arr.repeat(len(index))
    return arr

def _try_cast(arr, dtype, copy, raise_cast_failure):
    "\n    Convert input to numpy ndarray and optionally cast to a given dtype.\n\n    Parameters\n    ----------\n    arr : ndarray, list, tuple, iterator (catchall)\n        Excludes: ExtensionArray, Series, Index.\n    dtype : np.dtype, ExtensionDtype or None\n    copy : bool\n        If False, don't copy the data if not needed.\n    raise_cast_failure : bool\n        If True, and if a dtype is specified, raise errors during casting.\n        Otherwise an object array is returned.\n    "
    if isinstance(arr, np.ndarray):
        if (maybe_castable(arr) and (not copy) and (dtype is None)):
            return arr
    if (isinstance(dtype, ExtensionDtype) and ((dtype.kind != 'M') or is_sparse(dtype))):
        array_type = dtype.construct_array_type()._from_sequence
        subarr = array_type(arr, dtype=dtype, copy=copy)
        return subarr
    if (is_object_dtype(dtype) and (not isinstance(arr, np.ndarray))):
        subarr = construct_1d_object_array_from_listlike(arr)
        return subarr
    try:
        if is_integer_dtype(dtype):
            maybe_cast_to_integer_array(arr, dtype)
            subarr = arr
        else:
            subarr = maybe_cast_to_datetime(arr, dtype)
        if (not isinstance(subarr, (ABCExtensionArray, ABCIndex))):
            subarr = construct_1d_ndarray_preserving_na(subarr, dtype, copy=copy)
    except OutOfBoundsDatetime:
        raise
    except (ValueError, TypeError) as err:
        if ((dtype is not None) and raise_cast_failure):
            raise
        elif ('Cannot cast' in str(err)):
            raise
        else:
            subarr = np.array(arr, dtype=object, copy=copy)
    return subarr

def is_empty_data(data):
    '\n    Utility to check if a Series is instantiated with empty data,\n    which does not contain dtype information.\n\n    Parameters\n    ----------\n    data : array-like, Iterable, dict, or scalar value\n        Contains data stored in Series.\n\n    Returns\n    -------\n    bool\n    '
    is_none = (data is None)
    is_list_like_without_dtype = (is_list_like(data) and (not hasattr(data, 'dtype')))
    is_simple_empty = (is_list_like_without_dtype and (not data))
    return (is_none or is_simple_empty)

def create_series_with_explicit_dtype(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False, dtype_if_empty=object):
    '\n    Helper to pass an explicit dtype when instantiating an empty Series.\n\n    This silences a DeprecationWarning described in GitHub-17261.\n\n    Parameters\n    ----------\n    data : Mirrored from Series.__init__\n    index : Mirrored from Series.__init__\n    dtype : Mirrored from Series.__init__\n    name : Mirrored from Series.__init__\n    copy : Mirrored from Series.__init__\n    fastpath : Mirrored from Series.__init__\n    dtype_if_empty : str, numpy.dtype, or ExtensionDtype\n        This dtype will be passed explicitly if an empty Series will\n        be instantiated.\n\n    Returns\n    -------\n    Series\n    '
    from pandas.core.series import Series
    if (is_empty_data(data) and (dtype is None)):
        dtype = dtype_if_empty
    return Series(data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)
