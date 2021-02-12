
'\nGeneric data algorithms. This module is experimental at the moment and not\nintended for public consumption\n'
from __future__ import annotations
import operator
from textwrap import dedent
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from pandas._libs import algos, hashtable as htable, iNaT, lib
from pandas._typing import AnyArrayLike, ArrayLike, DtypeObj, FrameOrSeriesUnion
from pandas.util._decorators import doc
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike, infer_dtype_from_array, maybe_promote
from pandas.core.dtypes.common import ensure_float64, ensure_int64, ensure_object, ensure_platform_int, ensure_uint64, is_array_like, is_bool_dtype, is_categorical_dtype, is_complex_dtype, is_datetime64_dtype, is_datetime64_ns_dtype, is_extension_array_dtype, is_float_dtype, is_integer, is_integer_dtype, is_interval_dtype, is_list_like, is_numeric_dtype, is_object_dtype, is_period_dtype, is_scalar, is_signed_integer_dtype, is_timedelta64_dtype, is_unsigned_integer_dtype, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.generic import ABCDatetimeArray, ABCExtensionArray, ABCIndex, ABCMultiIndex, ABCRangeIndex, ABCSeries, ABCTimedeltaArray
from pandas.core.dtypes.missing import isna, na_value_for_dtype
from pandas.core.construction import array, ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexers import validate_indices
if TYPE_CHECKING:
    from pandas import Categorical, DataFrame, Index, Series
    from pandas.core.arrays import DatetimeArray, IntervalArray, TimedeltaArray
_shared_docs = {}

def _ensure_data(values, dtype=None):
    '\n    routine to ensure that our data is of the correct\n    input dtype for lower-level routines\n\n    This will coerce:\n    - ints -> int64\n    - uint -> uint64\n    - bool -> uint64 (TODO this should be uint8)\n    - datetimelike -> i8\n    - datetime64tz -> i8 (in local tz)\n    - categorical -> codes\n\n    Parameters\n    ----------\n    values : array-like\n    dtype : pandas_dtype, optional\n        coerce to this dtype\n\n    Returns\n    -------\n    values : ndarray\n    pandas_dtype : np.dtype or ExtensionDtype\n    '
    if (dtype is not None):
        assert (not needs_i8_conversion(dtype))
        assert (not is_categorical_dtype(dtype))
    if (not isinstance(values, ABCMultiIndex)):
        values = extract_array(values, extract_numpy=True)
    if is_object_dtype(dtype):
        return (ensure_object(np.asarray(values)), np.dtype('object'))
    elif (is_object_dtype(values) and (dtype is None)):
        return (ensure_object(np.asarray(values)), np.dtype('object'))
    try:
        if (is_bool_dtype(values) or is_bool_dtype(dtype)):
            return (np.asarray(values).astype('uint64'), np.dtype('bool'))
        elif (is_signed_integer_dtype(values) or is_signed_integer_dtype(dtype)):
            return (ensure_int64(values), np.dtype('int64'))
        elif (is_unsigned_integer_dtype(values) or is_unsigned_integer_dtype(dtype)):
            return (ensure_uint64(values), np.dtype('uint64'))
        elif (is_float_dtype(values) or is_float_dtype(dtype)):
            return (ensure_float64(values), np.dtype('float64'))
        elif (is_complex_dtype(values) or is_complex_dtype(dtype)):
            with catch_warnings():
                simplefilter('ignore', np.ComplexWarning)
                values = ensure_float64(values)
            return (values, np.dtype('float64'))
    except (TypeError, ValueError, OverflowError):
        return (ensure_object(values), np.dtype('object'))
    if (needs_i8_conversion(values.dtype) or needs_i8_conversion(dtype)):
        if (is_period_dtype(values.dtype) or is_period_dtype(dtype)):
            from pandas import PeriodIndex
            values = PeriodIndex(values)._data
            dtype = values.dtype
        elif (is_timedelta64_dtype(values.dtype) or is_timedelta64_dtype(dtype)):
            from pandas import TimedeltaIndex
            values = TimedeltaIndex(values)._data
            dtype = values.dtype
        else:
            if ((values.ndim > 1) and is_datetime64_ns_dtype(values.dtype)):
                asi8 = values.view('i8')
                dtype = values.dtype
                return (asi8, dtype)
            from pandas import DatetimeIndex
            values = DatetimeIndex(values)._data
            dtype = values.dtype
        return (values.asi8, dtype)
    elif (is_categorical_dtype(values.dtype) and (is_categorical_dtype(dtype) or (dtype is None))):
        values = cast('Categorical', values)
        values = values.codes
        dtype = pandas_dtype('category')
        values = ensure_int64(values)
        return (values, dtype)
    values = np.asarray(values, dtype=object)
    return (ensure_object(values), np.dtype('object'))

def _reconstruct_data(values, dtype, original):
    '\n    reverse of _ensure_data\n\n    Parameters\n    ----------\n    values : np.ndarray or ExtensionArray\n    dtype : np.ndtype or ExtensionDtype\n    original : AnyArrayLike\n\n    Returns\n    -------\n    ExtensionArray or np.ndarray\n    '
    if (isinstance(values, ABCExtensionArray) and (values.dtype == dtype)):
        return values
    if is_extension_array_dtype(dtype):
        cls = dtype.construct_array_type()
        if (isinstance(values, cls) and (values.dtype == dtype)):
            return values
        values = cls._from_sequence(values)
    elif is_bool_dtype(dtype):
        values = values.astype(dtype, copy=False)
        if isinstance(original, ABCIndex):
            values = values.astype(object, copy=False)
    elif (dtype is not None):
        if is_datetime64_dtype(dtype):
            dtype = 'datetime64[ns]'
        elif is_timedelta64_dtype(dtype):
            dtype = 'timedelta64[ns]'
        values = values.astype(dtype, copy=False)
    return values

def _ensure_arraylike(values):
    '\n    ensure that we are arraylike if not already\n    '
    if (not is_array_like(values)):
        inferred = lib.infer_dtype(values, skipna=False)
        if (inferred in ['mixed', 'string', 'mixed-integer']):
            if isinstance(values, tuple):
                values = list(values)
            values = construct_1d_object_array_from_listlike(values)
        else:
            values = np.asarray(values)
    return values
_hashtables = {'float64': htable.Float64HashTable, 'uint64': htable.UInt64HashTable, 'int64': htable.Int64HashTable, 'string': htable.StringHashTable, 'object': htable.PyObjectHashTable}

def _get_hashtable_algo(values):
    '\n    Parameters\n    ----------\n    values : np.ndarray\n\n    Returns\n    -------\n    htable : HashTable subclass\n    values : ndarray\n    '
    (values, _) = _ensure_data(values)
    ndtype = _check_object_for_strings(values)
    htable = _hashtables[ndtype]
    return (htable, values)

def _get_values_for_rank(values):
    if is_categorical_dtype(values):
        values = cast('Categorical', values)._values_for_rank()
    (values, _) = _ensure_data(values)
    return values

def get_data_algo(values):
    values = _get_values_for_rank(values)
    ndtype = _check_object_for_strings(values)
    htable = _hashtables.get(ndtype, _hashtables['object'])
    return (htable, values)

def _check_object_for_strings(values):
    '\n    Check if we can use string hashtable instead of object hashtable.\n\n    Parameters\n    ----------\n    values : ndarray\n\n    Returns\n    -------\n    str\n    '
    ndtype = values.dtype.name
    if (ndtype == 'object'):
        if (lib.infer_dtype(values, skipna=False) in ['string']):
            ndtype = 'string'
    return ndtype

def unique(values):
    "\n    Hash table-based unique. Uniques are returned in order\n    of appearance. This does NOT sort.\n\n    Significantly faster than numpy.unique. Includes NA values.\n\n    Parameters\n    ----------\n    values : 1d array-like\n\n    Returns\n    -------\n    numpy.ndarray or ExtensionArray\n\n        The return can be:\n\n        * Index : when the input is an Index\n        * Categorical : when the input is a Categorical dtype\n        * ndarray : when the input is a Series/ndarray\n\n        Return numpy.ndarray or ExtensionArray.\n\n    See Also\n    --------\n    Index.unique : Return unique values from an Index.\n    Series.unique : Return unique values of Series object.\n\n    Examples\n    --------\n    >>> pd.unique(pd.Series([2, 1, 3, 3]))\n    array([2, 1, 3])\n\n    >>> pd.unique(pd.Series([2] + [1] * 5))\n    array([2, 1])\n\n    >>> pd.unique(pd.Series([pd.Timestamp('20160101'),\n    ...                     pd.Timestamp('20160101')]))\n    array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')\n\n    >>> pd.unique(pd.Series([pd.Timestamp('20160101', tz='US/Eastern'),\n    ...                      pd.Timestamp('20160101', tz='US/Eastern')]))\n    array([Timestamp('2016-01-01 00:00:00-0500', tz='US/Eastern')],\n          dtype=object)\n\n    >>> pd.unique(pd.Index([pd.Timestamp('20160101', tz='US/Eastern'),\n    ...                     pd.Timestamp('20160101', tz='US/Eastern')]))\n    DatetimeIndex(['2016-01-01 00:00:00-05:00'],\n    ...           dtype='datetime64[ns, US/Eastern]', freq=None)\n\n    >>> pd.unique(list('baabc'))\n    array(['b', 'a', 'c'], dtype=object)\n\n    An unordered Categorical will return categories in the\n    order of appearance.\n\n    >>> pd.unique(pd.Series(pd.Categorical(list('baabc'))))\n    [b, a, c]\n    Categories (3, object): [b, a, c]\n\n    >>> pd.unique(pd.Series(pd.Categorical(list('baabc'),\n    ...                                    categories=list('abc'))))\n    [b, a, c]\n    Categories (3, object): [b, a, c]\n\n    An ordered Categorical preserves the category ordering.\n\n    >>> pd.unique(pd.Series(pd.Categorical(list('baabc'),\n    ...                                    categories=list('abc'),\n    ...                                    ordered=True)))\n    [b, a, c]\n    Categories (3, object): [a < b < c]\n\n    An array of tuples\n\n    >>> pd.unique([('a', 'b'), ('b', 'a'), ('a', 'c'), ('b', 'a')])\n    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)\n    "
    values = _ensure_arraylike(values)
    if is_extension_array_dtype(values):
        return values.unique()
    original = values
    (htable, values) = _get_hashtable_algo(values)
    table = htable(len(values))
    uniques = table.unique(values)
    uniques = _reconstruct_data(uniques, original.dtype, original)
    return uniques
unique1d = unique

def isin(comps, values):
    '\n    Compute the isin boolean array.\n\n    Parameters\n    ----------\n    comps : array-like\n    values : array-like\n\n    Returns\n    -------\n    ndarray[bool]\n        Same length as `comps`.\n    '
    if (not is_list_like(comps)):
        raise TypeError(f'only list-like objects are allowed to be passed to isin(), you passed a [{type(comps).__name__}]')
    if (not is_list_like(values)):
        raise TypeError(f'only list-like objects are allowed to be passed to isin(), you passed a [{type(values).__name__}]')
    if (not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray))):
        values = _ensure_arraylike(list(values))
    elif isinstance(values, ABCMultiIndex):
        values = np.array(values)
    else:
        values = extract_array(values, extract_numpy=True)
    comps = _ensure_arraylike(comps)
    comps = extract_array(comps, extract_numpy=True)
    if is_categorical_dtype(comps.dtype):
        return cast('Categorical', comps).isin(values)
    elif is_interval_dtype(comps.dtype):
        return cast('IntervalArray', comps).isin(values)
    elif needs_i8_conversion(comps.dtype):
        return array(comps).isin(values)
    elif (needs_i8_conversion(values.dtype) and (not is_object_dtype(comps.dtype))):
        return np.zeros(comps.shape, dtype=bool)
    elif needs_i8_conversion(values.dtype):
        return isin(comps, values.astype(object))
    elif (is_extension_array_dtype(comps.dtype) or is_extension_array_dtype(values.dtype)):
        return isin(np.asarray(comps), np.asarray(values))
    if ((len(comps) > 1000000) and (len(values) <= 26) and (not is_object_dtype(comps))):
        if isna(values).any():
            f = (lambda c, v: np.logical_or(np.in1d(c, v), np.isnan(c)))
        else:
            f = np.in1d
    else:
        common = np.find_common_type([values.dtype, comps.dtype], [])
        values = values.astype(common, copy=False)
        comps = comps.astype(common, copy=False)
        name = common.name
        if (name == 'bool'):
            name = 'uint8'
        f = getattr(htable, f'ismember_{name}')
    return f(comps, values)

def factorize_array(values, na_sentinel=(- 1), size_hint=None, na_value=None, mask=None):
    '\n    Factorize an array-like to codes and uniques.\n\n    This doesn\'t do any coercion of types or unboxing before factorization.\n\n    Parameters\n    ----------\n    values : ndarray\n    na_sentinel : int, default -1\n    size_hint : int, optional\n        Passed through to the hashtable\'s \'get_labels\' method\n    na_value : object, optional\n        A value in `values` to consider missing. Note: only use this\n        parameter when you know that you don\'t have any values pandas would\n        consider missing in the array (NaN for float data, iNaT for\n        datetimes, etc.).\n    mask : ndarray[bool], optional\n        If not None, the mask is used as indicator for missing values\n        (True = missing, False = valid) instead of `na_value` or\n        condition "val != val".\n\n    Returns\n    -------\n    codes : ndarray\n    uniques : ndarray\n    '
    (hash_klass, values) = get_data_algo(values)
    table = hash_klass((size_hint or len(values)))
    (uniques, codes) = table.factorize(values, na_sentinel=na_sentinel, na_value=na_value, mask=mask)
    codes = ensure_platform_int(codes)
    return (codes, uniques)

@doc(values=dedent("    values : sequence\n        A 1-D sequence. Sequences that aren't pandas objects are\n        coerced to ndarrays before factorization.\n    "), sort=dedent('    sort : bool, default False\n        Sort `uniques` and shuffle `codes` to maintain the\n        relationship.\n    '), size_hint=dedent('    size_hint : int, optional\n        Hint to the hashtable sizer.\n    '))
def factorize(values, sort=False, na_sentinel=(- 1), size_hint=None):
    '\n    Encode the object as an enumerated type or categorical variable.\n\n    This method is useful for obtaining a numeric representation of an\n    array when all that matters is identifying distinct values. `factorize`\n    is available as both a top-level function :func:`pandas.factorize`,\n    and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.\n\n    Parameters\n    ----------\n    {values}{sort}\n    na_sentinel : int or None, default -1\n        Value to mark "not found". If None, will not drop the NaN\n        from the uniques of the values.\n\n        .. versionchanged:: 1.1.2\n    {size_hint}\n    Returns\n    -------\n    codes : ndarray\n        An integer ndarray that\'s an indexer into `uniques`.\n        ``uniques.take(codes)`` will have the same values as `values`.\n    uniques : ndarray, Index, or Categorical\n        The unique valid values. When `values` is Categorical, `uniques`\n        is a Categorical. When `values` is some other pandas object, an\n        `Index` is returned. Otherwise, a 1-D ndarray is returned.\n\n        .. note ::\n\n           Even if there\'s a missing value in `values`, `uniques` will\n           *not* contain an entry for it.\n\n    See Also\n    --------\n    cut : Discretize continuous-valued array.\n    unique : Find the unique value in an array.\n\n    Examples\n    --------\n    These examples all show factorize as a top-level method like\n    ``pd.factorize(values)``. The results are identical for methods like\n    :meth:`Series.factorize`.\n\n    >>> codes, uniques = pd.factorize([\'b\', \'b\', \'a\', \'c\', \'b\'])\n    >>> codes\n    array([0, 0, 1, 2, 0]...)\n    >>> uniques\n    array([\'b\', \'a\', \'c\'], dtype=object)\n\n    With ``sort=True``, the `uniques` will be sorted, and `codes` will be\n    shuffled so that the relationship is the maintained.\n\n    >>> codes, uniques = pd.factorize([\'b\', \'b\', \'a\', \'c\', \'b\'], sort=True)\n    >>> codes\n    array([1, 1, 0, 2, 1]...)\n    >>> uniques\n    array([\'a\', \'b\', \'c\'], dtype=object)\n\n    Missing values are indicated in `codes` with `na_sentinel`\n    (``-1`` by default). Note that missing values are never\n    included in `uniques`.\n\n    >>> codes, uniques = pd.factorize([\'b\', None, \'a\', \'c\', \'b\'])\n    >>> codes\n    array([ 0, -1,  1,  2,  0]...)\n    >>> uniques\n    array([\'b\', \'a\', \'c\'], dtype=object)\n\n    Thus far, we\'ve only factorized lists (which are internally coerced to\n    NumPy arrays). When factorizing pandas objects, the type of `uniques`\n    will differ. For Categoricals, a `Categorical` is returned.\n\n    >>> cat = pd.Categorical([\'a\', \'a\', \'c\'], categories=[\'a\', \'b\', \'c\'])\n    >>> codes, uniques = pd.factorize(cat)\n    >>> codes\n    array([0, 0, 1]...)\n    >>> uniques\n    [\'a\', \'c\']\n    Categories (3, object): [\'a\', \'b\', \'c\']\n\n    Notice that ``\'b\'`` is in ``uniques.categories``, despite not being\n    present in ``cat.values``.\n\n    For all other pandas objects, an Index of the appropriate type is\n    returned.\n\n    >>> cat = pd.Series([\'a\', \'a\', \'c\'])\n    >>> codes, uniques = pd.factorize(cat)\n    >>> codes\n    array([0, 0, 1]...)\n    >>> uniques\n    Index([\'a\', \'c\'], dtype=\'object\')\n\n    If NaN is in the values, and we want to include NaN in the uniques of the\n    values, it can be achieved by setting ``na_sentinel=None``.\n\n    >>> values = np.array([1, 2, 1, np.nan])\n    >>> codes, uniques = pd.factorize(values)  # default: na_sentinel=-1\n    >>> codes\n    array([ 0,  1,  0, -1])\n    >>> uniques\n    array([1., 2.])\n\n    >>> codes, uniques = pd.factorize(values, na_sentinel=None)\n    >>> codes\n    array([0, 1, 0, 2])\n    >>> uniques\n    array([ 1.,  2., nan])\n    '
    if isinstance(values, ABCRangeIndex):
        return values.factorize(sort=sort)
    values = _ensure_arraylike(values)
    original = values
    if (not isinstance(values, ABCMultiIndex)):
        values = extract_array(values, extract_numpy=True)
    dropna = True
    if (na_sentinel is None):
        na_sentinel = (- 1)
        dropna = False
    if (isinstance(values, (ABCDatetimeArray, ABCTimedeltaArray)) and (values.freq is not None)):
        (codes, uniques) = values.factorize(sort=sort)
        if isinstance(original, ABCIndex):
            uniques = original._shallow_copy(uniques, name=None)
        elif isinstance(original, ABCSeries):
            from pandas import Index
            uniques = Index(uniques)
        return (codes, uniques)
    if is_extension_array_dtype(values.dtype):
        (codes, uniques) = values.factorize(na_sentinel=na_sentinel)
        dtype = original.dtype
    else:
        (values, dtype) = _ensure_data(values)
        if (original.dtype.kind in ['m', 'M']):
            na_value = na_value_for_dtype(original.dtype)
        else:
            na_value = None
        (codes, uniques) = factorize_array(values, na_sentinel=na_sentinel, size_hint=size_hint, na_value=na_value)
    if (sort and (len(uniques) > 0)):
        (uniques, codes) = safe_sort(uniques, codes, na_sentinel=na_sentinel, assume_unique=True, verify=False)
    code_is_na = (codes == na_sentinel)
    if ((not dropna) and code_is_na.any()):
        na_value = na_value_for_dtype(uniques.dtype, compat=False)
        uniques = np.append(uniques, [na_value])
        codes = np.where(code_is_na, (len(uniques) - 1), codes)
    uniques = _reconstruct_data(uniques, dtype, original)
    if isinstance(original, ABCIndex):
        if ((original.dtype.kind in ['m', 'M']) and isinstance(uniques, np.ndarray)):
            original._data = cast('Union[DatetimeArray, TimedeltaArray]', original._data)
            uniques = type(original._data)._simple_new(uniques, dtype=original.dtype)
        uniques = original._shallow_copy(uniques, name=None)
    elif isinstance(original, ABCSeries):
        from pandas import Index
        uniques = Index(uniques)
    return (codes, uniques)

def value_counts(values, sort=True, ascending=False, normalize=False, bins=None, dropna=True):
    "\n    Compute a histogram of the counts of non-null values.\n\n    Parameters\n    ----------\n    values : ndarray (1-d)\n    sort : bool, default True\n        Sort by values\n    ascending : bool, default False\n        Sort in ascending order\n    normalize: bool, default False\n        If True then compute a relative histogram\n    bins : integer, optional\n        Rather than count values, group them into half-open bins,\n        convenience for pd.cut, only works with numeric data\n    dropna : bool, default True\n        Don't include counts of NaN\n\n    Returns\n    -------\n    Series\n    "
    from pandas.core.series import Series
    name = getattr(values, 'name', None)
    if (bins is not None):
        from pandas.core.reshape.tile import cut
        values = Series(values)
        try:
            ii = cut(values, bins, include_lowest=True)
        except TypeError as err:
            raise TypeError('bins argument only works with numeric data.') from err
        result = ii.value_counts(dropna=dropna)
        result = result[result.index.notna()]
        result.index = result.index.astype('interval')
        result = result.sort_index()
        if (dropna and (result._values == 0).all()):
            result = result.iloc[0:0]
        counts = np.array([len(ii)])
    elif is_extension_array_dtype(values):
        result = Series(values)._values.value_counts(dropna=dropna)
        result.name = name
        counts = result._values
    else:
        (keys, counts) = value_counts_arraylike(values, dropna)
        result = Series(counts, index=keys, name=name)
    if sort:
        result = result.sort_values(ascending=ascending)
    if normalize:
        result = (result / float(counts.sum()))
    return result

def value_counts_arraylike(values, dropna):
    '\n    Parameters\n    ----------\n    values : arraylike\n    dropna : bool\n\n    Returns\n    -------\n    uniques : np.ndarray or ExtensionArray\n    counts : np.ndarray\n    '
    values = _ensure_arraylike(values)
    original = values
    (values, _) = _ensure_data(values)
    ndtype = values.dtype.name
    if needs_i8_conversion(original.dtype):
        (keys, counts) = htable.value_count_int64(values, dropna)
        if dropna:
            msk = (keys != iNaT)
            (keys, counts) = (keys[msk], counts[msk])
    else:
        f = getattr(htable, f'value_count_{ndtype}')
        (keys, counts) = f(values, dropna)
        mask = isna(values)
        if ((not dropna) and mask.any()):
            if (not isna(keys).any()):
                keys = np.insert(keys, 0, np.NaN)
                counts = np.insert(counts, 0, mask.sum())
    keys = _reconstruct_data(keys, original.dtype, original)
    return (keys, counts)

def duplicated(values, keep='first'):
    "\n    Return boolean ndarray denoting duplicate values.\n\n    Parameters\n    ----------\n    values : ndarray-like\n        Array over which to check for duplicate values.\n    keep : {'first', 'last', False}, default 'first'\n        - ``first`` : Mark duplicates as ``True`` except for the first\n          occurrence.\n        - ``last`` : Mark duplicates as ``True`` except for the last\n          occurrence.\n        - False : Mark all duplicates as ``True``.\n\n    Returns\n    -------\n    duplicated : ndarray\n    "
    (values, _) = _ensure_data(values)
    ndtype = values.dtype.name
    f = getattr(htable, f'duplicated_{ndtype}')
    return f(values, keep=keep)

def mode(values, dropna=True):
    "\n    Returns the mode(s) of an array.\n\n    Parameters\n    ----------\n    values : array-like\n        Array over which to check for duplicate values.\n    dropna : boolean, default True\n        Don't consider counts of NaN/NaT.\n\n        .. versionadded:: 0.24.0\n\n    Returns\n    -------\n    mode : Series\n    "
    from pandas import Series
    import pandas.core.indexes.base as ibase
    values = _ensure_arraylike(values)
    original = values
    if is_categorical_dtype(values):
        if isinstance(values, Series):
            return Series(values._values.mode(dropna=dropna), name=values.name)
        return values.mode(dropna=dropna)
    if (dropna and needs_i8_conversion(values.dtype)):
        mask = values.isnull()
        values = values[(~ mask)]
    (values, _) = _ensure_data(values)
    ndtype = values.dtype.name
    f = getattr(htable, f'mode_{ndtype}')
    result = f(values, dropna=dropna)
    try:
        result = np.sort(result)
    except TypeError as err:
        warn(f'Unable to sort modes: {err}')
    result = _reconstruct_data(result, original.dtype, original)
    return Series(result, index=ibase.default_index(len(result)))

def rank(values, axis=0, method='average', na_option='keep', ascending=True, pct=False):
    "\n    Rank the values along a given axis.\n\n    Parameters\n    ----------\n    values : array-like\n        Array whose values will be ranked. The number of dimensions in this\n        array must not exceed 2.\n    axis : int, default 0\n        Axis over which to perform rankings.\n    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'\n        The method by which tiebreaks are broken during the ranking.\n    na_option : {'keep', 'top'}, default 'keep'\n        The method by which NaNs are placed in the ranking.\n        - ``keep``: rank each NaN value with a NaN ranking\n        - ``top``: replace each NaN with either +/- inf so that they\n                   there are ranked at the top\n    ascending : boolean, default True\n        Whether or not the elements should be ranked in ascending order.\n    pct : boolean, default False\n        Whether or not to the display the returned rankings in integer form\n        (e.g. 1, 2, 3) or in percentile form (e.g. 0.333..., 0.666..., 1).\n    "
    if (values.ndim == 1):
        values = _get_values_for_rank(values)
        ranks = algos.rank_1d(values, labels=np.zeros(len(values), dtype=np.int64), ties_method=method, ascending=ascending, na_option=na_option, pct=pct)
    elif (values.ndim == 2):
        values = _get_values_for_rank(values)
        ranks = algos.rank_2d(values, axis=axis, ties_method=method, ascending=ascending, na_option=na_option, pct=pct)
    else:
        raise TypeError('Array with ndim > 2 are not supported.')
    return ranks

def checked_add_with_arr(arr, b, arr_mask=None, b_mask=None):
    '\n    Perform array addition that checks for underflow and overflow.\n\n    Performs the addition of an int64 array and an int64 integer (or array)\n    but checks that they do not result in overflow first. For elements that\n    are indicated to be NaN, whether or not there is overflow for that element\n    is automatically ignored.\n\n    Parameters\n    ----------\n    arr : array addend.\n    b : array or scalar addend.\n    arr_mask : boolean array or None\n        array indicating which elements to exclude from checking\n    b_mask : boolean array or boolean or None\n        array or scalar indicating which element(s) to exclude from checking\n\n    Returns\n    -------\n    sum : An array for elements x + b for each element x in arr if b is\n          a scalar or an array for elements x + y for each element pair\n          (x, y) in (arr, b).\n\n    Raises\n    ------\n    OverflowError if any x + y exceeds the maximum or minimum int64 value.\n    '
    b2 = np.broadcast_to(b, arr.shape)
    if (b_mask is not None):
        b2_mask = np.broadcast_to(b_mask, arr.shape)
    else:
        b2_mask = None
    if ((arr_mask is not None) and (b2_mask is not None)):
        not_nan = np.logical_not((arr_mask | b2_mask))
    elif (arr_mask is not None):
        not_nan = np.logical_not(arr_mask)
    elif (b_mask is not None):
        not_nan = np.logical_not(b2_mask)
    else:
        not_nan = np.empty(arr.shape, dtype=bool)
        not_nan.fill(True)
    mask1 = (b2 > 0)
    mask2 = (b2 < 0)
    if (not mask1.any()):
        to_raise = (((np.iinfo(np.int64).min - b2) > arr) & not_nan).any()
    elif (not mask2.any()):
        to_raise = (((np.iinfo(np.int64).max - b2) < arr) & not_nan).any()
    else:
        to_raise = ((((np.iinfo(np.int64).max - b2[mask1]) < arr[mask1]) & not_nan[mask1]).any() or (((np.iinfo(np.int64).min - b2[mask2]) > arr[mask2]) & not_nan[mask2]).any())
    if to_raise:
        raise OverflowError('Overflow in int64 addition')
    return (arr + b)

def quantile(x, q, interpolation_method='fraction'):
    "\n    Compute sample quantile or quantiles of the input array. For example, q=0.5\n    computes the median.\n\n    The `interpolation_method` parameter supports three values, namely\n    `fraction` (default), `lower` and `higher`. Interpolation is done only,\n    if the desired quantile lies between two data points `i` and `j`. For\n    `fraction`, the result is an interpolated value between `i` and `j`;\n    for `lower`, the result is `i`, for `higher` the result is `j`.\n\n    Parameters\n    ----------\n    x : ndarray\n        Values from which to extract score.\n    q : scalar or array\n        Percentile at which to extract score.\n    interpolation_method : {'fraction', 'lower', 'higher'}, optional\n        This optional parameter specifies the interpolation method to use,\n        when the desired quantile lies between two data points `i` and `j`:\n\n        - fraction: `i + (j - i)*fraction`, where `fraction` is the\n                    fractional part of the index surrounded by `i` and `j`.\n        -lower: `i`.\n        - higher: `j`.\n\n    Returns\n    -------\n    score : float\n        Score at percentile.\n\n    Examples\n    --------\n    >>> from scipy import stats\n    >>> a = np.arange(100)\n    >>> stats.scoreatpercentile(a, 50)\n    49.5\n\n    "
    x = np.asarray(x)
    mask = isna(x)
    x = x[(~ mask)]
    values = np.sort(x)

    def _interpolate(a, b, fraction):
        "\n        Returns the point at the given fraction between a and b, where\n        'fraction' must be between 0 and 1.\n        "
        return (a + ((b - a) * fraction))

    def _get_score(at):
        if (len(values) == 0):
            return np.nan
        idx = (at * (len(values) - 1))
        if ((idx % 1) == 0):
            score = values[int(idx)]
        elif (interpolation_method == 'fraction'):
            score = _interpolate(values[int(idx)], values[(int(idx) + 1)], (idx % 1))
        elif (interpolation_method == 'lower'):
            score = values[np.floor(idx)]
        elif (interpolation_method == 'higher'):
            score = values[np.ceil(idx)]
        else:
            raise ValueError("interpolation_method can only be 'fraction' , 'lower' or 'higher'")
        return score
    if is_scalar(q):
        return _get_score(q)
    else:
        q = np.asarray(q, np.float64)
        result = [_get_score(x) for x in q]
        result = np.array(result, dtype=np.float64)
        return result

class SelectN():

    def __init__(self, obj, n, keep):
        self.obj = obj
        self.n = n
        self.keep = keep
        if (self.keep not in ('first', 'last', 'all')):
            raise ValueError('keep must be either "first", "last" or "all"')

    def compute(self, method):
        raise NotImplementedError

    def nlargest(self):
        return self.compute('nlargest')

    def nsmallest(self):
        return self.compute('nsmallest')

    @staticmethod
    def is_valid_dtype_n_method(dtype):
        '\n        Helper function to determine if dtype is valid for\n        nsmallest/nlargest methods\n        '
        return ((is_numeric_dtype(dtype) and (not is_complex_dtype(dtype))) or needs_i8_conversion(dtype))

class SelectNSeries(SelectN):
    "\n    Implement n largest/smallest for Series\n\n    Parameters\n    ----------\n    obj : Series\n    n : int\n    keep : {'first', 'last'}, default 'first'\n\n    Returns\n    -------\n    nordered : Series\n    "

    def compute(self, method):
        n = self.n
        dtype = self.obj.dtype
        if (not self.is_valid_dtype_n_method(dtype)):
            raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")
        if (n <= 0):
            return self.obj[[]]
        dropped = self.obj.dropna()
        if (n >= len(self.obj)):
            ascending = (method == 'nsmallest')
            return dropped.sort_values(ascending=ascending).head(n)
        (arr, pandas_dtype) = _ensure_data(dropped.values)
        if (method == 'nlargest'):
            arr = (- arr)
            if is_integer_dtype(pandas_dtype):
                arr -= 1
            elif is_bool_dtype(pandas_dtype):
                arr = (1 - (- arr))
        if (self.keep == 'last'):
            arr = arr[::(- 1)]
        narr = len(arr)
        n = min(n, narr)
        kth_val = algos.kth_smallest(arr.copy(), (n - 1))
        (ns,) = np.nonzero((arr <= kth_val))
        inds = ns[arr[ns].argsort(kind='mergesort')]
        if (self.keep != 'all'):
            inds = inds[:n]
        if (self.keep == 'last'):
            inds = ((narr - 1) - inds)
        return dropped.iloc[inds]

class SelectNFrame(SelectN):
    "\n    Implement n largest/smallest for DataFrame\n\n    Parameters\n    ----------\n    obj : DataFrame\n    n : int\n    keep : {'first', 'last'}, default 'first'\n    columns : list or str\n\n    Returns\n    -------\n    nordered : DataFrame\n    "

    def __init__(self, obj, n, keep, columns):
        super().__init__(obj, n, keep)
        if ((not is_list_like(columns)) or isinstance(columns, tuple)):
            columns = [columns]
        columns = list(columns)
        self.columns = columns

    def compute(self, method):
        from pandas import Int64Index
        n = self.n
        frame = self.obj
        columns = self.columns
        for column in columns:
            dtype = frame[column].dtype
            if (not self.is_valid_dtype_n_method(dtype)):
                raise TypeError(f'Column {repr(column)} has dtype {dtype}, cannot use method {repr(method)} with this dtype')

        def get_indexer(current_indexer, other_indexer):
            '\n            Helper function to concat `current_indexer` and `other_indexer`\n            depending on `method`\n            '
            if (method == 'nsmallest'):
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer = Int64Index([])
        for (i, column) in enumerate(columns):
            series = cur_frame[column]
            is_last_column = ((len(columns) - 1) == i)
            values = getattr(series, method)(cur_n, keep=(self.keep if is_last_column else 'all'))
            if (is_last_column or (len(values) <= cur_n)):
                indexer = get_indexer(indexer, values.index)
                break
            border_value = (values == values[values.index[(- 1)]])
            unsafe_values = values[border_value]
            safe_values = values[(~ border_value)]
            indexer = get_indexer(indexer, safe_values.index)
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = (n - len(indexer))
        frame = frame.take(indexer)
        frame.index = original_index.take(indexer)
        if (len(columns) == 1):
            return frame
        ascending = (method == 'nsmallest')
        return frame.sort_values(columns, ascending=ascending, kind='mergesort')

def _view_wrapper(f, arr_dtype=None, out_dtype=None, fill_wrap=None):

    def wrapper(arr, indexer, out, fill_value=np.nan):
        if (arr_dtype is not None):
            arr = arr.view(arr_dtype)
        if (out_dtype is not None):
            out = out.view(out_dtype)
        if (fill_wrap is not None):
            fill_value = fill_wrap(fill_value)
        f(arr, indexer, out, fill_value=fill_value)
    return wrapper

def _convert_wrapper(f, conv_dtype):

    def wrapper(arr, indexer, out, fill_value=np.nan):
        arr = arr.astype(conv_dtype)
        f(arr, indexer, out, fill_value=fill_value)
    return wrapper

def _take_2d_multi_object(arr, indexer, out, fill_value, mask_info):
    (row_idx, col_idx) = indexer
    if (mask_info is not None):
        ((row_mask, col_mask), (row_needs, col_needs)) = mask_info
    else:
        row_mask = (row_idx == (- 1))
        col_mask = (col_idx == (- 1))
        row_needs = row_mask.any()
        col_needs = col_mask.any()
    if (fill_value is not None):
        if row_needs:
            out[row_mask, :] = fill_value
        if col_needs:
            out[:, col_mask] = fill_value
    for i in range(len(row_idx)):
        u_ = row_idx[i]
        for j in range(len(col_idx)):
            v = col_idx[j]
            out[(i, j)] = arr[(u_, v)]

def _take_nd_object(arr, indexer, out, axis, fill_value, mask_info):
    if (mask_info is not None):
        (mask, needs_masking) = mask_info
    else:
        mask = (indexer == (- 1))
        needs_masking = mask.any()
    if (arr.dtype != out.dtype):
        arr = arr.astype(out.dtype)
    if (arr.shape[axis] > 0):
        arr.take(ensure_platform_int(indexer), axis=axis, out=out)
    if needs_masking:
        outindexer = ([slice(None)] * arr.ndim)
        outindexer[axis] = mask
        out[tuple(outindexer)] = fill_value
_take_1d_dict = {('int8', 'int8'): algos.take_1d_int8_int8, ('int8', 'int32'): algos.take_1d_int8_int32, ('int8', 'int64'): algos.take_1d_int8_int64, ('int8', 'float64'): algos.take_1d_int8_float64, ('int16', 'int16'): algos.take_1d_int16_int16, ('int16', 'int32'): algos.take_1d_int16_int32, ('int16', 'int64'): algos.take_1d_int16_int64, ('int16', 'float64'): algos.take_1d_int16_float64, ('int32', 'int32'): algos.take_1d_int32_int32, ('int32', 'int64'): algos.take_1d_int32_int64, ('int32', 'float64'): algos.take_1d_int32_float64, ('int64', 'int64'): algos.take_1d_int64_int64, ('int64', 'float64'): algos.take_1d_int64_float64, ('float32', 'float32'): algos.take_1d_float32_float32, ('float32', 'float64'): algos.take_1d_float32_float64, ('float64', 'float64'): algos.take_1d_float64_float64, ('object', 'object'): algos.take_1d_object_object, ('bool', 'bool'): _view_wrapper(algos.take_1d_bool_bool, np.uint8, np.uint8), ('bool', 'object'): _view_wrapper(algos.take_1d_bool_object, np.uint8, None), ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(algos.take_1d_int64_int64, np.int64, np.int64, np.int64)}
_take_2d_axis0_dict = {('int8', 'int8'): algos.take_2d_axis0_int8_int8, ('int8', 'int32'): algos.take_2d_axis0_int8_int32, ('int8', 'int64'): algos.take_2d_axis0_int8_int64, ('int8', 'float64'): algos.take_2d_axis0_int8_float64, ('int16', 'int16'): algos.take_2d_axis0_int16_int16, ('int16', 'int32'): algos.take_2d_axis0_int16_int32, ('int16', 'int64'): algos.take_2d_axis0_int16_int64, ('int16', 'float64'): algos.take_2d_axis0_int16_float64, ('int32', 'int32'): algos.take_2d_axis0_int32_int32, ('int32', 'int64'): algos.take_2d_axis0_int32_int64, ('int32', 'float64'): algos.take_2d_axis0_int32_float64, ('int64', 'int64'): algos.take_2d_axis0_int64_int64, ('int64', 'float64'): algos.take_2d_axis0_int64_float64, ('float32', 'float32'): algos.take_2d_axis0_float32_float32, ('float32', 'float64'): algos.take_2d_axis0_float32_float64, ('float64', 'float64'): algos.take_2d_axis0_float64_float64, ('object', 'object'): algos.take_2d_axis0_object_object, ('bool', 'bool'): _view_wrapper(algos.take_2d_axis0_bool_bool, np.uint8, np.uint8), ('bool', 'object'): _view_wrapper(algos.take_2d_axis0_bool_object, np.uint8, None), ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(algos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64)}
_take_2d_axis1_dict = {('int8', 'int8'): algos.take_2d_axis1_int8_int8, ('int8', 'int32'): algos.take_2d_axis1_int8_int32, ('int8', 'int64'): algos.take_2d_axis1_int8_int64, ('int8', 'float64'): algos.take_2d_axis1_int8_float64, ('int16', 'int16'): algos.take_2d_axis1_int16_int16, ('int16', 'int32'): algos.take_2d_axis1_int16_int32, ('int16', 'int64'): algos.take_2d_axis1_int16_int64, ('int16', 'float64'): algos.take_2d_axis1_int16_float64, ('int32', 'int32'): algos.take_2d_axis1_int32_int32, ('int32', 'int64'): algos.take_2d_axis1_int32_int64, ('int32', 'float64'): algos.take_2d_axis1_int32_float64, ('int64', 'int64'): algos.take_2d_axis1_int64_int64, ('int64', 'float64'): algos.take_2d_axis1_int64_float64, ('float32', 'float32'): algos.take_2d_axis1_float32_float32, ('float32', 'float64'): algos.take_2d_axis1_float32_float64, ('float64', 'float64'): algos.take_2d_axis1_float64_float64, ('object', 'object'): algos.take_2d_axis1_object_object, ('bool', 'bool'): _view_wrapper(algos.take_2d_axis1_bool_bool, np.uint8, np.uint8), ('bool', 'object'): _view_wrapper(algos.take_2d_axis1_bool_object, np.uint8, None), ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(algos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64)}
_take_2d_multi_dict = {('int8', 'int8'): algos.take_2d_multi_int8_int8, ('int8', 'int32'): algos.take_2d_multi_int8_int32, ('int8', 'int64'): algos.take_2d_multi_int8_int64, ('int8', 'float64'): algos.take_2d_multi_int8_float64, ('int16', 'int16'): algos.take_2d_multi_int16_int16, ('int16', 'int32'): algos.take_2d_multi_int16_int32, ('int16', 'int64'): algos.take_2d_multi_int16_int64, ('int16', 'float64'): algos.take_2d_multi_int16_float64, ('int32', 'int32'): algos.take_2d_multi_int32_int32, ('int32', 'int64'): algos.take_2d_multi_int32_int64, ('int32', 'float64'): algos.take_2d_multi_int32_float64, ('int64', 'int64'): algos.take_2d_multi_int64_int64, ('int64', 'float64'): algos.take_2d_multi_int64_float64, ('float32', 'float32'): algos.take_2d_multi_float32_float32, ('float32', 'float64'): algos.take_2d_multi_float32_float64, ('float64', 'float64'): algos.take_2d_multi_float64_float64, ('object', 'object'): algos.take_2d_multi_object_object, ('bool', 'bool'): _view_wrapper(algos.take_2d_multi_bool_bool, np.uint8, np.uint8), ('bool', 'object'): _view_wrapper(algos.take_2d_multi_bool_object, np.uint8, None), ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(algos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64)}

def _get_take_nd_function(ndim, arr_dtype, out_dtype, axis=0, mask_info=None):
    if (ndim <= 2):
        tup = (arr_dtype.name, out_dtype.name)
        if (ndim == 1):
            func = _take_1d_dict.get(tup, None)
        elif (ndim == 2):
            if (axis == 0):
                func = _take_2d_axis0_dict.get(tup, None)
            else:
                func = _take_2d_axis1_dict.get(tup, None)
        if (func is not None):
            return func
        tup = (out_dtype.name, out_dtype.name)
        if (ndim == 1):
            func = _take_1d_dict.get(tup, None)
        elif (ndim == 2):
            if (axis == 0):
                func = _take_2d_axis0_dict.get(tup, None)
            else:
                func = _take_2d_axis1_dict.get(tup, None)
        if (func is not None):
            func = _convert_wrapper(func, out_dtype)
            return func

    def func2(arr, indexer, out, fill_value=np.nan):
        indexer = ensure_int64(indexer)
        _take_nd_object(arr, indexer, out, axis=axis, fill_value=fill_value, mask_info=mask_info)
    return func2

def take(arr, indices, axis=0, allow_fill=False, fill_value=None):
    '\n    Take elements from an array.\n\n    Parameters\n    ----------\n    arr : sequence\n        Non array-likes (sequences without a dtype) are coerced\n        to an ndarray.\n    indices : sequence of integers\n        Indices to be taken.\n    axis : int, default 0\n        The axis over which to select values.\n    allow_fill : bool, default False\n        How to handle negative values in `indices`.\n\n        * False: negative values in `indices` indicate positional indices\n          from the right (the default). This is similar to :func:`numpy.take`.\n\n        * True: negative values in `indices` indicate\n          missing values. These values are set to `fill_value`. Any other\n          negative values raise a ``ValueError``.\n\n    fill_value : any, optional\n        Fill value to use for NA-indices when `allow_fill` is True.\n        This may be ``None``, in which case the default NA value for\n        the type (``self.dtype.na_value``) is used.\n\n        For multi-dimensional `arr`, each *element* is filled with\n        `fill_value`.\n\n    Returns\n    -------\n    ndarray or ExtensionArray\n        Same type as the input.\n\n    Raises\n    ------\n    IndexError\n        When `indices` is out of bounds for the array.\n    ValueError\n        When the indexer contains negative values other than ``-1``\n        and `allow_fill` is True.\n\n    Notes\n    -----\n    When `allow_fill` is False, `indices` may be whatever dimensionality\n    is accepted by NumPy for `arr`.\n\n    When `allow_fill` is True, `indices` should be 1-D.\n\n    See Also\n    --------\n    numpy.take : Take elements from an array along an axis.\n\n    Examples\n    --------\n    >>> from pandas.api.extensions import take\n\n    With the default ``allow_fill=False``, negative numbers indicate\n    positional indices from the right.\n\n    >>> take(np.array([10, 20, 30]), [0, 0, -1])\n    array([10, 10, 30])\n\n    Setting ``allow_fill=True`` will place `fill_value` in those positions.\n\n    >>> take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)\n    array([10., 10., nan])\n\n    >>> take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True,\n    ...      fill_value=-10)\n    array([ 10,  10, -10])\n    '
    if (not is_array_like(arr)):
        arr = np.asarray(arr)
    indices = np.asarray(indices, dtype=np.intp)
    if allow_fill:
        validate_indices(indices, arr.shape[axis])
        result = take_1d(arr, indices, axis=axis, allow_fill=True, fill_value=fill_value)
    else:
        result = arr.take(indices, axis=axis)
    return result

def take_nd(arr, indexer, axis=0, out=None, fill_value=np.nan, allow_fill=True):
    '\n    Specialized Cython take which sets NaN values in one pass\n\n    This dispatches to ``take`` defined on ExtensionArrays. It does not\n    currently dispatch to ``SparseArray.take`` for sparse ``arr``.\n\n    Parameters\n    ----------\n    arr : array-like\n        Input array.\n    indexer : ndarray\n        1-D array of indices to take, subarrays corresponding to -1 value\n        indices are filed with fill_value\n    axis : int, default 0\n        Axis to take from\n    out : ndarray or None, default None\n        Optional output array, must be appropriate type to hold input and\n        fill_value together, if indexer has any -1 value entries; call\n        maybe_promote to determine this type for any fill_value\n    fill_value : any, default np.nan\n        Fill value to replace -1 values with\n    allow_fill : boolean, default True\n        If False, indexer is assumed to contain no -1 values so no filling\n        will be done.  This short-circuits computation of a mask.  Result is\n        undefined if allow_fill == False and -1 is present in indexer.\n\n    Returns\n    -------\n    subarray : array-like\n        May be the same type as the input, or cast to an ndarray.\n    '
    mask_info = None
    if isinstance(arr, ABCExtensionArray):
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)
    arr = extract_array(arr)
    arr = np.asarray(arr)
    if (indexer is None):
        indexer = np.arange(arr.shape[axis], dtype=np.int64)
        (dtype, fill_value) = (arr.dtype, arr.dtype.type())
    else:
        indexer = ensure_int64(indexer, copy=False)
        if (not allow_fill):
            (dtype, fill_value) = (arr.dtype, arr.dtype.type())
            mask_info = (None, False)
        else:
            (dtype, fill_value) = maybe_promote(arr.dtype, fill_value)
            if ((dtype != arr.dtype) and ((out is None) or (out.dtype != dtype))):
                mask = (indexer == (- 1))
                needs_masking = mask.any()
                mask_info = (mask, needs_masking)
                if needs_masking:
                    if ((out is not None) and (out.dtype != dtype)):
                        raise TypeError('Incompatible type for fill_value')
                else:
                    (dtype, fill_value) = (arr.dtype, arr.dtype.type())
    flip_order = False
    if (arr.ndim == 2):
        if arr.flags.f_contiguous:
            flip_order = True
    if flip_order:
        arr = arr.T
        axis = ((arr.ndim - axis) - 1)
        if (out is not None):
            out = out.T
    if (out is None):
        out_shape_ = list(arr.shape)
        out_shape_[axis] = len(indexer)
        out_shape = tuple(out_shape_)
        if (arr.flags.f_contiguous and (axis == (arr.ndim - 1))):
            out = np.empty(out_shape, dtype=dtype, order='F')
        else:
            out = np.empty(out_shape, dtype=dtype)
    func = _get_take_nd_function(arr.ndim, arr.dtype, out.dtype, axis=axis, mask_info=mask_info)
    func(arr, indexer, out, fill_value)
    if flip_order:
        out = out.T
    return out
take_1d = take_nd

def take_2d_multi(arr, indexer, fill_value=np.nan):
    '\n    Specialized Cython take which sets NaN values in one pass.\n    '
    assert (indexer is not None)
    assert (indexer[0] is not None)
    assert (indexer[1] is not None)
    (row_idx, col_idx) = indexer
    row_idx = ensure_int64(row_idx)
    col_idx = ensure_int64(col_idx)
    indexer = (row_idx, col_idx)
    mask_info = None
    (dtype, fill_value) = maybe_promote(arr.dtype, fill_value)
    if (dtype != arr.dtype):
        row_mask = (row_idx == (- 1))
        col_mask = (col_idx == (- 1))
        row_needs = row_mask.any()
        col_needs = col_mask.any()
        mask_info = ((row_mask, col_mask), (row_needs, col_needs))
        if (not (row_needs or col_needs)):
            (dtype, fill_value) = (arr.dtype, arr.dtype.type())
    out_shape = (len(row_idx), len(col_idx))
    out = np.empty(out_shape, dtype=dtype)
    func = _take_2d_multi_dict.get((arr.dtype.name, out.dtype.name), None)
    if ((func is None) and (arr.dtype != out.dtype)):
        func = _take_2d_multi_dict.get((out.dtype.name, out.dtype.name), None)
        if (func is not None):
            func = _convert_wrapper(func, out.dtype)
    if (func is None):

        def func(arr, indexer, out, fill_value=np.nan):
            _take_2d_multi_object(arr, indexer, out, fill_value=fill_value, mask_info=mask_info)
    func(arr, indexer, out=out, fill_value=fill_value)
    return out

def searchsorted(arr, value, side='left', sorter=None):
    "\n    Find indices where elements should be inserted to maintain order.\n\n    .. versionadded:: 0.25.0\n\n    Find the indices into a sorted array `arr` (a) such that, if the\n    corresponding elements in `value` were inserted before the indices,\n    the order of `arr` would be preserved.\n\n    Assuming that `arr` is sorted:\n\n    ======  ================================\n    `side`  returned index `i` satisfies\n    ======  ================================\n    left    ``arr[i-1] < value <= self[i]``\n    right   ``arr[i-1] <= value < self[i]``\n    ======  ================================\n\n    Parameters\n    ----------\n    arr: array-like\n        Input array. If `sorter` is None, then it must be sorted in\n        ascending order, otherwise `sorter` must be an array of indices\n        that sort it.\n    value : array_like\n        Values to insert into `arr`.\n    side : {'left', 'right'}, optional\n        If 'left', the index of the first suitable location found is given.\n        If 'right', return the last such index.  If there is no suitable\n        index, return either 0 or N (where N is the length of `self`).\n    sorter : 1-D array_like, optional\n        Optional array of integer indices that sort array a into ascending\n        order. They are typically the result of argsort.\n\n    Returns\n    -------\n    array of ints\n        Array of insertion points with the same shape as `value`.\n\n    See Also\n    --------\n    numpy.searchsorted : Similar method from NumPy.\n    "
    if (sorter is not None):
        sorter = ensure_platform_int(sorter)
    if (isinstance(arr, np.ndarray) and is_integer_dtype(arr.dtype) and (is_integer(value) or is_integer_dtype(value))):
        iinfo = np.iinfo(arr.dtype.type)
        value_arr = (np.array([value]) if is_scalar(value) else np.array(value))
        if ((value_arr >= iinfo.min).all() and (value_arr <= iinfo.max).all()):
            dtype = arr.dtype
        else:
            dtype = value_arr.dtype
        if is_scalar(value):
            value = dtype.type(value)
        else:
            value = array(value, dtype=dtype)
    elif (not (is_object_dtype(arr) or is_numeric_dtype(arr) or is_categorical_dtype(arr))):
        arr = ensure_wrapped_if_datetimelike(arr)
    result = arr.searchsorted(value, side=side, sorter=sorter)
    return result
_diff_special = {'float64', 'float32', 'int64', 'int32', 'int16', 'int8'}

def diff(arr, n, axis=0, stacklevel=3):
    '\n    difference of n between self,\n    analogous to s-s.shift(n)\n\n    Parameters\n    ----------\n    arr : ndarray\n    n : int\n        number of periods\n    axis : int\n        axis to shift on\n    stacklevel : int\n        The stacklevel for the lost dtype warning.\n\n    Returns\n    -------\n    shifted\n    '
    from pandas.core.arrays import PandasDtype
    n = int(n)
    na = np.nan
    dtype = arr.dtype
    if (dtype.kind == 'b'):
        op = operator.xor
    else:
        op = operator.sub
    if isinstance(dtype, PandasDtype):
        arr = np.asarray(arr)
        dtype = arr.dtype
    if is_extension_array_dtype(dtype):
        if hasattr(arr, f'__{op.__name__}__'):
            if (axis != 0):
                raise ValueError(f'cannot diff {type(arr).__name__} on axis={axis}')
            return op(arr, arr.shift(n))
        else:
            warn("dtype lost in 'diff()'. In the future this will raise a TypeError. Convert to a suitable dtype prior to calling 'diff'.", FutureWarning, stacklevel=stacklevel)
            arr = np.asarray(arr)
            dtype = arr.dtype
    is_timedelta = False
    is_bool = False
    if needs_i8_conversion(arr.dtype):
        dtype = np.int64
        arr = arr.view('i8')
        na = iNaT
        is_timedelta = True
    elif is_bool_dtype(dtype):
        dtype = np.object_
        is_bool = True
    elif is_integer_dtype(dtype):
        dtype = np.float64
    orig_ndim = arr.ndim
    if (orig_ndim == 1):
        arr = arr.reshape((- 1), 1)
    dtype = np.dtype(dtype)
    out_arr = np.empty(arr.shape, dtype=dtype)
    na_indexer = ([slice(None)] * arr.ndim)
    na_indexer[axis] = (slice(None, n) if (n >= 0) else slice(n, None))
    out_arr[tuple(na_indexer)] = na
    if ((arr.ndim == 2) and (arr.dtype.name in _diff_special)):
        algos.diff_2d(arr, out_arr, n, axis, datetimelike=is_timedelta)
    else:
        _res_indexer = ([slice(None)] * arr.ndim)
        _res_indexer[axis] = (slice(n, None) if (n >= 0) else slice(None, n))
        res_indexer = tuple(_res_indexer)
        _lag_indexer = ([slice(None)] * arr.ndim)
        _lag_indexer[axis] = (slice(None, (- n)) if (n > 0) else slice((- n), None))
        lag_indexer = tuple(_lag_indexer)
        if is_timedelta:
            res = arr[res_indexer]
            lag = arr[lag_indexer]
            mask = ((arr[res_indexer] == na) | (arr[lag_indexer] == na))
            if mask.any():
                res = res.copy()
                res[mask] = 0
                lag = lag.copy()
                lag[mask] = 0
            result = (res - lag)
            result[mask] = na
            out_arr[res_indexer] = result
        elif is_bool:
            out_arr[res_indexer] = (arr[res_indexer] ^ arr[lag_indexer])
        else:
            out_arr[res_indexer] = (arr[res_indexer] - arr[lag_indexer])
    if is_timedelta:
        out_arr = out_arr.view('timedelta64[ns]')
    if (orig_ndim == 1):
        out_arr = out_arr[:, 0]
    return out_arr

def safe_sort(values, codes=None, na_sentinel=(- 1), assume_unique=False, verify=True):
    '\n    Sort ``values`` and reorder corresponding ``codes``.\n\n    ``values`` should be unique if ``codes`` is not None.\n    Safe for use with mixed types (int, str), orders ints before strs.\n\n    Parameters\n    ----------\n    values : list-like\n        Sequence; must be unique if ``codes`` is not None.\n    codes : list_like, optional\n        Indices to ``values``. All out of bound indices are treated as\n        "not found" and will be masked with ``na_sentinel``.\n    na_sentinel : int, default -1\n        Value in ``codes`` to mark "not found".\n        Ignored when ``codes`` is None.\n    assume_unique : bool, default False\n        When True, ``values`` are assumed to be unique, which can speed up\n        the calculation. Ignored when ``codes`` is None.\n    verify : bool, default True\n        Check if codes are out of bound for the values and put out of bound\n        codes equal to na_sentinel. If ``verify=False``, it is assumed there\n        are no out of bound codes. Ignored when ``codes`` is None.\n\n        .. versionadded:: 0.25.0\n\n    Returns\n    -------\n    ordered : ndarray\n        Sorted ``values``\n    new_codes : ndarray\n        Reordered ``codes``; returned when ``codes`` is not None.\n\n    Raises\n    ------\n    TypeError\n        * If ``values`` is not list-like or if ``codes`` is neither None\n        nor list-like\n        * If ``values`` cannot be sorted\n    ValueError\n        * If ``codes`` is not None and ``values`` contain duplicates.\n    '
    if (not is_list_like(values)):
        raise TypeError('Only list-like objects are allowed to be passed to safe_sort as values')
    if (not isinstance(values, (np.ndarray, ABCExtensionArray))):
        (dtype, _) = infer_dtype_from_array(values)
        values = np.asarray(values, dtype=dtype)
    sorter = None
    if ((not is_extension_array_dtype(values)) and (lib.infer_dtype(values, skipna=False) == 'mixed-integer')):
        ordered = _sort_mixed(values)
    else:
        try:
            sorter = values.argsort()
            ordered = values.take(sorter)
        except TypeError:
            if (values.size and isinstance(values[0], tuple)):
                ordered = _sort_tuples(values)
            else:
                ordered = _sort_mixed(values)
    if (codes is None):
        return ordered
    if (not is_list_like(codes)):
        raise TypeError('Only list-like objects or None are allowed to be passed to safe_sort as codes')
    codes = ensure_platform_int(np.asarray(codes))
    if ((not assume_unique) and (not (len(unique(values)) == len(values)))):
        raise ValueError('values should be unique if codes is not None')
    if (sorter is None):
        (hash_klass, values) = get_data_algo(values)
        t = hash_klass(len(values))
        t.map_locations(values)
        sorter = ensure_platform_int(t.lookup(ordered))
    if (na_sentinel == (- 1)):
        order2 = sorter.argsort()
        new_codes = take_1d(order2, codes, fill_value=(- 1))
        if verify:
            mask = ((codes < (- len(values))) | (codes >= len(values)))
        else:
            mask = None
    else:
        reverse_indexer = np.empty(len(sorter), dtype=np.int_)
        reverse_indexer.put(sorter, np.arange(len(sorter)))
        new_codes = reverse_indexer.take(codes, mode='wrap')
        mask = (codes == na_sentinel)
        if verify:
            mask = ((mask | (codes < (- len(values)))) | (codes >= len(values)))
    if (mask is not None):
        np.putmask(new_codes, mask, na_sentinel)
    return (ordered, ensure_platform_int(new_codes))

def _sort_mixed(values):
    ' order ints before strings in 1d arrays, safe in py3 '
    str_pos = np.array([isinstance(x, str) for x in values], dtype=bool)
    nums = np.sort(values[(~ str_pos)])
    strs = np.sort(values[str_pos])
    return np.concatenate([nums, np.asarray(strs, dtype=object)])

def _sort_tuples(values):
    "\n    Convert array of tuples (1d) to array or array (2d).\n    We need to keep the columns separately as they contain different types and\n    nans (can't use `np.sort` as it may fail when str and nan are mixed in a\n    column as types cannot be compared).\n    "
    from pandas.core.internals.construction import to_arrays
    from pandas.core.sorting import lexsort_indexer
    (arrays, _) = to_arrays(values, None)
    indexer = lexsort_indexer(arrays, orders=True)
    return values[indexer]
