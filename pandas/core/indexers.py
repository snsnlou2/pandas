
'\nLow-dependency indexing utilities.\n'
import warnings
import numpy as np
from pandas._typing import Any, AnyArrayLike
from pandas.core.dtypes.common import is_array_like, is_bool_dtype, is_extension_array_dtype, is_integer, is_integer_dtype, is_list_like
from pandas.core.dtypes.generic import ABCIndex, ABCSeries

def is_valid_positional_slice(slc):
    '\n    Check if a slice object can be interpreted as a positional indexer.\n\n    Parameters\n    ----------\n    slc : slice\n\n    Returns\n    -------\n    bool\n\n    Notes\n    -----\n    A valid positional slice may also be interpreted as a label-based slice\n    depending on the index being sliced.\n    '

    def is_int_or_none(val):
        return ((val is None) or is_integer(val))
    return (is_int_or_none(slc.start) and is_int_or_none(slc.stop) and is_int_or_none(slc.step))

def is_list_like_indexer(key):
    '\n    Check if we have a list-like indexer that is *not* a NamedTuple.\n\n    Parameters\n    ----------\n    key : object\n\n    Returns\n    -------\n    bool\n    '
    return (is_list_like(key) and (not (isinstance(key, tuple) and (type(key) is not tuple))))

def is_scalar_indexer(indexer, ndim):
    '\n    Return True if we are all scalar indexers.\n\n    Parameters\n    ----------\n    indexer : object\n    ndim : int\n        Number of dimensions in the object being indexed.\n\n    Returns\n    -------\n    bool\n    '
    if ((ndim == 1) and is_integer(indexer)):
        return True
    if isinstance(indexer, tuple):
        if (len(indexer) == ndim):
            return all(((is_integer(x) or (isinstance(x, np.ndarray) and (x.ndim == len(x) == 1))) for x in indexer))
    return False

def is_empty_indexer(indexer, arr_value):
    '\n    Check if we have an empty indexer.\n\n    Parameters\n    ----------\n    indexer : object\n    arr_value : np.ndarray\n\n    Returns\n    -------\n    bool\n    '
    if (is_list_like(indexer) and (not len(indexer))):
        return True
    if (arr_value.ndim == 1):
        if (not isinstance(indexer, tuple)):
            indexer = (indexer,)
        return any(((isinstance(idx, np.ndarray) and (len(idx) == 0)) for idx in indexer))
    return False

def check_setitem_lengths(indexer, value, values):
    "\n    Validate that value and indexer are the same length.\n\n    An special-case is allowed for when the indexer is a boolean array\n    and the number of true values equals the length of ``value``. In\n    this case, no exception is raised.\n\n    Parameters\n    ----------\n    indexer : sequence\n        Key for the setitem.\n    value : array-like\n        Value for the setitem.\n    values : array-like\n        Values being set into.\n\n    Returns\n    -------\n    bool\n        Whether this is an empty listlike setting which is a no-op.\n\n    Raises\n    ------\n    ValueError\n        When the indexer is an ndarray or list and the lengths don't match.\n    "
    no_op = False
    if isinstance(indexer, (np.ndarray, list)):
        if is_list_like(value):
            if (len(indexer) != len(value)):
                if (not (isinstance(indexer, np.ndarray) and (indexer.dtype == np.bool_) and (len(indexer[indexer]) == len(value)))):
                    raise ValueError('cannot set using a list-like indexer with a different length than the value')
            if (not len(indexer)):
                no_op = True
    elif isinstance(indexer, slice):
        if is_list_like(value):
            if (len(value) != length_of_indexer(indexer, values)):
                raise ValueError('cannot set using a slice indexer with a different length than the value')
            if (not len(value)):
                no_op = True
    return no_op

def validate_indices(indices, n):
    '\n    Perform bounds-checking for an indexer.\n\n    -1 is allowed for indicating missing values.\n\n    Parameters\n    ----------\n    indices : ndarray\n    n : int\n        Length of the array being indexed.\n\n    Raises\n    ------\n    ValueError\n\n    Examples\n    --------\n    >>> validate_indices([1, 2], 3)\n    # OK\n    >>> validate_indices([1, -2], 3)\n    ValueError\n    >>> validate_indices([1, 2, 3], 3)\n    IndexError\n    >>> validate_indices([-1, -1], 0)\n    # OK\n    >>> validate_indices([0, 1], 0)\n    IndexError\n    '
    if len(indices):
        min_idx = indices.min()
        if (min_idx < (- 1)):
            msg = f"'indices' contains values less than allowed ({min_idx} < -1)"
            raise ValueError(msg)
        max_idx = indices.max()
        if (max_idx >= n):
            raise IndexError('indices are out-of-bounds')

def maybe_convert_indices(indices, n):
    '\n    Attempt to convert indices into valid, positive indices.\n\n    If we have negative indices, translate to positive here.\n    If we have indices that are out-of-bounds, raise an IndexError.\n\n    Parameters\n    ----------\n    indices : array-like\n        Array of indices that we are to convert.\n    n : int\n        Number of elements in the array that we are indexing.\n\n    Returns\n    -------\n    array-like\n        An array-like of positive indices that correspond to the ones\n        that were passed in initially to this function.\n\n    Raises\n    ------\n    IndexError\n        One of the converted indices either exceeded the number of,\n        elements (specified by `n`), or was still negative.\n    '
    if isinstance(indices, list):
        indices = np.array(indices)
        if (len(indices) == 0):
            return np.empty(0, dtype=np.intp)
    mask = (indices < 0)
    if mask.any():
        indices = indices.copy()
        indices[mask] += n
    mask = ((indices >= n) | (indices < 0))
    if mask.any():
        raise IndexError('indices are out-of-bounds')
    return indices

def length_of_indexer(indexer, target=None):
    '\n    Return the expected length of target[indexer]\n\n    Returns\n    -------\n    int\n    '
    if ((target is not None) and isinstance(indexer, slice)):
        target_len = len(target)
        start = indexer.start
        stop = indexer.stop
        step = indexer.step
        if (start is None):
            start = 0
        elif (start < 0):
            start += target_len
        if ((stop is None) or (stop > target_len)):
            stop = target_len
        elif (stop < 0):
            stop += target_len
        if (step is None):
            step = 1
        elif (step < 0):
            (start, stop) = ((stop + 1), (start + 1))
            step = (- step)
        return ((((stop - start) + step) - 1) // step)
    elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
        if isinstance(indexer, list):
            indexer = np.array(indexer)
        if (indexer.dtype == bool):
            return indexer.sum()
        return len(indexer)
    elif (not is_list_like_indexer(indexer)):
        return 1
    raise AssertionError('cannot find the length of the indexer')

def deprecate_ndim_indexing(result, stacklevel=3):
    '\n    Helper function to raise the deprecation warning for multi-dimensional\n    indexing on 1D Series/Index.\n\n    GH#27125 indexer like idx[:, None] expands dim, but we cannot do that\n    and keep an index, so we currently return ndarray, which is deprecated\n    (Deprecation GH#30588).\n    '
    if (np.ndim(result) > 1):
        warnings.warn('Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.', FutureWarning, stacklevel=stacklevel)

def unpack_1tuple(tup):
    '\n    If we have a length-1 tuple/list that contains a slice, unpack to just\n    the slice.\n\n    Notes\n    -----\n    The list case is deprecated.\n    '
    if ((len(tup) == 1) and isinstance(tup[0], slice)):
        if isinstance(tup, list):
            warnings.warn('Indexing with a single-item list containing a slice is deprecated and will raise in a future version.  Pass a tuple instead.', FutureWarning, stacklevel=3)
        return tup[0]
    return tup

def check_array_indexer(array, indexer):
    '\n    Check if `indexer` is a valid array indexer for `array`.\n\n    For a boolean mask, `array` and `indexer` are checked to have the same\n    length. The dtype is validated, and if it is an integer or boolean\n    ExtensionArray, it is checked if there are missing values present, and\n    it is converted to the appropriate numpy array. Other dtypes will raise\n    an error.\n\n    Non-array indexers (integer, slice, Ellipsis, tuples, ..) are passed\n    through as is.\n\n    .. versionadded:: 1.0.0\n\n    Parameters\n    ----------\n    array : array-like\n        The array that is being indexed (only used for the length).\n    indexer : array-like or list-like\n        The array-like that\'s used to index. List-like input that is not yet\n        a numpy array or an ExtensionArray is converted to one. Other input\n        types are passed through as is.\n\n    Returns\n    -------\n    numpy.ndarray\n        The validated indexer as a numpy array that can be used to index.\n\n    Raises\n    ------\n    IndexError\n        When the lengths don\'t match.\n    ValueError\n        When `indexer` cannot be converted to a numpy ndarray to index\n        (e.g. presence of missing values).\n\n    See Also\n    --------\n    api.types.is_bool_dtype : Check if `key` is of boolean dtype.\n\n    Examples\n    --------\n    When checking a boolean mask, a boolean ndarray is returned when the\n    arguments are all valid.\n\n    >>> mask = pd.array([True, False])\n    >>> arr = pd.array([1, 2])\n    >>> pd.api.indexers.check_array_indexer(arr, mask)\n    array([ True, False])\n\n    An IndexError is raised when the lengths don\'t match.\n\n    >>> mask = pd.array([True, False, True])\n    >>> pd.api.indexers.check_array_indexer(arr, mask)\n    Traceback (most recent call last):\n    ...\n    IndexError: Boolean index has wrong length: 3 instead of 2.\n\n    NA values in a boolean array are treated as False.\n\n    >>> mask = pd.array([True, pd.NA])\n    >>> pd.api.indexers.check_array_indexer(arr, mask)\n    array([ True, False])\n\n    A numpy boolean mask will get passed through (if the length is correct):\n\n    >>> mask = np.array([True, False])\n    >>> pd.api.indexers.check_array_indexer(arr, mask)\n    array([ True, False])\n\n    Similarly for integer indexers, an integer ndarray is returned when it is\n    a valid indexer, otherwise an error is  (for integer indexers, a matching\n    length is not required):\n\n    >>> indexer = pd.array([0, 2], dtype="Int64")\n    >>> arr = pd.array([1, 2, 3])\n    >>> pd.api.indexers.check_array_indexer(arr, indexer)\n    array([0, 2])\n\n    >>> indexer = pd.array([0, pd.NA], dtype="Int64")\n    >>> pd.api.indexers.check_array_indexer(arr, indexer)\n    Traceback (most recent call last):\n    ...\n    ValueError: Cannot index with an integer indexer containing NA values\n\n    For non-integer/boolean dtypes, an appropriate error is raised:\n\n    >>> indexer = np.array([0., 2.], dtype="float64")\n    >>> pd.api.indexers.check_array_indexer(arr, indexer)\n    Traceback (most recent call last):\n    ...\n    IndexError: arrays used as indices must be of integer or boolean type\n    '
    from pandas.core.construction import array as pd_array
    if is_list_like(indexer):
        if isinstance(indexer, tuple):
            return indexer
    else:
        return indexer
    if (not is_array_like(indexer)):
        indexer = pd_array(indexer)
        if (len(indexer) == 0):
            indexer = np.array([], dtype=np.intp)
    dtype = indexer.dtype
    if is_bool_dtype(dtype):
        if is_extension_array_dtype(dtype):
            indexer = indexer.to_numpy(dtype=bool, na_value=False)
        else:
            indexer = np.asarray(indexer, dtype=bool)
        if (len(indexer) != len(array)):
            raise IndexError(f'Boolean index has wrong length: {len(indexer)} instead of {len(array)}')
    elif is_integer_dtype(dtype):
        try:
            indexer = np.asarray(indexer, dtype=np.intp)
        except ValueError as err:
            raise ValueError('Cannot index with an integer indexer containing NA values') from err
    else:
        raise IndexError('arrays used as indices must be of integer or boolean type')
    return indexer
