
'\ndata hash pandas / numpy objects\n'
import itertools
from typing import Optional
import numpy as np
import pandas._libs.hashing as hashing
from pandas.core.dtypes.common import is_categorical_dtype, is_extension_array_dtype, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCMultiIndex, ABCSeries
_default_hash_key = '0123456789123456'

def combine_hash_arrays(arrays, num_items):
    "\n    Parameters\n    ----------\n    arrays : generator\n    num_items : int\n\n    Should be the same as CPython's tupleobject.c\n    "
    try:
        first = next(arrays)
    except StopIteration:
        return np.array([], dtype=np.uint64)
    arrays = itertools.chain([first], arrays)
    mult = np.uint64(1000003)
    out = (np.zeros_like(first) + np.uint64(3430008))
    for (i, a) in enumerate(arrays):
        inverse_i = (num_items - i)
        out ^= a
        out *= mult
        mult += np.uint64(((82520 + inverse_i) + inverse_i))
    assert ((i + 1) == num_items), 'Fed in wrong num_items'
    out += np.uint64(97531)
    return out

def hash_pandas_object(obj, index=True, encoding='utf8', hash_key=_default_hash_key, categorize=True):
    "\n    Return a data hash of the Index/Series/DataFrame.\n\n    Parameters\n    ----------\n    index : bool, default True\n        Include the index in the hash (if Series/DataFrame).\n    encoding : str, default 'utf8'\n        Encoding for data & key when strings.\n    hash_key : str, default _default_hash_key\n        Hash_key for string key to encode.\n    categorize : bool, default True\n        Whether to first categorize object arrays before hashing. This is more\n        efficient when the array contains duplicate values.\n\n    Returns\n    -------\n    Series of uint64, same length as the object\n    "
    from pandas import Series
    if (hash_key is None):
        hash_key = _default_hash_key
    if isinstance(obj, ABCMultiIndex):
        return Series(hash_tuples(obj, encoding, hash_key), dtype='uint64', copy=False)
    elif isinstance(obj, ABCIndex):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype('uint64', copy=False)
        h = Series(h, index=obj, dtype='uint64', copy=False)
    elif isinstance(obj, ABCSeries):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype('uint64', copy=False)
        if index:
            index_iter = (hash_pandas_object(obj.index, index=False, encoding=encoding, hash_key=hash_key, categorize=categorize)._values for _ in [None])
            arrays = itertools.chain([h], index_iter)
            h = combine_hash_arrays(arrays, 2)
        h = Series(h, index=obj.index, dtype='uint64', copy=False)
    elif isinstance(obj, ABCDataFrame):
        hashes = (hash_array(series._values) for (_, series) in obj.items())
        num_items = len(obj.columns)
        if index:
            index_hash_generator = (hash_pandas_object(obj.index, index=False, encoding=encoding, hash_key=hash_key, categorize=categorize)._values for _ in [None])
            num_items += 1
            _hashes = itertools.chain(hashes, index_hash_generator)
            hashes = (x for x in _hashes)
        h = combine_hash_arrays(hashes, num_items)
        h = Series(h, index=obj.index, dtype='uint64', copy=False)
    else:
        raise TypeError(f'Unexpected type for hashing {type(obj)}')
    return h

def hash_tuples(vals, encoding='utf8', hash_key=_default_hash_key):
    "\n    Hash an MultiIndex / list-of-tuples efficiently\n\n    Parameters\n    ----------\n    vals : MultiIndex, list-of-tuples, or single tuple\n    encoding : str, default 'utf8'\n    hash_key : str, default _default_hash_key\n\n    Returns\n    -------\n    ndarray of hashed values array\n    "
    is_tuple = False
    if isinstance(vals, tuple):
        vals = [vals]
        is_tuple = True
    elif (not is_list_like(vals)):
        raise TypeError('must be convertible to a list-of-tuples')
    from pandas import Categorical, MultiIndex
    if (not isinstance(vals, ABCMultiIndex)):
        vals = MultiIndex.from_tuples(vals)
    vals = [Categorical(vals.codes[level], vals.levels[level], ordered=False, fastpath=True) for level in range(vals.nlevels)]
    hashes = (_hash_categorical(cat, encoding=encoding, hash_key=hash_key) for cat in vals)
    h = combine_hash_arrays(hashes, len(vals))
    if is_tuple:
        h = h[0]
    return h

def _hash_categorical(c, encoding, hash_key):
    '\n    Hash a Categorical by hashing its categories, and then mapping the codes\n    to the hashes\n\n    Parameters\n    ----------\n    c : Categorical\n    encoding : str\n    hash_key : str\n\n    Returns\n    -------\n    ndarray of hashed values array, same size as len(c)\n    '
    values = np.asarray(c.categories._values)
    hashed = hash_array(values, encoding, hash_key, categorize=False)
    mask = c.isna()
    if len(hashed):
        result = hashed.take(c.codes)
    else:
        result = np.zeros(len(mask), dtype='uint64')
    if mask.any():
        result[mask] = np.iinfo(np.uint64).max
    return result

def hash_array(vals, encoding='utf8', hash_key=_default_hash_key, categorize=True):
    "\n    Given a 1d array, return an array of deterministic integers.\n\n    Parameters\n    ----------\n    vals : ndarray, Categorical\n    encoding : str, default 'utf8'\n        Encoding for data & key when strings.\n    hash_key : str, default _default_hash_key\n        Hash_key for string key to encode.\n    categorize : bool, default True\n        Whether to first categorize object arrays before hashing. This is more\n        efficient when the array contains duplicate values.\n\n    Returns\n    -------\n    1d uint64 numpy array of hash values, same length as the vals\n    "
    if (not hasattr(vals, 'dtype')):
        raise TypeError('must pass a ndarray-like')
    dtype = vals.dtype
    if is_categorical_dtype(dtype):
        return _hash_categorical(vals, encoding, hash_key)
    elif is_extension_array_dtype(dtype):
        (vals, _) = vals._values_for_factorize()
        dtype = vals.dtype
    if np.issubdtype(dtype, np.complex128):
        return (hash_array(np.real(vals)) + (23 * hash_array(np.imag(vals))))
    elif isinstance(dtype, bool):
        vals = vals.astype('u8')
    elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        vals = vals.view('i8').astype('u8', copy=False)
    elif (issubclass(dtype.type, np.number) and (dtype.itemsize <= 8)):
        vals = vals.view(f'u{vals.dtype.itemsize}').astype('u8')
    else:
        if categorize:
            from pandas import Categorical, Index, factorize
            (codes, categories) = factorize(vals, sort=False)
            cat = Categorical(codes, Index(categories), ordered=False, fastpath=True)
            return _hash_categorical(cat, encoding, hash_key)
        try:
            vals = hashing.hash_object_array(vals, hash_key, encoding)
        except TypeError:
            vals = hashing.hash_object_array(vals.astype(str).astype(object), hash_key, encoding)
    vals ^= (vals >> 30)
    vals *= np.uint64(13787848793156543929)
    vals ^= (vals >> 27)
    vals *= np.uint64(10723151780598845931)
    vals ^= (vals >> 31)
    return vals
