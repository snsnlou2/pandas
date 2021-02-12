
'\nMisc tools for implementing data structures\n\nNote: pandas.core.common is *not* part of the public API.\n'
from collections import abc, defaultdict
import contextlib
from functools import partial
import inspect
from typing import Any, Collection, Iterable, Iterator, List, Union, cast
import warnings
import numpy as np
from pandas._libs import lib
from pandas._typing import AnyArrayLike, Scalar, T
from pandas.compat.numpy import np_version_under1p18
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_array_like, is_bool_dtype, is_extension_array_dtype, is_integer
from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCSeries
from pandas.core.dtypes.inference import iterable_not_string
from pandas.core.dtypes.missing import isna, isnull, notnull

class SettingWithCopyError(ValueError):
    pass

class SettingWithCopyWarning(Warning):
    pass

def flatten(line):
    "\n    Flatten an arbitrarily nested sequence.\n\n    Parameters\n    ----------\n    line : sequence\n        The non string sequence to flatten\n\n    Notes\n    -----\n    This doesn't consider strings sequences.\n\n    Returns\n    -------\n    flattened : generator\n    "
    for element in line:
        if iterable_not_string(element):
            (yield from flatten(element))
        else:
            (yield element)

def consensus_name_attr(objs):
    name = objs[0].name
    for obj in objs[1:]:
        try:
            if (obj.name != name):
                name = None
        except ValueError:
            name = None
    return name

def is_bool_indexer(key):
    '\n    Check whether `key` is a valid boolean indexer.\n\n    Parameters\n    ----------\n    key : Any\n        Only list-likes may be considered boolean indexers.\n        All other types are not considered a boolean indexer.\n        For array-like input, boolean ndarrays or ExtensionArrays\n        with ``_is_boolean`` set are considered boolean indexers.\n\n    Returns\n    -------\n    bool\n        Whether `key` is a valid boolean indexer.\n\n    Raises\n    ------\n    ValueError\n        When the array is an object-dtype ndarray or ExtensionArray\n        and contains missing values.\n\n    See Also\n    --------\n    check_array_indexer : Check that `key` is a valid array to index,\n        and convert to an ndarray.\n    '
    if (isinstance(key, (ABCSeries, np.ndarray, ABCIndex)) or (is_array_like(key) and is_extension_array_dtype(key.dtype))):
        if (key.dtype == np.object_):
            key = np.asarray(key)
            if (not lib.is_bool_array(key)):
                na_msg = 'Cannot mask with non-boolean array containing NA / NaN values'
                if ((lib.infer_dtype(key) == 'boolean') and isna(key).any()):
                    raise ValueError(na_msg)
                return False
            return True
        elif is_bool_dtype(key.dtype):
            return True
    elif isinstance(key, list):
        try:
            arr = np.asarray(key)
            return ((arr.dtype == np.bool_) and (len(arr) == len(key)))
        except TypeError:
            return False
    return False

def cast_scalar_indexer(val, warn_float=False):
    '\n    To avoid numpy DeprecationWarnings, cast float to integer where valid.\n\n    Parameters\n    ----------\n    val : scalar\n    warn_float : bool, default False\n        If True, issue deprecation warning for a float indexer.\n\n    Returns\n    -------\n    outval : scalar\n    '
    if (lib.is_float(val) and val.is_integer()):
        if warn_float:
            warnings.warn('Indexing with a float is deprecated, and will raise an IndexError in pandas 2.0. You can manually convert to an integer key instead.', FutureWarning, stacklevel=3)
        return int(val)
    return val

def not_none(*args):
    '\n    Returns a generator consisting of the arguments that are not None.\n    '
    return (arg for arg in args if (arg is not None))

def any_none(*args):
    '\n    Returns a boolean indicating if any argument is None.\n    '
    return any(((arg is None) for arg in args))

def all_none(*args):
    '\n    Returns a boolean indicating if all arguments are None.\n    '
    return all(((arg is None) for arg in args))

def any_not_none(*args):
    '\n    Returns a boolean indicating if any argument is not None.\n    '
    return any(((arg is not None) for arg in args))

def all_not_none(*args):
    '\n    Returns a boolean indicating if all arguments are not None.\n    '
    return all(((arg is not None) for arg in args))

def count_not_none(*args):
    '\n    Returns the count of arguments that are not None.\n    '
    return sum(((x is not None) for x in args))

def asarray_tuplesafe(values, dtype=None):
    if (not (isinstance(values, (list, tuple)) or hasattr(values, '__array__'))):
        values = list(values)
    elif isinstance(values, ABCIndex):
        return values._values
    if (isinstance(values, list) and (dtype in [np.object_, object])):
        return construct_1d_object_array_from_listlike(values)
    result = np.asarray(values, dtype=dtype)
    if issubclass(result.dtype.type, str):
        result = np.asarray(values, dtype=object)
    if (result.ndim == 2):
        values = [tuple(x) for x in values]
        result = construct_1d_object_array_from_listlike(values)
    return result

def index_labels_to_array(labels, dtype=None):
    '\n    Transform label or iterable of labels to array, for use in Index.\n\n    Parameters\n    ----------\n    dtype : dtype\n        If specified, use as dtype of the resulting array, otherwise infer.\n\n    Returns\n    -------\n    array\n    '
    if isinstance(labels, (str, tuple)):
        labels = [labels]
    if (not isinstance(labels, (list, np.ndarray))):
        try:
            labels = list(labels)
        except TypeError:
            labels = [labels]
    labels = asarray_tuplesafe(labels, dtype=dtype)
    return labels

def maybe_make_list(obj):
    if ((obj is not None) and (not isinstance(obj, (tuple, list)))):
        return [obj]
    return obj

def maybe_iterable_to_list(obj):
    '\n    If obj is Iterable but not list-like, consume into list.\n    '
    if (isinstance(obj, abc.Iterable) and (not isinstance(obj, abc.Sized))):
        return list(obj)
    obj = cast(Collection, obj)
    return obj

def is_null_slice(obj):
    '\n    We have a null slice.\n    '
    return (isinstance(obj, slice) and (obj.start is None) and (obj.stop is None) and (obj.step is None))

def is_true_slices(line):
    '\n    Find non-trivial slices in "line": return a list of booleans with same length.\n    '
    return [(isinstance(k, slice) and (not is_null_slice(k))) for k in line]

def is_full_slice(obj, line):
    '\n    We have a full length slice.\n    '
    return (isinstance(obj, slice) and (obj.start == 0) and (obj.stop == line) and (obj.step is None))

def get_callable_name(obj):
    if hasattr(obj, '__name__'):
        return getattr(obj, '__name__')
    if isinstance(obj, partial):
        return get_callable_name(obj.func)
    if hasattr(obj, '__call__'):
        return type(obj).__name__
    return None

def apply_if_callable(maybe_callable, obj, **kwargs):
    '\n    Evaluate possibly callable input using obj and kwargs if it is callable,\n    otherwise return as it is.\n\n    Parameters\n    ----------\n    maybe_callable : possibly a callable\n    obj : NDFrame\n    **kwargs\n    '
    if callable(maybe_callable):
        return maybe_callable(obj, **kwargs)
    return maybe_callable

def standardize_mapping(into):
    '\n    Helper function to standardize a supplied mapping.\n\n    Parameters\n    ----------\n    into : instance or subclass of collections.abc.Mapping\n        Must be a class, an initialized collections.defaultdict,\n        or an instance of a collections.abc.Mapping subclass.\n\n    Returns\n    -------\n    mapping : a collections.abc.Mapping subclass or other constructor\n        a callable object that can accept an iterator to create\n        the desired Mapping.\n\n    See Also\n    --------\n    DataFrame.to_dict\n    Series.to_dict\n    '
    if (not inspect.isclass(into)):
        if isinstance(into, defaultdict):
            return partial(defaultdict, into.default_factory)
        into = type(into)
    if (not issubclass(into, abc.Mapping)):
        raise TypeError(f'unsupported type: {into}')
    elif (into == defaultdict):
        raise TypeError('to_dict() only accepts initialized defaultdicts')
    return into

def random_state(state=None):
    '\n    Helper function for processing random_state arguments.\n\n    Parameters\n    ----------\n    state : int, array-like, BitGenerator (NumPy>=1.17), np.random.RandomState, None.\n        If receives an int, array-like, or BitGenerator, passes to\n        np.random.RandomState() as seed.\n        If receives an np.random.RandomState object, just returns object.\n        If receives `None`, returns np.random.\n        If receives anything else, raises an informative ValueError.\n\n        .. versionchanged:: 1.1.0\n\n            array-like and BitGenerator (for NumPy>=1.18) object now passed to\n            np.random.RandomState() as seed\n\n        Default None.\n\n    Returns\n    -------\n    np.random.RandomState\n\n    '
    if (is_integer(state) or is_array_like(state) or ((not np_version_under1p18) and isinstance(state, np.random.BitGenerator))):
        return np.random.RandomState(state)
    elif isinstance(state, np.random.RandomState):
        return state
    elif (state is None):
        return np.random
    else:
        raise ValueError('random_state must be an integer, array-like, a BitGenerator, a numpy RandomState, or None')

def pipe(obj, func, *args, **kwargs):
    '\n    Apply a function ``func`` to object ``obj`` either by passing obj as the\n    first argument to the function or, in the case that the func is a tuple,\n    interpret the first element of the tuple as a function and pass the obj to\n    that function as a keyword argument whose key is the value of the second\n    element of the tuple.\n\n    Parameters\n    ----------\n    func : callable or tuple of (callable, str)\n        Function to apply to this object or, alternatively, a\n        ``(callable, data_keyword)`` tuple where ``data_keyword`` is a\n        string indicating the keyword of `callable`` that expects the\n        object.\n    *args : iterable, optional\n        Positional arguments passed into ``func``.\n    **kwargs : dict, optional\n        A dictionary of keyword arguments passed into ``func``.\n\n    Returns\n    -------\n    object : the return type of ``func``.\n    '
    if isinstance(func, tuple):
        (func, target) = func
        if (target in kwargs):
            msg = f'{target} is both the pipe target and a keyword argument'
            raise ValueError(msg)
        kwargs[target] = obj
        return func(*args, **kwargs)
    else:
        return func(obj, *args, **kwargs)

def get_rename_function(mapper):
    '\n    Returns a function that will map names/labels, dependent if mapper\n    is a dict, Series or just a function.\n    '
    if isinstance(mapper, (abc.Mapping, ABCSeries)):

        def f(x):
            if (x in mapper):
                return mapper[x]
            else:
                return x
    else:
        f = mapper
    return f

def convert_to_list_like(values):
    '\n    Convert list-like or scalar input to list-like. List, numpy and pandas array-like\n    inputs are returned unmodified whereas others are converted to list.\n    '
    if isinstance(values, (list, np.ndarray, ABCIndex, ABCSeries, ABCExtensionArray)):
        return values
    elif (isinstance(values, abc.Iterable) and (not isinstance(values, str))):
        return list(values)
    return [values]

@contextlib.contextmanager
def temp_setattr(obj, attr, value):
    'Temporarily set attribute on an object.\n\n    Args:\n        obj: Object whose attribute will be modified.\n        attr: Attribute to modify.\n        value: Value to temporarily set attribute to.\n\n    Yields:\n        obj with modified attribute.\n    '
    old_value = getattr(obj, attr)
    setattr(obj, attr, value)
    (yield obj)
    setattr(obj, attr, old_value)
