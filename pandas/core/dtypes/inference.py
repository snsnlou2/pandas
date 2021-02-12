
' basic inference routines '
from collections import abc
from numbers import Number
import re
from typing import Pattern
import numpy as np
from pandas._libs import lib
is_bool = lib.is_bool
is_integer = lib.is_integer
is_float = lib.is_float
is_complex = lib.is_complex
is_scalar = lib.is_scalar
is_decimal = lib.is_decimal
is_interval = lib.is_interval
is_list_like = lib.is_list_like
is_iterator = lib.is_iterator

def is_number(obj):
    '\n    Check if the object is a number.\n\n    Returns True when the object is a number, and False if is not.\n\n    Parameters\n    ----------\n    obj : any type\n        The object to check if is a number.\n\n    Returns\n    -------\n    is_number : bool\n        Whether `obj` is a number or not.\n\n    See Also\n    --------\n    api.types.is_integer: Checks a subgroup of numbers.\n\n    Examples\n    --------\n    >>> pd.api.types.is_number(1)\n    True\n    >>> pd.api.types.is_number(7.15)\n    True\n\n    Booleans are valid because they are int subclass.\n\n    >>> pd.api.types.is_number(False)\n    True\n\n    >>> pd.api.types.is_number("foo")\n    False\n    >>> pd.api.types.is_number("5")\n    False\n    '
    return isinstance(obj, (Number, np.number))

def iterable_not_string(obj):
    '\n    Check if the object is an iterable but not a string.\n\n    Parameters\n    ----------\n    obj : The object to check.\n\n    Returns\n    -------\n    is_iter_not_string : bool\n        Whether `obj` is a non-string iterable.\n\n    Examples\n    --------\n    >>> iterable_not_string([1, 2, 3])\n    True\n    >>> iterable_not_string("foo")\n    False\n    >>> iterable_not_string(1)\n    False\n    '
    return (isinstance(obj, abc.Iterable) and (not isinstance(obj, str)))

def is_file_like(obj):
    '\n    Check if the object is a file-like object.\n\n    For objects to be considered file-like, they must\n    be an iterator AND have either a `read` and/or `write`\n    method as an attribute.\n\n    Note: file-like objects must be iterable, but\n    iterable objects need not be file-like.\n\n    Parameters\n    ----------\n    obj : The object to check\n\n    Returns\n    -------\n    is_file_like : bool\n        Whether `obj` has file-like properties.\n\n    Examples\n    --------\n    >>> import io\n    >>> buffer = io.StringIO("data")\n    >>> is_file_like(buffer)\n    True\n    >>> is_file_like([1, 2, 3])\n    False\n    '
    if (not (hasattr(obj, 'read') or hasattr(obj, 'write'))):
        return False
    if (not hasattr(obj, '__iter__')):
        return False
    return True

def is_re(obj):
    '\n    Check if the object is a regex pattern instance.\n\n    Parameters\n    ----------\n    obj : The object to check\n\n    Returns\n    -------\n    is_regex : bool\n        Whether `obj` is a regex pattern.\n\n    Examples\n    --------\n    >>> is_re(re.compile(".*"))\n    True\n    >>> is_re("foo")\n    False\n    '
    return isinstance(obj, Pattern)

def is_re_compilable(obj):
    '\n    Check if the object can be compiled into a regex pattern instance.\n\n    Parameters\n    ----------\n    obj : The object to check\n\n    Returns\n    -------\n    is_regex_compilable : bool\n        Whether `obj` can be compiled as a regex pattern.\n\n    Examples\n    --------\n    >>> is_re_compilable(".*")\n    True\n    >>> is_re_compilable(1)\n    False\n    '
    try:
        re.compile(obj)
    except TypeError:
        return False
    else:
        return True

def is_array_like(obj):
    '\n    Check if the object is array-like.\n\n    For an object to be considered array-like, it must be list-like and\n    have a `dtype` attribute.\n\n    Parameters\n    ----------\n    obj : The object to check\n\n    Returns\n    -------\n    is_array_like : bool\n        Whether `obj` has array-like properties.\n\n    Examples\n    --------\n    >>> is_array_like(np.array([1, 2, 3]))\n    True\n    >>> is_array_like(pd.Series(["a", "b"]))\n    True\n    >>> is_array_like(pd.Index(["2016-01-01"]))\n    True\n    >>> is_array_like([1, 2, 3])\n    False\n    >>> is_array_like(("a", "b"))\n    False\n    '
    return (is_list_like(obj) and hasattr(obj, 'dtype'))

def is_nested_list_like(obj):
    '\n    Check if the object is list-like, and that all of its elements\n    are also list-like.\n\n    Parameters\n    ----------\n    obj : The object to check\n\n    Returns\n    -------\n    is_list_like : bool\n        Whether `obj` has list-like properties.\n\n    Examples\n    --------\n    >>> is_nested_list_like([[1, 2, 3]])\n    True\n    >>> is_nested_list_like([{1, 2, 3}, {1, 2, 3}])\n    True\n    >>> is_nested_list_like(["foo"])\n    False\n    >>> is_nested_list_like([])\n    False\n    >>> is_nested_list_like([[1, 2, 3], 1])\n    False\n\n    Notes\n    -----\n    This won\'t reliably detect whether a consumable iterator (e. g.\n    a generator) is a nested-list-like without consuming the iterator.\n    To avoid consuming it, we always return False if the outer container\n    doesn\'t define `__len__`.\n\n    See Also\n    --------\n    is_list_like\n    '
    return (is_list_like(obj) and hasattr(obj, '__len__') and (len(obj) > 0) and all((is_list_like(item) for item in obj)))

def is_dict_like(obj):
    '\n    Check if the object is dict-like.\n\n    Parameters\n    ----------\n    obj : The object to check\n\n    Returns\n    -------\n    is_dict_like : bool\n        Whether `obj` has dict-like properties.\n\n    Examples\n    --------\n    >>> is_dict_like({1: 2})\n    True\n    >>> is_dict_like([1, 2, 3])\n    False\n    >>> is_dict_like(dict)\n    False\n    >>> is_dict_like(dict())\n    True\n    '
    dict_like_attrs = ('__getitem__', 'keys', '__contains__')
    return (all((hasattr(obj, attr) for attr in dict_like_attrs)) and (not isinstance(obj, type)))

def is_named_tuple(obj):
    '\n    Check if the object is a named tuple.\n\n    Parameters\n    ----------\n    obj : The object to check\n\n    Returns\n    -------\n    is_named_tuple : bool\n        Whether `obj` is a named tuple.\n\n    Examples\n    --------\n    >>> from collections import namedtuple\n    >>> Point = namedtuple("Point", ["x", "y"])\n    >>> p = Point(1, 2)\n    >>>\n    >>> is_named_tuple(p)\n    True\n    >>> is_named_tuple((1, 2))\n    False\n    '
    return (isinstance(obj, tuple) and hasattr(obj, '_fields'))

def is_hashable(obj):
    '\n    Return True if hash(obj) will succeed, False otherwise.\n\n    Some types will pass a test against collections.abc.Hashable but fail when\n    they are actually hashed with hash().\n\n    Distinguish between these and other types by trying the call to hash() and\n    seeing if they raise TypeError.\n\n    Returns\n    -------\n    bool\n\n    Examples\n    --------\n    >>> import collections\n    >>> a = ([],)\n    >>> isinstance(a, collections.abc.Hashable)\n    True\n    >>> is_hashable(a)\n    False\n    '
    try:
        hash(obj)
    except TypeError:
        return False
    else:
        return True

def is_sequence(obj):
    '\n    Check if the object is a sequence of objects.\n    String types are not included as sequences here.\n\n    Parameters\n    ----------\n    obj : The object to check\n\n    Returns\n    -------\n    is_sequence : bool\n        Whether `obj` is a sequence of objects.\n\n    Examples\n    --------\n    >>> l = [1, 2, 3]\n    >>>\n    >>> is_sequence(l)\n    True\n    >>> is_sequence(iter(l))\n    False\n    '
    try:
        iter(obj)
        len(obj)
        return (not isinstance(obj, (str, bytes)))
    except (TypeError, AttributeError):
        return False

def is_dataclass(item):
    '\n    Checks if the object is a data-class instance\n\n    Parameters\n    ----------\n    item : object\n\n    Returns\n    --------\n    is_dataclass : bool\n        True if the item is an instance of a data-class,\n        will return false if you pass the data class itself\n\n    Examples\n    --------\n    >>> from dataclasses import dataclass\n    >>> @dataclass\n    ... class Point:\n    ...     x: int\n    ...     y: int\n\n    >>> is_dataclass(Point)\n    False\n    >>> is_dataclass(Point(0,2))\n    True\n\n    '
    try:
        from dataclasses import is_dataclass
        return (is_dataclass(item) and (not isinstance(item, type)))
    except ImportError:
        return False
