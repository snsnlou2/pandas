
'\nShared methods for Index subclasses backed by ExtensionArray.\n'
from typing import List, Optional, TypeVar
import numpy as np
from pandas._libs import lib
from pandas._typing import Label
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import is_dtype_equal, is_object_dtype, pandas_dtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.indexers import deprecate_ndim_indexing
from pandas.core.indexes.base import Index
from pandas.core.ops import get_op_result_name
_T = TypeVar('_T', bound='NDArrayBackedExtensionIndex')

def inherit_from_data(name, delegate, cache=False, wrap=False):
    '\n    Make an alias for a method of the underlying ExtensionArray.\n\n    Parameters\n    ----------\n    name : str\n        Name of an attribute the class should inherit from its EA parent.\n    delegate : class\n    cache : bool, default False\n        Whether to convert wrapped properties into cache_readonly\n    wrap : bool, default False\n        Whether to wrap the inherited result in an Index.\n\n    Returns\n    -------\n    attribute, method, property, or cache_readonly\n    '
    attr = getattr(delegate, name)
    if isinstance(attr, property):
        if cache:

            def cached(self):
                return getattr(self._data, name)
            cached.__name__ = name
            cached.__doc__ = attr.__doc__
            method = cache_readonly(cached)
        else:

            def fget(self):
                result = getattr(self._data, name)
                if wrap:
                    if isinstance(result, type(self._data)):
                        return type(self)._simple_new(result, name=self.name)
                    elif isinstance(result, ABCDataFrame):
                        return result.set_index(self)
                    return Index(result, name=self.name)
                return result

            def fset(self, value):
                setattr(self._data, name, value)
            fget.__name__ = name
            fget.__doc__ = attr.__doc__
            method = property(fget, fset)
    elif (not callable(attr)):
        method = attr
    else:

        def method(self, *args, **kwargs):
            result = attr(self._data, *args, **kwargs)
            if wrap:
                if isinstance(result, type(self._data)):
                    return type(self)._simple_new(result, name=self.name)
                elif isinstance(result, ABCDataFrame):
                    return result.set_index(self)
                return Index(result, name=self.name)
            return result
        method.__name__ = name
        method.__doc__ = attr.__doc__
    return method

def inherit_names(names, delegate, cache=False, wrap=False):
    '\n    Class decorator to pin attributes from an ExtensionArray to a Index subclass.\n\n    Parameters\n    ----------\n    names : List[str]\n    delegate : class\n    cache : bool, default False\n    wrap : bool, default False\n        Whether to wrap the inherited result in an Index.\n    '

    def wrapper(cls):
        for name in names:
            meth = inherit_from_data(name, delegate, cache=cache, wrap=wrap)
            setattr(cls, name, meth)
        return cls
    return wrapper

def _make_wrapped_comparison_op(opname):
    '\n    Create a comparison method that dispatches to ``._data``.\n    '

    def wrapper(self, other):
        if isinstance(other, ABCSeries):
            other = other._values
        other = _maybe_unwrap_index(other)
        op = getattr(self._data, opname)
        return op(other)
    wrapper.__name__ = opname
    return wrapper

def make_wrapped_arith_op(opname):

    def method(self, other):
        if (isinstance(other, Index) and is_object_dtype(other.dtype) and (type(other) is not Index)):
            return NotImplemented
        meth = getattr(self._data, opname)
        result = meth(_maybe_unwrap_index(other))
        return _wrap_arithmetic_op(self, other, result)
    method.__name__ = opname
    return method

def _wrap_arithmetic_op(self, other, result):
    if (result is NotImplemented):
        return NotImplemented
    if isinstance(result, tuple):
        assert (len(result) == 2)
        return (_wrap_arithmetic_op(self, other, result[0]), _wrap_arithmetic_op(self, other, result[1]))
    if (not isinstance(result, Index)):
        result = Index(result)
    res_name = get_op_result_name(self, other)
    result.name = res_name
    return result

def _maybe_unwrap_index(obj):
    '\n    If operating against another Index object, we need to unwrap the underlying\n    data before deferring to the DatetimeArray/TimedeltaArray/PeriodArray\n    implementation, otherwise we will incorrectly return NotImplemented.\n\n    Parameters\n    ----------\n    obj : object\n\n    Returns\n    -------\n    unwrapped object\n    '
    if isinstance(obj, Index):
        return obj._data
    return obj

class ExtensionIndex(Index):
    '\n    Index subclass for indexes backed by ExtensionArray.\n    '
    __eq__ = _make_wrapped_comparison_op('__eq__')
    __ne__ = _make_wrapped_comparison_op('__ne__')
    __lt__ = _make_wrapped_comparison_op('__lt__')
    __gt__ = _make_wrapped_comparison_op('__gt__')
    __le__ = _make_wrapped_comparison_op('__le__')
    __ge__ = _make_wrapped_comparison_op('__ge__')

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values=None, name=lib.no_default):
        name = (self.name if (name is lib.no_default) else name)
        if (values is not None):
            return self._simple_new(values, name=name)
        result = self._simple_new(self._data, name=name)
        result._cache = self._cache
        return result

    @property
    def _has_complex_internals(self):
        return True

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, type(self._data)):
            if (result.ndim == 1):
                return type(self)(result, name=self.name)
            result = result._data
        deprecate_ndim_indexing(result)
        return result

    def searchsorted(self, value, side='left', sorter=None):
        return self._data.searchsorted(value, side=side, sorter=sorter)

    def _get_engine_target(self):
        return np.asarray(self._data)

    def repeat(self, repeats, axis=None):
        nv.validate_repeat((), {'axis': axis})
        result = self._data.repeat(repeats, axis=axis)
        return type(self)._simple_new(result, name=self.name)

    def insert(self, loc, item):
        raise AbstractMethodError(self)

    def _get_unique_index(self, dropna=False):
        if (self.is_unique and (not dropna)):
            return self
        result = self._data.unique()
        if (dropna and self.hasnans):
            result = result[(~ result.isna())]
        return self._shallow_copy(result)

    @doc(Index.map)
    def map(self, mapper, na_action=None):
        try:
            result = mapper(self)
            if isinstance(result, np.ndarray):
                result = Index(result)
            if (not isinstance(result, Index)):
                raise TypeError('The map function must return an Index object')
            return result
        except Exception:
            return self.astype(object).map(mapper)

    @doc(Index.astype)
    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(self.dtype, dtype):
            if (not copy):
                return self
            return self.copy()
        new_values = self._data.astype(dtype, copy=copy)
        return Index(new_values, dtype=new_values.dtype, name=self.name, copy=False)

    @cache_readonly
    def _isnan(self):
        return self._data.isna()

    @doc(Index.equals)
    def equals(self, other):
        if self.is_(other):
            return True
        if (not isinstance(other, type(self))):
            return False
        return self._data.equals(other._data)

class NDArrayBackedExtensionIndex(ExtensionIndex):
    '\n    Index subclass for indexes backed by NDArrayBackedExtensionArray.\n    '

    def _get_engine_target(self):
        return self._data._ndarray

    def delete(self, loc):
        '\n        Make new Index with passed location(-s) deleted\n\n        Returns\n        -------\n        new_index : Index\n        '
        arr = self._data.delete(loc)
        return type(self)._simple_new(arr, name=self.name)

    def insert(self, loc, item):
        '\n        Make new Index inserting new item at location. Follows\n        Python list.append semantics for negative values.\n\n        Parameters\n        ----------\n        loc : int\n        item : object\n\n        Returns\n        -------\n        new_index : Index\n\n        Raises\n        ------\n        ValueError if the item is not valid for this dtype.\n        '
        arr = self._data
        code = arr._validate_scalar(item)
        new_vals = np.concatenate((arr._ndarray[:loc], [code], arr._ndarray[loc:]))
        new_arr = arr._from_backing_data(new_vals)
        return type(self)._simple_new(new_arr, name=self.name)

    @doc(Index.where)
    def where(self, cond, other=None):
        res_values = self._data.where(cond, other)
        return type(self)._simple_new(res_values, name=self.name)

    def putmask(self, mask, value):
        res_values = self._data.copy()
        try:
            res_values.putmask(mask, value)
        except (TypeError, ValueError):
            return self.astype(object).putmask(mask, value)
        return type(self)._simple_new(res_values, name=self.name)

    def _wrap_joined_index(self, joined, other):
        name = get_op_result_name(self, other)
        arr = self._data._from_backing_data(joined)
        return type(self)._simple_new(arr, name=name)
