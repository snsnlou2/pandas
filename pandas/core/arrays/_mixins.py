
from __future__ import annotations
from functools import wraps
from typing import Any, Optional, Sequence, Type, TypeVar, Union
import numpy as np
from pandas._libs import lib
from pandas._typing import Shape
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly, doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.common import is_dtype_equal
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import array_equivalent
from pandas.core import missing
from pandas.core.algorithms import take, unique
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
NDArrayBackedExtensionArrayT = TypeVar('NDArrayBackedExtensionArrayT', bound='NDArrayBackedExtensionArray')

def ravel_compat(meth):
    '\n    Decorator to ravel a 2D array before passing it to a cython operation,\n    then reshape the result to our own shape.\n    '

    @wraps(meth)
    def method(self, *args, **kwargs):
        if (self.ndim == 1):
            return meth(self, *args, **kwargs)
        flags = self._ndarray.flags
        flat = self.ravel('K')
        result = meth(flat, *args, **kwargs)
        order = ('F' if flags.f_contiguous else 'C')
        return result.reshape(self.shape, order=order)
    return method

class NDArrayBackedExtensionArray(ExtensionArray):
    '\n    ExtensionArray that is backed by a single NumPy ndarray.\n    '

    def _from_backing_data(self, arr):
        '\n        Construct a new ExtensionArray `new_array` with `arr` as its _ndarray.\n\n        This should round-trip:\n            self == self._from_backing_data(self._ndarray)\n        '
        raise AbstractMethodError(self)

    def _box_func(self, x):
        '\n        Wrap numpy type in our dtype.type if necessary.\n        '
        return x

    def _validate_scalar(self, value):
        raise AbstractMethodError(self)

    def take(self, indices, *, allow_fill=False, fill_value=None, axis=0):
        if allow_fill:
            fill_value = self._validate_fill_value(fill_value)
        new_data = take(self._ndarray, indices, allow_fill=allow_fill, fill_value=fill_value, axis=axis)
        return self._from_backing_data(new_data)

    def _validate_fill_value(self, fill_value):
        '\n        If a fill_value is passed to `take` convert it to a representation\n        suitable for self._ndarray, raising TypeError if this is not possible.\n\n        Parameters\n        ----------\n        fill_value : object\n\n        Returns\n        -------\n        fill_value : native representation\n\n        Raises\n        ------\n        TypeError\n        '
        raise AbstractMethodError(self)

    @property
    def shape(self):
        return self._ndarray.shape

    def __len__(self):
        return self.shape[0]

    @cache_readonly
    def ndim(self):
        return len(self.shape)

    @cache_readonly
    def size(self):
        return np.prod(self.shape)

    @cache_readonly
    def nbytes(self):
        return self._ndarray.nbytes

    def reshape(self, *args, **kwargs):
        new_data = self._ndarray.reshape(*args, **kwargs)
        return self._from_backing_data(new_data)

    def ravel(self, *args, **kwargs):
        new_data = self._ndarray.ravel(*args, **kwargs)
        return self._from_backing_data(new_data)

    @property
    def T(self):
        new_data = self._ndarray.T
        return self._from_backing_data(new_data)

    def equals(self, other):
        if (type(self) is not type(other)):
            return False
        if (not is_dtype_equal(self.dtype, other.dtype)):
            return False
        return bool(array_equivalent(self._ndarray, other._ndarray))

    def _values_for_argsort(self):
        return self._ndarray

    def copy(self):
        new_data = self._ndarray.copy()
        return self._from_backing_data(new_data)

    def repeat(self, repeats, axis=None):
        '\n        Repeat elements of an array.\n\n        See Also\n        --------\n        numpy.ndarray.repeat\n        '
        nv.validate_repeat((), {'axis': axis})
        new_data = self._ndarray.repeat(repeats, axis=axis)
        return self._from_backing_data(new_data)

    def unique(self):
        new_data = unique(self._ndarray)
        return self._from_backing_data(new_data)

    @classmethod
    @doc(ExtensionArray._concat_same_type)
    def _concat_same_type(cls, to_concat, axis=0):
        dtypes = {str(x.dtype) for x in to_concat}
        if (len(dtypes) != 1):
            raise ValueError('to_concat must have the same dtype (tz)', dtypes)
        new_values = [x._ndarray for x in to_concat]
        new_values = np.concatenate(new_values, axis=axis)
        return to_concat[0]._from_backing_data(new_values)

    @doc(ExtensionArray.searchsorted)
    def searchsorted(self, value, side='left', sorter=None):
        value = self._validate_searchsorted_value(value)
        return self._ndarray.searchsorted(value, side=side, sorter=sorter)

    def _validate_searchsorted_value(self, value):
        return value

    @doc(ExtensionArray.shift)
    def shift(self, periods=1, fill_value=None, axis=0):
        fill_value = self._validate_shift_value(fill_value)
        new_values = shift(self._ndarray, periods, axis, fill_value)
        return self._from_backing_data(new_values)

    def _validate_shift_value(self, fill_value):
        return self._validate_fill_value(fill_value)

    def __setitem__(self, key, value):
        key = check_array_indexer(self, key)
        value = self._validate_setitem_value(value)
        self._ndarray[key] = value

    def _validate_setitem_value(self, value):
        return value

    def __getitem__(self, key):
        if lib.is_integer(key):
            result = self._ndarray[key]
            if (self.ndim == 1):
                return self._box_func(result)
            return self._from_backing_data(result)
        key = extract_array(key, extract_numpy=True)
        key = check_array_indexer(self, key)
        result = self._ndarray[key]
        if lib.is_scalar(result):
            return self._box_func(result)
        result = self._from_backing_data(result)
        return result

    @doc(ExtensionArray.fillna)
    def fillna(self, value=None, method=None, limit=None):
        (value, method) = validate_fillna_kwargs(value, method)
        mask = self.isna()
        if is_array_like(value):
            if (len(value) != len(self)):
                raise ValueError(f"Length of 'value' does not match. Got ({len(value)})  expected {len(self)}")
            value = value[mask]
        if mask.any():
            if (method is not None):
                func = missing.get_fill_func(method)
                new_values = func(self._ndarray.copy(), limit=limit, mask=mask)
                new_values = self._from_backing_data(new_values)
            else:
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    def _reduce(self, name, *, skipna=True, **kwargs):
        meth = getattr(self, name, None)
        if meth:
            return meth(skipna=skipna, **kwargs)
        else:
            msg = f"'{type(self).__name__}' does not implement reduction '{name}'"
            raise TypeError(msg)

    def _wrap_reduction_result(self, axis, result):
        if ((axis is None) or (self.ndim == 1)):
            return self._box_func(result)
        return self._from_backing_data(result)

    def __repr__(self):
        if (self.ndim == 1):
            return super().__repr__()
        from pandas.io.formats.printing import format_object_summary
        lines = [format_object_summary(x, self._formatter(), indent_for_name=False).rstrip(', \n') for x in self]
        data = ',\n'.join(lines)
        class_name = f'<{type(self).__name__}>'
        return f'''{class_name}
[
{data}
]
Shape: {self.shape}, dtype: {self.dtype}'''

    def putmask(self, mask, value):
        '\n        Analogue to np.putmask(self, mask, value)\n\n        Parameters\n        ----------\n        mask : np.ndarray[bool]\n        value : scalar or listlike\n\n        Raises\n        ------\n        TypeError\n            If value cannot be cast to self.dtype.\n        '
        value = self._validate_setitem_value(value)
        np.putmask(self._ndarray, mask, value)

    def where(self, mask, value):
        '\n        Analogue to np.where(mask, self, value)\n\n        Parameters\n        ----------\n        mask : np.ndarray[bool]\n        value : scalar or listlike\n\n        Raises\n        ------\n        TypeError\n            If value cannot be cast to self.dtype.\n        '
        value = self._validate_setitem_value(value)
        res_values = np.where(mask, self._ndarray, value)
        return self._from_backing_data(res_values)

    def delete(self, loc):
        res_values = np.delete(self._ndarray, loc)
        return self._from_backing_data(res_values)
