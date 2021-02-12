
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Type, TypeVar, Union
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas._typing import ArrayLike, Dtype, Scalar
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import is_dtype_equal, is_integer, is_object_dtype, is_scalar, is_string_dtype, pandas_dtype
from pandas.core.dtypes.missing import isna, notna
from pandas.core import nanops
from pandas.core.algorithms import factorize_array, take
from pandas.core.array_algos import masked_reductions
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.indexers import check_array_indexer
if TYPE_CHECKING:
    from pandas import Series
BaseMaskedArrayT = TypeVar('BaseMaskedArrayT', bound='BaseMaskedArray')

class BaseMaskedDtype(ExtensionDtype):
    '\n    Base class for dtypes for BasedMaskedArray subclasses.\n    '
    base = None
    na_value = libmissing.NA

    @cache_readonly
    def numpy_dtype(self):
        ' Return an instance of our numpy dtype '
        return np.dtype(self.type)

    @cache_readonly
    def kind(self):
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self):
        ' Return the number of bytes in this dtype '
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        raise NotImplementedError

class BaseMaskedArray(OpsMixin, ExtensionArray):
    '\n    Base class for masked arrays (which use _data and _mask to store the data).\n\n    numpy based\n    '

    def __init__(self, values, mask, copy=False):
        if (not (isinstance(mask, np.ndarray) and (mask.dtype == np.bool_))):
            raise TypeError("mask should be boolean numpy array. Use the 'pd.array' function instead")
        if (values.ndim != 1):
            raise ValueError('values must be a 1D array')
        if (mask.ndim != 1):
            raise ValueError('mask must be a 1D array')
        if copy:
            values = values.copy()
            mask = mask.copy()
        self._data = values
        self._mask = mask

    @property
    def dtype(self):
        raise AbstractMethodError(self)

    def __getitem__(self, item):
        if is_integer(item):
            if self._mask[item]:
                return self.dtype.na_value
            return self._data[item]
        item = check_array_indexer(self, item)
        return type(self)(self._data[item], self._mask[item])

    def _coerce_to_array(self, values):
        raise AbstractMethodError(self)

    def __setitem__(self, key, value):
        _is_scalar = is_scalar(value)
        if _is_scalar:
            value = [value]
        (value, mask) = self._coerce_to_array(value)
        if _is_scalar:
            value = value[0]
            mask = mask[0]
        key = check_array_indexer(self, key)
        self._data[key] = value
        self._mask[key] = mask

    def __iter__(self):
        for i in range(len(self)):
            if self._mask[i]:
                (yield self.dtype.na_value)
            else:
                (yield self._data[i])

    def __len__(self):
        return len(self._data)

    def __invert__(self):
        return type(self)((~ self._data), self._mask)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        '\n        Convert to a NumPy Array.\n\n        By default converts to an object-dtype NumPy array. Specify the `dtype` and\n        `na_value` keywords to customize the conversion.\n\n        Parameters\n        ----------\n        dtype : dtype, default object\n            The numpy dtype to convert to.\n        copy : bool, default False\n            Whether to ensure that the returned value is a not a view on\n            the array. Note that ``copy=False`` does not *ensure* that\n            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that\n            a copy is made, even if not strictly necessary. This is typically\n            only possible when no missing values are present and `dtype`\n            is the equivalent numpy dtype.\n        na_value : scalar, optional\n             Scalar missing value indicator to use in numpy array. Defaults\n             to the native missing value indicator of this array (pd.NA).\n\n        Returns\n        -------\n        numpy.ndarray\n\n        Examples\n        --------\n        An object-dtype is the default result\n\n        >>> a = pd.array([True, False, pd.NA], dtype="boolean")\n        >>> a.to_numpy()\n        array([True, False, <NA>], dtype=object)\n\n        When no missing values are present, an equivalent dtype can be used.\n\n        >>> pd.array([True, False], dtype="boolean").to_numpy(dtype="bool")\n        array([ True, False])\n        >>> pd.array([1, 2], dtype="Int64").to_numpy("int64")\n        array([1, 2])\n\n        However, requesting such dtype will raise a ValueError if\n        missing values are present and the default missing value :attr:`NA`\n        is used.\n\n        >>> a = pd.array([True, False, pd.NA], dtype="boolean")\n        >>> a\n        <BooleanArray>\n        [True, False, <NA>]\n        Length: 3, dtype: boolean\n\n        >>> a.to_numpy(dtype="bool")\n        Traceback (most recent call last):\n        ...\n        ValueError: cannot convert to bool numpy array in presence of missing values\n\n        Specify a valid `na_value` instead\n\n        >>> a.to_numpy(dtype="bool", na_value=False)\n        array([ True, False, False])\n        '
        if (na_value is lib.no_default):
            na_value = libmissing.NA
        if (dtype is None):
            dtype = object
        if self._hasna:
            if ((not is_object_dtype(dtype)) and (not is_string_dtype(dtype)) and (na_value is libmissing.NA)):
                raise ValueError(f"cannot convert to '{dtype}'-dtype NumPy array with missing values. Specify an appropriate 'na_value' for this dtype.")
            data = self._data.astype(dtype)
            data[self._mask] = na_value
        else:
            data = self._data.astype(dtype, copy=copy)
        return data

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, self.dtype):
            if copy:
                return self.copy()
            return self
        if isinstance(dtype, BaseMaskedDtype):
            data = self._data.astype(dtype.numpy_dtype, copy=copy)
            mask = (self._mask if (data is self._data) else self._mask.copy())
            cls = dtype.construct_array_type()
            return cls(data, mask, copy=False)
        if isinstance(dtype, ExtensionDtype):
            eacls = dtype.construct_array_type()
            return eacls._from_sequence(self, dtype=dtype, copy=copy)
        raise NotImplementedError('subclass must implement astype to np.dtype')
    __array_priority__ = 1000

    def __array__(self, dtype=None):
        '\n        the array interface, return my values\n        We return an object array here to preserve our scalar values\n        '
        return self.to_numpy(dtype=dtype)

    def __arrow_array__(self, type=None):
        '\n        Convert myself into a pyarrow Array.\n        '
        import pyarrow as pa
        return pa.array(self._data, mask=self._mask, type=type)

    @property
    def _hasna(self):
        return self._mask.any()

    def isna(self):
        return self._mask

    @property
    def _na_value(self):
        return self.dtype.na_value

    @property
    def nbytes(self):
        return (self._data.nbytes + self._mask.nbytes)

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = np.concatenate([x._data for x in to_concat])
        mask = np.concatenate([x._mask for x in to_concat])
        return cls(data, mask)

    def take(self, indexer, *, allow_fill=False, fill_value=None):
        data_fill_value = (self._internal_fill_value if isna(fill_value) else fill_value)
        result = take(self._data, indexer, fill_value=data_fill_value, allow_fill=allow_fill)
        mask = take(self._mask, indexer, fill_value=True, allow_fill=allow_fill)
        if (allow_fill and notna(fill_value)):
            fill_mask = (np.asarray(indexer) == (- 1))
            result[fill_mask] = fill_value
            mask = (mask ^ fill_mask)
        return type(self)(result, mask, copy=False)

    def copy(self):
        (data, mask) = (self._data, self._mask)
        data = data.copy()
        mask = mask.copy()
        return type(self)(data, mask, copy=False)

    @doc(ExtensionArray.factorize)
    def factorize(self, na_sentinel=(- 1)):
        arr = self._data
        mask = self._mask
        (codes, uniques) = factorize_array(arr, na_sentinel=na_sentinel, mask=mask)
        uniques = uniques.astype(self.dtype.numpy_dtype, copy=False)
        uniques = type(self)(uniques, np.zeros(len(uniques), dtype=bool))
        return (codes, uniques)

    def value_counts(self, dropna=True):
        "\n        Returns a Series containing counts of each unique value.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include counts of missing values.\n\n        Returns\n        -------\n        counts : Series\n\n        See Also\n        --------\n        Series.value_counts\n        "
        from pandas import Index, Series
        from pandas.arrays import IntegerArray
        data = self._data[(~ self._mask)]
        value_counts = Index(data).value_counts()
        index = value_counts.index._values.astype(object)
        if dropna:
            counts = value_counts._values
        else:
            counts = np.empty((len(value_counts) + 1), dtype='int64')
            counts[:(- 1)] = value_counts
            counts[(- 1)] = self._mask.sum()
            index = Index(np.concatenate([index, np.array([self.dtype.na_value], dtype=object)]), dtype=object)
        mask = np.zeros(len(counts), dtype='bool')
        counts = IntegerArray(counts, mask)
        return Series(counts, index=index)

    def _reduce(self, name, *, skipna=True, **kwargs):
        data = self._data
        mask = self._mask
        if (name in {'sum', 'prod', 'min', 'max', 'mean'}):
            op = getattr(masked_reductions, name)
            return op(data, mask, skipna=skipna, **kwargs)
        if self._hasna:
            data = self.to_numpy('float64', na_value=np.nan)
        op = getattr(nanops, ('nan' + name))
        result = op(data, axis=0, skipna=skipna, mask=mask, **kwargs)
        if np.isnan(result):
            return libmissing.NA
        return result
