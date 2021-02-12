
from __future__ import annotations
from distutils.version import LooseVersion
from typing import TYPE_CHECKING, Any, Sequence, Type, Union
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.missing import isna
from pandas.api.types import is_array_like, is_bool_dtype, is_integer, is_integer_dtype, is_scalar
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays.base import ExtensionArray
from pandas.core.indexers import check_array_indexer, validate_indices
from pandas.core.missing import get_fill_func
try:
    import pyarrow as pa
except ImportError:
    pa = None
else:
    try:
        import pyarrow.compute as pc
    except ImportError:
        pass
    else:
        ARROW_CMP_FUNCS = {'eq': pc.equal, 'ne': pc.not_equal, 'lt': pc.less, 'gt': pc.greater, 'le': pc.less_equal, 'ge': pc.greater_equal}
if TYPE_CHECKING:
    from pandas import Series

@register_extension_dtype
class ArrowStringDtype(ExtensionDtype):
    '\n    Extension dtype for string data in a ``pyarrow.ChunkedArray``.\n\n    .. versionadded:: 1.2.0\n\n    .. warning::\n\n       ArrowStringDtype is considered experimental. The implementation and\n       parts of the API may change without warning.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    Examples\n    --------\n    >>> from pandas.core.arrays.string_arrow import ArrowStringDtype\n    >>> ArrowStringDtype()\n    ArrowStringDtype\n    '
    name = 'arrow_string'
    na_value = libmissing.NA

    @property
    def type(self):
        return str

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        return ArrowStringArray

    def __hash__(self):
        return hash('ArrowStringDtype')

    def __repr__(self):
        return 'ArrowStringDtype'

    def __from_arrow__(self, array):
        '\n        Construct StringArray from pyarrow Array/ChunkedArray.\n        '
        return ArrowStringArray(array)

    def __eq__(self, other):
        "Check whether 'other' is equal to self.\n\n        By default, 'other' is considered equal if\n        * it's a string matching 'self.name'.\n        * it's an instance of this type.\n\n        Parameters\n        ----------\n        other : Any\n\n        Returns\n        -------\n        bool\n        "
        if isinstance(other, ArrowStringDtype):
            return True
        elif (isinstance(other, str) and (other == 'arrow_string')):
            return True
        else:
            return False

class ArrowStringArray(OpsMixin, ExtensionArray):
    '\n    Extension array for string data in a ``pyarrow.ChunkedArray``.\n\n    .. versionadded:: 1.2.0\n\n    .. warning::\n\n       ArrowStringArray is considered experimental. The implementation and\n       parts of the API may change without warning.\n\n    Parameters\n    ----------\n    values : pyarrow.Array or pyarrow.ChunkedArray\n        The array of data.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    See Also\n    --------\n    array\n        The recommended function for creating a ArrowStringArray.\n    Series.str\n        The string methods are available on Series backed by\n        a ArrowStringArray.\n\n    Notes\n    -----\n    ArrowStringArray returns a BooleanArray for comparison methods.\n\n    Examples\n    --------\n    >>> pd.array([\'This is\', \'some text\', None, \'data.\'], dtype="arrow_string")\n    <ArrowStringArray>\n    [\'This is\', \'some text\', <NA>, \'data.\']\n    Length: 4, dtype: arrow_string\n    '
    _dtype = ArrowStringDtype()

    def __init__(self, values):
        self._chk_pyarrow_available()
        if isinstance(values, pa.Array):
            self._data = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._data = values
        else:
            raise ValueError(f"Unsupported type '{type(values)}' for ArrowStringArray")
        if (not pa.types.is_string(self._data.type)):
            raise ValueError('ArrowStringArray requires a PyArrow (chunked) array of string type')

    @classmethod
    def _chk_pyarrow_available(cls):
        if ((pa is None) or (LooseVersion(pa.__version__) < '1.0.0')):
            msg = 'pyarrow>=1.0.0 is required for PyArrow backed StringArray.'
            raise ImportError(msg)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        cls._chk_pyarrow_available()
        scalars = lib.ensure_string_array(scalars, copy=False)
        return cls(pa.array(scalars, type=pa.string(), from_pandas=True))

    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @property
    def dtype(self):
        "\n        An instance of 'ArrowStringDtype'.\n        "
        return self._dtype

    def __array__(self, dtype=None):
        'Correctly construct numpy arrays when passed to `np.asarray()`.'
        return self.to_numpy(dtype=dtype)

    def __arrow_array__(self, type=None):
        'Convert myself to a pyarrow Array or ChunkedArray.'
        return self._data

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        '\n        Convert to a NumPy ndarray.\n        '
        if (na_value is lib.no_default):
            na_value = self._dtype.na_value
        result = self._data.__array__(dtype=dtype)
        result[isna(result)] = na_value
        return result

    def __len__(self):
        '\n        Length of this array.\n\n        Returns\n        -------\n        length : int\n        '
        return len(self._data)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values)

    @classmethod
    def _concat_same_type(cls, to_concat):
        '\n        Concatenate multiple ArrowStringArray.\n\n        Parameters\n        ----------\n        to_concat : sequence of ArrowStringArray\n\n        Returns\n        -------\n        ArrowStringArray\n        '
        return cls(pa.chunked_array([array for ea in to_concat for array in ea._data.iterchunks()]))

    def __getitem__(self, item):
        "Select a subset of self.\n\n        Parameters\n        ----------\n        item : int, slice, or ndarray\n            * int: The position in 'self' to get.\n            * slice: A slice object, where 'start', 'stop', and 'step' are\n              integers or None\n            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'\n\n        Returns\n        -------\n        item : scalar or ExtensionArray\n\n        Notes\n        -----\n        For scalar ``item``, return a scalar value suitable for the array's\n        type. This should be an instance of ``self.dtype.type``.\n        For slice ``key``, return an instance of ``ExtensionArray``, even\n        if the slice is length 0 or 1.\n        For a boolean mask, return an instance of ``ExtensionArray``, filtered\n        to the values where ``item`` is True.\n        "
        item = check_array_indexer(self, item)
        if isinstance(item, np.ndarray):
            if (not len(item)):
                return type(self)(pa.chunked_array([], type=pa.string()))
            elif is_integer_dtype(item.dtype):
                return self.take(item)
            elif is_bool_dtype(item.dtype):
                return type(self)(self._data.filter(item))
            else:
                raise IndexError('Only integers, slices and integer or boolean arrays are valid indices.')
        value = self._data[item]
        if isinstance(value, pa.ChunkedArray):
            return type(self)(value)
        else:
            return self._as_pandas_scalar(value)

    def _as_pandas_scalar(self, arrow_scalar):
        scalar = arrow_scalar.as_py()
        if (scalar is None):
            return self._dtype.na_value
        else:
            return scalar

    def fillna(self, value=None, method=None, limit=None):
        "\n        Fill NA/NaN values using the specified method.\n\n        Parameters\n        ----------\n        value : scalar, array-like\n            If a scalar value is passed it is used to fill all missing values.\n            Alternatively, an array-like 'value' can be given. It's expected\n            that the array-like have the same length as 'self'.\n        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None\n            Method to use for filling holes in reindexed Series\n            pad / ffill: propagate last valid observation forward to next valid\n            backfill / bfill: use NEXT valid observation to fill gap.\n        limit : int, default None\n            If method is specified, this is the maximum number of consecutive\n            NaN values to forward/backward fill. In other words, if there is\n            a gap with more than this number of consecutive NaNs, it will only\n            be partially filled. If method is not specified, this is the\n            maximum number of entries along the entire axis where NaNs will be\n            filled.\n\n        Returns\n        -------\n        ExtensionArray\n            With NA/NaN filled.\n        "
        (value, method) = validate_fillna_kwargs(value, method)
        mask = self.isna()
        if is_array_like(value):
            if (len(value) != len(self)):
                raise ValueError(f"Length of 'value' does not match. Got ({len(value)}) expected {len(self)}")
            value = value[mask]
        if mask.any():
            if (method is not None):
                func = get_fill_func(method)
                new_values = func(self.to_numpy(object), limit=limit, mask=mask)
                new_values = self._from_sequence(new_values)
            else:
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    def _reduce(self, name, skipna=True, **kwargs):
        if (name in ['min', 'max']):
            return getattr(self, name)(skipna=skipna)
        raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

    @property
    def nbytes(self):
        '\n        The number of bytes needed to store this object in memory.\n        '
        return self._data.nbytes

    def isna(self):
        "\n        Boolean NumPy array indicating if each value is missing.\n\n        This should return a 1-D array the same length as 'self'.\n        "
        return self._data.is_null().to_pandas().values

    def copy(self):
        '\n        Return a shallow copy of the array.\n\n        Returns\n        -------\n        ArrowStringArray\n        '
        return type(self)(self._data)

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray
        pc_func = ARROW_CMP_FUNCS[op.__name__]
        if isinstance(other, ArrowStringArray):
            result = pc_func(self._data, other._data)
        elif isinstance(other, np.ndarray):
            result = pc_func(self._data, other)
        elif is_scalar(other):
            try:
                result = pc_func(self._data, pa.scalar(other))
            except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid):
                mask = (isna(self) | isna(other))
                valid = (~ mask)
                result = np.zeros(len(self), dtype='bool')
                result[valid] = op(np.array(self)[valid], other)
                return BooleanArray(result, mask)
        else:
            return NotImplemented
        return BooleanArray._from_sequence(result.to_pandas().values)

    def __setitem__(self, key, value):
        'Set one or more values inplace.\n\n        Parameters\n        ----------\n        key : int, ndarray, or slice\n            When called from, e.g. ``Series.__setitem__``, ``key`` will be\n            one of\n\n            * scalar int\n            * ndarray of integers.\n            * boolean ndarray\n            * slice object\n\n        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object\n            value or values to be set of ``key``.\n\n        Returns\n        -------\n        None\n        '
        key = check_array_indexer(self, key)
        if is_integer(key):
            if (not is_scalar(value)):
                raise ValueError('Must pass scalars with scalar indexer')
            elif isna(value):
                value = None
            elif (not isinstance(value, str)):
                raise ValueError('Scalar must be NA or str')
            new_data = [*self._data[0:key].chunks, pa.array([value], type=pa.string()), *self._data[(key + 1):].chunks]
            self._data = pa.chunked_array(new_data)
        else:
            if is_bool_dtype(key):
                key_array = np.argwhere(key).flatten()
            elif isinstance(key, slice):
                key_array = np.array(range(len(self))[key])
            else:
                key_array = np.asanyarray(key)
            if is_scalar(value):
                value = np.broadcast_to(value, len(key_array))
            else:
                value = np.asarray(value)
            if (len(key_array) != len(value)):
                raise ValueError('Length of indexer and values mismatch')
            for (k, v) in zip(key_array, value):
                self[k] = v

    def take(self, indices, allow_fill=False, fill_value=None):
        '\n        Take elements from an array.\n\n        Parameters\n        ----------\n        indices : sequence of int\n            Indices to be taken.\n        allow_fill : bool, default False\n            How to handle negative values in `indices`.\n\n            * False: negative values in `indices` indicate positional indices\n              from the right (the default). This is similar to\n              :func:`numpy.take`.\n\n            * True: negative values in `indices` indicate\n              missing values. These values are set to `fill_value`. Any other\n              other negative values raise a ``ValueError``.\n\n        fill_value : any, optional\n            Fill value to use for NA-indices when `allow_fill` is True.\n            This may be ``None``, in which case the default NA value for\n            the type, ``self.dtype.na_value``, is used.\n\n            For many ExtensionArrays, there will be two representations of\n            `fill_value`: a user-facing "boxed" scalar, and a low-level\n            physical NA value. `fill_value` should be the user-facing version,\n            and the implementation should handle translating that to the\n            physical version for processing the take if necessary.\n\n        Returns\n        -------\n        ExtensionArray\n\n        Raises\n        ------\n        IndexError\n            When the indices are out of bounds for the array.\n        ValueError\n            When `indices` contains negative values other than ``-1``\n            and `allow_fill` is True.\n\n        See Also\n        --------\n        numpy.take\n        api.extensions.take\n\n        Notes\n        -----\n        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,\n        ``iloc``, when `indices` is a sequence of values. Additionally,\n        it\'s called by :meth:`Series.reindex`, or any other method\n        that causes realignment, with a `fill_value`.\n        '
        if (not is_array_like(indices)):
            indices_array = np.asanyarray(indices)
        else:
            indices_array = indices
        if ((len(self._data) == 0) and (indices_array >= 0).any()):
            raise IndexError('cannot do a non-empty take')
        if ((indices_array.size > 0) and (indices_array.max() >= len(self._data))):
            raise IndexError("out of bounds value in 'indices'.")
        if allow_fill:
            fill_mask = (indices_array < 0)
            if fill_mask.any():
                validate_indices(indices_array, len(self._data))
                indices_array = pa.array(indices_array, mask=fill_mask)
                result = self._data.take(indices_array)
                if isna(fill_value):
                    return type(self)(result)
                result = type(self)(result)
                result[fill_mask] = fill_value
                return result
            else:
                return type(self)(self._data.take(indices))
        else:
            if (indices_array < 0).any():
                indices_array = np.copy(indices_array)
                indices_array[(indices_array < 0)] += len(self._data)
            return type(self)(self._data.take(indices_array))

    def value_counts(self, dropna=True):
        "\n        Return a Series containing counts of each unique value.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include counts of missing values.\n\n        Returns\n        -------\n        counts : Series\n\n        See Also\n        --------\n        Series.value_counts\n        "
        from pandas import Index, Series
        vc = self._data.value_counts()
        index = Index(type(self)(vc.field(0)).astype(object))
        counts = np.array(vc.field(1))
        if (dropna and (self._data.null_count > 0)):
            raise NotImplementedError('yo')
        return Series(counts, index=index).astype('Int64')
