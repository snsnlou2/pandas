
'\nAn interface for extending pandas with custom arrays.\n\n.. warning::\n\n   This is an experimental API and subject to breaking changes\n   without warning.\n'
from __future__ import annotations
import operator
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union, cast
import numpy as np
from pandas._libs import lib
from pandas._typing import ArrayLike, Dtype, Shape
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import Appender, Substitution
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.cast import maybe_cast_to_extension_array
from pandas.core.dtypes.common import is_array_like, is_dtype_equal, is_list_like, is_scalar, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import ops
from pandas.core.algorithms import factorize_array, unique
from pandas.core.missing import get_fill_func
from pandas.core.sorting import nargminmax, nargsort
_extension_array_shared_docs = {}
ExtensionArrayT = TypeVar('ExtensionArrayT', bound='ExtensionArray')

class ExtensionArray():
    "\n    Abstract base class for custom 1-D array types.\n\n    pandas will recognize instances of this class as proper arrays\n    with a custom type and will not attempt to coerce them to objects. They\n    may be stored directly inside a :class:`DataFrame` or :class:`Series`.\n\n    Attributes\n    ----------\n    dtype\n    nbytes\n    ndim\n    shape\n\n    Methods\n    -------\n    argsort\n    astype\n    copy\n    dropna\n    factorize\n    fillna\n    equals\n    isna\n    ravel\n    repeat\n    searchsorted\n    shift\n    take\n    unique\n    view\n    _concat_same_type\n    _formatter\n    _from_factorized\n    _from_sequence\n    _from_sequence_of_strings\n    _reduce\n    _values_for_argsort\n    _values_for_factorize\n\n    Notes\n    -----\n    The interface includes the following abstract methods that must be\n    implemented by subclasses:\n\n    * _from_sequence\n    * _from_factorized\n    * __getitem__\n    * __len__\n    * __eq__\n    * dtype\n    * nbytes\n    * isna\n    * take\n    * copy\n    * _concat_same_type\n\n    A default repr displaying the type, (truncated) data, length,\n    and dtype is provided. It can be customized or replaced by\n    by overriding:\n\n    * __repr__ : A default repr for the ExtensionArray.\n    * _formatter : Print scalars inside a Series or DataFrame.\n\n    Some methods require casting the ExtensionArray to an ndarray of Python\n    objects with ``self.astype(object)``, which may be expensive. When\n    performance is a concern, we highly recommend overriding the following\n    methods:\n\n    * fillna\n    * dropna\n    * unique\n    * factorize / _values_for_factorize\n    * argsort / _values_for_argsort\n    * searchsorted\n\n    The remaining methods implemented on this class should be performant,\n    as they only compose abstract methods. Still, a more efficient\n    implementation may be available, and these methods can be overridden.\n\n    One can implement methods to handle array reductions.\n\n    * _reduce\n\n    One can implement methods to handle parsing from strings that will be used\n    in methods such as ``pandas.io.parsers.read_csv``.\n\n    * _from_sequence_of_strings\n\n    This class does not inherit from 'abc.ABCMeta' for performance reasons.\n    Methods and properties required by the interface raise\n    ``pandas.errors.AbstractMethodError`` and no ``register`` method is\n    provided for registering virtual subclasses.\n\n    ExtensionArrays are limited to 1 dimension.\n\n    They may be backed by none, one, or many NumPy arrays. For example,\n    ``pandas.Categorical`` is an extension array backed by two arrays,\n    one for codes and one for categories. An array of IPv6 address may\n    be backed by a NumPy structured array with two fields, one for the\n    lower 64 bits and one for the upper 64 bits. Or they may be backed\n    by some other storage type, like Python lists. Pandas makes no\n    assumptions on how the data are stored, just that it can be converted\n    to a NumPy array.\n    The ExtensionArray interface does not impose any rules on how this data\n    is stored. However, currently, the backing data cannot be stored in\n    attributes called ``.values`` or ``._values`` to ensure full compatibility\n    with pandas internals. But other names as ``.data``, ``._data``,\n    ``._items``, ... can be freely used.\n\n    If implementing NumPy's ``__array_ufunc__`` interface, pandas expects\n    that\n\n    1. You defer by returning ``NotImplemented`` when any Series are present\n       in `inputs`. Pandas will extract the arrays and call the ufunc again.\n    2. You define a ``_HANDLED_TYPES`` tuple as an attribute on the class.\n       Pandas inspect this to determine whether the ufunc is valid for the\n       types present.\n\n    See :ref:`extending.extension.ufunc` for more.\n\n    By default, ExtensionArrays are not hashable.  Immutable subclasses may\n    override this behavior.\n    "
    _typ = 'extension'

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        '\n        Construct a new ExtensionArray from a sequence of scalars.\n\n        Parameters\n        ----------\n        scalars : Sequence\n            Each element will be an instance of the scalar type for this\n            array, ``cls.dtype.type`` or be converted into this type in this method.\n        dtype : dtype, optional\n            Construct for this particular dtype. This should be a Dtype\n            compatible with the ExtensionArray.\n        copy : bool, default False\n            If True, copy the underlying data.\n\n        Returns\n        -------\n        ExtensionArray\n        '
        raise AbstractMethodError(cls)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        '\n        Construct a new ExtensionArray from a sequence of strings.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        strings : Sequence\n            Each element will be an instance of the scalar type for this\n            array, ``cls.dtype.type``.\n        dtype : dtype, optional\n            Construct for this particular dtype. This should be a Dtype\n            compatible with the ExtensionArray.\n        copy : bool, default False\n            If True, copy the underlying data.\n\n        Returns\n        -------\n        ExtensionArray\n        '
        raise AbstractMethodError(cls)

    @classmethod
    def _from_factorized(cls, values, original):
        '\n        Reconstruct an ExtensionArray after factorization.\n\n        Parameters\n        ----------\n        values : ndarray\n            An integer ndarray with the factorized values.\n        original : ExtensionArray\n            The original ExtensionArray that factorize was called on.\n\n        See Also\n        --------\n        factorize : Top-level factorize method that dispatches here.\n        ExtensionArray.factorize : Encode the extension array as an enumerated type.\n        '
        raise AbstractMethodError(cls)

    def __getitem__(self, item):
        "\n        Select a subset of self.\n\n        Parameters\n        ----------\n        item : int, slice, or ndarray\n            * int: The position in 'self' to get.\n\n            * slice: A slice object, where 'start', 'stop', and 'step' are\n              integers or None\n\n            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'\n\n        Returns\n        -------\n        item : scalar or ExtensionArray\n\n        Notes\n        -----\n        For scalar ``item``, return a scalar value suitable for the array's\n        type. This should be an instance of ``self.dtype.type``.\n\n        For slice ``key``, return an instance of ``ExtensionArray``, even\n        if the slice is length 0 or 1.\n\n        For a boolean mask, return an instance of ``ExtensionArray``, filtered\n        to the values where ``item`` is True.\n        "
        raise AbstractMethodError(self)

    def __setitem__(self, key, value):
        '\n        Set one or more values inplace.\n\n        This method is not required to satisfy the pandas extension array\n        interface.\n\n        Parameters\n        ----------\n        key : int, ndarray, or slice\n            When called from, e.g. ``Series.__setitem__``, ``key`` will be\n            one of\n\n            * scalar int\n            * ndarray of integers.\n            * boolean ndarray\n            * slice object\n\n        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object\n            value or values to be set of ``key``.\n\n        Returns\n        -------\n        None\n        '
        raise NotImplementedError(f'{type(self)} does not implement __setitem__.')

    def __len__(self):
        '\n        Length of this array\n\n        Returns\n        -------\n        length : int\n        '
        raise AbstractMethodError(self)

    def __iter__(self):
        '\n        Iterate over elements of the array.\n        '
        for i in range(len(self)):
            (yield self[i])

    def __contains__(self, item):
        '\n        Return for `item in self`.\n        '
        if (is_scalar(item) and isna(item)):
            if (not self._can_hold_na):
                return False
            elif ((item is self.dtype.na_value) or isinstance(item, self.dtype.type)):
                return self.isna().any()
            else:
                return False
        else:
            return (item == self).any()

    def __eq__(self, other):
        '\n        Return for `self == other` (element-wise equality).\n        '
        raise AbstractMethodError(self)

    def __ne__(self, other):
        '\n        Return for `self != other` (element-wise in-equality).\n        '
        return (~ (self == other))

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        '\n        Convert to a NumPy ndarray.\n\n        .. versionadded:: 1.0.0\n\n        This is similar to :meth:`numpy.asarray`, but may provide additional control\n        over how the conversion is done.\n\n        Parameters\n        ----------\n        dtype : str or numpy.dtype, optional\n            The dtype to pass to :meth:`numpy.asarray`.\n        copy : bool, default False\n            Whether to ensure that the returned value is a not a view on\n            another array. Note that ``copy=False`` does not *ensure* that\n            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that\n            a copy is made, even if not strictly necessary.\n        na_value : Any, optional\n            The value to use for missing values. The default value depends\n            on `dtype` and the type of the array.\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        result = np.asarray(self, dtype=dtype)
        if (copy or (na_value is not lib.no_default)):
            result = result.copy()
        if (na_value is not lib.no_default):
            result[self.isna()] = na_value
        return result

    @property
    def dtype(self):
        "\n        An instance of 'ExtensionDtype'.\n        "
        raise AbstractMethodError(self)

    @property
    def shape(self):
        '\n        Return a tuple of the array dimensions.\n        '
        return (len(self),)

    @property
    def size(self):
        '\n        The number of elements in the array.\n        '
        return np.prod(self.shape)

    @property
    def ndim(self):
        '\n        Extension Arrays are only allowed to be 1-dimensional.\n        '
        return 1

    @property
    def nbytes(self):
        '\n        The number of bytes needed to store this object in memory.\n        '
        raise AbstractMethodError(self)

    def astype(self, dtype, copy=True):
        "\n        Cast to a NumPy array with 'dtype'.\n\n        Parameters\n        ----------\n        dtype : str or dtype\n            Typecode or data-type to which the array is cast.\n        copy : bool, default True\n            Whether to copy the data, even if not necessary. If False,\n            a copy is made only if the old dtype does not match the\n            new dtype.\n\n        Returns\n        -------\n        array : ndarray\n            NumPy ndarray with 'dtype' for its dtype.\n        "
        from pandas.core.arrays.string_ import StringDtype
        from pandas.core.arrays.string_arrow import ArrowStringDtype
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, self.dtype):
            if (not copy):
                return self
            else:
                return self.copy()
        if isinstance(dtype, (ArrowStringDtype, StringDtype)):
            return dtype.construct_array_type()._from_sequence(self, copy=False)
        return np.array(self, dtype=dtype, copy=copy)

    def isna(self):
        '\n        A 1-D array indicating if each value is missing.\n\n        Returns\n        -------\n        na_values : Union[np.ndarray, ExtensionArray]\n            In most cases, this should return a NumPy ndarray. For\n            exceptional cases like ``SparseArray``, where returning\n            an ndarray would be expensive, an ExtensionArray may be\n            returned.\n\n        Notes\n        -----\n        If returning an ExtensionArray, then\n\n        * ``na_values._is_boolean`` should be True\n        * `na_values` should implement :func:`ExtensionArray._reduce`\n        * ``na_values.any`` and ``na_values.all`` should be implemented\n        '
        raise AbstractMethodError(self)

    def _values_for_argsort(self):
        '\n        Return values for sorting.\n\n        Returns\n        -------\n        ndarray\n            The transformed values should maintain the ordering between values\n            within the array.\n\n        See Also\n        --------\n        ExtensionArray.argsort : Return the indices that would sort this array.\n        '
        return np.array(self)

    def argsort(self, ascending=True, kind='quicksort', na_position='last', *args, **kwargs):
        "\n        Return the indices that would sort this array.\n\n        Parameters\n        ----------\n        ascending : bool, default True\n            Whether the indices should result in an ascending\n            or descending sort.\n        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional\n            Sorting algorithm.\n        *args, **kwargs:\n            Passed through to :func:`numpy.argsort`.\n\n        Returns\n        -------\n        ndarray\n            Array of indices that sort ``self``. If NaN values are contained,\n            NaN values are placed at the end.\n\n        See Also\n        --------\n        numpy.argsort : Sorting implementation used internally.\n        "
        ascending = nv.validate_argsort_with_ascending(ascending, args, kwargs)
        values = self._values_for_argsort()
        return nargsort(values, kind=kind, ascending=ascending, na_position=na_position, mask=np.asarray(self.isna()))

    def argmin(self):
        '\n        Return the index of minimum value.\n\n        In case of multiple occurrences of the minimum value, the index\n        corresponding to the first occurrence is returned.\n\n        Returns\n        -------\n        int\n\n        See Also\n        --------\n        ExtensionArray.argmax\n        '
        return nargminmax(self, 'argmin')

    def argmax(self):
        '\n        Return the index of maximum value.\n\n        In case of multiple occurrences of the maximum value, the index\n        corresponding to the first occurrence is returned.\n\n        Returns\n        -------\n        int\n\n        See Also\n        --------\n        ExtensionArray.argmin\n        '
        return nargminmax(self, 'argmax')

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
                new_values = func(self.astype(object), limit=limit, mask=mask)
                new_values = self._from_sequence(new_values, dtype=self.dtype)
            else:
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    def dropna(self):
        '\n        Return ExtensionArray without NA values.\n\n        Returns\n        -------\n        valid : ExtensionArray\n        '
        return self[(~ self.isna())]

    def shift(self, periods=1, fill_value=None):
        '\n        Shift values by desired number.\n\n        Newly introduced missing values are filled with\n        ``self.dtype.na_value``.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        periods : int, default 1\n            The number of periods to shift. Negative values are allowed\n            for shifting backwards.\n\n        fill_value : object, optional\n            The scalar value to use for newly introduced missing values.\n            The default is ``self.dtype.na_value``.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        ExtensionArray\n            Shifted.\n\n        Notes\n        -----\n        If ``self`` is empty or ``periods`` is 0, a copy of ``self`` is\n        returned.\n\n        If ``periods > len(self)``, then an array of size\n        len(self) is returned, with all values filled with\n        ``self.dtype.na_value``.\n        '
        if ((not len(self)) or (periods == 0)):
            return self.copy()
        if isna(fill_value):
            fill_value = self.dtype.na_value
        empty = self._from_sequence(([fill_value] * min(abs(periods), len(self))), dtype=self.dtype)
        if (periods > 0):
            a = empty
            b = self[:(- periods)]
        else:
            a = self[abs(periods):]
            b = empty
        return self._concat_same_type([a, b])

    def unique(self):
        '\n        Compute the ExtensionArray of unique values.\n\n        Returns\n        -------\n        uniques : ExtensionArray\n        '
        uniques = unique(self.astype(object))
        return self._from_sequence(uniques, dtype=self.dtype)

    def searchsorted(self, value, side='left', sorter=None):
        "\n        Find indices where elements should be inserted to maintain order.\n\n        .. versionadded:: 0.24.0\n\n        Find the indices into a sorted array `self` (a) such that, if the\n        corresponding elements in `value` were inserted before the indices,\n        the order of `self` would be preserved.\n\n        Assuming that `self` is sorted:\n\n        ======  ================================\n        `side`  returned index `i` satisfies\n        ======  ================================\n        left    ``self[i-1] < value <= self[i]``\n        right   ``self[i-1] <= value < self[i]``\n        ======  ================================\n\n        Parameters\n        ----------\n        value : array_like\n            Values to insert into `self`.\n        side : {'left', 'right'}, optional\n            If 'left', the index of the first suitable location found is given.\n            If 'right', return the last such index.  If there is no suitable\n            index, return either 0 or N (where N is the length of `self`).\n        sorter : 1-D array_like, optional\n            Optional array of integer indices that sort array a into ascending\n            order. They are typically the result of argsort.\n\n        Returns\n        -------\n        array of ints\n            Array of insertion points with the same shape as `value`.\n\n        See Also\n        --------\n        numpy.searchsorted : Similar method from NumPy.\n        "
        arr = self.astype(object)
        return arr.searchsorted(value, side=side, sorter=sorter)

    def equals(self, other):
        '\n        Return if another array is equivalent to this array.\n\n        Equivalent means that both arrays have the same shape and dtype, and\n        all values compare equal. Missing values in the same location are\n        considered equal (in contrast with normal equality).\n\n        Parameters\n        ----------\n        other : ExtensionArray\n            Array to compare to this Array.\n\n        Returns\n        -------\n        boolean\n            Whether the arrays are equivalent.\n        '
        if (type(self) != type(other)):
            return False
        other = cast(ExtensionArray, other)
        if (not is_dtype_equal(self.dtype, other.dtype)):
            return False
        elif (len(self) != len(other)):
            return False
        else:
            equal_values = (self == other)
            if isinstance(equal_values, ExtensionArray):
                equal_values = equal_values.fillna(False)
            equal_na = (self.isna() & other.isna())
            return bool((equal_values | equal_na).all())

    def _values_for_factorize(self):
        '\n        Return an array and missing value suitable for factorization.\n\n        Returns\n        -------\n        values : ndarray\n\n            An array suitable for factorization. This should maintain order\n            and be a supported dtype (Float64, Int64, UInt64, String, Object).\n            By default, the extension array is cast to object dtype.\n        na_value : object\n            The value in `values` to consider missing. This will be treated\n            as NA in the factorization routines, so it will be coded as\n            `na_sentinel` and not included in `uniques`. By default,\n            ``np.nan`` is used.\n\n        Notes\n        -----\n        The values returned by this method are also used in\n        :func:`pandas.util.hash_pandas_object`.\n        '
        return (self.astype(object), np.nan)

    def factorize(self, na_sentinel=(- 1)):
        "\n        Encode the extension array as an enumerated type.\n\n        Parameters\n        ----------\n        na_sentinel : int, default -1\n            Value to use in the `codes` array to indicate missing values.\n\n        Returns\n        -------\n        codes : ndarray\n            An integer NumPy array that's an indexer into the original\n            ExtensionArray.\n        uniques : ExtensionArray\n            An ExtensionArray containing the unique values of `self`.\n\n            .. note::\n\n               uniques will *not* contain an entry for the NA value of\n               the ExtensionArray if there are any missing values present\n               in `self`.\n\n        See Also\n        --------\n        factorize : Top-level factorize method that dispatches here.\n\n        Notes\n        -----\n        :meth:`pandas.factorize` offers a `sort` keyword as well.\n        "
        (arr, na_value) = self._values_for_factorize()
        (codes, uniques) = factorize_array(arr, na_sentinel=na_sentinel, na_value=na_value)
        uniques = self._from_factorized(uniques, self)
        return (codes, uniques)
    _extension_array_shared_docs['repeat'] = "\n        Repeat elements of a %(klass)s.\n\n        Returns a new %(klass)s where each element of the current %(klass)s\n        is repeated consecutively a given number of times.\n\n        Parameters\n        ----------\n        repeats : int or array of ints\n            The number of repetitions for each element. This should be a\n            non-negative integer. Repeating 0 times will return an empty\n            %(klass)s.\n        axis : None\n            Must be ``None``. Has no effect but is accepted for compatibility\n            with numpy.\n\n        Returns\n        -------\n        repeated_array : %(klass)s\n            Newly created %(klass)s with repeated elements.\n\n        See Also\n        --------\n        Series.repeat : Equivalent function for Series.\n        Index.repeat : Equivalent function for Index.\n        numpy.repeat : Similar method for :class:`numpy.ndarray`.\n        ExtensionArray.take : Take arbitrary positions.\n\n        Examples\n        --------\n        >>> cat = pd.Categorical(['a', 'b', 'c'])\n        >>> cat\n        ['a', 'b', 'c']\n        Categories (3, object): ['a', 'b', 'c']\n        >>> cat.repeat(2)\n        ['a', 'a', 'b', 'b', 'c', 'c']\n        Categories (3, object): ['a', 'b', 'c']\n        >>> cat.repeat([1, 2, 3])\n        ['a', 'b', 'b', 'c', 'c', 'c']\n        Categories (3, object): ['a', 'b', 'c']\n        "

    @Substitution(klass='ExtensionArray')
    @Appender(_extension_array_shared_docs['repeat'])
    def repeat(self, repeats, axis=None):
        nv.validate_repeat((), {'axis': axis})
        ind = np.arange(len(self)).repeat(repeats)
        return self.take(ind)

    def take(self, indices, *, allow_fill=False, fill_value=None):
        '\n        Take elements from an array.\n\n        Parameters\n        ----------\n        indices : sequence of int\n            Indices to be taken.\n        allow_fill : bool, default False\n            How to handle negative values in `indices`.\n\n            * False: negative values in `indices` indicate positional indices\n              from the right (the default). This is similar to\n              :func:`numpy.take`.\n\n            * True: negative values in `indices` indicate\n              missing values. These values are set to `fill_value`. Any other\n              other negative values raise a ``ValueError``.\n\n        fill_value : any, optional\n            Fill value to use for NA-indices when `allow_fill` is True.\n            This may be ``None``, in which case the default NA value for\n            the type, ``self.dtype.na_value``, is used.\n\n            For many ExtensionArrays, there will be two representations of\n            `fill_value`: a user-facing "boxed" scalar, and a low-level\n            physical NA value. `fill_value` should be the user-facing version,\n            and the implementation should handle translating that to the\n            physical version for processing the take if necessary.\n\n        Returns\n        -------\n        ExtensionArray\n\n        Raises\n        ------\n        IndexError\n            When the indices are out of bounds for the array.\n        ValueError\n            When `indices` contains negative values other than ``-1``\n            and `allow_fill` is True.\n\n        See Also\n        --------\n        numpy.take : Take elements from an array along an axis.\n        api.extensions.take : Take elements from an array.\n\n        Notes\n        -----\n        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,\n        ``iloc``, when `indices` is a sequence of values. Additionally,\n        it\'s called by :meth:`Series.reindex`, or any other method\n        that causes realignment, with a `fill_value`.\n\n        Examples\n        --------\n        Here\'s an example implementation, which relies on casting the\n        extension array to object dtype. This uses the helper method\n        :func:`pandas.api.extensions.take`.\n\n        .. code-block:: python\n\n           def take(self, indices, allow_fill=False, fill_value=None):\n               from pandas.core.algorithms import take\n\n               # If the ExtensionArray is backed by an ndarray, then\n               # just pass that here instead of coercing to object.\n               data = self.astype(object)\n\n               if allow_fill and fill_value is None:\n                   fill_value = self.dtype.na_value\n\n               # fill value should always be translated from the scalar\n               # type for the array, to the physical storage type for\n               # the data, before passing to take.\n\n               result = take(data, indices, fill_value=fill_value,\n                             allow_fill=allow_fill)\n               return self._from_sequence(result, dtype=self.dtype)\n        '
        raise AbstractMethodError(self)

    def copy(self):
        '\n        Return a copy of the array.\n\n        Returns\n        -------\n        ExtensionArray\n        '
        raise AbstractMethodError(self)

    def view(self, dtype=None):
        "\n        Return a view on the array.\n\n        Parameters\n        ----------\n        dtype : str, np.dtype, or ExtensionDtype, optional\n            Default None.\n\n        Returns\n        -------\n        ExtensionArray or np.ndarray\n            A view on the :class:`ExtensionArray`'s data.\n        "
        if (dtype is not None):
            raise NotImplementedError(dtype)
        return self[:]

    def __repr__(self):
        from pandas.io.formats.printing import format_object_summary
        data = format_object_summary(self, self._formatter(), indent_for_name=False).rstrip(', \n')
        class_name = f'''<{type(self).__name__}>
'''
        return f'''{class_name}{data}
Length: {len(self)}, dtype: {self.dtype}'''

    def _formatter(self, boxed=False):
        "\n        Formatting function for scalar values.\n\n        This is used in the default '__repr__'. The returned formatting\n        function receives instances of your scalar type.\n\n        Parameters\n        ----------\n        boxed : bool, default False\n            An indicated for whether or not your array is being printed\n            within a Series, DataFrame, or Index (True), or just by\n            itself (False). This may be useful if you want scalar values\n            to appear differently within a Series versus on its own (e.g.\n            quoted or not).\n\n        Returns\n        -------\n        Callable[[Any], str]\n            A callable that gets instances of the scalar type and\n            returns a string. By default, :func:`repr` is used\n            when ``boxed=False`` and :func:`str` is used when\n            ``boxed=True``.\n        "
        if boxed:
            return str
        return repr

    def transpose(self, *axes):
        '\n        Return a transposed view on this array.\n\n        Because ExtensionArrays are always 1D, this is a no-op.  It is included\n        for compatibility with np.ndarray.\n        '
        return self[:]

    @property
    def T(self):
        return self.transpose()

    def ravel(self, order='C'):
        '\n        Return a flattened view on this array.\n\n        Parameters\n        ----------\n        order : {None, \'C\', \'F\', \'A\', \'K\'}, default \'C\'\n\n        Returns\n        -------\n        ExtensionArray\n\n        Notes\n        -----\n        - Because ExtensionArrays are 1D-only, this is a no-op.\n        - The "order" argument is ignored, is for compatibility with NumPy.\n        '
        return self

    @classmethod
    def _concat_same_type(cls, to_concat):
        '\n        Concatenate multiple array of this dtype.\n\n        Parameters\n        ----------\n        to_concat : sequence of this type\n\n        Returns\n        -------\n        ExtensionArray\n        '
        raise AbstractMethodError(cls)
    _can_hold_na = True

    def _reduce(self, name, *, skipna=True, **kwargs):
        '\n        Return a scalar result of performing the reduction operation.\n\n        Parameters\n        ----------\n        name : str\n            Name of the function, supported values are:\n            { any, all, min, max, sum, mean, median, prod,\n            std, var, sem, kurt, skew }.\n        skipna : bool, default True\n            If True, skip NaN values.\n        **kwargs\n            Additional keyword arguments passed to the reduction function.\n            Currently, `ddof` is the only supported kwarg.\n\n        Returns\n        -------\n        scalar\n\n        Raises\n        ------\n        TypeError : subclass does not define reductions\n        '
        raise TypeError(f'cannot perform {name} with type {self.dtype}')

    def __hash__(self):
        raise TypeError(f'unhashable type: {repr(type(self).__name__)}')

class ExtensionOpsMixin():
    '\n    A base class for linking the operators to their dunder names.\n\n    .. note::\n\n       You may want to set ``__array_priority__`` if you want your\n       implementation to be called when involved in binary operations\n       with NumPy arrays.\n    '

    @classmethod
    def _create_arithmetic_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_arithmetic_ops(cls):
        setattr(cls, '__add__', cls._create_arithmetic_method(operator.add))
        setattr(cls, '__radd__', cls._create_arithmetic_method(ops.radd))
        setattr(cls, '__sub__', cls._create_arithmetic_method(operator.sub))
        setattr(cls, '__rsub__', cls._create_arithmetic_method(ops.rsub))
        setattr(cls, '__mul__', cls._create_arithmetic_method(operator.mul))
        setattr(cls, '__rmul__', cls._create_arithmetic_method(ops.rmul))
        setattr(cls, '__pow__', cls._create_arithmetic_method(operator.pow))
        setattr(cls, '__rpow__', cls._create_arithmetic_method(ops.rpow))
        setattr(cls, '__mod__', cls._create_arithmetic_method(operator.mod))
        setattr(cls, '__rmod__', cls._create_arithmetic_method(ops.rmod))
        setattr(cls, '__floordiv__', cls._create_arithmetic_method(operator.floordiv))
        setattr(cls, '__rfloordiv__', cls._create_arithmetic_method(ops.rfloordiv))
        setattr(cls, '__truediv__', cls._create_arithmetic_method(operator.truediv))
        setattr(cls, '__rtruediv__', cls._create_arithmetic_method(ops.rtruediv))
        setattr(cls, '__divmod__', cls._create_arithmetic_method(divmod))
        setattr(cls, '__rdivmod__', cls._create_arithmetic_method(ops.rdivmod))

    @classmethod
    def _create_comparison_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_comparison_ops(cls):
        setattr(cls, '__eq__', cls._create_comparison_method(operator.eq))
        setattr(cls, '__ne__', cls._create_comparison_method(operator.ne))
        setattr(cls, '__lt__', cls._create_comparison_method(operator.lt))
        setattr(cls, '__gt__', cls._create_comparison_method(operator.gt))
        setattr(cls, '__le__', cls._create_comparison_method(operator.le))
        setattr(cls, '__ge__', cls._create_comparison_method(operator.ge))

    @classmethod
    def _create_logical_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_logical_ops(cls):
        setattr(cls, '__and__', cls._create_logical_method(operator.and_))
        setattr(cls, '__rand__', cls._create_logical_method(ops.rand_))
        setattr(cls, '__or__', cls._create_logical_method(operator.or_))
        setattr(cls, '__ror__', cls._create_logical_method(ops.ror_))
        setattr(cls, '__xor__', cls._create_logical_method(operator.xor))
        setattr(cls, '__rxor__', cls._create_logical_method(ops.rxor))

class ExtensionScalarOpsMixin(ExtensionOpsMixin):
    '\n    A mixin for defining  ops on an ExtensionArray.\n\n    It is assumed that the underlying scalar objects have the operators\n    already defined.\n\n    Notes\n    -----\n    If you have defined a subclass MyExtensionArray(ExtensionArray), then\n    use MyExtensionArray(ExtensionArray, ExtensionScalarOpsMixin) to\n    get the arithmetic operators.  After the definition of MyExtensionArray,\n    insert the lines\n\n    MyExtensionArray._add_arithmetic_ops()\n    MyExtensionArray._add_comparison_ops()\n\n    to link the operators to your class.\n\n    .. note::\n\n       You may want to set ``__array_priority__`` if you want your\n       implementation to be called when involved in binary operations\n       with NumPy arrays.\n    '

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
        "\n        A class method that returns a method that will correspond to an\n        operator for an ExtensionArray subclass, by dispatching to the\n        relevant operator defined on the individual elements of the\n        ExtensionArray.\n\n        Parameters\n        ----------\n        op : function\n            An operator that takes arguments op(a, b)\n        coerce_to_dtype : bool, default True\n            boolean indicating whether to attempt to convert\n            the result to the underlying ExtensionArray dtype.\n            If it's not possible to create a new ExtensionArray with the\n            values, an ndarray is returned instead.\n\n        Returns\n        -------\n        Callable[[Any, Any], Union[ndarray, ExtensionArray]]\n            A method that can be bound to a class. When used, the method\n            receives the two arguments, one of which is the instance of\n            this class, and should return an ExtensionArray or an ndarray.\n\n            Returning an ndarray may be necessary when the result of the\n            `op` cannot be stored in the ExtensionArray. The dtype of the\n            ndarray uses NumPy's normal inference rules.\n\n        Examples\n        --------\n        Given an ExtensionArray subclass called MyExtensionArray, use\n\n            __add__ = cls._create_method(operator.add)\n\n        in the class definition of MyExtensionArray to create the operator\n        for addition, that will be based on the operator implementation\n        of the underlying elements of the ExtensionArray\n        "

        def _binop(self, other):

            def convert_values(param):
                if (isinstance(param, ExtensionArray) or is_list_like(param)):
                    ovalues = param
                else:
                    ovalues = ([param] * len(self))
                return ovalues
            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                return NotImplemented
            lvalues = self
            rvalues = convert_values(other)
            res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

            def _maybe_convert(arr):
                if coerce_to_dtype:
                    res = maybe_cast_to_extension_array(type(self), arr)
                    if (not isinstance(res, type(self))):
                        res = np.asarray(arr)
                else:
                    res = np.asarray(arr, dtype=result_dtype)
                return res
            if (op.__name__ in {'divmod', 'rdivmod'}):
                (a, b) = zip(*res)
                return (_maybe_convert(a), _maybe_convert(b))
            return _maybe_convert(res)
        op_name = f'__{op.__name__}__'
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False, result_dtype=bool)
