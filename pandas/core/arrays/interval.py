
import operator
from operator import le, lt
import textwrap
from typing import Sequence, Type, TypeVar
import numpy as np
from pandas._config import get_option
from pandas._libs.interval import VALID_CLOSED, Interval, IntervalMixin, intervals_to_interval_bounds
from pandas._libs.missing import NA
from pandas._typing import ArrayLike
from pandas.compat.numpy import function as nv
from pandas.util._decorators import Appender
from pandas.core.dtypes.cast import maybe_convert_platform
from pandas.core.dtypes.common import is_categorical_dtype, is_datetime64_any_dtype, is_dtype_equal, is_float_dtype, is_integer_dtype, is_interval_dtype, is_list_like, is_object_dtype, is_scalar, is_string_dtype, is_timedelta64_dtype, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas.core.dtypes.generic import ABCDatetimeIndex, ABCIntervalIndex, ABCPeriodIndex, ABCSeries
from pandas.core.dtypes.missing import is_valid_nat_for_dtype, isna, notna
from pandas.core.algorithms import isin, take, value_counts
from pandas.core.arrays.base import ExtensionArray, _extension_array_shared_docs
from pandas.core.arrays.categorical import Categorical
import pandas.core.common as com
from pandas.core.construction import array, ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.indexes.base import ensure_index
from pandas.core.ops import invalid_comparison, unpack_zerodim_and_defer
IntervalArrayT = TypeVar('IntervalArrayT', bound='IntervalArray')
_interval_shared_docs = {}
_shared_docs_kwargs = {'klass': 'IntervalArray', 'qualname': 'arrays.IntervalArray', 'name': ''}
_interval_shared_docs['class'] = "\n%(summary)s\n\n.. versionadded:: %(versionadded)s\n\nParameters\n----------\ndata : array-like (1-dimensional)\n    Array-like containing Interval objects from which to build the\n    %(klass)s.\nclosed : {'left', 'right', 'both', 'neither'}, default 'right'\n    Whether the intervals are closed on the left-side, right-side, both or\n    neither.\ndtype : dtype or None, default None\n    If None, dtype will be inferred.\ncopy : bool, default False\n    Copy the input data.\n%(name)sverify_integrity : bool, default True\n    Verify that the %(klass)s is valid.\n\nAttributes\n----------\nleft\nright\nclosed\nmid\nlength\nis_empty\nis_non_overlapping_monotonic\n%(extra_attributes)s\nMethods\n-------\nfrom_arrays\nfrom_tuples\nfrom_breaks\ncontains\noverlaps\nset_closed\nto_tuples\n%(extra_methods)s\nSee Also\n--------\nIndex : The base pandas Index type.\nInterval : A bounded slice-like interval; the elements of an %(klass)s.\ninterval_range : Function to create a fixed frequency IntervalIndex.\ncut : Bin values into discrete Intervals.\nqcut : Bin values into equal-sized Intervals based on rank or sample quantiles.\n\nNotes\n-----\nSee the `user guide\n<https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`_\nfor more.\n\n%(examples)s"

@Appender((_interval_shared_docs['class'] % {'klass': 'IntervalArray', 'summary': 'Pandas array for interval data that are closed on the same side.', 'versionadded': '0.24.0', 'name': '', 'extra_attributes': '', 'extra_methods': '', 'examples': textwrap.dedent('    Examples\n    --------\n    A new ``IntervalArray`` can be constructed directly from an array-like of\n    ``Interval`` objects:\n\n    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])\n    <IntervalArray>\n    [(0, 1], (1, 5]]\n    Length: 2, closed: right, dtype: interval[int64]\n\n    It may also be constructed using one of the constructor\n    methods: :meth:`IntervalArray.from_arrays`,\n    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.\n    ')}))
class IntervalArray(IntervalMixin, ExtensionArray):
    ndim = 1
    can_hold_na = True
    _na_value = _fill_value = np.nan

    def __new__(cls, data, closed=None, dtype=None, copy=False, verify_integrity=True):
        if (isinstance(data, (ABCSeries, ABCIntervalIndex)) and is_interval_dtype(data.dtype)):
            data = data._values
        if isinstance(data, cls):
            left = data._left
            right = data._right
            closed = (closed or data.closed)
        else:
            if is_scalar(data):
                msg = f'{cls.__name__}(...) must be called with a collection of some kind, {data} was passed'
                raise TypeError(msg)
            data = maybe_convert_platform_interval(data)
            (left, right, infer_closed) = intervals_to_interval_bounds(data, validate_closed=(closed is None))
            closed = (closed or infer_closed)
        return cls._simple_new(left, right, closed, copy=copy, dtype=dtype, verify_integrity=verify_integrity)

    @classmethod
    def _simple_new(cls, left, right, closed=None, copy=False, dtype=None, verify_integrity=True):
        result = IntervalMixin.__new__(cls)
        closed = (closed or 'right')
        left = ensure_index(left, copy=copy)
        right = ensure_index(right, copy=copy)
        if (dtype is not None):
            dtype = pandas_dtype(dtype)
            if (not is_interval_dtype(dtype)):
                msg = f'dtype must be an IntervalDtype, got {dtype}'
                raise TypeError(msg)
            elif (dtype.subtype is not None):
                left = left.astype(dtype.subtype)
                right = right.astype(dtype.subtype)
        if (is_float_dtype(left) and is_integer_dtype(right)):
            right = right.astype(left.dtype)
        elif (is_float_dtype(right) and is_integer_dtype(left)):
            left = left.astype(right.dtype)
        if (type(left) != type(right)):
            msg = f'must not have differing left [{type(left).__name__}] and right [{type(right).__name__}] types'
            raise ValueError(msg)
        elif (is_categorical_dtype(left.dtype) or is_string_dtype(left.dtype)):
            msg = 'category, object, and string subtypes are not supported for IntervalArray'
            raise TypeError(msg)
        elif isinstance(left, ABCPeriodIndex):
            msg = 'Period dtypes are not supported, use a PeriodIndex instead'
            raise ValueError(msg)
        elif (isinstance(left, ABCDatetimeIndex) and (str(left.tz) != str(right.tz))):
            msg = f"left and right must have the same time zone, got '{left.tz}' and '{right.tz}'"
            raise ValueError(msg)
        left = ensure_wrapped_if_datetimelike(left)
        left = extract_array(left, extract_numpy=True)
        right = ensure_wrapped_if_datetimelike(right)
        right = extract_array(right, extract_numpy=True)
        lbase = getattr(left, '_ndarray', left).base
        rbase = getattr(right, '_ndarray', right).base
        if ((lbase is not None) and (lbase is rbase)):
            right = right.copy()
        result._left = left
        result._right = right
        result._closed = closed
        if verify_integrity:
            result._validate()
        return result

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        if (len(values) == 0):
            values = values.astype(original.dtype.subtype)
        return cls(values, closed=original.closed)
    _interval_shared_docs['from_breaks'] = textwrap.dedent("\n        Construct an %(klass)s from an array of splits.\n\n        Parameters\n        ----------\n        breaks : array-like (1-dimensional)\n            Left and right bounds for each interval.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.\n        copy : bool, default False\n            Copy the data.\n        dtype : dtype or None, default None\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_arrays : Construct from a left and right array.\n        %(klass)s.from_tuples : Construct from a sequence of tuples.\n\n        %(examples)s        ")

    @classmethod
    @Appender((_interval_shared_docs['from_breaks'] % {'klass': 'IntervalArray', 'examples': textwrap.dedent('        Examples\n        --------\n        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])\n        <IntervalArray>\n        [(0, 1], (1, 2], (2, 3]]\n        Length: 3, closed: right, dtype: interval[int64]\n        ')}))
    def from_breaks(cls, breaks, closed='right', copy=False, dtype=None):
        breaks = maybe_convert_platform_interval(breaks)
        return cls.from_arrays(breaks[:(- 1)], breaks[1:], closed, copy=copy, dtype=dtype)
    _interval_shared_docs['from_arrays'] = textwrap.dedent("\n        Construct from two arrays defining the left and right bounds.\n\n        Parameters\n        ----------\n        left : array-like (1-dimensional)\n            Left bounds for each interval.\n        right : array-like (1-dimensional)\n            Right bounds for each interval.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.\n        copy : bool, default False\n            Copy the data.\n        dtype : dtype, optional\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        Raises\n        ------\n        ValueError\n            When a value is missing in only one of `left` or `right`.\n            When a value in `left` is greater than the corresponding value\n            in `right`.\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_breaks : Construct an %(klass)s from an array of\n            splits.\n        %(klass)s.from_tuples : Construct an %(klass)s from an\n            array-like of tuples.\n\n        Notes\n        -----\n        Each element of `left` must be less than or equal to the `right`\n        element at the same position. If an element is missing, it must be\n        missing in both `left` and `right`. A TypeError is raised when\n        using an unsupported type for `left` or `right`. At the moment,\n        'category', 'object', and 'string' subtypes are not supported.\n\n        %(examples)s        ")

    @classmethod
    @Appender((_interval_shared_docs['from_arrays'] % {'klass': 'IntervalArray', 'examples': textwrap.dedent('        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])\n        <IntervalArray>\n        [(0, 1], (1, 2], (2, 3]]\n        Length: 3, closed: right, dtype: interval[int64]\n        ')}))
    def from_arrays(cls, left, right, closed='right', copy=False, dtype=None):
        left = maybe_convert_platform_interval(left)
        right = maybe_convert_platform_interval(right)
        return cls._simple_new(left, right, closed, copy=copy, dtype=dtype, verify_integrity=True)
    _interval_shared_docs['from_tuples'] = textwrap.dedent("\n        Construct an %(klass)s from an array-like of tuples.\n\n        Parameters\n        ----------\n        data : array-like (1-dimensional)\n            Array of tuples.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.\n        copy : bool, default False\n            By-default copy the data, this is compat only and ignored.\n        dtype : dtype or None, default None\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_arrays : Construct an %(klass)s from a left and\n                                    right array.\n        %(klass)s.from_breaks : Construct an %(klass)s from an array of\n                                    splits.\n\n        %(examples)s        ")

    @classmethod
    @Appender((_interval_shared_docs['from_tuples'] % {'klass': 'IntervalArray', 'examples': textwrap.dedent('        Examples\n        --------\n        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])\n        <IntervalArray>\n        [(0, 1], (1, 2]]\n        Length: 2, closed: right, dtype: interval[int64]\n        ')}))
    def from_tuples(cls, data, closed='right', copy=False, dtype=None):
        if len(data):
            (left, right) = ([], [])
        else:
            left = right = data
        for d in data:
            if isna(d):
                lhs = rhs = np.nan
            else:
                name = cls.__name__
                try:
                    (lhs, rhs) = d
                except ValueError as err:
                    msg = f'{name}.from_tuples requires tuples of length 2, got {d}'
                    raise ValueError(msg) from err
                except TypeError as err:
                    msg = f'{name}.from_tuples received an invalid item, {d}'
                    raise TypeError(msg) from err
            left.append(lhs)
            right.append(rhs)
        return cls.from_arrays(left, right, closed, copy=False, dtype=dtype)

    def _validate(self):
        '\n        Verify that the IntervalArray is valid.\n\n        Checks that\n\n        * closed is valid\n        * left and right match lengths\n        * left and right have the same missing values\n        * left is always below right\n        '
        if (self.closed not in VALID_CLOSED):
            msg = f"invalid option for 'closed': {self.closed}"
            raise ValueError(msg)
        if (len(self._left) != len(self._right)):
            msg = 'left and right must have the same length'
            raise ValueError(msg)
        left_mask = notna(self._left)
        right_mask = notna(self._right)
        if (not (left_mask == right_mask).all()):
            msg = 'missing values must be missing in the same location both left and right sides'
            raise ValueError(msg)
        if (not (self._left[left_mask] <= self._right[left_mask]).all()):
            msg = 'left side of interval must be <= right side'
            raise ValueError(msg)

    def _shallow_copy(self, left, right):
        '\n        Return a new IntervalArray with the replacement attributes\n\n        Parameters\n        ----------\n        left : Index\n            Values to be used for the left-side of the intervals.\n        right : Index\n            Values to be used for the right-side of the intervals.\n        '
        return self._simple_new(left, right, closed=self.closed, verify_integrity=False)

    @property
    def dtype(self):
        return IntervalDtype(self.left.dtype)

    @property
    def nbytes(self):
        return (self.left.nbytes + self.right.nbytes)

    @property
    def size(self):
        return self.left.size

    def __iter__(self):
        return iter(np.asarray(self))

    def __len__(self):
        return len(self._left)

    def __getitem__(self, key):
        key = check_array_indexer(self, key)
        left = self._left[key]
        right = self._right[key]
        if (not isinstance(left, (np.ndarray, ExtensionArray))):
            if (is_scalar(left) and isna(left)):
                return self._fill_value
            return Interval(left, right, self.closed)
        if (np.ndim(left) > 1):
            raise ValueError('multi-dimensional indexing not allowed')
        return self._shallow_copy(left, right)

    def __setitem__(self, key, value):
        (value_left, value_right) = self._validate_setitem_value(value)
        key = check_array_indexer(self, key)
        self._left[key] = value_left
        self._right[key] = value_right

    def _cmp_method(self, other, op):
        if is_list_like(other):
            if (len(self) != len(other)):
                raise ValueError('Lengths must match to compare')
            other = array(other)
        elif (not isinstance(other, Interval)):
            return invalid_comparison(self, other, op)
        if isinstance(other, Interval):
            other_dtype = pandas_dtype('interval')
        elif (not is_categorical_dtype(other.dtype)):
            other_dtype = other.dtype
        else:
            other_dtype = other.categories.dtype
            if is_interval_dtype(other_dtype):
                if (self.closed != other.categories.closed):
                    return invalid_comparison(self, other, op)
                other = other.categories.take(other.codes, allow_fill=True, fill_value=other.categories._na_value)
        if is_interval_dtype(other_dtype):
            if (self.closed != other.closed):
                return invalid_comparison(self, other, op)
            elif (not isinstance(other, Interval)):
                other = type(self)(other)
            if (op is operator.eq):
                return ((self._left == other.left) & (self._right == other.right))
            elif (op is operator.ne):
                return ((self._left != other.left) | (self._right != other.right))
            elif (op is operator.gt):
                return ((self._left > other.left) | ((self._left == other.left) & (self._right > other.right)))
            elif (op is operator.ge):
                return ((self == other) | (self > other))
            elif (op is operator.lt):
                return ((self._left < other.left) | ((self._left == other.left) & (self._right < other.right)))
            else:
                return ((self == other) | (self < other))
        if (not is_object_dtype(other_dtype)):
            return invalid_comparison(self, other, op)
        result = np.zeros(len(self), dtype=bool)
        for (i, obj) in enumerate(other):
            try:
                result[i] = op(self[i], obj)
            except TypeError:
                if (obj is NA):
                    result[i] = (op is operator.ne)
                else:
                    raise
        return result

    @unpack_zerodim_and_defer('__eq__')
    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer('__ne__')
    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer('__gt__')
    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer('__ge__')
    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    @unpack_zerodim_and_defer('__lt__')
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer('__le__')
    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    def argsort(self, ascending=True, kind='quicksort', na_position='last', *args, **kwargs):
        ascending = nv.validate_argsort_with_ascending(ascending, args, kwargs)
        if (ascending and (kind == 'quicksort') and (na_position == 'last')):
            return np.lexsort((self.right, self.left))
        return super().argsort(ascending=ascending, kind=kind, na_position=na_position, **kwargs)

    def fillna(self, value=None, method=None, limit=None):
        "\n        Fill NA/NaN values using the specified method.\n\n        Parameters\n        ----------\n        value : scalar, dict, Series\n            If a scalar value is passed it is used to fill all missing values.\n            Alternatively, a Series or dict can be used to fill in different\n            values for each index. The value should not be a list. The\n            value(s) passed should be either Interval objects or NA/NaN.\n        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None\n            (Not implemented yet for IntervalArray)\n            Method to use for filling holes in reindexed Series\n        limit : int, default None\n            (Not implemented yet for IntervalArray)\n            If method is specified, this is the maximum number of consecutive\n            NaN values to forward/backward fill. In other words, if there is\n            a gap with more than this number of consecutive NaNs, it will only\n            be partially filled. If method is not specified, this is the\n            maximum number of entries along the entire axis where NaNs will be\n            filled.\n\n        Returns\n        -------\n        filled : IntervalArray with NA/NaN filled\n        "
        if (method is not None):
            raise TypeError('Filling by method is not supported for IntervalArray.')
        if (limit is not None):
            raise TypeError('limit is not supported for IntervalArray.')
        (value_left, value_right) = self._validate_fill_value(value)
        left = self.left.fillna(value=value_left)
        right = self.right.fillna(value=value_right)
        return self._shallow_copy(left, right)

    def astype(self, dtype, copy=True):
        "\n        Cast to an ExtensionArray or NumPy array with dtype 'dtype'.\n\n        Parameters\n        ----------\n        dtype : str or dtype\n            Typecode or data-type to which the array is cast.\n\n        copy : bool, default True\n            Whether to copy the data, even if not necessary. If False,\n            a copy is made only if the old dtype does not match the\n            new dtype.\n\n        Returns\n        -------\n        array : ExtensionArray or ndarray\n            ExtensionArray or NumPy ndarray with 'dtype' for its dtype.\n        "
        from pandas import Index
        from pandas.core.arrays.string_ import StringDtype
        if (dtype is not None):
            dtype = pandas_dtype(dtype)
        if is_interval_dtype(dtype):
            if (dtype == self.dtype):
                return (self.copy() if copy else self)
            try:
                new_left = Index(self._left, copy=False).astype(dtype.subtype)
                new_right = Index(self._right, copy=False).astype(dtype.subtype)
            except TypeError as err:
                msg = f'Cannot convert {self.dtype} to {dtype}; subtypes are incompatible'
                raise TypeError(msg) from err
            return self._shallow_copy(new_left, new_right)
        elif is_categorical_dtype(dtype):
            return Categorical(np.asarray(self), dtype=dtype)
        elif isinstance(dtype, StringDtype):
            return dtype.construct_array_type()._from_sequence(self, copy=False)
        try:
            return np.asarray(self).astype(dtype, copy=copy)
        except (TypeError, ValueError) as err:
            msg = f'Cannot cast {type(self).__name__} to dtype {dtype}'
            raise TypeError(msg) from err

    def equals(self, other):
        if (type(self) != type(other)):
            return False
        return bool(((self.closed == other.closed) and self.left.equals(other.left) and self.right.equals(other.right)))

    @classmethod
    def _concat_same_type(cls, to_concat):
        '\n        Concatenate multiple IntervalArray\n\n        Parameters\n        ----------\n        to_concat : sequence of IntervalArray\n\n        Returns\n        -------\n        IntervalArray\n        '
        closed = {interval.closed for interval in to_concat}
        if (len(closed) != 1):
            raise ValueError('Intervals must all be closed on the same side.')
        closed = closed.pop()
        left = np.concatenate([interval.left for interval in to_concat])
        right = np.concatenate([interval.right for interval in to_concat])
        return cls._simple_new(left, right, closed=closed, copy=False)

    def copy(self):
        '\n        Return a copy of the array.\n\n        Returns\n        -------\n        IntervalArray\n        '
        left = self._left.copy()
        right = self._right.copy()
        closed = self.closed
        return type(self).from_arrays(left, right, closed=closed)

    def isna(self):
        return isna(self._left)

    def shift(self, periods=1, fill_value=None):
        if ((not len(self)) or (periods == 0)):
            return self.copy()
        if isna(fill_value):
            fill_value = self.dtype.na_value
        empty_len = min(abs(periods), len(self))
        if isna(fill_value):
            from pandas import Index
            fill_value = Index(self._left, copy=False)._na_value
            empty = IntervalArray.from_breaks(([fill_value] * (empty_len + 1)))
        else:
            empty = self._from_sequence(([fill_value] * empty_len))
        if (periods > 0):
            a = empty
            b = self[:(- periods)]
        else:
            a = self[abs(periods):]
            b = empty
        return self._concat_same_type([a, b])

    def take(self, indices, *, allow_fill=False, fill_value=None, axis=None, **kwargs):
        '\n        Take elements from the IntervalArray.\n\n        Parameters\n        ----------\n        indices : sequence of integers\n            Indices to be taken.\n\n        allow_fill : bool, default False\n            How to handle negative values in `indices`.\n\n            * False: negative values in `indices` indicate positional indices\n              from the right (the default). This is similar to\n              :func:`numpy.take`.\n\n            * True: negative values in `indices` indicate\n              missing values. These values are set to `fill_value`. Any other\n              other negative values raise a ``ValueError``.\n\n        fill_value : Interval or NA, optional\n            Fill value to use for NA-indices when `allow_fill` is True.\n            This may be ``None``, in which case the default NA value for\n            the type, ``self.dtype.na_value``, is used.\n\n            For many ExtensionArrays, there will be two representations of\n            `fill_value`: a user-facing "boxed" scalar, and a low-level\n            physical NA value. `fill_value` should be the user-facing version,\n            and the implementation should handle translating that to the\n            physical version for processing the take if necessary.\n\n        axis : any, default None\n            Present for compat with IntervalIndex; does nothing.\n\n        Returns\n        -------\n        IntervalArray\n\n        Raises\n        ------\n        IndexError\n            When the indices are out of bounds for the array.\n        ValueError\n            When `indices` contains negative values other than ``-1``\n            and `allow_fill` is True.\n        '
        nv.validate_take((), kwargs)
        fill_left = fill_right = fill_value
        if allow_fill:
            (fill_left, fill_right) = self._validate_fill_value(fill_value)
        left_take = take(self._left, indices, allow_fill=allow_fill, fill_value=fill_left)
        right_take = take(self._right, indices, allow_fill=allow_fill, fill_value=fill_right)
        return self._shallow_copy(left_take, right_take)

    def _validate_listlike(self, value):
        try:
            array = IntervalArray(value)
            (value_left, value_right) = (array.left, array.right)
        except TypeError as err:
            msg = f"'value' should be an interval type, got {type(value)} instead."
            raise TypeError(msg) from err
        return (value_left, value_right)

    def _validate_scalar(self, value):
        if isinstance(value, Interval):
            self._check_closed_matches(value, name='value')
            (left, right) = (value.left, value.right)
        elif is_valid_nat_for_dtype(value, self.left.dtype):
            left = right = value
        else:
            raise TypeError('can only insert Interval objects and NA into an IntervalArray')
        return (left, right)

    def _validate_fill_value(self, value):
        return self._validate_scalar(value)

    def _validate_setitem_value(self, value):
        needs_float_conversion = False
        if is_valid_nat_for_dtype(value, self.left.dtype):
            if is_integer_dtype(self.dtype.subtype):
                needs_float_conversion = True
            elif is_datetime64_any_dtype(self.dtype.subtype):
                value = np.datetime64('NaT')
            elif is_timedelta64_dtype(self.dtype.subtype):
                value = np.timedelta64('NaT')
            (value_left, value_right) = (value, value)
        elif (is_interval_dtype(value) or isinstance(value, Interval)):
            self._check_closed_matches(value, name='value')
            (value_left, value_right) = (value.left, value.right)
        else:
            return self._validate_listlike(value)
        if needs_float_conversion:
            raise ValueError('Cannot set float NaN to integer-backed IntervalArray')
        return (value_left, value_right)

    def value_counts(self, dropna=True):
        "\n        Returns a Series containing counts of each interval.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include counts of NaN.\n\n        Returns\n        -------\n        counts : Series\n\n        See Also\n        --------\n        Series.value_counts\n        "
        return value_counts(np.asarray(self), dropna=dropna)

    def _format_data(self):
        n = len(self)
        max_seq_items = min(((get_option('display.max_seq_items') or n) // 10), 10)
        formatter = str
        if (n == 0):
            summary = '[]'
        elif (n == 1):
            first = formatter(self[0])
            summary = f'[{first}]'
        elif (n == 2):
            first = formatter(self[0])
            last = formatter(self[(- 1)])
            summary = f'[{first}, {last}]'
        elif (n > max_seq_items):
            n = min((max_seq_items // 2), 10)
            head = [formatter(x) for x in self[:n]]
            tail = [formatter(x) for x in self[(- n):]]
            head_str = ', '.join(head)
            tail_str = ', '.join(tail)
            summary = f'[{head_str} ... {tail_str}]'
        else:
            tail = [formatter(x) for x in self]
            tail_str = ', '.join(tail)
            summary = f'[{tail_str}]'
        return summary

    def __repr__(self):
        data = self._format_data()
        class_name = f'''<{type(self).__name__}>
'''
        template = f'''{class_name}{data}
Length: {len(self)}, closed: {self.closed}, dtype: {self.dtype}'''
        return template

    def _format_space(self):
        space = (' ' * (len(type(self).__name__) + 1))
        return f'''
{space}'''

    @property
    def left(self):
        '\n        Return the left endpoints of each Interval in the IntervalArray as\n        an Index.\n        '
        from pandas import Index
        return Index(self._left, copy=False)

    @property
    def right(self):
        '\n        Return the right endpoints of each Interval in the IntervalArray as\n        an Index.\n        '
        from pandas import Index
        return Index(self._right, copy=False)

    @property
    def length(self):
        '\n        Return an Index with entries denoting the length of each Interval in\n        the IntervalArray.\n        '
        try:
            return (self.right - self.left)
        except TypeError as err:
            msg = 'IntervalArray contains Intervals without defined length, e.g. Intervals with string endpoints'
            raise TypeError(msg) from err

    @property
    def mid(self):
        '\n        Return the midpoint of each Interval in the IntervalArray as an Index.\n        '
        try:
            return (0.5 * (self.left + self.right))
        except TypeError:
            return (self.left + (0.5 * self.length))
    _interval_shared_docs['overlaps'] = textwrap.dedent("\n        Check elementwise if an Interval overlaps the values in the %(klass)s.\n\n        Two intervals overlap if they share a common point, including closed\n        endpoints. Intervals that only have an open endpoint in common do not\n        overlap.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        other : %(klass)s\n            Interval to check against for an overlap.\n\n        Returns\n        -------\n        ndarray\n            Boolean array positionally indicating where an overlap occurs.\n\n        See Also\n        --------\n        Interval.overlaps : Check whether two Interval objects overlap.\n\n        Examples\n        --------\n        %(examples)s\n        >>> intervals.overlaps(pd.Interval(0.5, 1.5))\n        array([ True,  True, False])\n\n        Intervals that share closed endpoints overlap:\n\n        >>> intervals.overlaps(pd.Interval(1, 3, closed='left'))\n        array([ True,  True, True])\n\n        Intervals that only have an open endpoint in common do not overlap:\n\n        >>> intervals.overlaps(pd.Interval(1, 2, closed='right'))\n        array([False,  True, False])\n        ")

    @Appender((_interval_shared_docs['overlaps'] % {'klass': 'IntervalArray', 'examples': textwrap.dedent('        >>> data = [(0, 1), (1, 3), (2, 4)]\n        >>> intervals = pd.arrays.IntervalArray.from_tuples(data)\n        >>> intervals\n        <IntervalArray>\n        [(0, 1], (1, 3], (2, 4]]\n        Length: 3, closed: right, dtype: interval[int64]\n        ')}))
    def overlaps(self, other):
        if isinstance(other, (IntervalArray, ABCIntervalIndex)):
            raise NotImplementedError
        elif (not isinstance(other, Interval)):
            msg = f'`other` must be Interval-like, got {type(other).__name__}'
            raise TypeError(msg)
        op1 = (le if (self.closed_left and other.closed_right) else lt)
        op2 = (le if (other.closed_left and self.closed_right) else lt)
        return (op1(self.left, other.right) & op2(other.left, self.right))

    @property
    def closed(self):
        '\n        Whether the intervals are closed on the left-side, right-side, both or\n        neither.\n        '
        return self._closed
    _interval_shared_docs['set_closed'] = textwrap.dedent("\n        Return an %(klass)s identical to the current one, but closed on the\n        specified side.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        closed : {'left', 'right', 'both', 'neither'}\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.\n\n        Returns\n        -------\n        new_index : %(klass)s\n\n        %(examples)s        ")

    @Appender((_interval_shared_docs['set_closed'] % {'klass': 'IntervalArray', 'examples': textwrap.dedent("        Examples\n        --------\n        >>> index = pd.arrays.IntervalArray.from_breaks(range(4))\n        >>> index\n        <IntervalArray>\n        [(0, 1], (1, 2], (2, 3]]\n        Length: 3, closed: right, dtype: interval[int64]\n        >>> index.set_closed('both')\n        <IntervalArray>\n        [[0, 1], [1, 2], [2, 3]]\n        Length: 3, closed: both, dtype: interval[int64]\n        ")}))
    def set_closed(self, closed):
        if (closed not in VALID_CLOSED):
            msg = f"invalid option for 'closed': {closed}"
            raise ValueError(msg)
        return type(self)._simple_new(left=self._left, right=self._right, closed=closed, verify_integrity=False)
    _interval_shared_docs['is_non_overlapping_monotonic'] = '\n        Return True if the %(klass)s is non-overlapping (no Intervals share\n        points) and is either monotonic increasing or monotonic decreasing,\n        else False.\n        '

    @property
    @Appender((_interval_shared_docs['is_non_overlapping_monotonic'] % _shared_docs_kwargs))
    def is_non_overlapping_monotonic(self):
        if (self.closed == 'both'):
            return bool(((self._right[:(- 1)] < self._left[1:]).all() or (self._left[:(- 1)] > self._right[1:]).all()))
        return bool(((self._right[:(- 1)] <= self._left[1:]).all() or (self._left[:(- 1)] >= self._right[1:]).all()))

    def __array__(self, dtype=None):
        "\n        Return the IntervalArray's data as a numpy array of Interval\n        objects (with dtype='object')\n        "
        left = self._left
        right = self._right
        mask = self.isna()
        closed = self._closed
        result = np.empty(len(left), dtype=object)
        for i in range(len(left)):
            if mask[i]:
                result[i] = np.nan
            else:
                result[i] = Interval(left[i], right[i], closed)
        return result

    def __arrow_array__(self, type=None):
        '\n        Convert myself into a pyarrow Array.\n        '
        import pyarrow
        from pandas.core.arrays._arrow_utils import ArrowIntervalType
        try:
            subtype = pyarrow.from_numpy_dtype(self.dtype.subtype)
        except TypeError as err:
            raise TypeError(f"Conversion to arrow with subtype '{self.dtype.subtype}' is not supported") from err
        interval_type = ArrowIntervalType(subtype, self.closed)
        storage_array = pyarrow.StructArray.from_arrays([pyarrow.array(self._left, type=subtype, from_pandas=True), pyarrow.array(self._right, type=subtype, from_pandas=True)], names=['left', 'right'])
        mask = self.isna()
        if mask.any():
            null_bitmap = pyarrow.array((~ mask)).buffers()[1]
            storage_array = pyarrow.StructArray.from_buffers(storage_array.type, len(storage_array), [null_bitmap], children=[storage_array.field(0), storage_array.field(1)])
        if (type is not None):
            if type.equals(interval_type.storage_type):
                return storage_array
            elif isinstance(type, ArrowIntervalType):
                if (not type.equals(interval_type)):
                    raise TypeError(f"Not supported to convert IntervalArray to type with different 'subtype' ({self.dtype.subtype} vs {type.subtype}) and 'closed' ({self.closed} vs {type.closed}) attributes")
            else:
                raise TypeError(f"Not supported to convert IntervalArray to '{type}' type")
        return pyarrow.ExtensionArray.from_storage(interval_type, storage_array)
    _interval_shared_docs['to_tuples'] = '\n        Return an %(return_type)s of tuples of the form (left, right).\n\n        Parameters\n        ----------\n        na_tuple : bool, default True\n            Returns NA as a tuple if True, ``(nan, nan)``, or just as the NA\n            value itself if False, ``nan``.\n\n        Returns\n        -------\n        tuples: %(return_type)s\n        %(examples)s        '

    @Appender((_interval_shared_docs['to_tuples'] % {'return_type': 'ndarray', 'examples': ''}))
    def to_tuples(self, na_tuple=True):
        tuples = com.asarray_tuplesafe(zip(self._left, self._right))
        if (not na_tuple):
            tuples = np.where((~ self.isna()), tuples, np.nan)
        return tuples

    @Appender((_extension_array_shared_docs['repeat'] % _shared_docs_kwargs))
    def repeat(self, repeats, axis=None):
        nv.validate_repeat((), {'axis': axis})
        left_repeat = self.left.repeat(repeats)
        right_repeat = self.right.repeat(repeats)
        return self._shallow_copy(left=left_repeat, right=right_repeat)
    _interval_shared_docs['contains'] = textwrap.dedent('\n        Check elementwise if the Intervals contain the value.\n\n        Return a boolean mask whether the value is contained in the Intervals\n        of the %(klass)s.\n\n        .. versionadded:: 0.25.0\n\n        Parameters\n        ----------\n        other : scalar\n            The value to check whether it is contained in the Intervals.\n\n        Returns\n        -------\n        boolean array\n\n        See Also\n        --------\n        Interval.contains : Check whether Interval object contains value.\n        %(klass)s.overlaps : Check if an Interval overlaps the values in the\n            %(klass)s.\n\n        Examples\n        --------\n        %(examples)s\n        >>> intervals.contains(0.5)\n        array([ True, False, False])\n    ')

    @Appender((_interval_shared_docs['contains'] % {'klass': 'IntervalArray', 'examples': textwrap.dedent('        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])\n        >>> intervals\n        <IntervalArray>\n        [(0, 1], (1, 3], (2, 4]]\n        Length: 3, closed: right, dtype: interval[int64]\n        ')}))
    def contains(self, other):
        if isinstance(other, Interval):
            raise NotImplementedError('contains not implemented for two intervals')
        return (((self._left < other) if self.open_left else (self._left <= other)) & ((other < self._right) if self.open_right else (other <= self._right)))

    def isin(self, values):
        if (not hasattr(values, 'dtype')):
            values = np.array(values)
        values = extract_array(values, extract_numpy=True)
        if is_interval_dtype(values.dtype):
            if (self.closed != values.closed):
                return np.zeros(self.shape, dtype=bool)
            if is_dtype_equal(self.dtype, values.dtype):
                left = self._combined.view('complex128')
                right = values._combined.view('complex128')
                return np.in1d(left, right)
            elif (needs_i8_conversion(self.left.dtype) ^ needs_i8_conversion(values.left.dtype)):
                return np.zeros(self.shape, dtype=bool)
        return isin(self.astype(object), values.astype(object))

    @property
    def _combined(self):
        left = self.left._values.reshape((- 1), 1)
        right = self.right._values.reshape((- 1), 1)
        if needs_i8_conversion(left.dtype):
            comb = left._concat_same_type([left, right], axis=1)
        else:
            comb = np.concatenate([left, right], axis=1)
        return comb

def maybe_convert_platform_interval(values):
    '\n    Try to do platform conversion, with special casing for IntervalArray.\n    Wrapper around maybe_convert_platform that alters the default return\n    dtype in certain cases to be compatible with IntervalArray.  For example,\n    empty lists return with integer dtype instead of object dtype, which is\n    prohibited for IntervalArray.\n\n    Parameters\n    ----------\n    values : array-like\n\n    Returns\n    -------\n    array\n    '
    if (isinstance(values, (list, tuple)) and (len(values) == 0)):
        return np.array([], dtype=np.int64)
    elif is_categorical_dtype(values):
        values = np.asarray(values)
    return maybe_convert_platform(values)
