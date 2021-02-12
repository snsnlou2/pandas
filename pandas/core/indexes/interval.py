
' define the IntervalIndex '
from functools import wraps
from operator import le, lt
import textwrap
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas._libs.interval import Interval, IntervalMixin, IntervalTree
from pandas._libs.tslibs import BaseOffset, Timedelta, Timestamp, to_offset
from pandas._typing import DtypeObj, Label
from pandas.errors import InvalidIndexError
from pandas.util._decorators import Appender, cache_readonly
from pandas.util._exceptions import rewrite_exception
from pandas.core.dtypes.cast import find_common_type, infer_dtype_from_scalar, maybe_box_datetimelike, maybe_downcast_numeric
from pandas.core.dtypes.common import ensure_platform_int, is_categorical_dtype, is_datetime64tz_dtype, is_datetime_or_timedelta_dtype, is_dtype_equal, is_float, is_float_dtype, is_integer, is_integer_dtype, is_interval_dtype, is_list_like, is_number, is_object_dtype, is_scalar
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas.core.algorithms import take_1d, unique
from pandas.core.arrays.interval import IntervalArray, _interval_shared_docs
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, _index_shared_docs, default_pprint, ensure_index, maybe_extract_name, unpack_nested_dtype
from pandas.core.indexes.datetimes import DatetimeIndex, date_range
from pandas.core.indexes.extension import ExtensionIndex, inherit_names
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex, timedelta_range
from pandas.core.ops import get_op_result_name
if TYPE_CHECKING:
    from pandas import CategoricalIndex
_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({'klass': 'IntervalIndex', 'qualname': 'IntervalIndex', 'target_klass': 'IntervalIndex or list of Intervals', 'name': textwrap.dedent('         name : object, optional\n              Name to be stored in the index.\n         ')})

def _get_next_label(label):
    dtype = getattr(label, 'dtype', type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = 'datetime64'
    if (is_datetime_or_timedelta_dtype(dtype) or is_datetime64tz_dtype(dtype)):
        return (label + np.timedelta64(1, 'ns'))
    elif is_integer_dtype(dtype):
        return (label + 1)
    elif is_float_dtype(dtype):
        return np.nextafter(label, np.infty)
    else:
        raise TypeError(f'cannot determine next label for type {repr(type(label))}')

def _get_prev_label(label):
    dtype = getattr(label, 'dtype', type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = 'datetime64'
    if (is_datetime_or_timedelta_dtype(dtype) or is_datetime64tz_dtype(dtype)):
        return (label - np.timedelta64(1, 'ns'))
    elif is_integer_dtype(dtype):
        return (label - 1)
    elif is_float_dtype(dtype):
        return np.nextafter(label, (- np.infty))
    else:
        raise TypeError(f'cannot determine next label for type {repr(type(label))}')

def _new_IntervalIndex(cls, d):
    "\n    This is called upon unpickling, rather than the default which doesn't have\n    arguments and breaks __new__.\n    "
    return cls.from_arrays(**d)

def setop_check(method):
    '\n    This is called to decorate the set operations of IntervalIndex\n    to perform the type check in advance.\n    '
    op_name = method.__name__

    @wraps(method)
    def wrapped(self, other, sort=False):
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        (other, _) = self._convert_can_do_setop(other)
        if (not isinstance(other, IntervalIndex)):
            result = getattr(self.astype(object), op_name)(other)
            if (op_name in ('difference',)):
                result = result.astype(self.dtype)
            return result
        return method(self, other, sort)
    return wrapped

@Appender((_interval_shared_docs['class'] % {'klass': 'IntervalIndex', 'summary': 'Immutable index of intervals that are closed on the same side.', 'name': _index_doc_kwargs['name'], 'versionadded': '0.20.0', 'extra_attributes': 'is_overlapping\nvalues\n', 'extra_methods': '', 'examples': textwrap.dedent("    Examples\n    --------\n    A new ``IntervalIndex`` is typically constructed using\n    :func:`interval_range`:\n\n    >>> pd.interval_range(start=0, end=5)\n    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],\n                  closed='right',\n                  dtype='interval[int64]')\n\n    It may also be constructed using one of the constructor\n    methods: :meth:`IntervalIndex.from_arrays`,\n    :meth:`IntervalIndex.from_breaks`, and :meth:`IntervalIndex.from_tuples`.\n\n    See further examples in the doc strings of ``interval_range`` and the\n    mentioned constructor methods.\n    ")}))
@inherit_names(['set_closed', 'to_tuples'], IntervalArray, wrap=True)
@inherit_names(['__array__', 'overlaps', 'contains'], IntervalArray)
@inherit_names(['is_non_overlapping_monotonic', 'closed'], IntervalArray, cache=True)
class IntervalIndex(IntervalMixin, ExtensionIndex):
    _typ = 'intervalindex'
    _comparables = ['name']
    _attributes = ['name', 'closed']
    _defer_to_indexing = True
    _can_hold_strings = False

    def __new__(cls, data, closed=None, dtype=None, copy=False, name=None, verify_integrity=True):
        name = maybe_extract_name(name, data, cls)
        with rewrite_exception('IntervalArray', cls.__name__):
            array = IntervalArray(data, closed=closed, copy=copy, dtype=dtype, verify_integrity=verify_integrity)
        return cls._simple_new(array, name)

    @classmethod
    def _simple_new(cls, array, name=None):
        '\n        Construct from an IntervalArray\n\n        Parameters\n        ----------\n        array : IntervalArray\n        name : Label, default None\n            Attached as result.name\n        '
        assert isinstance(array, IntervalArray), type(array)
        result = IntervalMixin.__new__(cls)
        result._data = array
        result.name = name
        result._cache = {}
        result._reset_identity()
        return result

    @classmethod
    @Appender((_interval_shared_docs['from_breaks'] % {'klass': 'IntervalIndex', 'examples': textwrap.dedent("        Examples\n        --------\n        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])\n        IntervalIndex([(0, 1], (1, 2], (2, 3]],\n                      closed='right',\n                      dtype='interval[int64]')\n        ")}))
    def from_breaks(cls, breaks, closed='right', name=None, copy=False, dtype=None):
        with rewrite_exception('IntervalArray', cls.__name__):
            array = IntervalArray.from_breaks(breaks, closed=closed, copy=copy, dtype=dtype)
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender((_interval_shared_docs['from_arrays'] % {'klass': 'IntervalIndex', 'examples': textwrap.dedent("        Examples\n        --------\n        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])\n        IntervalIndex([(0, 1], (1, 2], (2, 3]],\n                      closed='right',\n                      dtype='interval[int64]')\n        ")}))
    def from_arrays(cls, left, right, closed='right', name=None, copy=False, dtype=None):
        with rewrite_exception('IntervalArray', cls.__name__):
            array = IntervalArray.from_arrays(left, right, closed, copy=copy, dtype=dtype)
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender((_interval_shared_docs['from_tuples'] % {'klass': 'IntervalIndex', 'examples': textwrap.dedent("        Examples\n        --------\n        >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])\n        IntervalIndex([(0, 1], (1, 2]],\n                       closed='right',\n                       dtype='interval[int64]')\n        ")}))
    def from_tuples(cls, data, closed='right', name=None, copy=False, dtype=None):
        with rewrite_exception('IntervalArray', cls.__name__):
            arr = IntervalArray.from_tuples(data, closed=closed, copy=copy, dtype=dtype)
        return cls._simple_new(arr, name=name)

    @cache_readonly
    def _engine(self):
        left = self._maybe_convert_i8(self.left)
        right = self._maybe_convert_i8(self.right)
        return IntervalTree(left, right, closed=self.closed)

    def __contains__(self, key):
        '\n        return a boolean if this key is IN the index\n        We *only* accept an Interval\n\n        Parameters\n        ----------\n        key : Interval\n\n        Returns\n        -------\n        bool\n        '
        hash(key)
        if (not isinstance(key, Interval)):
            return False
        try:
            self.get_loc(key)
            return True
        except KeyError:
            return False

    @cache_readonly
    def _multiindex(self):
        return MultiIndex.from_arrays([self.left, self.right], names=['left', 'right'])

    def __array_wrap__(self, result, context=None):
        return result

    def __reduce__(self):
        d = {'left': self.left, 'right': self.right}
        d.update(self._get_attributes_dict())
        return (_new_IntervalIndex, (type(self), d), None)

    @Appender(Index.astype.__doc__)
    def astype(self, dtype, copy=True):
        with rewrite_exception('IntervalArray', type(self).__name__):
            new_values = self._values.astype(dtype, copy=copy)
        return Index(new_values, dtype=new_values.dtype, name=self.name)

    @property
    def inferred_type(self):
        'Return a string of the type inferred from the values'
        return 'interval'

    @Appender(Index.memory_usage.__doc__)
    def memory_usage(self, deep=False):
        return (self.left.memory_usage(deep=deep) + self.right.memory_usage(deep=deep))

    @cache_readonly
    def is_monotonic_decreasing(self):
        '\n        Return True if the IntervalIndex is monotonic decreasing (only equal or\n        decreasing values), else False\n        '
        return self[::(- 1)].is_monotonic_increasing

    @cache_readonly
    def is_unique(self):
        '\n        Return True if the IntervalIndex contains unique elements, else False.\n        '
        left = self.left
        right = self.right
        if (self.isna().sum() > 1):
            return False
        if (left.is_unique or right.is_unique):
            return True
        seen_pairs = set()
        check_idx = np.where(left.duplicated(keep=False))[0]
        for idx in check_idx:
            pair = (left[idx], right[idx])
            if (pair in seen_pairs):
                return False
            seen_pairs.add(pair)
        return True

    @property
    def is_overlapping(self):
        "\n        Return True if the IntervalIndex has overlapping intervals, else False.\n\n        Two intervals overlap if they share a common point, including closed\n        endpoints. Intervals that only have an open endpoint in common do not\n        overlap.\n\n        .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        bool\n            Boolean indicating if the IntervalIndex has overlapping intervals.\n\n        See Also\n        --------\n        Interval.overlaps : Check whether two Interval objects overlap.\n        IntervalIndex.overlaps : Check an IntervalIndex elementwise for\n            overlaps.\n\n        Examples\n        --------\n        >>> index = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])\n        >>> index\n        IntervalIndex([(0, 2], (1, 3], (4, 5]],\n              closed='right',\n              dtype='interval[int64]')\n        >>> index.is_overlapping\n        True\n\n        Intervals that share closed endpoints overlap:\n\n        >>> index = pd.interval_range(0, 3, closed='both')\n        >>> index\n        IntervalIndex([[0, 1], [1, 2], [2, 3]],\n              closed='both',\n              dtype='interval[int64]')\n        >>> index.is_overlapping\n        True\n\n        Intervals that only have an open endpoint in common do not overlap:\n\n        >>> index = pd.interval_range(0, 3, closed='left')\n        >>> index\n        IntervalIndex([[0, 1), [1, 2), [2, 3)],\n              closed='left',\n              dtype='interval[int64]')\n        >>> index.is_overlapping\n        False\n        "
        return self._engine.is_overlapping

    def _needs_i8_conversion(self, key):
        '\n        Check if a given key needs i8 conversion. Conversion is necessary for\n        Timestamp, Timedelta, DatetimeIndex, and TimedeltaIndex keys. An\n        Interval-like requires conversion if its endpoints are one of the\n        aforementioned types.\n\n        Assumes that any list-like data has already been cast to an Index.\n\n        Parameters\n        ----------\n        key : scalar or Index-like\n            The key that should be checked for i8 conversion\n\n        Returns\n        -------\n        bool\n        '
        if (is_interval_dtype(key) or isinstance(key, Interval)):
            return self._needs_i8_conversion(key.left)
        i8_types = (Timestamp, Timedelta, DatetimeIndex, TimedeltaIndex)
        return isinstance(key, i8_types)

    def _maybe_convert_i8(self, key):
        '\n        Maybe convert a given key to its equivalent i8 value(s). Used as a\n        preprocessing step prior to IntervalTree queries (self._engine), which\n        expects numeric data.\n\n        Parameters\n        ----------\n        key : scalar or list-like\n            The key that should maybe be converted to i8.\n\n        Returns\n        -------\n        scalar or list-like\n            The original key if no conversion occurred, int if converted scalar,\n            Int64Index if converted list-like.\n        '
        original = key
        if is_list_like(key):
            key = ensure_index(key)
        if (not self._needs_i8_conversion(key)):
            return original
        scalar = is_scalar(key)
        if (is_interval_dtype(key) or isinstance(key, Interval)):
            left = self._maybe_convert_i8(key.left)
            right = self._maybe_convert_i8(key.right)
            constructor = (Interval if scalar else IntervalIndex.from_arrays)
            return constructor(left, right, closed=self.closed)
        if scalar:
            (key_dtype, key_i8) = infer_dtype_from_scalar(key, pandas_dtype=True)
            if lib.is_period(key):
                key_i8 = key.ordinal
        else:
            (key_dtype, key_i8) = (key.dtype, Index(key.asi8))
            if key.hasnans:
                key_i8 = key_i8.where((~ key._isnan))
        subtype = self.dtype.subtype
        if (not is_dtype_equal(subtype, key_dtype)):
            raise ValueError(f'Cannot index an IntervalIndex of subtype {subtype} with values of dtype {key_dtype}')
        return key_i8

    def _searchsorted_monotonic(self, label, side, exclude_label=False):
        if (not self.is_non_overlapping_monotonic):
            raise KeyError('can only get slices from an IntervalIndex if bounds are non-overlapping and all monotonic increasing or decreasing')
        if isinstance(label, IntervalMixin):
            raise NotImplementedError('Interval objects are not currently supported')
        if (((side == 'left') and self.left.is_monotonic_increasing) or ((side == 'right') and (not self.left.is_monotonic_increasing))):
            sub_idx = self.right
            if (self.open_right or exclude_label):
                label = _get_next_label(label)
        else:
            sub_idx = self.left
            if (self.open_left or exclude_label):
                label = _get_prev_label(label)
        return sub_idx._searchsorted_monotonic(label, side)

    def get_loc(self, key, method=None, tolerance=None):
        '\n        Get integer location, slice or boolean mask for requested label.\n\n        Parameters\n        ----------\n        key : label\n        method : {None}, optional\n            * default: matches where the label is within an interval only.\n\n        Returns\n        -------\n        int if unique index, slice if monotonic index, else mask\n\n        Examples\n        --------\n        >>> i1, i2 = pd.Interval(0, 1), pd.Interval(1, 2)\n        >>> index = pd.IntervalIndex([i1, i2])\n        >>> index.get_loc(1)\n        0\n\n        You can also supply a point inside an interval.\n\n        >>> index.get_loc(1.5)\n        1\n\n        If a label is in several intervals, you get the locations of all the\n        relevant intervals.\n\n        >>> i3 = pd.Interval(0, 2)\n        >>> overlapping_index = pd.IntervalIndex([i1, i2, i3])\n        >>> overlapping_index.get_loc(0.5)\n        array([ True, False,  True])\n\n        Only exact matches will be returned if an interval is provided.\n\n        >>> index.get_loc(pd.Interval(0, 1))\n        0\n        '
        self._check_indexing_method(method)
        if (not is_scalar(key)):
            raise InvalidIndexError(key)
        if isinstance(key, Interval):
            if (self.closed != key.closed):
                raise KeyError(key)
            mask = ((self.left == key.left) & (self.right == key.right))
        else:
            op_left = (le if self.closed_left else lt)
            op_right = (le if self.closed_right else lt)
            try:
                mask = (op_left(self.left, key) & op_right(key, self.right))
            except TypeError as err:
                raise KeyError(key) from err
        matches = mask.sum()
        if (matches == 0):
            raise KeyError(key)
        elif (matches == 1):
            return mask.argmax()
        return lib.maybe_booleans_to_slice(mask.view('u1'))

    def _get_indexer(self, target, method=None, limit=None, tolerance=None):
        if isinstance(target, IntervalIndex):
            if self.equals(target):
                return np.arange(len(self), dtype='intp')
            if (not self._should_compare(target)):
                return self._get_indexer_non_comparable(target, method, unique=True)
            left_indexer = self.left.get_indexer(target.left)
            right_indexer = self.right.get_indexer(target.right)
            indexer = np.where((left_indexer == right_indexer), left_indexer, (- 1))
        elif is_categorical_dtype(target.dtype):
            target = cast('CategoricalIndex', target)
            categories_indexer = self.get_indexer(target.categories)
            indexer = take_1d(categories_indexer, target.codes, fill_value=(- 1))
        elif (not is_object_dtype(target)):
            target = self._maybe_convert_i8(target)
            indexer = self._engine.get_indexer(target.values)
        else:
            return self._get_indexer_pointwise(target)[0]
        return ensure_platform_int(indexer)

    @Appender((_index_shared_docs['get_indexer_non_unique'] % _index_doc_kwargs))
    def get_indexer_non_unique(self, target):
        target = ensure_index(target)
        if (isinstance(target, IntervalIndex) and (not self._should_compare(target))):
            return self._get_indexer_non_comparable(target, None, unique=False)
        elif (is_object_dtype(target.dtype) or isinstance(target, IntervalIndex)):
            return self._get_indexer_pointwise(target)
        else:
            target = self._maybe_convert_i8(target)
            (indexer, missing) = self._engine.get_indexer_non_unique(target.values)
        return (ensure_platform_int(indexer), ensure_platform_int(missing))

    def _get_indexer_pointwise(self, target):
        '\n        pointwise implementation for get_indexer and get_indexer_non_unique.\n        '
        (indexer, missing) = ([], [])
        for (i, key) in enumerate(target):
            try:
                locs = self.get_loc(key)
                if isinstance(locs, slice):
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype='intp')
                locs = np.array(locs, ndmin=1)
            except KeyError:
                missing.append(i)
                locs = np.array([(- 1)])
            except InvalidIndexError as err:
                raise TypeError(key) from err
            indexer.append(locs)
        indexer = np.concatenate(indexer)
        return (ensure_platform_int(indexer), ensure_platform_int(missing))

    @property
    def _index_as_unique(self):
        return (not self.is_overlapping)
    _requires_unique_msg = 'cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique'

    def _convert_slice_indexer(self, key, kind):
        if (not ((key.step is None) or (key.step == 1))):
            msg = 'label-based slicing with step!=1 is not supported for IntervalIndex'
            if (kind == 'loc'):
                raise ValueError(msg)
            elif (kind == 'getitem'):
                if (not is_valid_positional_slice(key)):
                    raise ValueError(msg)
        return super()._convert_slice_indexer(key, kind)

    def _should_fallback_to_positional(self):
        return (self.dtype.subtype.kind in ['m', 'M'])

    def _maybe_cast_slice_bound(self, label, side, kind):
        return getattr(self, side)._maybe_cast_slice_bound(label, side, kind)

    @Appender(Index._convert_list_indexer.__doc__)
    def _convert_list_indexer(self, keyarr):
        '\n        we are passed a list-like indexer. Return the\n        indexer for matching intervals.\n        '
        locs = self.get_indexer_for(keyarr)
        if (locs == (- 1)).any():
            raise KeyError(keyarr[(locs == (- 1))].tolist())
        return locs

    def _is_comparable_dtype(self, dtype):
        if (not isinstance(dtype, IntervalDtype)):
            return False
        common_subtype = find_common_type([self.dtype.subtype, dtype.subtype])
        return (not is_object_dtype(common_subtype))

    def _should_compare(self, other):
        other = unpack_nested_dtype(other)
        if is_object_dtype(other.dtype):
            return True
        if (not self._is_comparable_dtype(other.dtype)):
            return False
        return (other.closed == self.closed)

    @cache_readonly
    def left(self):
        return Index(self._data.left, copy=False)

    @cache_readonly
    def right(self):
        return Index(self._data.right, copy=False)

    @cache_readonly
    def mid(self):
        return Index(self._data.mid, copy=False)

    @property
    def length(self):
        return Index(self._data.length, copy=False)

    def putmask(self, mask, value):
        arr = self._data.copy()
        try:
            (value_left, value_right) = arr._validate_setitem_value(value)
        except (ValueError, TypeError):
            return self.astype(object).putmask(mask, value)
        if isinstance(self._data._left, np.ndarray):
            np.putmask(arr._left, mask, value_left)
            np.putmask(arr._right, mask, value_right)
        else:
            arr._left.putmask(mask, value_left)
            arr._right.putmask(mask, value_right)
        return type(self)._simple_new(arr, name=self.name)

    @Appender(Index.where.__doc__)
    def where(self, cond, other=None):
        if (other is None):
            other = self._na_value
        values = np.where(cond, self._values, other)
        result = IntervalArray(values)
        return type(self)._simple_new(result, name=self.name)

    def delete(self, loc):
        '\n        Return a new IntervalIndex with passed location(-s) deleted\n\n        Returns\n        -------\n        IntervalIndex\n        '
        new_left = self.left.delete(loc)
        new_right = self.right.delete(loc)
        result = self._data._shallow_copy(new_left, new_right)
        return type(self)._simple_new(result, name=self.name)

    def insert(self, loc, item):
        '\n        Return a new IntervalIndex inserting new item at location. Follows\n        Python list.append semantics for negative values.  Only Interval\n        objects and NA can be inserted into an IntervalIndex\n\n        Parameters\n        ----------\n        loc : int\n        item : object\n\n        Returns\n        -------\n        IntervalIndex\n        '
        (left_insert, right_insert) = self._data._validate_scalar(item)
        new_left = self.left.insert(loc, left_insert)
        new_right = self.right.insert(loc, right_insert)
        result = self._data._shallow_copy(new_left, new_right)
        return type(self)._simple_new(result, name=self.name)

    def _format_with_header(self, header, na_rep='NaN'):
        return (header + list(self._format_native_types(na_rep=na_rep)))

    def _format_native_types(self, na_rep='NaN', quoting=None, **kwargs):
        return super()._format_native_types(na_rep=na_rep, quoting=quoting, **kwargs)

    def _format_data(self, name=None):
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
            head_joined = ', '.join(head)
            tail_joined = ', '.join(tail)
            summary = f'[{head_joined} ... {tail_joined}]'
        else:
            tail = [formatter(x) for x in self]
            joined = ', '.join(tail)
            summary = f'[{joined}]'
        return ((summary + ',') + self._format_space())

    def _format_attrs(self):
        attrs = [('closed', repr(self.closed))]
        if (self.name is not None):
            attrs.append(('name', default_pprint(self.name)))
        attrs.append(('dtype', f"'{self.dtype}'"))
        return attrs

    def _format_space(self):
        space = (' ' * (len(type(self).__name__) + 1))
        return f'''
{space}'''

    def _assert_can_do_setop(self, other):
        super()._assert_can_do_setop(other)
        if (isinstance(other, IntervalIndex) and (not self._should_compare(other))):
            raise TypeError('can only do set operations between two IntervalIndex objects that are closed on the same side and have compatible dtypes')

    def _intersection(self, other, sort):
        '\n        intersection specialized to the case with matching dtypes.\n        '
        if (self.left.is_unique and self.right.is_unique):
            taken = self._intersection_unique(other)
        elif (other.left.is_unique and other.right.is_unique and (self.isna().sum() <= 1)):
            taken = other._intersection_unique(self)
        else:
            taken = self._intersection_non_unique(other)
        if (sort is None):
            taken = taken.sort_values()
        return taken

    def _intersection_unique(self, other):
        '\n        Used when the IntervalIndex does not have any common endpoint,\n        no matter left or right.\n        Return the intersection with another IntervalIndex.\n\n        Parameters\n        ----------\n        other : IntervalIndex\n\n        Returns\n        -------\n        IntervalIndex\n        '
        lindexer = self.left.get_indexer(other.left)
        rindexer = self.right.get_indexer(other.right)
        match = ((lindexer == rindexer) & (lindexer != (- 1)))
        indexer = lindexer.take(match.nonzero()[0])
        indexer = unique(indexer)
        return self.take(indexer)

    def _intersection_non_unique(self, other):
        '\n        Used when the IntervalIndex does have some common endpoints,\n        on either sides.\n        Return the intersection with another IntervalIndex.\n\n        Parameters\n        ----------\n        other : IntervalIndex\n\n        Returns\n        -------\n        IntervalIndex\n        '
        mask = np.zeros(len(self), dtype=bool)
        if (self.hasnans and other.hasnans):
            first_nan_loc = np.arange(len(self))[self.isna()][0]
            mask[first_nan_loc] = True
        other_tups = set(zip(other.left, other.right))
        for (i, tup) in enumerate(zip(self.left, self.right)):
            if (tup in other_tups):
                mask[i] = True
        return self[mask]

    def _setop(op_name, sort=None):

        def func(self, other, sort=sort):
            result = getattr(self._multiindex, op_name)(other._multiindex, sort=sort)
            result_name = get_op_result_name(self, other)
            if result.empty:
                result = result._values.astype(self.dtype.subtype)
            else:
                result = result._values
            return type(self).from_tuples(result, closed=self.closed, name=result_name)
        func.__name__ = op_name
        return setop_check(func)
    _union = _setop('union')
    difference = _setop('difference')

    @property
    def _is_all_dates(self):
        '\n        This is False even when left/right contain datetime-like objects,\n        as the check is done on the Interval itself\n        '
        return False

def _is_valid_endpoint(endpoint):
    '\n    Helper for interval_range to check if start/end are valid types.\n    '
    return any([is_number(endpoint), isinstance(endpoint, Timestamp), isinstance(endpoint, Timedelta), (endpoint is None)])

def _is_type_compatible(a, b):
    '\n    Helper for interval_range to check type compat of start/end/freq.\n    '
    is_ts_compat = (lambda x: isinstance(x, (Timestamp, BaseOffset)))
    is_td_compat = (lambda x: isinstance(x, (Timedelta, BaseOffset)))
    return ((is_number(a) and is_number(b)) or (is_ts_compat(a) and is_ts_compat(b)) or (is_td_compat(a) and is_td_compat(b)) or com.any_none(a, b))

def interval_range(start=None, end=None, periods=None, freq=None, name=None, closed='right'):
    "\n    Return a fixed frequency IntervalIndex.\n\n    Parameters\n    ----------\n    start : numeric or datetime-like, default None\n        Left bound for generating intervals.\n    end : numeric or datetime-like, default None\n        Right bound for generating intervals.\n    periods : int, default None\n        Number of periods to generate.\n    freq : numeric, str, or DateOffset, default None\n        The length of each interval. Must be consistent with the type of start\n        and end, e.g. 2 for numeric, or '5H' for datetime-like.  Default is 1\n        for numeric and 'D' for datetime-like.\n    name : str, default None\n        Name of the resulting IntervalIndex.\n    closed : {'left', 'right', 'both', 'neither'}, default 'right'\n        Whether the intervals are closed on the left-side, right-side, both\n        or neither.\n\n    Returns\n    -------\n    IntervalIndex\n\n    See Also\n    --------\n    IntervalIndex : An Index of intervals that are all closed on the same side.\n\n    Notes\n    -----\n    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,\n    exactly three must be specified. If ``freq`` is omitted, the resulting\n    ``IntervalIndex`` will have ``periods`` linearly spaced elements between\n    ``start`` and ``end``, inclusively.\n\n    To learn more about datetime-like frequency strings, please see `this link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n\n    Examples\n    --------\n    Numeric ``start`` and  ``end`` is supported.\n\n    >>> pd.interval_range(start=0, end=5)\n    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],\n                  closed='right', dtype='interval[int64]')\n\n    Additionally, datetime-like input is also supported.\n\n    >>> pd.interval_range(start=pd.Timestamp('2017-01-01'),\n    ...                   end=pd.Timestamp('2017-01-04'))\n    IntervalIndex([(2017-01-01, 2017-01-02], (2017-01-02, 2017-01-03],\n                   (2017-01-03, 2017-01-04]],\n                  closed='right', dtype='interval[datetime64[ns]]')\n\n    The ``freq`` parameter specifies the frequency between the left and right.\n    endpoints of the individual intervals within the ``IntervalIndex``.  For\n    numeric ``start`` and ``end``, the frequency must also be numeric.\n\n    >>> pd.interval_range(start=0, periods=4, freq=1.5)\n    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],\n                  closed='right', dtype='interval[float64]')\n\n    Similarly, for datetime-like ``start`` and ``end``, the frequency must be\n    convertible to a DateOffset.\n\n    >>> pd.interval_range(start=pd.Timestamp('2017-01-01'),\n    ...                   periods=3, freq='MS')\n    IntervalIndex([(2017-01-01, 2017-02-01], (2017-02-01, 2017-03-01],\n                   (2017-03-01, 2017-04-01]],\n                  closed='right', dtype='interval[datetime64[ns]]')\n\n    Specify ``start``, ``end``, and ``periods``; the frequency is generated\n    automatically (linearly spaced).\n\n    >>> pd.interval_range(start=0, end=6, periods=4)\n    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],\n              closed='right',\n              dtype='interval[float64]')\n\n    The ``closed`` parameter specifies which endpoints of the individual\n    intervals within the ``IntervalIndex`` are closed.\n\n    >>> pd.interval_range(end=5, periods=4, closed='both')\n    IntervalIndex([[1, 2], [2, 3], [3, 4], [4, 5]],\n                  closed='both', dtype='interval[int64]')\n    "
    start = maybe_box_datetimelike(start)
    end = maybe_box_datetimelike(end)
    endpoint = (start if (start is not None) else end)
    if ((freq is None) and com.any_none(periods, start, end)):
        freq = (1 if is_number(endpoint) else 'D')
    if (com.count_not_none(start, end, periods, freq) != 3):
        raise ValueError('Of the four parameters: start, end, periods, and freq, exactly three must be specified')
    if (not _is_valid_endpoint(start)):
        raise ValueError(f'start must be numeric or datetime-like, got {start}')
    elif (not _is_valid_endpoint(end)):
        raise ValueError(f'end must be numeric or datetime-like, got {end}')
    if is_float(periods):
        periods = int(periods)
    elif ((not is_integer(periods)) and (periods is not None)):
        raise TypeError(f'periods must be a number, got {periods}')
    if ((freq is not None) and (not is_number(freq))):
        try:
            freq = to_offset(freq)
        except ValueError as err:
            raise ValueError(f'freq must be numeric or convertible to DateOffset, got {freq}') from err
    if (not all([_is_type_compatible(start, end), _is_type_compatible(start, freq), _is_type_compatible(end, freq)])):
        raise TypeError('start, end, freq need to be type compatible')
    if (periods is not None):
        periods += 1
    if is_number(endpoint):
        if com.all_not_none(start, end, freq):
            end -= ((end - start) % freq)
        if (periods is None):
            periods = (int(((end - start) // freq)) + 1)
        elif (start is None):
            start = (end - ((periods - 1) * freq))
        elif (end is None):
            end = (start + ((periods - 1) * freq))
        breaks = np.linspace(start, end, periods)
        if all((is_integer(x) for x in com.not_none(start, end, freq))):
            breaks = maybe_downcast_numeric(breaks, np.dtype('int64'))
    elif isinstance(endpoint, Timestamp):
        breaks = date_range(start=start, end=end, periods=periods, freq=freq)
    else:
        breaks = timedelta_range(start=start, end=end, periods=periods, freq=freq)
    return IntervalIndex.from_breaks(breaks, name=name, closed=closed)
