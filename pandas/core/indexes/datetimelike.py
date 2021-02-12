
'\nBase and utility classes for tseries type pandas objects.\n'
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, TypeVar, Union, cast
import numpy as np
from pandas._libs import NaT, Timedelta, iNaT, join as libjoin, lib
from pandas._libs.tslibs import BaseOffset, Resolution, Tick
from pandas._typing import Callable, Label
from pandas.compat.numpy import function as nv
from pandas.util._decorators import Appender, cache_readonly, doc
from pandas.core.dtypes.common import is_bool_dtype, is_categorical_dtype, is_dtype_equal, is_integer, is_list_like, is_period_dtype, is_scalar
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex, inherit_names, make_wrapped_arith_op
from pandas.core.indexes.numeric import Int64Index
from pandas.core.tools.timedeltas import to_timedelta
if TYPE_CHECKING:
    from pandas import CategoricalIndex
_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_T = TypeVar('_T', bound='DatetimeIndexOpsMixin')

def _join_i8_wrapper(joinf, with_indexers=True):
    '\n    Create the join wrapper methods.\n    '

    @staticmethod
    def wrapper(left, right):
        if isinstance(left, (np.ndarray, DatetimeIndexOpsMixin, ABCSeries, DatetimeLikeArrayMixin)):
            left = left.view('i8')
        if isinstance(right, (np.ndarray, DatetimeIndexOpsMixin, ABCSeries, DatetimeLikeArrayMixin)):
            right = right.view('i8')
        results = joinf(left, right)
        if with_indexers:
            dtype = cast(np.dtype, left.dtype).base
            (join_index, left_indexer, right_indexer) = results
            join_index = join_index.view(dtype)
            return (join_index, left_indexer, right_indexer)
        return results
    return wrapper

@inherit_names(['inferred_freq', '_resolution_obj', 'resolution'], DatetimeLikeArrayMixin, cache=True)
@inherit_names(['mean', 'asi8', 'freq', 'freqstr'], DatetimeLikeArrayMixin)
class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex):
    '\n    Common ops mixin to support a unified interface datetimelike Index.\n    '
    _can_hold_strings = False
    _bool_ops = []
    _field_ops = []
    hasnans = cache_readonly(DatetimeLikeArrayMixin._hasnans.fget)
    _hasnans = hasnans

    @classmethod
    def _simple_new(cls, values, name=None):
        assert isinstance(values, cls._data_cls), type(values)
        result = object.__new__(cls)
        result._data = values
        result._name = name
        result._cache = {}
        result._index_data = values._data
        result._reset_identity()
        return result

    @property
    def _is_all_dates(self):
        return True

    @property
    def values(self):
        return self._data._data

    def __array_wrap__(self, result, context=None):
        '\n        Gets called after a ufunc and other functions.\n        '
        result = lib.item_from_zerodim(result)
        if (is_bool_dtype(result) or lib.is_scalar(result)):
            return result
        attrs = self._get_attributes_dict()
        if ((not is_period_dtype(self.dtype)) and attrs['freq']):
            attrs['freq'] = 'infer'
        return type(self)(result, **attrs)

    def equals(self, other):
        '\n        Determines if two Index objects contain the same elements.\n        '
        if self.is_(other):
            return True
        if (not isinstance(other, Index)):
            return False
        elif (other.dtype.kind in ['f', 'i', 'u', 'c']):
            return False
        elif (not isinstance(other, type(self))):
            should_try = False
            inferable = self._data._infer_matches
            if (other.dtype == object):
                should_try = (other.inferred_type in inferable)
            elif is_categorical_dtype(other.dtype):
                other = cast('CategoricalIndex', other)
                should_try = (other.categories.inferred_type in inferable)
            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    return False
        if (not is_dtype_equal(self.dtype, other.dtype)):
            return False
        return np.array_equal(self.asi8, other.asi8)

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key):
        hash(key)
        try:
            res = self.get_loc(key)
        except (KeyError, TypeError, ValueError):
            return False
        return bool((is_scalar(res) or isinstance(res, slice) or (is_list_like(res) and len(res))))

    @Appender((_index_shared_docs['take'] % _index_doc_kwargs))
    def take(self, indices, axis=0, allow_fill=True, fill_value=None, **kwargs):
        nv.validate_take((), kwargs)
        indices = np.asarray(indices, dtype=np.intp)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))
        result = NDArrayBackedExtensionIndex.take(self, indices, axis, allow_fill, fill_value, **kwargs)
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result
    _can_hold_na = True
    _na_value = NaT
    'The expected NA value to use with this index.'

    def _convert_tolerance(self, tolerance, target):
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        if ((target.size != tolerance.size) and (tolerance.size > 1)):
            raise ValueError('list-like tolerance size must match target index size')
        return tolerance

    def tolist(self):
        '\n        Return a list of the underlying data.\n        '
        return list(self.astype(object))

    def min(self, axis=None, skipna=True, *args, **kwargs):
        '\n        Return the minimum value of the Index or minimum along\n        an axis.\n\n        See Also\n        --------\n        numpy.ndarray.min\n        Series.min : Return the minimum value in a Series.\n        '
        nv.validate_min(args, kwargs)
        nv.validate_minmax_axis(axis)
        if (not len(self)):
            return self._na_value
        i8 = self.asi8
        if (len(i8) and self.is_monotonic_increasing):
            if (i8[0] != iNaT):
                return self._data._box_func(i8[0])
        if self.hasnans:
            if (not skipna):
                return self._na_value
            i8 = i8[(~ self._isnan)]
        if (not len(i8)):
            return self._na_value
        min_stamp = i8.min()
        return self._data._box_func(min_stamp)

    def argmin(self, axis=None, skipna=True, *args, **kwargs):
        '\n        Returns the indices of the minimum values along an axis.\n\n        See `numpy.ndarray.argmin` for more information on the\n        `axis` parameter.\n\n        See Also\n        --------\n        numpy.ndarray.argmin\n        '
        nv.validate_argmin(args, kwargs)
        nv.validate_minmax_axis(axis)
        i8 = self.asi8
        if self.hasnans:
            mask = self._isnan
            if (mask.all() or (not skipna)):
                return (- 1)
            i8 = i8.copy()
            i8[mask] = np.iinfo('int64').max
        return i8.argmin()

    def max(self, axis=None, skipna=True, *args, **kwargs):
        '\n        Return the maximum value of the Index or maximum along\n        an axis.\n\n        See Also\n        --------\n        numpy.ndarray.max\n        Series.max : Return the maximum value in a Series.\n        '
        nv.validate_max(args, kwargs)
        nv.validate_minmax_axis(axis)
        if (not len(self)):
            return self._na_value
        i8 = self.asi8
        if (len(i8) and self.is_monotonic):
            if (i8[(- 1)] != iNaT):
                return self._data._box_func(i8[(- 1)])
        if self.hasnans:
            if (not skipna):
                return self._na_value
            i8 = i8[(~ self._isnan)]
        if (not len(i8)):
            return self._na_value
        max_stamp = i8.max()
        return self._data._box_func(max_stamp)

    def argmax(self, axis=None, skipna=True, *args, **kwargs):
        '\n        Returns the indices of the maximum values along an axis.\n\n        See `numpy.ndarray.argmax` for more information on the\n        `axis` parameter.\n\n        See Also\n        --------\n        numpy.ndarray.argmax\n        '
        nv.validate_argmax(args, kwargs)
        nv.validate_minmax_axis(axis)
        i8 = self.asi8
        if self.hasnans:
            mask = self._isnan
            if (mask.all() or (not skipna)):
                return (- 1)
            i8 = i8.copy()
            i8[mask] = 0
        return i8.argmax()

    def format(self, name=False, formatter=None, na_rep='NaT', date_format=None):
        '\n        Render a string representation of the Index.\n        '
        header = []
        if name:
            header.append((ibase.pprint_thing(self.name, escape_chars=('\t', '\r', '\n')) if (self.name is not None) else ''))
        if (formatter is not None):
            return (header + list(self.map(formatter)))
        return self._format_with_header(header, na_rep=na_rep, date_format=date_format)

    def _format_with_header(self, header, na_rep='NaT', date_format=None):
        return (header + list(self._format_native_types(na_rep=na_rep, date_format=date_format)))

    @property
    def _formatter_func(self):
        return self._data._formatter()

    def _format_attrs(self):
        '\n        Return a list of tuples of the (attr,formatted_value).\n        '
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            if (attrib == 'freq'):
                freq = self.freqstr
                if (freq is not None):
                    freq = repr(freq)
                attrs.append(('freq', freq))
        return attrs

    def _summary(self, name=None):
        '\n        Return a summarized representation.\n\n        Parameters\n        ----------\n        name : str\n            Name to use in the summary representation.\n\n        Returns\n        -------\n        str\n            Summarized representation of the index.\n        '
        formatter = self._formatter_func
        if (len(self) > 0):
            index_summary = f', {formatter(self[0])} to {formatter(self[(- 1)])}'
        else:
            index_summary = ''
        if (name is None):
            name = type(self).__name__
        result = f'{name}: {len(self)} entries{index_summary}'
        if self.freq:
            result += f'''
Freq: {self.freqstr}'''
        result = result.replace("'", '')
        return result

    def _validate_partial_date_slice(self, reso):
        raise NotImplementedError

    def _parsed_string_to_bounds(self, reso, parsed):
        raise NotImplementedError

    def _partial_date_slice(self, reso, parsed):
        '\n        Parameters\n        ----------\n        reso : Resolution\n        parsed : datetime\n\n        Returns\n        -------\n        slice or ndarray[intp]\n        '
        self._validate_partial_date_slice(reso)
        (t1, t2) = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox
        if self.is_monotonic_increasing:
            if (len(self) and (((t1 < self[0]) and (t2 < self[0])) or ((t1 > self[(- 1)]) and (t2 > self[(- 1)])))):
                raise KeyError
            left = vals.searchsorted(unbox(t1), side='left')
            right = vals.searchsorted(unbox(t2), side='right')
            return slice(left, right)
        else:
            lhs_mask = (vals >= unbox(t1))
            rhs_mask = (vals <= unbox(t2))
            return (lhs_mask & rhs_mask).nonzero()[0]
    __add__ = make_wrapped_arith_op('__add__')
    __sub__ = make_wrapped_arith_op('__sub__')
    __radd__ = make_wrapped_arith_op('__radd__')
    __rsub__ = make_wrapped_arith_op('__rsub__')
    __pow__ = make_wrapped_arith_op('__pow__')
    __rpow__ = make_wrapped_arith_op('__rpow__')
    __mul__ = make_wrapped_arith_op('__mul__')
    __rmul__ = make_wrapped_arith_op('__rmul__')
    __floordiv__ = make_wrapped_arith_op('__floordiv__')
    __rfloordiv__ = make_wrapped_arith_op('__rfloordiv__')
    __mod__ = make_wrapped_arith_op('__mod__')
    __rmod__ = make_wrapped_arith_op('__rmod__')
    __divmod__ = make_wrapped_arith_op('__divmod__')
    __rdivmod__ = make_wrapped_arith_op('__rdivmod__')
    __truediv__ = make_wrapped_arith_op('__truediv__')
    __rtruediv__ = make_wrapped_arith_op('__rtruediv__')

    def shift(self, periods=1, freq=None):
        "\n        Shift index by desired number of time frequency increments.\n\n        This method is for shifting the values of datetime-like indexes\n        by a specified time increment a given number of times.\n\n        Parameters\n        ----------\n        periods : int, default 1\n            Number of periods (or increments) to shift by,\n            can be positive or negative.\n\n            .. versionchanged:: 0.24.0\n\n        freq : pandas.DateOffset, pandas.Timedelta or string, optional\n            Frequency increment to shift by.\n            If None, the index is shifted by its own `freq` attribute.\n            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.\n\n        Returns\n        -------\n        pandas.DatetimeIndex\n            Shifted index.\n\n        See Also\n        --------\n        Index.shift : Shift values of Index.\n        PeriodIndex.shift : Shift values of PeriodIndex.\n        "
        arr = self._data.view()
        arr._freq = self.freq
        result = arr._time_shift(periods, freq=freq)
        return type(self)(result, name=self.name)

    def _get_delete_freq(self, loc):
        '\n        Find the `freq` for self.delete(loc).\n        '
        freq = None
        if is_period_dtype(self.dtype):
            freq = self.freq
        elif (self.freq is not None):
            if is_integer(loc):
                if (loc in (0, (- len(self)), (- 1), (len(self) - 1))):
                    freq = self.freq
            else:
                if is_list_like(loc):
                    loc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=np.intp), len(self))
                if (isinstance(loc, slice) and (loc.step in (1, None))):
                    if ((loc.start in (0, None)) or (loc.stop in (len(self), None))):
                        freq = self.freq
        return freq

    def _get_insert_freq(self, loc, item):
        '\n        Find the `freq` for self.insert(loc, item).\n        '
        value = self._data._validate_scalar(item)
        item = self._data._box_func(value)
        freq = None
        if is_period_dtype(self.dtype):
            freq = self.freq
        elif (self.freq is not None):
            if self.size:
                if (item is NaT):
                    pass
                elif (((loc == 0) or (loc == (- len(self)))) and ((item + self.freq) == self[0])):
                    freq = self.freq
                elif ((loc == len(self)) and ((item - self.freq) == self[(- 1)])):
                    freq = self.freq
            elif self.freq.is_on_offset(item):
                freq = self.freq
        return freq

    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self, loc):
        result = super().delete(loc)
        result._data._freq = self._get_delete_freq(loc)
        return result

    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc, item):
        result = super().insert(loc, item)
        result._data._freq = self._get_insert_freq(loc, item)
        return result

    def _get_join_freq(self, other):
        '\n        Get the freq to attach to the result of a join operation.\n        '
        if is_period_dtype(self.dtype):
            freq = self.freq
        else:
            self = cast(DatetimeTimedeltaMixin, self)
            freq = (self.freq if self._can_fast_union(other) else None)
        return freq

    def _wrap_joined_index(self, joined, other):
        assert (other.dtype == self.dtype), (other.dtype, self.dtype)
        result = super()._wrap_joined_index(joined, other)
        result._data._freq = self._get_join_freq(other)
        return result

    @doc(Index._convert_arr_indexer)
    def _convert_arr_indexer(self, keyarr):
        try:
            return self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            return com.asarray_tuplesafe(keyarr)

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, Int64Index):
    '\n    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,\n    but not PeriodIndex\n    '
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    def _with_freq(self, freq):
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self.name)

    @property
    def _has_complex_internals(self):
        return False

    def is_type_compatible(self, kind):
        return (kind in self._data._infer_matches)

    @Appender(Index.difference.__doc__)
    def difference(self, other, sort=None):
        new_idx = super().difference(other, sort=sort)._with_freq(None)
        return new_idx

    def _intersection(self, other, sort=False):
        '\n        intersection specialized to the case with matching dtypes.\n        '
        other = cast('DatetimeTimedeltaMixin', other)
        if (len(self) == 0):
            return self.copy()._get_reconciled_name_object(other)
        if (len(other) == 0):
            return other.copy()._get_reconciled_name_object(self)
        elif (not self._can_fast_intersect(other)):
            result = Index._intersection(self, other, sort=sort)
            result = self._wrap_setop_result(other, result)
            return result._with_freq(None)._with_freq('infer')
        if (self[0] <= other[0]):
            (left, right) = (self, other)
        else:
            (left, right) = (other, self)
        end = min(left[(- 1)], right[(- 1)])
        start = right[0]
        if (end < start):
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            left_chunk = left._values[lslice]
            result = type(self)._simple_new(left_chunk)
        return self._wrap_setop_result(other, result)

    def _can_fast_intersect(self, other):
        if (self.freq is None):
            return False
        elif (other.freq != self.freq):
            return False
        elif (not self.is_monotonic_increasing):
            return False
        elif self.freq.is_anchored():
            return True
        elif ((not len(self)) or (not len(other))):
            return False
        elif isinstance(self.freq, Tick):
            diff = (self[0] - other[0])
            remainder = (diff % self.freq.delta)
            return (remainder == Timedelta(0))
        return True

    def _can_fast_union(self, other):
        if (not isinstance(other, type(self))):
            return False
        freq = self.freq
        if ((freq is None) or (freq != other.freq)):
            return False
        if (not self.is_monotonic_increasing):
            return False
        if ((len(self) == 0) or (len(other) == 0)):
            return True
        if (self[0] <= other[0]):
            (left, right) = (self, other)
        else:
            (left, right) = (other, self)
        right_start = right[0]
        left_end = left[(- 1)]
        return ((right_start == (left_end + freq)) or (right_start in left))

    def _fast_union(self, other, sort=None):
        if (len(other) == 0):
            return self.view(type(self))
        if (len(self) == 0):
            return other.view(type(self))
        if (self[0] <= other[0]):
            (left, right) = (self, other)
        elif (sort is False):
            (left, right) = (self, other)
            left_start = left[0]
            loc = right.searchsorted(left_start, side='left')
            right_chunk = right._values[:loc]
            dates = concat_compat((left._values, right_chunk))
            result = self._shallow_copy(dates)._with_freq('infer')
            return result
        else:
            (left, right) = (other, self)
        left_end = left[(- 1)]
        right_end = right[(- 1)]
        if (left_end < right_end):
            loc = right.searchsorted(left_end, side='right')
            right_chunk = right._values[loc:]
            dates = concat_compat([left._values, right_chunk])
            dates = type(self._data)(dates, freq=self.freq)
            result = type(self)._simple_new(dates)
            return result
        else:
            return left

    def _union(self, other, sort):
        assert isinstance(other, type(self))
        (this, other) = self._maybe_utc_convert(other)
        if this._can_fast_union(other):
            result = this._fast_union(other, sort=sort)
            if (sort is None):
                assert (result.freq == self.freq), (result.freq, self.freq)
            elif (result.freq is None):
                result = result._with_freq('infer')
            return result
        else:
            i8self = Int64Index._simple_new(self.asi8)
            i8other = Int64Index._simple_new(other.asi8)
            i8result = i8self._union(i8other, sort=sort)
            result = type(self)(i8result, dtype=self.dtype, freq='infer')
            return result
    _join_precedence = 10
    _inner_indexer = _join_i8_wrapper(libjoin.inner_join_indexer)
    _outer_indexer = _join_i8_wrapper(libjoin.outer_join_indexer)
    _left_indexer = _join_i8_wrapper(libjoin.left_join_indexer)
    _left_indexer_unique = _join_i8_wrapper(libjoin.left_join_indexer_unique, with_indexers=False)

    def join(self, other, how='left', level=None, return_indexers=False, sort=False):
        '\n        See Index.join\n        '
        (pself, pother) = self._maybe_promote(other)
        if ((pself is not self) or (pother is not other)):
            return pself.join(pother, how=how, level=level, return_indexers=return_indexers, sort=sort)
        (this, other) = self._maybe_utc_convert(other)
        return Index.join(this, other, how=how, level=level, return_indexers=return_indexers, sort=sort)

    def _maybe_utc_convert(self, other):
        return (self, other)

    @Appender(DatetimeIndexOpsMixin.insert.__doc__)
    def insert(self, loc, item):
        if isinstance(item, str):
            return self.astype(object).insert(loc, item)
        return DatetimeIndexOpsMixin.insert(self, loc, item)
