
from datetime import timedelta
import operator
from sys import getsizeof
from typing import Any, List, Optional, Tuple
import warnings
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.lib import no_default
from pandas._typing import Label
from pandas.compat.numpy import function as nv
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import ensure_platform_int, ensure_python_int, is_float, is_integer, is_list_like, is_scalar, is_signed_integer_dtype, is_timedelta64_dtype
from pandas.core.dtypes.generic import ABCTimedeltaIndex
from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.numeric import Float64Index, Int64Index
from pandas.core.ops.common import unpack_zerodim_and_defer
_empty_range = range(0)

class RangeIndex(Int64Index):
    '\n    Immutable Index implementing a monotonic integer range.\n\n    RangeIndex is a memory-saving special case of Int64Index limited to\n    representing monotonic ranges. Using RangeIndex may in some instances\n    improve computing speed.\n\n    This is the default index type used\n    by DataFrame and Series when no explicit index is provided by the user.\n\n    Parameters\n    ----------\n    start : int (default: 0), or other RangeIndex instance\n        If int and "stop" is not given, interpreted as "stop" instead.\n    stop : int (default: 0)\n    step : int (default: 1)\n    dtype : np.int64\n        Unused, accepted for homogeneity with other index types.\n    copy : bool, default False\n        Unused, accepted for homogeneity with other index types.\n    name : object, optional\n        Name to be stored in the index.\n\n    Attributes\n    ----------\n    start\n    stop\n    step\n\n    Methods\n    -------\n    from_range\n\n    See Also\n    --------\n    Index : The base pandas Index type.\n    Int64Index : Index of int64 data.\n    '
    _typ = 'rangeindex'
    _engine_type = libindex.Int64Engine

    def __new__(cls, start=None, stop=None, step=None, dtype=None, copy=False, name=None):
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)
        if isinstance(start, RangeIndex):
            start = start._range
            return cls._simple_new(start, name=name)
        if com.all_none(start, stop, step):
            raise TypeError('RangeIndex(...) must be called with integers')
        start = (ensure_python_int(start) if (start is not None) else 0)
        if (stop is None):
            (start, stop) = (0, start)
        else:
            stop = ensure_python_int(stop)
        step = (ensure_python_int(step) if (step is not None) else 1)
        if (step == 0):
            raise ValueError('Step must not be zero')
        rng = range(start, stop, step)
        return cls._simple_new(rng, name=name)

    @classmethod
    def from_range(cls, data, name=None, dtype=None):
        '\n        Create RangeIndex from a range object.\n\n        Returns\n        -------\n        RangeIndex\n        '
        if (not isinstance(data, range)):
            raise TypeError(f'{cls.__name__}(...) must be called with object coercible to a range, {repr(data)} was passed')
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    @classmethod
    def _simple_new(cls, values, name=None):
        result = object.__new__(cls)
        assert isinstance(values, range)
        result._range = values
        result.name = name
        result._cache = {}
        result._reset_identity()
        return result

    @cache_readonly
    def _constructor(self):
        ' return the class to use for construction '
        return Int64Index

    @cache_readonly
    def _data(self):
        '\n        An int array that for performance reasons is created only when needed.\n\n        The constructed array is saved in ``_cache``.\n        '
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    @cache_readonly
    def _int64index(self):
        return Int64Index._simple_new(self._data, name=self.name)

    def _get_data_as_items(self):
        ' return a list of tuples of start, stop, step '
        rng = self._range
        return [('start', rng.start), ('stop', rng.stop), ('step', rng.step)]

    def __reduce__(self):
        d = self._get_attributes_dict()
        d.update(dict(self._get_data_as_items()))
        return (ibase._new_Index, (type(self), d), None)

    def _format_attrs(self):
        '\n        Return a list of tuples of the (attr, formatted_value)\n        '
        attrs = self._get_data_as_items()
        if (self.name is not None):
            attrs.append(('name', ibase.default_pprint(self.name)))
        return attrs

    def _format_data(self, name=None):
        return None

    def _format_with_header(self, header, na_rep='NaN'):
        if (not len(self._range)):
            return header
        first_val_str = str(self._range[0])
        last_val_str = str(self._range[(- 1)])
        max_length = max(len(first_val_str), len(last_val_str))
        return (header + [f'{x:<{max_length}}' for x in self._range])
    _deprecation_message = 'RangeIndex.{} is deprecated and will be removed in a future version. Use RangeIndex.{} instead'

    @cache_readonly
    def start(self):
        '\n        The value of the `start` parameter (``0`` if this was not supplied).\n        '
        return self._range.start

    @property
    def _start(self):
        '\n        The value of the `start` parameter (``0`` if this was not supplied).\n\n         .. deprecated:: 0.25.0\n            Use ``start`` instead.\n        '
        warnings.warn(self._deprecation_message.format('_start', 'start'), FutureWarning, stacklevel=2)
        return self.start

    @cache_readonly
    def stop(self):
        '\n        The value of the `stop` parameter.\n        '
        return self._range.stop

    @property
    def _stop(self):
        '\n        The value of the `stop` parameter.\n\n         .. deprecated:: 0.25.0\n            Use ``stop`` instead.\n        '
        warnings.warn(self._deprecation_message.format('_stop', 'stop'), FutureWarning, stacklevel=2)
        return self.stop

    @cache_readonly
    def step(self):
        '\n        The value of the `step` parameter (``1`` if this was not supplied).\n        '
        return self._range.step

    @property
    def _step(self):
        '\n        The value of the `step` parameter (``1`` if this was not supplied).\n\n         .. deprecated:: 0.25.0\n            Use ``step`` instead.\n        '
        warnings.warn(self._deprecation_message.format('_step', 'step'), FutureWarning, stacklevel=2)
        return self.step

    @cache_readonly
    def nbytes(self):
        '\n        Return the number of bytes in the underlying data.\n        '
        rng = self._range
        return (getsizeof(rng) + sum((getsizeof(getattr(rng, attr_name)) for attr_name in ['start', 'stop', 'step'])))

    def memory_usage(self, deep=False):
        '\n        Memory usage of my values\n\n        Parameters\n        ----------\n        deep : bool\n            Introspect the data deeply, interrogate\n            `object` dtypes for system-level memory consumption\n\n        Returns\n        -------\n        bytes used\n\n        Notes\n        -----\n        Memory usage does not include memory consumed by elements that\n        are not components of the array if deep=False\n\n        See Also\n        --------\n        numpy.ndarray.nbytes\n        '
        return self.nbytes

    @property
    def dtype(self):
        return np.dtype(np.int64)

    @property
    def is_unique(self):
        ' return if the index has unique values '
        return True

    @cache_readonly
    def is_monotonic_increasing(self):
        return ((self._range.step > 0) or (len(self) <= 1))

    @cache_readonly
    def is_monotonic_decreasing(self):
        return ((self._range.step < 0) or (len(self) <= 1))

    @property
    def has_duplicates(self):
        return False

    def __contains__(self, key):
        hash(key)
        try:
            key = ensure_python_int(key)
        except TypeError:
            return False
        return (key in self._range)

    @doc(Int64Index.get_loc)
    def get_loc(self, key, method=None, tolerance=None):
        if ((method is None) and (tolerance is None)):
            if (is_integer(key) or (is_float(key) and key.is_integer())):
                new_key = int(key)
                try:
                    return self._range.index(new_key)
                except ValueError as err:
                    raise KeyError(key) from err
            raise KeyError(key)
        return super().get_loc(key, method=method, tolerance=tolerance)

    def _get_indexer(self, target, method=None, limit=None, tolerance=None):
        if (com.any_not_none(method, tolerance, limit) or (not is_list_like(target))):
            return super()._get_indexer(target, method=method, tolerance=tolerance, limit=limit)
        if (self.step > 0):
            (start, stop, step) = (self.start, self.stop, self.step)
        else:
            reverse = self._range[::(- 1)]
            (start, stop, step) = (reverse.start, reverse.stop, reverse.step)
        target_array = np.asarray(target)
        if (not (is_signed_integer_dtype(target_array) and (target_array.ndim == 1))):
            return super()._get_indexer(target, method=method, tolerance=tolerance)
        locs = (target_array - start)
        valid = ((((locs % step) == 0) & (locs >= 0)) & (target_array < stop))
        locs[(~ valid)] = (- 1)
        locs[valid] = (locs[valid] / step)
        if (step != self.step):
            locs[valid] = ((len(self) - 1) - locs[valid])
        return ensure_platform_int(locs)

    def tolist(self):
        return list(self._range)

    @doc(Int64Index.__iter__)
    def __iter__(self):
        (yield from self._range)

    @doc(Int64Index._shallow_copy)
    def _shallow_copy(self, values=None, name=no_default):
        name = (self.name if (name is no_default) else name)
        if (values is not None):
            if (values.dtype.kind == 'f'):
                return Float64Index(values, name=name)
            return Int64Index._simple_new(values, name=name)
        result = self._simple_new(self._range, name=name)
        result._cache = self._cache
        return result

    @doc(Int64Index.copy)
    def copy(self, name=None, deep=False, dtype=None, names=None):
        name = self._validate_names(name=name, names=names, deep=deep)[0]
        new_index = self._shallow_copy(name=name)
        if dtype:
            warnings.warn('parameter dtype is deprecated and will be removed in a future version. Use the astype method instead.', FutureWarning, stacklevel=2)
            new_index = new_index.astype(dtype)
        return new_index

    def _minmax(self, meth):
        no_steps = (len(self) - 1)
        if (no_steps == (- 1)):
            return np.nan
        elif (((meth == 'min') and (self.step > 0)) or ((meth == 'max') and (self.step < 0))):
            return self.start
        return (self.start + (self.step * no_steps))

    def min(self, axis=None, skipna=True, *args, **kwargs):
        'The minimum value of the RangeIndex'
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax('min')

    def max(self, axis=None, skipna=True, *args, **kwargs):
        'The maximum value of the RangeIndex'
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return self._minmax('max')

    def argsort(self, *args, **kwargs):
        '\n        Returns the indices that would sort the index and its\n        underlying data.\n\n        Returns\n        -------\n        argsorted : numpy array\n\n        See Also\n        --------\n        numpy.ndarray.argsort\n        '
        nv.validate_argsort(args, kwargs)
        if (self._range.step > 0):
            return np.arange(len(self))
        else:
            return np.arange((len(self) - 1), (- 1), (- 1))

    def factorize(self, sort=False, na_sentinel=(- 1)):
        codes = np.arange(len(self), dtype=np.intp)
        uniques = self
        if (sort and (self.step < 0)):
            codes = codes[::(- 1)]
            uniques = uniques[::(- 1)]
        return (codes, uniques)

    def equals(self, other):
        '\n        Determines if two Index objects contain the same elements.\n        '
        if isinstance(other, RangeIndex):
            return (self._range == other._range)
        return super().equals(other)

    def _intersection(self, other, sort=False):
        if (not isinstance(other, RangeIndex)):
            return super()._intersection(other, sort=sort)
        if ((not len(self)) or (not len(other))):
            return self._simple_new(_empty_range)
        first = (self._range[::(- 1)] if (self.step < 0) else self._range)
        second = (other._range[::(- 1)] if (other.step < 0) else other._range)
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if (int_high <= int_low):
            return self._simple_new(_empty_range)
        (gcd, s, t) = self._extended_gcd(first.step, second.step)
        if ((first.start - second.start) % gcd):
            return self._simple_new(_empty_range)
        tmp_start = (first.start + ((((second.start - first.start) * first.step) // gcd) * s))
        new_step = ((first.step * second.step) // gcd)
        new_range = range(tmp_start, int_high, new_step)
        new_index = self._simple_new(new_range)
        new_start = new_index._min_fitting_element(int_low)
        new_range = range(new_start, new_index.stop, new_index.step)
        new_index = self._simple_new(new_range)
        if (((self.step < 0) and (other.step < 0)) is not (new_index.step < 0)):
            new_index = new_index[::(- 1)]
        if (sort is None):
            new_index = new_index.sort_values()
        return new_index

    def _min_fitting_element(self, lower_limit):
        'Returns the smallest element greater than or equal to the limit'
        no_steps = (- ((- (lower_limit - self.start)) // abs(self.step)))
        return (self.start + (abs(self.step) * no_steps))

    def _max_fitting_element(self, upper_limit):
        'Returns the largest element smaller than or equal to the limit'
        no_steps = ((upper_limit - self.start) // abs(self.step))
        return (self.start + (abs(self.step) * no_steps))

    def _extended_gcd(self, a, b):
        "\n        Extended Euclidean algorithms to solve Bezout's identity:\n           a*x + b*y = gcd(x, y)\n        Finds one particular solution for x, y: s, t\n        Returns: gcd, s, t\n        "
        (s, old_s) = (0, 1)
        (t, old_t) = (1, 0)
        (r, old_r) = (b, a)
        while r:
            quotient = (old_r // r)
            (old_r, r) = (r, (old_r - (quotient * r)))
            (old_s, s) = (s, (old_s - (quotient * s)))
            (old_t, t) = (t, (old_t - (quotient * t)))
        return (old_r, old_s, old_t)

    def _union(self, other, sort):
        '\n        Form the union of two Index objects and sorts if possible\n\n        Parameters\n        ----------\n        other : Index or array-like\n\n        sort : False or None, default None\n            Whether to sort resulting index. ``sort=None`` returns a\n            monotonically increasing ``RangeIndex`` if possible or a sorted\n            ``Int64Index`` if not. ``sort=False`` always returns an\n            unsorted ``Int64Index``\n\n            .. versionadded:: 0.25.0\n\n        Returns\n        -------\n        union : Index\n        '
        if (isinstance(other, RangeIndex) and (sort is None)):
            (start_s, step_s) = (self.start, self.step)
            end_s = (self.start + (self.step * (len(self) - 1)))
            (start_o, step_o) = (other.start, other.step)
            end_o = (other.start + (other.step * (len(other) - 1)))
            if (self.step < 0):
                (start_s, step_s, end_s) = (end_s, (- step_s), start_s)
            if (other.step < 0):
                (start_o, step_o, end_o) = (end_o, (- step_o), start_o)
            if ((len(self) == 1) and (len(other) == 1)):
                step_s = step_o = abs((self.start - other.start))
            elif (len(self) == 1):
                step_s = step_o
            elif (len(other) == 1):
                step_o = step_s
            start_r = min(start_s, start_o)
            end_r = max(end_s, end_o)
            if (step_o == step_s):
                if ((((start_s - start_o) % step_s) == 0) and ((start_s - end_o) <= step_s) and ((start_o - end_s) <= step_s)):
                    return type(self)(start_r, (end_r + step_s), step_s)
                if (((step_s % 2) == 0) and (abs((start_s - start_o)) <= (step_s / 2)) and (abs((end_s - end_o)) <= (step_s / 2))):
                    return type(self)(start_r, (end_r + (step_s / 2)), (step_s / 2))
            elif ((step_o % step_s) == 0):
                if ((((start_o - start_s) % step_s) == 0) and ((start_o + step_s) >= start_s) and ((end_o - step_s) <= end_s)):
                    return type(self)(start_r, (end_r + step_s), step_s)
            elif ((step_s % step_o) == 0):
                if ((((start_s - start_o) % step_o) == 0) and ((start_s + step_o) >= start_o) and ((end_s - step_o) <= end_o)):
                    return type(self)(start_r, (end_r + step_o), step_o)
        return self._int64index._union(other, sort=sort)

    def difference(self, other, sort=None):
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        (other, result_name) = self._convert_can_do_setop(other)
        if (not isinstance(other, RangeIndex)):
            return super().difference(other, sort=sort)
        res_name = ops.get_op_result_name(self, other)
        first = (self._range[::(- 1)] if (self.step < 0) else self._range)
        overlap = self.intersection(other)
        if (overlap.step < 0):
            overlap = overlap[::(- 1)]
        if (len(overlap) == 0):
            return self._shallow_copy(name=res_name)
        if (len(overlap) == len(self)):
            return self[:0].rename(res_name)
        if (not isinstance(overlap, RangeIndex)):
            return super().difference(other, sort=sort)
        if (overlap.step != first.step):
            return super().difference(other, sort=sort)
        if (overlap[0] == first.start):
            new_rng = range((overlap[(- 1)] + first.step), first.stop, first.step)
        elif (overlap[(- 1)] == first[(- 1)]):
            new_rng = range(first.start, overlap[0], first.step)
        else:
            return super().difference(other, sort=sort)
        new_index = type(self)._simple_new(new_rng, name=res_name)
        if (first is not self._range):
            new_index = new_index[::(- 1)]
        return new_index

    def symmetric_difference(self, other, result_name=None, sort=None):
        if ((not isinstance(other, RangeIndex)) or (sort is not None)):
            return super().symmetric_difference(other, result_name, sort)
        left = self.difference(other)
        right = other.difference(self)
        result = left.union(right)
        if (result_name is not None):
            result = result.rename(result_name)
        return result

    @doc(Int64Index.join)
    def join(self, other, how='left', level=None, return_indexers=False, sort=False):
        if ((how == 'outer') and (self is not other)):
            return self._int64index.join(other, how, level, return_indexers, sort)
        return super().join(other, how, level, return_indexers, sort)

    def _concat(self, indexes, name):
        '\n        Overriding parent method for the case of all RangeIndex instances.\n\n        When all members of "indexes" are of type RangeIndex: result will be\n        RangeIndex if possible, Int64Index otherwise. E.g.:\n        indexes = [RangeIndex(3), RangeIndex(3, 6)] -> RangeIndex(6)\n        indexes = [RangeIndex(3), RangeIndex(4, 6)] -> Int64Index([0,1,2,4,5])\n        '
        if (not all((isinstance(x, RangeIndex) for x in indexes))):
            return super()._concat(indexes, name)
        start = step = next_ = None
        non_empty_indexes = [obj for obj in indexes if len(obj)]
        for obj in non_empty_indexes:
            rng: range = obj._range
            if (start is None):
                start = rng.start
                if ((step is None) and (len(rng) > 1)):
                    step = rng.step
            elif (step is None):
                if (rng.start == start):
                    result = Int64Index(np.concatenate([x._values for x in indexes]))
                    return result.rename(name)
                step = (rng.start - start)
            non_consecutive = (((step != rng.step) and (len(rng) > 1)) or ((next_ is not None) and (rng.start != next_)))
            if non_consecutive:
                result = Int64Index(np.concatenate([x._values for x in indexes]))
                return result.rename(name)
            if (step is not None):
                next_ = (rng[(- 1)] + step)
        if non_empty_indexes:
            stop = (non_empty_indexes[(- 1)].stop if (next_ is None) else next_)
            return RangeIndex(start, stop, step).rename(name)
        return RangeIndex(0, 0).rename(name)

    def __len__(self):
        '\n        return the length of the RangeIndex\n        '
        return len(self._range)

    @property
    def size(self):
        return len(self)

    def __getitem__(self, key):
        '\n        Conserve RangeIndex type for scalar and slice keys.\n        '
        if isinstance(key, slice):
            new_range = self._range[key]
            return self._simple_new(new_range, name=self.name)
        elif is_integer(key):
            new_key = int(key)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(f'index {key} is out of bounds for axis 0 with size {len(self)}') from err
        elif is_scalar(key):
            raise IndexError('only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices')
        return super().__getitem__(key)

    @unpack_zerodim_and_defer('__floordiv__')
    def __floordiv__(self, other):
        if (is_integer(other) and (other != 0)):
            if ((len(self) == 0) or (((self.start % other) == 0) and ((self.step % other) == 0))):
                start = (self.start // other)
                step = (self.step // other)
                stop = (start + (len(self) * step))
                new_range = range(start, stop, (step or 1))
                return self._simple_new(new_range, name=self.name)
            if (len(self) == 1):
                start = (self.start // other)
                new_range = range(start, (start + 1), 1)
                return self._simple_new(new_range, name=self.name)
        return (self._int64index // other)

    def all(self, *args, **kwargs):
        return (0 not in self._range)

    def any(self, *args, **kwargs):
        return any(self._range)

    def _cmp_method(self, other, op):
        if (isinstance(other, RangeIndex) and (self._range == other._range)):
            return super()._cmp_method(self, op)
        return super()._cmp_method(other, op)

    def _arith_method(self, other, op):
        '\n        Parameters\n        ----------\n        other : Any\n        op : callable that accepts 2 params\n            perform the binary op\n        '
        if isinstance(other, ABCTimedeltaIndex):
            return NotImplemented
        elif isinstance(other, (timedelta, np.timedelta64)):
            return op(self._int64index, other)
        elif is_timedelta64_dtype(other):
            return op(self._int64index, other)
        if (op in [operator.pow, ops.rpow, operator.mod, ops.rmod, ops.rfloordiv, divmod, ops.rdivmod]):
            return op(self._int64index, other)
        step = False
        if (op in [operator.mul, ops.rmul, operator.truediv, ops.rtruediv]):
            step = op
        other = extract_array(other, extract_numpy=True)
        attrs = self._get_attributes_dict()
        (left, right) = (self, other)
        try:
            if step:
                with np.errstate(all='ignore'):
                    rstep = step(left.step, right)
                if ((not is_integer(rstep)) or (not rstep)):
                    raise ValueError
            else:
                rstep = left.step
            with np.errstate(all='ignore'):
                rstart = op(left.start, right)
                rstop = op(left.stop, right)
            result = type(self)(rstart, rstop, rstep, **attrs)
            if (not all((is_integer(x) for x in [rstart, rstop, rstep]))):
                result = result.astype('float64')
            return result
        except (ValueError, TypeError, ZeroDivisionError):
            return op(self._int64index, other)
