
from __future__ import annotations
from datetime import datetime, timedelta
import operator
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple, Type, TypeVar, Union, cast
import warnings
import numpy as np
from pandas._libs import algos, lib
from pandas._libs.tslibs import BaseOffset, NaT, NaTType, Period, Resolution, Tick, Timestamp, delta_to_nanoseconds, iNaT, to_offset
from pandas._libs.tslibs.timestamps import RoundTo, integer_op_not_supported, round_nsint64
from pandas._typing import DatetimeLikeScalar, Dtype, DtypeObj, NpDtype
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError, NullFrequencyError, PerformanceWarning
from pandas.util._decorators import Appender, Substitution, cache_readonly
from pandas.core.dtypes.common import is_categorical_dtype, is_datetime64_any_dtype, is_datetime64_dtype, is_datetime64tz_dtype, is_datetime_or_timedelta_dtype, is_dtype_equal, is_extension_array_dtype, is_float_dtype, is_integer_dtype, is_list_like, is_object_dtype, is_period_dtype, is_string_dtype, is_timedelta64_dtype, is_unsigned_integer_dtype, pandas_dtype
from pandas.core.dtypes.missing import is_valid_nat_for_dtype, isna
from pandas.core import nanops, ops
from pandas.core.algorithms import checked_add_with_arr, isin, unique1d, value_counts
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray, ravel_compat
import pandas.core.common as com
from pandas.core.construction import array, extract_array
from pandas.core.indexers import check_array_indexer, check_setitem_lengths
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import invalid_comparison, make_invalid_op
from pandas.tseries import frequencies
if TYPE_CHECKING:
    from pandas.core.arrays import DatetimeArray, TimedeltaArray
DTScalarOrNaT = Union[(DatetimeLikeScalar, NaTType)]
DatetimeLikeArrayT = TypeVar('DatetimeLikeArrayT', bound='DatetimeLikeArrayMixin')

class InvalidComparison(Exception):
    '\n    Raised by _validate_comparison_value to indicate to caller it should\n    return invalid_comparison.\n    '
    pass

class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray):
    '\n    Shared Base/Mixin class for DatetimeArray, TimedeltaArray, PeriodArray\n\n    Assumes that __new__/__init__ defines:\n        _data\n        _freq\n\n    and that the inheriting class has methods:\n        _generate_range\n    '

    def __init__(self, data, dtype=None, freq=None, copy=False):
        raise AbstractMethodError(self)

    @classmethod
    def _simple_new(cls, values, freq=None, dtype=None):
        raise AbstractMethodError(cls)

    @property
    def _scalar_type(self):
        '\n        The scalar associated with this datelike\n\n        * PeriodArray : Period\n        * DatetimeArray : Timestamp\n        * TimedeltaArray : Timedelta\n        '
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value):
        '\n        Construct a scalar type from a string.\n\n        Parameters\n        ----------\n        value : str\n\n        Returns\n        -------\n        Period, Timestamp, or Timedelta, or NaT\n            Whatever the type of ``self._scalar_type`` is.\n\n        Notes\n        -----\n        This should call ``self._check_compatible_with`` before\n        unboxing the result.\n        '
        raise AbstractMethodError(self)

    def _unbox_scalar(self, value, setitem=False):
        '\n        Unbox the integer value of a scalar `value`.\n\n        Parameters\n        ----------\n        value : Period, Timestamp, Timedelta, or NaT\n            Depending on subclass.\n        setitem : bool, default False\n            Whether to check compatibility with setitem strictness.\n\n        Returns\n        -------\n        int\n\n        Examples\n        --------\n        >>> self._unbox_scalar(Timedelta("10s"))  # doctest: +SKIP\n        10000000000\n        '
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other, setitem=False):
        '\n        Verify that `self` and `other` are compatible.\n\n        * DatetimeArray verifies that the timezones (if any) match\n        * PeriodArray verifies that the freq matches\n        * Timedelta has no verification\n\n        In each case, NaT is considered compatible.\n\n        Parameters\n        ----------\n        other\n        setitem : bool, default False\n            For __setitem__ we may have stricter compatibility restrictions than\n            for comparisons.\n\n        Raises\n        ------\n        Exception\n        '
        raise AbstractMethodError(self)

    @cache_readonly
    def _ndarray(self):
        return self._data

    def _from_backing_data(self, arr):
        return type(self)._simple_new(arr, dtype=self.dtype)

    def _box_func(self, x):
        '\n        box function to get object from internal representation\n        '
        raise AbstractMethodError(self)

    def _box_values(self, values):
        '\n        apply box func to passed values\n        '
        return lib.map_infer(values, self._box_func)

    def __iter__(self):
        if (self.ndim > 1):
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self):
        '\n        Integer representation of the values.\n\n        Returns\n        -------\n        ndarray\n            An ndarray with int64 dtype.\n        '
        return self._data.view('i8')

    def _format_native_types(self, na_rep='NaT', date_format=None):
        '\n        Helper method for astype when converting to strings.\n\n        Returns\n        -------\n        ndarray[str]\n        '
        raise AbstractMethodError(self)

    def _formatter(self, boxed=False):
        return "'{}'".format

    def __array__(self, dtype=None):
        if is_object_dtype(dtype):
            return np.array(list(self), dtype=object)
        return self._ndarray

    def __getitem__(self, key):
        '\n        This getitem defers to the underlying array, which by-definition can\n        only handle list-likes, slices, and integer scalars\n        '
        result = super().__getitem__(key)
        if lib.is_scalar(result):
            return result
        result._freq = self._get_getitem_freq(key)
        return result

    def _get_getitem_freq(self, key):
        '\n        Find the `freq` attribute to assign to the result of a __getitem__ lookup.\n        '
        is_period = is_period_dtype(self.dtype)
        if is_period:
            freq = self.freq
        elif (self.ndim != 1):
            freq = None
        else:
            key = check_array_indexer(self, key)
            freq = None
            if isinstance(key, slice):
                if ((self.freq is not None) and (key.step is not None)):
                    freq = (key.step * self.freq)
                else:
                    freq = self.freq
            elif (key is Ellipsis):
                freq = self.freq
            elif com.is_bool_indexer(key):
                new_key = lib.maybe_booleans_to_slice(key.view(np.uint8))
                if isinstance(new_key, slice):
                    return self._get_getitem_freq(new_key)
        return freq

    def __setitem__(self, key, value):
        no_op = check_setitem_lengths(key, value, self)
        if no_op:
            return
        super().__setitem__(key, value)
        self._maybe_clear_freq()

    def _maybe_clear_freq(self):
        pass

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_object_dtype(dtype):
            return self._box_values(self.asi8.ravel()).reshape(self.shape)
        elif (is_string_dtype(dtype) and (not is_categorical_dtype(dtype))):
            if is_extension_array_dtype(dtype):
                arr_cls = dtype.construct_array_type()
                return arr_cls._from_sequence(self, dtype=dtype, copy=copy)
            else:
                return self._format_native_types()
        elif is_integer_dtype(dtype):
            warnings.warn(f'casting {self.dtype} values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.', FutureWarning, stacklevel=3)
            values = self.asi8
            if is_unsigned_integer_dtype(dtype):
                values = values.view('uint64')
            if copy:
                values = values.copy()
            return values
        elif ((is_datetime_or_timedelta_dtype(dtype) and (not is_dtype_equal(self.dtype, dtype))) or is_float_dtype(dtype)):
            msg = f'Cannot cast {type(self).__name__} to dtype {dtype}'
            raise TypeError(msg)
        elif is_categorical_dtype(dtype):
            arr_cls = dtype.construct_array_type()
            return arr_cls(self, dtype=dtype)
        else:
            return np.asarray(self, dtype=dtype)

    def view(self, dtype=None):
        if ((dtype is None) or (dtype is self.dtype)):
            return type(self)(self._ndarray, dtype=self.dtype)
        return self._ndarray.view(dtype=dtype)

    @classmethod
    def _concat_same_type(cls, to_concat, axis=0):
        new_obj = super()._concat_same_type(to_concat, axis)
        obj = to_concat[0]
        dtype = obj.dtype
        new_freq = None
        if is_period_dtype(dtype):
            new_freq = obj.freq
        elif (axis == 0):
            to_concat = [x for x in to_concat if len(x)]
            if ((obj.freq is not None) and all(((x.freq == obj.freq) for x in to_concat))):
                pairs = zip(to_concat[:(- 1)], to_concat[1:])
                if all((((pair[0][(- 1)] + obj.freq) == pair[1][0]) for pair in pairs)):
                    new_freq = obj.freq
        new_obj._freq = new_freq
        return new_obj

    def copy(self):
        new_obj = super().copy()
        new_obj._freq = self.freq
        return new_obj

    def _values_for_factorize(self):
        return (self._ndarray, iNaT)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    def _validate_comparison_value(self, other):
        if isinstance(other, str):
            try:
                other = self._scalar_from_string(other)
            except ValueError:
                raise InvalidComparison(other)
        if (isinstance(other, self._recognized_scalars) or (other is NaT)):
            other = self._scalar_type(other)
            try:
                self._check_compatible_with(other)
            except TypeError as err:
                raise InvalidComparison(other) from err
        elif (not is_list_like(other)):
            raise InvalidComparison(other)
        elif (len(other) != len(self)):
            raise ValueError('Lengths must match')
        else:
            try:
                other = self._validate_listlike(other, allow_object=True)
                self._check_compatible_with(other)
            except TypeError as err:
                if is_object_dtype(getattr(other, 'dtype', None)):
                    pass
                else:
                    raise InvalidComparison(other) from err
        return other

    def _validate_fill_value(self, fill_value):
        '\n        If a fill_value is passed to `take` convert it to an i8 representation,\n        raising TypeError if this is not possible.\n\n        Parameters\n        ----------\n        fill_value : object\n\n        Returns\n        -------\n        fill_value : np.int64, np.datetime64, or np.timedelta64\n\n        Raises\n        ------\n        TypeError\n        '
        return self._validate_scalar(fill_value)

    def _validate_shift_value(self, fill_value):
        if is_valid_nat_for_dtype(fill_value, self.dtype):
            fill_value = NaT
        elif isinstance(fill_value, self._recognized_scalars):
            fill_value = self._scalar_type(fill_value)
        else:
            if ((self._scalar_type is Period) and lib.is_integer(fill_value)):
                new_fill = Period._from_ordinal(fill_value, freq=self.freq)
            else:
                new_fill = self._scalar_type(fill_value)
            warnings.warn(f'Passing {type(fill_value)} to shift is deprecated and will raise in a future version, pass {self._scalar_type.__name__} instead.', FutureWarning, stacklevel=8)
            fill_value = new_fill
        return self._unbox(fill_value, setitem=True)

    def _validate_scalar(self, value, *, allow_listlike=False, setitem=True, unbox=True):
        '\n        Validate that the input value can be cast to our scalar_type.\n\n        Parameters\n        ----------\n        value : object\n        allow_listlike: bool, default False\n            When raising an exception, whether the message should say\n            listlike inputs are allowed.\n        setitem : bool, default True\n            Whether to check compatibility with setitem strictness.\n        unbox : bool, default True\n            Whether to unbox the result before returning.  Note: unbox=False\n            skips the setitem compatibility check.\n\n        Returns\n        -------\n        self._scalar_type or NaT\n        '
        if isinstance(value, str):
            try:
                value = self._scalar_from_string(value)
            except ValueError as err:
                msg = self._validation_error_message(value, allow_listlike)
                raise TypeError(msg) from err
        elif is_valid_nat_for_dtype(value, self.dtype):
            value = NaT
        elif isinstance(value, self._recognized_scalars):
            value = self._scalar_type(value)
        else:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)
        if (not unbox):
            return value
        return self._unbox_scalar(value, setitem=setitem)

    def _validation_error_message(self, value, allow_listlike=False):
        '\n        Construct an exception message on validation error.\n\n        Some methods allow only scalar inputs, while others allow either scalar\n        or listlike.\n\n        Parameters\n        ----------\n        allow_listlike: bool, default False\n\n        Returns\n        -------\n        str\n        '
        if allow_listlike:
            msg = f"value should be a '{self._scalar_type.__name__}', 'NaT', or array of those. Got '{type(value).__name__}' instead."
        else:
            msg = f"value should be a '{self._scalar_type.__name__}' or 'NaT'. Got '{type(value).__name__}' instead."
        return msg

    def _validate_listlike(self, value, allow_object=False):
        if isinstance(value, type(self)):
            return value
        if (isinstance(value, list) and (len(value) == 0)):
            return type(self)._from_sequence([], dtype=self.dtype)
        value = array(value)
        value = extract_array(value, extract_numpy=True)
        if is_dtype_equal(value.dtype, 'string'):
            try:
                value = type(self)._from_sequence(value, dtype=self.dtype)
            except ValueError:
                pass
        if is_categorical_dtype(value.dtype):
            if is_dtype_equal(value.categories.dtype, self.dtype):
                value = value._internal_get_values()
                value = extract_array(value, extract_numpy=True)
        if (allow_object and is_object_dtype(value.dtype)):
            pass
        elif (not type(self)._is_recognized_dtype(value.dtype)):
            msg = self._validation_error_message(value, True)
            raise TypeError(msg)
        return value

    def _validate_searchsorted_value(self, value):
        if (not is_list_like(value)):
            return self._validate_scalar(value, allow_listlike=True, setitem=False)
        else:
            value = self._validate_listlike(value)
        return self._unbox(value)

    def _validate_setitem_value(self, value):
        if is_list_like(value):
            value = self._validate_listlike(value)
        else:
            return self._validate_scalar(value, allow_listlike=True)
        return self._unbox(value, setitem=True)

    def _unbox(self, other, setitem=False):
        '\n        Unbox either a scalar with _unbox_scalar or an instance of our own type.\n        '
        if lib.is_scalar(other):
            other = self._unbox_scalar(other, setitem=setitem)
        else:
            self._check_compatible_with(other, setitem=setitem)
            other = other._ndarray
        return other

    def value_counts(self, dropna=False):
        "\n        Return a Series containing counts of unique values.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include counts of NaT values.\n\n        Returns\n        -------\n        Series\n        "
        if (self.ndim != 1):
            raise NotImplementedError
        from pandas import Index, Series
        if dropna:
            values = self[(~ self.isna())]._ndarray
        else:
            values = self._ndarray
        cls = type(self)
        result = value_counts(values, sort=False, dropna=dropna)
        index = Index(cls(result.index.view('i8'), dtype=self.dtype), name=result.index.name)
        return Series(result._values, index=index, name=result.name)

    @ravel_compat
    def map(self, mapper):
        from pandas import Index
        return Index(self).map(mapper).array

    def isin(self, values):
        '\n        Compute boolean array of whether each value is found in the\n        passed set of values.\n\n        Parameters\n        ----------\n        values : set or sequence of values\n\n        Returns\n        -------\n        ndarray[bool]\n        '
        if (not hasattr(values, 'dtype')):
            values = np.asarray(values)
        if (values.dtype.kind in ['f', 'i', 'u', 'c']):
            return np.zeros(self.shape, dtype=bool)
        if (not isinstance(values, type(self))):
            inferable = ['timedelta', 'timedelta64', 'datetime', 'datetime64', 'date', 'period']
            if (values.dtype == object):
                inferred = lib.infer_dtype(values, skipna=False)
                if (inferred not in inferable):
                    if (inferred == 'string'):
                        pass
                    elif ('mixed' in inferred):
                        return isin(self.astype(object), values)
                    else:
                        return np.zeros(self.shape, dtype=bool)
            try:
                values = type(self)._from_sequence(values)
            except ValueError:
                return isin(self.astype(object), values)
        try:
            self._check_compatible_with(values)
        except (TypeError, ValueError):
            return np.zeros(self.shape, dtype=bool)
        return isin(self.asi8, values.asi8)

    def isna(self):
        return self._isnan

    @property
    def _isnan(self):
        '\n        return if each value is nan\n        '
        return (self.asi8 == iNaT)

    @property
    def _hasnans(self):
        '\n        return if I have any nans; enables various perf speedups\n        '
        return bool(self._isnan.any())

    def _maybe_mask_results(self, result, fill_value=iNaT, convert=None):
        '\n        Parameters\n        ----------\n        result : np.ndarray\n        fill_value : object, default iNaT\n        convert : str, dtype or None\n\n        Returns\n        -------\n        result : ndarray with values replace by the fill_value\n\n        mask the result if needed, convert to the provided dtype if its not\n        None\n\n        This is an internal routine.\n        '
        if self._hasnans:
            if convert:
                result = result.astype(convert)
            if (fill_value is None):
                fill_value = np.nan
            np.putmask(result, self._isnan, fill_value)
        return result

    @property
    def freq(self):
        '\n        Return the frequency object if it is set, otherwise None.\n        '
        return self._freq

    @freq.setter
    def freq(self, value):
        if (value is not None):
            value = to_offset(value)
            self._validate_frequency(self, value)
            if (self.ndim > 1):
                raise ValueError('Cannot set freq with ndim > 1')
        self._freq = value

    @property
    def freqstr(self):
        '\n        Return the frequency object as a string if its set, otherwise None.\n        '
        if (self.freq is None):
            return None
        return self.freq.freqstr

    @property
    def inferred_freq(self):
        "\n        Tries to return a string representing a frequency guess,\n        generated by infer_freq.  Returns None if it can't autodetect the\n        frequency.\n        "
        if (self.ndim != 1):
            return None
        try:
            return frequencies.infer_freq(self)
        except ValueError:
            return None

    @property
    def _resolution_obj(self):
        try:
            return Resolution.get_reso_from_freq(self.freqstr)
        except KeyError:
            return None

    @property
    def resolution(self):
        '\n        Returns day, hour, minute, second, millisecond or microsecond\n        '
        return self._resolution_obj.attrname

    @classmethod
    def _validate_frequency(cls, index, freq, **kwargs):
        '\n        Validate that a frequency is compatible with the values of a given\n        Datetime Array/Index or Timedelta Array/Index\n\n        Parameters\n        ----------\n        index : DatetimeIndex or TimedeltaIndex\n            The index on which to determine if the given frequency is valid\n        freq : DateOffset\n            The frequency to validate\n        '
        inferred = index.inferred_freq
        if ((index.size == 0) or (inferred == freq.freqstr)):
            return None
        try:
            on_freq = cls._generate_range(start=index[0], end=None, periods=len(index), freq=freq, **kwargs)
            if (not np.array_equal(index.asi8, on_freq.asi8)):
                raise ValueError
        except ValueError as e:
            if ('non-fixed' in str(e)):
                raise e
            raise ValueError(f'Inferred frequency {inferred} from passed values does not conform to passed frequency {freq.freqstr}') from e

    @classmethod
    def _generate_range(cls, start, end, periods, freq, *args, **kwargs):
        raise AbstractMethodError(cls)

    @property
    def _is_monotonic_increasing(self):
        return algos.is_monotonic(self.asi8, timelike=True)[0]

    @property
    def _is_monotonic_decreasing(self):
        return algos.is_monotonic(self.asi8, timelike=True)[1]

    @property
    def _is_unique(self):
        return (len(unique1d(self.asi8.ravel('K'))) == self.size)

    def _cmp_method(self, other, op):
        if ((self.ndim > 1) and (getattr(other, 'shape', None) == self.shape)):
            return op(self.ravel(), other.ravel()).reshape(self.shape)
        try:
            other = self._validate_comparison_value(other)
        except InvalidComparison:
            return invalid_comparison(self, other, op)
        dtype = getattr(other, 'dtype', None)
        if is_object_dtype(dtype):
            with np.errstate(all='ignore'):
                result = ops.comp_method_OBJECT_ARRAY(op, np.asarray(self.astype(object)), other)
            return result
        other_vals = self._unbox(other)
        result = op(self._ndarray.view('i8'), other_vals.view('i8'))
        o_mask = isna(other)
        mask = (self._isnan | o_mask)
        if mask.any():
            nat_result = (op is operator.ne)
            np.putmask(result, mask, nat_result)
        return result
    __pow__ = make_invalid_op('__pow__')
    __rpow__ = make_invalid_op('__rpow__')
    __mul__ = make_invalid_op('__mul__')
    __rmul__ = make_invalid_op('__rmul__')
    __truediv__ = make_invalid_op('__truediv__')
    __rtruediv__ = make_invalid_op('__rtruediv__')
    __floordiv__ = make_invalid_op('__floordiv__')
    __rfloordiv__ = make_invalid_op('__rfloordiv__')
    __mod__ = make_invalid_op('__mod__')
    __rmod__ = make_invalid_op('__rmod__')
    __divmod__ = make_invalid_op('__divmod__')
    __rdivmod__ = make_invalid_op('__rdivmod__')

    def _add_datetimelike_scalar(self, other):
        raise TypeError(f'cannot add {type(self).__name__} and {type(other).__name__}')
    _add_datetime_arraylike = _add_datetimelike_scalar

    def _sub_datetimelike_scalar(self, other):
        assert (other is not NaT)
        raise TypeError(f'cannot subtract a datelike from a {type(self).__name__}')
    _sub_datetime_arraylike = _sub_datetimelike_scalar

    def _sub_period(self, other):
        raise TypeError(f'cannot subtract Period from a {type(self).__name__}')

    def _add_period(self, other):
        raise TypeError(f'cannot add Period to a {type(self).__name__}')

    def _add_offset(self, offset):
        raise AbstractMethodError(self)

    def _add_timedeltalike_scalar(self, other):
        '\n        Add a delta of a timedeltalike\n\n        Returns\n        -------\n        Same type as self\n        '
        if isna(other):
            new_values = np.empty(self.shape, dtype='i8')
            new_values.fill(iNaT)
            return type(self)(new_values, dtype=self.dtype)
        inc = delta_to_nanoseconds(other)
        new_values = checked_add_with_arr(self.asi8, inc, arr_mask=self._isnan).view('i8')
        new_values = self._maybe_mask_results(new_values)
        new_freq = None
        if (isinstance(self.freq, Tick) or is_period_dtype(self.dtype)):
            new_freq = self.freq
        return type(self)._simple_new(new_values, dtype=self.dtype, freq=new_freq)

    def _add_timedelta_arraylike(self, other):
        '\n        Add a delta of a TimedeltaIndex\n\n        Returns\n        -------\n        Same type as self\n        '
        if (len(self) != len(other)):
            raise ValueError('cannot add indices of unequal length')
        if isinstance(other, np.ndarray):
            from pandas.core.arrays import TimedeltaArray
            other = TimedeltaArray._from_sequence(other)
        self_i8 = self.asi8
        other_i8 = other.asi8
        new_values = checked_add_with_arr(self_i8, other_i8, arr_mask=self._isnan, b_mask=other._isnan)
        if (self._hasnans or other._hasnans):
            mask = (self._isnan | other._isnan)
            np.putmask(new_values, mask, iNaT)
        return type(self)(new_values, dtype=self.dtype)

    def _add_nat(self):
        '\n        Add pd.NaT to self\n        '
        if is_period_dtype(self.dtype):
            raise TypeError(f'Cannot add {type(self).__name__} and {type(NaT).__name__}')
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        return type(self)(result, dtype=self.dtype, freq=None)

    def _sub_nat(self):
        '\n        Subtract pd.NaT from self\n        '
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        return result.view('timedelta64[ns]')

    def _sub_period_array(self, other):
        raise TypeError(f'cannot subtract {other.dtype}-dtype from {type(self).__name__}')

    def _addsub_object_array(self, other, op):
        '\n        Add or subtract array-like of DateOffset objects\n\n        Parameters\n        ----------\n        other : np.ndarray[object]\n        op : {operator.add, operator.sub}\n\n        Returns\n        -------\n        result : same class as self\n        '
        assert (op in [operator.add, operator.sub])
        if ((len(other) == 1) and (self.ndim == 1)):
            return op(self, other[0])
        warnings.warn(f'Adding/subtracting object-dtype array to {type(self).__name__} not vectorized', PerformanceWarning)
        assert (self.shape == other.shape), (self.shape, other.shape)
        res_values = op(self.astype('O'), np.asarray(other))
        result = array(res_values.ravel())
        result = extract_array(result, extract_numpy=True).reshape(self.shape)
        return result

    def _time_shift(self, periods, freq=None):
        '\n        Shift each value by `periods`.\n\n        Note this is different from ExtensionArray.shift, which\n        shifts the *position* of each element, padding the end with\n        missing values.\n\n        Parameters\n        ----------\n        periods : int\n            Number of periods to shift by.\n        freq : pandas.DateOffset, pandas.Timedelta, or str\n            Frequency increment to shift by.\n        '
        if ((freq is not None) and (freq != self.freq)):
            if isinstance(freq, str):
                freq = to_offset(freq)
            offset = (periods * freq)
            return (self + offset)
        if ((periods == 0) or (len(self) == 0)):
            return self.copy()
        if (self.freq is None):
            raise NullFrequencyError('Cannot shift with no freq')
        start = (self[0] + (periods * self.freq))
        end = (self[(- 1)] + (periods * self.freq))
        return self._generate_range(start=start, end=end, periods=None, freq=self.freq)

    @unpack_zerodim_and_defer('__add__')
    def __add__(self, other):
        other_dtype = getattr(other, 'dtype', None)
        if (other is NaT):
            result = self._add_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(other)
        elif isinstance(other, BaseOffset):
            result = self._add_offset(other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._add_datetimelike_scalar(other)
        elif (isinstance(other, Period) and is_timedelta64_dtype(self.dtype)):
            result = self._add_period(other)
        elif lib.is_integer(other):
            if (not is_period_dtype(self.dtype)):
                raise integer_op_not_supported(self)
            result = self._time_shift(other)
        elif is_timedelta64_dtype(other_dtype):
            result = self._add_timedelta_arraylike(other)
        elif is_object_dtype(other_dtype):
            result = self._addsub_object_array(other, operator.add)
        elif (is_datetime64_dtype(other_dtype) or is_datetime64tz_dtype(other_dtype)):
            return self._add_datetime_arraylike(other)
        elif is_integer_dtype(other_dtype):
            if (not is_period_dtype(self.dtype)):
                raise integer_op_not_supported(self)
            result = self._addsub_int_array(other, operator.add)
        else:
            return NotImplemented
        if (isinstance(result, np.ndarray) and is_timedelta64_dtype(result.dtype)):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray(result)
        return result

    def __radd__(self, other):
        return self.__add__(other)

    @unpack_zerodim_and_defer('__sub__')
    def __sub__(self, other):
        other_dtype = getattr(other, 'dtype', None)
        if (other is NaT):
            result = self._sub_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar((- other))
        elif isinstance(other, BaseOffset):
            result = self._add_offset((- other))
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._sub_datetimelike_scalar(other)
        elif lib.is_integer(other):
            if (not is_period_dtype(self.dtype)):
                raise integer_op_not_supported(self)
            result = self._time_shift((- other))
        elif isinstance(other, Period):
            result = self._sub_period(other)
        elif is_timedelta64_dtype(other_dtype):
            result = self._add_timedelta_arraylike((- other))
        elif is_object_dtype(other_dtype):
            result = self._addsub_object_array(other, operator.sub)
        elif (is_datetime64_dtype(other_dtype) or is_datetime64tz_dtype(other_dtype)):
            result = self._sub_datetime_arraylike(other)
        elif is_period_dtype(other_dtype):
            result = self._sub_period_array(other)
        elif is_integer_dtype(other_dtype):
            if (not is_period_dtype(self.dtype)):
                raise integer_op_not_supported(self)
            result = self._addsub_int_array(other, operator.sub)
        else:
            return NotImplemented
        if (isinstance(result, np.ndarray) and is_timedelta64_dtype(result.dtype)):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray(result)
        return result

    def __rsub__(self, other):
        other_dtype = getattr(other, 'dtype', None)
        if (is_datetime64_any_dtype(other_dtype) and is_timedelta64_dtype(self.dtype)):
            if lib.is_scalar(other):
                return (Timestamp(other) - self)
            if (not isinstance(other, DatetimeLikeArrayMixin)):
                from pandas.core.arrays import DatetimeArray
                other = DatetimeArray(other)
            return (other - self)
        elif (is_datetime64_any_dtype(self.dtype) and hasattr(other, 'dtype') and (not is_datetime64_any_dtype(other.dtype))):
            raise TypeError(f'cannot subtract {type(self).__name__} from {type(other).__name__}')
        elif (is_period_dtype(self.dtype) and is_timedelta64_dtype(other_dtype)):
            raise TypeError(f'cannot subtract {type(self).__name__} from {other.dtype}')
        elif is_timedelta64_dtype(self.dtype):
            self = cast('TimedeltaArray', self)
            return ((- self) + other)
        return (- (self - other))

    def __iadd__(self, other):
        result = (self + other)
        self[:] = result[:]
        if (not is_period_dtype(self.dtype)):
            self._freq = result._freq
        return self

    def __isub__(self, other):
        result = (self - other)
        self[:] = result[:]
        if (not is_period_dtype(self.dtype)):
            self._freq = result._freq
        return self

    def min(self, *, axis=None, skipna=True, **kwargs):
        '\n        Return the minimum value of the Array or minimum along\n        an axis.\n\n        See Also\n        --------\n        numpy.ndarray.min\n        Index.min : Return the minimum value in an Index.\n        Series.min : Return the minimum value in a Series.\n        '
        nv.validate_min((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)
        if is_period_dtype(self.dtype):
            result = nanops.nanmin(self._ndarray.view('M8[ns]'), axis=axis, skipna=skipna)
            if (result is NaT):
                return NaT
            result = result.view('i8')
            if ((axis is None) or (self.ndim == 1)):
                return self._box_func(result)
            return self._from_backing_data(result)
        result = nanops.nanmin(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def max(self, *, axis=None, skipna=True, **kwargs):
        '\n        Return the maximum value of the Array or maximum along\n        an axis.\n\n        See Also\n        --------\n        numpy.ndarray.max\n        Index.max : Return the maximum value in an Index.\n        Series.max : Return the maximum value in a Series.\n        '
        nv.validate_max((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)
        if is_period_dtype(self.dtype):
            result = nanops.nanmax(self._ndarray.view('M8[ns]'), axis=axis, skipna=skipna)
            if (result is NaT):
                return result
            result = result.view('i8')
            if ((axis is None) or (self.ndim == 1)):
                return self._box_func(result)
            return self._from_backing_data(result)
        result = nanops.nanmax(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def mean(self, *, skipna=True, axis=0):
        '\n        Return the mean value of the Array.\n\n        .. versionadded:: 0.25.0\n\n        Parameters\n        ----------\n        skipna : bool, default True\n            Whether to ignore any NaT elements.\n        axis : int, optional, default 0\n\n        Returns\n        -------\n        scalar\n            Timestamp or Timedelta.\n\n        See Also\n        --------\n        numpy.ndarray.mean : Returns the average of array elements along a given axis.\n        Series.mean : Return the mean value in a Series.\n\n        Notes\n        -----\n        mean is only defined for Datetime and Timedelta dtypes, not for Period.\n        '
        if is_period_dtype(self.dtype):
            raise TypeError(f"mean is not implemented for {type(self).__name__} since the meaning is ambiguous.  An alternative is obj.to_timestamp(how='start').mean()")
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())
        return self._wrap_reduction_result(axis, result)

    def median(self, *, axis=None, skipna=True, **kwargs):
        nv.validate_median((), kwargs)
        if ((axis is not None) and (abs(axis) >= self.ndim)):
            raise ValueError('abs(axis) must be less than ndim')
        if is_period_dtype(self.dtype):
            result = nanops.nanmedian(self._ndarray.view('M8[ns]'), axis=axis, skipna=skipna)
            result = result.view('i8')
            if ((axis is None) or (self.ndim == 1)):
                return self._box_func(result)
            return self._from_backing_data(result)
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

class DatelikeOps(DatetimeLikeArrayMixin):
    '\n    Common ops for DatetimeIndex/PeriodIndex, but not TimedeltaIndex.\n    '

    @Substitution(URL='https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior')
    def strftime(self, date_format):
        '\n        Convert to Index using specified date_format.\n\n        Return an Index of formatted strings specified by date_format, which\n        supports the same string format as the python standard library. Details\n        of the string format can be found in `python string format\n        doc <%(URL)s>`__.\n\n        Parameters\n        ----------\n        date_format : str\n            Date format string (e.g. "%%Y-%%m-%%d").\n\n        Returns\n        -------\n        ndarray\n            NumPy ndarray of formatted strings.\n\n        See Also\n        --------\n        to_datetime : Convert the given argument to datetime.\n        DatetimeIndex.normalize : Return DatetimeIndex with times to midnight.\n        DatetimeIndex.round : Round the DatetimeIndex to the specified freq.\n        DatetimeIndex.floor : Floor the DatetimeIndex to the specified freq.\n\n        Examples\n        --------\n        >>> rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"),\n        ...                     periods=3, freq=\'s\')\n        >>> rng.strftime(\'%%B %%d, %%Y, %%r\')\n        Index([\'March 10, 2018, 09:00:00 AM\', \'March 10, 2018, 09:00:01 AM\',\n               \'March 10, 2018, 09:00:02 AM\'],\n              dtype=\'object\')\n        '
        result = self._format_native_types(date_format=date_format, na_rep=np.nan)
        return result.astype(object)
_round_doc = "\n    Perform {op} operation on the data to the specified `freq`.\n\n    Parameters\n    ----------\n    freq : str or Offset\n        The frequency level to {op} the index to. Must be a fixed\n        frequency like 'S' (second) not 'ME' (month end). See\n        :ref:`frequency aliases <timeseries.offset_aliases>` for\n        a list of possible `freq` values.\n    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'\n        Only relevant for DatetimeIndex:\n\n        - 'infer' will attempt to infer fall dst-transition hours based on\n          order\n        - bool-ndarray where True signifies a DST time, False designates\n          a non-DST time (note that this flag is only applicable for\n          ambiguous times)\n        - 'NaT' will return NaT where there are ambiguous times\n        - 'raise' will raise an AmbiguousTimeError if there are ambiguous\n          times.\n\n        .. versionadded:: 0.24.0\n\n    nonexistent : 'shift_forward', 'shift_backward', 'NaT', timedelta, default 'raise'\n        A nonexistent time does not exist in a particular timezone\n        where clocks moved forward due to DST.\n\n        - 'shift_forward' will shift the nonexistent time forward to the\n          closest existing time\n        - 'shift_backward' will shift the nonexistent time backward to the\n          closest existing time\n        - 'NaT' will return NaT where there are nonexistent times\n        - timedelta objects will shift nonexistent times by the timedelta\n        - 'raise' will raise an NonExistentTimeError if there are\n          nonexistent times.\n\n        .. versionadded:: 0.24.0\n\n    Returns\n    -------\n    DatetimeIndex, TimedeltaIndex, or Series\n        Index of the same type for a DatetimeIndex or TimedeltaIndex,\n        or a Series with the same index for a Series.\n\n    Raises\n    ------\n    ValueError if the `freq` cannot be converted.\n\n    Examples\n    --------\n    **DatetimeIndex**\n\n    >>> rng = pd.date_range('1/1/2018 11:59:00', periods=3, freq='min')\n    >>> rng\n    DatetimeIndex(['2018-01-01 11:59:00', '2018-01-01 12:00:00',\n                   '2018-01-01 12:01:00'],\n                  dtype='datetime64[ns]', freq='T')\n    "
_round_example = '>>> rng.round(\'H\')\n    DatetimeIndex([\'2018-01-01 12:00:00\', \'2018-01-01 12:00:00\',\n                   \'2018-01-01 12:00:00\'],\n                  dtype=\'datetime64[ns]\', freq=None)\n\n    **Series**\n\n    >>> pd.Series(rng).dt.round("H")\n    0   2018-01-01 12:00:00\n    1   2018-01-01 12:00:00\n    2   2018-01-01 12:00:00\n    dtype: datetime64[ns]\n    '
_floor_example = '>>> rng.floor(\'H\')\n    DatetimeIndex([\'2018-01-01 11:00:00\', \'2018-01-01 12:00:00\',\n                   \'2018-01-01 12:00:00\'],\n                  dtype=\'datetime64[ns]\', freq=None)\n\n    **Series**\n\n    >>> pd.Series(rng).dt.floor("H")\n    0   2018-01-01 11:00:00\n    1   2018-01-01 12:00:00\n    2   2018-01-01 12:00:00\n    dtype: datetime64[ns]\n    '
_ceil_example = '>>> rng.ceil(\'H\')\n    DatetimeIndex([\'2018-01-01 12:00:00\', \'2018-01-01 12:00:00\',\n                   \'2018-01-01 13:00:00\'],\n                  dtype=\'datetime64[ns]\', freq=None)\n\n    **Series**\n\n    >>> pd.Series(rng).dt.ceil("H")\n    0   2018-01-01 12:00:00\n    1   2018-01-01 12:00:00\n    2   2018-01-01 13:00:00\n    dtype: datetime64[ns]\n    '

class TimelikeOps(DatetimeLikeArrayMixin):
    '\n    Common ops for TimedeltaIndex/DatetimeIndex, but not PeriodIndex.\n    '

    def _round(self, freq, mode, ambiguous, nonexistent):
        if is_datetime64tz_dtype(self.dtype):
            self = cast('DatetimeArray', self)
            naive = self.tz_localize(None)
            result = naive._round(freq, mode, ambiguous, nonexistent)
            return result.tz_localize(self.tz, ambiguous=ambiguous, nonexistent=nonexistent)
        values = self.view('i8')
        result = round_nsint64(values, mode, freq)
        result = self._maybe_mask_results(result, fill_value=NaT)
        return self._simple_new(result, dtype=self.dtype)

    @Appender((_round_doc + _round_example).format(op='round'))
    def round(self, freq, ambiguous='raise', nonexistent='raise'):
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent)

    @Appender((_round_doc + _floor_example).format(op='floor'))
    def floor(self, freq, ambiguous='raise', nonexistent='raise'):
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    @Appender((_round_doc + _ceil_example).format(op='ceil'))
    def ceil(self, freq, ambiguous='raise', nonexistent='raise'):
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    def any(self, *, axis=None, skipna=True):
        return nanops.nanany(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def all(self, *, axis=None, skipna=True):
        return nanops.nanall(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def _maybe_clear_freq(self):
        self._freq = None

    def _with_freq(self, freq):
        '\n        Helper to get a view on the same data, with a new freq.\n\n        Parameters\n        ----------\n        freq : DateOffset, None, or "infer"\n\n        Returns\n        -------\n        Same type as self\n        '
        if (freq is None):
            pass
        elif ((len(self) == 0) and isinstance(freq, BaseOffset)):
            pass
        else:
            assert (freq == 'infer')
            freq = to_offset(self.inferred_freq)
        arr = self.view()
        arr._freq = freq
        return arr

    def factorize(self, na_sentinel=(- 1), sort=False):
        if (self.freq is not None):
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self.copy()
            if (sort and (self.freq.n < 0)):
                codes = codes[::(- 1)]
                uniques = uniques[::(- 1)]
            return (codes, uniques)
        return super().factorize(na_sentinel=na_sentinel)

def validate_periods(periods):
    '\n    If a `periods` argument is passed to the Datetime/Timedelta Array/Index\n    constructor, cast it to an integer.\n\n    Parameters\n    ----------\n    periods : None, float, int\n\n    Returns\n    -------\n    periods : None or int\n\n    Raises\n    ------\n    TypeError\n        if periods is None, float, or int\n    '
    if (periods is not None):
        if lib.is_float(periods):
            periods = int(periods)
        elif (not lib.is_integer(periods)):
            raise TypeError(f'periods must be a number, got {periods}')
    return periods

def validate_endpoints(closed):
    '\n    Check that the `closed` argument is among [None, "left", "right"]\n\n    Parameters\n    ----------\n    closed : {None, "left", "right"}\n\n    Returns\n    -------\n    left_closed : bool\n    right_closed : bool\n\n    Raises\n    ------\n    ValueError : if argument is not among valid values\n    '
    left_closed = False
    right_closed = False
    if (closed is None):
        left_closed = True
        right_closed = True
    elif (closed == 'left'):
        left_closed = True
    elif (closed == 'right'):
        right_closed = True
    else:
        raise ValueError("Closed has to be either 'left', 'right' or None")
    return (left_closed, right_closed)

def validate_inferred_freq(freq, inferred_freq, freq_infer):
    '\n    If the user passes a freq and another freq is inferred from passed data,\n    require that they match.\n\n    Parameters\n    ----------\n    freq : DateOffset or None\n    inferred_freq : DateOffset or None\n    freq_infer : bool\n\n    Returns\n    -------\n    freq : DateOffset or None\n    freq_infer : bool\n\n    Notes\n    -----\n    We assume at this point that `maybe_infer_freq` has been called, so\n    `freq` is either a DateOffset object or None.\n    '
    if (inferred_freq is not None):
        if ((freq is not None) and (freq != inferred_freq)):
            raise ValueError(f'Inferred frequency {inferred_freq} from passed values does not conform to passed frequency {freq.freqstr}')
        elif (freq is None):
            freq = inferred_freq
        freq_infer = False
    return (freq, freq_infer)

def maybe_infer_freq(freq):
    '\n    Comparing a DateOffset to the string "infer" raises, so we need to\n    be careful about comparisons.  Make a dummy variable `freq_infer` to\n    signify the case where the given freq is "infer" and set freq to None\n    to avoid comparison trouble later on.\n\n    Parameters\n    ----------\n    freq : {DateOffset, None, str}\n\n    Returns\n    -------\n    freq : {DateOffset, None}\n    freq_infer : bool\n        Whether we should inherit the freq of passed data.\n    '
    freq_infer = False
    if (not isinstance(freq, BaseOffset)):
        if (freq != 'infer'):
            freq = to_offset(freq)
        else:
            freq_infer = True
            freq = None
    return (freq, freq_infer)
