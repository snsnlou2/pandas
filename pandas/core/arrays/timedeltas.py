
from datetime import timedelta
from typing import List, Optional, Union
import numpy as np
from pandas._libs import lib, tslibs
from pandas._libs.tslibs import BaseOffset, NaT, NaTType, Period, Tick, Timedelta, Timestamp, iNaT, to_offset
from pandas._libs.tslibs.conversion import precision_from_unit
from pandas._libs.tslibs.fields import get_timedelta_field
from pandas._libs.tslibs.timedeltas import array_to_timedelta64, ints_to_pytimedelta, parse_timedelta_unit
from pandas.compat.numpy import function as nv
from pandas.core.dtypes.cast import astype_td64_unit_conversion
from pandas.core.dtypes.common import DT64NS_DTYPE, TD64NS_DTYPE, is_categorical_dtype, is_dtype_equal, is_float_dtype, is_integer_dtype, is_object_dtype, is_scalar, is_string_dtype, is_timedelta64_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries, ABCTimedeltaIndex
from pandas.core.dtypes.missing import isna
from pandas.core import nanops
from pandas.core.algorithms import checked_add_with_arr
from pandas.core.arrays import IntegerArray, datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer

def _field_accessor(name, alias, docstring):

    def f(self) -> np.ndarray:
        values = self.asi8
        result = get_timedelta_field(values, alias)
        if self._hasnans:
            result = self._maybe_mask_results(result, fill_value=None, convert='float64')
        return result
    f.__name__ = name
    f.__doc__ = f'''
{docstring}
'''
    return property(f)

class TimedeltaArray(dtl.TimelikeOps):
    '\n    Pandas ExtensionArray for timedelta data.\n\n    .. versionadded:: 0.24.0\n\n    .. warning::\n\n       TimedeltaArray is currently experimental, and its API may change\n       without warning. In particular, :attr:`TimedeltaArray.dtype` is\n       expected to change to be an instance of an ``ExtensionDtype``\n       subclass.\n\n    Parameters\n    ----------\n    values : array-like\n        The timedelta data.\n\n    dtype : numpy.dtype\n        Currently, only ``numpy.dtype("timedelta64[ns]")`` is accepted.\n    freq : Offset, optional\n    copy : bool, default False\n        Whether to copy the underlying array of data.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n    '
    _typ = 'timedeltaarray'
    _scalar_type = Timedelta
    _recognized_scalars = (timedelta, np.timedelta64, Tick)
    _is_recognized_dtype = is_timedelta64_dtype
    _infer_matches = ('timedelta', 'timedelta64')
    __array_priority__ = 1000
    _other_ops = []
    _bool_ops = []
    _object_ops = ['freq']
    _field_ops = ['days', 'seconds', 'microseconds', 'nanoseconds']
    _datetimelike_ops = ((_field_ops + _object_ops) + _bool_ops)
    _datetimelike_methods = ['to_pytimedelta', 'total_seconds', 'round', 'floor', 'ceil']

    def _box_func(self, x):
        return Timedelta(x, unit='ns')

    @property
    def dtype(self):
        '\n        The dtype for the TimedeltaArray.\n\n        .. warning::\n\n           A future version of pandas will change dtype to be an instance\n           of a :class:`pandas.api.extensions.ExtensionDtype` subclass,\n           not a ``numpy.dtype``.\n\n        Returns\n        -------\n        numpy.dtype\n        '
        return TD64NS_DTYPE

    def __init__(self, values, dtype=TD64NS_DTYPE, freq=lib.no_default, copy=False):
        values = extract_array(values)
        inferred_freq = getattr(values, '_freq', None)
        explicit_none = (freq is None)
        freq = (freq if (freq is not lib.no_default) else None)
        if isinstance(values, type(self)):
            if explicit_none:
                pass
            elif (freq is None):
                freq = values.freq
            elif (freq and values.freq):
                freq = to_offset(freq)
                (freq, _) = dtl.validate_inferred_freq(freq, values.freq, False)
            values = values._data
        if (not isinstance(values, np.ndarray)):
            msg = f"Unexpected type '{type(values).__name__}'. 'values' must be a TimedeltaArray ndarray, or Series or Index containing one of those."
            raise ValueError(msg)
        if (values.ndim not in [1, 2]):
            raise ValueError('Only 1-dimensional input arrays are supported.')
        if (values.dtype == 'i8'):
            values = values.view(TD64NS_DTYPE)
        _validate_td64_dtype(values.dtype)
        dtype = _validate_td64_dtype(dtype)
        if (freq == 'infer'):
            msg = "Frequency inference not allowed in TimedeltaArray.__init__. Use 'pd.array()' instead."
            raise ValueError(msg)
        if copy:
            values = values.copy()
        if freq:
            freq = to_offset(freq)
        self._data = values
        self._dtype = dtype
        self._freq = freq
        if ((inferred_freq is None) and (freq is not None)):
            type(self)._validate_frequency(self, freq)

    @classmethod
    def _simple_new(cls, values, freq=None, dtype=TD64NS_DTYPE):
        assert (dtype == TD64NS_DTYPE), dtype
        assert isinstance(values, np.ndarray), type(values)
        if (values.dtype != TD64NS_DTYPE):
            assert (values.dtype == 'i8')
            values = values.view(TD64NS_DTYPE)
        result = object.__new__(cls)
        result._data = values
        result._freq = to_offset(freq)
        result._dtype = TD64NS_DTYPE
        return result

    @classmethod
    def _from_sequence(cls, data, *, dtype=TD64NS_DTYPE, copy=False):
        if dtype:
            _validate_td64_dtype(dtype)
        (data, inferred_freq) = sequence_to_td64ns(data, copy=copy, unit=None)
        (freq, _) = dtl.validate_inferred_freq(None, inferred_freq, False)
        return cls._simple_new(data, freq=freq)

    @classmethod
    def _from_sequence_not_strict(cls, data, dtype=TD64NS_DTYPE, copy=False, freq=lib.no_default, unit=None):
        if dtype:
            _validate_td64_dtype(dtype)
        explicit_none = (freq is None)
        freq = (freq if (freq is not lib.no_default) else None)
        (freq, freq_infer) = dtl.maybe_infer_freq(freq)
        (data, inferred_freq) = sequence_to_td64ns(data, copy=copy, unit=unit)
        (freq, freq_infer) = dtl.validate_inferred_freq(freq, inferred_freq, freq_infer)
        if explicit_none:
            freq = None
        result = cls._simple_new(data, freq=freq)
        if ((inferred_freq is None) and (freq is not None)):
            cls._validate_frequency(result, freq)
        elif freq_infer:
            result._freq = to_offset(result.inferred_freq)
        return result

    @classmethod
    def _generate_range(cls, start, end, periods, freq, closed=None):
        periods = dtl.validate_periods(periods)
        if ((freq is None) and any(((x is None) for x in [periods, start, end]))):
            raise ValueError('Must provide freq argument if no data is supplied')
        if (com.count_not_none(start, end, periods, freq) != 3):
            raise ValueError('Of the four parameters: start, end, periods, and freq, exactly three must be specified')
        if (start is not None):
            start = Timedelta(start)
        if (end is not None):
            end = Timedelta(end)
        (left_closed, right_closed) = dtl.validate_endpoints(closed)
        if (freq is not None):
            index = generate_regular_range(start, end, periods, freq)
        else:
            index = np.linspace(start.value, end.value, periods).astype('i8')
        if (not left_closed):
            index = index[1:]
        if (not right_closed):
            index = index[:(- 1)]
        return cls._simple_new(index, freq=freq)

    def _unbox_scalar(self, value, setitem=False):
        if ((not isinstance(value, self._scalar_type)) and (value is not NaT)):
            raise ValueError("'value' should be a Timedelta.")
        self._check_compatible_with(value, setitem=setitem)
        return np.timedelta64(value.value, 'ns')

    def _scalar_from_string(self, value):
        return Timedelta(value)

    def _check_compatible_with(self, other, setitem=False):
        pass

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if (dtype.kind == 'm'):
            return astype_td64_unit_conversion(self._data, dtype, copy=copy)
        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy=copy)

    def __iter__(self):
        if (self.ndim > 1):
            for i in range(len(self)):
                (yield self[i])
        else:
            data = self.asi8
            length = len(self)
            chunksize = 10000
            chunks = (int((length / chunksize)) + 1)
            for i in range(chunks):
                start_i = (i * chunksize)
                end_i = min(((i + 1) * chunksize), length)
                converted = ints_to_pytimedelta(data[start_i:end_i], box=True)
                (yield from converted)

    def sum(self, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, skipna=True, min_count=0):
        nv.validate_sum((), {'dtype': dtype, 'out': out, 'keepdims': keepdims, 'initial': initial})
        result = nanops.nansum(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def std(self, *, axis=None, dtype=None, out=None, ddof=1, keepdims=False, skipna=True):
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='std')
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        if ((axis is None) or (self.ndim == 1)):
            return self._box_func(result)
        return self._from_backing_data(result)

    def _formatter(self, boxed=False):
        from pandas.io.formats.format import get_format_timedelta64
        return get_format_timedelta64(self, box=True)

    @dtl.ravel_compat
    def _format_native_types(self, na_rep='NaT', date_format=None, **kwargs):
        from pandas.io.formats.format import get_format_timedelta64
        formatter = get_format_timedelta64(self._data, na_rep)
        return np.array([formatter(x) for x in self._data])

    def _add_offset(self, other):
        assert (not isinstance(other, Tick))
        raise TypeError(f'cannot add the type {type(other).__name__} to a {type(self).__name__}')

    def _add_period(self, other):
        '\n        Add a Period object.\n        '
        from .period import PeriodArray
        i8vals = np.broadcast_to(other.ordinal, self.shape)
        oth = PeriodArray(i8vals, freq=other.freq)
        return (oth + self)

    def _add_datetime_arraylike(self, other):
        '\n        Add DatetimeArray/Index or ndarray[datetime64] to TimedeltaArray.\n        '
        if isinstance(other, np.ndarray):
            from pandas.core.arrays import DatetimeArray
            other = DatetimeArray(other)
        return (other + self)

    def _add_datetimelike_scalar(self, other):
        from pandas.core.arrays import DatetimeArray
        assert (other is not NaT)
        other = Timestamp(other)
        if (other is NaT):
            result = (self.asi8.view('m8[ms]') + NaT.to_datetime64())
            return DatetimeArray(result)
        i8 = self.asi8
        result = checked_add_with_arr(i8, other.value, arr_mask=self._isnan)
        result = self._maybe_mask_results(result)
        dtype = (DatetimeTZDtype(tz=other.tz) if other.tz else DT64NS_DTYPE)
        return DatetimeArray(result, dtype=dtype, freq=self.freq)

    def _addsub_object_array(self, other, op):
        try:
            return super()._addsub_object_array(other, op)
        except AttributeError as err:
            raise TypeError(f'Cannot add/subtract non-tick DateOffset to {type(self).__name__}') from err

    @unpack_zerodim_and_defer('__mul__')
    def __mul__(self, other):
        if is_scalar(other):
            result = (self._data * other)
            freq = None
            if ((self.freq is not None) and (not isna(other))):
                freq = (self.freq * other)
            return type(self)(result, freq=freq)
        if (not hasattr(other, 'dtype')):
            other = np.array(other)
        if ((len(other) != len(self)) and (not is_timedelta64_dtype(other.dtype))):
            raise ValueError('Cannot multiply with unequal lengths')
        if is_object_dtype(other.dtype):
            result = [(self[n] * other[n]) for n in range(len(self))]
            result = np.array(result)
            return type(self)(result)
        result = (self._data * other)
        return type(self)(result)
    __rmul__ = __mul__

    @unpack_zerodim_and_defer('__truediv__')
    def __truediv__(self, other):
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
            if (other is NaT):
                result = np.empty(self.shape, dtype=np.float64)
                result.fill(np.nan)
                return result
            return (self._data / other)
        elif lib.is_scalar(other):
            result = (self._data / other)
            freq = None
            if (self.freq is not None):
                freq = (self.freq.delta / other)
            return type(self)(result, freq=freq)
        if (not hasattr(other, 'dtype')):
            other = np.array(other)
        if (len(other) != len(self)):
            raise ValueError('Cannot divide vectors with unequal lengths')
        elif is_timedelta64_dtype(other.dtype):
            return (self._data / other)
        elif is_object_dtype(other.dtype):
            srav = self.ravel()
            orav = other.ravel()
            result = [(srav[n] / orav[n]) for n in range(len(srav))]
            result = np.array(result).reshape(self.shape)
            inferred = lib.infer_dtype(result)
            if (inferred == 'timedelta'):
                flat = result.ravel()
                result = type(self)._from_sequence(flat).reshape(result.shape)
            elif (inferred == 'floating'):
                result = result.astype(float)
            return result
        else:
            result = (self._data / other)
            return type(self)(result)

    @unpack_zerodim_and_defer('__rtruediv__')
    def __rtruediv__(self, other):
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
            if (other is NaT):
                result = np.empty(self.shape, dtype=np.float64)
                result.fill(np.nan)
                return result
            return (other / self._data)
        elif lib.is_scalar(other):
            raise TypeError(f'Cannot divide {type(other).__name__} by {type(self).__name__}')
        if (not hasattr(other, 'dtype')):
            other = np.array(other)
        if (len(other) != len(self)):
            raise ValueError('Cannot divide vectors with unequal lengths')
        elif is_timedelta64_dtype(other.dtype):
            return (other / self._data)
        elif is_object_dtype(other.dtype):
            result = [(other[n] / self[n]) for n in range(len(self))]
            return np.array(result)
        else:
            raise TypeError(f'Cannot divide {other.dtype} data by {type(self).__name__}')

    @unpack_zerodim_and_defer('__floordiv__')
    def __floordiv__(self, other):
        if is_scalar(other):
            if isinstance(other, self._recognized_scalars):
                other = Timedelta(other)
                if (other is NaT):
                    result = np.empty(self.shape, dtype=np.float64)
                    result.fill(np.nan)
                    return result
                result = other.__rfloordiv__(self._data)
                return result
            result = (self.asi8 // other)
            np.putmask(result, self._isnan, iNaT)
            freq = None
            if (self.freq is not None):
                freq = (self.freq / other)
                if ((freq.nanos == 0) and (self.freq.nanos != 0)):
                    freq = None
            return type(self)(result.view('m8[ns]'), freq=freq)
        if (not hasattr(other, 'dtype')):
            other = np.array(other)
        if (len(other) != len(self)):
            raise ValueError('Cannot divide with unequal lengths')
        elif is_timedelta64_dtype(other.dtype):
            other = type(self)(other)
            result = (self.asi8 // other.asi8)
            mask = (self._isnan | other._isnan)
            if mask.any():
                result = result.astype(np.float64)
                np.putmask(result, mask, np.nan)
            return result
        elif is_object_dtype(other.dtype):
            result = [(self[n] // other[n]) for n in range(len(self))]
            result = np.array(result)
            if (lib.infer_dtype(result, skipna=False) == 'timedelta'):
                (result, _) = sequence_to_td64ns(result)
                return type(self)(result)
            return result
        elif (is_integer_dtype(other.dtype) or is_float_dtype(other.dtype)):
            result = (self._data // other)
            return type(self)(result)
        else:
            dtype = getattr(other, 'dtype', type(other).__name__)
            raise TypeError(f'Cannot divide {dtype} by {type(self).__name__}')

    @unpack_zerodim_and_defer('__rfloordiv__')
    def __rfloordiv__(self, other):
        if is_scalar(other):
            if isinstance(other, self._recognized_scalars):
                other = Timedelta(other)
                if (other is NaT):
                    result = np.empty(self.shape, dtype=np.float64)
                    result.fill(np.nan)
                    return result
                result = other.__floordiv__(self._data)
                return result
            raise TypeError(f'Cannot divide {type(other).__name__} by {type(self).__name__}')
        if (not hasattr(other, 'dtype')):
            other = np.array(other)
        if (len(other) != len(self)):
            raise ValueError('Cannot divide with unequal lengths')
        elif is_timedelta64_dtype(other.dtype):
            other = type(self)(other)
            result = (other.asi8 // self.asi8)
            mask = (self._isnan | other._isnan)
            if mask.any():
                result = result.astype(np.float64)
                np.putmask(result, mask, np.nan)
            return result
        elif is_object_dtype(other.dtype):
            result = [(other[n] // self[n]) for n in range(len(self))]
            result = np.array(result)
            return result
        else:
            dtype = getattr(other, 'dtype', type(other).__name__)
            raise TypeError(f'Cannot divide {dtype} by {type(self).__name__}')

    @unpack_zerodim_and_defer('__mod__')
    def __mod__(self, other):
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return (self - ((self // other) * other))

    @unpack_zerodim_and_defer('__rmod__')
    def __rmod__(self, other):
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return (other - ((other // self) * self))

    @unpack_zerodim_and_defer('__divmod__')
    def __divmod__(self, other):
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        res1 = (self // other)
        res2 = (self - (res1 * other))
        return (res1, res2)

    @unpack_zerodim_and_defer('__rdivmod__')
    def __rdivmod__(self, other):
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        res1 = (other // self)
        res2 = (other - (res1 * self))
        return (res1, res2)

    def __neg__(self):
        if (self.freq is not None):
            return type(self)((- self._data), freq=(- self.freq))
        return type(self)((- self._data))

    def __pos__(self):
        return type(self)(self._data, freq=self.freq)

    def __abs__(self):
        return type(self)(np.abs(self._data))

    def total_seconds(self):
        "\n        Return total duration of each element expressed in seconds.\n\n        This method is available directly on TimedeltaArray, TimedeltaIndex\n        and on Series containing timedelta values under the ``.dt`` namespace.\n\n        Returns\n        -------\n        seconds : [ndarray, Float64Index, Series]\n            When the calling object is a TimedeltaArray, the return type\n            is ndarray.  When the calling object is a TimedeltaIndex,\n            the return type is a Float64Index. When the calling object\n            is a Series, the return type is Series of type `float64` whose\n            index is the same as the original.\n\n        See Also\n        --------\n        datetime.timedelta.total_seconds : Standard library version\n            of this method.\n        TimedeltaIndex.components : Return a DataFrame with components of\n            each Timedelta.\n\n        Examples\n        --------\n        **Series**\n\n        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='d'))\n        >>> s\n        0   0 days\n        1   1 days\n        2   2 days\n        3   3 days\n        4   4 days\n        dtype: timedelta64[ns]\n\n        >>> s.dt.total_seconds()\n        0         0.0\n        1     86400.0\n        2    172800.0\n        3    259200.0\n        4    345600.0\n        dtype: float64\n\n        **TimedeltaIndex**\n\n        >>> idx = pd.to_timedelta(np.arange(5), unit='d')\n        >>> idx\n        TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],\n                       dtype='timedelta64[ns]', freq=None)\n\n        >>> idx.total_seconds()\n        Float64Index([0.0, 86400.0, 172800.0, 259200.00000000003, 345600.0],\n                     dtype='float64')\n        "
        return self._maybe_mask_results((1e-09 * self.asi8), fill_value=None)

    def to_pytimedelta(self):
        '\n        Return Timedelta Array/Index as object ndarray of datetime.timedelta\n        objects.\n\n        Returns\n        -------\n        datetimes : ndarray\n        '
        return tslibs.ints_to_pytimedelta(self.asi8)
    days = _field_accessor('days', 'days', 'Number of days for each element.')
    seconds = _field_accessor('seconds', 'seconds', 'Number of seconds (>= 0 and less than 1 day) for each element.')
    microseconds = _field_accessor('microseconds', 'microseconds', 'Number of microseconds (>= 0 and less than 1 second) for each element.')
    nanoseconds = _field_accessor('nanoseconds', 'nanoseconds', 'Number of nanoseconds (>= 0 and less than 1 microsecond) for each element.')

    @property
    def components(self):
        '\n        Return a dataframe of the components (days, hours, minutes,\n        seconds, milliseconds, microseconds, nanoseconds) of the Timedeltas.\n\n        Returns\n        -------\n        a DataFrame\n        '
        from pandas import DataFrame
        columns = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds']
        hasnans = self._hasnans
        if hasnans:

            def f(x):
                if isna(x):
                    return ([np.nan] * len(columns))
                return x.components
        else:

            def f(x):
                return x.components
        result = DataFrame([f(x) for x in self], columns=columns)
        if (not hasnans):
            result = result.astype('int64')
        return result

def sequence_to_td64ns(data, copy=False, unit=None, errors='raise'):
    '\n    Parameters\n    ----------\n    data : list-like\n    copy : bool, default False\n    unit : str, optional\n        The timedelta unit to treat integers as multiples of. For numeric\n        data this defaults to ``\'ns\'``.\n        Must be un-specified if the data contains a str and ``errors=="raise"``.\n    errors : {"raise", "coerce", "ignore"}, default "raise"\n        How to handle elements that cannot be converted to timedelta64[ns].\n        See ``pandas.to_timedelta`` for details.\n\n    Returns\n    -------\n    converted : numpy.ndarray\n        The sequence converted to a numpy array with dtype ``timedelta64[ns]``.\n    inferred_freq : Tick or None\n        The inferred frequency of the sequence.\n\n    Raises\n    ------\n    ValueError : Data cannot be converted to timedelta64[ns].\n\n    Notes\n    -----\n    Unlike `pandas.to_timedelta`, if setting ``errors=ignore`` will not cause\n    errors to be ignored; they are caught and subsequently ignored at a\n    higher level.\n    '
    inferred_freq = None
    if (unit is not None):
        unit = parse_timedelta_unit(unit)
    if (not hasattr(data, 'dtype')):
        if (np.ndim(data) == 0):
            data = list(data)
        data = np.array(data, copy=False)
    elif isinstance(data, ABCSeries):
        data = data._values
    elif isinstance(data, (ABCTimedeltaIndex, TimedeltaArray)):
        inferred_freq = data.freq
        data = data._data
    elif isinstance(data, IntegerArray):
        data = data.to_numpy('int64', na_value=tslibs.iNaT)
    elif is_categorical_dtype(data.dtype):
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False
    if (is_object_dtype(data.dtype) or is_string_dtype(data.dtype)):
        data = objects_to_td64ns(data, unit=unit, errors=errors)
        copy = False
    elif is_integer_dtype(data.dtype):
        (data, copy_made) = ints_to_td64ns(data, unit=unit)
        copy = (copy and (not copy_made))
    elif is_float_dtype(data.dtype):
        mask = np.isnan(data)
        (m, p) = precision_from_unit((unit or 'ns'))
        base = data.astype(np.int64)
        frac = (data - base)
        if p:
            frac = np.round(frac, p)
        data = ((base * m) + (frac * m).astype(np.int64)).view('timedelta64[ns]')
        data[mask] = iNaT
        copy = False
    elif is_timedelta64_dtype(data.dtype):
        if (data.dtype != TD64NS_DTYPE):
            data = data.astype(TD64NS_DTYPE)
            copy = False
    else:
        raise TypeError(f'dtype {data.dtype} cannot be converted to timedelta64[ns]')
    data = np.array(data, copy=copy)
    assert (data.dtype == 'm8[ns]'), data
    return (data, inferred_freq)

def ints_to_td64ns(data, unit='ns'):
    '\n    Convert an ndarray with integer-dtype to timedelta64[ns] dtype, treating\n    the integers as multiples of the given timedelta unit.\n\n    Parameters\n    ----------\n    data : numpy.ndarray with integer-dtype\n    unit : str, default "ns"\n        The timedelta unit to treat integers as multiples of.\n\n    Returns\n    -------\n    numpy.ndarray : timedelta64[ns] array converted from data\n    bool : whether a copy was made\n    '
    copy_made = False
    unit = (unit if (unit is not None) else 'ns')
    if (data.dtype != np.int64):
        data = data.astype(np.int64)
        copy_made = True
    if (unit != 'ns'):
        dtype_str = f'timedelta64[{unit}]'
        data = data.view(dtype_str)
        data = data.astype('timedelta64[ns]')
        copy_made = True
    else:
        data = data.view('timedelta64[ns]')
    return (data, copy_made)

def objects_to_td64ns(data, unit=None, errors='raise'):
    '\n    Convert a object-dtyped or string-dtyped array into an\n    timedelta64[ns]-dtyped array.\n\n    Parameters\n    ----------\n    data : ndarray or Index\n    unit : str, default "ns"\n        The timedelta unit to treat integers as multiples of.\n        Must not be specified if the data contains a str.\n    errors : {"raise", "coerce", "ignore"}, default "raise"\n        How to handle elements that cannot be converted to timedelta64[ns].\n        See ``pandas.to_timedelta`` for details.\n\n    Returns\n    -------\n    numpy.ndarray : timedelta64[ns] array converted from data\n\n    Raises\n    ------\n    ValueError : Data cannot be converted to timedelta64[ns].\n\n    Notes\n    -----\n    Unlike `pandas.to_timedelta`, if setting `errors=ignore` will not cause\n    errors to be ignored; they are caught and subsequently ignored at a\n    higher level.\n    '
    values = np.array(data, dtype=np.object_, copy=False)
    result = array_to_timedelta64(values, unit=unit, errors=errors)
    return result.view('timedelta64[ns]')

def _validate_td64_dtype(dtype):
    dtype = pandas_dtype(dtype)
    if is_dtype_equal(dtype, np.dtype('timedelta64')):
        msg = "Passing in 'timedelta' dtype with no precision is not allowed. Please pass in 'timedelta64[ns]' instead."
        raise ValueError(msg)
    if (not is_dtype_equal(dtype, TD64NS_DTYPE)):
        raise ValueError(f'dtype {dtype} cannot be converted to timedelta64[ns]')
    return dtype
