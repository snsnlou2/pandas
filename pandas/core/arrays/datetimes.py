
from datetime import datetime, time, timedelta, tzinfo
from typing import Optional, Union, cast
import warnings
import numpy as np
from pandas._libs import lib, tslib
from pandas._libs.tslibs import BaseOffset, NaT, NaTType, Resolution, Timestamp, conversion, fields, get_resolution, iNaT, ints_to_pydatetime, is_date_array_normalized, normalize_i8_timestamps, timezones, to_offset, tzconversion
from pandas.errors import PerformanceWarning
from pandas.core.dtypes.cast import astype_dt64_to_dt64tz
from pandas.core.dtypes.common import DT64NS_DTYPE, INT64_DTYPE, is_bool_dtype, is_categorical_dtype, is_datetime64_any_dtype, is_datetime64_dtype, is_datetime64_ns_dtype, is_datetime64tz_dtype, is_dtype_equal, is_extension_array_dtype, is_float_dtype, is_object_dtype, is_period_dtype, is_sparse, is_string_dtype, is_timedelta64_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCIndex, ABCPandasArray, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.algorithms import checked_add_with_arr
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import BDay, Day, Tick
_midnight = time(0, 0)

def tz_to_dtype(tz):
    '\n    Return a datetime64[ns] dtype appropriate for the given timezone.\n\n    Parameters\n    ----------\n    tz : tzinfo or None\n\n    Returns\n    -------\n    np.dtype or Datetime64TZDType\n    '
    if (tz is None):
        return DT64NS_DTYPE
    else:
        return DatetimeTZDtype(tz=tz)

def _field_accessor(name, field, docstring=None):

    def f(self):
        values = self._local_timestamps()
        if (field in self._bool_ops):
            if field.endswith(('start', 'end')):
                freq = self.freq
                month_kw = 12
                if freq:
                    kwds = freq.kwds
                    month_kw = kwds.get('startingMonth', kwds.get('month', 12))
                result = fields.get_start_end_field(values, field, self.freqstr, month_kw)
            else:
                result = fields.get_date_field(values, field)
            return result
        if (field in self._object_ops):
            result = fields.get_date_name_field(values, field)
            result = self._maybe_mask_results(result, fill_value=None)
        else:
            result = fields.get_date_field(values, field)
            result = self._maybe_mask_results(result, fill_value=None, convert='float64')
        return result
    f.__name__ = name
    f.__doc__ = docstring
    return property(f)

class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):
    "\n    Pandas ExtensionArray for tz-naive or tz-aware datetime data.\n\n    .. versionadded:: 0.24.0\n\n    .. warning::\n\n       DatetimeArray is currently experimental, and its API may change\n       without warning. In particular, :attr:`DatetimeArray.dtype` is\n       expected to change to always be an instance of an ``ExtensionDtype``\n       subclass.\n\n    Parameters\n    ----------\n    values : Series, Index, DatetimeArray, ndarray\n        The datetime data.\n\n        For DatetimeArray `values` (or a Series or Index boxing one),\n        `dtype` and `freq` will be extracted from `values`.\n\n    dtype : numpy.dtype or DatetimeTZDtype\n        Note that the only NumPy dtype allowed is 'datetime64[ns]'.\n    freq : str or Offset, optional\n        The frequency.\n    copy : bool, default False\n        Whether to copy the underlying array of values.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n    "
    _typ = 'datetimearray'
    _scalar_type = Timestamp
    _recognized_scalars = (datetime, np.datetime64)
    _is_recognized_dtype = is_datetime64_any_dtype
    _infer_matches = ('datetime', 'datetime64', 'date')
    _bool_ops = ['is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_leap_year']
    _object_ops = ['freq', 'tz']
    _field_ops = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekofyear', 'week', 'weekday', 'dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'days_in_month', 'daysinmonth', 'microsecond', 'nanosecond']
    _other_ops = ['date', 'time', 'timetz']
    _datetimelike_ops = (((_field_ops + _object_ops) + _bool_ops) + _other_ops)
    _datetimelike_methods = ['to_period', 'tz_localize', 'tz_convert', 'normalize', 'strftime', 'round', 'floor', 'ceil', 'month_name', 'day_name']
    __array_priority__ = 1000
    _freq = None

    def __init__(self, values, dtype=DT64NS_DTYPE, freq=None, copy=False):
        if isinstance(values, (ABCSeries, ABCIndex)):
            values = values._values
        inferred_freq = getattr(values, '_freq', None)
        if isinstance(values, type(self)):
            dtz = getattr(dtype, 'tz', None)
            if (dtz and (values.tz is None)):
                dtype = DatetimeTZDtype(tz=dtype.tz)
            elif (dtz and values.tz):
                if (not timezones.tz_compare(dtz, values.tz)):
                    msg = f"Timezone of the array and 'dtype' do not match. '{dtz}' != '{values.tz}'"
                    raise TypeError(msg)
            elif values.tz:
                dtype = values.dtype
            if (freq is None):
                freq = values.freq
            values = values._data
        if (not isinstance(values, np.ndarray)):
            raise ValueError(f"Unexpected type '{type(values).__name__}'. 'values' must be a DatetimeArray ndarray, or Series or Index containing one of those.")
        if (values.ndim not in [1, 2]):
            raise ValueError('Only 1-dimensional input arrays are supported.')
        if (values.dtype == 'i8'):
            values = values.view(DT64NS_DTYPE)
        if (values.dtype != DT64NS_DTYPE):
            raise ValueError(f"The dtype of 'values' is incorrect. Must be 'datetime64[ns]'. Got {values.dtype} instead.")
        dtype = _validate_dt64_dtype(dtype)
        if (freq == 'infer'):
            raise ValueError("Frequency inference not allowed in DatetimeArray.__init__. Use 'pd.array()' instead.")
        if copy:
            values = values.copy()
        if freq:
            freq = to_offset(freq)
        if getattr(dtype, 'tz', None):
            dtype = DatetimeTZDtype(tz=timezones.tz_standardize(dtype.tz))
        self._data = values
        self._dtype = dtype
        self._freq = freq
        if ((inferred_freq is None) and (freq is not None)):
            type(self)._validate_frequency(self, freq)

    @classmethod
    def _simple_new(cls, values, freq=None, dtype=DT64NS_DTYPE):
        assert isinstance(values, np.ndarray)
        if (values.dtype != DT64NS_DTYPE):
            assert (values.dtype == 'i8')
            values = values.view(DT64NS_DTYPE)
        result = object.__new__(cls)
        result._data = values
        result._freq = freq
        result._dtype = dtype
        return result

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_not_strict(cls, data, dtype=None, copy=False, tz=None, freq=lib.no_default, dayfirst=False, yearfirst=False, ambiguous='raise'):
        explicit_none = (freq is None)
        freq = (freq if (freq is not lib.no_default) else None)
        (freq, freq_infer) = dtl.maybe_infer_freq(freq)
        (subarr, tz, inferred_freq) = sequence_to_dt64ns(data, dtype=dtype, copy=copy, tz=tz, dayfirst=dayfirst, yearfirst=yearfirst, ambiguous=ambiguous)
        (freq, freq_infer) = dtl.validate_inferred_freq(freq, inferred_freq, freq_infer)
        if explicit_none:
            freq = None
        dtype = tz_to_dtype(tz)
        result = cls._simple_new(subarr, freq=freq, dtype=dtype)
        if ((inferred_freq is None) and (freq is not None)):
            cls._validate_frequency(result, freq, ambiguous=ambiguous)
        elif freq_infer:
            result._freq = to_offset(result.inferred_freq)
        return result

    @classmethod
    def _generate_range(cls, start, end, periods, freq, tz=None, normalize=False, ambiguous='raise', nonexistent='raise', closed=None):
        periods = dtl.validate_periods(periods)
        if ((freq is None) and any(((x is None) for x in [periods, start, end]))):
            raise ValueError('Must provide freq argument if no data is supplied')
        if (com.count_not_none(start, end, periods, freq) != 3):
            raise ValueError('Of the four parameters: start, end, periods, and freq, exactly three must be specified')
        freq = to_offset(freq)
        if (start is not None):
            start = Timestamp(start)
        if (end is not None):
            end = Timestamp(end)
        if ((start is NaT) or (end is NaT)):
            raise ValueError('Neither `start` nor `end` can be NaT')
        (left_closed, right_closed) = dtl.validate_endpoints(closed)
        (start, end, _normalized) = _maybe_normalize_endpoints(start, end, normalize)
        tz = _infer_tz_from_endpoints(start, end, tz)
        if (tz is not None):
            start_tz = (None if (start is None) else start.tz)
            end_tz = (None if (end is None) else end.tz)
            start = _maybe_localize_point(start, start_tz, start, freq, tz, ambiguous, nonexistent)
            end = _maybe_localize_point(end, end_tz, end, freq, tz, ambiguous, nonexistent)
        if (freq is not None):
            if isinstance(freq, Day):
                if (start is not None):
                    start = start.tz_localize(None)
                if (end is not None):
                    end = end.tz_localize(None)
            if isinstance(freq, Tick):
                values = generate_regular_range(start, end, periods, freq)
            else:
                xdr = generate_range(start=start, end=end, periods=periods, offset=freq)
                values = np.array([x.value for x in xdr], dtype=np.int64)
            _tz = (start.tz if (start is not None) else end.tz)
            index = cls._simple_new(values, freq=freq, dtype=tz_to_dtype(_tz))
            if ((tz is not None) and (index.tz is None)):
                arr = tzconversion.tz_localize_to_utc(index.asi8, tz, ambiguous=ambiguous, nonexistent=nonexistent)
                index = cls(arr)
                if (start is not None):
                    start = start.tz_localize(tz, ambiguous, nonexistent).asm8
                if (end is not None):
                    end = end.tz_localize(tz, ambiguous, nonexistent).asm8
        else:
            arr = (np.linspace(0, (end.value - start.value), periods, dtype='int64') + start.value)
            dtype = tz_to_dtype(tz)
            index = cls._simple_new(arr.astype('M8[ns]', copy=False), freq=None, dtype=dtype)
        if ((not left_closed) and len(index) and (index[0] == start)):
            index = cast(DatetimeArray, index[1:])
        if ((not right_closed) and len(index) and (index[(- 1)] == end)):
            index = cast(DatetimeArray, index[:(- 1)])
        dtype = tz_to_dtype(tz)
        return cls._simple_new(index.asi8, freq=freq, dtype=dtype)

    def _unbox_scalar(self, value, setitem=False):
        if ((not isinstance(value, self._scalar_type)) and (value is not NaT)):
            raise ValueError("'value' should be a Timestamp.")
        if (not isna(value)):
            self._check_compatible_with(value, setitem=setitem)
            return value.asm8
        return np.datetime64(value.value, 'ns')

    def _scalar_from_string(self, value):
        return Timestamp(value, tz=self.tz)

    def _check_compatible_with(self, other, setitem=False):
        if (other is NaT):
            return
        self._assert_tzawareness_compat(other)
        if setitem:
            if (not timezones.tz_compare(self.tz, other.tz)):
                raise ValueError(f"Timezones don't match. '{self.tz}' != '{other.tz}'")

    def _box_func(self, x):
        return Timestamp(x, freq=self.freq, tz=self.tz)

    @property
    def dtype(self):
        "\n        The dtype for the DatetimeArray.\n\n        .. warning::\n\n           A future version of pandas will change dtype to never be a\n           ``numpy.dtype``. Instead, :attr:`DatetimeArray.dtype` will\n           always be an instance of an ``ExtensionDtype`` subclass.\n\n        Returns\n        -------\n        numpy.dtype or DatetimeTZDtype\n            If the values are tz-naive, then ``np.dtype('datetime64[ns]')``\n            is returned.\n\n            If the values are tz-aware, then the ``DatetimeTZDtype``\n            is returned.\n        "
        return self._dtype

    @property
    def tz(self):
        '\n        Return timezone, if any.\n\n        Returns\n        -------\n        datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, or None\n            Returns None when the array is tz-naive.\n        '
        return getattr(self.dtype, 'tz', None)

    @tz.setter
    def tz(self, value):
        raise AttributeError('Cannot directly set timezone. Use tz_localize() or tz_convert() as appropriate')

    @property
    def tzinfo(self):
        '\n        Alias for tz attribute\n        '
        return self.tz

    @property
    def is_normalized(self):
        '\n        Returns True if all of the dates are at midnight ("no time")\n        '
        return is_date_array_normalized(self.asi8, self.tz)

    @property
    def _resolution_obj(self):
        return get_resolution(self.asi8, self.tz)

    def __array__(self, dtype=None):
        if ((dtype is None) and self.tz):
            dtype = object
        return super().__array__(dtype=dtype)

    def __iter__(self):
        '\n        Return an iterator over the boxed values\n\n        Yields\n        ------\n        tstamp : Timestamp\n        '
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
                converted = ints_to_pydatetime(data[start_i:end_i], tz=self.tz, freq=self.freq, box='timestamp')
                (yield from converted)

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, self.dtype):
            if copy:
                return self.copy()
            return self
        elif is_datetime64_ns_dtype(dtype):
            return astype_dt64_to_dt64tz(self, dtype, copy, via_utc=False)
        elif is_period_dtype(dtype):
            return self.to_period(freq=dtype.freq)
        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)

    @dtl.ravel_compat
    def _format_native_types(self, na_rep='NaT', date_format=None, **kwargs):
        from pandas.io.formats.format import get_format_datetime64_from_values
        fmt = get_format_datetime64_from_values(self, date_format)
        return tslib.format_array_from_datetime(self.asi8, tz=self.tz, format=fmt, na_rep=na_rep)

    def _has_same_tz(self, other):
        if isinstance(other, np.datetime64):
            other = Timestamp(other)
        if (not hasattr(other, 'tzinfo')):
            return False
        other_tz = other.tzinfo
        return timezones.tz_compare(self.tzinfo, other_tz)

    def _assert_tzawareness_compat(self, other):
        other_tz = getattr(other, 'tzinfo', None)
        other_dtype = getattr(other, 'dtype', None)
        if is_datetime64tz_dtype(other_dtype):
            other_tz = other.dtype.tz
        if (other is NaT):
            pass
        elif (self.tz is None):
            if (other_tz is not None):
                raise TypeError('Cannot compare tz-naive and tz-aware datetime-like objects.')
        elif (other_tz is None):
            raise TypeError('Cannot compare tz-naive and tz-aware datetime-like objects')

    def _sub_datetime_arraylike(self, other):
        'subtract DatetimeArray/Index or ndarray[datetime64]'
        if (len(self) != len(other)):
            raise ValueError('cannot add indices of unequal length')
        if isinstance(other, np.ndarray):
            assert is_datetime64_dtype(other)
            other = type(self)(other)
        if (not self._has_same_tz(other)):
            raise TypeError(f'{type(self).__name__} subtraction must have the same timezones or no timezones')
        self_i8 = self.asi8
        other_i8 = other.asi8
        arr_mask = (self._isnan | other._isnan)
        new_values = checked_add_with_arr(self_i8, (- other_i8), arr_mask=arr_mask)
        if (self._hasnans or other._hasnans):
            np.putmask(new_values, arr_mask, iNaT)
        return new_values.view('timedelta64[ns]')

    def _add_offset(self, offset):
        if (self.ndim == 2):
            return self.ravel()._add_offset(offset).reshape(self.shape)
        assert (not isinstance(offset, Tick))
        try:
            if (self.tz is not None):
                values = self.tz_localize(None)
            else:
                values = self
            result = offset._apply_array(values)
            result = DatetimeArray._simple_new(result)
            result = result.tz_localize(self.tz)
        except NotImplementedError:
            warnings.warn('Non-vectorized DateOffset being applied to Series or DatetimeIndex', PerformanceWarning)
            result = (self.astype('O') + offset)
            if (not len(self)):
                return type(self)._from_sequence(result).tz_localize(self.tz)
        return type(self)._from_sequence(result)

    def _sub_datetimelike_scalar(self, other):
        assert isinstance(other, (datetime, np.datetime64))
        assert (other is not NaT)
        other = Timestamp(other)
        if (other is NaT):
            return (self - NaT)
        if (not self._has_same_tz(other)):
            raise TypeError('Timestamp subtraction must have the same timezones or no timezones')
        i8 = self.asi8
        result = checked_add_with_arr(i8, (- other.value), arr_mask=self._isnan)
        result = self._maybe_mask_results(result)
        return result.view('timedelta64[ns]')

    def _local_timestamps(self):
        '\n        Convert to an i8 (unix-like nanosecond timestamp) representation\n        while keeping the local timezone and not using UTC.\n        This is used to calculate time-of-day information as if the timestamps\n        were timezone-naive.\n        '
        if ((self.tz is None) or timezones.is_utc(self.tz)):
            return self.asi8
        return tzconversion.tz_convert_from_utc(self.asi8, self.tz)

    def tz_convert(self, tz):
        "\n        Convert tz-aware Datetime Array/Index from one time zone to another.\n\n        Parameters\n        ----------\n        tz : str, pytz.timezone, dateutil.tz.tzfile or None\n            Time zone for time. Corresponding timestamps would be converted\n            to this time zone of the Datetime Array/Index. A `tz` of None will\n            convert to UTC and remove the timezone information.\n\n        Returns\n        -------\n        Array or Index\n\n        Raises\n        ------\n        TypeError\n            If Datetime Array/Index is tz-naive.\n\n        See Also\n        --------\n        DatetimeIndex.tz : A timezone that has a variable offset from UTC.\n        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a\n            given time zone, or remove timezone from a tz-aware DatetimeIndex.\n\n        Examples\n        --------\n        With the `tz` parameter, we can change the DatetimeIndex\n        to other time zones:\n\n        >>> dti = pd.date_range(start='2014-08-01 09:00',\n        ...                     freq='H', periods=3, tz='Europe/Berlin')\n\n        >>> dti\n        DatetimeIndex(['2014-08-01 09:00:00+02:00',\n                       '2014-08-01 10:00:00+02:00',\n                       '2014-08-01 11:00:00+02:00'],\n                      dtype='datetime64[ns, Europe/Berlin]', freq='H')\n\n        >>> dti.tz_convert('US/Central')\n        DatetimeIndex(['2014-08-01 02:00:00-05:00',\n                       '2014-08-01 03:00:00-05:00',\n                       '2014-08-01 04:00:00-05:00'],\n                      dtype='datetime64[ns, US/Central]', freq='H')\n\n        With the ``tz=None``, we can remove the timezone (after converting\n        to UTC if necessary):\n\n        >>> dti = pd.date_range(start='2014-08-01 09:00', freq='H',\n        ...                     periods=3, tz='Europe/Berlin')\n\n        >>> dti\n        DatetimeIndex(['2014-08-01 09:00:00+02:00',\n                       '2014-08-01 10:00:00+02:00',\n                       '2014-08-01 11:00:00+02:00'],\n                        dtype='datetime64[ns, Europe/Berlin]', freq='H')\n\n        >>> dti.tz_convert(None)\n        DatetimeIndex(['2014-08-01 07:00:00',\n                       '2014-08-01 08:00:00',\n                       '2014-08-01 09:00:00'],\n                        dtype='datetime64[ns]', freq='H')\n        "
        tz = timezones.maybe_get_tz(tz)
        if (self.tz is None):
            raise TypeError('Cannot convert tz-naive timestamps, use tz_localize to localize')
        dtype = tz_to_dtype(tz)
        return self._simple_new(self.asi8, dtype=dtype, freq=self.freq)

    @dtl.ravel_compat
    def tz_localize(self, tz, ambiguous='raise', nonexistent='raise'):
        "\n        Localize tz-naive Datetime Array/Index to tz-aware\n        Datetime Array/Index.\n\n        This method takes a time zone (tz) naive Datetime Array/Index object\n        and makes this time zone aware. It does not move the time to another\n        time zone.\n        Time zone localization helps to switch from time zone aware to time\n        zone unaware objects.\n\n        Parameters\n        ----------\n        tz : str, pytz.timezone, dateutil.tz.tzfile or None\n            Time zone to convert timestamps to. Passing ``None`` will\n            remove the time zone information preserving local time.\n        ambiguous : 'infer', 'NaT', bool array, default 'raise'\n            When clocks moved backward due to DST, ambiguous times may arise.\n            For example in Central European Time (UTC+01), when going from\n            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at\n            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the\n            `ambiguous` parameter dictates how ambiguous times should be\n            handled.\n\n            - 'infer' will attempt to infer fall dst-transition hours based on\n              order\n            - bool-ndarray where True signifies a DST time, False signifies a\n              non-DST time (note that this flag is only applicable for\n              ambiguous times)\n            - 'NaT' will return NaT where there are ambiguous times\n            - 'raise' will raise an AmbiguousTimeError if there are ambiguous\n              times.\n\n        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, default 'raise'\n            A nonexistent time does not exist in a particular timezone\n            where clocks moved forward due to DST.\n\n            - 'shift_forward' will shift the nonexistent time forward to the\n              closest existing time\n            - 'shift_backward' will shift the nonexistent time backward to the\n              closest existing time\n            - 'NaT' will return NaT where there are nonexistent times\n            - timedelta objects will shift nonexistent times by the timedelta\n            - 'raise' will raise an NonExistentTimeError if there are\n              nonexistent times.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        Same type as self\n            Array/Index converted to the specified time zone.\n\n        Raises\n        ------\n        TypeError\n            If the Datetime Array/Index is tz-aware and tz is not None.\n\n        See Also\n        --------\n        DatetimeIndex.tz_convert : Convert tz-aware DatetimeIndex from\n            one time zone to another.\n\n        Examples\n        --------\n        >>> tz_naive = pd.date_range('2018-03-01 09:00', periods=3)\n        >>> tz_naive\n        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',\n                       '2018-03-03 09:00:00'],\n                      dtype='datetime64[ns]', freq='D')\n\n        Localize DatetimeIndex in US/Eastern time zone:\n\n        >>> tz_aware = tz_naive.tz_localize(tz='US/Eastern')\n        >>> tz_aware\n        DatetimeIndex(['2018-03-01 09:00:00-05:00',\n                       '2018-03-02 09:00:00-05:00',\n                       '2018-03-03 09:00:00-05:00'],\n                      dtype='datetime64[ns, US/Eastern]', freq=None)\n\n        With the ``tz=None``, we can remove the time zone information\n        while keeping the local time (not converted to UTC):\n\n        >>> tz_aware.tz_localize(None)\n        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',\n                       '2018-03-03 09:00:00'],\n                      dtype='datetime64[ns]', freq=None)\n\n        Be careful with DST changes. When there is sequential data, pandas can\n        infer the DST time:\n\n        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:30:00',\n        ...                               '2018-10-28 02:00:00',\n        ...                               '2018-10-28 02:30:00',\n        ...                               '2018-10-28 02:00:00',\n        ...                               '2018-10-28 02:30:00',\n        ...                               '2018-10-28 03:00:00',\n        ...                               '2018-10-28 03:30:00']))\n        >>> s.dt.tz_localize('CET', ambiguous='infer')\n        0   2018-10-28 01:30:00+02:00\n        1   2018-10-28 02:00:00+02:00\n        2   2018-10-28 02:30:00+02:00\n        3   2018-10-28 02:00:00+01:00\n        4   2018-10-28 02:30:00+01:00\n        5   2018-10-28 03:00:00+01:00\n        6   2018-10-28 03:30:00+01:00\n        dtype: datetime64[ns, CET]\n\n        In some cases, inferring the DST is impossible. In such cases, you can\n        pass an ndarray to the ambiguous parameter to set the DST explicitly\n\n        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:20:00',\n        ...                               '2018-10-28 02:36:00',\n        ...                               '2018-10-28 03:46:00']))\n        >>> s.dt.tz_localize('CET', ambiguous=np.array([True, True, False]))\n        0   2018-10-28 01:20:00+02:00\n        1   2018-10-28 02:36:00+02:00\n        2   2018-10-28 03:46:00+01:00\n        dtype: datetime64[ns, CET]\n\n        If the DST transition causes nonexistent times, you can shift these\n        dates forward or backwards with a timedelta object or `'shift_forward'`\n        or `'shift_backwards'`.\n\n        >>> s = pd.to_datetime(pd.Series(['2015-03-29 02:30:00',\n        ...                               '2015-03-29 03:30:00']))\n        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_forward')\n        0   2015-03-29 03:00:00+02:00\n        1   2015-03-29 03:30:00+02:00\n        dtype: datetime64[ns, Europe/Warsaw]\n\n        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_backward')\n        0   2015-03-29 01:59:59.999999999+01:00\n        1   2015-03-29 03:30:00+02:00\n        dtype: datetime64[ns, Europe/Warsaw]\n\n        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent=pd.Timedelta('1H'))\n        0   2015-03-29 03:30:00+02:00\n        1   2015-03-29 03:30:00+02:00\n        dtype: datetime64[ns, Europe/Warsaw]\n        "
        nonexistent_options = ('raise', 'NaT', 'shift_forward', 'shift_backward')
        if ((nonexistent not in nonexistent_options) and (not isinstance(nonexistent, timedelta))):
            raise ValueError("The nonexistent argument must be one of 'raise', 'NaT', 'shift_forward', 'shift_backward' or a timedelta object")
        if (self.tz is not None):
            if (tz is None):
                new_dates = tzconversion.tz_convert_from_utc(self.asi8, self.tz)
            else:
                raise TypeError('Already tz-aware, use tz_convert to convert.')
        else:
            tz = timezones.maybe_get_tz(tz)
            new_dates = tzconversion.tz_localize_to_utc(self.asi8, tz, ambiguous=ambiguous, nonexistent=nonexistent)
        new_dates = new_dates.view(DT64NS_DTYPE)
        dtype = tz_to_dtype(tz)
        freq = None
        if (timezones.is_utc(tz) or ((len(self) == 1) and (not isna(new_dates[0])))):
            freq = self.freq
        elif ((tz is None) and (self.tz is None)):
            freq = self.freq
        return self._simple_new(new_dates, dtype=dtype, freq=freq)

    def to_pydatetime(self):
        '\n        Return Datetime Array/Index as object ndarray of datetime.datetime\n        objects.\n\n        Returns\n        -------\n        datetimes : ndarray\n        '
        return ints_to_pydatetime(self.asi8, tz=self.tz)

    def normalize(self):
        "\n        Convert times to midnight.\n\n        The time component of the date-time is converted to midnight i.e.\n        00:00:00. This is useful in cases, when the time does not matter.\n        Length is unaltered. The timezones are unaffected.\n\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on Datetime Array/Index.\n\n        Returns\n        -------\n        DatetimeArray, DatetimeIndex or Series\n            The same type as the original data. Series will have the same\n            name and index. DatetimeIndex will have the same name.\n\n        See Also\n        --------\n        floor : Floor the datetimes to the specified freq.\n        ceil : Ceil the datetimes to the specified freq.\n        round : Round the datetimes to the specified freq.\n\n        Examples\n        --------\n        >>> idx = pd.date_range(start='2014-08-01 10:00', freq='H',\n        ...                     periods=3, tz='Asia/Calcutta')\n        >>> idx\n        DatetimeIndex(['2014-08-01 10:00:00+05:30',\n                       '2014-08-01 11:00:00+05:30',\n                       '2014-08-01 12:00:00+05:30'],\n                        dtype='datetime64[ns, Asia/Calcutta]', freq='H')\n        >>> idx.normalize()\n        DatetimeIndex(['2014-08-01 00:00:00+05:30',\n                       '2014-08-01 00:00:00+05:30',\n                       '2014-08-01 00:00:00+05:30'],\n                       dtype='datetime64[ns, Asia/Calcutta]', freq=None)\n        "
        new_values = normalize_i8_timestamps(self.asi8, self.tz)
        return type(self)(new_values)._with_freq('infer').tz_localize(self.tz)

    @dtl.ravel_compat
    def to_period(self, freq=None):
        '\n        Cast to PeriodArray/Index at a particular frequency.\n\n        Converts DatetimeArray/Index to PeriodArray/Index.\n\n        Parameters\n        ----------\n        freq : str or Offset, optional\n            One of pandas\' :ref:`offset strings <timeseries.offset_aliases>`\n            or an Offset object. Will be inferred by default.\n\n        Returns\n        -------\n        PeriodArray/Index\n\n        Raises\n        ------\n        ValueError\n            When converting a DatetimeArray/Index with non-regular values,\n            so that a frequency cannot be inferred.\n\n        See Also\n        --------\n        PeriodIndex: Immutable ndarray holding ordinal values.\n        DatetimeIndex.to_pydatetime: Return DatetimeIndex as object.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"y": [1, 2, 3]},\n        ...                   index=pd.to_datetime(["2000-03-31 00:00:00",\n        ...                                         "2000-05-31 00:00:00",\n        ...                                         "2000-08-31 00:00:00"]))\n        >>> df.index.to_period("M")\n        PeriodIndex([\'2000-03\', \'2000-05\', \'2000-08\'],\n                    dtype=\'period[M]\', freq=\'M\')\n\n        Infer the daily frequency\n\n        >>> idx = pd.date_range("2017-01-01", periods=2)\n        >>> idx.to_period()\n        PeriodIndex([\'2017-01-01\', \'2017-01-02\'],\n                    dtype=\'period[D]\', freq=\'D\')\n        '
        from pandas.core.arrays import PeriodArray
        if (self.tz is not None):
            warnings.warn('Converting to PeriodArray/Index representation will drop timezone information.', UserWarning)
        if (freq is None):
            freq = (self.freqstr or self.inferred_freq)
            if (freq is None):
                raise ValueError('You must pass a freq argument as current index has none.')
            res = get_period_alias(freq)
            if (res is None):
                res = freq
            freq = res
        return PeriodArray._from_datetime64(self._data, freq, tz=self.tz)

    def to_perioddelta(self, freq):
        '\n        Calculate TimedeltaArray of difference between index\n        values and index converted to PeriodArray at specified\n        freq. Used for vectorized offsets.\n\n        Parameters\n        ----------\n        freq : Period frequency\n\n        Returns\n        -------\n        TimedeltaArray/Index\n        '
        warnings.warn('to_perioddelta is deprecated and will be removed in a future version.  Use `dtindex - dtindex.to_period(freq).to_timestamp()` instead', FutureWarning, stacklevel=3)
        from pandas.core.arrays.timedeltas import TimedeltaArray
        i8delta = (self.asi8 - self.to_period(freq).to_timestamp().asi8)
        m8delta = i8delta.view('m8[ns]')
        return TimedeltaArray(m8delta)

    def month_name(self, locale=None):
        "\n        Return the month names of the DateTimeIndex with specified locale.\n\n        Parameters\n        ----------\n        locale : str, optional\n            Locale determining the language in which to return the month name.\n            Default is English locale.\n\n        Returns\n        -------\n        Index\n            Index of month names.\n\n        Examples\n        --------\n        >>> idx = pd.date_range(start='2018-01', freq='M', periods=3)\n        >>> idx\n        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],\n                      dtype='datetime64[ns]', freq='M')\n        >>> idx.month_name()\n        Index(['January', 'February', 'March'], dtype='object')\n        "
        values = self._local_timestamps()
        result = fields.get_date_name_field(values, 'month_name', locale=locale)
        result = self._maybe_mask_results(result, fill_value=None)
        return result

    def day_name(self, locale=None):
        "\n        Return the day names of the DateTimeIndex with specified locale.\n\n        Parameters\n        ----------\n        locale : str, optional\n            Locale determining the language in which to return the day name.\n            Default is English locale.\n\n        Returns\n        -------\n        Index\n            Index of day names.\n\n        Examples\n        --------\n        >>> idx = pd.date_range(start='2018-01-01', freq='D', periods=3)\n        >>> idx\n        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],\n                      dtype='datetime64[ns]', freq='D')\n        >>> idx.day_name()\n        Index(['Monday', 'Tuesday', 'Wednesday'], dtype='object')\n        "
        values = self._local_timestamps()
        result = fields.get_date_name_field(values, 'day_name', locale=locale)
        result = self._maybe_mask_results(result, fill_value=None)
        return result

    @property
    def time(self):
        '\n        Returns numpy array of datetime.time. The time part of the Timestamps.\n        '
        timestamps = self._local_timestamps()
        return ints_to_pydatetime(timestamps, box='time')

    @property
    def timetz(self):
        '\n        Returns numpy array of datetime.time also containing timezone\n        information. The time part of the Timestamps.\n        '
        return ints_to_pydatetime(self.asi8, self.tz, box='time')

    @property
    def date(self):
        '\n        Returns numpy array of python datetime.date objects (namely, the date\n        part of Timestamps without timezone information).\n        '
        timestamps = self._local_timestamps()
        return ints_to_pydatetime(timestamps, box='date')

    def isocalendar(self):
        "\n        Returns a DataFrame with the year, week, and day calculated according to\n        the ISO 8601 standard.\n\n        .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        DataFrame\n            with columns year, week and day\n\n        See Also\n        --------\n        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,\n            week number, and weekday for the given Timestamp object.\n        datetime.date.isocalendar : Return a named tuple object with\n            three components: year, week and weekday.\n\n        Examples\n        --------\n        >>> idx = pd.date_range(start='2019-12-29', freq='D', periods=4)\n        >>> idx.isocalendar()\n                    year  week  day\n        2019-12-29  2019    52    7\n        2019-12-30  2020     1    1\n        2019-12-31  2020     1    2\n        2020-01-01  2020     1    3\n        >>> idx.isocalendar().week\n        2019-12-29    52\n        2019-12-30     1\n        2019-12-31     1\n        2020-01-01     1\n        Freq: D, Name: week, dtype: UInt32\n        "
        from pandas import DataFrame
        values = self._local_timestamps()
        sarray = fields.build_isocalendar_sarray(values)
        iso_calendar_df = DataFrame(sarray, columns=['year', 'week', 'day'], dtype='UInt32')
        if self._hasnans:
            iso_calendar_df.iloc[self._isnan] = None
        return iso_calendar_df

    @property
    def weekofyear(self):
        '\n        The week ordinal of the year.\n\n        .. deprecated:: 1.1.0\n\n        weekofyear and week have been deprecated.\n        Please use DatetimeIndex.isocalendar().week instead.\n        '
        warnings.warn('weekofyear and week have been deprecated, please use DatetimeIndex.isocalendar().week instead, which returns a Series.  To exactly reproduce the behavior of week and weekofyear and return an Index, you may call pd.Int64Index(idx.isocalendar().week)', FutureWarning, stacklevel=3)
        week_series = self.isocalendar().week
        if week_series.hasnans:
            return week_series.to_numpy(dtype='float64', na_value=np.nan)
        return week_series.to_numpy(dtype='int64')
    week = weekofyear
    year = _field_accessor('year', 'Y', '\n        The year of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="Y")\n        ... )\n        >>> datetime_series\n        0   2000-12-31\n        1   2001-12-31\n        2   2002-12-31\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.year\n        0    2000\n        1    2001\n        2    2002\n        dtype: int64\n        ')
    month = _field_accessor('month', 'M', '\n        The month as January=1, December=12.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="M")\n        ... )\n        >>> datetime_series\n        0   2000-01-31\n        1   2000-02-29\n        2   2000-03-31\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.month\n        0    1\n        1    2\n        2    3\n        dtype: int64\n        ')
    day = _field_accessor('day', 'D', '\n        The day of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="D")\n        ... )\n        >>> datetime_series\n        0   2000-01-01\n        1   2000-01-02\n        2   2000-01-03\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.day\n        0    1\n        1    2\n        2    3\n        dtype: int64\n        ')
    hour = _field_accessor('hour', 'h', '\n        The hours of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="h")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00\n        1   2000-01-01 01:00:00\n        2   2000-01-01 02:00:00\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.hour\n        0    0\n        1    1\n        2    2\n        dtype: int64\n        ')
    minute = _field_accessor('minute', 'm', '\n        The minutes of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="T")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00\n        1   2000-01-01 00:01:00\n        2   2000-01-01 00:02:00\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.minute\n        0    0\n        1    1\n        2    2\n        dtype: int64\n        ')
    second = _field_accessor('second', 's', '\n        The seconds of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="s")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00\n        1   2000-01-01 00:00:01\n        2   2000-01-01 00:00:02\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.second\n        0    0\n        1    1\n        2    2\n        dtype: int64\n        ')
    microsecond = _field_accessor('microsecond', 'us', '\n        The microseconds of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="us")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00.000000\n        1   2000-01-01 00:00:00.000001\n        2   2000-01-01 00:00:00.000002\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.microsecond\n        0       0\n        1       1\n        2       2\n        dtype: int64\n        ')
    nanosecond = _field_accessor('nanosecond', 'ns', '\n        The nanoseconds of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="ns")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00.000000000\n        1   2000-01-01 00:00:00.000000001\n        2   2000-01-01 00:00:00.000000002\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.nanosecond\n        0       0\n        1       1\n        2       2\n        dtype: int64\n        ')
    _dayofweek_doc = "\n    The day of the week with Monday=0, Sunday=6.\n\n    Return the day of the week. It is assumed the week starts on\n    Monday, which is denoted by 0 and ends on Sunday which is denoted\n    by 6. This method is available on both Series with datetime\n    values (using the `dt` accessor) or DatetimeIndex.\n\n    Returns\n    -------\n    Series or Index\n        Containing integers indicating the day number.\n\n    See Also\n    --------\n    Series.dt.dayofweek : Alias.\n    Series.dt.weekday : Alias.\n    Series.dt.day_name : Returns the name of the day of the week.\n\n    Examples\n    --------\n    >>> s = pd.date_range('2016-12-31', '2017-01-08', freq='D').to_series()\n    >>> s.dt.dayofweek\n    2016-12-31    5\n    2017-01-01    6\n    2017-01-02    0\n    2017-01-03    1\n    2017-01-04    2\n    2017-01-05    3\n    2017-01-06    4\n    2017-01-07    5\n    2017-01-08    6\n    Freq: D, dtype: int64\n    "
    day_of_week = _field_accessor('day_of_week', 'dow', _dayofweek_doc)
    dayofweek = day_of_week
    weekday = day_of_week
    day_of_year = _field_accessor('dayofyear', 'doy', '\n        The ordinal day of the year.\n        ')
    dayofyear = day_of_year
    quarter = _field_accessor('quarter', 'q', '\n        The quarter of the date.\n        ')
    days_in_month = _field_accessor('days_in_month', 'dim', '\n        The number of days in the month.\n        ')
    daysinmonth = days_in_month
    _is_month_doc = '\n        Indicates whether the date is the {first_or_last} day of the month.\n\n        Returns\n        -------\n        Series or array\n            For Series, returns a Series with boolean values.\n            For DatetimeIndex, returns a boolean array.\n\n        See Also\n        --------\n        is_month_start : Return a boolean indicating whether the date\n            is the first day of the month.\n        is_month_end : Return a boolean indicating whether the date\n            is the last day of the month.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> s = pd.Series(pd.date_range("2018-02-27", periods=3))\n        >>> s\n        0   2018-02-27\n        1   2018-02-28\n        2   2018-03-01\n        dtype: datetime64[ns]\n        >>> s.dt.is_month_start\n        0    False\n        1    False\n        2    True\n        dtype: bool\n        >>> s.dt.is_month_end\n        0    False\n        1    True\n        2    False\n        dtype: bool\n\n        >>> idx = pd.date_range("2018-02-27", periods=3)\n        >>> idx.is_month_start\n        array([False, False, True])\n        >>> idx.is_month_end\n        array([False, True, False])\n    '
    is_month_start = _field_accessor('is_month_start', 'is_month_start', _is_month_doc.format(first_or_last='first'))
    is_month_end = _field_accessor('is_month_end', 'is_month_end', _is_month_doc.format(first_or_last='last'))
    is_quarter_start = _field_accessor('is_quarter_start', 'is_quarter_start', '\n        Indicator for whether the date is the first day of a quarter.\n\n        Returns\n        -------\n        is_quarter_start : Series or DatetimeIndex\n            The same type as the original data with boolean values. Series will\n            have the same name and index. DatetimeIndex will have the same\n            name.\n\n        See Also\n        --------\n        quarter : Return the quarter of the date.\n        is_quarter_end : Similar property for indicating the quarter start.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> df = pd.DataFrame({\'dates\': pd.date_range("2017-03-30",\n        ...                   periods=4)})\n        >>> df.assign(quarter=df.dates.dt.quarter,\n        ...           is_quarter_start=df.dates.dt.is_quarter_start)\n               dates  quarter  is_quarter_start\n        0 2017-03-30        1             False\n        1 2017-03-31        1             False\n        2 2017-04-01        2              True\n        3 2017-04-02        2             False\n\n        >>> idx = pd.date_range(\'2017-03-30\', periods=4)\n        >>> idx\n        DatetimeIndex([\'2017-03-30\', \'2017-03-31\', \'2017-04-01\', \'2017-04-02\'],\n                      dtype=\'datetime64[ns]\', freq=\'D\')\n\n        >>> idx.is_quarter_start\n        array([False, False,  True, False])\n        ')
    is_quarter_end = _field_accessor('is_quarter_end', 'is_quarter_end', '\n        Indicator for whether the date is the last day of a quarter.\n\n        Returns\n        -------\n        is_quarter_end : Series or DatetimeIndex\n            The same type as the original data with boolean values. Series will\n            have the same name and index. DatetimeIndex will have the same\n            name.\n\n        See Also\n        --------\n        quarter : Return the quarter of the date.\n        is_quarter_start : Similar property indicating the quarter start.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> df = pd.DataFrame({\'dates\': pd.date_range("2017-03-30",\n        ...                    periods=4)})\n        >>> df.assign(quarter=df.dates.dt.quarter,\n        ...           is_quarter_end=df.dates.dt.is_quarter_end)\n               dates  quarter    is_quarter_end\n        0 2017-03-30        1             False\n        1 2017-03-31        1              True\n        2 2017-04-01        2             False\n        3 2017-04-02        2             False\n\n        >>> idx = pd.date_range(\'2017-03-30\', periods=4)\n        >>> idx\n        DatetimeIndex([\'2017-03-30\', \'2017-03-31\', \'2017-04-01\', \'2017-04-02\'],\n                      dtype=\'datetime64[ns]\', freq=\'D\')\n\n        >>> idx.is_quarter_end\n        array([False,  True, False, False])\n        ')
    is_year_start = _field_accessor('is_year_start', 'is_year_start', '\n        Indicate whether the date is the first day of a year.\n\n        Returns\n        -------\n        Series or DatetimeIndex\n            The same type as the original data with boolean values. Series will\n            have the same name and index. DatetimeIndex will have the same\n            name.\n\n        See Also\n        --------\n        is_year_end : Similar property indicating the last day of the year.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))\n        >>> dates\n        0   2017-12-30\n        1   2017-12-31\n        2   2018-01-01\n        dtype: datetime64[ns]\n\n        >>> dates.dt.is_year_start\n        0    False\n        1    False\n        2    True\n        dtype: bool\n\n        >>> idx = pd.date_range("2017-12-30", periods=3)\n        >>> idx\n        DatetimeIndex([\'2017-12-30\', \'2017-12-31\', \'2018-01-01\'],\n                      dtype=\'datetime64[ns]\', freq=\'D\')\n\n        >>> idx.is_year_start\n        array([False, False,  True])\n        ')
    is_year_end = _field_accessor('is_year_end', 'is_year_end', '\n        Indicate whether the date is the last day of the year.\n\n        Returns\n        -------\n        Series or DatetimeIndex\n            The same type as the original data with boolean values. Series will\n            have the same name and index. DatetimeIndex will have the same\n            name.\n\n        See Also\n        --------\n        is_year_start : Similar property indicating the start of the year.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))\n        >>> dates\n        0   2017-12-30\n        1   2017-12-31\n        2   2018-01-01\n        dtype: datetime64[ns]\n\n        >>> dates.dt.is_year_end\n        0    False\n        1     True\n        2    False\n        dtype: bool\n\n        >>> idx = pd.date_range("2017-12-30", periods=3)\n        >>> idx\n        DatetimeIndex([\'2017-12-30\', \'2017-12-31\', \'2018-01-01\'],\n                      dtype=\'datetime64[ns]\', freq=\'D\')\n\n        >>> idx.is_year_end\n        array([False,  True, False])\n        ')
    is_leap_year = _field_accessor('is_leap_year', 'is_leap_year', '\n        Boolean indicator if the date belongs to a leap year.\n\n        A leap year is a year, which has 366 days (instead of 365) including\n        29th of February as an intercalary day.\n        Leap years are years which are multiples of four with the exception\n        of years divisible by 100 but not by 400.\n\n        Returns\n        -------\n        Series or ndarray\n             Booleans indicating if dates belong to a leap year.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> idx = pd.date_range("2012-01-01", "2015-01-01", freq="Y")\n        >>> idx\n        DatetimeIndex([\'2012-12-31\', \'2013-12-31\', \'2014-12-31\'],\n                      dtype=\'datetime64[ns]\', freq=\'A-DEC\')\n        >>> idx.is_leap_year\n        array([ True, False, False])\n\n        >>> dates_series = pd.Series(idx)\n        >>> dates_series\n        0   2012-12-31\n        1   2013-12-31\n        2   2014-12-31\n        dtype: datetime64[ns]\n        >>> dates_series.dt.is_leap_year\n        0     True\n        1    False\n        2    False\n        dtype: bool\n        ')

    def to_julian_date(self):
        '\n        Convert Datetime Array to float64 ndarray of Julian Dates.\n        0 Julian date is noon January 1, 4713 BC.\n        https://en.wikipedia.org/wiki/Julian_day\n        '
        year = np.asarray(self.year)
        month = np.asarray(self.month)
        day = np.asarray(self.day)
        testarr = (month < 3)
        year[testarr] -= 1
        month[testarr] += 12
        return (((((((day + np.fix((((153 * month) - 457) / 5))) + (365 * year)) + np.floor((year / 4))) - np.floor((year / 100))) + np.floor((year / 400))) + 1721118.5) + (((((self.hour + (self.minute / 60.0)) + (self.second / 3600.0)) + ((self.microsecond / 3600.0) / 1000000.0)) + ((self.nanosecond / 3600.0) / 1000000000.0)) / 24.0))

    def std(self, axis=None, dtype=None, out=None, ddof=1, keepdims=False, skipna=True):
        from pandas.core.arrays import TimedeltaArray
        tda = TimedeltaArray(self._ndarray.view('i8'))
        return tda.std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, skipna=skipna)

def sequence_to_dt64ns(data, dtype=None, copy=False, tz=None, dayfirst=False, yearfirst=False, ambiguous='raise'):
    "\n    Parameters\n    ----------\n    data : list-like\n    dtype : dtype, str, or None, default None\n    copy : bool, default False\n    tz : tzinfo, str, or None, default None\n    dayfirst : bool, default False\n    yearfirst : bool, default False\n    ambiguous : str, bool, or arraylike, default 'raise'\n        See pandas._libs.tslibs.tzconversion.tz_localize_to_utc.\n\n    Returns\n    -------\n    result : numpy.ndarray\n        The sequence converted to a numpy array with dtype ``datetime64[ns]``.\n    tz : tzinfo or None\n        Either the user-provided tzinfo or one inferred from the data.\n    inferred_freq : Tick or None\n        The inferred frequency of the sequence.\n\n    Raises\n    ------\n    TypeError : PeriodDType data is passed\n    "
    inferred_freq = None
    dtype = _validate_dt64_dtype(dtype)
    tz = timezones.maybe_get_tz(tz)
    if (not hasattr(data, 'dtype')):
        if (np.ndim(data) == 0):
            data = list(data)
        data = np.asarray(data)
        copy = False
    elif isinstance(data, ABCSeries):
        data = data._values
    if isinstance(data, ABCPandasArray):
        data = data.to_numpy()
    if hasattr(data, 'freq'):
        inferred_freq = data.freq
    tz = validate_tz_from_dtype(dtype, tz)
    if isinstance(data, ABCIndex):
        if (data.nlevels > 1):
            raise TypeError('Cannot create a DatetimeArray from a MultiIndex.')
        data = data._data
    (data, copy) = maybe_convert_dtype(data, copy)
    data_dtype = getattr(data, 'dtype', None)
    if (is_object_dtype(data_dtype) or is_string_dtype(data_dtype) or is_sparse(data_dtype)):
        copy = False
        if (lib.infer_dtype(data, skipna=False) == 'integer'):
            data = data.astype(np.int64)
        else:
            (data, inferred_tz) = objects_to_datetime64ns(data, dayfirst=dayfirst, yearfirst=yearfirst)
            if (tz and inferred_tz):
                data = tzconversion.tz_convert_from_utc(data.view('i8'), tz)
                data = data.view(DT64NS_DTYPE)
            elif inferred_tz:
                tz = inferred_tz
        data_dtype = data.dtype
    if is_datetime64tz_dtype(data_dtype):
        tz = _maybe_infer_tz(tz, data.tz)
        result = data._data
    elif is_datetime64_dtype(data_dtype):
        data = getattr(data, '_data', data)
        if (data.dtype != DT64NS_DTYPE):
            data = conversion.ensure_datetime64ns(data)
        if (tz is not None):
            tz = timezones.maybe_get_tz(tz)
            data = tzconversion.tz_localize_to_utc(data.view('i8'), tz, ambiguous=ambiguous)
            data = data.view(DT64NS_DTYPE)
        assert (data.dtype == DT64NS_DTYPE), data.dtype
        result = data
    else:
        if tz:
            tz = timezones.maybe_get_tz(tz)
        if (data.dtype != INT64_DTYPE):
            data = data.astype(np.int64, copy=False)
        result = data.view(DT64NS_DTYPE)
    if copy:
        result = result.copy()
    assert isinstance(result, np.ndarray), type(result)
    assert (result.dtype == 'M8[ns]'), result.dtype
    validate_tz_from_dtype(dtype, tz)
    return (result, tz, inferred_freq)

def objects_to_datetime64ns(data, dayfirst, yearfirst, utc=False, errors='raise', require_iso8601=False, allow_object=False):
    "\n    Convert data to array of timestamps.\n\n    Parameters\n    ----------\n    data : np.ndarray[object]\n    dayfirst : bool\n    yearfirst : bool\n    utc : bool, default False\n        Whether to convert timezone-aware timestamps to UTC.\n    errors : {'raise', 'ignore', 'coerce'}\n    require_iso8601 : bool, default False\n    allow_object : bool\n        Whether to return an object-dtype ndarray instead of raising if the\n        data contains more than one timezone.\n\n    Returns\n    -------\n    result : ndarray\n        np.int64 dtype if returned values represent UTC timestamps\n        np.datetime64[ns] if returned values represent wall times\n        object if mixed timezones\n    inferred_tz : tzinfo or None\n\n    Raises\n    ------\n    ValueError : if data cannot be converted to datetimes\n    "
    assert (errors in ['raise', 'ignore', 'coerce'])
    data = np.array(data, copy=False, dtype=np.object_)
    flags = data.flags
    order = ('F' if flags.f_contiguous else 'C')
    try:
        (result, tz_parsed) = tslib.array_to_datetime(data.ravel('K'), errors=errors, utc=utc, dayfirst=dayfirst, yearfirst=yearfirst, require_iso8601=require_iso8601)
        result = result.reshape(data.shape, order=order)
    except ValueError as err:
        try:
            (values, tz_parsed) = conversion.datetime_to_datetime64(data.ravel('K'))
            values = values.reshape(data.shape, order=order)
            return (values.view('i8'), tz_parsed)
        except (ValueError, TypeError):
            raise err
    if (tz_parsed is not None):
        return (result.view('i8'), tz_parsed)
    elif is_datetime64_dtype(result):
        return (result, tz_parsed)
    elif is_object_dtype(result):
        if allow_object:
            return (result, tz_parsed)
        raise TypeError(result)
    else:
        raise TypeError(result)

def maybe_convert_dtype(data, copy):
    '\n    Convert data based on dtype conventions, issuing deprecation warnings\n    or errors where appropriate.\n\n    Parameters\n    ----------\n    data : np.ndarray or pd.Index\n    copy : bool\n\n    Returns\n    -------\n    data : np.ndarray or pd.Index\n    copy : bool\n\n    Raises\n    ------\n    TypeError : PeriodDType data is passed\n    '
    if (not hasattr(data, 'dtype')):
        return (data, copy)
    if is_float_dtype(data.dtype):
        data = data.astype(DT64NS_DTYPE)
        copy = False
    elif (is_timedelta64_dtype(data.dtype) or is_bool_dtype(data.dtype)):
        raise TypeError(f'dtype {data.dtype} cannot be converted to datetime64[ns]')
    elif is_period_dtype(data.dtype):
        raise TypeError('Passing PeriodDtype data is invalid. Use `data.to_timestamp()` instead')
    elif is_categorical_dtype(data.dtype):
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False
    elif (is_extension_array_dtype(data.dtype) and (not is_datetime64tz_dtype(data.dtype))):
        data = np.array(data, dtype=np.object_)
        copy = False
    return (data, copy)

def _maybe_infer_tz(tz, inferred_tz):
    '\n    If a timezone is inferred from data, check that it is compatible with\n    the user-provided timezone, if any.\n\n    Parameters\n    ----------\n    tz : tzinfo or None\n    inferred_tz : tzinfo or None\n\n    Returns\n    -------\n    tz : tzinfo or None\n\n    Raises\n    ------\n    TypeError : if both timezones are present but do not match\n    '
    if (tz is None):
        tz = inferred_tz
    elif (inferred_tz is None):
        pass
    elif (not timezones.tz_compare(tz, inferred_tz)):
        raise TypeError(f'data is already tz-aware {inferred_tz}, unable to set specified tz: {tz}')
    return tz

def _validate_dt64_dtype(dtype):
    '\n    Check that a dtype, if passed, represents either a numpy datetime64[ns]\n    dtype or a pandas DatetimeTZDtype.\n\n    Parameters\n    ----------\n    dtype : object\n\n    Returns\n    -------\n    dtype : None, numpy.dtype, or DatetimeTZDtype\n\n    Raises\n    ------\n    ValueError : invalid dtype\n\n    Notes\n    -----\n    Unlike validate_tz_from_dtype, this does _not_ allow non-existent\n    tz errors to go through\n    '
    if (dtype is not None):
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, np.dtype('M8')):
            msg = "Passing in 'datetime64' dtype with no precision is not allowed. Please pass in 'datetime64[ns]' instead."
            raise ValueError(msg)
        if ((isinstance(dtype, np.dtype) and (dtype != DT64NS_DTYPE)) or (not isinstance(dtype, (np.dtype, DatetimeTZDtype)))):
            raise ValueError(f"Unexpected value for 'dtype': '{dtype}'. Must be 'datetime64[ns]' or DatetimeTZDtype'.")
    return dtype

def validate_tz_from_dtype(dtype, tz):
    '\n    If the given dtype is a DatetimeTZDtype, extract the implied\n    tzinfo object from it and check that it does not conflict with the given\n    tz.\n\n    Parameters\n    ----------\n    dtype : dtype, str\n    tz : None, tzinfo\n\n    Returns\n    -------\n    tz : consensus tzinfo\n\n    Raises\n    ------\n    ValueError : on tzinfo mismatch\n    '
    if (dtype is not None):
        if isinstance(dtype, str):
            try:
                dtype = DatetimeTZDtype.construct_from_string(dtype)
            except TypeError:
                pass
        dtz = getattr(dtype, 'tz', None)
        if (dtz is not None):
            if ((tz is not None) and (not timezones.tz_compare(tz, dtz))):
                raise ValueError('cannot supply both a tz and a dtype with a tz')
            tz = dtz
        if ((tz is not None) and is_datetime64_dtype(dtype)):
            if ((tz is not None) and (not timezones.tz_compare(tz, dtz))):
                raise ValueError('cannot supply both a tz and a timezone-naive dtype (i.e. datetime64[ns])')
    return tz

def _infer_tz_from_endpoints(start, end, tz):
    '\n    If a timezone is not explicitly given via `tz`, see if one can\n    be inferred from the `start` and `end` endpoints.  If more than one\n    of these inputs provides a timezone, require that they all agree.\n\n    Parameters\n    ----------\n    start : Timestamp\n    end : Timestamp\n    tz : tzinfo or None\n\n    Returns\n    -------\n    tz : tzinfo or None\n\n    Raises\n    ------\n    TypeError : if start and end timezones do not agree\n    '
    try:
        inferred_tz = timezones.infer_tzinfo(start, end)
    except AssertionError as err:
        raise TypeError('Start and end cannot both be tz-aware with different timezones') from err
    inferred_tz = timezones.maybe_get_tz(inferred_tz)
    tz = timezones.maybe_get_tz(tz)
    if ((tz is not None) and (inferred_tz is not None)):
        if (not timezones.tz_compare(inferred_tz, tz)):
            raise AssertionError('Inferred time zone not equal to passed time zone')
    elif (inferred_tz is not None):
        tz = inferred_tz
    return tz

def _maybe_normalize_endpoints(start, end, normalize):
    _normalized = True
    if (start is not None):
        if normalize:
            start = start.normalize()
            _normalized = True
        else:
            _normalized = (_normalized and (start.time() == _midnight))
    if (end is not None):
        if normalize:
            end = end.normalize()
            _normalized = True
        else:
            _normalized = (_normalized and (end.time() == _midnight))
    return (start, end, _normalized)

def _maybe_localize_point(ts, is_none, is_not_none, freq, tz, ambiguous, nonexistent):
    '\n    Localize a start or end Timestamp to the timezone of the corresponding\n    start or end Timestamp\n\n    Parameters\n    ----------\n    ts : start or end Timestamp to potentially localize\n    is_none : argument that should be None\n    is_not_none : argument that should not be None\n    freq : Tick, DateOffset, or None\n    tz : str, timezone object or None\n    ambiguous: str, localization behavior for ambiguous times\n    nonexistent: str, localization behavior for nonexistent times\n\n    Returns\n    -------\n    ts : Timestamp\n    '
    if ((is_none is None) and (is_not_none is not None)):
        ambiguous = (ambiguous if (ambiguous != 'infer') else False)
        localize_args = {'ambiguous': ambiguous, 'nonexistent': nonexistent, 'tz': None}
        if (isinstance(freq, Tick) or (freq is None)):
            localize_args['tz'] = tz
        ts = ts.tz_localize(**localize_args)
    return ts

def generate_range(start=None, end=None, periods=None, offset=BDay()):
    '\n    Generates a sequence of dates corresponding to the specified time\n    offset. Similar to dateutil.rrule except uses pandas DateOffset\n    objects to represent time increments.\n\n    Parameters\n    ----------\n    start : datetime, (default None)\n    end : datetime, (default None)\n    periods : int, (default None)\n    offset : DateOffset, (default BDay())\n\n    Notes\n    -----\n    * This method is faster for generating weekdays than dateutil.rrule\n    * At least two of (start, end, periods) must be specified.\n    * If both start and end are specified, the returned dates will\n    satisfy start <= date <= end.\n\n    Returns\n    -------\n    dates : generator object\n    '
    offset = to_offset(offset)
    start = Timestamp(start)
    start = (start if (start is not NaT) else None)
    end = Timestamp(end)
    end = (end if (end is not NaT) else None)
    if (start and (not offset.is_on_offset(start))):
        start = offset.rollforward(start)
    elif (end and (not offset.is_on_offset(end))):
        end = offset.rollback(end)
    if ((periods is None) and (end < start) and (offset.n >= 0)):
        end = None
        periods = 0
    if (end is None):
        end = (start + ((periods - 1) * offset))
    if (start is None):
        start = (end - ((periods - 1) * offset))
    cur = start
    if (offset.n >= 0):
        while (cur <= end):
            (yield cur)
            if (cur == end):
                break
            next_date = offset.apply(cur)
            if (next_date <= cur):
                raise ValueError(f'Offset {offset} did not increment date')
            cur = next_date
    else:
        while (cur >= end):
            (yield cur)
            if (cur == end):
                break
            next_date = offset.apply(cur)
            if (next_date >= cur):
                raise ValueError(f'Offset {offset} did not decrement date')
            cur = next_date
