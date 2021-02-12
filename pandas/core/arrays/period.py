
from datetime import timedelta
import operator
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import numpy as np
from pandas._libs.tslibs import BaseOffset, NaT, NaTType, Timedelta, delta_to_nanoseconds, dt64arr_to_periodarr as c_dt64arr_to_periodarr, iNaT, period as libperiod, to_offset
from pandas._libs.tslibs.dtypes import FreqGroup
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import Tick, delta_to_tick
from pandas._libs.tslibs.period import DIFFERENT_FREQ, IncompatibleFrequency, Period, PeriodMixin, get_period_field_arr, period_asfreq_arr
from pandas._typing import AnyArrayLike, Dtype
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import TD64NS_DTYPE, ensure_object, is_datetime64_dtype, is_dtype_equal, is_float_dtype, is_period_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCIndex, ABCPeriodIndex, ABCSeries, ABCTimedeltaArray
from pandas.core.dtypes.missing import isna, notna
import pandas.core.algorithms as algos
from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com
_shared_doc_kwargs = {'klass': 'PeriodArray'}

def _field_accessor(name, docstring=None):

    def f(self):
        base = self.freq._period_dtype_code
        result = get_period_field_arr(name, self.asi8, base)
        return result
    f.__name__ = name
    f.__doc__ = docstring
    return property(f)

class PeriodArray(PeriodMixin, dtl.DatelikeOps):
    '\n    Pandas ExtensionArray for storing Period data.\n\n    Users should use :func:`~pandas.period_array` to create new instances.\n    Alternatively, :func:`~pandas.array` can be used to create new instances\n    from a sequence of Period scalars.\n\n    Parameters\n    ----------\n    values : Union[PeriodArray, Series[period], ndarray[int], PeriodIndex]\n        The data to store. These should be arrays that can be directly\n        converted to ordinals without inference or copy (PeriodArray,\n        ndarray[int64]), or a box around such an array (Series[period],\n        PeriodIndex).\n    dtype : PeriodDtype, optional\n        A PeriodDtype instance from which to extract a `freq`. If both\n        `freq` and `dtype` are specified, then the frequencies must match.\n    freq : str or DateOffset\n        The `freq` to use for the array. Mostly applicable when `values`\n        is an ndarray of integers, when `freq` is required. When `values`\n        is a PeriodArray (or box around), it\'s checked that ``values.freq``\n        matches `freq`.\n    copy : bool, default False\n        Whether to copy the ordinals before storing.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    See Also\n    --------\n    Period: Represents a period of time.\n    PeriodIndex : Immutable Index for period data.\n    period_range: Create a fixed-frequency PeriodArray.\n    array: Construct a pandas array.\n\n    Notes\n    -----\n    There are two components to a PeriodArray\n\n    - ordinals : integer ndarray\n    - freq : pd.tseries.offsets.Offset\n\n    The values are physically stored as a 1-D ndarray of integers. These are\n    called "ordinals" and represent some kind of offset from a base.\n\n    The `freq` indicates the span covered by each element of the array.\n    All elements in the PeriodArray have the same `freq`.\n    '
    __array_priority__ = 1000
    _typ = 'periodarray'
    _scalar_type = Period
    _recognized_scalars = (Period,)
    _is_recognized_dtype = is_period_dtype
    _infer_matches = ('period',)
    _other_ops = []
    _bool_ops = ['is_leap_year']
    _object_ops = ['start_time', 'end_time', 'freq']
    _field_ops = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekofyear', 'weekday', 'week', 'dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'qyear', 'days_in_month', 'daysinmonth']
    _datetimelike_ops = ((_field_ops + _object_ops) + _bool_ops)
    _datetimelike_methods = ['strftime', 'to_timestamp', 'asfreq']

    def __init__(self, values, dtype=None, freq=None, copy=False):
        freq = validate_dtype_freq(dtype, freq)
        if (freq is not None):
            freq = Period._maybe_convert_freq(freq)
        if isinstance(values, ABCSeries):
            values = values._values
            if (not isinstance(values, type(self))):
                raise TypeError('Incorrect dtype')
        elif isinstance(values, ABCPeriodIndex):
            values = values._values
        if isinstance(values, type(self)):
            if ((freq is not None) and (freq != values.freq)):
                raise raise_on_incompatible(values, freq)
            (values, freq) = (values._data, values.freq)
        values = np.array(values, dtype='int64', copy=copy)
        self._data = values
        if (freq is None):
            raise ValueError('freq is not specified and cannot be inferred')
        self._dtype = PeriodDtype(freq)

    @classmethod
    def _simple_new(cls, values, freq=None, dtype=None):
        assertion_msg = 'Should be numpy array of type i8'
        assert (isinstance(values, np.ndarray) and (values.dtype == 'i8')), assertion_msg
        return cls(values, freq=freq, dtype=dtype)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if (dtype and isinstance(dtype, PeriodDtype)):
            freq = dtype.freq
        else:
            freq = None
        if isinstance(scalars, cls):
            validate_dtype_freq(scalars.dtype, freq)
            if copy:
                scalars = scalars.copy()
            return scalars
        periods = np.asarray(scalars, dtype=object)
        freq = (freq or libperiod.extract_freq(periods))
        ordinals = libperiod.extract_ordinals(periods, freq)
        return cls(ordinals, freq=freq)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_datetime64(cls, data, freq, tz=None):
        '\n        Construct a PeriodArray from a datetime64 array\n\n        Parameters\n        ----------\n        data : ndarray[datetime64[ns], datetime64[ns, tz]]\n        freq : str or Tick\n        tz : tzinfo, optional\n\n        Returns\n        -------\n        PeriodArray[freq]\n        '
        (data, freq) = dt64arr_to_periodarr(data, freq, tz)
        return cls(data, freq=freq)

    @classmethod
    def _generate_range(cls, start, end, periods, freq, fields):
        periods = dtl.validate_periods(periods)
        if (freq is not None):
            freq = Period._maybe_convert_freq(freq)
        field_count = len(fields)
        if ((start is not None) or (end is not None)):
            if (field_count > 0):
                raise ValueError('Can either instantiate from fields or endpoints, but not both')
            (subarr, freq) = _get_ordinal_range(start, end, periods, freq)
        elif (field_count > 0):
            (subarr, freq) = _range_from_fields(freq=freq, **fields)
        else:
            raise ValueError('Not enough parameters to construct Period range')
        return (subarr, freq)

    def _unbox_scalar(self, value, setitem=False):
        if (value is NaT):
            return np.int64(value.value)
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value, setitem=setitem)
            return np.int64(value.ordinal)
        else:
            raise ValueError(f"'value' should be a Period. Got '{value}' instead.")

    def _scalar_from_string(self, value):
        return Period(value, freq=self.freq)

    def _check_compatible_with(self, other, setitem=False):
        if (other is NaT):
            return
        if (self.freqstr != other.freqstr):
            raise raise_on_incompatible(self, other)

    @cache_readonly
    def dtype(self):
        return self._dtype

    @property
    def freq(self):
        '\n        Return the frequency object for this PeriodArray.\n        '
        return self.dtype.freq

    def __array__(self, dtype=None):
        if (dtype == 'i8'):
            return self.asi8
        elif (dtype == bool):
            return (~ self._isnan)
        return np.array(list(self), dtype=object)

    def __arrow_array__(self, type=None):
        '\n        Convert myself into a pyarrow Array.\n        '
        import pyarrow
        from pandas.core.arrays._arrow_utils import ArrowPeriodType
        if (type is not None):
            if pyarrow.types.is_integer(type):
                return pyarrow.array(self._data, mask=self.isna(), type=type)
            elif isinstance(type, ArrowPeriodType):
                if (self.freqstr != type.freq):
                    raise TypeError(f"Not supported to convert PeriodArray to array with different 'freq' ({self.freqstr} vs {type.freq})")
            else:
                raise TypeError(f"Not supported to convert PeriodArray to '{type}' type")
        period_type = ArrowPeriodType(self.freqstr)
        storage_array = pyarrow.array(self._data, mask=self.isna(), type='int64')
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)
    year = _field_accessor('year', '\n        The year of the period.\n        ')
    month = _field_accessor('month', '\n        The month as January=1, December=12.\n        ')
    day = _field_accessor('day', '\n        The days of the period.\n        ')
    hour = _field_accessor('hour', '\n        The hour of the period.\n        ')
    minute = _field_accessor('minute', '\n        The minute of the period.\n        ')
    second = _field_accessor('second', '\n        The second of the period.\n        ')
    weekofyear = _field_accessor('week', '\n        The week ordinal of the year.\n        ')
    week = weekofyear
    day_of_week = _field_accessor('day_of_week', '\n        The day of the week with Monday=0, Sunday=6.\n        ')
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor('day_of_year', '\n        The ordinal day of the year.\n        ')
    quarter = _field_accessor('quarter', '\n        The quarter of the date.\n        ')
    qyear = _field_accessor('qyear')
    days_in_month = _field_accessor('days_in_month', '\n        The number of days in the month.\n        ')
    daysinmonth = days_in_month

    @property
    def is_leap_year(self):
        '\n        Logical indicating if the date belongs to a leap year.\n        '
        return isleapyear_arr(np.asarray(self.year))

    @property
    def start_time(self):
        return self.to_timestamp(how='start')

    @property
    def end_time(self):
        return self.to_timestamp(how='end')

    def to_timestamp(self, freq=None, how='start'):
        "\n        Cast to DatetimeArray/Index.\n\n        Parameters\n        ----------\n        freq : str or DateOffset, optional\n            Target frequency. The default is 'D' for week or longer,\n            'S' otherwise.\n        how : {'s', 'e', 'start', 'end'}\n            Whether to use the start or end of the time period being converted.\n\n        Returns\n        -------\n        DatetimeArray/Index\n        "
        from pandas.core.arrays import DatetimeArray
        how = libperiod.validate_end_alias(how)
        end = (how == 'E')
        if end:
            if ((freq == 'B') or (self.freq == 'B')):
                adjust = (Timedelta(1, 'D') - Timedelta(1, 'ns'))
                return (self.to_timestamp(how='start') + adjust)
            else:
                adjust = Timedelta(1, 'ns')
                return ((self + self.freq).to_timestamp(how='start') - adjust)
        if (freq is None):
            freq = self._get_to_timestamp_base()
            base = freq
        else:
            freq = Period._maybe_convert_freq(freq)
            base = freq._period_dtype_code
        new_data = self.asfreq(freq, how=how)
        new_data = libperiod.periodarr_to_dt64arr(new_data.asi8, base)
        return DatetimeArray(new_data)._with_freq('infer')

    def _time_shift(self, periods, freq=None):
        '\n        Shift each value by `periods`.\n\n        Note this is different from ExtensionArray.shift, which\n        shifts the *position* of each element, padding the end with\n        missing values.\n\n        Parameters\n        ----------\n        periods : int\n            Number of periods to shift by.\n        freq : pandas.DateOffset, pandas.Timedelta, or str\n            Frequency increment to shift by.\n        '
        if (freq is not None):
            raise TypeError(f'`freq` argument is not supported for {type(self).__name__}._time_shift')
        values = (self.asi8 + (periods * self.freq.n))
        if self._hasnans:
            values[self._isnan] = iNaT
        return type(self)(values, freq=self.freq)

    def _box_func(self, x):
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    @doc(**_shared_doc_kwargs, other='PeriodIndex', other_name='PeriodIndex')
    def asfreq(self, freq=None, how='E'):
        "\n        Convert the {klass} to the specified frequency `freq`.\n\n        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments\n        to each :class:`~pandas.Period` in this {klass}.\n\n        Parameters\n        ----------\n        freq : str\n            A frequency.\n        how : str {{'E', 'S'}}, default 'E'\n            Whether the elements should be aligned to the end\n            or start within pa period.\n\n            * 'E', 'END', or 'FINISH' for end,\n            * 'S', 'START', or 'BEGIN' for start.\n\n            January 31st ('END') vs. January 1st ('START') for example.\n\n        Returns\n        -------\n        {klass}\n            The transformed {klass} with the new frequency.\n\n        See Also\n        --------\n        {other}.asfreq: Convert each Period in a {other_name} to the given frequency.\n        Period.asfreq : Convert a :class:`~pandas.Period` object to the given frequency.\n\n        Examples\n        --------\n        >>> pidx = pd.period_range('2010-01-01', '2015-01-01', freq='A')\n        >>> pidx\n        PeriodIndex(['2010', '2011', '2012', '2013', '2014', '2015'],\n        dtype='period[A-DEC]', freq='A-DEC')\n\n        >>> pidx.asfreq('M')\n        PeriodIndex(['2010-12', '2011-12', '2012-12', '2013-12', '2014-12',\n        '2015-12'], dtype='period[M]', freq='M')\n\n        >>> pidx.asfreq('M', how='S')\n        PeriodIndex(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01',\n        '2015-01'], dtype='period[M]', freq='M')\n        "
        how = libperiod.validate_end_alias(how)
        freq = Period._maybe_convert_freq(freq)
        base1 = self.freq._period_dtype_code
        base2 = freq._period_dtype_code
        asi8 = self.asi8
        end = (how == 'E')
        if end:
            ordinal = ((asi8 + self.freq.n) - 1)
        else:
            ordinal = asi8
        new_data = period_asfreq_arr(ordinal, base1, base2, end)
        if self._hasnans:
            new_data[self._isnan] = iNaT
        return type(self)(new_data, freq=freq)

    def _formatter(self, boxed=False):
        if boxed:
            return str
        return "'{}'".format

    @dtl.ravel_compat
    def _format_native_types(self, na_rep='NaT', date_format=None, **kwargs):
        '\n        actually format my specific types\n        '
        values = self.astype(object)
        if date_format:
            formatter = (lambda dt: dt.strftime(date_format))
        else:
            formatter = (lambda dt: str(dt))
        if self._hasnans:
            mask = self._isnan
            values[mask] = na_rep
            imask = (~ mask)
            values[imask] = np.array([formatter(dt) for dt in values[imask]])
        else:
            values = np.array([formatter(dt) for dt in values])
        return values

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, self._dtype):
            if (not copy):
                return self
            else:
                return self.copy()
        if is_period_dtype(dtype):
            return self.asfreq(dtype.freq)
        return super().astype(dtype, copy=copy)

    def searchsorted(self, value, side='left', sorter=None):
        value = self._validate_searchsorted_value(value).view('M8[ns]')
        m8arr = self._ndarray.view('M8[ns]')
        return m8arr.searchsorted(value, side=side, sorter=sorter)

    def _sub_datelike(self, other):
        assert (other is not NaT)
        return NotImplemented

    def _sub_period(self, other):
        self._check_compatible_with(other)
        asi8 = self.asi8
        new_data = (asi8 - other.ordinal)
        new_data = np.array([(self.freq * x) for x in new_data])
        if self._hasnans:
            new_data[self._isnan] = NaT
        return new_data

    def _sub_period_array(self, other):
        '\n        Subtract a Period Array/Index from self.  This is only valid if self\n        is itself a Period Array/Index, raises otherwise.  Both objects must\n        have the same frequency.\n\n        Parameters\n        ----------\n        other : PeriodIndex or PeriodArray\n\n        Returns\n        -------\n        result : np.ndarray[object]\n            Array of DateOffset objects; nulls represented by NaT.\n        '
        if (self.freq != other.freq):
            msg = DIFFERENT_FREQ.format(cls=type(self).__name__, own_freq=self.freqstr, other_freq=other.freqstr)
            raise IncompatibleFrequency(msg)
        new_values = algos.checked_add_with_arr(self.asi8, (- other.asi8), arr_mask=self._isnan, b_mask=other._isnan)
        new_values = np.array([(self.freq.base * x) for x in new_values])
        if (self._hasnans or other._hasnans):
            mask = (self._isnan | other._isnan)
            new_values[mask] = NaT
        return new_values

    def _addsub_int_array(self, other, op):
        '\n        Add or subtract array of integers; equivalent to applying\n        `_time_shift` pointwise.\n\n        Parameters\n        ----------\n        other : np.ndarray[integer-dtype]\n        op : {operator.add, operator.sub}\n\n        Returns\n        -------\n        result : PeriodArray\n        '
        assert (op in [operator.add, operator.sub])
        if (op is operator.sub):
            other = (- other)
        res_values = algos.checked_add_with_arr(self.asi8, other, arr_mask=self._isnan)
        res_values = res_values.view('i8')
        np.putmask(res_values, self._isnan, iNaT)
        return type(self)(res_values, freq=self.freq)

    def _add_offset(self, other):
        assert (not isinstance(other, Tick))
        if (other.base != self.freq.base):
            raise raise_on_incompatible(self, other)
        result = super()._add_timedeltalike_scalar(other.n)
        return type(self)(result, freq=self.freq)

    def _add_timedeltalike_scalar(self, other):
        '\n        Parameters\n        ----------\n        other : timedelta, Tick, np.timedelta64\n\n        Returns\n        -------\n        PeriodArray\n        '
        if (not isinstance(self.freq, Tick)):
            raise raise_on_incompatible(self, other)
        if notna(other):
            other = self._check_timedeltalike_freq_compat(other)
        return super()._add_timedeltalike_scalar(other)

    def _add_timedelta_arraylike(self, other):
        '\n        Parameters\n        ----------\n        other : TimedeltaArray or ndarray[timedelta64]\n\n        Returns\n        -------\n        result : ndarray[int64]\n        '
        if (not isinstance(self.freq, Tick)):
            raise TypeError(f'Cannot add or subtract timedelta64[ns] dtype from {self.dtype}')
        if (not np.all(isna(other))):
            delta = self._check_timedeltalike_freq_compat(other)
        else:
            return (self + np.timedelta64('NaT'))
        ordinals = self._addsub_int_array(delta, operator.add).asi8
        return type(self)(ordinals, dtype=self.dtype)

    def _check_timedeltalike_freq_compat(self, other):
        '\n        Arithmetic operations with timedelta-like scalars or array `other`\n        are only valid if `other` is an integer multiple of `self.freq`.\n        If the operation is valid, find that integer multiple.  Otherwise,\n        raise because the operation is invalid.\n\n        Parameters\n        ----------\n        other : timedelta, np.timedelta64, Tick,\n                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex\n\n        Returns\n        -------\n        multiple : int or ndarray[int64]\n\n        Raises\n        ------\n        IncompatibleFrequency\n        '
        assert isinstance(self.freq, Tick)
        base_nanos = self.freq.base.nanos
        if isinstance(other, (timedelta, np.timedelta64, Tick)):
            nanos = delta_to_nanoseconds(other)
        elif isinstance(other, np.ndarray):
            assert (other.dtype.kind == 'm')
            if (other.dtype != TD64NS_DTYPE):
                other = other.astype(TD64NS_DTYPE)
            nanos = other.view('i8')
        else:
            nanos = other.asi8
        if np.all(((nanos % base_nanos) == 0)):
            delta = (nanos // base_nanos)
            return delta
        raise raise_on_incompatible(self, other)

def raise_on_incompatible(left, right):
    '\n    Helper function to render a consistent error message when raising\n    IncompatibleFrequency.\n\n    Parameters\n    ----------\n    left : PeriodArray\n    right : None, DateOffset, Period, ndarray, or timedelta-like\n\n    Returns\n    -------\n    IncompatibleFrequency\n        Exception to be raised by the caller.\n    '
    if (isinstance(right, (np.ndarray, ABCTimedeltaArray)) or (right is None)):
        other_freq = None
    elif isinstance(right, (ABCPeriodIndex, PeriodArray, Period, BaseOffset)):
        other_freq = right.freqstr
    else:
        other_freq = delta_to_tick(Timedelta(right)).freqstr
    msg = DIFFERENT_FREQ.format(cls=type(left).__name__, own_freq=left.freqstr, other_freq=other_freq)
    return IncompatibleFrequency(msg)

def period_array(data, freq=None, copy=False):
    "\n    Construct a new PeriodArray from a sequence of Period scalars.\n\n    Parameters\n    ----------\n    data : Sequence of Period objects\n        A sequence of Period objects. These are required to all have\n        the same ``freq.`` Missing values can be indicated by ``None``\n        or ``pandas.NaT``.\n    freq : str, Tick, or Offset\n        The frequency of every element of the array. This can be specified\n        to avoid inferring the `freq` from `data`.\n    copy : bool, default False\n        Whether to ensure a copy of the data is made.\n\n    Returns\n    -------\n    PeriodArray\n\n    See Also\n    --------\n    PeriodArray\n    pandas.PeriodIndex\n\n    Examples\n    --------\n    >>> period_array([pd.Period('2017', freq='A'),\n    ...               pd.Period('2018', freq='A')])\n    <PeriodArray>\n    ['2017', '2018']\n    Length: 2, dtype: period[A-DEC]\n\n    >>> period_array([pd.Period('2017', freq='A'),\n    ...               pd.Period('2018', freq='A'),\n    ...               pd.NaT])\n    <PeriodArray>\n    ['2017', '2018', 'NaT']\n    Length: 3, dtype: period[A-DEC]\n\n    Integers that look like years are handled\n\n    >>> period_array([2000, 2001, 2002], freq='D')\n    <PeriodArray>\n    ['2000-01-01', '2001-01-01', '2002-01-01']\n    Length: 3, dtype: period[D]\n\n    Datetime-like strings may also be passed\n\n    >>> period_array(['2000-Q1', '2000-Q2', '2000-Q3', '2000-Q4'], freq='Q')\n    <PeriodArray>\n    ['2000Q1', '2000Q2', '2000Q3', '2000Q4']\n    Length: 4, dtype: period[Q-DEC]\n    "
    data_dtype = getattr(data, 'dtype', None)
    if is_datetime64_dtype(data_dtype):
        return PeriodArray._from_datetime64(data, freq)
    if is_period_dtype(data_dtype):
        return PeriodArray(data, freq=freq)
    if (not isinstance(data, (np.ndarray, list, tuple, ABCSeries))):
        data = list(data)
    data = np.asarray(data)
    dtype: Optional[PeriodDtype]
    if freq:
        dtype = PeriodDtype(freq)
    else:
        dtype = None
    if (is_float_dtype(data) and (len(data) > 0)):
        raise TypeError('PeriodIndex does not allow floating point in construction')
    data = ensure_object(data)
    return PeriodArray._from_sequence(data, dtype=dtype)

def validate_dtype_freq(dtype, freq):
    '\n    If both a dtype and a freq are available, ensure they match.  If only\n    dtype is available, extract the implied freq.\n\n    Parameters\n    ----------\n    dtype : dtype\n    freq : DateOffset or None\n\n    Returns\n    -------\n    freq : DateOffset\n\n    Raises\n    ------\n    ValueError : non-period dtype\n    IncompatibleFrequency : mismatch between dtype and freq\n    '
    if (freq is not None):
        freq = to_offset(freq)
    if (dtype is not None):
        dtype = pandas_dtype(dtype)
        if (not is_period_dtype(dtype)):
            raise ValueError('dtype must be PeriodDtype')
        if (freq is None):
            freq = dtype.freq
        elif (freq != dtype.freq):
            raise IncompatibleFrequency('specified freq and dtype are different')
    return freq

def dt64arr_to_periodarr(data, freq, tz=None):
    "\n    Convert an datetime-like array to values Period ordinals.\n\n    Parameters\n    ----------\n    data : Union[Series[datetime64[ns]], DatetimeIndex, ndarray[datetime64ns]]\n    freq : Optional[Union[str, Tick]]\n        Must match the `freq` on the `data` if `data` is a DatetimeIndex\n        or Series.\n    tz : Optional[tzinfo]\n\n    Returns\n    -------\n    ordinals : ndarray[int]\n    freq : Tick\n        The frequency extracted from the Series or DatetimeIndex if that's\n        used.\n\n    "
    if (data.dtype != np.dtype('M8[ns]')):
        raise ValueError(f'Wrong dtype: {data.dtype}')
    if (freq is None):
        if isinstance(data, ABCIndex):
            (data, freq) = (data._values, data.freq)
        elif isinstance(data, ABCSeries):
            (data, freq) = (data._values, data.dt.freq)
    freq = Period._maybe_convert_freq(freq)
    if isinstance(data, (ABCIndex, ABCSeries)):
        data = data._values
    base = freq._period_dtype_code
    return (c_dt64arr_to_periodarr(data.view('i8'), base, tz), freq)

def _get_ordinal_range(start, end, periods, freq, mult=1):
    if (com.count_not_none(start, end, periods) != 2):
        raise ValueError('Of the three parameters: start, end, and periods, exactly two must be specified')
    if (freq is not None):
        freq = to_offset(freq)
        mult = freq.n
    if (start is not None):
        start = Period(start, freq)
    if (end is not None):
        end = Period(end, freq)
    is_start_per = isinstance(start, Period)
    is_end_per = isinstance(end, Period)
    if (is_start_per and is_end_per and (start.freq != end.freq)):
        raise ValueError('start and end must have same freq')
    if ((start is NaT) or (end is NaT)):
        raise ValueError('start and end must not be NaT')
    if (freq is None):
        if is_start_per:
            freq = start.freq
        elif is_end_per:
            freq = end.freq
        else:
            raise ValueError('Could not infer freq from start/end')
    if (periods is not None):
        periods = (periods * mult)
        if (start is None):
            data = np.arange(((end.ordinal - periods) + mult), (end.ordinal + 1), mult, dtype=np.int64)
        else:
            data = np.arange(start.ordinal, (start.ordinal + periods), mult, dtype=np.int64)
    else:
        data = np.arange(start.ordinal, (end.ordinal + 1), mult, dtype=np.int64)
    return (data, freq)

def _range_from_fields(year=None, month=None, quarter=None, day=None, hour=None, minute=None, second=None, freq=None):
    if (hour is None):
        hour = 0
    if (minute is None):
        minute = 0
    if (second is None):
        second = 0
    if (day is None):
        day = 1
    ordinals = []
    if (quarter is not None):
        if (freq is None):
            freq = to_offset('Q')
            base = FreqGroup.FR_QTR
        else:
            freq = to_offset(freq)
            base = libperiod.freq_to_dtype_code(freq)
            if (base != FreqGroup.FR_QTR):
                raise AssertionError('base must equal FR_QTR')
        freqstr = freq.freqstr
        (year, quarter) = _make_field_arrays(year, quarter)
        for (y, q) in zip(year, quarter):
            (y, m) = libperiod.quarter_to_myear(y, q, freqstr)
            val = libperiod.period_ordinal(y, m, 1, 1, 1, 1, 0, 0, base)
            ordinals.append(val)
    else:
        freq = to_offset(freq)
        base = libperiod.freq_to_dtype_code(freq)
        arrays = _make_field_arrays(year, month, day, hour, minute, second)
        for (y, mth, d, h, mn, s) in zip(*arrays):
            ordinals.append(libperiod.period_ordinal(y, mth, d, h, mn, s, 0, 0, base))
    return (np.array(ordinals, dtype=np.int64), freq)

def _make_field_arrays(*fields):
    length = None
    for x in fields:
        if isinstance(x, (list, np.ndarray, ABCSeries)):
            if ((length is not None) and (len(x) != length)):
                raise ValueError('Mismatched Period array lengths')
            elif (length is None):
                length = len(x)
    return [(np.asarray(x) if isinstance(x, (np.ndarray, list, ABCSeries)) else np.repeat(x, length)) for x in fields]
