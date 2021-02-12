
from datetime import date, datetime, time, timedelta, tzinfo
import operator
from typing import TYPE_CHECKING, Optional, Tuple
import warnings
import numpy as np
from pandas._libs import NaT, Period, Timestamp, index as libindex, lib
from pandas._libs.tslibs import Resolution, ints_to_pydatetime, parsing, timezones, to_offset
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas._typing import DtypeObj
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import DT64NS_DTYPE, is_datetime64_dtype, is_datetime64tz_dtype, is_scalar
from pandas.core.dtypes.missing import is_valid_nat_for_dtype
from pandas.core.arrays.datetimes import DatetimeArray, tz_to_dtype
import pandas.core.common as com
from pandas.core.indexes.base import Index, get_unanimous_names, maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time
if TYPE_CHECKING:
    from pandas import DataFrame, Float64Index, PeriodIndex, TimedeltaIndex

def _new_DatetimeIndex(cls, d):
    "\n    This is called upon unpickling, rather than the default which doesn't\n    have arguments and breaks __new__\n    "
    if (('data' in d) and (not isinstance(d['data'], DatetimeIndex))):
        data = d.pop('data')
        if (not isinstance(data, DatetimeArray)):
            tz = d.pop('tz')
            freq = d.pop('freq')
            dta = DatetimeArray._simple_new(data, dtype=tz_to_dtype(tz), freq=freq)
        else:
            dta = data
            for key in ['tz', 'freq']:
                if (key in d):
                    assert (d.pop(key) == getattr(dta, key))
        result = cls._simple_new(dta, **d)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = cls.__new__(cls, **d)
    return result

@inherit_names((DatetimeArray._field_ops + [method for method in DatetimeArray._datetimelike_methods if (method not in ('tz_localize', 'tz_convert'))]), DatetimeArray, wrap=True)
@inherit_names(['is_normalized', '_resolution_obj'], DatetimeArray, cache=True)
@inherit_names((['_bool_ops', '_object_ops', '_field_ops', '_datetimelike_ops', '_datetimelike_methods', 'tz', 'tzinfo', 'dtype', 'to_pydatetime', '_has_same_tz', '_format_native_types', 'date', 'time', 'timetz', 'std'] + DatetimeArray._bool_ops), DatetimeArray)
class DatetimeIndex(DatetimeTimedeltaMixin):
    "\n    Immutable ndarray-like of datetime64 data.\n\n    Represented internally as int64, and which can be boxed to Timestamp objects\n    that are subclasses of datetime and carry metadata.\n\n    Parameters\n    ----------\n    data : array-like (1-dimensional), optional\n        Optional datetime-like data to construct index with.\n    freq : str or pandas offset object, optional\n        One of pandas date offset strings or corresponding objects. The string\n        'infer' can be passed in order to set the frequency of the index as the\n        inferred frequency upon creation.\n    tz : pytz.timezone or dateutil.tz.tzfile or datetime.tzinfo or str\n        Set the Timezone of the data.\n    normalize : bool, default False\n        Normalize start/end dates to midnight before generating date range.\n    closed : {'left', 'right'}, optional\n        Set whether to include `start` and `end` that are on the\n        boundary. The default includes boundary points on either end.\n    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'\n        When clocks moved backward due to DST, ambiguous times may arise.\n        For example in Central European Time (UTC+01), when going from 03:00\n        DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC\n        and at 01:30:00 UTC. In such a situation, the `ambiguous` parameter\n        dictates how ambiguous times should be handled.\n\n        - 'infer' will attempt to infer fall dst-transition hours based on\n          order\n        - bool-ndarray where True signifies a DST time, False signifies a\n          non-DST time (note that this flag is only applicable for ambiguous\n          times)\n        - 'NaT' will return NaT where there are ambiguous times\n        - 'raise' will raise an AmbiguousTimeError if there are ambiguous times.\n    dayfirst : bool, default False\n        If True, parse dates in `data` with the day first order.\n    yearfirst : bool, default False\n        If True parse dates in `data` with the year first order.\n    dtype : numpy.dtype or DatetimeTZDtype or str, default None\n        Note that the only NumPy dtype allowed is ‘datetime64[ns]’.\n    copy : bool, default False\n        Make a copy of input ndarray.\n    name : label, default None\n        Name to be stored in the index.\n\n    Attributes\n    ----------\n    year\n    month\n    day\n    hour\n    minute\n    second\n    microsecond\n    nanosecond\n    date\n    time\n    timetz\n    dayofyear\n    day_of_year\n    weekofyear\n    week\n    dayofweek\n    day_of_week\n    weekday\n    quarter\n    tz\n    freq\n    freqstr\n    is_month_start\n    is_month_end\n    is_quarter_start\n    is_quarter_end\n    is_year_start\n    is_year_end\n    is_leap_year\n    inferred_freq\n\n    Methods\n    -------\n    normalize\n    strftime\n    snap\n    tz_convert\n    tz_localize\n    round\n    floor\n    ceil\n    to_period\n    to_perioddelta\n    to_pydatetime\n    to_series\n    to_frame\n    month_name\n    day_name\n    mean\n    std\n\n    See Also\n    --------\n    Index : The base pandas Index type.\n    TimedeltaIndex : Index of timedelta64 data.\n    PeriodIndex : Index of Period data.\n    to_datetime : Convert argument to datetime.\n    date_range : Create a fixed-frequency DatetimeIndex.\n\n    Notes\n    -----\n    To learn more about the frequency strings, please see `this link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n    "
    _typ = 'datetimeindex'
    _data_cls = DatetimeArray
    _engine_type = libindex.DatetimeEngine
    _supports_partial_string_indexing = True
    _comparables = ['name', 'freqstr', 'tz']
    _attributes = ['name', 'tz', 'freq']
    _is_numeric_dtype = False

    @doc(DatetimeArray.strftime)
    def strftime(self, date_format):
        arr = self._data.strftime(date_format)
        return Index(arr, name=self.name)

    @doc(DatetimeArray.tz_convert)
    def tz_convert(self, tz):
        arr = self._data.tz_convert(tz)
        return type(self)._simple_new(arr, name=self.name)

    @doc(DatetimeArray.tz_localize)
    def tz_localize(self, tz, ambiguous='raise', nonexistent='raise'):
        arr = self._data.tz_localize(tz, ambiguous, nonexistent)
        return type(self)._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_period)
    def to_period(self, freq=None):
        from pandas.core.indexes.api import PeriodIndex
        arr = self._data.to_period(freq)
        return PeriodIndex._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_perioddelta)
    def to_perioddelta(self, freq):
        from pandas.core.indexes.api import TimedeltaIndex
        arr = self._data.to_perioddelta(freq)
        return TimedeltaIndex._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_julian_date)
    def to_julian_date(self):
        from pandas.core.indexes.api import Float64Index
        arr = self._data.to_julian_date()
        return Float64Index._simple_new(arr, name=self.name)

    @doc(DatetimeArray.isocalendar)
    def isocalendar(self):
        df = self._data.isocalendar()
        return df.set_index(self)

    def __new__(cls, data=None, freq=lib.no_default, tz=None, normalize=False, closed=None, ambiguous='raise', dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None):
        if is_scalar(data):
            raise TypeError(f'{cls.__name__}() must be called with a collection of some kind, {repr(data)} was passed')
        name = maybe_extract_name(name, data, cls)
        dtarr = DatetimeArray._from_sequence_not_strict(data, dtype=dtype, copy=copy, tz=tz, freq=freq, dayfirst=dayfirst, yearfirst=yearfirst, ambiguous=ambiguous)
        subarr = cls._simple_new(dtarr, name=name)
        return subarr

    @cache_readonly
    def _is_dates_only(self):
        "\n        Return a boolean if we are only dates (and don't have a timezone)\n\n        Returns\n        -------\n        bool\n        "
        from pandas.io.formats.format import is_dates_only
        return ((self.tz is None) and is_dates_only(self._values))

    def __reduce__(self):
        d = {'data': self._data}
        d.update(self._get_attributes_dict())
        return (_new_DatetimeIndex, (type(self), d), None)

    def _validate_fill_value(self, value):
        '\n        Convert value to be insertable to ndarray.\n        '
        return self._data._validate_setitem_value(value)

    def _is_comparable_dtype(self, dtype):
        '\n        Can we compare values of the given dtype to our own?\n        '
        if (self.tz is not None):
            return is_datetime64tz_dtype(dtype)
        return is_datetime64_dtype(dtype)

    def _mpl_repr(self):
        return ints_to_pydatetime(self.asi8, self.tz)

    @property
    def _formatter_func(self):
        from pandas.io.formats.format import get_format_datetime64
        formatter = get_format_datetime64(is_dates_only=self._is_dates_only)
        return (lambda x: f"'{formatter(x)}'")

    def union_many(self, others):
        '\n        A bit of a hack to accelerate unioning a collection of indexes.\n        '
        this = self
        for other in others:
            if (not isinstance(this, DatetimeIndex)):
                this = Index.union(this, other)
                continue
            if (not isinstance(other, DatetimeIndex)):
                try:
                    other = DatetimeIndex(other)
                except TypeError:
                    pass
            (this, other) = this._maybe_utc_convert(other)
            if this._can_fast_union(other):
                this = this._fast_union(other)
            else:
                this = Index.union(this, other)
        res_name = get_unanimous_names(self, *others)[0]
        if (this.name != res_name):
            return this.rename(res_name)
        return this

    def _maybe_utc_convert(self, other):
        this = self
        if isinstance(other, DatetimeIndex):
            if ((self.tz is None) ^ (other.tz is None)):
                raise TypeError('Cannot join tz-naive with tz-aware DatetimeIndex')
            if (not timezones.tz_compare(self.tz, other.tz)):
                this = self.tz_convert('UTC')
                other = other.tz_convert('UTC')
        return (this, other)

    def _get_time_micros(self):
        '\n        Return the number of microseconds since midnight.\n\n        Returns\n        -------\n        ndarray[int64_t]\n        '
        values = self._data._local_timestamps()
        nanos = (values % ((24 * 3600) * 1000000000))
        micros = (nanos // 1000)
        micros[self._isnan] = (- 1)
        return micros

    def to_series(self, keep_tz=lib.no_default, index=None, name=None):
        '\n        Create a Series with both index and values equal to the index keys\n        useful with map for returning an indexer based on an index.\n\n        Parameters\n        ----------\n        keep_tz : optional, defaults True\n            Return the data keeping the timezone.\n\n            If keep_tz is True:\n\n              If the timezone is not set, the resulting\n              Series will have a datetime64[ns] dtype.\n\n              Otherwise the Series will have an datetime64[ns, tz] dtype; the\n              tz will be preserved.\n\n            If keep_tz is False:\n\n              Series will have a datetime64[ns] dtype. TZ aware\n              objects will have the tz removed.\n\n            .. versionchanged:: 1.0.0\n                The default value is now True.  In a future version,\n                this keyword will be removed entirely.  Stop passing the\n                argument to obtain the future behavior and silence the warning.\n\n        index : Index, optional\n            Index of resulting Series. If None, defaults to original index.\n        name : str, optional\n            Name of resulting Series. If None, defaults to name of original\n            index.\n\n        Returns\n        -------\n        Series\n        '
        from pandas import Series
        if (index is None):
            index = self._shallow_copy()
        if (name is None):
            name = self.name
        if (keep_tz is not lib.no_default):
            if keep_tz:
                warnings.warn("The 'keep_tz' keyword in DatetimeIndex.to_series is deprecated and will be removed in a future version.  You can stop passing 'keep_tz' to silence this warning.", FutureWarning, stacklevel=2)
            else:
                warnings.warn("Specifying 'keep_tz=False' is deprecated and this option will be removed in a future release. If you want to remove the timezone information, you can do 'idx.tz_convert(None)' before calling 'to_series'.", FutureWarning, stacklevel=2)
        else:
            keep_tz = True
        if (keep_tz and (self.tz is not None)):
            values = self.copy(deep=True)
        else:
            values = self._values.view('M8[ns]').copy()
        return Series(values, index=index, name=name)

    def snap(self, freq='S'):
        '\n        Snap time stamps to nearest occurring frequency.\n\n        Returns\n        -------\n        DatetimeIndex\n        '
        freq = to_offset(freq)
        snapped = np.empty(len(self), dtype=DT64NS_DTYPE)
        for (i, v) in enumerate(self):
            s = v
            if (not freq.is_on_offset(s)):
                t0 = freq.rollback(s)
                t1 = freq.rollforward(s)
                if (abs((s - t0)) < abs((t1 - s))):
                    s = t0
                else:
                    s = t1
            snapped[i] = s
        dta = DatetimeArray(snapped, dtype=self.dtype)
        return DatetimeIndex._simple_new(dta, name=self.name)

    def _parsed_string_to_bounds(self, reso, parsed):
        '\n        Calculate datetime bounds for parsed time string and its resolution.\n\n        Parameters\n        ----------\n        reso : str\n            Resolution provided by parsed string.\n        parsed : datetime\n            Datetime from parsed string.\n\n        Returns\n        -------\n        lower, upper: pd.Timestamp\n        '
        assert isinstance(reso, Resolution), (type(reso), reso)
        valid_resos = {'year', 'month', 'quarter', 'day', 'hour', 'minute', 'second', 'minute', 'second', 'microsecond'}
        if (reso.attrname not in valid_resos):
            raise KeyError
        grp = reso.freq_group
        per = Period(parsed, freq=grp)
        (start, end) = (per.start_time, per.end_time)
        if (parsed.tzinfo is not None):
            if (self.tz is None):
                raise ValueError('The index must be timezone aware when indexing with a date string with a UTC offset')
            start = start.tz_localize(parsed.tzinfo).tz_convert(self.tz)
            end = end.tz_localize(parsed.tzinfo).tz_convert(self.tz)
        elif (self.tz is not None):
            start = start.tz_localize(self.tz)
            end = end.tz_localize(self.tz)
        return (start, end)

    def _validate_partial_date_slice(self, reso):
        assert isinstance(reso, Resolution), (type(reso), reso)
        if (self.is_monotonic and (reso.attrname in ['day', 'hour', 'minute', 'second']) and (self._resolution_obj >= reso)):
            raise KeyError
        if (reso == 'microsecond'):
            raise KeyError

    def _deprecate_mismatched_indexing(self, key):
        try:
            self._data._assert_tzawareness_compat(key)
        except TypeError:
            if (self.tz is None):
                msg = 'Indexing a timezone-naive DatetimeIndex with a timezone-aware datetime is deprecated and will raise KeyError in a future version.  Use a timezone-naive object instead.'
            else:
                msg = 'Indexing a timezone-aware DatetimeIndex with a timezone-naive datetime is deprecated and will raise KeyError in a future version.  Use a timezone-aware object instead.'
            warnings.warn(msg, FutureWarning, stacklevel=5)

    def get_loc(self, key, method=None, tolerance=None):
        '\n        Get integer location for requested label\n\n        Returns\n        -------\n        loc : int\n        '
        if (not is_scalar(key)):
            raise InvalidIndexError(key)
        orig_key = key
        if is_valid_nat_for_dtype(key, self.dtype):
            key = NaT
        if isinstance(key, self._data._recognized_scalars):
            self._deprecate_mismatched_indexing(key)
            key = self._maybe_cast_for_get_loc(key)
        elif isinstance(key, str):
            try:
                return self._get_string_slice(key)
            except (TypeError, KeyError, ValueError, OverflowError):
                pass
            try:
                key = self._maybe_cast_for_get_loc(key)
            except ValueError as err:
                raise KeyError(key) from err
        elif isinstance(key, timedelta):
            raise TypeError(f'Cannot index {type(self).__name__} with {type(key).__name__}')
        elif isinstance(key, time):
            if (method is not None):
                raise NotImplementedError('cannot yet lookup inexact labels when key is a time object')
            return self.indexer_at_time(key)
        else:
            raise KeyError(key)
        try:
            return Index.get_loc(self, key, method, tolerance)
        except KeyError as err:
            raise KeyError(orig_key) from err

    def _maybe_cast_for_get_loc(self, key):
        key = Timestamp(key)
        if (key.tzinfo is None):
            key = key.tz_localize(self.tz)
        else:
            key = key.tz_convert(self.tz)
        return key

    def _maybe_cast_slice_bound(self, label, side, kind):
        "\n        If label is a string, cast it to datetime according to resolution.\n\n        Parameters\n        ----------\n        label : object\n        side : {'left', 'right'}\n        kind : {'loc', 'getitem'} or None\n\n        Returns\n        -------\n        label : object\n\n        Notes\n        -----\n        Value of `side` parameter should be validated in caller.\n        "
        assert (kind in ['loc', 'getitem', None])
        if isinstance(label, str):
            freq = getattr(self, 'freqstr', getattr(self, 'inferred_freq', None))
            try:
                (parsed, reso) = parsing.parse_time_string(label, freq)
            except parsing.DateParseError as err:
                raise self._invalid_indexer('slice', label) from err
            reso = Resolution.from_attrname(reso)
            (lower, upper) = self._parsed_string_to_bounds(reso, parsed)
            if (self._is_strictly_monotonic_decreasing and (len(self) > 1)):
                return (upper if (side == 'left') else lower)
            return (lower if (side == 'left') else upper)
        elif isinstance(label, (self._data._recognized_scalars, date)):
            self._deprecate_mismatched_indexing(label)
        else:
            raise self._invalid_indexer('slice', label)
        return self._maybe_cast_for_get_loc(label)

    def _get_string_slice(self, key):
        freq = getattr(self, 'freqstr', getattr(self, 'inferred_freq', None))
        (parsed, reso) = parsing.parse_time_string(key, freq)
        reso = Resolution.from_attrname(reso)
        return self._partial_date_slice(reso, parsed)

    def slice_indexer(self, start=None, end=None, step=None, kind=None):
        '\n        Return indexer for specified label slice.\n        Index.slice_indexer, customized to handle time slicing.\n\n        In addition to functionality provided by Index.slice_indexer, does the\n        following:\n\n        - if both `start` and `end` are instances of `datetime.time`, it\n          invokes `indexer_between_time`\n        - if `start` and `end` are both either string or None perform\n          value-based selection in non-monotonic cases.\n\n        '
        if (isinstance(start, time) and isinstance(end, time)):
            if ((step is not None) and (step != 1)):
                raise ValueError('Must have step size of 1 with time slices')
            return self.indexer_between_time(start, end)
        if (isinstance(start, time) or isinstance(end, time)):
            raise KeyError('Cannot mix time and non-time slice keys')
        if (isinstance(start, date) and (not isinstance(start, datetime))):
            start = datetime.combine(start, time(0, 0))
        if (isinstance(end, date) and (not isinstance(end, datetime))):
            end = datetime.combine(end, time(0, 0))
        try:
            return Index.slice_indexer(self, start, end, step, kind=kind)
        except KeyError:
            if (((start is None) or isinstance(start, str)) and ((end is None) or isinstance(end, str))):
                mask = np.array(True)
                deprecation_mask = np.array(True)
                if (start is not None):
                    start_casted = self._maybe_cast_slice_bound(start, 'left', kind)
                    mask = (start_casted <= self)
                    deprecation_mask = (start_casted == self)
                if (end is not None):
                    end_casted = self._maybe_cast_slice_bound(end, 'right', kind)
                    mask = ((self <= end_casted) & mask)
                    deprecation_mask = ((end_casted == self) | deprecation_mask)
                if (not deprecation_mask.any()):
                    warnings.warn('Value based partial slicing on non-monotonic DatetimeIndexes with non-existing keys is deprecated and will raise a KeyError in a future Version.', FutureWarning, stacklevel=5)
                indexer = mask.nonzero()[0][::step]
                if (len(indexer) == len(self)):
                    return slice(None)
                else:
                    return indexer
            else:
                raise

    @property
    def inferred_type(self):
        return 'datetime64'

    def indexer_at_time(self, time, asof=False):
        '\n        Return index locations of values at particular time of day\n        (e.g. 9:30AM).\n\n        Parameters\n        ----------\n        time : datetime.time or str\n            Time passed in either as object (datetime.time) or as string in\n            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",\n            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p", "%I%M%S%p").\n\n        Returns\n        -------\n        values_at_time : array of integers\n\n        See Also\n        --------\n        indexer_between_time : Get index locations of values between particular\n            times of day.\n        DataFrame.at_time : Select values at particular time of day.\n        '
        if asof:
            raise NotImplementedError("'asof' argument is not supported")
        if isinstance(time, str):
            from dateutil.parser import parse
            time = parse(time).time()
        if time.tzinfo:
            if (self.tz is None):
                raise ValueError('Index must be timezone aware.')
            time_micros = self.tz_convert(time.tzinfo)._get_time_micros()
        else:
            time_micros = self._get_time_micros()
        micros = _time_to_micros(time)
        return (micros == time_micros).nonzero()[0]

    def indexer_between_time(self, start_time, end_time, include_start=True, include_end=True):
        '\n        Return index locations of values between particular times of day\n        (e.g., 9:00-9:30AM).\n\n        Parameters\n        ----------\n        start_time, end_time : datetime.time, str\n            Time passed either as object (datetime.time) or as string in\n            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",\n            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p","%I%M%S%p").\n        include_start : bool, default True\n        include_end : bool, default True\n\n        Returns\n        -------\n        values_between_time : array of integers\n\n        See Also\n        --------\n        indexer_at_time : Get index locations of values at particular time of day.\n        DataFrame.between_time : Select values between particular times of day.\n        '
        start_time = to_time(start_time)
        end_time = to_time(end_time)
        time_micros = self._get_time_micros()
        start_micros = _time_to_micros(start_time)
        end_micros = _time_to_micros(end_time)
        if (include_start and include_end):
            lop = rop = operator.le
        elif include_start:
            lop = operator.le
            rop = operator.lt
        elif include_end:
            lop = operator.lt
            rop = operator.le
        else:
            lop = rop = operator.lt
        if (start_time <= end_time):
            join_op = operator.and_
        else:
            join_op = operator.or_
        mask = join_op(lop(start_micros, time_micros), rop(time_micros, end_micros))
        return mask.nonzero()[0]

def date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs):
    "\n    Return a fixed frequency DatetimeIndex.\n\n    Returns the range of equally spaced time points (where the difference between any\n    two adjacent points is specified by the given frequency) such that they all\n    satisfy `start <[=] x <[=] end`, where the first one and the last one are, resp.,\n    the first and last time points in that range that fall on the boundary of ``freq``\n    (if given as a frequency string) or that are valid for ``freq`` (if given as a\n    :class:`pandas.tseries.offsets.DateOffset`). (If exactly one of ``start``,\n    ``end``, or ``freq`` is *not* specified, this missing parameter can be computed\n    given ``periods``, the number of timesteps in the range. See the note below.)\n\n    Parameters\n    ----------\n    start : str or datetime-like, optional\n        Left bound for generating dates.\n    end : str or datetime-like, optional\n        Right bound for generating dates.\n    periods : int, optional\n        Number of periods to generate.\n    freq : str or DateOffset, default 'D'\n        Frequency strings can have multiples, e.g. '5H'. See\n        :ref:`here <timeseries.offset_aliases>` for a list of\n        frequency aliases.\n    tz : str or tzinfo, optional\n        Time zone name for returning localized DatetimeIndex, for example\n        'Asia/Hong_Kong'. By default, the resulting DatetimeIndex is\n        timezone-naive.\n    normalize : bool, default False\n        Normalize start/end dates to midnight before generating date range.\n    name : str, default None\n        Name of the resulting DatetimeIndex.\n    closed : {None, 'left', 'right'}, optional\n        Make the interval closed with respect to the given frequency to\n        the 'left', 'right', or both sides (None, the default).\n    **kwargs\n        For compatibility. Has no effect on the result.\n\n    Returns\n    -------\n    rng : DatetimeIndex\n\n    See Also\n    --------\n    DatetimeIndex : An immutable container for datetimes.\n    timedelta_range : Return a fixed frequency TimedeltaIndex.\n    period_range : Return a fixed frequency PeriodIndex.\n    interval_range : Return a fixed frequency IntervalIndex.\n\n    Notes\n    -----\n    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,\n    exactly three must be specified. If ``freq`` is omitted, the resulting\n    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between\n    ``start`` and ``end`` (closed on both sides).\n\n    To learn more about the frequency strings, please see `this link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n\n    Examples\n    --------\n    **Specifying the values**\n\n    The next four examples generate the same `DatetimeIndex`, but vary\n    the combination of `start`, `end` and `periods`.\n\n    Specify `start` and `end`, with the default daily frequency.\n\n    >>> pd.date_range(start='1/1/2018', end='1/08/2018')\n    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',\n                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],\n                  dtype='datetime64[ns]', freq='D')\n\n    Specify `start` and `periods`, the number of periods (days).\n\n    >>> pd.date_range(start='1/1/2018', periods=8)\n    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',\n                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],\n                  dtype='datetime64[ns]', freq='D')\n\n    Specify `end` and `periods`, the number of periods (days).\n\n    >>> pd.date_range(end='1/1/2018', periods=8)\n    DatetimeIndex(['2017-12-25', '2017-12-26', '2017-12-27', '2017-12-28',\n                   '2017-12-29', '2017-12-30', '2017-12-31', '2018-01-01'],\n                  dtype='datetime64[ns]', freq='D')\n\n    Specify `start`, `end`, and `periods`; the frequency is generated\n    automatically (linearly spaced).\n\n    >>> pd.date_range(start='2018-04-24', end='2018-04-27', periods=3)\n    DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00',\n                   '2018-04-27 00:00:00'],\n                  dtype='datetime64[ns]', freq=None)\n\n    **Other Parameters**\n\n    Changed the `freq` (frequency) to ``'M'`` (month end frequency).\n\n    >>> pd.date_range(start='1/1/2018', periods=5, freq='M')\n    DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30',\n                   '2018-05-31'],\n                  dtype='datetime64[ns]', freq='M')\n\n    Multiples are allowed\n\n    >>> pd.date_range(start='1/1/2018', periods=5, freq='3M')\n    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',\n                   '2019-01-31'],\n                  dtype='datetime64[ns]', freq='3M')\n\n    `freq` can also be specified as an Offset object.\n\n    >>> pd.date_range(start='1/1/2018', periods=5, freq=pd.offsets.MonthEnd(3))\n    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',\n                   '2019-01-31'],\n                  dtype='datetime64[ns]', freq='3M')\n\n    Specify `tz` to set the timezone.\n\n    >>> pd.date_range(start='1/1/2018', periods=5, tz='Asia/Tokyo')\n    DatetimeIndex(['2018-01-01 00:00:00+09:00', '2018-01-02 00:00:00+09:00',\n                   '2018-01-03 00:00:00+09:00', '2018-01-04 00:00:00+09:00',\n                   '2018-01-05 00:00:00+09:00'],\n                  dtype='datetime64[ns, Asia/Tokyo]', freq='D')\n\n    `closed` controls whether to include `start` and `end` that are on the\n    boundary. The default includes boundary points on either end.\n\n    >>> pd.date_range(start='2017-01-01', end='2017-01-04', closed=None)\n    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04'],\n                  dtype='datetime64[ns]', freq='D')\n\n    Use ``closed='left'`` to exclude `end` if it falls on the boundary.\n\n    >>> pd.date_range(start='2017-01-01', end='2017-01-04', closed='left')\n    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03'],\n                  dtype='datetime64[ns]', freq='D')\n\n    Use ``closed='right'`` to exclude `start` if it falls on the boundary.\n\n    >>> pd.date_range(start='2017-01-01', end='2017-01-04', closed='right')\n    DatetimeIndex(['2017-01-02', '2017-01-03', '2017-01-04'],\n                  dtype='datetime64[ns]', freq='D')\n    "
    if ((freq is None) and com.any_none(periods, start, end)):
        freq = 'D'
    dtarr = DatetimeArray._generate_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize, closed=closed, **kwargs)
    return DatetimeIndex._simple_new(dtarr, name=name)

def bdate_range(start=None, end=None, periods=None, freq='B', tz=None, normalize=True, name=None, weekmask=None, holidays=None, closed=None, **kwargs):
    "\n    Return a fixed frequency DatetimeIndex, with business day as the default\n    frequency.\n\n    Parameters\n    ----------\n    start : str or datetime-like, default None\n        Left bound for generating dates.\n    end : str or datetime-like, default None\n        Right bound for generating dates.\n    periods : int, default None\n        Number of periods to generate.\n    freq : str or DateOffset, default 'B' (business daily)\n        Frequency strings can have multiples, e.g. '5H'.\n    tz : str or None\n        Time zone name for returning localized DatetimeIndex, for example\n        Asia/Beijing.\n    normalize : bool, default False\n        Normalize start/end dates to midnight before generating date range.\n    name : str, default None\n        Name of the resulting DatetimeIndex.\n    weekmask : str or None, default None\n        Weekmask of valid business days, passed to ``numpy.busdaycalendar``,\n        only used when custom frequency strings are passed.  The default\n        value None is equivalent to 'Mon Tue Wed Thu Fri'.\n    holidays : list-like or None, default None\n        Dates to exclude from the set of valid business days, passed to\n        ``numpy.busdaycalendar``, only used when custom frequency strings\n        are passed.\n    closed : str, default None\n        Make the interval closed with respect to the given frequency to\n        the 'left', 'right', or both sides (None).\n    **kwargs\n        For compatibility. Has no effect on the result.\n\n    Returns\n    -------\n    DatetimeIndex\n\n    Notes\n    -----\n    Of the four parameters: ``start``, ``end``, ``periods``, and ``freq``,\n    exactly three must be specified.  Specifying ``freq`` is a requirement\n    for ``bdate_range``.  Use ``date_range`` if specifying ``freq`` is not\n    desired.\n\n    To learn more about the frequency strings, please see `this link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n\n    Examples\n    --------\n    Note how the two weekend days are skipped in the result.\n\n    >>> pd.bdate_range(start='1/1/2018', end='1/08/2018')\n    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',\n               '2018-01-05', '2018-01-08'],\n              dtype='datetime64[ns]', freq='B')\n    "
    if (freq is None):
        msg = 'freq must be specified for bdate_range; use date_range instead'
        raise TypeError(msg)
    if (isinstance(freq, str) and freq.startswith('C')):
        try:
            weekmask = (weekmask or 'Mon Tue Wed Thu Fri')
            freq = prefix_mapping[freq](holidays=holidays, weekmask=weekmask)
        except (KeyError, TypeError) as err:
            msg = f'invalid custom frequency string: {freq}'
            raise ValueError(msg) from err
    elif (holidays or weekmask):
        msg = f'a custom frequency string is required when holidays or weekmask are passed, got frequency {freq}'
        raise ValueError(msg)
    return date_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize, name=name, closed=closed, **kwargs)

def _time_to_micros(time_obj):
    seconds = ((((time_obj.hour * 60) * 60) + (60 * time_obj.minute)) + time_obj.second)
    return ((1000000 * seconds) + time_obj.microsecond)
