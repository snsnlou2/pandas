
from datetime import datetime, timedelta
from typing import Any
import warnings
import numpy as np
from pandas._libs import index as libindex, lib
from pandas._libs.tslibs import BaseOffset, Period, Resolution, Tick
from pandas._libs.tslibs.parsing import DateParseError, parse_time_string
from pandas._typing import DtypeObj
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import is_bool_dtype, is_datetime64_any_dtype, is_float, is_integer, is_scalar, pandas_dtype
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.arrays.period import PeriodArray, period_array, raise_on_incompatible, validate_dtype_freq
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import ensure_index, maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex, Index
from pandas.core.indexes.extension import inherit_names
from pandas.core.indexes.numeric import Int64Index
from pandas.core.ops import get_op_result_name
_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({'target_klass': 'PeriodIndex or list of Periods'})
_shared_doc_kwargs = {'klass': 'PeriodArray'}

def _new_PeriodIndex(cls, **d):
    values = d.pop('data')
    if (values.dtype == 'int64'):
        freq = d.pop('freq', None)
        values = PeriodArray(values, freq=freq)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)

@inherit_names((['strftime', 'start_time', 'end_time'] + PeriodArray._field_ops), PeriodArray, wrap=True)
@inherit_names(['is_leap_year', '_format_native_types'], PeriodArray)
class PeriodIndex(DatetimeIndexOpsMixin):
    "\n    Immutable ndarray holding ordinal values indicating regular periods in time.\n\n    Index keys are boxed to Period objects which carries the metadata (eg,\n    frequency information).\n\n    Parameters\n    ----------\n    data : array-like (1d int np.ndarray or PeriodArray), optional\n        Optional period-like data to construct index with.\n    copy : bool\n        Make a copy of input ndarray.\n    freq : str or period object, optional\n        One of pandas period strings or corresponding objects.\n    year : int, array, or Series, default None\n    month : int, array, or Series, default None\n    quarter : int, array, or Series, default None\n    day : int, array, or Series, default None\n    hour : int, array, or Series, default None\n    minute : int, array, or Series, default None\n    second : int, array, or Series, default None\n    dtype : str or PeriodDtype, default None\n\n    Attributes\n    ----------\n    day\n    dayofweek\n    day_of_week\n    dayofyear\n    day_of_year\n    days_in_month\n    daysinmonth\n    end_time\n    freq\n    freqstr\n    hour\n    is_leap_year\n    minute\n    month\n    quarter\n    qyear\n    second\n    start_time\n    week\n    weekday\n    weekofyear\n    year\n\n    Methods\n    -------\n    asfreq\n    strftime\n    to_timestamp\n\n    See Also\n    --------\n    Index : The base pandas Index type.\n    Period : Represents a period of time.\n    DatetimeIndex : Index with datetime64 data.\n    TimedeltaIndex : Index of timedelta64 data.\n    period_range : Create a fixed-frequency PeriodIndex.\n\n    Examples\n    --------\n    >>> idx = pd.PeriodIndex(year=[2000, 2002], quarter=[1, 3])\n    >>> idx\n    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]', freq='Q-DEC')\n    "
    _typ = 'periodindex'
    _attributes = ['name', 'freq']
    _is_numeric_dtype = False
    _data_cls = PeriodArray
    _engine_type = libindex.PeriodEngine
    _supports_partial_string_indexing = True

    @doc(PeriodArray.asfreq, other='pandas.arrays.PeriodArray', other_name='PeriodArray', **_shared_doc_kwargs)
    def asfreq(self, freq=None, how='E'):
        arr = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)

    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq=None, how='start'):
        arr = self._data.to_timestamp(freq, how)
        return DatetimeIndex._simple_new(arr, name=self.name)

    @property
    @doc(PeriodArray.hour.fget)
    def hour(self):
        return Int64Index(self._data.hour, name=self.name)

    @property
    @doc(PeriodArray.minute.fget)
    def minute(self):
        return Int64Index(self._data.minute, name=self.name)

    @property
    @doc(PeriodArray.second.fget)
    def second(self):
        return Int64Index(self._data.second, name=self.name)

    def __new__(cls, data=None, ordinal=None, freq=None, dtype=None, copy=False, name=None, **fields):
        valid_field_set = {'year', 'month', 'day', 'quarter', 'hour', 'minute', 'second'}
        if (not set(fields).issubset(valid_field_set)):
            argument = list((set(fields) - valid_field_set))[0]
            raise TypeError(f'__new__() got an unexpected keyword argument {argument}')
        name = maybe_extract_name(name, data, cls)
        if ((data is None) and (ordinal is None)):
            (data, freq2) = PeriodArray._generate_range(None, None, None, freq, fields)
            freq = freq2
            data = PeriodArray(data, freq=freq)
        else:
            freq = validate_dtype_freq(dtype, freq)
            if (freq and isinstance(data, cls) and (data.freq != freq)):
                data = data.asfreq(freq)
            if ((data is None) and (ordinal is not None)):
                ordinal = np.asarray(ordinal, dtype=np.int64)
                data = PeriodArray(ordinal, freq=freq)
            else:
                data = period_array(data=data, freq=freq)
        if copy:
            data = data.copy()
        return cls._simple_new(data, name=name)

    @property
    def values(self):
        return np.asarray(self, dtype=object)

    def _maybe_convert_timedelta(self, other):
        '\n        Convert timedelta-like input to an integer multiple of self.freq\n\n        Parameters\n        ----------\n        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray\n\n        Returns\n        -------\n        converted : int, np.ndarray[int64]\n\n        Raises\n        ------\n        IncompatibleFrequency : if the input cannot be written as a multiple\n            of self.freq.  Note IncompatibleFrequency subclasses ValueError.\n        '
        if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            if isinstance(self.freq, Tick):
                delta = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, BaseOffset):
            if (other.base == self.freq.base):
                return other.n
            raise raise_on_incompatible(self, other)
        elif is_integer(other):
            return other
        raise raise_on_incompatible(self, None)

    def _is_comparable_dtype(self, dtype):
        '\n        Can we compare values of the given dtype to our own?\n        '
        if (not isinstance(dtype, PeriodDtype)):
            return False
        return (dtype.freq == self.freq)

    def _mpl_repr(self):
        return self.astype(object)._values

    @doc(Index.__contains__)
    def __contains__(self, key):
        if isinstance(key, Period):
            if (key.freq != self.freq):
                return False
            else:
                return (key.ordinal in self._engine)
        else:
            hash(key)
            try:
                self.get_loc(key)
                return True
            except KeyError:
                return False

    @cache_readonly
    def _int64index(self):
        return Int64Index._simple_new(self.asi8, name=self.name)

    def __array_wrap__(self, result, context=None):
        '\n        Gets called after a ufunc and other functions.\n\n        Needs additional handling as PeriodIndex stores internal data as int\n        dtype\n\n        Replace this to __numpy_ufunc__ in future version and implement\n        __array_function__ for Indexes\n        '
        if (isinstance(context, tuple) and (len(context) > 0)):
            func = context[0]
            if (func is np.add):
                pass
            elif (func is np.subtract):
                name = self.name
                left = context[1][0]
                right = context[1][1]
                if (isinstance(left, PeriodIndex) and isinstance(right, PeriodIndex)):
                    name = (left.name if (left.name == right.name) else None)
                    return Index(result, name=name)
                elif (isinstance(left, Period) or isinstance(right, Period)):
                    return Index(result, name=name)
            elif isinstance(func, np.ufunc):
                if ('M->M' not in func.types):
                    msg = f"ufunc '{func.__name__}' not supported for the PeriodIndex"
                    raise ValueError(msg)
        if is_bool_dtype(result):
            return result
        return type(self)(result, freq=self.freq, name=self.name)

    def asof_locs(self, where, mask):
        '\n        where : array of timestamps\n        mask : array of booleans where data is not NA\n        '
        if isinstance(where, DatetimeIndex):
            where = PeriodIndex(where._values, freq=self.freq)
        elif (not isinstance(where, PeriodIndex)):
            raise TypeError('asof_locs `where` must be DatetimeIndex or PeriodIndex')
        return super().asof_locs(where, mask)

    @doc(Index.astype)
    def astype(self, dtype, copy=True, how=lib.no_default):
        dtype = pandas_dtype(dtype)
        if (how is not lib.no_default):
            warnings.warn("The 'how' keyword in PeriodIndex.astype is deprecated and will be removed in a future version. Use index.to_timestamp(how=how) instead", FutureWarning, stacklevel=2)
        else:
            how = 'start'
        if is_datetime64_any_dtype(dtype):
            tz = getattr(dtype, 'tz', None)
            return self.to_timestamp(how=how).tz_localize(tz)
        return super().astype(dtype, copy=copy)

    @property
    def is_full(self):
        '\n        Returns True if this PeriodIndex is range-like in that all Periods\n        between start and end are present, in order.\n        '
        if (len(self) == 0):
            return True
        if (not self.is_monotonic_increasing):
            raise ValueError('Index is not monotonic')
        values = self.asi8
        return ((values[1:] - values[:(- 1)]) < 2).all()

    @property
    def inferred_type(self):
        return 'period'

    def insert(self, loc, item):
        if ((not isinstance(item, Period)) or (self.freq != item.freq)):
            return self.astype(object).insert(loc, item)
        return DatetimeIndexOpsMixin.insert(self, loc, item)

    def join(self, other, how='left', level=None, return_indexers=False, sort=False):
        '\n        See Index.join\n        '
        self._assert_can_do_setop(other)
        if (not isinstance(other, PeriodIndex)):
            return self.astype(object).join(other, how=how, level=level, return_indexers=return_indexers, sort=sort)
        result = super().join(other, how=how, level=level, return_indexers=return_indexers, sort=sort)
        return result

    def _get_indexer(self, target, method=None, limit=None, tolerance=None):
        if (not self._should_compare(target)):
            return self._get_indexer_non_comparable(target, method, unique=True)
        if isinstance(target, PeriodIndex):
            target = target._int64index
            self_index = self._int64index
        else:
            self_index = self
        if (tolerance is not None):
            tolerance = self._convert_tolerance(tolerance, target)
            if (self_index is not self):
                tolerance = self._maybe_convert_timedelta(tolerance)
        return Index._get_indexer(self_index, target, method, limit, tolerance)

    def get_loc(self, key, method=None, tolerance=None):
        '\n        Get integer location for requested label.\n\n        Parameters\n        ----------\n        key : Period, NaT, str, or datetime\n            String or datetime key must be parsable as Period.\n\n        Returns\n        -------\n        loc : int or ndarray[int64]\n\n        Raises\n        ------\n        KeyError\n            Key is not present in the index.\n        TypeError\n            If key is listlike or otherwise not hashable.\n        '
        orig_key = key
        if (not is_scalar(key)):
            raise InvalidIndexError(key)
        if isinstance(key, str):
            try:
                loc = self._get_string_slice(key)
                return loc
            except (TypeError, ValueError):
                pass
            try:
                (asdt, reso) = parse_time_string(key, self.freq)
            except (ValueError, DateParseError) as err:
                raise KeyError(f"Cannot interpret '{key}' as period") from err
            reso = Resolution.from_attrname(reso)
            grp = reso.freq_group
            freqn = self.dtype.freq_group
            assert (grp >= freqn)
            if ((grp == freqn) or ((reso == Resolution.RESO_DAY) and (self.dtype.freq.name == 'B'))):
                key = Period(asdt, freq=self.freq)
                loc = self.get_loc(key, method=method, tolerance=tolerance)
                return loc
            elif (method is None):
                raise KeyError(key)
            else:
                key = asdt
        elif is_integer(key):
            raise KeyError(key)
        try:
            key = Period(key, freq=self.freq)
        except ValueError as err:
            raise KeyError(orig_key) from err
        try:
            return Index.get_loc(self, key, method, tolerance)
        except KeyError as err:
            raise KeyError(orig_key) from err

    def _maybe_cast_slice_bound(self, label, side, kind):
        "\n        If label is a string or a datetime, cast it to Period.ordinal according\n        to resolution.\n\n        Parameters\n        ----------\n        label : object\n        side : {'left', 'right'}\n        kind : {'loc', 'getitem'}\n\n        Returns\n        -------\n        bound : Period or object\n\n        Notes\n        -----\n        Value of `side` parameter should be validated in caller.\n\n        "
        assert (kind in ['loc', 'getitem'])
        if isinstance(label, datetime):
            return Period(label, freq=self.freq)
        elif isinstance(label, str):
            try:
                (parsed, reso) = parse_time_string(label, self.freq)
                reso = Resolution.from_attrname(reso)
                bounds = self._parsed_string_to_bounds(reso, parsed)
                return bounds[(0 if (side == 'left') else 1)]
            except ValueError as err:
                raise self._invalid_indexer('slice', label) from err
        elif (is_integer(label) or is_float(label)):
            raise self._invalid_indexer('slice', label)
        return label

    def _parsed_string_to_bounds(self, reso, parsed):
        grp = reso.freq_group
        iv = Period(parsed, freq=grp)
        return (iv.asfreq(self.freq, how='start'), iv.asfreq(self.freq, how='end'))

    def _validate_partial_date_slice(self, reso):
        assert isinstance(reso, Resolution), (type(reso), reso)
        grp = reso.freq_group
        freqn = self.dtype.freq_group
        if (not (grp < freqn)):
            raise ValueError

    def _get_string_slice(self, key):
        (parsed, reso) = parse_time_string(key, self.freq)
        reso = Resolution.from_attrname(reso)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    def _assert_can_do_setop(self, other):
        super()._assert_can_do_setop(other)
        if (isinstance(other, PeriodIndex) and (self.freq != other.freq)):
            raise raise_on_incompatible(self, other)

    def _setop(self, other, sort, opname):
        '\n        Perform a set operation by dispatching to the Int64Index implementation.\n        '
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        res_name = get_op_result_name(self, other)
        other = ensure_index(other)
        i8self = Int64Index._simple_new(self.asi8)
        i8other = Int64Index._simple_new(other.asi8)
        i8result = getattr(i8self, opname)(i8other, sort=sort)
        parr = type(self._data)(np.asarray(i8result, dtype=np.int64), dtype=self.dtype)
        result = type(self)._simple_new(parr, name=res_name)
        return result

    def _intersection(self, other, sort=False):
        return self._setop(other, sort, opname='intersection')

    def _union(self, other, sort):
        return self._setop(other, sort, opname='_union')

    def memory_usage(self, deep=False):
        result = super().memory_usage(deep=deep)
        if (hasattr(self, '_cache') and ('_int64index' in self._cache)):
            result += self._int64index.memory_usage(deep=deep)
        return result

def period_range(start=None, end=None, periods=None, freq=None, name=None):
    '\n    Return a fixed frequency PeriodIndex.\n\n    The day (calendar) is the default frequency.\n\n    Parameters\n    ----------\n    start : str or period-like, default None\n        Left bound for generating periods.\n    end : str or period-like, default None\n        Right bound for generating periods.\n    periods : int, default None\n        Number of periods to generate.\n    freq : str or DateOffset, optional\n        Frequency alias. By default the freq is taken from `start` or `end`\n        if those are Period objects. Otherwise, the default is ``"D"`` for\n        daily frequency.\n    name : str, default None\n        Name of the resulting PeriodIndex.\n\n    Returns\n    -------\n    PeriodIndex\n\n    Notes\n    -----\n    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two\n    must be specified.\n\n    To learn more about the frequency strings, please see `this link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n\n    Examples\n    --------\n    >>> pd.period_range(start=\'2017-01-01\', end=\'2018-01-01\', freq=\'M\')\n    PeriodIndex([\'2017-01\', \'2017-02\', \'2017-03\', \'2017-04\', \'2017-05\', \'2017-06\',\n             \'2017-07\', \'2017-08\', \'2017-09\', \'2017-10\', \'2017-11\', \'2017-12\',\n             \'2018-01\'],\n            dtype=\'period[M]\', freq=\'M\')\n\n    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor\n    endpoints for a ``PeriodIndex`` with frequency matching that of the\n    ``period_range`` constructor.\n\n    >>> pd.period_range(start=pd.Period(\'2017Q1\', freq=\'Q\'),\n    ...                 end=pd.Period(\'2017Q2\', freq=\'Q\'), freq=\'M\')\n    PeriodIndex([\'2017-03\', \'2017-04\', \'2017-05\', \'2017-06\'],\n                dtype=\'period[M]\', freq=\'M\')\n    '
    if (com.count_not_none(start, end, periods) != 2):
        raise ValueError('Of the three parameters: start, end, and periods, exactly two must be specified')
    if ((freq is None) and ((not isinstance(start, Period)) and (not isinstance(end, Period)))):
        freq = 'D'
    (data, freq) = PeriodArray._generate_range(start, end, periods, freq, fields={})
    data = PeriodArray(data, freq=freq)
    return PeriodIndex(data, name=name)
