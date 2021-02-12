
'\ndatetimelike delegation\n'
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas.core.dtypes.common import is_categorical_dtype, is_datetime64_dtype, is_datetime64tz_dtype, is_integer_dtype, is_list_like, is_period_dtype, is_timedelta64_dtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.accessor import PandasDelegate, delegate_names
from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray
from pandas.core.base import NoNewAttributesMixin, PandasObject
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
if TYPE_CHECKING:
    from pandas import Series

class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _hidden_attrs = (PandasObject._hidden_attrs | {'orig', 'name'})

    def __init__(self, data, orig):
        if (not isinstance(data, ABCSeries)):
            raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')
        self._parent = data
        self.orig = orig
        self.name = getattr(data, 'name', None)
        self._freeze()

    def _get_values(self):
        data = self._parent
        if is_datetime64_dtype(data.dtype):
            return DatetimeIndex(data, copy=False, name=self.name)
        elif is_datetime64tz_dtype(data.dtype):
            return DatetimeIndex(data, copy=False, name=self.name)
        elif is_timedelta64_dtype(data.dtype):
            return TimedeltaIndex(data, copy=False, name=self.name)
        elif is_period_dtype(data.dtype):
            return PeriodArray(data, copy=False)
        raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')

    def _delegate_property_get(self, name):
        from pandas import Series
        values = self._get_values()
        result = getattr(values, name)
        if isinstance(result, np.ndarray):
            if is_integer_dtype(result):
                result = result.astype('int64')
        elif (not is_list_like(result)):
            return result
        result = np.asarray(result)
        if (self.orig is not None):
            index = self.orig.index
        else:
            index = self._parent.index
        result = Series(result, index=index, name=self.name).__finalize__(self._parent)
        result._is_copy = 'modifications to a property of a datetimelike object are not supported and are discarded. Change values on the original.'
        return result

    def _delegate_property_set(self, name, value, *args, **kwargs):
        raise ValueError('modifications to a property of a datetimelike object are not supported. Change values on the original.')

    def _delegate_method(self, name, *args, **kwargs):
        from pandas import Series
        values = self._get_values()
        method = getattr(values, name)
        result = method(*args, **kwargs)
        if (not is_list_like(result)):
            return result
        result = Series(result, index=self._parent.index, name=self.name).__finalize__(self._parent)
        result._is_copy = 'modifications to a method of a datetimelike object are not supported and are discarded. Change values on the original.'
        return result

@delegate_names(delegate=DatetimeArray, accessors=DatetimeArray._datetimelike_ops, typ='property')
@delegate_names(delegate=DatetimeArray, accessors=DatetimeArray._datetimelike_methods, typ='method')
class DatetimeProperties(Properties):
    '\n    Accessor object for datetimelike properties of the Series values.\n\n    Examples\n    --------\n    >>> seconds_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="s"))\n    >>> seconds_series\n    0   2000-01-01 00:00:00\n    1   2000-01-01 00:00:01\n    2   2000-01-01 00:00:02\n    dtype: datetime64[ns]\n    >>> seconds_series.dt.second\n    0    0\n    1    1\n    2    2\n    dtype: int64\n\n    >>> hours_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="h"))\n    >>> hours_series\n    0   2000-01-01 00:00:00\n    1   2000-01-01 01:00:00\n    2   2000-01-01 02:00:00\n    dtype: datetime64[ns]\n    >>> hours_series.dt.hour\n    0    0\n    1    1\n    2    2\n    dtype: int64\n\n    >>> quarters_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="q"))\n    >>> quarters_series\n    0   2000-03-31\n    1   2000-06-30\n    2   2000-09-30\n    dtype: datetime64[ns]\n    >>> quarters_series.dt.quarter\n    0    1\n    1    2\n    2    3\n    dtype: int64\n\n    Returns a Series indexed like the original Series.\n    Raises TypeError if the Series does not contain datetimelike values.\n    '

    def to_pydatetime(self):
        "\n        Return the data as an array of native Python datetime objects.\n\n        Timezone information is retained if present.\n\n        .. warning::\n\n           Python's datetime uses microsecond resolution, which is lower than\n           pandas (nanosecond). The values are truncated.\n\n        Returns\n        -------\n        numpy.ndarray\n            Object dtype array containing native Python datetime objects.\n\n        See Also\n        --------\n        datetime.datetime : Standard library value for a datetime.\n\n        Examples\n        --------\n        >>> s = pd.Series(pd.date_range('20180310', periods=2))\n        >>> s\n        0   2018-03-10\n        1   2018-03-11\n        dtype: datetime64[ns]\n\n        >>> s.dt.to_pydatetime()\n        array([datetime.datetime(2018, 3, 10, 0, 0),\n               datetime.datetime(2018, 3, 11, 0, 0)], dtype=object)\n\n        pandas' nanosecond precision is truncated to microseconds.\n\n        >>> s = pd.Series(pd.date_range('20180310', periods=2, freq='ns'))\n        >>> s\n        0   2018-03-10 00:00:00.000000000\n        1   2018-03-10 00:00:00.000000001\n        dtype: datetime64[ns]\n\n        >>> s.dt.to_pydatetime()\n        array([datetime.datetime(2018, 3, 10, 0, 0),\n               datetime.datetime(2018, 3, 10, 0, 0)], dtype=object)\n        "
        return self._get_values().to_pydatetime()

    @property
    def freq(self):
        return self._get_values().inferred_freq

    def isocalendar(self):
        '\n        Returns a DataFrame with the year, week, and day calculated according to\n        the ISO 8601 standard.\n\n        .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        DataFrame\n            with columns year, week and day\n\n        See Also\n        --------\n        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,\n            week number, and weekday for the given Timestamp object.\n        datetime.date.isocalendar : Return a named tuple object with\n            three components: year, week and weekday.\n\n        Examples\n        --------\n        >>> ser = pd.to_datetime(pd.Series(["2010-01-01", pd.NaT]))\n        >>> ser.dt.isocalendar()\n           year  week  day\n        0  2009    53     5\n        1  <NA>  <NA>  <NA>\n        >>> ser.dt.isocalendar().week\n        0      53\n        1    <NA>\n        Name: week, dtype: UInt32\n        '
        return self._get_values().isocalendar().set_index(self._parent.index)

    @property
    def weekofyear(self):
        '\n        The week ordinal of the year.\n\n        .. deprecated:: 1.1.0\n\n        Series.dt.weekofyear and Series.dt.week have been deprecated.\n        Please use Series.dt.isocalendar().week instead.\n        '
        warnings.warn('Series.dt.weekofyear and Series.dt.week have been deprecated.  Please use Series.dt.isocalendar().week instead.', FutureWarning, stacklevel=2)
        week_series = self.isocalendar().week
        week_series.name = self.name
        if week_series.hasnans:
            return week_series.astype('float64')
        return week_series.astype('int64')
    week = weekofyear

@delegate_names(delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ='property')
@delegate_names(delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_methods, typ='method')
class TimedeltaProperties(Properties):
    '\n    Accessor object for datetimelike properties of the Series values.\n\n    Returns a Series indexed like the original Series.\n    Raises TypeError if the Series does not contain datetimelike values.\n\n    Examples\n    --------\n    >>> seconds_series = pd.Series(\n    ...     pd.timedelta_range(start="1 second", periods=3, freq="S")\n    ... )\n    >>> seconds_series\n    0   0 days 00:00:01\n    1   0 days 00:00:02\n    2   0 days 00:00:03\n    dtype: timedelta64[ns]\n    >>> seconds_series.dt.seconds\n    0    1\n    1    2\n    2    3\n    dtype: int64\n    '

    def to_pytimedelta(self):
        '\n        Return an array of native `datetime.timedelta` objects.\n\n        Python\'s standard `datetime` library uses a different representation\n        timedelta\'s. This method converts a Series of pandas Timedeltas\n        to `datetime.timedelta` format with the same length as the original\n        Series.\n\n        Returns\n        -------\n        numpy.ndarray\n            Array of 1D containing data with `datetime.timedelta` type.\n\n        See Also\n        --------\n        datetime.timedelta : A duration expressing the difference\n            between two date, time, or datetime.\n\n        Examples\n        --------\n        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="d"))\n        >>> s\n        0   0 days\n        1   1 days\n        2   2 days\n        3   3 days\n        4   4 days\n        dtype: timedelta64[ns]\n\n        >>> s.dt.to_pytimedelta()\n        array([datetime.timedelta(0), datetime.timedelta(days=1),\n        datetime.timedelta(days=2), datetime.timedelta(days=3),\n        datetime.timedelta(days=4)], dtype=object)\n        '
        return self._get_values().to_pytimedelta()

    @property
    def components(self):
        "\n        Return a Dataframe of the components of the Timedeltas.\n\n        Returns\n        -------\n        DataFrame\n\n        Examples\n        --------\n        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='s'))\n        >>> s\n        0   0 days 00:00:00\n        1   0 days 00:00:01\n        2   0 days 00:00:02\n        3   0 days 00:00:03\n        4   0 days 00:00:04\n        dtype: timedelta64[ns]\n        >>> s.dt.components\n           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds\n        0     0      0        0        0             0             0            0\n        1     0      0        0        1             0             0            0\n        2     0      0        0        2             0             0            0\n        3     0      0        0        3             0             0            0\n        4     0      0        0        4             0             0            0\n        "
        return self._get_values().components.set_index(self._parent.index).__finalize__(self._parent)

    @property
    def freq(self):
        return self._get_values().inferred_freq

@delegate_names(delegate=PeriodArray, accessors=PeriodArray._datetimelike_ops, typ='property')
@delegate_names(delegate=PeriodArray, accessors=PeriodArray._datetimelike_methods, typ='method')
class PeriodProperties(Properties):
    '\n    Accessor object for datetimelike properties of the Series values.\n\n    Returns a Series indexed like the original Series.\n    Raises TypeError if the Series does not contain datetimelike values.\n\n    Examples\n    --------\n    >>> seconds_series = pd.Series(\n    ...     pd.period_range(\n    ...         start="2000-01-01 00:00:00", end="2000-01-01 00:00:03", freq="s"\n    ...     )\n    ... )\n    >>> seconds_series\n    0    2000-01-01 00:00:00\n    1    2000-01-01 00:00:01\n    2    2000-01-01 00:00:02\n    3    2000-01-01 00:00:03\n    dtype: period[S]\n    >>> seconds_series.dt.second\n    0    0\n    1    1\n    2    2\n    3    3\n    dtype: int64\n\n    >>> hours_series = pd.Series(\n    ...     pd.period_range(start="2000-01-01 00:00", end="2000-01-01 03:00", freq="h")\n    ... )\n    >>> hours_series\n    0    2000-01-01 00:00\n    1    2000-01-01 01:00\n    2    2000-01-01 02:00\n    3    2000-01-01 03:00\n    dtype: period[H]\n    >>> hours_series.dt.hour\n    0    0\n    1    1\n    2    2\n    3    3\n    dtype: int64\n\n    >>> quarters_series = pd.Series(\n    ...     pd.period_range(start="2000-01-01", end="2000-12-31", freq="Q-DEC")\n    ... )\n    >>> quarters_series\n    0    2000Q1\n    1    2000Q2\n    2    2000Q3\n    3    2000Q4\n    dtype: period[Q-DEC]\n    >>> quarters_series.dt.quarter\n    0    1\n    1    2\n    2    3\n    3    4\n    dtype: int64\n    '

class CombinedDatetimelikeProperties(DatetimeProperties, TimedeltaProperties, PeriodProperties):

    def __new__(cls, data):
        if (not isinstance(data, ABCSeries)):
            raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')
        orig = (data if is_categorical_dtype(data.dtype) else None)
        if (orig is not None):
            data = data._constructor(orig.array, name=orig.name, copy=False, dtype=orig._values.categories.dtype)
        if is_datetime64_dtype(data.dtype):
            return DatetimeProperties(data, orig)
        elif is_datetime64tz_dtype(data.dtype):
            return DatetimeProperties(data, orig)
        elif is_timedelta64_dtype(data.dtype):
            return TimedeltaProperties(data, orig)
        elif is_period_dtype(data.dtype):
            return PeriodProperties(data, orig)
        raise AttributeError('Can only use .dt accessor with datetimelike values')
