
from collections import abc
from datetime import datetime
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, TypeVar, Union, overload
import warnings
import numpy as np
from pandas._libs import tslib
from pandas._libs.tslibs import OutOfBoundsDatetime, Timedelta, Timestamp, conversion, iNaT, nat_strings, parsing
from pandas._libs.tslibs.parsing import DateParseError, format_is_iso, guess_datetime_format
from pandas._libs.tslibs.strptime import array_strptime
from pandas._typing import AnyArrayLike, ArrayLike, Label, Timezone
from pandas.core.dtypes.common import ensure_object, is_datetime64_dtype, is_datetime64_ns_dtype, is_datetime64tz_dtype, is_float, is_integer, is_integer_dtype, is_list_like, is_numeric_dtype, is_scalar
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import notna
from pandas.arrays import DatetimeArray, IntegerArray
from pandas.core import algorithms
from pandas.core.algorithms import unique
from pandas.core.arrays.datetimes import maybe_convert_dtype, objects_to_datetime64ns, tz_to_dtype
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
if TYPE_CHECKING:
    from pandas._libs.tslibs.nattype import NaTType
    from pandas import Series
ArrayConvertible = Union[(List, Tuple, AnyArrayLike, 'Series')]
Scalar = Union[(int, float, str)]
DatetimeScalar = TypeVar('DatetimeScalar', Scalar, datetime)
DatetimeScalarOrArrayConvertible = Union[(DatetimeScalar, ArrayConvertible)]

def _guess_datetime_format_for_array(arr, **kwargs):
    non_nan_elements = notna(arr).nonzero()[0]
    if len(non_nan_elements):
        return guess_datetime_format(arr[non_nan_elements[0]], **kwargs)

def should_cache(arg, unique_share=0.7, check_count=None):
    "\n    Decides whether to do caching.\n\n    If the percent of unique elements among `check_count` elements less\n    than `unique_share * 100` then we can do caching.\n\n    Parameters\n    ----------\n    arg: listlike, tuple, 1-d array, Series\n    unique_share: float, default=0.7, optional\n        0 < unique_share < 1\n    check_count: int, optional\n        0 <= check_count <= len(arg)\n\n    Returns\n    -------\n    do_caching: bool\n\n    Notes\n    -----\n    By default for a sequence of less than 50 items in size, we don't do\n    caching; for the number of elements less than 5000, we take ten percent of\n    all elements to check for a uniqueness share; if the sequence size is more\n    than 5000, then we check only the first 500 elements.\n    All constants were chosen empirically by.\n    "
    do_caching = True
    if (check_count is None):
        if (len(arg) <= 50):
            return False
        if (len(arg) <= 5000):
            check_count = int((len(arg) * 0.1))
        else:
            check_count = 500
    else:
        assert (0 <= check_count <= len(arg)), 'check_count must be in next bounds: [0; len(arg)]'
        if (check_count == 0):
            return False
    assert (0 < unique_share < 1), 'unique_share must be in next bounds: (0; 1)'
    unique_elements = set(islice(arg, check_count))
    if (len(unique_elements) > (check_count * unique_share)):
        do_caching = False
    return do_caching

def _maybe_cache(arg, format, cache, convert_listlike):
    '\n    Create a cache of unique dates from an array of dates\n\n    Parameters\n    ----------\n    arg : listlike, tuple, 1-d array, Series\n    format : string\n        Strftime format to parse time\n    cache : boolean\n        True attempts to create a cache of converted values\n    convert_listlike : function\n        Conversion function to apply on dates\n\n    Returns\n    -------\n    cache_array : Series\n        Cache of converted, unique dates. Can be empty\n    '
    from pandas import Series
    cache_array = Series(dtype=object)
    if cache:
        if (not should_cache(arg)):
            return cache_array
        unique_dates = unique(arg)
        if (len(unique_dates) < len(arg)):
            cache_dates = convert_listlike(unique_dates, format)
            cache_array = Series(cache_dates, index=unique_dates)
    return cache_array

def _box_as_indexlike(dt_array, utc=None, name=None):
    "\n    Properly boxes the ndarray of datetimes to DatetimeIndex\n    if it is possible or to generic Index instead\n\n    Parameters\n    ----------\n    dt_array: 1-d array\n        Array of datetimes to be wrapped in an Index.\n    tz : object\n        None or 'utc'\n    name : string, default None\n        Name for a resulting index\n\n    Returns\n    -------\n    result : datetime of converted dates\n        - DatetimeIndex if convertible to sole datetime64 type\n        - general Index otherwise\n    "
    if is_datetime64_dtype(dt_array):
        tz = ('utc' if utc else None)
        return DatetimeIndex(dt_array, tz=tz, name=name)
    return Index(dt_array, name=name)

def _convert_and_box_cache(arg, cache_array, name=None):
    '\n    Convert array of dates with a cache and wrap the result in an Index.\n\n    Parameters\n    ----------\n    arg : integer, float, string, datetime, list, tuple, 1-d array, Series\n    cache_array : Series\n        Cache of converted, unique dates\n    name : string, default None\n        Name for a DatetimeIndex\n\n    Returns\n    -------\n    result : Index-like of converted dates\n    '
    from pandas import Series
    result = Series(arg).map(cache_array)
    return _box_as_indexlike(result, utc=None, name=name)

def _return_parsed_timezone_results(result, timezones, tz, name):
    '\n    Return results from array_strptime if a %z or %Z directive was passed.\n\n    Parameters\n    ----------\n    result : ndarray\n        int64 date representations of the dates\n    timezones : ndarray\n        pytz timezone objects\n    tz : object\n        None or pytz timezone object\n    name : string, default None\n        Name for a DatetimeIndex\n\n    Returns\n    -------\n    tz_result : Index-like of parsed dates with timezone\n    '
    tz_results = np.array([Timestamp(res).tz_localize(zone) for (res, zone) in zip(result, timezones)])
    if (tz is not None):
        tz_results = np.array([tz_result.tz_convert(tz) for tz_result in tz_results])
    return Index(tz_results, name=name)

def _convert_listlike_datetimes(arg, format, name=None, tz=None, unit=None, errors=None, infer_datetime_format=None, dayfirst=None, yearfirst=None, exact=None):
    "\n    Helper function for to_datetime. Performs the conversions of 1D listlike\n    of dates\n\n    Parameters\n    ----------\n    arg : list, tuple, ndarray, Series, Index\n        date to be parsed\n    name : object\n        None or string for the Index name\n    tz : object\n        None or 'utc'\n    unit : string\n        None or string of the frequency of the passed data\n    errors : string\n        error handing behaviors from to_datetime, 'raise', 'coerce', 'ignore'\n    infer_datetime_format : boolean\n        inferring format behavior from to_datetime\n    dayfirst : boolean\n        dayfirst parsing behavior from to_datetime\n    yearfirst : boolean\n        yearfirst parsing behavior from to_datetime\n    exact : boolean\n        exact format matching behavior from to_datetime\n\n    Returns\n    -------\n    Index-like of parsed dates\n    "
    if isinstance(arg, (list, tuple)):
        arg = np.array(arg, dtype='O')
    arg_dtype = getattr(arg, 'dtype', None)
    if is_datetime64tz_dtype(arg_dtype):
        if (not isinstance(arg, (DatetimeArray, DatetimeIndex))):
            return DatetimeIndex(arg, tz=tz, name=name)
        if (tz == 'utc'):
            arg = arg.tz_convert(None).tz_localize(tz)
        return arg
    elif is_datetime64_ns_dtype(arg_dtype):
        if (not isinstance(arg, (DatetimeArray, DatetimeIndex))):
            try:
                return DatetimeIndex(arg, tz=tz, name=name)
            except ValueError:
                pass
        elif tz:
            return arg.tz_localize(tz)
        return arg
    elif (unit is not None):
        if (format is not None):
            raise ValueError('cannot specify both format and unit')
        arg = getattr(arg, '_values', arg)
        if isinstance(arg, IntegerArray):
            result = arg.astype(f'datetime64[{unit}]')
            tz_parsed = None
        else:
            (result, tz_parsed) = tslib.array_with_unit_to_datetime(arg, unit, errors=errors)
        if (errors == 'ignore'):
            result = Index(result, name=name)
        else:
            result = DatetimeIndex(result, name=name)
        try:
            result = result.tz_localize('UTC').tz_convert(tz_parsed)
        except AttributeError:
            return result
        if (tz is not None):
            if (result.tz is None):
                result = result.tz_localize(tz)
            else:
                result = result.tz_convert(tz)
        return result
    elif (getattr(arg, 'ndim', 1) > 1):
        raise TypeError('arg must be a string, datetime, list, tuple, 1-d array, or Series')
    orig_arg = arg
    try:
        (arg, _) = maybe_convert_dtype(arg, copy=False)
    except TypeError:
        if (errors == 'coerce'):
            result = np.array(['NaT'], dtype='datetime64[ns]').repeat(len(arg))
            return DatetimeIndex(result, name=name)
        elif (errors == 'ignore'):
            result = Index(arg, name=name)
            return result
        raise
    arg = ensure_object(arg)
    require_iso8601 = False
    if (infer_datetime_format and (format is None)):
        format = _guess_datetime_format_for_array(arg, dayfirst=dayfirst)
    if (format is not None):
        format_is_iso8601 = format_is_iso(format)
        if format_is_iso8601:
            require_iso8601 = (not infer_datetime_format)
            format = None
    tz_parsed = None
    result = None
    if (format is not None):
        try:
            if (format == '%Y%m%d'):
                try:
                    orig_arg = ensure_object(orig_arg)
                    result = _attempt_YYYYMMDD(orig_arg, errors=errors)
                except (ValueError, TypeError, OutOfBoundsDatetime) as err:
                    raise ValueError("cannot convert the input to '%Y%m%d' date format") from err
            if (result is None):
                try:
                    (result, timezones) = array_strptime(arg, format, exact=exact, errors=errors)
                    if (('%Z' in format) or ('%z' in format)):
                        return _return_parsed_timezone_results(result, timezones, tz, name)
                except OutOfBoundsDatetime:
                    if (errors == 'raise'):
                        raise
                    elif (errors == 'coerce'):
                        result = np.empty(arg.shape, dtype='M8[ns]')
                        iresult = result.view('i8')
                        iresult.fill(iNaT)
                    else:
                        result = arg
                except ValueError:
                    if (not infer_datetime_format):
                        if (errors == 'raise'):
                            raise
                        elif (errors == 'coerce'):
                            result = np.empty(arg.shape, dtype='M8[ns]')
                            iresult = result.view('i8')
                            iresult.fill(iNaT)
                        else:
                            result = arg
        except ValueError as e:
            try:
                (values, tz) = conversion.datetime_to_datetime64(arg)
                dta = DatetimeArray(values, dtype=tz_to_dtype(tz))
                return DatetimeIndex._simple_new(dta, name=name)
            except (ValueError, TypeError):
                raise e
    if (result is None):
        assert ((format is None) or infer_datetime_format)
        utc = (tz == 'utc')
        (result, tz_parsed) = objects_to_datetime64ns(arg, dayfirst=dayfirst, yearfirst=yearfirst, utc=utc, errors=errors, require_iso8601=require_iso8601, allow_object=True)
    if (tz_parsed is not None):
        dta = DatetimeArray(result, dtype=tz_to_dtype(tz_parsed))
        return DatetimeIndex._simple_new(dta, name=name)
    utc = (tz == 'utc')
    return _box_as_indexlike(result, utc=utc, name=name)

def _adjust_to_origin(arg, origin, unit):
    "\n    Helper function for to_datetime.\n    Adjust input argument to the specified origin\n\n    Parameters\n    ----------\n    arg : list, tuple, ndarray, Series, Index\n        date to be adjusted\n    origin : 'julian' or Timestamp\n        origin offset for the arg\n    unit : string\n        passed unit from to_datetime, must be 'D'\n\n    Returns\n    -------\n    ndarray or scalar of adjusted date(s)\n    "
    if (origin == 'julian'):
        original = arg
        j0 = Timestamp(0).to_julian_date()
        if (unit != 'D'):
            raise ValueError("unit must be 'D' for origin='julian'")
        try:
            arg = (arg - j0)
        except TypeError as err:
            raise ValueError("incompatible 'arg' type for given 'origin'='julian'") from err
        j_max = (Timestamp.max.to_julian_date() - j0)
        j_min = (Timestamp.min.to_julian_date() - j0)
        if (np.any((arg > j_max)) or np.any((arg < j_min))):
            raise OutOfBoundsDatetime(f"{original} is Out of Bounds for origin='julian'")
    else:
        if (not ((is_scalar(arg) and (is_integer(arg) or is_float(arg))) or is_numeric_dtype(np.asarray(arg)))):
            raise ValueError(f"'{arg}' is not compatible with origin='{origin}'; it must be numeric with a unit specified")
        try:
            offset = Timestamp(origin)
        except OutOfBoundsDatetime as err:
            raise OutOfBoundsDatetime(f'origin {origin} is Out of Bounds') from err
        except ValueError as err:
            raise ValueError(f'origin {origin} cannot be converted to a Timestamp') from err
        if (offset.tz is not None):
            raise ValueError(f'origin offset {offset} must be tz-naive')
        offset -= Timestamp(0)
        offset = (offset // Timedelta(1, unit=unit))
        if (is_list_like(arg) and (not isinstance(arg, (ABCSeries, Index, np.ndarray)))):
            arg = np.asarray(arg)
        arg = (arg + offset)
    return arg

@overload
def to_datetime(arg, errors=..., dayfirst=..., yearfirst=..., utc=..., format=..., exact=..., unit=..., infer_datetime_format=..., origin=..., cache=...):
    ...

@overload
def to_datetime(arg, errors=..., dayfirst=..., yearfirst=..., utc=..., format=..., exact=..., unit=..., infer_datetime_format=..., origin=..., cache=...):
    ...

@overload
def to_datetime(arg, errors=..., dayfirst=..., yearfirst=..., utc=..., format=..., exact=..., unit=..., infer_datetime_format=..., origin=..., cache=...):
    ...

def to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True):
    '\n    Convert argument to datetime.\n\n    Parameters\n    ----------\n    arg : int, float, str, datetime, list, tuple, 1-d array, Series, DataFrame/dict-like\n        The object to convert to a datetime.\n    errors : {\'ignore\', \'raise\', \'coerce\'}, default \'raise\'\n        - If \'raise\', then invalid parsing will raise an exception.\n        - If \'coerce\', then invalid parsing will be set as NaT.\n        - If \'ignore\', then invalid parsing will return the input.\n    dayfirst : bool, default False\n        Specify a date parse order if `arg` is str or its list-likes.\n        If True, parses dates with the day first, eg 10/11/12 is parsed as\n        2012-11-10.\n        Warning: dayfirst=True is not strict, but will prefer to parse\n        with day first (this is a known bug, based on dateutil behavior).\n    yearfirst : bool, default False\n        Specify a date parse order if `arg` is str or its list-likes.\n\n        - If True parses dates with the year first, eg 10/11/12 is parsed as\n          2010-11-12.\n        - If both dayfirst and yearfirst are True, yearfirst is preceded (same\n          as dateutil).\n\n        Warning: yearfirst=True is not strict, but will prefer to parse\n        with year first (this is a known bug, based on dateutil behavior).\n    utc : bool, default None\n        Return UTC DatetimeIndex if True (converting any tz-aware\n        datetime.datetime objects as well).\n    format : str, default None\n        The strftime to parse time, eg "%d/%m/%Y", note that "%f" will parse\n        all the way up to nanoseconds.\n        See strftime documentation for more information on choices:\n        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.\n    exact : bool, True by default\n        Behaves as:\n        - If True, require an exact format match.\n        - If False, allow the format to match anywhere in the target string.\n\n    unit : str, default \'ns\'\n        The unit of the arg (D,s,ms,us,ns) denote the unit, which is an\n        integer or float number. This will be based off the origin.\n        Example, with unit=\'ms\' and origin=\'unix\' (the default), this\n        would calculate the number of milliseconds to the unix epoch start.\n    infer_datetime_format : bool, default False\n        If True and no `format` is given, attempt to infer the format of the\n        datetime strings based on the first non-NaN element,\n        and if it can be inferred, switch to a faster method of parsing them.\n        In some cases this can increase the parsing speed by ~5-10x.\n    origin : scalar, default \'unix\'\n        Define the reference date. The numeric values would be parsed as number\n        of units (defined by `unit`) since this reference date.\n\n        - If \'unix\' (or POSIX) time; origin is set to 1970-01-01.\n        - If \'julian\', unit must be \'D\', and origin is set to beginning of\n          Julian Calendar. Julian day number 0 is assigned to the day starting\n          at noon on January 1, 4713 BC.\n        - If Timestamp convertible, origin is set to Timestamp identified by\n          origin.\n    cache : bool, default True\n        If True, use a cache of unique, converted dates to apply the datetime\n        conversion. May produce significant speed-up when parsing duplicate\n        date strings, especially ones with timezone offsets. The cache is only\n        used when there are at least 50 values. The presence of out-of-bounds\n        values will render the cache unusable and may slow down parsing.\n\n        .. versionchanged:: 0.25.0\n            - changed default value from False to True.\n\n    Returns\n    -------\n    datetime\n        If parsing succeeded.\n        Return type depends on input:\n\n        - list-like: DatetimeIndex\n        - Series: Series of datetime64 dtype\n        - scalar: Timestamp\n\n        In case when it is not possible to return designated types (e.g. when\n        any element of input is before Timestamp.min or after Timestamp.max)\n        return will have datetime.datetime type (or corresponding\n        array/Series).\n\n    See Also\n    --------\n    DataFrame.astype : Cast argument to a specified dtype.\n    to_timedelta : Convert argument to timedelta.\n    convert_dtypes : Convert dtypes.\n\n    Examples\n    --------\n    Assembling a datetime from multiple columns of a DataFrame. The keys can be\n    common abbreviations like [\'year\', \'month\', \'day\', \'minute\', \'second\',\n    \'ms\', \'us\', \'ns\']) or plurals of the same\n\n    >>> df = pd.DataFrame({\'year\': [2015, 2016],\n    ...                    \'month\': [2, 3],\n    ...                    \'day\': [4, 5]})\n    >>> pd.to_datetime(df)\n    0   2015-02-04\n    1   2016-03-05\n    dtype: datetime64[ns]\n\n    If a date does not meet the `timestamp limitations\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html\n    #timeseries-timestamp-limits>`_, passing errors=\'ignore\'\n    will return the original input instead of raising any exception.\n\n    Passing errors=\'coerce\' will force an out-of-bounds date to NaT,\n    in addition to forcing non-dates (or non-parseable dates) to NaT.\n\n    >>> pd.to_datetime(\'13000101\', format=\'%Y%m%d\', errors=\'ignore\')\n    datetime.datetime(1300, 1, 1, 0, 0)\n    >>> pd.to_datetime(\'13000101\', format=\'%Y%m%d\', errors=\'coerce\')\n    NaT\n\n    Passing infer_datetime_format=True can often-times speedup a parsing\n    if its not an ISO8601 format exactly, but in a regular format.\n\n    >>> s = pd.Series([\'3/11/2000\', \'3/12/2000\', \'3/13/2000\'] * 1000)\n    >>> s.head()\n    0    3/11/2000\n    1    3/12/2000\n    2    3/13/2000\n    3    3/11/2000\n    4    3/12/2000\n    dtype: object\n\n    >>> %timeit pd.to_datetime(s, infer_datetime_format=True)  # doctest: +SKIP\n    100 loops, best of 3: 10.4 ms per loop\n\n    >>> %timeit pd.to_datetime(s, infer_datetime_format=False)  # doctest: +SKIP\n    1 loop, best of 3: 471 ms per loop\n\n    Using a unix epoch time\n\n    >>> pd.to_datetime(1490195805, unit=\'s\')\n    Timestamp(\'2017-03-22 15:16:45\')\n    >>> pd.to_datetime(1490195805433502912, unit=\'ns\')\n    Timestamp(\'2017-03-22 15:16:45.433502912\')\n\n    .. warning:: For float arg, precision rounding might happen. To prevent\n        unexpected behavior use a fixed-width exact type.\n\n    Using a non-unix epoch origin\n\n    >>> pd.to_datetime([1, 2, 3], unit=\'D\',\n    ...                origin=pd.Timestamp(\'1960-01-01\'))\n    DatetimeIndex([\'1960-01-02\', \'1960-01-03\', \'1960-01-04\'], dtype=\'datetime64[ns]\', freq=None)\n    '
    if (arg is None):
        return None
    if (origin != 'unix'):
        arg = _adjust_to_origin(arg, origin, unit)
    tz = ('utc' if utc else None)
    convert_listlike = partial(_convert_listlike_datetimes, tz=tz, unit=unit, dayfirst=dayfirst, yearfirst=yearfirst, errors=errors, exact=exact, infer_datetime_format=infer_datetime_format)
    if isinstance(arg, Timestamp):
        result = arg
        if (tz is not None):
            if (arg.tz is not None):
                result = result.tz_convert(tz)
            else:
                result = result.tz_localize(tz)
    elif isinstance(arg, ABCSeries):
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if (not cache_array.empty):
            result = arg.map(cache_array)
        else:
            values = convert_listlike(arg._values, format)
            result = arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, (ABCDataFrame, abc.MutableMapping)):
        result = _assemble_from_unit_mappings(arg, errors, tz)
    elif isinstance(arg, Index):
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if (not cache_array.empty):
            result = _convert_and_box_cache(arg, cache_array, name=arg.name)
        else:
            result = convert_listlike(arg, format, name=arg.name)
    elif is_list_like(arg):
        try:
            cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        except OutOfBoundsDatetime:
            if (errors == 'raise'):
                raise
            from pandas import Series
            cache_array = Series([], dtype=object)
        if (not cache_array.empty):
            result = _convert_and_box_cache(arg, cache_array)
        else:
            result = convert_listlike(arg, format)
    else:
        result = convert_listlike(np.array([arg]), format)[0]
    return result
_unit_map = {'year': 'year', 'years': 'year', 'month': 'month', 'months': 'month', 'day': 'day', 'days': 'day', 'hour': 'h', 'hours': 'h', 'minute': 'm', 'minutes': 'm', 'second': 's', 'seconds': 's', 'ms': 'ms', 'millisecond': 'ms', 'milliseconds': 'ms', 'us': 'us', 'microsecond': 'us', 'microseconds': 'us', 'ns': 'ns', 'nanosecond': 'ns', 'nanoseconds': 'ns'}

def _assemble_from_unit_mappings(arg, errors, tz):
    "\n    assemble the unit specified fields from the arg (DataFrame)\n    Return a Series for actual parsing\n\n    Parameters\n    ----------\n    arg : DataFrame\n    errors : {'ignore', 'raise', 'coerce'}, default 'raise'\n\n        - If 'raise', then invalid parsing will raise an exception\n        - If 'coerce', then invalid parsing will be set as NaT\n        - If 'ignore', then invalid parsing will return the input\n    tz : None or 'utc'\n\n    Returns\n    -------\n    Series\n    "
    from pandas import DataFrame, to_numeric, to_timedelta
    arg = DataFrame(arg)
    if (not arg.columns.is_unique):
        raise ValueError('cannot assemble with duplicate keys')

    def f(value):
        if (value in _unit_map):
            return _unit_map[value]
        if (value.lower() in _unit_map):
            return _unit_map[value.lower()]
        return value
    unit = {k: f(k) for k in arg.keys()}
    unit_rev = {v: k for (k, v) in unit.items()}
    required = ['year', 'month', 'day']
    req = sorted((set(required) - set(unit_rev.keys())))
    if len(req):
        _required = ','.join(req)
        raise ValueError(f'to assemble mappings requires at least that [year, month, day] be specified: [{_required}] is missing')
    excess = sorted((set(unit_rev.keys()) - set(_unit_map.values())))
    if len(excess):
        _excess = ','.join(excess)
        raise ValueError(f'extra keys have been passed to the datetime assemblage: [{_excess}]')

    def coerce(values):
        values = to_numeric(values, errors=errors)
        if is_integer_dtype(values):
            values = values.astype('int64', copy=False)
        return values
    values = (((coerce(arg[unit_rev['year']]) * 10000) + (coerce(arg[unit_rev['month']]) * 100)) + coerce(arg[unit_rev['day']]))
    try:
        values = to_datetime(values, format='%Y%m%d', errors=errors, utc=tz)
    except (TypeError, ValueError) as err:
        raise ValueError(f'cannot assemble the datetimes: {err}') from err
    for u in ['h', 'm', 's', 'ms', 'us', 'ns']:
        value = unit_rev.get(u)
        if ((value is not None) and (value in arg)):
            try:
                values += to_timedelta(coerce(arg[value]), unit=u, errors=errors)
            except (TypeError, ValueError) as err:
                raise ValueError(f'cannot assemble the datetimes [{value}]: {err}') from err
    return values

def _attempt_YYYYMMDD(arg, errors):
    "\n    try to parse the YYYYMMDD/%Y%m%d format, try to deal with NaT-like,\n    arg is a passed in as an object dtype, but could really be ints/strings\n    with nan-like/or floats (e.g. with nan)\n\n    Parameters\n    ----------\n    arg : passed value\n    errors : 'raise','ignore','coerce'\n    "

    def calc(carg):
        carg = carg.astype(object)
        parsed = parsing.try_parse_year_month_day((carg / 10000), ((carg / 100) % 100), (carg % 100))
        return tslib.array_to_datetime(parsed, errors=errors)[0]

    def calc_with_mask(carg, mask):
        result = np.empty(carg.shape, dtype='M8[ns]')
        iresult = result.view('i8')
        iresult[(~ mask)] = iNaT
        masked_result = calc(carg[mask].astype(np.float64).astype(np.int64))
        result[mask] = masked_result.astype('M8[ns]')
        return result
    try:
        return calc(arg.astype(np.int64))
    except (ValueError, OverflowError, TypeError):
        pass
    try:
        carg = arg.astype(np.float64)
        return calc_with_mask(carg, notna(carg))
    except (ValueError, OverflowError, TypeError):
        pass
    try:
        mask = (~ algorithms.isin(arg, list(nat_strings)))
        return calc_with_mask(arg, mask)
    except (ValueError, OverflowError, TypeError):
        pass
    return None

def to_time(arg, format=None, infer_time_format=False, errors='raise'):
    warnings.warn('`to_time` has been moved, should be imported from pandas.core.tools.times.  This alias will be removed in a future version.', FutureWarning, stacklevel=2)
    from pandas.core.tools.times import to_time
    return to_time(arg, format, infer_time_format, errors)
