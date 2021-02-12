
' implement the TimedeltaIndex '
from pandas._libs import index as libindex, lib
from pandas._libs.tslibs import Timedelta, to_offset
from pandas._typing import DtypeObj
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import TD64NS_DTYPE, is_scalar, is_timedelta64_dtype
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names

@inherit_names((['__neg__', '__pos__', '__abs__', 'total_seconds', 'round', 'floor', 'ceil'] + TimedeltaArray._field_ops), TimedeltaArray, wrap=True)
@inherit_names(['_bool_ops', '_object_ops', '_field_ops', '_datetimelike_ops', '_datetimelike_methods', '_other_ops', 'components', 'to_pytimedelta', 'sum', 'std', 'median', '_format_native_types'], TimedeltaArray)
class TimedeltaIndex(DatetimeTimedeltaMixin):
    "\n    Immutable ndarray of timedelta64 data, represented internally as int64, and\n    which can be boxed to timedelta objects.\n\n    Parameters\n    ----------\n    data  : array-like (1-dimensional), optional\n        Optional timedelta-like data to construct index with.\n    unit : unit of the arg (D,h,m,s,ms,us,ns) denote the unit, optional\n        Which is an integer/float number.\n    freq : str or pandas offset object, optional\n        One of pandas date offset strings or corresponding objects. The string\n        'infer' can be passed in order to set the frequency of the index as the\n        inferred frequency upon creation.\n    copy  : bool\n        Make a copy of input ndarray.\n    name : object\n        Name to be stored in the index.\n\n    Attributes\n    ----------\n    days\n    seconds\n    microseconds\n    nanoseconds\n    components\n    inferred_freq\n\n    Methods\n    -------\n    to_pytimedelta\n    to_series\n    round\n    floor\n    ceil\n    to_frame\n    mean\n\n    See Also\n    --------\n    Index : The base pandas Index type.\n    Timedelta : Represents a duration between two dates or times.\n    DatetimeIndex : Index of datetime64 data.\n    PeriodIndex : Index of Period data.\n    timedelta_range : Create a fixed-frequency TimedeltaIndex.\n\n    Notes\n    -----\n    To learn more about the frequency strings, please see `this link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n    "
    _typ = 'timedeltaindex'
    _data_cls = TimedeltaArray
    _engine_type = libindex.TimedeltaEngine
    _comparables = ['name', 'freq']
    _attributes = ['name', 'freq']
    _is_numeric_dtype = False

    def __new__(cls, data=None, unit=None, freq=lib.no_default, closed=None, dtype=TD64NS_DTYPE, copy=False, name=None):
        name = maybe_extract_name(name, data, cls)
        if is_scalar(data):
            raise TypeError(f'{cls.__name__}() must be called with a collection of some kind, {repr(data)} was passed')
        if (unit in {'Y', 'y', 'M'}):
            raise ValueError("Units 'M', 'Y', and 'y' are no longer supported, as they do not represent unambiguous timedelta values durations.")
        if (isinstance(data, TimedeltaArray) and (freq is lib.no_default)):
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)
        if (isinstance(data, TimedeltaIndex) and (freq is lib.no_default) and (name is None)):
            if copy:
                return data.copy()
            else:
                return data._shallow_copy()
        tdarr = TimedeltaArray._from_sequence_not_strict(data, freq=freq, unit=unit, dtype=dtype, copy=copy)
        return cls._simple_new(tdarr, name=name)

    def _is_comparable_dtype(self, dtype):
        '\n        Can we compare values of the given dtype to our own?\n        '
        return is_timedelta64_dtype(dtype)

    def get_loc(self, key, method=None, tolerance=None):
        '\n        Get integer location for requested label\n\n        Returns\n        -------\n        loc : int, slice, or ndarray[int]\n        '
        if (not is_scalar(key)):
            raise InvalidIndexError(key)
        try:
            key = self._data._validate_scalar(key, unbox=False)
        except TypeError as err:
            raise KeyError(key) from err
        return Index.get_loc(self, key, method, tolerance)

    def _maybe_cast_slice_bound(self, label, side, kind):
        "\n        If label is a string, cast it to timedelta according to resolution.\n\n        Parameters\n        ----------\n        label : object\n        side : {'left', 'right'}\n        kind : {'loc', 'getitem'} or None\n\n        Returns\n        -------\n        label : object\n        "
        assert (kind in ['loc', 'getitem', None])
        if isinstance(label, str):
            parsed = Timedelta(label)
            lbound = parsed.round(parsed.resolution_string)
            if (side == 'left'):
                return lbound
            else:
                return ((lbound + to_offset(parsed.resolution_string)) - Timedelta(1, 'ns'))
        elif (not isinstance(label, self._data._recognized_scalars)):
            raise self._invalid_indexer('slice', label)
        return label

    @property
    def inferred_type(self):
        return 'timedelta64'

def timedelta_range(start=None, end=None, periods=None, freq=None, name=None, closed=None):
    "\n    Return a fixed frequency TimedeltaIndex, with day as the default\n    frequency.\n\n    Parameters\n    ----------\n    start : str or timedelta-like, default None\n        Left bound for generating timedeltas.\n    end : str or timedelta-like, default None\n        Right bound for generating timedeltas.\n    periods : int, default None\n        Number of periods to generate.\n    freq : str or DateOffset, default 'D'\n        Frequency strings can have multiples, e.g. '5H'.\n    name : str, default None\n        Name of the resulting TimedeltaIndex.\n    closed : str, default None\n        Make the interval closed with respect to the given frequency to\n        the 'left', 'right', or both sides (None).\n\n    Returns\n    -------\n    rng : TimedeltaIndex\n\n    Notes\n    -----\n    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,\n    exactly three must be specified. If ``freq`` is omitted, the resulting\n    ``TimedeltaIndex`` will have ``periods`` linearly spaced elements between\n    ``start`` and ``end`` (closed on both sides).\n\n    To learn more about the frequency strings, please see `this link\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.\n\n    Examples\n    --------\n    >>> pd.timedelta_range(start='1 day', periods=4)\n    TimedeltaIndex(['1 days', '2 days', '3 days', '4 days'],\n                   dtype='timedelta64[ns]', freq='D')\n\n    The ``closed`` parameter specifies which endpoint is included.  The default\n    behavior is to include both endpoints.\n\n    >>> pd.timedelta_range(start='1 day', periods=4, closed='right')\n    TimedeltaIndex(['2 days', '3 days', '4 days'],\n                   dtype='timedelta64[ns]', freq='D')\n\n    The ``freq`` parameter specifies the frequency of the TimedeltaIndex.\n    Only fixed frequencies can be passed, non-fixed frequencies such as\n    'M' (month end) will raise.\n\n    >>> pd.timedelta_range(start='1 day', end='2 days', freq='6H')\n    TimedeltaIndex(['1 days 00:00:00', '1 days 06:00:00', '1 days 12:00:00',\n                    '1 days 18:00:00', '2 days 00:00:00'],\n                   dtype='timedelta64[ns]', freq='6H')\n\n    Specify ``start``, ``end``, and ``periods``; the frequency is generated\n    automatically (linearly spaced).\n\n    >>> pd.timedelta_range(start='1 day', end='5 days', periods=4)\n    TimedeltaIndex(['1 days 00:00:00', '2 days 08:00:00', '3 days 16:00:00',\n                    '5 days 00:00:00'],\n                   dtype='timedelta64[ns]', freq=None)\n    "
    if ((freq is None) and com.any_none(periods, start, end)):
        freq = 'D'
    (freq, _) = dtl.maybe_infer_freq(freq)
    tdarr = TimedeltaArray._generate_range(start, end, periods, freq, closed=closed)
    return TimedeltaIndex._simple_new(tdarr, name=name)
