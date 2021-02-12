
'\ntimedelta support tools\n'
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import NaT
from pandas._libs.tslibs.timedeltas import Timedelta, parse_timedelta_unit
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import ABCIndex, ABCSeries
from pandas.core.arrays.timedeltas import sequence_to_td64ns

def to_timedelta(arg, unit=None, errors='raise'):
    '\n    Convert argument to timedelta.\n\n    Timedeltas are absolute differences in times, expressed in difference\n    units (e.g. days, hours, minutes, seconds). This method converts\n    an argument from a recognized timedelta format / value into\n    a Timedelta type.\n\n    Parameters\n    ----------\n    arg : str, timedelta, list-like or Series\n        The data to be converted to timedelta.\n\n        .. deprecated:: 1.2\n            Strings with units \'M\', \'Y\' and \'y\' do not represent\n            unambiguous timedelta values and will be removed in a future version\n\n    unit : str, optional\n        Denotes the unit of the arg for numeric `arg`. Defaults to ``"ns"``.\n\n        Possible values:\n\n        * \'W\'\n        * \'D\' / \'days\' / \'day\'\n        * \'hours\' / \'hour\' / \'hr\' / \'h\'\n        * \'m\' / \'minute\' / \'min\' / \'minutes\' / \'T\'\n        * \'S\' / \'seconds\' / \'sec\' / \'second\'\n        * \'ms\' / \'milliseconds\' / \'millisecond\' / \'milli\' / \'millis\' / \'L\'\n        * \'us\' / \'microseconds\' / \'microsecond\' / \'micro\' / \'micros\' / \'U\'\n        * \'ns\' / \'nanoseconds\' / \'nano\' / \'nanos\' / \'nanosecond\' / \'N\'\n\n        .. versionchanged:: 1.1.0\n\n           Must not be specified when `arg` context strings and\n           ``errors="raise"``.\n\n    errors : {\'ignore\', \'raise\', \'coerce\'}, default \'raise\'\n        - If \'raise\', then invalid parsing will raise an exception.\n        - If \'coerce\', then invalid parsing will be set as NaT.\n        - If \'ignore\', then invalid parsing will return the input.\n\n    Returns\n    -------\n    timedelta64 or numpy.array of timedelta64\n        Output type returned if parsing succeeded.\n\n    See Also\n    --------\n    DataFrame.astype : Cast argument to a specified dtype.\n    to_datetime : Convert argument to datetime.\n    convert_dtypes : Convert dtypes.\n\n    Notes\n    -----\n    If the precision is higher than nanoseconds, the precision of the duration is\n    truncated to nanoseconds for string inputs.\n\n    Examples\n    --------\n    Parsing a single string to a Timedelta:\n\n    >>> pd.to_timedelta(\'1 days 06:05:01.00003\')\n    Timedelta(\'1 days 06:05:01.000030\')\n    >>> pd.to_timedelta(\'15.5us\')\n    Timedelta(\'0 days 00:00:00.000015500\')\n\n    Parsing a list or array of strings:\n\n    >>> pd.to_timedelta([\'1 days 06:05:01.00003\', \'15.5us\', \'nan\'])\n    TimedeltaIndex([\'1 days 06:05:01.000030\', \'0 days 00:00:00.000015500\', NaT],\n                   dtype=\'timedelta64[ns]\', freq=None)\n\n    Converting numbers by specifying the `unit` keyword argument:\n\n    >>> pd.to_timedelta(np.arange(5), unit=\'s\')\n    TimedeltaIndex([\'0 days 00:00:00\', \'0 days 00:00:01\', \'0 days 00:00:02\',\n                    \'0 days 00:00:03\', \'0 days 00:00:04\'],\n                   dtype=\'timedelta64[ns]\', freq=None)\n    >>> pd.to_timedelta(np.arange(5), unit=\'d\')\n    TimedeltaIndex([\'0 days\', \'1 days\', \'2 days\', \'3 days\', \'4 days\'],\n                   dtype=\'timedelta64[ns]\', freq=None)\n    '
    if (unit is not None):
        unit = parse_timedelta_unit(unit)
    if (errors not in ('ignore', 'raise', 'coerce')):
        raise ValueError("errors must be one of 'ignore', 'raise', or 'coerce'.")
    if (unit in {'Y', 'y', 'M'}):
        raise ValueError("Units 'M', 'Y', and 'y' are no longer supported, as they do not represent unambiguous timedelta values durations.")
    if (arg is None):
        return arg
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, unit=unit, errors=errors)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, unit=unit, errors=errors, name=arg.name)
    elif (isinstance(arg, np.ndarray) and (arg.ndim == 0)):
        arg = lib.item_from_zerodim(arg)
    elif (is_list_like(arg) and (getattr(arg, 'ndim', 1) == 1)):
        return _convert_listlike(arg, unit=unit, errors=errors)
    elif (getattr(arg, 'ndim', 1) > 1):
        raise TypeError('arg must be a string, timedelta, list, tuple, 1-d array, or Series')
    if (isinstance(arg, str) and (unit is not None)):
        raise ValueError('unit must not be specified if the input is/contains a str')
    return _coerce_scalar_to_timedelta_type(arg, unit=unit, errors=errors)

def _coerce_scalar_to_timedelta_type(r, unit='ns', errors='raise'):
    "Convert string 'r' to a timedelta object."
    try:
        result = Timedelta(r, unit)
    except ValueError:
        if (errors == 'raise'):
            raise
        elif (errors == 'ignore'):
            return r
        result = NaT
    return result

def _convert_listlike(arg, unit=None, errors='raise', name=None):
    'Convert a list of objects to a timedelta index object.'
    if (isinstance(arg, (list, tuple)) or (not hasattr(arg, 'dtype'))):
        arg = np.array(list(arg), dtype=object)
    try:
        value = sequence_to_td64ns(arg, unit=unit, errors=errors, copy=False)[0]
    except ValueError:
        if (errors == 'ignore'):
            return arg
        else:
            raise
    from pandas import TimedeltaIndex
    value = TimedeltaIndex(value, unit='ns', name=name)
    return value
