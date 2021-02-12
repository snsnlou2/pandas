
from typing import Optional
import warnings
import numpy as np
from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import Timestamp, tzconversion
from pandas._libs.tslibs.ccalendar import DAYS, MONTH_ALIASES, MONTH_NUMBERS, MONTHS, int_to_weekday
from pandas._libs.tslibs.fields import build_field_sarray, month_position_check
from pandas._libs.tslibs.offsets import DateOffset, Day, _get_offset, to_offset
from pandas._libs.tslibs.parsing import get_rule_month
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_datetime64_dtype, is_period_dtype, is_timedelta64_dtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.algorithms import unique
_ONE_MICRO = 1000
_ONE_MILLI = (_ONE_MICRO * 1000)
_ONE_SECOND = (_ONE_MILLI * 1000)
_ONE_MINUTE = (60 * _ONE_SECOND)
_ONE_HOUR = (60 * _ONE_MINUTE)
_ONE_DAY = (24 * _ONE_HOUR)
_offset_to_period_map = {'WEEKDAY': 'D', 'EOM': 'M', 'BM': 'M', 'BQS': 'Q', 'QS': 'Q', 'BQ': 'Q', 'BA': 'A', 'AS': 'A', 'BAS': 'A', 'MS': 'M', 'D': 'D', 'C': 'C', 'B': 'B', 'T': 'T', 'S': 'S', 'L': 'L', 'U': 'U', 'N': 'N', 'H': 'H', 'Q': 'Q', 'A': 'A', 'W': 'W', 'M': 'M', 'Y': 'A', 'BY': 'A', 'YS': 'A', 'BYS': 'A'}
_need_suffix = ['QS', 'BQ', 'BQS', 'YS', 'AS', 'BY', 'BA', 'BYS', 'BAS']
for _prefix in _need_suffix:
    for _m in MONTHS:
        key = f'{_prefix}-{_m}'
        _offset_to_period_map[key] = _offset_to_period_map[_prefix]
for _prefix in ['A', 'Q']:
    for _m in MONTHS:
        _alias = f'{_prefix}-{_m}'
        _offset_to_period_map[_alias] = _alias
for _d in DAYS:
    _offset_to_period_map[f'W-{_d}'] = f'W-{_d}'

def get_period_alias(offset_str):
    '\n    Alias to closest period strings BQ->Q etc.\n    '
    return _offset_to_period_map.get(offset_str, None)

def get_offset(name):
    "\n    Return DateOffset object associated with rule name.\n\n    .. deprecated:: 1.0.0\n\n    Examples\n    --------\n    get_offset('EOM') --> BMonthEnd(1)\n    "
    warnings.warn('get_offset is deprecated and will be removed in a future version, use to_offset instead', FutureWarning, stacklevel=2)
    return _get_offset(name)

def infer_freq(index, warn=True):
    '\n    Infer the most likely frequency given the input index. If the frequency is\n    uncertain, a warning will be printed.\n\n    Parameters\n    ----------\n    index : DatetimeIndex or TimedeltaIndex\n      If passed a Series will use the values of the series (NOT THE INDEX).\n    warn : bool, default True\n\n    Returns\n    -------\n    str or None\n        None if no discernible frequency.\n\n    Raises\n    ------\n    TypeError\n        If the index is not datetime-like.\n    ValueError\n        If there are fewer than three values.\n    '
    import pandas as pd
    if isinstance(index, ABCSeries):
        values = index._values
        if (not (is_datetime64_dtype(values) or is_timedelta64_dtype(values) or (values.dtype == object))):
            raise TypeError(f'cannot infer freq from a non-convertible dtype on a Series of {index.dtype}')
        index = values
    inferer: _FrequencyInferer
    if (not hasattr(index, 'dtype')):
        pass
    elif is_period_dtype(index.dtype):
        raise TypeError('PeriodIndex given. Check the `freq` attribute instead of using infer_freq.')
    elif is_timedelta64_dtype(index.dtype):
        inferer = _TimedeltaFrequencyInferer(index, warn=warn)
        return inferer.get_freq()
    if (isinstance(index, pd.Index) and (not isinstance(index, pd.DatetimeIndex))):
        if isinstance(index, (pd.Int64Index, pd.Float64Index)):
            raise TypeError(f'cannot infer freq from a non-convertible index type {type(index)}')
        index = index._values
    if (not isinstance(index, pd.DatetimeIndex)):
        index = pd.DatetimeIndex(index)
    inferer = _FrequencyInferer(index, warn=warn)
    return inferer.get_freq()

class _FrequencyInferer():
    '\n    Not sure if I can avoid the state machine here\n    '

    def __init__(self, index, warn=True):
        self.index = index
        self.i8values = index.asi8
        if hasattr(index, 'tz'):
            if (index.tz is not None):
                self.i8values = tzconversion.tz_convert_from_utc(self.i8values, index.tz)
        self.warn = warn
        if (len(index) < 3):
            raise ValueError('Need at least 3 dates to infer frequency')
        self.is_monotonic = (self.index._is_monotonic_increasing or self.index._is_monotonic_decreasing)

    @cache_readonly
    def deltas(self):
        return unique_deltas(self.i8values)

    @cache_readonly
    def deltas_asi8(self):
        return unique_deltas(self.index.asi8)

    @cache_readonly
    def is_unique(self):
        return (len(self.deltas) == 1)

    @cache_readonly
    def is_unique_asi8(self):
        return (len(self.deltas_asi8) == 1)

    def get_freq(self):
        '\n        Find the appropriate frequency string to describe the inferred\n        frequency of self.i8values\n\n        Returns\n        -------\n        str or None\n        '
        if ((not self.is_monotonic) or (not self.index._is_unique)):
            return None
        delta = self.deltas[0]
        if _is_multiple(delta, _ONE_DAY):
            return self._infer_daily_rule()
        if (self.hour_deltas in ([1, 17], [1, 65], [1, 17, 65])):
            return 'BH'
        elif (not self.is_unique_asi8):
            return None
        delta = self.deltas_asi8[0]
        if _is_multiple(delta, _ONE_HOUR):
            return _maybe_add_count('H', (delta / _ONE_HOUR))
        elif _is_multiple(delta, _ONE_MINUTE):
            return _maybe_add_count('T', (delta / _ONE_MINUTE))
        elif _is_multiple(delta, _ONE_SECOND):
            return _maybe_add_count('S', (delta / _ONE_SECOND))
        elif _is_multiple(delta, _ONE_MILLI):
            return _maybe_add_count('L', (delta / _ONE_MILLI))
        elif _is_multiple(delta, _ONE_MICRO):
            return _maybe_add_count('U', (delta / _ONE_MICRO))
        else:
            return _maybe_add_count('N', delta)

    @cache_readonly
    def day_deltas(self):
        return [(x / _ONE_DAY) for x in self.deltas]

    @cache_readonly
    def hour_deltas(self):
        return [(x / _ONE_HOUR) for x in self.deltas]

    @cache_readonly
    def fields(self):
        return build_field_sarray(self.i8values)

    @cache_readonly
    def rep_stamp(self):
        return Timestamp(self.i8values[0])

    def month_position_check(self):
        return month_position_check(self.fields, self.index.dayofweek)

    @cache_readonly
    def mdiffs(self):
        nmonths = ((self.fields['Y'] * 12) + self.fields['M'])
        return unique_deltas(nmonths.astype('i8'))

    @cache_readonly
    def ydiffs(self):
        return unique_deltas(self.fields['Y'].astype('i8'))

    def _infer_daily_rule(self):
        annual_rule = self._get_annual_rule()
        if annual_rule:
            nyears = self.ydiffs[0]
            month = MONTH_ALIASES[self.rep_stamp.month]
            alias = f'{annual_rule}-{month}'
            return _maybe_add_count(alias, nyears)
        quarterly_rule = self._get_quarterly_rule()
        if quarterly_rule:
            nquarters = (self.mdiffs[0] / 3)
            mod_dict = {0: 12, 2: 11, 1: 10}
            month = MONTH_ALIASES[mod_dict[(self.rep_stamp.month % 3)]]
            alias = f'{quarterly_rule}-{month}'
            return _maybe_add_count(alias, nquarters)
        monthly_rule = self._get_monthly_rule()
        if monthly_rule:
            return _maybe_add_count(monthly_rule, self.mdiffs[0])
        if self.is_unique:
            return self._get_daily_rule()
        if self._is_business_daily():
            return 'B'
        wom_rule = self._get_wom_rule()
        if wom_rule:
            return wom_rule
        return None

    def _get_daily_rule(self):
        days = (self.deltas[0] / _ONE_DAY)
        if ((days % 7) == 0):
            wd = int_to_weekday[self.rep_stamp.weekday()]
            alias = f'W-{wd}'
            return _maybe_add_count(alias, (days / 7))
        else:
            return _maybe_add_count('D', days)

    def _get_annual_rule(self):
        if (len(self.ydiffs) > 1):
            return None
        if (len(unique(self.fields['M'])) > 1):
            return None
        pos_check = self.month_position_check()
        return {'cs': 'AS', 'bs': 'BAS', 'ce': 'A', 'be': 'BA'}.get(pos_check)

    def _get_quarterly_rule(self):
        if (len(self.mdiffs) > 1):
            return None
        if (not ((self.mdiffs[0] % 3) == 0)):
            return None
        pos_check = self.month_position_check()
        return {'cs': 'QS', 'bs': 'BQS', 'ce': 'Q', 'be': 'BQ'}.get(pos_check)

    def _get_monthly_rule(self):
        if (len(self.mdiffs) > 1):
            return None
        pos_check = self.month_position_check()
        return {'cs': 'MS', 'bs': 'BMS', 'ce': 'M', 'be': 'BM'}.get(pos_check)

    def _is_business_daily(self):
        if (self.day_deltas != [1, 3]):
            return False
        first_weekday = self.index[0].weekday()
        shifts = np.diff(self.index.asi8)
        shifts = np.floor_divide(shifts, _ONE_DAY)
        weekdays = np.mod((first_weekday + np.cumsum(shifts)), 7)
        return np.all((((weekdays == 0) & (shifts == 3)) | (((weekdays > 0) & (weekdays <= 4)) & (shifts == 1))))

    def _get_wom_rule(self):
        weekdays = unique(self.index.weekday)
        if (len(weekdays) > 1):
            return None
        week_of_months = unique(((self.index.day - 1) // 7))
        week_of_months = week_of_months[(week_of_months < 4)]
        if ((len(week_of_months) == 0) or (len(week_of_months) > 1)):
            return None
        week = (week_of_months[0] + 1)
        wd = int_to_weekday[weekdays[0]]
        return f'WOM-{week}{wd}'

class _TimedeltaFrequencyInferer(_FrequencyInferer):

    def _infer_daily_rule(self):
        if self.is_unique:
            return self._get_daily_rule()

def _is_multiple(us, mult):
    return ((us % mult) == 0)

def _maybe_add_count(base, count):
    if (count != 1):
        assert (count == int(count))
        count = int(count)
        return f'{count}{base}'
    else:
        return base

def is_subperiod(source, target):
    '\n    Returns True if downsampling is possible between source and target\n    frequencies\n\n    Parameters\n    ----------\n    source : str or DateOffset\n        Frequency converting from\n    target : str or DateOffset\n        Frequency converting to\n\n    Returns\n    -------\n    bool\n    '
    if ((target is None) or (source is None)):
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)
    if _is_annual(target):
        if _is_quarterly(source):
            return _quarter_months_conform(get_rule_month(source), get_rule_month(target))
        return (source in {'D', 'C', 'B', 'M', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif _is_quarterly(target):
        return (source in {'D', 'C', 'B', 'M', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif _is_monthly(target):
        return (source in {'D', 'C', 'B', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif _is_weekly(target):
        return (source in {target, 'D', 'C', 'B', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif (target == 'B'):
        return (source in {'B', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif (target == 'C'):
        return (source in {'C', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif (target == 'D'):
        return (source in {'D', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif (target == 'H'):
        return (source in {'H', 'T', 'S', 'L', 'U', 'N'})
    elif (target == 'T'):
        return (source in {'T', 'S', 'L', 'U', 'N'})
    elif (target == 'S'):
        return (source in {'S', 'L', 'U', 'N'})
    elif (target == 'L'):
        return (source in {'L', 'U', 'N'})
    elif (target == 'U'):
        return (source in {'U', 'N'})
    elif (target == 'N'):
        return (source in {'N'})
    else:
        return False

def is_superperiod(source, target):
    '\n    Returns True if upsampling is possible between source and target\n    frequencies\n\n    Parameters\n    ----------\n    source : str or DateOffset\n        Frequency converting from\n    target : str or DateOffset\n        Frequency converting to\n\n    Returns\n    -------\n    bool\n    '
    if ((target is None) or (source is None)):
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)
    if _is_annual(source):
        if _is_annual(target):
            return (get_rule_month(source) == get_rule_month(target))
        if _is_quarterly(target):
            smonth = get_rule_month(source)
            tmonth = get_rule_month(target)
            return _quarter_months_conform(smonth, tmonth)
        return (target in {'D', 'C', 'B', 'M', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif _is_quarterly(source):
        return (target in {'D', 'C', 'B', 'M', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif _is_monthly(source):
        return (target in {'D', 'C', 'B', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif _is_weekly(source):
        return (target in {source, 'D', 'C', 'B', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif (source == 'B'):
        return (target in {'D', 'C', 'B', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif (source == 'C'):
        return (target in {'D', 'C', 'B', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif (source == 'D'):
        return (target in {'D', 'C', 'B', 'H', 'T', 'S', 'L', 'U', 'N'})
    elif (source == 'H'):
        return (target in {'H', 'T', 'S', 'L', 'U', 'N'})
    elif (source == 'T'):
        return (target in {'T', 'S', 'L', 'U', 'N'})
    elif (source == 'S'):
        return (target in {'S', 'L', 'U', 'N'})
    elif (source == 'L'):
        return (target in {'L', 'U', 'N'})
    elif (source == 'U'):
        return (target in {'U', 'N'})
    elif (source == 'N'):
        return (target in {'N'})
    else:
        return False

def _maybe_coerce_freq(code):
    'we might need to coerce a code to a rule_code\n    and uppercase it\n\n    Parameters\n    ----------\n    source : string or DateOffset\n        Frequency converting from\n\n    Returns\n    -------\n    str\n    '
    assert (code is not None)
    if isinstance(code, DateOffset):
        code = code.rule_code
    return code.upper()

def _quarter_months_conform(source, target):
    snum = MONTH_NUMBERS[source]
    tnum = MONTH_NUMBERS[target]
    return ((snum % 3) == (tnum % 3))

def _is_annual(rule):
    rule = rule.upper()
    return ((rule == 'A') or rule.startswith('A-'))

def _is_quarterly(rule):
    rule = rule.upper()
    return ((rule == 'Q') or rule.startswith('Q-') or rule.startswith('BQ'))

def _is_monthly(rule):
    rule = rule.upper()
    return ((rule == 'M') or (rule == 'BM'))

def _is_weekly(rule):
    rule = rule.upper()
    return ((rule == 'W') or rule.startswith('W-'))
