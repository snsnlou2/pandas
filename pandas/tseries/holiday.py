
from datetime import datetime, timedelta
from typing import List
import warnings
from dateutil.relativedelta import FR, MO, SA, SU, TH, TU, WE
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import DateOffset, DatetimeIndex, Series, Timestamp, concat, date_range
from pandas.tseries.offsets import Day, Easter

def next_monday(dt):
    '\n    If holiday falls on Saturday, use following Monday instead;\n    if holiday falls on Sunday, use Monday instead\n    '
    if (dt.weekday() == 5):
        return (dt + timedelta(2))
    elif (dt.weekday() == 6):
        return (dt + timedelta(1))
    return dt

def next_monday_or_tuesday(dt):
    '\n    For second holiday of two adjacent ones!\n    If holiday falls on Saturday, use following Monday instead;\n    if holiday falls on Sunday or Monday, use following Tuesday instead\n    (because Monday is already taken by adjacent holiday on the day before)\n    '
    dow = dt.weekday()
    if ((dow == 5) or (dow == 6)):
        return (dt + timedelta(2))
    elif (dow == 0):
        return (dt + timedelta(1))
    return dt

def previous_friday(dt):
    '\n    If holiday falls on Saturday or Sunday, use previous Friday instead.\n    '
    if (dt.weekday() == 5):
        return (dt - timedelta(1))
    elif (dt.weekday() == 6):
        return (dt - timedelta(2))
    return dt

def sunday_to_monday(dt):
    '\n    If holiday falls on Sunday, use day thereafter (Monday) instead.\n    '
    if (dt.weekday() == 6):
        return (dt + timedelta(1))
    return dt

def weekend_to_monday(dt):
    '\n    If holiday falls on Sunday or Saturday,\n    use day thereafter (Monday) instead.\n    Needed for holidays such as Christmas observation in Europe\n    '
    if (dt.weekday() == 6):
        return (dt + timedelta(1))
    elif (dt.weekday() == 5):
        return (dt + timedelta(2))
    return dt

def nearest_workday(dt):
    '\n    If holiday falls on Saturday, use day before (Friday) instead;\n    if holiday falls on Sunday, use day thereafter (Monday) instead.\n    '
    if (dt.weekday() == 5):
        return (dt - timedelta(1))
    elif (dt.weekday() == 6):
        return (dt + timedelta(1))
    return dt

def next_workday(dt):
    '\n    returns next weekday used for observances\n    '
    dt += timedelta(days=1)
    while (dt.weekday() > 4):
        dt += timedelta(days=1)
    return dt

def previous_workday(dt):
    '\n    returns previous weekday used for observances\n    '
    dt -= timedelta(days=1)
    while (dt.weekday() > 4):
        dt -= timedelta(days=1)
    return dt

def before_nearest_workday(dt):
    '\n    returns previous workday after nearest workday\n    '
    return previous_workday(nearest_workday(dt))

def after_nearest_workday(dt):
    '\n    returns next workday after nearest workday\n    needed for Boxing day or multiple holidays in a series\n    '
    return next_workday(nearest_workday(dt))

class Holiday():
    '\n    Class that defines a holiday with start/end dates and rules\n    for observance.\n    '

    def __init__(self, name, year=None, month=None, day=None, offset=None, observance=None, start_date=None, end_date=None, days_of_week=None):
        '\n        Parameters\n        ----------\n        name : str\n            Name of the holiday , defaults to class name\n        offset : array of pandas.tseries.offsets or\n                class from pandas.tseries.offsets\n            computes offset from date\n        observance: function\n            computes when holiday is given a pandas Timestamp\n        days_of_week:\n            provide a tuple of days e.g  (0,1,2,3,) for Monday Through Thursday\n            Monday=0,..,Sunday=6\n\n        Examples\n        --------\n        >>> from pandas.tseries.holiday import Holiday, nearest_workday\n        >>> from dateutil.relativedelta import MO\n\n        >>> USMemorialDay = Holiday(\n        ...     "Memorial Day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))\n        ... )\n        >>> USMemorialDay\n        Holiday: Memorial Day (month=5, day=31, offset=<DateOffset: weekday=MO(-1)>)\n\n        >>> USLaborDay = Holiday(\n        ...     "Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))\n        ... )\n        >>> USLaborDay\n        Holiday: Labor Day (month=9, day=1, offset=<DateOffset: weekday=MO(+1)>)\n\n        >>> July3rd = Holiday("July 3rd", month=7, day=3)\n        >>> July3rd\n        Holiday: July 3rd (month=7, day=3, )\n\n        >>> NewYears = Holiday(\n        ...     "New Years Day", month=1,  day=1, observance=nearest_workday\n        ... )\n        >>> NewYears  # doctest: +SKIP\n        Holiday: New Years Day (\n            month=1, day=1, observance=<function nearest_workday at 0x66545e9bc440>\n        )\n\n        >>> July3rd = Holiday("July 3rd", month=7, day=3, days_of_week=(0, 1, 2, 3))\n        >>> July3rd\n        Holiday: July 3rd (month=7, day=3, )\n        '
        if ((offset is not None) and (observance is not None)):
            raise NotImplementedError('Cannot use both offset and observance.')
        self.name = name
        self.year = year
        self.month = month
        self.day = day
        self.offset = offset
        self.start_date = (Timestamp(start_date) if (start_date is not None) else start_date)
        self.end_date = (Timestamp(end_date) if (end_date is not None) else end_date)
        self.observance = observance
        assert ((days_of_week is None) or (type(days_of_week) == tuple))
        self.days_of_week = days_of_week

    def __repr__(self):
        info = ''
        if (self.year is not None):
            info += f'year={self.year}, '
        info += f'month={self.month}, day={self.day}, '
        if (self.offset is not None):
            info += f'offset={self.offset}'
        if (self.observance is not None):
            info += f'observance={self.observance}'
        repr = f'Holiday: {self.name} ({info})'
        return repr

    def dates(self, start_date, end_date, return_name=False):
        '\n        Calculate holidays observed between start date and end date\n\n        Parameters\n        ----------\n        start_date : starting date, datetime-like, optional\n        end_date : ending date, datetime-like, optional\n        return_name : bool, optional, default=False\n            If True, return a series that has dates and holiday names.\n            False will only return dates.\n        '
        start_date = Timestamp(start_date)
        end_date = Timestamp(end_date)
        filter_start_date = start_date
        filter_end_date = end_date
        if (self.year is not None):
            dt = Timestamp(datetime(self.year, self.month, self.day))
            if return_name:
                return Series(self.name, index=[dt])
            else:
                return [dt]
        dates = self._reference_dates(start_date, end_date)
        holiday_dates = self._apply_rule(dates)
        if (self.days_of_week is not None):
            holiday_dates = holiday_dates[np.in1d(holiday_dates.dayofweek, self.days_of_week)]
        if (self.start_date is not None):
            filter_start_date = max(self.start_date.tz_localize(filter_start_date.tz), filter_start_date)
        if (self.end_date is not None):
            filter_end_date = min(self.end_date.tz_localize(filter_end_date.tz), filter_end_date)
        holiday_dates = holiday_dates[((holiday_dates >= filter_start_date) & (holiday_dates <= filter_end_date))]
        if return_name:
            return Series(self.name, index=holiday_dates)
        return holiday_dates

    def _reference_dates(self, start_date, end_date):
        '\n        Get reference dates for the holiday.\n\n        Return reference dates for the holiday also returning the year\n        prior to the start_date and year following the end_date.  This ensures\n        that any offsets to be applied will yield the holidays within\n        the passed in dates.\n        '
        if (self.start_date is not None):
            start_date = self.start_date.tz_localize(start_date.tz)
        if (self.end_date is not None):
            end_date = self.end_date.tz_localize(start_date.tz)
        year_offset = DateOffset(years=1)
        reference_start_date = Timestamp(datetime((start_date.year - 1), self.month, self.day))
        reference_end_date = Timestamp(datetime((end_date.year + 1), self.month, self.day))
        dates = date_range(start=reference_start_date, end=reference_end_date, freq=year_offset, tz=start_date.tz)
        return dates

    def _apply_rule(self, dates):
        '\n        Apply the given offset/observance to a DatetimeIndex of dates.\n\n        Parameters\n        ----------\n        dates : DatetimeIndex\n            Dates to apply the given offset/observance rule\n\n        Returns\n        -------\n        Dates with rules applied\n        '
        if (self.observance is not None):
            return dates.map((lambda d: self.observance(d)))
        if (self.offset is not None):
            if (not isinstance(self.offset, list)):
                offsets = [self.offset]
            else:
                offsets = self.offset
            for offset in offsets:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', PerformanceWarning)
                    dates += offset
        return dates
holiday_calendars = {}

def register(cls):
    try:
        name = cls.name
    except AttributeError:
        name = cls.__name__
    holiday_calendars[name] = cls

def get_calendar(name):
    '\n    Return an instance of a calendar based on its name.\n\n    Parameters\n    ----------\n    name : str\n        Calendar name to return an instance of\n    '
    return holiday_calendars[name]()

class HolidayCalendarMetaClass(type):

    def __new__(cls, clsname, bases, attrs):
        calendar_class = super().__new__(cls, clsname, bases, attrs)
        register(calendar_class)
        return calendar_class

class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    '\n    Abstract interface to create holidays following certain rules.\n    '
    rules = []
    start_date = Timestamp(datetime(1970, 1, 1))
    end_date = Timestamp(datetime(2200, 12, 31))
    _cache = None

    def __init__(self, name=None, rules=None):
        '\n        Initializes holiday object with a given set a rules.  Normally\n        classes just have the rules defined within them.\n\n        Parameters\n        ----------\n        name : str\n            Name of the holiday calendar, defaults to class name\n        rules : array of Holiday objects\n            A set of rules used to create the holidays.\n        '
        super().__init__()
        if (name is None):
            name = type(self).__name__
        self.name = name
        if (rules is not None):
            self.rules = rules

    def rule_from_name(self, name):
        for rule in self.rules:
            if (rule.name == name):
                return rule
        return None

    def holidays(self, start=None, end=None, return_name=False):
        '\n        Returns a curve with holidays between start_date and end_date\n\n        Parameters\n        ----------\n        start : starting date, datetime-like, optional\n        end : ending date, datetime-like, optional\n        return_name : bool, optional\n            If True, return a series that has dates and holiday names.\n            False will only return a DatetimeIndex of dates.\n\n        Returns\n        -------\n            DatetimeIndex of holidays\n        '
        if (self.rules is None):
            raise Exception(f'Holiday Calendar {self.name} does not have any rules specified')
        if (start is None):
            start = AbstractHolidayCalendar.start_date
        if (end is None):
            end = AbstractHolidayCalendar.end_date
        start = Timestamp(start)
        end = Timestamp(end)
        if ((self._cache is None) or (start < self._cache[0]) or (end > self._cache[1])):
            pre_holidays = [rule.dates(start, end, return_name=True) for rule in self.rules]
            if pre_holidays:
                holidays = concat(pre_holidays)
            else:
                holidays = Series(index=DatetimeIndex([]), dtype=object)
            self._cache = (start, end, holidays.sort_index())
        holidays = self._cache[2]
        holidays = holidays[start:end]
        if return_name:
            return holidays
        else:
            return holidays.index

    @staticmethod
    def merge_class(base, other):
        "\n        Merge holiday calendars together. The base calendar\n        will take precedence to other. The merge will be done\n        based on each holiday's name.\n\n        Parameters\n        ----------\n        base : AbstractHolidayCalendar\n          instance/subclass or array of Holiday objects\n        other : AbstractHolidayCalendar\n          instance/subclass or array of Holiday objects\n        "
        try:
            other = other.rules
        except AttributeError:
            pass
        if (not isinstance(other, list)):
            other = [other]
        other_holidays = {holiday.name: holiday for holiday in other}
        try:
            base = base.rules
        except AttributeError:
            pass
        if (not isinstance(base, list)):
            base = [base]
        base_holidays = {holiday.name: holiday for holiday in base}
        other_holidays.update(base_holidays)
        return list(other_holidays.values())

    def merge(self, other, inplace=False):
        "\n        Merge holiday calendars together.  The caller's class\n        rules take precedence.  The merge will be done\n        based on each holiday's name.\n\n        Parameters\n        ----------\n        other : holiday calendar\n        inplace : bool (default=False)\n            If True set rule_table to holidays, else return array of Holidays\n        "
        holidays = self.merge_class(self, other)
        if inplace:
            self.rules = holidays
        else:
            return holidays
USMemorialDay = Holiday('Memorial Day', month=5, day=31, offset=DateOffset(weekday=MO((- 1))))
USLaborDay = Holiday('Labor Day', month=9, day=1, offset=DateOffset(weekday=MO(1)))
USColumbusDay = Holiday('Columbus Day', month=10, day=1, offset=DateOffset(weekday=MO(2)))
USThanksgivingDay = Holiday('Thanksgiving', month=11, day=1, offset=DateOffset(weekday=TH(4)))
USMartinLutherKingJr = Holiday('Martin Luther King Jr. Day', start_date=datetime(1986, 1, 1), month=1, day=1, offset=DateOffset(weekday=MO(3)))
USPresidentsDay = Holiday('Presidents Day', month=2, day=1, offset=DateOffset(weekday=MO(3)))
GoodFriday = Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day((- 2))])
EasterMonday = Holiday('Easter Monday', month=1, day=1, offset=[Easter(), Day(1)])

class USFederalHolidayCalendar(AbstractHolidayCalendar):
    '\n    US Federal Government Holiday Calendar based on rules specified by:\n    https://www.opm.gov/policy-data-oversight/\n       snow-dismissal-procedures/federal-holidays/\n    '
    rules = [Holiday('New Years Day', month=1, day=1, observance=nearest_workday), USMartinLutherKingJr, USPresidentsDay, USMemorialDay, Holiday('July 4th', month=7, day=4, observance=nearest_workday), USLaborDay, USColumbusDay, Holiday('Veterans Day', month=11, day=11, observance=nearest_workday), USThanksgivingDay, Holiday('Christmas', month=12, day=25, observance=nearest_workday)]

def HolidayCalendarFactory(name, base, other, base_class=AbstractHolidayCalendar):
    rules = AbstractHolidayCalendar.merge_class(base, other)
    calendar_class = type(name, (base_class,), {'rules': rules, 'name': name})
    return calendar_class
