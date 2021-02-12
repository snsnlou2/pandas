
'\nBehavioral based tests for offsets and date_range.\n\nThis file is adapted from https://github.com/pandas-dev/pandas/pull/18761 -\nwhich was more ambitious but less idiomatic in its use of Hypothesis.\n\nYou may wish to consult the previous version for inspiration on further\ntests, or when trying to pin down the bugs exposed by the tests below.\n'
import warnings
from hypothesis import assume, given, strategies as st
from hypothesis.extra.dateutil import timezones as dateutil_timezones
from hypothesis.extra.pytz import timezones as pytz_timezones
import pytest
import pytz
import pandas as pd
from pandas import Timestamp
from pandas.tseries.offsets import BMonthBegin, BMonthEnd, BQuarterBegin, BQuarterEnd, BYearBegin, BYearEnd, MonthBegin, MonthEnd, QuarterBegin, QuarterEnd, YearBegin, YearEnd
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    min_dt = Timestamp(1900, 1, 1).to_pydatetime()
    max_dt = Timestamp(1900, 1, 1).to_pydatetime()
gen_date_range = st.builds(pd.date_range, start=st.datetimes(min_value=Timestamp(1900, 1, 1).to_pydatetime(), max_value=Timestamp(2100, 1, 1).to_pydatetime()), periods=st.integers(min_value=2, max_value=100), freq=st.sampled_from('Y Q M D H T s ms us ns'.split()), tz=st.one_of(st.none(), dateutil_timezones(), pytz_timezones()))
gen_random_datetime = st.datetimes(min_value=min_dt, max_value=max_dt, timezones=st.one_of(st.none(), dateutil_timezones(), pytz_timezones()))
gen_yqm_offset = st.one_of(*map(st.from_type, [MonthBegin, MonthEnd, BMonthBegin, BMonthEnd, QuarterBegin, QuarterEnd, BQuarterBegin, BQuarterEnd, YearBegin, YearEnd, BYearBegin, BYearEnd]))

@pytest.mark.arm_slow
@given(gen_random_datetime, gen_yqm_offset)
def test_on_offset_implementations(dt, offset):
    assume((not offset.normalize))
    try:
        compare = ((dt + offset) - offset)
    except pytz.NonExistentTimeError:
        assume(False)
    assert (offset.is_on_offset(dt) == (compare == dt))

@given(gen_yqm_offset)
def test_shift_across_dst(offset):
    assume((not offset.normalize))
    dti = pd.date_range(start='2017-10-30 12:00:00', end='2017-11-06', freq='D', tz='US/Eastern')
    assert (dti.hour == 12).all()
    res = (dti + offset)
    assert (res.hour == 12).all()
