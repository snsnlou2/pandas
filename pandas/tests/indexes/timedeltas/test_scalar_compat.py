
'\nTests for TimedeltaIndex methods behaving like their Timedelta counterparts\n'
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import Index, Series, Timedelta, TimedeltaIndex, timedelta_range
import pandas._testing as tm

class TestVectorizedTimedelta():

    def test_tdi_total_seconds(self):
        rng = timedelta_range('1 days, 10:11:12.100123456', periods=2, freq='s')
        expt = [(((((1 * 86400) + (10 * 3600)) + (11 * 60)) + 12) + (100123456.0 / 1000000000.0)), (((((1 * 86400) + (10 * 3600)) + (11 * 60)) + 13) + (100123456.0 / 1000000000.0))]
        tm.assert_almost_equal(rng.total_seconds(), Index(expt))
        ser = Series(rng)
        s_expt = Series(expt, index=[0, 1])
        tm.assert_series_equal(ser.dt.total_seconds(), s_expt)
        ser[1] = np.nan
        s_expt = Series([(((((1 * 86400) + (10 * 3600)) + (11 * 60)) + 12) + (100123456.0 / 1000000000.0)), np.nan], index=[0, 1])
        tm.assert_series_equal(ser.dt.total_seconds(), s_expt)
        ser = Series([np.nan, np.nan], dtype='timedelta64[ns]')
        tm.assert_series_equal(ser.dt.total_seconds(), Series([np.nan, np.nan], index=[0, 1]))

    def test_tdi_round(self):
        td = timedelta_range(start='16801 days', periods=5, freq='30Min')
        elt = td[1]
        expected_rng = TimedeltaIndex([Timedelta('16801 days 00:00:00'), Timedelta('16801 days 00:00:00'), Timedelta('16801 days 01:00:00'), Timedelta('16801 days 02:00:00'), Timedelta('16801 days 02:00:00')])
        expected_elt = expected_rng[1]
        tm.assert_index_equal(td.round(freq='H'), expected_rng)
        assert (elt.round(freq='H') == expected_elt)
        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            td.round(freq='foo')
        with pytest.raises(ValueError, match=msg):
            elt.round(freq='foo')
        msg = '<MonthEnd> is a non-fixed frequency'
        with pytest.raises(ValueError, match=msg):
            td.round(freq='M')
        with pytest.raises(ValueError, match=msg):
            elt.round(freq='M')

    @pytest.mark.parametrize('freq,msg', [('Y', '<YearEnd: month=12> is a non-fixed frequency'), ('M', '<MonthEnd> is a non-fixed frequency'), ('foobar', 'Invalid frequency: foobar')])
    def test_tdi_round_invalid(self, freq, msg):
        t1 = timedelta_range('1 days', periods=3, freq='1 min 2 s 3 us')
        with pytest.raises(ValueError, match=msg):
            t1.round(freq)
        with pytest.raises(ValueError, match=msg):
            t1._data.round(freq)

    def test_round(self):
        t1 = timedelta_range('1 days', periods=3, freq='1 min 2 s 3 us')
        t2 = ((- 1) * t1)
        t1a = timedelta_range('1 days', periods=3, freq='1 min 2 s')
        t1c = TimedeltaIndex([1, 1, 1], unit='D')
        for (freq, s1, s2) in [('N', t1, t2), ('U', t1, t2), ('L', t1a, TimedeltaIndex(['-1 days +00:00:00', '-2 days +23:58:58', '-2 days +23:57:56'])), ('S', t1a, TimedeltaIndex(['-1 days +00:00:00', '-2 days +23:58:58', '-2 days +23:57:56'])), ('12T', t1c, TimedeltaIndex(['-1 days', '-1 days', '-1 days'])), ('H', t1c, TimedeltaIndex(['-1 days', '-1 days', '-1 days'])), ('d', t1c, TimedeltaIndex([(- 1), (- 1), (- 1)], unit='D'))]:
            r1 = t1.round(freq)
            tm.assert_index_equal(r1, s1)
            r2 = t2.round(freq)
            tm.assert_index_equal(r2, s2)

    def test_components(self):
        rng = timedelta_range('1 days, 10:11:12', periods=2, freq='s')
        rng.components
        s = Series(rng)
        s[1] = np.nan
        result = s.dt.components
        assert (not result.iloc[0].isna().all())
        assert result.iloc[1].isna().all()
