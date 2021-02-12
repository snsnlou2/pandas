
import numpy as np
import pytest
import pandas as pd
from pandas import Categorical, DataFrame, Index, Series
import pandas._testing as tm

class TestConcatAppendCommon():
    '\n    Test common dtype coercion rules between concat and append.\n    '

    def setup_method(self, method):
        dt_data = [pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-02'), pd.Timestamp('2011-01-03')]
        tz_data = [pd.Timestamp('2011-01-01', tz='US/Eastern'), pd.Timestamp('2011-01-02', tz='US/Eastern'), pd.Timestamp('2011-01-03', tz='US/Eastern')]
        td_data = [pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Timedelta('3 days')]
        period_data = [pd.Period('2011-01', freq='M'), pd.Period('2011-02', freq='M'), pd.Period('2011-03', freq='M')]
        self.data = {'bool': [True, False, True], 'int64': [1, 2, 3], 'float64': [1.1, np.nan, 3.3], 'category': Categorical(['X', 'Y', 'Z']), 'object': ['a', 'b', 'c'], 'datetime64[ns]': dt_data, 'datetime64[ns, US/Eastern]': tz_data, 'timedelta64[ns]': td_data, 'period[M]': period_data}

    def _check_expected_dtype(self, obj, label):
        '\n        Check whether obj has expected dtype depending on label\n        considering not-supported dtypes\n        '
        if isinstance(obj, Index):
            if (label == 'bool'):
                assert (obj.dtype == 'object')
            else:
                assert (obj.dtype == label)
        elif isinstance(obj, Series):
            if label.startswith('period'):
                assert (obj.dtype == 'Period[M]')
            else:
                assert (obj.dtype == label)
        else:
            raise ValueError

    def test_dtypes(self):
        for (typ, vals) in self.data.items():
            self._check_expected_dtype(Index(vals), typ)
            self._check_expected_dtype(Series(vals), typ)

    def test_concatlike_same_dtypes(self):
        for (typ1, vals1) in self.data.items():
            vals2 = vals1
            vals3 = vals1
            if (typ1 == 'category'):
                exp_data = Categorical((list(vals1) + list(vals2)))
                exp_data3 = Categorical(((list(vals1) + list(vals2)) + list(vals3)))
            else:
                exp_data = (vals1 + vals2)
                exp_data3 = ((vals1 + vals2) + vals3)
            res = Index(vals1).append(Index(vals2))
            exp = Index(exp_data)
            tm.assert_index_equal(res, exp)
            res = Index(vals1).append([Index(vals2), Index(vals3)])
            exp = Index(exp_data3)
            tm.assert_index_equal(res, exp)
            i1 = Index(vals1, name='x')
            i2 = Index(vals2, name='y')
            res = i1.append(i2)
            exp = Index(exp_data)
            tm.assert_index_equal(res, exp)
            i1 = Index(vals1, name='x')
            i2 = Index(vals2, name='x')
            res = i1.append(i2)
            exp = Index(exp_data, name='x')
            tm.assert_index_equal(res, exp)
            with pytest.raises(TypeError, match='all inputs must be Index'):
                Index(vals1).append(vals2)
            with pytest.raises(TypeError, match='all inputs must be Index'):
                Index(vals1).append([Index(vals2), vals3])
            res = Series(vals1).append(Series(vals2), ignore_index=True)
            exp = Series(exp_data)
            tm.assert_series_equal(res, exp, check_index_type=True)
            res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
            tm.assert_series_equal(res, exp, check_index_type=True)
            res = Series(vals1).append([Series(vals2), Series(vals3)], ignore_index=True)
            exp = Series(exp_data3)
            tm.assert_series_equal(res, exp)
            res = pd.concat([Series(vals1), Series(vals2), Series(vals3)], ignore_index=True)
            tm.assert_series_equal(res, exp)
            s1 = Series(vals1, name='x')
            s2 = Series(vals2, name='y')
            res = s1.append(s2, ignore_index=True)
            exp = Series(exp_data)
            tm.assert_series_equal(res, exp, check_index_type=True)
            res = pd.concat([s1, s2], ignore_index=True)
            tm.assert_series_equal(res, exp, check_index_type=True)
            s1 = Series(vals1, name='x')
            s2 = Series(vals2, name='x')
            res = s1.append(s2, ignore_index=True)
            exp = Series(exp_data, name='x')
            tm.assert_series_equal(res, exp, check_index_type=True)
            res = pd.concat([s1, s2], ignore_index=True)
            tm.assert_series_equal(res, exp, check_index_type=True)
            msg = "cannot concatenate object of type '.+'; only Series and DataFrame objs are valid"
            with pytest.raises(TypeError, match=msg):
                Series(vals1).append(vals2)
            with pytest.raises(TypeError, match=msg):
                Series(vals1).append([Series(vals2), vals3])
            with pytest.raises(TypeError, match=msg):
                pd.concat([Series(vals1), vals2])
            with pytest.raises(TypeError, match=msg):
                pd.concat([Series(vals1), Series(vals2), vals3])

    def test_concatlike_dtypes_coercion(self):
        for (typ1, vals1) in self.data.items():
            for (typ2, vals2) in self.data.items():
                vals3 = vals2
                exp_index_dtype = None
                exp_series_dtype = None
                if (typ1 == typ2):
                    continue
                elif ((typ1 == 'category') or (typ2 == 'category')):
                    continue
                if ((typ1 == 'bool') and (typ2 in ('int64', 'float64'))):
                    exp_series_dtype = typ2
                elif ((typ2 == 'bool') and (typ1 in ('int64', 'float64'))):
                    exp_series_dtype = typ1
                elif ((typ1 == 'datetime64[ns, US/Eastern]') or (typ2 == 'datetime64[ns, US/Eastern]') or (typ1 == 'timedelta64[ns]') or (typ2 == 'timedelta64[ns]')):
                    exp_index_dtype = object
                    exp_series_dtype = object
                exp_data = (vals1 + vals2)
                exp_data3 = ((vals1 + vals2) + vals3)
                res = Index(vals1).append(Index(vals2))
                exp = Index(exp_data, dtype=exp_index_dtype)
                tm.assert_index_equal(res, exp)
                res = Index(vals1).append([Index(vals2), Index(vals3)])
                exp = Index(exp_data3, dtype=exp_index_dtype)
                tm.assert_index_equal(res, exp)
                res = Series(vals1).append(Series(vals2), ignore_index=True)
                exp = Series(exp_data, dtype=exp_series_dtype)
                tm.assert_series_equal(res, exp, check_index_type=True)
                res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
                tm.assert_series_equal(res, exp, check_index_type=True)
                res = Series(vals1).append([Series(vals2), Series(vals3)], ignore_index=True)
                exp = Series(exp_data3, dtype=exp_series_dtype)
                tm.assert_series_equal(res, exp)
                res = pd.concat([Series(vals1), Series(vals2), Series(vals3)], ignore_index=True)
                tm.assert_series_equal(res, exp)

    def test_concatlike_common_coerce_to_pandas_object(self):
        dti = pd.DatetimeIndex(['2011-01-01', '2011-01-02'])
        tdi = pd.TimedeltaIndex(['1 days', '2 days'])
        exp = Index([pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-02'), pd.Timedelta('1 days'), pd.Timedelta('2 days')])
        res = dti.append(tdi)
        tm.assert_index_equal(res, exp)
        assert isinstance(res[0], pd.Timestamp)
        assert isinstance(res[(- 1)], pd.Timedelta)
        dts = Series(dti)
        tds = Series(tdi)
        res = dts.append(tds)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        assert isinstance(res.iloc[0], pd.Timestamp)
        assert isinstance(res.iloc[(- 1)], pd.Timedelta)
        res = pd.concat([dts, tds])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        assert isinstance(res.iloc[0], pd.Timestamp)
        assert isinstance(res.iloc[(- 1)], pd.Timedelta)

    def test_concatlike_datetimetz(self, tz_aware_fixture):
        tz = tz_aware_fixture
        dti1 = pd.DatetimeIndex(['2011-01-01', '2011-01-02'], tz=tz)
        dti2 = pd.DatetimeIndex(['2012-01-01', '2012-01-02'], tz=tz)
        exp = pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2012-01-01', '2012-01-02'], tz=tz)
        res = dti1.append(dti2)
        tm.assert_index_equal(res, exp)
        dts1 = Series(dti1)
        dts2 = Series(dti2)
        res = dts1.append(dts2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        res = pd.concat([dts1, dts2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    @pytest.mark.parametrize('tz', ['UTC', 'US/Eastern', 'Asia/Tokyo', 'EST5EDT'])
    def test_concatlike_datetimetz_short(self, tz):
        ix1 = pd.date_range(start='2014-07-15', end='2014-07-17', freq='D', tz=tz)
        ix2 = pd.DatetimeIndex(['2014-07-11', '2014-07-21'], tz=tz)
        df1 = DataFrame(0, index=ix1, columns=['A', 'B'])
        df2 = DataFrame(0, index=ix2, columns=['A', 'B'])
        exp_idx = pd.DatetimeIndex(['2014-07-15', '2014-07-16', '2014-07-17', '2014-07-11', '2014-07-21'], tz=tz)
        exp = DataFrame(0, index=exp_idx, columns=['A', 'B'])
        tm.assert_frame_equal(df1.append(df2), exp)
        tm.assert_frame_equal(pd.concat([df1, df2]), exp)

    def test_concatlike_datetimetz_to_object(self, tz_aware_fixture):
        tz = tz_aware_fixture
        dti1 = pd.DatetimeIndex(['2011-01-01', '2011-01-02'], tz=tz)
        dti2 = pd.DatetimeIndex(['2012-01-01', '2012-01-02'])
        exp = Index([pd.Timestamp('2011-01-01', tz=tz), pd.Timestamp('2011-01-02', tz=tz), pd.Timestamp('2012-01-01'), pd.Timestamp('2012-01-02')], dtype=object)
        res = dti1.append(dti2)
        tm.assert_index_equal(res, exp)
        dts1 = Series(dti1)
        dts2 = Series(dti2)
        res = dts1.append(dts2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        res = pd.concat([dts1, dts2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        dti3 = pd.DatetimeIndex(['2012-01-01', '2012-01-02'], tz='US/Pacific')
        exp = Index([pd.Timestamp('2011-01-01', tz=tz), pd.Timestamp('2011-01-02', tz=tz), pd.Timestamp('2012-01-01', tz='US/Pacific'), pd.Timestamp('2012-01-02', tz='US/Pacific')], dtype=object)
        res = dti1.append(dti3)
        dts1 = Series(dti1)
        dts3 = Series(dti3)
        res = dts1.append(dts3)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        res = pd.concat([dts1, dts3])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period(self):
        pi1 = pd.PeriodIndex(['2011-01', '2011-02'], freq='M')
        pi2 = pd.PeriodIndex(['2012-01', '2012-02'], freq='M')
        exp = pd.PeriodIndex(['2011-01', '2011-02', '2012-01', '2012-02'], freq='M')
        res = pi1.append(pi2)
        tm.assert_index_equal(res, exp)
        ps1 = Series(pi1)
        ps2 = Series(pi2)
        res = ps1.append(ps2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        res = pd.concat([ps1, ps2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period_diff_freq_to_object(self):
        pi1 = pd.PeriodIndex(['2011-01', '2011-02'], freq='M')
        pi2 = pd.PeriodIndex(['2012-01-01', '2012-02-01'], freq='D')
        exp = Index([pd.Period('2011-01', freq='M'), pd.Period('2011-02', freq='M'), pd.Period('2012-01-01', freq='D'), pd.Period('2012-02-01', freq='D')], dtype=object)
        res = pi1.append(pi2)
        tm.assert_index_equal(res, exp)
        ps1 = Series(pi1)
        ps2 = Series(pi2)
        res = ps1.append(ps2)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        res = pd.concat([ps1, ps2])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concatlike_common_period_mixed_dt_to_object(self):
        pi1 = pd.PeriodIndex(['2011-01', '2011-02'], freq='M')
        tdi = pd.TimedeltaIndex(['1 days', '2 days'])
        exp = Index([pd.Period('2011-01', freq='M'), pd.Period('2011-02', freq='M'), pd.Timedelta('1 days'), pd.Timedelta('2 days')], dtype=object)
        res = pi1.append(tdi)
        tm.assert_index_equal(res, exp)
        ps1 = Series(pi1)
        tds = Series(tdi)
        res = ps1.append(tds)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        res = pd.concat([ps1, tds])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        exp = Index([pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Period('2011-01', freq='M'), pd.Period('2011-02', freq='M')], dtype=object)
        res = tdi.append(pi1)
        tm.assert_index_equal(res, exp)
        ps1 = Series(pi1)
        tds = Series(tdi)
        res = tds.append(ps1)
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
        res = pd.concat([tds, ps1])
        tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))

    def test_concat_categorical(self):
        s1 = Series([1, 2, np.nan], dtype='category')
        s2 = Series([2, 1, 2], dtype='category')
        exp = Series([1, 2, np.nan, 2, 1, 2], dtype='category')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        s1 = Series([3, 2], dtype='category')
        s2 = Series([2, 1], dtype='category')
        exp = Series([3, 2, 2, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        s1 = Series([10, 11, np.nan], dtype='category')
        s2 = Series([np.nan, 1, 3, 2], dtype='category')
        exp = Series([10, 11, np.nan, np.nan, 1, 3, 2], dtype='object')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)

    def test_union_categorical_same_categories_different_order(self):
        a = Series(Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c']))
        b = Series(Categorical(['a', 'b', 'c'], categories=['b', 'a', 'c']))
        result = pd.concat([a, b], ignore_index=True)
        expected = Series(Categorical(['a', 'b', 'c', 'a', 'b', 'c'], categories=['a', 'b', 'c']))
        tm.assert_series_equal(result, expected)

    def test_concat_categorical_coercion(self):
        s1 = Series([1, 2, np.nan], dtype='category')
        s2 = Series([2, 1, 2])
        exp = Series([1, 2, np.nan, 2, 1, 2], dtype='object')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        exp = Series([2, 1, 2, 1, 2, np.nan], dtype='object')
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), exp)
        s1 = Series([3, 2], dtype='category')
        s2 = Series([2, 1])
        exp = Series([3, 2, 2, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        exp = Series([2, 1, 3, 2])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), exp)
        s1 = Series([10, 11, np.nan], dtype='category')
        s2 = Series([1, 3, 2])
        exp = Series([10, 11, np.nan, 1, 3, 2], dtype='object')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        exp = Series([1, 3, 2, 10, 11, np.nan], dtype='object')
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), exp)
        s1 = Series([10, 11, np.nan], dtype='category')
        s2 = Series(['a', 'b', 'c'])
        exp = Series([10, 11, np.nan, 'a', 'b', 'c'])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        exp = Series(['a', 'b', 'c', 10, 11, np.nan])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), exp)
        s1 = Series([10, 11], dtype='category')
        s2 = Series([np.nan, np.nan, np.nan])
        exp = Series([10, 11, np.nan, np.nan, np.nan])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        exp = Series([np.nan, np.nan, np.nan, 10, 11])
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), exp)

    def test_concat_categorical_3elem_coercion(self):
        s1 = Series([1, 2, np.nan], dtype='category')
        s2 = Series([2, 1, 2], dtype='category')
        s3 = Series([1, 2, 1, 2, np.nan])
        exp = Series([1, 2, np.nan, 2, 1, 2, 1, 2, 1, 2, np.nan], dtype='float')
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        tm.assert_series_equal(s1.append([s2, s3], ignore_index=True), exp)
        exp = Series([1, 2, 1, 2, np.nan, 1, 2, np.nan, 2, 1, 2], dtype='float')
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s3.append([s1, s2], ignore_index=True), exp)
        s1 = Series([4, 5, 6], dtype='category')
        s2 = Series([1, 2, 3], dtype='category')
        s3 = Series([1, 3, 4])
        exp = Series([4, 5, 6, 1, 2, 3, 1, 3, 4])
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        tm.assert_series_equal(s1.append([s2, s3], ignore_index=True), exp)
        exp = Series([1, 3, 4, 4, 5, 6, 1, 2, 3])
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s3.append([s1, s2], ignore_index=True), exp)
        s1 = Series([4, 5, 6], dtype='category')
        s2 = Series([1, 2, 3], dtype='category')
        s3 = Series([10, 11, 12])
        exp = Series([4, 5, 6, 1, 2, 3, 10, 11, 12])
        tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
        tm.assert_series_equal(s1.append([s2, s3], ignore_index=True), exp)
        exp = Series([10, 11, 12, 4, 5, 6, 1, 2, 3])
        tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s3.append([s1, s2], ignore_index=True), exp)

    def test_concat_categorical_multi_coercion(self):
        s1 = Series([1, 3], dtype='category')
        s2 = Series([3, 4], dtype='category')
        s3 = Series([2, 3])
        s4 = Series([2, 2], dtype='category')
        s5 = Series([1, np.nan])
        s6 = Series([1, 3, 2], dtype='category')
        exp = Series([1, 3, 3, 4, 2, 3, 2, 2, 1, np.nan, 1, 3, 2])
        res = pd.concat([s1, s2, s3, s4, s5, s6], ignore_index=True)
        tm.assert_series_equal(res, exp)
        res = s1.append([s2, s3, s4, s5, s6], ignore_index=True)
        tm.assert_series_equal(res, exp)
        exp = Series([1, 3, 2, 1, np.nan, 2, 2, 2, 3, 3, 4, 1, 3])
        res = pd.concat([s6, s5, s4, s3, s2, s1], ignore_index=True)
        tm.assert_series_equal(res, exp)
        res = s6.append([s5, s4, s3, s2, s1], ignore_index=True)
        tm.assert_series_equal(res, exp)

    def test_concat_categorical_ordered(self):
        s1 = Series(Categorical([1, 2, np.nan], ordered=True))
        s2 = Series(Categorical([2, 1, 2], ordered=True))
        exp = Series(Categorical([1, 2, np.nan, 2, 1, 2], ordered=True))
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        exp = Series(Categorical([1, 2, np.nan, 2, 1, 2, 1, 2, np.nan], ordered=True))
        tm.assert_series_equal(pd.concat([s1, s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s1.append([s2, s1], ignore_index=True), exp)

    def test_concat_categorical_coercion_nan(self):
        s1 = Series(np.array([np.nan, np.nan], dtype=np.float64), dtype='category')
        s2 = Series([np.nan, 1])
        exp = Series([np.nan, np.nan, np.nan, 1])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        s1 = Series([1, np.nan], dtype='category')
        s2 = Series([np.nan, np.nan])
        exp = Series([1, np.nan, np.nan, np.nan], dtype='float')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        s1 = Series([np.nan, np.nan], dtype='category')
        s2 = Series([np.nan, np.nan])
        exp = Series([np.nan, np.nan, np.nan, np.nan])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), exp)
        s1 = Series([np.nan, np.nan], dtype='category')
        s2 = Series([np.nan, np.nan], dtype='category')
        exp = Series([np.nan, np.nan, np.nan, np.nan], dtype='category')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)

    def test_concat_categorical_empty(self):
        s1 = Series([], dtype='category')
        s2 = Series([1, 2], dtype='category')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), s2)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), s2)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), s2)
        s1 = Series([], dtype='category')
        s2 = Series([], dtype='category')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), s2)
        s1 = Series([], dtype='category')
        s2 = Series([], dtype='object')
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), s2)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), s2)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), s2)
        s1 = Series([], dtype='category')
        s2 = Series([np.nan, np.nan])
        exp = Series([np.nan, np.nan])
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1.append(s2, ignore_index=True), exp)
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2.append(s1, ignore_index=True), exp)

    def test_categorical_concat_append(self):
        cat = Categorical(['a', 'b'], categories=['a', 'b'])
        vals = [1, 2]
        df = DataFrame({'cats': cat, 'vals': vals})
        cat2 = Categorical(['a', 'b', 'a', 'b'], categories=['a', 'b'])
        vals2 = [1, 2, 1, 2]
        exp = DataFrame({'cats': cat2, 'vals': vals2}, index=Index([0, 1, 0, 1]))
        tm.assert_frame_equal(pd.concat([df, df]), exp)
        tm.assert_frame_equal(df.append(df), exp)
        cat3 = Categorical(['a', 'b'], categories=['a', 'b', 'c'])
        vals3 = [1, 2]
        df_different_categories = DataFrame({'cats': cat3, 'vals': vals3})
        res = pd.concat([df, df_different_categories], ignore_index=True)
        exp = DataFrame({'cats': list('abab'), 'vals': [1, 2, 1, 2]})
        tm.assert_frame_equal(res, exp)
        res = df.append(df_different_categories, ignore_index=True)
        tm.assert_frame_equal(res, exp)
