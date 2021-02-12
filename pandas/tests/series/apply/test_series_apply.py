
from collections import Counter, defaultdict
from itertools import chain
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, isna, timedelta_range
import pandas._testing as tm
from pandas.core.base import SpecificationError

class TestSeriesApply():

    def test_series_map_box_timedelta(self):
        ser = Series(timedelta_range('1 day 1 s', periods=5, freq='h'))

        def f(x):
            return x.total_seconds()
        ser.map(f)
        ser.apply(f)
        DataFrame(ser).applymap(f)

    def test_apply(self, datetime_series):
        with np.errstate(all='ignore'):
            tm.assert_series_equal(datetime_series.apply(np.sqrt), np.sqrt(datetime_series))
            import math
            tm.assert_series_equal(datetime_series.apply(math.exp), np.exp(datetime_series))
        s = Series(dtype=object, name='foo', index=Index([], name='bar'))
        rs = s.apply((lambda x: x))
        tm.assert_series_equal(s, rs)
        assert (s is not rs)
        assert (s.index is rs.index)
        assert (s.dtype == rs.dtype)
        assert (s.name == rs.name)
        s = Series(index=[1, 2, 3], dtype=np.float64)
        rs = s.apply((lambda x: x))
        tm.assert_series_equal(s, rs)

    def test_apply_same_length_inference_bug(self):
        s = Series([1, 2])

        def f(x):
            return (x, (x + 1))
        result = s.apply(f)
        expected = s.map(f)
        tm.assert_series_equal(result, expected)
        s = Series([1, 2, 3])
        result = s.apply(f)
        expected = s.map(f)
        tm.assert_series_equal(result, expected)

    def test_apply_dont_convert_dtype(self):
        s = Series(np.random.randn(10))

        def f(x):
            return (x if (x > 0) else np.nan)
        result = s.apply(f, convert_dtype=False)
        assert (result.dtype == object)

    def test_with_string_args(self, datetime_series):
        for arg in ['sum', 'mean', 'min', 'max', 'std']:
            result = datetime_series.apply(arg)
            expected = getattr(datetime_series, arg)()
            assert (result == expected)

    def test_apply_args(self):
        s = Series(['foo,bar'])
        result = s.apply(str.split, args=(',',))
        assert (result[0] == ['foo', 'bar'])
        assert isinstance(result[0], list)

    def test_series_map_box_timestamps(self):
        ser = Series(pd.date_range('1/1/2000', periods=10))

        def func(x):
            return (x.hour, x.day, x.month)
        ser.map(func)
        ser.apply(func)

    def test_apply_box(self):
        vals = [pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-02')]
        s = Series(vals)
        assert (s.dtype == 'datetime64[ns]')
        res = s.apply((lambda x: f'{type(x).__name__}_{x.day}_{x.tz}'))
        exp = Series(['Timestamp_1_None', 'Timestamp_2_None'])
        tm.assert_series_equal(res, exp)
        vals = [pd.Timestamp('2011-01-01', tz='US/Eastern'), pd.Timestamp('2011-01-02', tz='US/Eastern')]
        s = Series(vals)
        assert (s.dtype == 'datetime64[ns, US/Eastern]')
        res = s.apply((lambda x: f'{type(x).__name__}_{x.day}_{x.tz}'))
        exp = Series(['Timestamp_1_US/Eastern', 'Timestamp_2_US/Eastern'])
        tm.assert_series_equal(res, exp)
        vals = [pd.Timedelta('1 days'), pd.Timedelta('2 days')]
        s = Series(vals)
        assert (s.dtype == 'timedelta64[ns]')
        res = s.apply((lambda x: f'{type(x).__name__}_{x.days}'))
        exp = Series(['Timedelta_1', 'Timedelta_2'])
        tm.assert_series_equal(res, exp)
        vals = [pd.Period('2011-01-01', freq='M'), pd.Period('2011-01-02', freq='M')]
        s = Series(vals)
        assert (s.dtype == 'Period[M]')
        res = s.apply((lambda x: f'{type(x).__name__}_{x.freqstr}'))
        exp = Series(['Period_M', 'Period_M'])
        tm.assert_series_equal(res, exp)

    def test_apply_datetimetz(self):
        values = pd.date_range('2011-01-01', '2011-01-02', freq='H').tz_localize('Asia/Tokyo')
        s = Series(values, name='XX')
        result = s.apply((lambda x: (x + pd.offsets.Day())))
        exp_values = pd.date_range('2011-01-02', '2011-01-03', freq='H').tz_localize('Asia/Tokyo')
        exp = Series(exp_values, name='XX')
        tm.assert_series_equal(result, exp)
        result = s.apply((lambda x: x.hour))
        exp = Series((list(range(24)) + [0]), name='XX', dtype=np.int64)
        tm.assert_series_equal(result, exp)

        def f(x):
            if (not isinstance(x, pd.Timestamp)):
                raise ValueError
            return str(x.tz)
        result = s.map(f)
        exp = Series((['Asia/Tokyo'] * 25), name='XX')
        tm.assert_series_equal(result, exp)

    def test_apply_dict_depr(self):
        tsdf = DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'], index=pd.date_range('1/1/2000', periods=10))
        msg = 'nested renamer is not supported'
        with pytest.raises(SpecificationError, match=msg):
            tsdf.A.agg({'foo': ['sum', 'mean']})

    def test_apply_categorical(self):
        values = pd.Categorical(list('ABBABCD'), categories=list('DCBA'), ordered=True)
        ser = Series(values, name='XX', index=list('abcdefg'))
        result = ser.apply((lambda x: x.lower()))
        values = pd.Categorical(list('abbabcd'), categories=list('dcba'), ordered=True)
        exp = Series(values, name='XX', index=list('abcdefg'))
        tm.assert_series_equal(result, exp)
        tm.assert_categorical_equal(result.values, exp.values)
        result = ser.apply((lambda x: 'A'))
        exp = Series((['A'] * 7), name='XX', index=list('abcdefg'))
        tm.assert_series_equal(result, exp)
        assert (result.dtype == object)

    @pytest.mark.parametrize('series', [['1-1', '1-1', np.NaN], ['1-1', '1-2', np.NaN]])
    def test_apply_categorical_with_nan_values(self, series):
        s = Series(series, dtype='category')
        result = s.apply((lambda x: x.split('-')[0]))
        result = result.astype(object)
        expected = Series(['1', '1', np.NaN], dtype='category')
        expected = expected.astype(object)
        tm.assert_series_equal(result, expected)

    def test_apply_empty_integer_series_with_datetime_index(self):
        s = Series([], index=pd.date_range(start='2018-01-01', periods=0), dtype=int)
        result = s.apply((lambda x: x))
        tm.assert_series_equal(result, s)

class TestSeriesAggregate():

    def test_transform(self, string_series):
        with np.errstate(all='ignore'):
            f_sqrt = np.sqrt(string_series)
            f_abs = np.abs(string_series)
            result = string_series.apply(np.sqrt)
            expected = f_sqrt.copy()
            tm.assert_series_equal(result, expected)
            result = string_series.apply([np.sqrt])
            expected = f_sqrt.to_frame().copy()
            expected.columns = ['sqrt']
            tm.assert_frame_equal(result, expected)
            result = string_series.apply(['sqrt'])
            tm.assert_frame_equal(result, expected)
            expected = pd.concat([f_sqrt, f_abs], axis=1)
            expected.columns = ['sqrt', 'absolute']
            result = string_series.apply([np.sqrt, np.abs])
            tm.assert_frame_equal(result, expected)
            expected = pd.concat([f_sqrt, f_abs], axis=1)
            expected.columns = ['foo', 'bar']
            expected = expected.unstack().rename('series')
            result = string_series.apply({'foo': np.sqrt, 'bar': np.abs})
            tm.assert_series_equal(result.reindex_like(expected), expected)

    def test_transform_and_agg_error(self, string_series):
        msg = 'cannot combine transform and aggregation'
        with pytest.raises(ValueError, match=msg):
            with np.errstate(all='ignore'):
                string_series.agg(['sqrt', 'max'])
        msg = 'cannot perform both aggregation and transformation'
        with pytest.raises(ValueError, match=msg):
            with np.errstate(all='ignore'):
                string_series.agg({'foo': np.sqrt, 'bar': 'sum'})

    def test_demo(self):
        s = Series(range(6), dtype='int64', name='series')
        result = s.agg(['min', 'max'])
        expected = Series([0, 5], index=['min', 'max'], name='series')
        tm.assert_series_equal(result, expected)
        result = s.agg({'foo': 'min'})
        expected = Series([0], index=['foo'], name='series')
        tm.assert_series_equal(result, expected)
        msg = 'nested renamer is not supported'
        with pytest.raises(SpecificationError, match=msg):
            s.agg({'foo': ['min', 'max']})

    def test_multiple_aggregators_with_dict_api(self):
        s = Series(range(6), dtype='int64', name='series')
        msg = 'nested renamer is not supported'
        with pytest.raises(SpecificationError, match=msg):
            s.agg({'foo': ['min', 'max'], 'bar': ['sum', 'mean']})

    def test_agg_apply_evaluate_lambdas_the_same(self, string_series):
        result = string_series.apply((lambda x: str(x)))
        expected = string_series.agg((lambda x: str(x)))
        tm.assert_series_equal(result, expected)
        result = string_series.apply(str)
        expected = string_series.agg(str)
        tm.assert_series_equal(result, expected)

    def test_with_nested_series(self, datetime_series):
        result = datetime_series.apply((lambda x: Series([x, (x ** 2)], index=['x', 'x^2'])))
        expected = DataFrame({'x': datetime_series, 'x^2': (datetime_series ** 2)})
        tm.assert_frame_equal(result, expected)
        result = datetime_series.agg((lambda x: Series([x, (x ** 2)], index=['x', 'x^2'])))
        tm.assert_frame_equal(result, expected)

    def test_replicate_describe(self, string_series):
        expected = string_series.describe()
        result = string_series.apply({'count': 'count', 'mean': 'mean', 'std': 'std', 'min': 'min', '25%': (lambda x: x.quantile(0.25)), '50%': 'median', '75%': (lambda x: x.quantile(0.75)), 'max': 'max'})
        tm.assert_series_equal(result, expected)

    def test_reduce(self, string_series):
        result = string_series.agg(['sum', 'mean'])
        expected = Series([string_series.sum(), string_series.mean()], ['sum', 'mean'], name=string_series.name)
        tm.assert_series_equal(result, expected)

    def test_non_callable_aggregates(self):
        s = Series([1, 2, None])
        result = s.agg('size')
        expected = s.size
        assert (result == expected)
        result = s.agg(['size', 'count', 'mean'])
        expected = Series({'size': 3.0, 'count': 2.0, 'mean': 1.5})
        tm.assert_series_equal(result[expected.index], expected)

    @pytest.mark.parametrize('series, func, expected', chain(tm.get_cython_table_params(Series(dtype=np.float64), [('sum', 0), ('max', np.nan), ('min', np.nan), ('all', True), ('any', False), ('mean', np.nan), ('prod', 1), ('std', np.nan), ('var', np.nan), ('median', np.nan)]), tm.get_cython_table_params(Series([np.nan, 1, 2, 3]), [('sum', 6), ('max', 3), ('min', 1), ('all', True), ('any', True), ('mean', 2), ('prod', 6), ('std', 1), ('var', 1), ('median', 2)]), tm.get_cython_table_params(Series('a b c'.split()), [('sum', 'abc'), ('max', 'c'), ('min', 'a'), ('all', 'c'), ('any', 'a')])))
    def test_agg_cython_table(self, series, func, expected):
        result = series.agg(func)
        if is_number(expected):
            assert np.isclose(result, expected, equal_nan=True)
        else:
            assert (result == expected)

    @pytest.mark.parametrize('series, func, expected', chain(tm.get_cython_table_params(Series(dtype=np.float64), [('cumprod', Series([], Index([]), dtype=np.float64)), ('cumsum', Series([], Index([]), dtype=np.float64))]), tm.get_cython_table_params(Series([np.nan, 1, 2, 3]), [('cumprod', Series([np.nan, 1, 2, 6])), ('cumsum', Series([np.nan, 1, 3, 6]))]), tm.get_cython_table_params(Series('a b c'.split()), [('cumsum', Series(['a', 'ab', 'abc']))])))
    def test_agg_cython_table_transform(self, series, func, expected):
        result = series.agg(func)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('series, func, expected', chain(tm.get_cython_table_params(Series('a b c'.split()), [('mean', TypeError), ('prod', TypeError), ('std', TypeError), ('var', TypeError), ('median', TypeError), ('cumprod', TypeError)])))
    def test_agg_cython_table_raises(self, series, func, expected):
        msg = "[Cc]ould not convert|can't multiply sequence by non-int of type"
        with pytest.raises(expected, match=msg):
            series.agg(func)

    def test_series_apply_no_suffix_index(self):
        s = Series(([4] * 3))
        result = s.apply(['sum', (lambda x: x.sum()), (lambda x: x.sum())])
        expected = Series([12, 12, 12], index=['sum', '<lambda>', '<lambda>'])
        tm.assert_series_equal(result, expected)

class TestSeriesMap():

    def test_map(self, datetime_series):
        (index, data) = tm.getMixedTypeDict()
        source = Series(data['B'], index=data['C'])
        target = Series(data['C'][:4], index=data['D'][:4])
        merged = target.map(source)
        for (k, v) in merged.items():
            assert (v == source[target[k]])
        merged = target.map(source.to_dict())
        for (k, v) in merged.items():
            assert (v == source[target[k]])
        result = datetime_series.map((lambda x: (x * 2)))
        tm.assert_series_equal(result, (datetime_series * 2))
        a = Series([1, 2, 3, 4])
        b = Series(['even', 'odd', 'even', 'odd'], dtype='category')
        c = Series(['even', 'odd', 'even', 'odd'])
        exp = Series(['odd', 'even', 'odd', np.nan], dtype='category')
        tm.assert_series_equal(a.map(b), exp)
        exp = Series(['odd', 'even', 'odd', np.nan])
        tm.assert_series_equal(a.map(c), exp)
        a = Series(['a', 'b', 'c', 'd'])
        b = Series([1, 2, 3, 4], index=pd.CategoricalIndex(['b', 'c', 'd', 'e']))
        c = Series([1, 2, 3, 4], index=Index(['b', 'c', 'd', 'e']))
        exp = Series([np.nan, 1, 2, 3])
        tm.assert_series_equal(a.map(b), exp)
        exp = Series([np.nan, 1, 2, 3])
        tm.assert_series_equal(a.map(c), exp)
        a = Series(['a', 'b', 'c', 'd'])
        b = Series(['B', 'C', 'D', 'E'], dtype='category', index=pd.CategoricalIndex(['b', 'c', 'd', 'e']))
        c = Series(['B', 'C', 'D', 'E'], index=Index(['b', 'c', 'd', 'e']))
        exp = Series(pd.Categorical([np.nan, 'B', 'C', 'D'], categories=['B', 'C', 'D', 'E']))
        tm.assert_series_equal(a.map(b), exp)
        exp = Series([np.nan, 'B', 'C', 'D'])
        tm.assert_series_equal(a.map(c), exp)

    def test_map_empty(self, index):
        if isinstance(index, MultiIndex):
            pytest.skip('Initializing a Series from a MultiIndex is not supported')
        s = Series(index)
        result = s.map({})
        expected = Series(np.nan, index=s.index)
        tm.assert_series_equal(result, expected)

    def test_map_compat(self):
        s = Series([True, True, False], index=[1, 2, 3])
        result = s.map({True: 'foo', False: 'bar'})
        expected = Series(['foo', 'foo', 'bar'], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_map_int(self):
        left = Series({'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4})
        right = Series({1: 11, 2: 22, 3: 33})
        assert (left.dtype == np.float_)
        assert issubclass(right.dtype.type, np.integer)
        merged = left.map(right)
        assert (merged.dtype == np.float_)
        assert isna(merged['d'])
        assert (not isna(merged['c']))

    def test_map_type_inference(self):
        s = Series(range(3))
        s2 = s.map((lambda x: np.where((x == 0), 0, 1)))
        assert issubclass(s2.dtype.type, np.integer)

    def test_map_decimal(self, string_series):
        from decimal import Decimal
        result = string_series.map((lambda x: Decimal(str(x))))
        assert (result.dtype == np.object_)
        assert isinstance(result[0], Decimal)

    def test_map_na_exclusion(self):
        s = Series([1.5, np.nan, 3, np.nan, 5])
        result = s.map((lambda x: (x * 2)), na_action='ignore')
        exp = (s * 2)
        tm.assert_series_equal(result, exp)

    def test_map_dict_with_tuple_keys(self):
        '\n        Due to new MultiIndex-ing behaviour in v0.14.0,\n        dicts with tuple keys passed to map were being\n        converted to a multi-index, preventing tuple values\n        from being mapped properly.\n        '
        df = DataFrame({'a': [(1,), (2,), (3, 4), (5, 6)]})
        label_mappings = {(1,): 'A', (2,): 'B', (3, 4): 'A', (5, 6): 'B'}
        df['labels'] = df['a'].map(label_mappings)
        df['expected_labels'] = Series(['A', 'B', 'A', 'B'], index=df.index)
        tm.assert_series_equal(df['labels'], df['expected_labels'], check_names=False)

    def test_map_counter(self):
        s = Series(['a', 'b', 'c'], index=[1, 2, 3])
        counter = Counter()
        counter['b'] = 5
        counter['c'] += 1
        result = s.map(counter)
        expected = Series([0, 5, 1], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_map_defaultdict(self):
        s = Series([1, 2, 3], index=['a', 'b', 'c'])
        default_dict = defaultdict((lambda : 'blank'))
        default_dict[1] = 'stuff'
        result = s.map(default_dict)
        expected = Series(['stuff', 'blank', 'blank'], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)

    def test_map_dict_na_key(self):
        s = Series([1, 2, np.nan])
        expected = Series(['a', 'b', 'c'])
        result = s.map({1: 'a', 2: 'b', np.nan: 'c'})
        tm.assert_series_equal(result, expected)

    def test_map_dict_subclass_with_missing(self):
        '\n        Test Series.map with a dictionary subclass that defines __missing__,\n        i.e. sets a default value (GH #15999).\n        '

        class DictWithMissing(dict):

            def __missing__(self, key):
                return 'missing'
        s = Series([1, 2, 3])
        dictionary = DictWithMissing({3: 'three'})
        result = s.map(dictionary)
        expected = Series(['missing', 'missing', 'three'])
        tm.assert_series_equal(result, expected)

    def test_map_dict_subclass_without_missing(self):

        class DictWithoutMissing(dict):
            pass
        s = Series([1, 2, 3])
        dictionary = DictWithoutMissing({3: 'three'})
        result = s.map(dictionary)
        expected = Series([np.nan, np.nan, 'three'])
        tm.assert_series_equal(result, expected)

    def test_map_abc_mapping(self, non_dict_mapping_subclass):
        s = Series([1, 2, 3])
        not_a_dictionary = non_dict_mapping_subclass({3: 'three'})
        result = s.map(not_a_dictionary)
        expected = Series([np.nan, np.nan, 'three'])
        tm.assert_series_equal(result, expected)

    def test_map_abc_mapping_with_missing(self, non_dict_mapping_subclass):

        class NonDictMappingWithMissing(non_dict_mapping_subclass):

            def __missing__(self, key):
                return 'missing'
        s = Series([1, 2, 3])
        not_a_dictionary = NonDictMappingWithMissing({3: 'three'})
        result = s.map(not_a_dictionary)
        expected = Series([np.nan, np.nan, 'three'])
        tm.assert_series_equal(result, expected)

    def test_map_box(self):
        vals = [pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-02')]
        s = Series(vals)
        assert (s.dtype == 'datetime64[ns]')
        res = s.apply((lambda x: f'{type(x).__name__}_{x.day}_{x.tz}'))
        exp = Series(['Timestamp_1_None', 'Timestamp_2_None'])
        tm.assert_series_equal(res, exp)
        vals = [pd.Timestamp('2011-01-01', tz='US/Eastern'), pd.Timestamp('2011-01-02', tz='US/Eastern')]
        s = Series(vals)
        assert (s.dtype == 'datetime64[ns, US/Eastern]')
        res = s.apply((lambda x: f'{type(x).__name__}_{x.day}_{x.tz}'))
        exp = Series(['Timestamp_1_US/Eastern', 'Timestamp_2_US/Eastern'])
        tm.assert_series_equal(res, exp)
        vals = [pd.Timedelta('1 days'), pd.Timedelta('2 days')]
        s = Series(vals)
        assert (s.dtype == 'timedelta64[ns]')
        res = s.apply((lambda x: f'{type(x).__name__}_{x.days}'))
        exp = Series(['Timedelta_1', 'Timedelta_2'])
        tm.assert_series_equal(res, exp)
        vals = [pd.Period('2011-01-01', freq='M'), pd.Period('2011-01-02', freq='M')]
        s = Series(vals)
        assert (s.dtype == 'Period[M]')
        res = s.apply((lambda x: f'{type(x).__name__}_{x.freqstr}'))
        exp = Series(['Period_M', 'Period_M'])
        tm.assert_series_equal(res, exp)

    def test_map_categorical(self):
        values = pd.Categorical(list('ABBABCD'), categories=list('DCBA'), ordered=True)
        s = Series(values, name='XX', index=list('abcdefg'))
        result = s.map((lambda x: x.lower()))
        exp_values = pd.Categorical(list('abbabcd'), categories=list('dcba'), ordered=True)
        exp = Series(exp_values, name='XX', index=list('abcdefg'))
        tm.assert_series_equal(result, exp)
        tm.assert_categorical_equal(result.values, exp_values)
        result = s.map((lambda x: 'A'))
        exp = Series((['A'] * 7), name='XX', index=list('abcdefg'))
        tm.assert_series_equal(result, exp)
        assert (result.dtype == object)
        with pytest.raises(NotImplementedError, match=tm.EMPTY_STRING_PATTERN):
            s.map((lambda x: x), na_action='ignore')

    def test_map_datetimetz(self):
        values = pd.date_range('2011-01-01', '2011-01-02', freq='H').tz_localize('Asia/Tokyo')
        s = Series(values, name='XX')
        result = s.map((lambda x: (x + pd.offsets.Day())))
        exp_values = pd.date_range('2011-01-02', '2011-01-03', freq='H').tz_localize('Asia/Tokyo')
        exp = Series(exp_values, name='XX')
        tm.assert_series_equal(result, exp)
        result = s.map((lambda x: x.hour))
        exp = Series((list(range(24)) + [0]), name='XX', dtype=np.int64)
        tm.assert_series_equal(result, exp)
        with pytest.raises(NotImplementedError, match=tm.EMPTY_STRING_PATTERN):
            s.map((lambda x: x), na_action='ignore')

        def f(x):
            if (not isinstance(x, pd.Timestamp)):
                raise ValueError
            return str(x.tz)
        result = s.map(f)
        exp = Series((['Asia/Tokyo'] * 25), name='XX')
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize('vals,mapping,exp', [(list('abc'), {np.nan: 'not NaN'}, (([np.nan] * 3) + ['not NaN'])), (list('abc'), {'a': 'a letter'}, (['a letter'] + ([np.nan] * 3))), (list(range(3)), {0: 42}, ([42] + ([np.nan] * 3)))])
    def test_map_missing_mixed(self, vals, mapping, exp):
        s = Series((vals + [np.nan]))
        result = s.map(mapping)
        tm.assert_series_equal(result, Series(exp))

    @pytest.mark.parametrize('dti,exp', [(Series([1, 2], index=pd.DatetimeIndex([0, 31536000000])), DataFrame(np.repeat([[1, 2]], 2, axis=0), dtype='int64')), (tm.makeTimeSeries(nper=30), DataFrame(np.repeat([[1, 2]], 30, axis=0), dtype='int64'))])
    @pytest.mark.parametrize('aware', [True, False])
    def test_apply_series_on_date_time_index_aware_series(self, dti, exp, aware):
        if aware:
            index = dti.tz_localize('UTC').index
        else:
            index = dti.index
        result = Series(index).apply((lambda x: Series([1, 2])))
        tm.assert_frame_equal(result, exp)

    def test_apply_scaler_on_date_time_index_aware_series(self):
        series = tm.makeTimeSeries(nper=30).tz_localize('UTC')
        result = Series(series.index).apply((lambda x: 1))
        tm.assert_series_equal(result, Series(np.ones(30), dtype='int64'))

    def test_map_float_to_string_precision(self):
        ser = Series((1 / 3))
        result = ser.map((lambda val: str(val))).to_dict()
        expected = {0: '0.3333333333333333'}
        assert (result == expected)

    def test_map_with_invalid_na_action_raises(self):
        s = Series([1, 2, 3])
        msg = "na_action must either be 'ignore' or None"
        with pytest.raises(ValueError, match=msg):
            s.map((lambda x: x), na_action='____')

    def test_apply_to_timedelta(self):
        list_of_valid_strings = ['00:00:01', '00:00:02']
        a = pd.to_timedelta(list_of_valid_strings)
        b = Series(list_of_valid_strings).apply(pd.to_timedelta)
        list_of_strings = ['00:00:01', np.nan, pd.NaT, pd.NaT]
        a = pd.to_timedelta(list_of_strings)
        b = Series(list_of_strings).apply(pd.to_timedelta)
