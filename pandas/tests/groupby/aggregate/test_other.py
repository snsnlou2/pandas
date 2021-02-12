
'\ntest all other .agg behavior\n'
import datetime as dt
from functools import partial
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, PeriodIndex, Series, date_range, period_range
import pandas._testing as tm
from pandas.core.base import SpecificationError
from pandas.io.formats.printing import pprint_thing

def test_agg_api():
    df = DataFrame({'data1': np.random.randn(5), 'data2': np.random.randn(5), 'key1': ['a', 'a', 'b', 'b', 'a'], 'key2': ['one', 'two', 'one', 'two', 'one']})
    grouped = df.groupby('key1')

    def peak_to_peak(arr):
        return (arr.max() - arr.min())
    expected = grouped.agg([peak_to_peak])
    expected.columns = ['data1', 'data2']
    result = grouped.agg(peak_to_peak)
    tm.assert_frame_equal(result, expected)

def test_agg_datetimes_mixed():
    data = [[1, '2012-01-01', 1.0], [2, '2012-01-02', 2.0], [3, None, 3.0]]
    df1 = DataFrame({'key': [x[0] for x in data], 'date': [x[1] for x in data], 'value': [x[2] for x in data]})
    data = [[row[0], (dt.datetime.strptime(row[1], '%Y-%m-%d').date() if row[1] else None), row[2]] for row in data]
    df2 = DataFrame({'key': [x[0] for x in data], 'date': [x[1] for x in data], 'value': [x[2] for x in data]})
    df1['weights'] = (df1['value'] / df1['value'].sum())
    gb1 = df1.groupby('date').aggregate(np.sum)
    df2['weights'] = (df1['value'] / df1['value'].sum())
    gb2 = df2.groupby('date').aggregate(np.sum)
    assert (len(gb1) == len(gb2))

def test_agg_period_index():
    prng = period_range('2012-1-1', freq='M', periods=3)
    df = DataFrame(np.random.randn(3, 2), index=prng)
    rs = df.groupby(level=0).sum()
    assert isinstance(rs.index, PeriodIndex)
    index = period_range(start='1999-01', periods=5, freq='M')
    s1 = Series(np.random.rand(len(index)), index=index)
    s2 = Series(np.random.rand(len(index)), index=index)
    df = DataFrame.from_dict({'s1': s1, 's2': s2})
    grouped = df.groupby(df.index.month)
    list(grouped)

def test_agg_dict_parameter_cast_result_dtypes():
    df = DataFrame({'class': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'], 'time': date_range('1/1/2011', periods=8, freq='H')})
    df.loc[([0, 1, 2, 5], 'time')] = None
    exp = df.loc[[0, 3, 4, 6]].set_index('class')
    grouped = df.groupby('class')
    tm.assert_frame_equal(grouped.first(), exp)
    tm.assert_frame_equal(grouped.agg('first'), exp)
    tm.assert_frame_equal(grouped.agg({'time': 'first'}), exp)
    tm.assert_series_equal(grouped.time.first(), exp['time'])
    tm.assert_series_equal(grouped.time.agg('first'), exp['time'])
    exp = df.loc[[0, 3, 4, 7]].set_index('class')
    grouped = df.groupby('class')
    tm.assert_frame_equal(grouped.last(), exp)
    tm.assert_frame_equal(grouped.agg('last'), exp)
    tm.assert_frame_equal(grouped.agg({'time': 'last'}), exp)
    tm.assert_series_equal(grouped.time.last(), exp['time'])
    tm.assert_series_equal(grouped.time.agg('last'), exp['time'])
    exp = Series([2, 2, 2, 2], index=Index(list('ABCD'), name='class'), name='time')
    tm.assert_series_equal(grouped.time.agg(len), exp)
    tm.assert_series_equal(grouped.time.size(), exp)
    exp = Series([0, 1, 1, 2], index=Index(list('ABCD'), name='class'), name='time')
    tm.assert_series_equal(grouped.time.count(), exp)

def test_agg_cast_results_dtypes():
    u = [dt.datetime(2015, (x + 1), 1) for x in range(12)]
    v = list('aaabbbbbbccd')
    df = DataFrame({'X': v, 'Y': u})
    result = df.groupby('X')['Y'].agg(len)
    expected = df.groupby('X')['Y'].count()
    tm.assert_series_equal(result, expected)

def test_aggregate_float64_no_int64():
    df = DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 2, 4, 5], 'c': [1, 2, 3, 4, 5]})
    expected = DataFrame({'a': [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = 'b'
    result = df.groupby('b')[['a']].mean()
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({'a': [1, 2.5, 4, 5], 'c': [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = 'b'
    result = df.groupby('b')[['a', 'c']].mean()
    tm.assert_frame_equal(result, expected)

def test_aggregate_api_consistency():
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': (np.random.randn(8) + 1.0), 'D': np.arange(8)})
    grouped = df.groupby(['A', 'B'])
    c_mean = grouped['C'].mean()
    c_sum = grouped['C'].sum()
    d_mean = grouped['D'].mean()
    d_sum = grouped['D'].sum()
    result = grouped['D'].agg(['sum', 'mean'])
    expected = pd.concat([d_sum, d_mean], axis=1)
    expected.columns = ['sum', 'mean']
    tm.assert_frame_equal(result, expected, check_like=True)
    result = grouped.agg([np.sum, np.mean])
    expected = pd.concat([c_sum, c_mean, d_sum, d_mean], axis=1)
    expected.columns = MultiIndex.from_product([['C', 'D'], ['sum', 'mean']])
    tm.assert_frame_equal(result, expected, check_like=True)
    result = grouped[['D', 'C']].agg([np.sum, np.mean])
    expected = pd.concat([d_sum, d_mean, c_sum, c_mean], axis=1)
    expected.columns = MultiIndex.from_product([['D', 'C'], ['sum', 'mean']])
    tm.assert_frame_equal(result, expected, check_like=True)
    result = grouped.agg({'C': 'mean', 'D': 'sum'})
    expected = pd.concat([d_sum, c_mean], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)
    result = grouped.agg({'C': ['mean', 'sum'], 'D': ['mean', 'sum']})
    expected = pd.concat([c_mean, c_sum, d_mean, d_sum], axis=1)
    expected.columns = MultiIndex.from_product([['C', 'D'], ['mean', 'sum']])
    msg = "Column\\(s\\) \\['r', 'r2'\\] do not exist"
    with pytest.raises(SpecificationError, match=msg):
        grouped[['D', 'C']].agg({'r': np.sum, 'r2': np.mean})

def test_agg_dict_renaming_deprecation():
    df = DataFrame({'A': [1, 1, 1, 2, 2], 'B': range(5), 'C': range(5)})
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        df.groupby('A').agg({'B': {'foo': ['sum', 'max']}, 'C': {'bar': ['count', 'min']}})
    msg = "Column\\(s\\) \\['ma'\\] do not exist"
    with pytest.raises(SpecificationError, match=msg):
        df.groupby('A')[['B', 'C']].agg({'ma': 'max'})
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        df.groupby('A').B.agg({'foo': 'count'})

def test_agg_compat():
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': (np.random.randn(8) + 1.0), 'D': np.arange(8)})
    g = df.groupby(['A', 'B'])
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        g['D'].agg({'C': ['sum', 'std']})
    with pytest.raises(SpecificationError, match=msg):
        g['D'].agg({'C': 'sum', 'D': 'std'})

def test_agg_nested_dicts():
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'two', 'two', 'two', 'one', 'two'], 'C': (np.random.randn(8) + 1.0), 'D': np.arange(8)})
    g = df.groupby(['A', 'B'])
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        g.aggregate({'r1': {'C': ['mean', 'sum']}, 'r2': {'D': ['mean', 'sum']}})
    with pytest.raises(SpecificationError, match=msg):
        g.agg({'C': {'ra': ['mean', 'std']}, 'D': {'rb': ['mean', 'std']}})
    with pytest.raises(SpecificationError, match=msg):
        g['D'].agg({'result1': np.sum, 'result2': np.mean})
    with pytest.raises(SpecificationError, match=msg):
        g['D'].agg({'D': np.sum, 'result2': np.mean})

def test_agg_item_by_item_raise_typeerror():
    df = DataFrame(np.random.randint(10, size=(20, 10)))

    def raiseException(df):
        pprint_thing('----------------------------------------')
        pprint_thing(df.to_string())
        raise TypeError('test')
    with pytest.raises(TypeError, match='test'):
        df.groupby(0).agg(raiseException)

def test_series_agg_multikey():
    ts = tm.makeTimeSeries()
    grouped = ts.groupby([(lambda x: x.year), (lambda x: x.month)])
    result = grouped.agg(np.sum)
    expected = grouped.sum()
    tm.assert_series_equal(result, expected)

def test_series_agg_multi_pure_python():
    data = DataFrame({'A': ['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar', 'foo', 'foo', 'foo'], 'B': ['one', 'one', 'one', 'two', 'one', 'one', 'one', 'two', 'two', 'two', 'one'], 'C': ['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny', 'shiny', 'dull', 'shiny', 'shiny', 'shiny'], 'D': np.random.randn(11), 'E': np.random.randn(11), 'F': np.random.randn(11)})

    def bad(x):
        assert (len(x.values.base) > 0)
        return 'foo'
    result = data.groupby(['A', 'B']).agg(bad)
    expected = data.groupby(['A', 'B']).agg((lambda x: 'foo'))
    tm.assert_frame_equal(result, expected)

def test_agg_consistency():

    def P1(a):
        return np.percentile(a.dropna(), q=1)
    df = DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 25, 26, 31], 'date': [dt.date(2013, 2, 10), dt.date(2013, 2, 10), dt.date(2013, 2, 11), dt.date(2013, 2, 11)]})
    g = df.groupby('date')
    expected = g.agg([P1])
    expected.columns = expected.columns.levels[0]
    result = g.agg(P1)
    tm.assert_frame_equal(result, expected)

def test_agg_callables():
    df = DataFrame({'foo': [1, 2], 'bar': [3, 4]}).astype(np.int64)

    class fn_class():

        def __call__(self, x):
            return sum(x)
    equiv_callables = [sum, np.sum, (lambda x: sum(x)), (lambda x: x.sum()), partial(sum), fn_class()]
    expected = df.groupby('foo').agg(sum)
    for ecall in equiv_callables:
        result = df.groupby('foo').agg(ecall)
        tm.assert_frame_equal(result, expected)

def test_agg_over_numpy_arrays():
    df = DataFrame([[1, np.array([10, 20, 30])], [1, np.array([40, 50, 60])], [2, np.array([20, 30, 40])]], columns=['category', 'arraydata'])
    result = df.groupby('category').agg(sum)
    expected_data = [[np.array([50, 70, 90])], [np.array([20, 30, 40])]]
    expected_index = Index([1, 2], name='category')
    expected_column = ['arraydata']
    expected = DataFrame(expected_data, index=expected_index, columns=expected_column)
    tm.assert_frame_equal(result, expected)

def test_agg_tzaware_non_datetime_result():
    dti = pd.date_range('2012-01-01', periods=4, tz='UTC')
    df = DataFrame({'a': [0, 0, 1, 1], 'b': dti})
    gb = df.groupby('a')
    result = gb['b'].agg((lambda x: x.iloc[0]))
    expected = Series(dti[::2], name='b')
    expected.index.name = 'a'
    tm.assert_series_equal(result, expected)
    result = gb['b'].agg((lambda x: x.iloc[0].year))
    expected = Series([2012, 2012], name='b')
    expected.index.name = 'a'
    tm.assert_series_equal(result, expected)
    result = gb['b'].agg((lambda x: (x.iloc[(- 1)] - x.iloc[0])))
    expected = Series([pd.Timedelta(days=1), pd.Timedelta(days=1)], name='b')
    expected.index.name = 'a'
    tm.assert_series_equal(result, expected)

def test_agg_timezone_round_trip():
    ts = pd.Timestamp('2016-01-01 12:00:00', tz='US/Pacific')
    df = DataFrame({'a': 1, 'b': [(ts + dt.timedelta(minutes=nn)) for nn in range(10)]})
    result1 = df.groupby('a')['b'].agg(np.min).iloc[0]
    result2 = df.groupby('a')['b'].agg((lambda x: np.min(x))).iloc[0]
    result3 = df.groupby('a')['b'].min().iloc[0]
    assert (result1 == ts)
    assert (result2 == ts)
    assert (result3 == ts)
    dates = [pd.Timestamp(f'2016-01-0{i:d} 12:00:00', tz='US/Pacific') for i in range(1, 5)]
    df = DataFrame({'A': (['a', 'b'] * 2), 'B': dates})
    grouped = df.groupby('A')
    ts = df['B'].iloc[0]
    assert (ts == grouped.nth(0)['B'].iloc[0])
    assert (ts == grouped.head(1)['B'].iloc[0])
    assert (ts == grouped.first()['B'].iloc[0])
    assert (ts == grouped.apply((lambda x: x.iloc[0])).iloc[(0, 1)])
    ts = df['B'].iloc[2]
    assert (ts == grouped.last()['B'].iloc[0])
    assert (ts == grouped.apply((lambda x: x.iloc[(- 1)])).iloc[(0, 1)])

def test_sum_uint64_overflow():
    df = DataFrame([[1, 2], [3, 4], [5, 6]], dtype=object)
    df = (df + 9223372036854775807)
    index = Index([9223372036854775808, 9223372036854775810, 9223372036854775812], dtype=np.uint64)
    expected = DataFrame({1: [9223372036854775809, 9223372036854775811, 9223372036854775813]}, index=index)
    expected.index.name = 0
    result = df.groupby(0).sum()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('structure, expected', [(tuple, DataFrame({'C': {(1, 1): (1, 1, 1), (3, 4): (3, 4, 4)}})), (list, DataFrame({'C': {(1, 1): [1, 1, 1], (3, 4): [3, 4, 4]}})), ((lambda x: tuple(x)), DataFrame({'C': {(1, 1): (1, 1, 1), (3, 4): (3, 4, 4)}})), ((lambda x: list(x)), DataFrame({'C': {(1, 1): [1, 1, 1], (3, 4): [3, 4, 4]}}))])
def test_agg_structs_dataframe(structure, expected):
    df = DataFrame({'A': [1, 1, 1, 3, 3, 3], 'B': [1, 1, 1, 4, 4, 4], 'C': [1, 1, 1, 3, 4, 4]})
    result = df.groupby(['A', 'B']).aggregate(structure)
    expected.index.names = ['A', 'B']
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('structure, expected', [(tuple, Series([(1, 1, 1), (3, 4, 4)], index=[1, 3], name='C')), (list, Series([[1, 1, 1], [3, 4, 4]], index=[1, 3], name='C')), ((lambda x: tuple(x)), Series([(1, 1, 1), (3, 4, 4)], index=[1, 3], name='C')), ((lambda x: list(x)), Series([[1, 1, 1], [3, 4, 4]], index=[1, 3], name='C'))])
def test_agg_structs_series(structure, expected):
    df = DataFrame({'A': [1, 1, 1, 3, 3, 3], 'B': [1, 1, 1, 4, 4, 4], 'C': [1, 1, 1, 3, 4, 4]})
    result = df.groupby('A')['C'].aggregate(structure)
    expected.index.name = 'A'
    tm.assert_series_equal(result, expected)

def test_agg_category_nansum(observed):
    categories = ['a', 'b', 'c']
    df = DataFrame({'A': pd.Categorical(['a', 'a', 'b'], categories=categories), 'B': [1, 2, 3]})
    result = df.groupby('A', observed=observed).B.agg(np.nansum)
    expected = Series([3, 3, 0], index=pd.CategoricalIndex(['a', 'b', 'c'], categories=categories, name='A'), name='B')
    if observed:
        expected = expected[(expected != 0)]
    tm.assert_series_equal(result, expected)

def test_agg_list_like_func():
    df = DataFrame({'A': [str(x) for x in range(3)], 'B': [str(x) for x in range(3)]})
    grouped = df.groupby('A', as_index=False, sort=False)
    result = grouped.agg({'B': (lambda x: list(x))})
    expected = DataFrame({'A': [str(x) for x in range(3)], 'B': [[str(x)] for x in range(3)]})
    tm.assert_frame_equal(result, expected)

def test_agg_lambda_with_timezone():
    df = DataFrame({'tag': [1, 1], 'date': [pd.Timestamp('2018-01-01', tz='UTC'), pd.Timestamp('2018-01-02', tz='UTC')]})
    result = df.groupby('tag').agg({'date': (lambda e: e.head(1))})
    expected = DataFrame([pd.Timestamp('2018-01-01', tz='UTC')], index=Index([1], name='tag'), columns=['date'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('err_cls', [NotImplementedError, RuntimeError, KeyError, IndexError, OSError, ValueError, ArithmeticError, AttributeError])
def test_groupby_agg_err_catching(err_cls):
    from pandas.tests.extension.decimal.array import DecimalArray, make_data, to_decimal
    data = make_data()[:5]
    df = DataFrame({'id1': [0, 0, 0, 1, 1], 'id2': [0, 1, 0, 1, 1], 'decimals': DecimalArray(data)})
    expected = Series(to_decimal([data[0], data[3]]))

    def weird_func(x):
        if (len(x) == 0):
            raise err_cls
        return x.iloc[0]
    result = df['decimals'].groupby(df['id1']).agg(weird_func)
    tm.assert_series_equal(result, expected, check_names=False)
