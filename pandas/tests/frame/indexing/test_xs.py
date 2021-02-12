
import re
import numpy as np
import pytest
from pandas import DataFrame, Index, IndexSlice, MultiIndex, Series, concat
import pandas._testing as tm
import pandas.core.common as com
from pandas.tseries.offsets import BDay

@pytest.fixture
def four_level_index_dataframe():
    arr = np.array([[(- 0.5109), (- 2.3358), (- 0.4645), 0.05076, 0.364], [0.4473, 1.4152, 0.2834, 1.00661, 0.1744], [(- 0.6662), (- 0.5243), (- 0.358), 0.89145, 2.5838]])
    index = MultiIndex(levels=[['a', 'x'], ['b', 'q'], [10.0032, 20.0, 30.0], [3, 4, 5]], codes=[[0, 0, 1], [0, 1, 1], [0, 1, 2], [2, 1, 0]], names=['one', 'two', 'three', 'four'])
    return DataFrame(arr, index=index, columns=list('ABCDE'))

class TestXS():

    def test_xs(self, float_frame, datetime_frame):
        idx = float_frame.index[5]
        xs = float_frame.xs(idx)
        for (item, value) in xs.items():
            if np.isnan(value):
                assert np.isnan(float_frame[item][idx])
            else:
                assert (value == float_frame[item][idx])
        test_data = {'A': {'1': 1, '2': 2}, 'B': {'1': '1', '2': '2', '3': '3'}}
        frame = DataFrame(test_data)
        xs = frame.xs('1')
        assert (xs.dtype == np.object_)
        assert (xs['A'] == 1)
        assert (xs['B'] == '1')
        with pytest.raises(KeyError, match=re.escape("Timestamp('1999-12-31 00:00:00', freq='B')")):
            datetime_frame.xs((datetime_frame.index[0] - BDay()))
        series = float_frame.xs('A', axis=1)
        expected = float_frame['A']
        tm.assert_series_equal(series, expected)
        series = float_frame.xs('A', axis=1)
        series[:] = 5
        assert (expected == 5).all()

    def test_xs_corner(self):
        df = DataFrame(index=[0])
        df['A'] = 1.0
        df['B'] = 'foo'
        df['C'] = 2.0
        df['D'] = 'bar'
        df['E'] = 3.0
        xs = df.xs(0)
        exp = Series([1.0, 'foo', 2.0, 'bar', 3.0], index=list('ABCDE'), name=0)
        tm.assert_series_equal(xs, exp)
        df = DataFrame(index=['a', 'b', 'c'])
        result = df.xs('a')
        expected = Series([], name='a', index=Index([]), dtype=np.float64)
        tm.assert_series_equal(result, expected)

    def test_xs_duplicates(self):
        df = DataFrame(np.random.randn(5, 2), index=['b', 'b', 'c', 'b', 'a'])
        cross = df.xs('c')
        exp = df.iloc[2]
        tm.assert_series_equal(cross, exp)

    def test_xs_keep_level(self):
        df = DataFrame({'day': {0: 'sat', 1: 'sun'}, 'flavour': {0: 'strawberry', 1: 'strawberry'}, 'sales': {0: 10, 1: 12}, 'year': {0: 2008, 1: 2008}}).set_index(['year', 'flavour', 'day'])
        result = df.xs('sat', level='day', drop_level=False)
        expected = df[:1]
        tm.assert_frame_equal(result, expected)
        result = df.xs([2008, 'sat'], level=['year', 'day'], drop_level=False)
        tm.assert_frame_equal(result, expected)

    def test_xs_view(self):
        dm = DataFrame(np.arange(20.0).reshape(4, 5), index=range(4), columns=range(5))
        dm.xs(2)[:] = 10
        assert (dm.xs(2) == 10).all()

class TestXSWithMultiIndex():

    def test_xs_integer_key(self):
        dates = range(20111201, 20111205)
        ids = list('abcde')
        index = MultiIndex.from_product([dates, ids], names=['date', 'secid'])
        df = DataFrame(np.random.randn(len(index), 3), index, ['X', 'Y', 'Z'])
        result = df.xs(20111201, level='date')
        expected = df.loc[20111201, :]
        tm.assert_frame_equal(result, expected)

    def test_xs_level(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data
        result = df.xs('two', level='second')
        expected = df[(df.index.get_level_values(1) == 'two')]
        expected.index = Index(['foo', 'bar', 'baz', 'qux'], name='first')
        tm.assert_frame_equal(result, expected)

    def test_xs_level_eq_2(self):
        arr = np.random.randn(3, 5)
        index = MultiIndex(levels=[['a', 'p', 'x'], ['b', 'q', 'y'], ['c', 'r', 'z']], codes=[[2, 0, 1], [2, 0, 1], [2, 0, 1]])
        df = DataFrame(arr, index=index)
        expected = DataFrame(arr[1:2], index=[['a'], ['b']])
        result = df.xs('c', level=2)
        tm.assert_frame_equal(result, expected)

    def test_xs_setting_with_copy_error(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data
        result = df.xs('two', level='second')
        msg = 'A value is trying to be set on a copy of a slice from a DataFrame'
        with pytest.raises(com.SettingWithCopyError, match=msg):
            result[:] = 10

    def test_xs_setting_with_copy_error_multiple(self, four_level_index_dataframe):
        df = four_level_index_dataframe
        result = df.xs(('a', 4), level=['one', 'four'])
        msg = 'A value is trying to be set on a copy of a slice from a DataFrame'
        with pytest.raises(com.SettingWithCopyError, match=msg):
            result[:] = 10

    @pytest.mark.parametrize('key, level', [('one', 'second'), (['one'], ['second'])])
    def test_xs_with_duplicates(self, key, level, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        df = concat(([frame] * 2))
        assert (df.index.is_unique is False)
        expected = concat(([frame.xs('one', level='second')] * 2))
        result = df.xs(key, level=level)
        tm.assert_frame_equal(result, expected)

    def test_xs_missing_values_in_index(self):
        acc = [('a', 'abcde', 1), ('b', 'bbcde', 2), ('y', 'yzcde', 25), ('z', 'xbcde', 24), ('z', None, 26), ('z', 'zbcde', 25), ('z', 'ybcde', 26)]
        df = DataFrame(acc, columns=['a1', 'a2', 'cnt']).set_index(['a1', 'a2'])
        expected = DataFrame({'cnt': [24, 26, 25, 26]}, index=Index(['xbcde', np.nan, 'zbcde', 'ybcde'], name='a2'))
        result = df.xs('z', level='a1')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('key, level, exp_arr, exp_index', [('a', 'lvl0', (lambda x: x[:, 0:2]), Index(['bar', 'foo'], name='lvl1')), ('foo', 'lvl1', (lambda x: x[:, 1:2]), Index(['a'], name='lvl0'))])
    def test_xs_named_levels_axis_eq_1(self, key, level, exp_arr, exp_index):
        arr = np.random.randn(4, 4)
        index = MultiIndex(levels=[['a', 'b'], ['bar', 'foo', 'hello', 'world']], codes=[[0, 0, 1, 1], [0, 1, 2, 3]], names=['lvl0', 'lvl1'])
        df = DataFrame(arr, columns=index)
        result = df.xs(key, level=level, axis=1)
        expected = DataFrame(exp_arr(arr), columns=exp_index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('indexer', [(lambda df: df.xs(('a', 4), level=['one', 'four'])), (lambda df: df.xs('a').xs(4, level='four'))])
    def test_xs_level_multiple(self, indexer, four_level_index_dataframe):
        df = four_level_index_dataframe
        expected_values = [[0.4473, 1.4152, 0.2834, 1.00661, 0.1744]]
        expected_index = MultiIndex(levels=[['q'], [20.0]], codes=[[0], [0]], names=['two', 'three'])
        expected = DataFrame(expected_values, index=expected_index, columns=list('ABCDE'))
        result = indexer(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('indexer', [(lambda df: df.xs('a', level=0)), (lambda df: df.xs('a'))])
    def test_xs_level0(self, indexer, four_level_index_dataframe):
        df = four_level_index_dataframe
        expected_values = [[(- 0.5109), (- 2.3358), (- 0.4645), 0.05076, 0.364], [0.4473, 1.4152, 0.2834, 1.00661, 0.1744]]
        expected_index = MultiIndex(levels=[['b', 'q'], [10.0032, 20.0], [4, 5]], codes=[[0, 1], [0, 1], [1, 0]], names=['two', 'three', 'four'])
        expected = DataFrame(expected_values, index=expected_index, columns=list('ABCDE'))
        result = indexer(df)
        tm.assert_frame_equal(result, expected)

    def test_xs_values(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data
        result = df.xs(('bar', 'two')).values
        expected = df.values[4]
        tm.assert_almost_equal(result, expected)

    def test_xs_loc_equality(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data
        result = df.xs(('bar', 'two'))
        expected = df.loc[('bar', 'two')]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('klass', [DataFrame, Series])
    def test_xs_IndexSlice_argument_not_implemented(self, klass):
        index = MultiIndex(levels=[[('foo', 'bar', 0), ('foo', 'baz', 0), ('foo', 'qux', 0)], [0, 1]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
        obj = DataFrame(np.random.randn(6, 4), index=index)
        if (klass is Series):
            obj = obj[0]
        msg = "Expected label or tuple of labels, got \\(\\('foo', 'qux', 0\\), slice\\(None, None, None\\)\\)"
        with pytest.raises(TypeError, match=msg):
            obj.xs(IndexSlice[('foo', 'qux', 0), :])

    @pytest.mark.parametrize('klass', [DataFrame, Series])
    def test_xs_levels_raises(self, klass):
        obj = DataFrame({'A': [1, 2, 3]})
        if (klass is Series):
            obj = obj['A']
        msg = 'Index must be a MultiIndex'
        with pytest.raises(TypeError, match=msg):
            obj.xs(0, level='as')

    def test_xs_multiindex_droplevel_false(self):
        mi = MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'x')], names=['level1', 'level2'])
        df = DataFrame([[1, 2, 3]], columns=mi)
        result = df.xs('a', axis=1, drop_level=False)
        expected = DataFrame([[1, 2]], columns=MultiIndex.from_tuples([('a', 'x'), ('a', 'y')], names=['level1', 'level2']))
        tm.assert_frame_equal(result, expected)

    def test_xs_droplevel_false(self):
        df = DataFrame([[1, 2, 3]], columns=Index(['a', 'b', 'c']))
        result = df.xs('a', axis=1, drop_level=False)
        expected = DataFrame({'a': [1]})
        tm.assert_frame_equal(result, expected)

    def test_xs_droplevel_false_view(self):
        df = DataFrame([[1, 2, 3]], columns=Index(['a', 'b', 'c']))
        result = df.xs('a', axis=1, drop_level=False)
        df.values[(0, 0)] = 2
        expected = DataFrame({'a': [2]})
        tm.assert_frame_equal(result, expected)
