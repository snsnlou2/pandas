
import random
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import Categorical, DataFrame, NaT, Timestamp, date_range
import pandas._testing as tm

class TestDataFrameSortValues():

    def test_sort_values(self):
        frame = DataFrame([[1, 1, 2], [3, 1, 0], [4, 5, 6]], index=[1, 2, 3], columns=list('ABC'))
        sorted_df = frame.sort_values(by='A')
        indexer = frame['A'].argsort().values
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by='A', ascending=False)
        indexer = indexer[::(- 1)]
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by='A', ascending=False)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by=['A'], ascending=[False])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by=['B', 'C'])
        expected = frame.loc[[2, 1, 3]]
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by=['B', 'C'], ascending=False)
        tm.assert_frame_equal(sorted_df, expected[::(- 1)])
        sorted_df = frame.sort_values(by=['B', 'A'], ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=['A', 'B'], axis=2, inplace=True)
        sorted_df = frame.sort_values(by=3, axis=1)
        expected = frame
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by=3, axis=1, ascending=False)
        expected = frame.reindex(columns=['C', 'B', 'A'])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by=[1, 2], axis='columns')
        expected = frame.reindex(columns=['B', 'A', 'C'])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=False)
        expected = frame.reindex(columns=['C', 'B', 'A'])
        tm.assert_frame_equal(sorted_df, expected)
        msg = 'Length of ascending \\(5\\) != length of by \\(2\\)'
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=['A', 'B'], axis=0, ascending=([True] * 5))

    def test_sort_values_inplace(self):
        frame = DataFrame(np.random.randn(4, 4), index=[1, 2, 3, 4], columns=['A', 'B', 'C', 'D'])
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by='A', inplace=True)
        assert (return_value is None)
        expected = frame.sort_values(by='A')
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by=1, axis=1, inplace=True)
        assert (return_value is None)
        expected = frame.sort_values(by=1, axis=1)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by='A', ascending=False, inplace=True)
        assert (return_value is None)
        expected = frame.sort_values(by='A', ascending=False)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by=['A', 'B'], ascending=False, inplace=True)
        assert (return_value is None)
        expected = frame.sort_values(by=['A', 'B'], ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_multicolumn(self):
        A = np.arange(5).repeat(20)
        B = np.tile(np.arange(5), 20)
        random.shuffle(A)
        random.shuffle(B)
        frame = DataFrame({'A': A, 'B': B, 'C': np.random.randn(100)})
        result = frame.sort_values(by=['A', 'B'])
        indexer = np.lexsort((frame['B'], frame['A']))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)
        result = frame.sort_values(by=['A', 'B'], ascending=False)
        indexer = np.lexsort((frame['B'].rank(ascending=False), frame['A'].rank(ascending=False)))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)
        result = frame.sort_values(by=['B', 'A'])
        indexer = np.lexsort((frame['A'], frame['B']))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_multicolumn_uint64(self):
        df = DataFrame({'a': pd.Series([18446637057563306014, 1162265347240853609]), 'b': pd.Series([1, 2])})
        df['a'] = df['a'].astype(np.uint64)
        result = df.sort_values(['a', 'b'])
        expected = DataFrame({'a': pd.Series([18446637057563306014, 1162265347240853609]), 'b': pd.Series([1, 2])}, index=pd.Index([1, 0]))
        tm.assert_frame_equal(result, expected)

    def test_sort_values_nan(self):
        df = DataFrame({'A': [1, 2, np.nan, 1, 6, 8, 4], 'B': [9, np.nan, 5, 2, 5, 4, 5]})
        expected = DataFrame({'A': [np.nan, 1, 1, 2, 4, 6, 8], 'B': [5, 9, 2, np.nan, 5, 5, 4]}, index=[2, 0, 3, 1, 6, 4, 5])
        sorted_df = df.sort_values(['A'], na_position='first')
        tm.assert_frame_equal(sorted_df, expected)
        expected = DataFrame({'A': [np.nan, 8, 6, 4, 2, 1, 1], 'B': [5, 4, 5, 5, np.nan, 9, 2]}, index=[2, 5, 4, 6, 1, 0, 3])
        sorted_df = df.sort_values(['A'], na_position='first', ascending=False)
        tm.assert_frame_equal(sorted_df, expected)
        expected = df.reindex(columns=['B', 'A'])
        sorted_df = df.sort_values(by=1, axis=1, na_position='first')
        tm.assert_frame_equal(sorted_df, expected)
        expected = DataFrame({'A': [1, 1, 2, 4, 6, 8, np.nan], 'B': [2, 9, np.nan, 5, 5, 4, 5]}, index=[3, 0, 1, 6, 4, 5, 2])
        sorted_df = df.sort_values(['A', 'B'])
        tm.assert_frame_equal(sorted_df, expected)
        expected = DataFrame({'A': [np.nan, 1, 1, 2, 4, 6, 8], 'B': [5, 2, 9, np.nan, 5, 5, 4]}, index=[2, 3, 0, 1, 6, 4, 5])
        sorted_df = df.sort_values(['A', 'B'], na_position='first')
        tm.assert_frame_equal(sorted_df, expected)
        expected = DataFrame({'A': [np.nan, 1, 1, 2, 4, 6, 8], 'B': [5, 9, 2, np.nan, 5, 5, 4]}, index=[2, 0, 3, 1, 6, 4, 5])
        sorted_df = df.sort_values(['A', 'B'], ascending=[1, 0], na_position='first')
        tm.assert_frame_equal(sorted_df, expected)
        expected = DataFrame({'A': [8, 6, 4, 2, 1, 1, np.nan], 'B': [4, 5, 5, np.nan, 2, 9, 5]}, index=[5, 4, 6, 1, 3, 0, 2])
        sorted_df = df.sort_values(['A', 'B'], ascending=[0, 1], na_position='last')
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_descending_sort(self):
        df = DataFrame([[2, 'first'], [2, 'second'], [1, 'a'], [1, 'b']], columns=['sort_col', 'order'])
        sorted_df = df.sort_values(by='sort_col', kind='mergesort', ascending=False)
        tm.assert_frame_equal(df, sorted_df)

    @pytest.mark.parametrize('expected_idx_non_na, ascending', [[[3, 4, 5, 0, 1, 8, 6, 9, 7, 10, 13, 14], [True, True]], [[0, 3, 4, 5, 1, 8, 6, 7, 10, 13, 14, 9], [True, False]], [[9, 7, 10, 13, 14, 6, 8, 1, 3, 4, 5, 0], [False, True]], [[7, 10, 13, 14, 9, 6, 8, 1, 0, 3, 4, 5], [False, False]]])
    @pytest.mark.parametrize('na_position', ['first', 'last'])
    def test_sort_values_stable_multicolumn_sort(self, expected_idx_non_na, ascending, na_position):
        df = DataFrame({'A': [1, 2, np.nan, 1, 1, 1, 6, 8, 4, 8, 8, np.nan, np.nan, 8, 8], 'B': [9, np.nan, 5, 2, 2, 2, 5, 4, 5, 3, 4, np.nan, np.nan, 4, 4]})
        expected_idx = (([11, 12, 2] + expected_idx_non_na) if (na_position == 'first') else (expected_idx_non_na + [2, 11, 12]))
        expected = df.take(expected_idx)
        sorted_df = df.sort_values(['A', 'B'], ascending=ascending, na_position=na_position)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_categorial(self):
        df = DataFrame({'x': Categorical(np.repeat([1, 2, 3, 4], 5), ordered=True)})
        expected = df.copy()
        sorted_df = df.sort_values('x', kind='mergesort')
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_datetimes(self):
        df = DataFrame(['a', 'a', 'a', 'b', 'c', 'd', 'e', 'f', 'g'], columns=['A'], index=date_range('20130101', periods=9))
        dts = [Timestamp(x) for x in ['2004-02-11', '2004-01-21', '2004-01-26', '2005-09-20', '2010-10-04', '2009-05-12', '2008-11-12', '2010-09-28', '2010-09-28']]
        df['B'] = (dts[::2] + dts[1::2])
        df['C'] = 2.0
        df['A1'] = 3.0
        df1 = df.sort_values(by='A')
        df2 = df.sort_values(by=['A'])
        tm.assert_frame_equal(df1, df2)
        df1 = df.sort_values(by='B')
        df2 = df.sort_values(by=['B'])
        tm.assert_frame_equal(df1, df2)
        df1 = df.sort_values(by='B')
        df2 = df.sort_values(by=['C', 'B'])
        tm.assert_frame_equal(df1, df2)

    def test_sort_values_frame_column_inplace_sort_exception(self, float_frame):
        s = float_frame['A']
        with pytest.raises(ValueError, match='This Series is a view'):
            s.sort_values(inplace=True)
        cp = s.copy()
        cp.sort_values()

    def test_sort_values_nat_values_in_int_column(self):
        int_values = (2, int(NaT))
        float_values = (2.0, (- 1.797693e+308))
        df = DataFrame({'int': int_values, 'float': float_values}, columns=['int', 'float'])
        df_reversed = DataFrame({'int': int_values[::(- 1)], 'float': float_values[::(- 1)]}, columns=['int', 'float'], index=[1, 0])
        df_sorted = df.sort_values(['int', 'float'], na_position='last')
        tm.assert_frame_equal(df_sorted, df_reversed)
        df_sorted = df.sort_values(['int', 'float'], na_position='first')
        tm.assert_frame_equal(df_sorted, df_reversed)
        df_sorted = df.sort_values(['int', 'float'], ascending=False)
        tm.assert_frame_equal(df_sorted, df)
        df = DataFrame({'datetime': [Timestamp('2016-01-01'), NaT], 'float': float_values}, columns=['datetime', 'float'])
        df_reversed = DataFrame({'datetime': [NaT, Timestamp('2016-01-01')], 'float': float_values[::(- 1)]}, columns=['datetime', 'float'], index=[1, 0])
        df_sorted = df.sort_values(['datetime', 'float'], na_position='first')
        tm.assert_frame_equal(df_sorted, df_reversed)
        df_sorted = df.sort_values(['datetime', 'float'], na_position='last')
        tm.assert_frame_equal(df_sorted, df)
        df_sorted = df.sort_values(['datetime', 'float'], ascending=False)
        tm.assert_frame_equal(df_sorted, df)

    def test_sort_nat(self):
        d1 = [Timestamp(x) for x in ['2016-01-01', '2015-01-01', np.nan, '2016-01-01']]
        d2 = [Timestamp(x) for x in ['2017-01-01', '2014-01-01', '2016-01-01', '2015-01-01']]
        df = DataFrame({'a': d1, 'b': d2}, index=[0, 1, 2, 3])
        d3 = [Timestamp(x) for x in ['2015-01-01', '2016-01-01', '2016-01-01', np.nan]]
        d4 = [Timestamp(x) for x in ['2014-01-01', '2015-01-01', '2017-01-01', '2016-01-01']]
        expected = DataFrame({'a': d3, 'b': d4}, index=[1, 3, 0, 2])
        sorted_df = df.sort_values(by=['a', 'b'])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_na_position_with_categories(self):
        categories = ['A', 'B', 'C']
        category_indices = [0, 2, 4]
        list_of_nans = [np.nan, np.nan]
        na_indices = [1, 3]
        na_position_first = 'first'
        na_position_last = 'last'
        column_name = 'c'
        reversed_categories = sorted(categories, reverse=True)
        reversed_category_indices = sorted(category_indices, reverse=True)
        reversed_na_indices = sorted(na_indices)
        df = DataFrame({column_name: Categorical(['A', np.nan, 'B', np.nan, 'C'], categories=categories, ordered=True)})
        result = df.sort_values(by=column_name, ascending=True, na_position=na_position_first)
        expected = DataFrame({column_name: Categorical((list_of_nans + categories), categories=categories, ordered=True)}, index=(na_indices + category_indices))
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=column_name, ascending=True, na_position=na_position_last)
        expected = DataFrame({column_name: Categorical((categories + list_of_nans), categories=categories, ordered=True)}, index=(category_indices + na_indices))
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=column_name, ascending=False, na_position=na_position_first)
        expected = DataFrame({column_name: Categorical((list_of_nans + reversed_categories), categories=categories, ordered=True)}, index=(reversed_na_indices + reversed_category_indices))
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=column_name, ascending=False, na_position=na_position_last)
        expected = DataFrame({column_name: Categorical((reversed_categories + list_of_nans), categories=categories, ordered=True)}, index=(reversed_category_indices + reversed_na_indices))
        tm.assert_frame_equal(result, expected)

    def test_sort_values_nat(self):
        d1 = [Timestamp(x) for x in ['2016-01-01', '2015-01-01', np.nan, '2016-01-01']]
        d2 = [Timestamp(x) for x in ['2017-01-01', '2014-01-01', '2016-01-01', '2015-01-01']]
        df = DataFrame({'a': d1, 'b': d2}, index=[0, 1, 2, 3])
        d3 = [Timestamp(x) for x in ['2015-01-01', '2016-01-01', '2016-01-01', np.nan]]
        d4 = [Timestamp(x) for x in ['2014-01-01', '2015-01-01', '2017-01-01', '2016-01-01']]
        expected = DataFrame({'a': d3, 'b': d4}, index=[1, 3, 0, 2])
        sorted_df = df.sort_values(by=['a', 'b'])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_na_position_with_categories_raises(self):
        df = DataFrame({'c': Categorical(['A', np.nan, 'B', np.nan, 'C'], categories=['A', 'B', 'C'], ordered=True)})
        with pytest.raises(ValueError, match='invalid na_position: bad_position'):
            df.sort_values(by='c', ascending=False, na_position='bad_position')

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_dict, sorted_dict, ignore_index, output_index', [({'A': [1, 2, 3]}, {'A': [3, 2, 1]}, True, [0, 1, 2]), ({'A': [1, 2, 3]}, {'A': [3, 2, 1]}, False, [2, 1, 0]), ({'A': [1, 2, 3], 'B': [2, 3, 4]}, {'A': [3, 2, 1], 'B': [4, 3, 2]}, True, [0, 1, 2]), ({'A': [1, 2, 3], 'B': [2, 3, 4]}, {'A': [3, 2, 1], 'B': [4, 3, 2]}, False, [2, 1, 0])])
    def test_sort_values_ignore_index(self, inplace, original_dict, sorted_dict, ignore_index, output_index):
        df = DataFrame(original_dict)
        expected = DataFrame(sorted_dict, index=output_index)
        kwargs = {'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_df = df.copy()
            result_df.sort_values('A', ascending=False, **kwargs)
        else:
            result_df = df.sort_values('A', ascending=False, **kwargs)
        tm.assert_frame_equal(result_df, expected)
        tm.assert_frame_equal(df, DataFrame(original_dict))

    def test_sort_values_nat_na_position_default(self):
        expected = DataFrame({'A': [1, 2, 3, 4, 4], 'date': pd.DatetimeIndex(['2010-01-01 09:00:00', '2010-01-01 09:00:01', '2010-01-01 09:00:02', '2010-01-01 09:00:03', 'NaT'])})
        result = expected.sort_values(['A', 'date'])
        tm.assert_frame_equal(result, expected)

    def test_sort_values_item_cache(self):
        df = DataFrame(np.random.randn(4, 3), columns=['A', 'B', 'C'])
        df['D'] = (df['A'] * 2)
        ser = df['A']
        assert (len(df._mgr.blocks) == 2)
        df.sort_values(by='A')
        ser.values[0] = 99
        assert (df.iloc[(0, 0)] == df['A'][0])

class TestDataFrameSortKey():

    def test_sort_values_inplace_key(self, sort_by_key):
        frame = DataFrame(np.random.randn(4, 4), index=[1, 2, 3, 4], columns=['A', 'B', 'C', 'D'])
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by='A', inplace=True, key=sort_by_key)
        assert (return_value is None)
        expected = frame.sort_values(by='A', key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by=1, axis=1, inplace=True, key=sort_by_key)
        assert (return_value is None)
        expected = frame.sort_values(by=1, axis=1, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by='A', ascending=False, inplace=True, key=sort_by_key)
        assert (return_value is None)
        expected = frame.sort_values(by='A', ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = frame.copy()
        sorted_df.sort_values(by=['A', 'B'], ascending=False, inplace=True, key=sort_by_key)
        expected = frame.sort_values(by=['A', 'B'], ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_key(self):
        df = DataFrame(np.array([0, 5, np.nan, 3, 2, np.nan]))
        result = df.sort_values(0)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(0, key=(lambda x: (x + 5)))
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(0, key=(lambda x: (- x)), ascending=False)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key(self):
        df = DataFrame({'a': np.array([0, 3, np.nan, 3, 2, np.nan]), 'b': np.array([0, 2, np.nan, 5, 2, np.nan])})
        result = df.sort_values('a', key=(lambda x: (- x)))
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['a', 'b'], key=(lambda x: (- x)))
        expected = df.iloc[[3, 1, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['a', 'b'], key=(lambda x: (- x)), ascending=False)
        expected = df.iloc[[0, 4, 1, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key_by_name(self):
        df = DataFrame({'a': np.array([0, 3, np.nan, 3, 2, np.nan]), 'b': np.array([0, 2, np.nan, 5, 2, np.nan])})

        def key(col):
            if (col.name == 'a'):
                return (- col)
            else:
                return col
        result = df.sort_values(by='a', key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['a'], key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by='b', key=key)
        expected = df.iloc[[0, 1, 4, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['a', 'b'], key=key)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_string(self):
        df = DataFrame(np.array([['hello', 'goodbye'], ['hello', 'Hello']]))
        result = df.sort_values(1)
        expected = df[::(- 1)]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values([0, 1], key=(lambda col: col.str.lower()))
        tm.assert_frame_equal(result, df)
        result = df.sort_values([0, 1], key=(lambda col: col.str.lower()), ascending=False)
        expected = df.sort_values(1, key=(lambda col: col.str.lower()), ascending=False)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_empty(self, sort_by_key):
        df = DataFrame(np.array([]))
        df.sort_values(0, key=sort_by_key)
        df.sort_index(key=sort_by_key)

    def test_changes_length_raises(self):
        df = DataFrame({'A': [1, 2, 3]})
        with pytest.raises(ValueError, match='change the shape'):
            df.sort_values('A', key=(lambda x: x[:1]))

    def test_sort_values_key_axes(self):
        df = DataFrame({0: ['Hello', 'goodbye'], 1: [0, 1]})
        result = df.sort_values(0, key=(lambda col: col.str.lower()))
        expected = df[::(- 1)]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(1, key=(lambda col: (- col)))
        expected = df[::(- 1)]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key_dict_axis(self):
        df = DataFrame({0: ['Hello', 0], 1: ['goodbye', 1]})
        result = df.sort_values(0, key=(lambda col: col.str.lower()), axis=1)
        expected = df.loc[:, ::(- 1)]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(1, key=(lambda col: (- col)), axis=1)
        expected = df.loc[:, ::(- 1)]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('ordered', [True, False])
    def test_sort_values_key_casts_to_categorical(self, ordered):
        categories = ['c', 'b', 'a']
        df = DataFrame({'x': [1, 1, 1], 'y': ['a', 'b', 'c']})

        def sorter(key):
            if (key.name == 'y'):
                return pd.Series(Categorical(key, categories=categories, ordered=ordered))
            return key
        result = df.sort_values(by=['x', 'y'], key=sorter)
        expected = DataFrame({'x': [1, 1, 1], 'y': ['c', 'b', 'a']}, index=pd.Index([2, 1, 0]))
        tm.assert_frame_equal(result, expected)

@pytest.fixture
def df_none():
    return DataFrame({'outer': ['a', 'a', 'a', 'b', 'b', 'b'], 'inner': [1, 2, 2, 2, 1, 1], 'A': np.arange(6, 0, (- 1)), ('B', 5): ['one', 'one', 'two', 'two', 'one', 'one']})

@pytest.fixture(params=[['outer'], ['outer', 'inner']])
def df_idx(request, df_none):
    levels = request.param
    return df_none.set_index(levels)

@pytest.fixture(params=['inner', ['outer'], 'A', [('B', 5)], ['inner', 'outer'], [('B', 5), 'outer'], ['A', ('B', 5)], ['inner', 'outer']])
def sort_names(request):
    return request.param

@pytest.fixture(params=[True, False])
def ascending(request):
    return request.param

class TestSortValuesLevelAsStr():

    def test_sort_index_level_and_column_label(self, df_none, df_idx, sort_names, ascending):
        levels = df_idx.index.names
        expected = df_none.sort_values(by=sort_names, ascending=ascending, axis=0).set_index(levels)
        result = df_idx.sort_values(by=sort_names, ascending=ascending, axis=0)
        tm.assert_frame_equal(result, expected)

    def test_sort_column_level_and_index_label(self, df_none, df_idx, sort_names, ascending):
        levels = df_idx.index.names
        expected = df_none.sort_values(by=sort_names, ascending=ascending, axis=0).set_index(levels).T
        result = df_idx.T.sort_values(by=sort_names, ascending=ascending, axis=1)
        if (len(levels) > 1):
            with tm.assert_produces_warning(PerformanceWarning, check_stacklevel=False):
                tm.assert_frame_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)
