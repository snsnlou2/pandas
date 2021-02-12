
' test fancy indexing & misc '
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas.core.dtypes.common import is_float_dtype, is_integer_dtype
import pandas as pd
from pandas import DataFrame, Index, NaT, Series, date_range, offsets, timedelta_range
import pandas._testing as tm
from pandas.core.indexing import maybe_numeric_slice, non_reducing_slice
from pandas.tests.indexing.common import _mklbl
from .test_floats import gen_obj

class TestFancy():
    ' pure get/set item & fancy indexing '

    def test_setitem_ndarray_1d(self):
        df = DataFrame(index=Index(np.arange(1, 11)))
        df['foo'] = np.zeros(10, dtype=np.float64)
        df['bar'] = np.zeros(10, dtype=complex)
        msg = 'Must have equal len keys and value when setting with an iterable'
        with pytest.raises(ValueError, match=msg):
            df.loc[(df.index[2:5], 'bar')] = np.array([2.33j, (1.23 + 0.1j), 2.2, 1.0])
        df.loc[(df.index[2:6], 'bar')] = np.array([2.33j, (1.23 + 0.1j), 2.2, 1.0])
        result = df.loc[(df.index[2:6], 'bar')]
        expected = Series([2.33j, (1.23 + 0.1j), 2.2, 1.0], index=[3, 4, 5, 6], name='bar')
        tm.assert_series_equal(result, expected)
        df = DataFrame(index=Index(np.arange(1, 11)))
        df['foo'] = np.zeros(10, dtype=np.float64)
        df['bar'] = np.zeros(10, dtype=complex)
        msg = 'Must have equal len keys and value when setting with an iterable'
        with pytest.raises(ValueError, match=msg):
            df[2:5] = (np.arange(1, 4) * 1j)

    @pytest.mark.parametrize('idxr', [tm.getitem, tm.loc, tm.iloc])
    def test_getitem_ndarray_3d(self, index, frame_or_series, idxr):
        obj = gen_obj(frame_or_series, index)
        idxr = idxr(obj)
        nd3 = np.random.randint(5, size=(2, 2, 2))
        msg = '|'.join(['Buffer has wrong number of dimensions \\(expected 1, got 3\\)', 'Cannot index with multidimensional key', 'Wrong number of dimensions. values.ndim != ndim \\[3 != 1\\]', 'Index data must be 1-dimensional', 'positional indexers are out-of-bounds', 'Indexing a MultiIndex with a multidimensional key is not implemented'])
        potential_errors = (IndexError, ValueError, NotImplementedError)
        with pytest.raises(potential_errors, match=msg):
            with tm.assert_produces_warning(DeprecationWarning, check_stacklevel=False):
                idxr[nd3]

    @pytest.mark.parametrize('indexer', [tm.setitem, tm.loc, tm.iloc])
    def test_setitem_ndarray_3d(self, index, frame_or_series, indexer):
        obj = gen_obj(frame_or_series, index)
        idxr = indexer(obj)
        nd3 = np.random.randint(5, size=(2, 2, 2))
        if (indexer.__name__ == 'iloc'):
            err = ValueError
            msg = f'Cannot set values with ndim > {obj.ndim}'
        elif (isinstance(index, pd.IntervalIndex) and (indexer.__name__ == 'setitem') and (obj.ndim == 1)):
            err = AttributeError
            msg = "'pandas._libs.interval.IntervalTree' object has no attribute 'get_loc'"
        else:
            err = ValueError
            msg = 'Buffer has wrong number of dimensions \\(expected 1, got 3\\)|'
        with pytest.raises(err, match=msg):
            idxr[nd3] = 0

    def test_inf_upcast(self):
        df = DataFrame(columns=[0])
        df.loc[1] = 1
        df.loc[2] = 2
        df.loc[np.inf] = 3
        assert (df.loc[(np.inf, 0)] == 3)
        result = df.index
        expected = pd.Float64Index([1, 2, np.inf])
        tm.assert_index_equal(result, expected)
        df = DataFrame()
        df.loc[(0, 0)] = 1
        df.loc[(1, 1)] = 2
        df.loc[(0, np.inf)] = 3
        result = df.columns
        expected = pd.Float64Index([0, 1, np.inf])
        tm.assert_index_equal(result, expected)

    def test_setitem_dtype_upcast(self):
        df = DataFrame([{'a': 1}, {'a': 3, 'b': 2}])
        df['c'] = np.nan
        assert (df['c'].dtype == np.float64)
        df.loc[(0, 'c')] = 'foo'
        expected = DataFrame([{'a': 1, 'b': np.nan, 'c': 'foo'}, {'a': 3, 'b': 2, 'c': np.nan}])
        tm.assert_frame_equal(df, expected)
        df = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=list('ab'), columns=['foo', 'bar', 'baz'])
        for val in [3.14, 'wxyz']:
            left = df.copy()
            left.loc[('a', 'bar')] = val
            right = DataFrame([[0, val, 2], [3, 4, 5]], index=list('ab'), columns=['foo', 'bar', 'baz'])
            tm.assert_frame_equal(left, right)
            assert is_integer_dtype(left['foo'])
            assert is_integer_dtype(left['baz'])
        left = DataFrame((np.arange(6, dtype='int64').reshape(2, 3) / 10.0), index=list('ab'), columns=['foo', 'bar', 'baz'])
        left.loc[('a', 'bar')] = 'wxyz'
        right = DataFrame([[0, 'wxyz', 0.2], [0.3, 0.4, 0.5]], index=list('ab'), columns=['foo', 'bar', 'baz'])
        tm.assert_frame_equal(left, right)
        assert is_float_dtype(left['foo'])
        assert is_float_dtype(left['baz'])

    def test_dups_fancy_indexing(self):
        df = tm.makeCustomDataframe(10, 3)
        df.columns = ['a', 'a', 'b']
        result = df[['b', 'a']].columns
        expected = Index(['b', 'a', 'a'])
        tm.assert_index_equal(result, expected)
        df = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']], columns=list('aaaaaaa'))
        df.head()
        str(df)
        result = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']])
        result.columns = list('aaaaaaa')
        df_v = df.iloc[:, 4]
        res_v = result.iloc[:, 4]
        tm.assert_frame_equal(df, result)
        df = DataFrame({'test': [5, 7, 9, 11], 'test1': [4.0, 5, 6, 7], 'other': list('abcd')}, index=['A', 'A', 'B', 'C'])
        rows = ['C', 'B']
        expected = DataFrame({'test': [11, 9], 'test1': [7.0, 6], 'other': ['d', 'c']}, index=rows)
        result = df.loc[rows]
        tm.assert_frame_equal(result, expected)
        result = df.loc[Index(rows)]
        tm.assert_frame_equal(result, expected)
        rows = ['C', 'B', 'E']
        with pytest.raises(KeyError, match='with any missing labels'):
            df.loc[rows]
        rows = ['F', 'G', 'H', 'C', 'B', 'E']
        with pytest.raises(KeyError, match='with any missing labels'):
            df.loc[rows]
        dfnu = DataFrame(np.random.randn(5, 3), index=list('AABCD'))
        with pytest.raises(KeyError, match=re.escape('"None of [Index([\'E\'], dtype=\'object\')] are in the [index]"')):
            dfnu.loc[['E']]
        df = DataFrame({'A': [0, 1, 2]})
        with pytest.raises(KeyError, match='with any missing labels'):
            df.loc[[0, 8, 0]]
        df = DataFrame({'A': list('abc')})
        with pytest.raises(KeyError, match='with any missing labels'):
            df.loc[[0, 8, 0]]
        df = DataFrame({'test': [5, 7, 9, 11]}, index=['A', 'A', 'B', 'C'])
        with pytest.raises(KeyError, match='with any missing labels'):
            df.loc[['A', 'A', 'E']]

    def test_dups_fancy_indexing2(self):
        df = DataFrame(np.random.randn(5, 5), columns=['A', 'B', 'B', 'B', 'A'])
        with pytest.raises(KeyError, match='with any missing labels'):
            df.loc[:, ['A', 'B', 'C']]
        df = DataFrame(np.random.randn(9, 2), index=[1, 1, 1, 2, 2, 2, 3, 3, 3], columns=['a', 'b'])
        expected = df.iloc[0:6]
        result = df.loc[[1, 2]]
        tm.assert_frame_equal(result, expected)
        expected = df
        result = df.loc[:, ['a', 'b']]
        tm.assert_frame_equal(result, expected)
        expected = df.iloc[0:6, :]
        result = df.loc[([1, 2], ['a', 'b'])]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('case', [tm.getitem, tm.loc])
    def test_duplicate_int_indexing(self, case):
        s = Series(range(3), index=[1, 1, 3])
        expected = s[1]
        result = case(s)[[1]]
        tm.assert_series_equal(result, expected)

    def test_indexing_mixed_frame_bug(self):
        df = DataFrame({'a': {1: 'aaa', 2: 'bbb', 3: 'ccc'}, 'b': {1: 111, 2: 222, 3: 333}})
        df['test'] = df['a'].apply((lambda x: ('_' if (x == 'aaa') else x)))
        idx = (df['test'] == '_')
        temp = df.loc[(idx, 'a')].apply((lambda x: ('-----' if (x == 'aaa') else x)))
        df.loc[(idx, 'test')] = temp
        assert (df.iloc[(0, 2)] == '-----')

    def test_multitype_list_index_access(self):
        df = DataFrame(np.random.random((10, 5)), columns=(['a'] + [20, 21, 22, 23]))
        with pytest.raises(KeyError, match=re.escape("'[-8, 26] not in index'")):
            df[[22, 26, (- 8)]]
        assert (df[21].shape[0] == df.shape[0])

    def test_set_index_nan(self):
        df = DataFrame({'PRuid': {17: 'nonQC', 18: 'nonQC', 19: 'nonQC', 20: '10', 21: '11', 22: '12', 23: '13', 24: '24', 25: '35', 26: '46', 27: '47', 28: '48', 29: '59', 30: '10'}, 'QC': {17: 0.0, 18: 0.0, 19: 0.0, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: 1.0, 25: np.nan, 26: np.nan, 27: np.nan, 28: np.nan, 29: np.nan, 30: np.nan}, 'data': {17: 7.95449, 18: 8.014261, 19: 7.859152000000001, 20: 0.8614035, 21: 0.8785311, 22: 0.8427041999999999, 23: 0.785877, 24: 0.7306246, 25: 0.8166856, 26: 0.8192708000000001, 27: 0.8070501, 28: 0.8144024000000001, 29: 0.8014085, 30: 0.8130774000000001}, 'year': {17: 2006, 18: 2007, 19: 2008, 20: 1985, 21: 1985, 22: 1985, 23: 1985, 24: 1985, 25: 1985, 26: 1985, 27: 1985, 28: 1985, 29: 1985, 30: 1986}}).reset_index()
        result = df.set_index(['year', 'PRuid', 'QC']).reset_index().reindex(columns=df.columns)
        tm.assert_frame_equal(result, df)

    def test_multi_assign(self):
        df = DataFrame({'FC': ['a', 'b', 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': list(range(6)), 'col2': list(range(6, 12))})
        df.iloc[(1, 0)] = np.nan
        df2 = df.copy()
        mask = (~ df2.FC.isna())
        cols = ['col1', 'col2']
        dft = (df2 * 2)
        dft.iloc[(3, 3)] = np.nan
        expected = DataFrame({'FC': ['a', np.nan, 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': Series([0, 1, 4, 6, 8, 10]), 'col2': [12, 7, 16, np.nan, 20, 22]})
        df2.loc[(mask, cols)] = dft.loc[(mask, cols)]
        tm.assert_frame_equal(df2, expected)
        df2.loc[(mask, cols)] = dft.loc[(mask, cols)]
        tm.assert_frame_equal(df2, expected)
        expected = DataFrame({'FC': ['a', np.nan, 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': [0.0, 1.0, 4.0, 6.0, 8.0, 10.0], 'col2': [12, 7, 16, np.nan, 20, 22]})
        df2 = df.copy()
        df2.loc[(mask, cols)] = dft.loc[(mask, cols)].values
        tm.assert_frame_equal(df2, expected)
        df2.loc[(mask, cols)] = dft.loc[(mask, cols)].values
        tm.assert_frame_equal(df2, expected)
        df = DataFrame({'A': [1, 2, 0, 0, 0], 'B': [0, 0, 0, 10, 11], 'C': [0, 0, 0, 10, 11], 'D': [3, 4, 5, 6, 7]})
        expected = df.copy()
        mask = (expected['A'] == 0)
        for col in ['A', 'B']:
            expected.loc[(mask, col)] = df['D']
        df.loc[((df['A'] == 0), ['A', 'B'])] = df['D']
        tm.assert_frame_equal(df, expected)

    def test_setitem_list(self):
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[(1, 0)] = [1, 2, 3]
        df.iloc[(1, 0)] = [1, 2]
        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[(1, 0)] = [1, 2]
        tm.assert_frame_equal(result, df)

        class TO():

            def __init__(self, value):
                self.value = value

            def __str__(self) -> str:
                return f'[{self.value}]'
            __repr__ = __str__

            def __eq__(self, other) -> bool:
                return (self.value == other.value)

            def view(self):
                return self
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[(1, 0)] = TO(1)
        df.iloc[(1, 0)] = TO(2)
        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[(1, 0)] = TO(2)
        tm.assert_frame_equal(result, df)
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[(1, 0)] = TO(1)
        df.iloc[(1, 0)] = np.nan
        result = DataFrame(index=[0, 1], columns=[0])
        tm.assert_frame_equal(result, df)

    def test_string_slice(self):
        df = DataFrame([1], Index([pd.Timestamp('2011-01-01')], dtype=object))
        assert df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df['2011']
        with pytest.raises(KeyError, match="'2011'"):
            df.loc[('2011', 0)]
        df = DataFrame()
        assert (not df.index._is_all_dates)
        with pytest.raises(KeyError, match="'2011'"):
            df['2011']
        with pytest.raises(KeyError, match="'2011'"):
            df.loc[('2011', 0)]

    def test_astype_assignment(self):
        df_orig = DataFrame([['1', '2', '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        df = df_orig.copy()
        df.iloc[:, 0:2] = df.iloc[:, 0:2].astype(np.int64)
        expected = DataFrame([[1, 2, '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.iloc[:, 0:2] = df.iloc[:, 0:2]._convert(datetime=True, numeric=True)
        expected = DataFrame([[1, 2, '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.loc[:, 'A'] = df.loc[:, 'A'].astype(np.int64)
        expected = DataFrame([[1, '2', '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.loc[:, ['B', 'C']] = df.loc[:, ['B', 'C']].astype(np.int64)
        expected = DataFrame([['1', 2, 3, '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        df.iloc[:, 0] = df['A'].astype(np.int64)
        expected = DataFrame({'A': [1, 2, 3, 4]})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        df.loc[:, 'A'] = df['A'].astype(np.int64)
        expected = DataFrame({'A': [1, 2, 3, 4]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [tm.getitem, tm.loc])
    def test_index_type_coercion(self, indexer):
        for s in [Series(range(5)), Series(range(5), index=range(1, 6))]:
            assert s.index.is_integer()
            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert s2.index.is_floating()
            assert (indexer(s2)[0.1] == 0)
            s2 = s.copy()
            indexer(s2)[0.0] = 0
            exp = s.index
            if (0 not in s):
                exp = Index((s.index.tolist() + [0]))
            tm.assert_index_equal(s2.index, exp)
            s2 = s.copy()
            indexer(s2)['0'] = 0
            assert s2.index.is_object()
        for s in [Series(range(5), index=np.arange(5.0))]:
            assert s.index.is_floating()
            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert s2.index.is_floating()
            assert (indexer(s2)[0.1] == 0)
            s2 = s.copy()
            indexer(s2)[0.0] = 0
            tm.assert_index_equal(s2.index, s.index)
            s2 = s.copy()
            indexer(s2)['0'] = 0
            assert s2.index.is_object()

class TestMisc():

    def test_float_index_to_mixed(self):
        df = DataFrame({0.0: np.random.rand(10), 1.0: np.random.rand(10)})
        df['a'] = 10
        tm.assert_frame_equal(DataFrame({0.0: df[0.0], 1.0: df[1.0], 'a': ([10] * 10)}), df)

    def test_float_index_non_scalar_assignment(self):
        df = DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        df.loc[df.index[:2]] = 1
        expected = DataFrame({'a': [1, 1, 3], 'b': [1, 1, 5]}, index=df.index)
        tm.assert_frame_equal(expected, df)
        df = DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        df2 = df.copy()
        df.loc[df.index] = df.loc[df.index]
        tm.assert_frame_equal(df, df2)

    def test_float_index_at_iat(self):
        s = Series([1, 2, 3], index=[0.1, 0.2, 0.3])
        for (el, item) in s.items():
            assert (s.at[el] == item)
        for i in range(len(s)):
            assert (s.iat[i] == (i + 1))

    def test_rhs_alignment(self):

        def run_tests(df, rhs, right_loc, right_iloc):
            (lbl_one, idx_one, slice_one) = (list('bcd'), [1, 2, 3], slice(1, 4))
            (lbl_two, idx_two, slice_two) = (['joe', 'jolie'], [1, 2], slice(1, 3))
            left = df.copy()
            left.loc[(lbl_one, lbl_two)] = rhs
            tm.assert_frame_equal(left, right_loc)
            left = df.copy()
            left.iloc[(idx_one, idx_two)] = rhs
            tm.assert_frame_equal(left, right_iloc)
            left = df.copy()
            left.iloc[(slice_one, slice_two)] = rhs
            tm.assert_frame_equal(left, right_iloc)
        xs = np.arange(20).reshape(5, 4)
        cols = ['jim', 'joe', 'jolie', 'joline']
        df = DataFrame(xs, columns=cols, index=list('abcde'), dtype='int64')
        rhs = ((- 2) * df.iloc[3:0:(- 1), 2:0:(- 1)])
        right_iloc = df.copy()
        right_iloc['joe'] = [1, 14, 10, 6, 17]
        right_iloc['jolie'] = [2, 13, 9, 5, 18]
        right_iloc.iloc[1:4, 1:3] *= (- 2)
        right_loc = df.copy()
        right_loc.iloc[1:4, 1:3] *= (- 2)
        run_tests(df, rhs, right_loc, right_iloc)
        for frame in [df, rhs, right_loc, right_iloc]:
            frame['joe'] = frame['joe'].astype('float64')
            frame['jolie'] = frame['jolie'].map('@{}'.format)
        right_iloc['joe'] = [1.0, '@-28', '@-20', '@-12', 17.0]
        right_iloc['jolie'] = ['@2', (- 26.0), (- 18.0), (- 10.0), '@18']
        run_tests(df, rhs, right_loc, right_iloc)

    def test_str_label_slicing_with_negative_step(self):
        SLC = pd.IndexSlice

        def assert_slices_equivalent(l_slc, i_slc):
            tm.assert_series_equal(s.loc[l_slc], s.iloc[i_slc])
            if (not idx.is_integer):
                tm.assert_series_equal(s[l_slc], s.iloc[i_slc])
                tm.assert_series_equal(s.loc[l_slc], s.iloc[i_slc])
        for idx in [_mklbl('A', 20), (np.arange(20) + 100), np.linspace(100, 150, 20)]:
            idx = Index(idx)
            s = Series(np.arange(20), index=idx)
            assert_slices_equivalent(SLC[idx[9]::(- 1)], SLC[9::(- 1)])
            assert_slices_equivalent(SLC[:idx[9]:(- 1)], SLC[:8:(- 1)])
            assert_slices_equivalent(SLC[idx[13]:idx[9]:(- 1)], SLC[13:8:(- 1)])
            assert_slices_equivalent(SLC[idx[9]:idx[13]:(- 1)], SLC[:0])

    def test_slice_with_zero_step_raises(self):
        s = Series(np.arange(20), index=_mklbl('A', 20))
        with pytest.raises(ValueError, match='slice step cannot be zero'):
            s[::0]
        with pytest.raises(ValueError, match='slice step cannot be zero'):
            s.loc[::0]

    def test_indexing_assignment_dict_already_exists(self):
        index = Index([(- 5), 0, 5], name='z')
        df = DataFrame({'x': [1, 2, 6], 'y': [2, 2, 8]}, index=index)
        expected = df.copy()
        rhs = {'x': 9, 'y': 99}
        df.loc[5] = rhs
        expected.loc[5] = [9, 99]
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'x': [1, 2, 6], 'y': [2.0, 2.0, 8.0]}, index=index)
        df.loc[5] = rhs
        expected = DataFrame({'x': [1, 2, 9], 'y': [2.0, 2.0, 99.0]}, index=index)
        tm.assert_frame_equal(df, expected)

    def test_indexing_dtypes_on_empty(self):
        df = DataFrame({'a': [1, 2, 3], 'b': ['b', 'b2', 'b3']})
        df2 = df.iloc[[], :]
        assert (df2.loc[:, 'a'].dtype == np.int64)
        tm.assert_series_equal(df2.loc[:, 'a'], df2.iloc[:, 0])

    @pytest.mark.parametrize('size', [5, 999999, 1000000])
    def test_range_in_series_indexing(self, size):
        s = Series(index=range(size), dtype=np.float64)
        s.loc[range(1)] = 42
        tm.assert_series_equal(s.loc[range(1)], Series(42.0, index=[0]))
        s.loc[range(2)] = 43
        tm.assert_series_equal(s.loc[range(2)], Series(43.0, index=[0, 1]))

    @pytest.mark.parametrize('slc', [pd.IndexSlice[:, :], pd.IndexSlice[:, 1], pd.IndexSlice[1, :], pd.IndexSlice[([1], [1])], pd.IndexSlice[(1, [1])], pd.IndexSlice[([1], 1)], pd.IndexSlice[1], pd.IndexSlice[(1, 1)], slice(None, None, None), [0, 1], np.array([0, 1]), Series([0, 1])])
    def test_non_reducing_slice(self, slc):
        df = DataFrame([[0, 1], [2, 3]])
        tslice_ = non_reducing_slice(slc)
        assert isinstance(df.loc[tslice_], DataFrame)

    def test_list_slice(self):
        slices = [['A'], Series(['A']), np.array(['A'])]
        df = DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['A', 'B'])
        expected = pd.IndexSlice[:, ['A']]
        for subset in slices:
            result = non_reducing_slice(subset)
            tm.assert_frame_equal(df.loc[result], df.loc[expected])

    def test_maybe_numeric_slice(self):
        df = DataFrame({'A': [1, 2], 'B': ['c', 'd'], 'C': [True, False]})
        result = maybe_numeric_slice(df, slice_=None)
        expected = pd.IndexSlice[:, ['A']]
        assert (result == expected)
        result = maybe_numeric_slice(df, None, include_bool=True)
        expected = pd.IndexSlice[:, ['A', 'C']]
        assert all((result[1] == expected[1]))
        result = maybe_numeric_slice(df, [1])
        expected = [1]
        assert (result == expected)

    def test_partial_boolean_frame_indexing(self):
        df = DataFrame(np.arange(9.0).reshape(3, 3), index=list('abc'), columns=list('ABC'))
        index_df = DataFrame(1, index=list('ab'), columns=list('AB'))
        result = df[index_df.notnull()]
        expected = DataFrame(np.array([[0.0, 1.0, np.nan], [3.0, 4.0, np.nan], ([np.nan] * 3)]), index=list('abc'), columns=list('ABC'))
        tm.assert_frame_equal(result, expected)

    def test_no_reference_cycle(self):
        df = DataFrame({'a': [0, 1], 'b': [2, 3]})
        for name in ('loc', 'iloc', 'at', 'iat'):
            getattr(df, name)
        wr = weakref.ref(df)
        del df
        assert (wr() is None)

    def test_label_indexing_on_nan(self):
        df = Series([1, '{1,2}', 1, None])
        vc = df.value_counts(dropna=False)
        result1 = vc.loc[np.nan]
        result2 = vc[np.nan]
        expected = 1
        assert (result1 == expected)
        assert (result2 == expected)

class TestSeriesNoneCoercion():
    EXPECTED_RESULTS = [([1, 2, 3], [np.nan, 2, 3]), ([1.0, 2.0, 3.0], [np.nan, 2.0, 3.0]), ([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)]), (['foo', 'bar', 'baz'], [None, 'bar', 'baz'])]

    @pytest.mark.parametrize('start_data,expected_result', EXPECTED_RESULTS)
    def test_coercion_with_setitem(self, start_data, expected_result):
        start_series = Series(start_data)
        start_series[0] = None
        expected_series = Series(expected_result)
        tm.assert_series_equal(start_series, expected_series)

    @pytest.mark.parametrize('start_data,expected_result', EXPECTED_RESULTS)
    def test_coercion_with_loc_setitem(self, start_data, expected_result):
        start_series = Series(start_data)
        start_series.loc[0] = None
        expected_series = Series(expected_result)
        tm.assert_series_equal(start_series, expected_series)

    @pytest.mark.parametrize('start_data,expected_result', EXPECTED_RESULTS)
    def test_coercion_with_setitem_and_series(self, start_data, expected_result):
        start_series = Series(start_data)
        start_series[(start_series == start_series[0])] = None
        expected_series = Series(expected_result)
        tm.assert_series_equal(start_series, expected_series)

    @pytest.mark.parametrize('start_data,expected_result', EXPECTED_RESULTS)
    def test_coercion_with_loc_and_series(self, start_data, expected_result):
        start_series = Series(start_data)
        start_series.loc[(start_series == start_series[0])] = None
        expected_series = Series(expected_result)
        tm.assert_series_equal(start_series, expected_series)

class TestDataframeNoneCoercion():
    EXPECTED_SINGLE_ROW_RESULTS = [([1, 2, 3], [np.nan, 2, 3]), ([1.0, 2.0, 3.0], [np.nan, 2.0, 3.0]), ([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)]), (['foo', 'bar', 'baz'], [None, 'bar', 'baz'])]

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_loc(self, expected):
        (start_data, expected_result) = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe.loc[(0, ['foo'])] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_setitem_and_dataframe(self, expected):
        (start_data, expected_result) = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe[(start_dataframe['foo'] == start_dataframe['foo'][0])] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_none_coercion_loc_and_dataframe(self, expected):
        (start_data, expected_result) = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe.loc[(start_dataframe['foo'] == start_dataframe['foo'][0])] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    def test_none_coercion_mixed_dtypes(self):
        start_dataframe = DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0], 'c': [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], 'd': ['a', 'b', 'c']})
        start_dataframe.iloc[0] = None
        exp = DataFrame({'a': [np.nan, 2, 3], 'b': [np.nan, 2.0, 3.0], 'c': [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)], 'd': [None, 'b', 'c']})
        tm.assert_frame_equal(start_dataframe, exp)

class TestDatetimelikeCoercion():

    @pytest.mark.parametrize('indexer', [tm.setitem, tm.loc, tm.iloc])
    def test_setitem_dt64_string_scalar(self, tz_naive_fixture, indexer):
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        ser = Series(dti)
        values = ser._values
        newval = '2018-01-01'
        values._validate_setitem_value(newval)
        indexer(ser)[0] = newval
        if (tz is None):
            assert (ser.dtype == dti.dtype)
            assert (ser._values._data is values._data)
        else:
            assert (ser._values is values)

    @pytest.mark.parametrize('box', [list, np.array, pd.array])
    @pytest.mark.parametrize('key', [[0, 1], slice(0, 2), np.array([True, True, False])])
    @pytest.mark.parametrize('indexer', [tm.setitem, tm.loc, tm.iloc])
    def test_setitem_dt64_string_values(self, tz_naive_fixture, indexer, key, box):
        tz = tz_naive_fixture
        if (isinstance(key, slice) and (indexer is tm.loc)):
            key = slice(0, 1)
        dti = date_range('2016-01-01', periods=3, tz=tz)
        ser = Series(dti)
        values = ser._values
        newvals = box(['2019-01-01', '2010-01-02'])
        values._validate_setitem_value(newvals)
        indexer(ser)[key] = newvals
        if (tz is None):
            assert (ser.dtype == dti.dtype)
            assert (ser._values._data is values._data)
        else:
            assert (ser._values is values)

    @pytest.mark.parametrize('scalar', ['3 Days', offsets.Hour(4)])
    @pytest.mark.parametrize('indexer', [tm.setitem, tm.loc, tm.iloc])
    def test_setitem_td64_scalar(self, indexer, scalar):
        tdi = timedelta_range('1 Day', periods=3)
        ser = Series(tdi)
        values = ser._values
        values._validate_setitem_value(scalar)
        indexer(ser)[0] = scalar
        assert (ser._values._data is values._data)

    @pytest.mark.parametrize('box', [list, np.array, pd.array])
    @pytest.mark.parametrize('key', [[0, 1], slice(0, 2), np.array([True, True, False])])
    @pytest.mark.parametrize('indexer', [tm.setitem, tm.loc, tm.iloc])
    def test_setitem_td64_string_values(self, indexer, key, box):
        if (isinstance(key, slice) and (indexer is tm.loc)):
            key = slice(0, 1)
        tdi = timedelta_range('1 Day', periods=3)
        ser = Series(tdi)
        values = ser._values
        newvals = box(['10 Days', '44 hours'])
        values._validate_setitem_value(newvals)
        indexer(ser)[key] = newvals
        assert (ser._values._data is values._data)

def test_extension_array_cross_section():
    df = DataFrame({'A': pd.array([1, 2], dtype='Int64'), 'B': pd.array([3, 4], dtype='Int64')}, index=['a', 'b'])
    expected = Series(pd.array([1, 3], dtype='Int64'), index=['A', 'B'], name='a')
    result = df.loc['a']
    tm.assert_series_equal(result, expected)
    result = df.iloc[0]
    tm.assert_series_equal(result, expected)

def test_extension_array_cross_section_converts():
    df = DataFrame({'A': pd.array([1, 2], dtype='Int64'), 'B': np.array([1, 2])}, index=['a', 'b'])
    result = df.loc['a']
    expected = Series([1, 1], dtype='Int64', index=['A', 'B'], name='a')
    tm.assert_series_equal(result, expected)
    result = df.iloc[0]
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': pd.array([1, 2], dtype='Int64'), 'B': np.array(['a', 'b'])}, index=['a', 'b'])
    result = df.loc['a']
    expected = Series([1, 'a'], dtype=object, index=['A', 'B'], name='a')
    tm.assert_series_equal(result, expected)
    result = df.iloc[0]
    tm.assert_series_equal(result, expected)

def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length():
    ser = Series(([None] * 10))
    mask = ((([False] * 3) + ([True] * 5)) + ([False] * 2))
    ser[mask] = range(5)
    result = ser
    expected = Series(((([None] * 3) + list(range(5))) + ([None] * 2))).astype('object')
    tm.assert_series_equal(result, expected)

def test_missing_labels_inside_loc_matched_in_error_message():
    s = Series({'a': 1, 'b': 2, 'c': 3})
    error_message_regex = 'missing_0.*missing_1.*missing_2'
    with pytest.raises(KeyError, match=error_message_regex):
        s.loc[['a', 'b', 'missing_0', 'c', 'missing_1', 'missing_2']]

def test_many_missing_labels_inside_loc_error_message_limited():
    n = 10000
    missing_labels = [f'missing_{label}' for label in range(n)]
    s = Series({'a': 1, 'b': 2, 'c': 3})
    error_message_regex = 'missing_4.*\\.\\.\\..*missing_9995'
    with pytest.raises(KeyError, match=error_message_regex):
        s.loc[(['a', 'c'] + missing_labels)]

def test_long_text_missing_labels_inside_loc_error_message_limited():
    s = Series({'a': 1, 'b': 2, 'c': 3})
    missing_labels = [(f'long_missing_label_text_{i}' * 5) for i in range(3)]
    error_message_regex = 'long_missing_label_text_0.*\\\\n.*long_missing_label_text_1'
    with pytest.raises(KeyError, match=error_message_regex):
        s.loc[(['a', 'c'] + missing_labels)]

def test_setitem_categorical():
    df = DataFrame({'h': Series(list('mn')).astype('category')})
    df.h = df.h.cat.reorder_categories(['n', 'm'])
    expected = DataFrame({'h': pd.Categorical(['m', 'n']).reorder_categories(['n', 'm'])})
    tm.assert_frame_equal(df, expected)
