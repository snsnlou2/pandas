
from datetime import timedelta
import numpy as np
import pytest
from pandas.errors import InvalidIndexError, PerformanceWarning
import pandas as pd
from pandas import Categorical, Index, MultiIndex, date_range
import pandas._testing as tm

class TestSliceLocs():

    def test_slice_locs_partial(self, idx):
        (sorted_idx, _) = idx.sortlevel(0)
        result = sorted_idx.slice_locs(('foo', 'two'), ('qux', 'one'))
        assert (result == (1, 5))
        result = sorted_idx.slice_locs(None, ('qux', 'one'))
        assert (result == (0, 5))
        result = sorted_idx.slice_locs(('foo', 'two'), None)
        assert (result == (1, len(sorted_idx)))
        result = sorted_idx.slice_locs('bar', 'baz')
        assert (result == (2, 4))

    def test_slice_locs(self):
        df = tm.makeTimeDataFrame()
        stacked = df.stack()
        idx = stacked.index
        slob = slice(*idx.slice_locs(df.index[5], df.index[15]))
        sliced = stacked[slob]
        expected = df[5:16].stack()
        tm.assert_almost_equal(sliced.values, expected.values)
        slob = slice(*idx.slice_locs((df.index[5] + timedelta(seconds=30)), (df.index[15] - timedelta(seconds=30))))
        sliced = stacked[slob]
        expected = df[6:15].stack()
        tm.assert_almost_equal(sliced.values, expected.values)

    def test_slice_locs_with_type_mismatch(self):
        df = tm.makeTimeDataFrame()
        stacked = df.stack()
        idx = stacked.index
        with pytest.raises(TypeError, match='^Level type mismatch'):
            idx.slice_locs((1, 3))
        with pytest.raises(TypeError, match='^Level type mismatch'):
            idx.slice_locs((df.index[5] + timedelta(seconds=30)), (5, 2))
        df = tm.makeCustomDataframe(5, 5)
        stacked = df.stack()
        idx = stacked.index
        with pytest.raises(TypeError, match='^Level type mismatch'):
            idx.slice_locs(timedelta(seconds=30))
        with pytest.raises(TypeError, match='^Level type mismatch'):
            idx.slice_locs(df.index[1], (16, 'a'))

    def test_slice_locs_not_sorted(self):
        index = MultiIndex(levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))], codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array([0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])])
        msg = '[Kk]ey length.*greater than MultiIndex lexsort depth'
        with pytest.raises(KeyError, match=msg):
            index.slice_locs((1, 0, 1), (2, 1, 0))
        (sorted_index, _) = index.sortlevel(0)
        sorted_index.slice_locs((1, 0, 1), (2, 1, 0))

    def test_slice_locs_not_contained(self):
        index = MultiIndex(levels=[[0, 2, 4, 6], [0, 2, 4]], codes=[[0, 0, 0, 1, 1, 2, 3, 3, 3], [0, 1, 2, 1, 2, 2, 0, 1, 2]])
        result = index.slice_locs((1, 0), (5, 2))
        assert (result == (3, 6))
        result = index.slice_locs(1, 5)
        assert (result == (3, 6))
        result = index.slice_locs((2, 2), (5, 2))
        assert (result == (3, 6))
        result = index.slice_locs(2, 5)
        assert (result == (3, 6))
        result = index.slice_locs((1, 0), (6, 3))
        assert (result == (3, 8))
        result = index.slice_locs((- 1), 10)
        assert (result == (0, len(index)))

    @pytest.mark.parametrize('index_arr,expected,start_idx,end_idx', [([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, None), ([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, 'b'), ([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, ('b', 'e')), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), None), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), 'c'), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), ('c', 'e'))])
    def test_slice_locs_with_missing_value(self, index_arr, expected, start_idx, end_idx):
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.slice_locs(start=start_idx, end=end_idx)
        assert (result == expected)

def test_putmask_with_wrong_mask(idx):
    msg = 'putmask: mask and data must be the same size'
    with pytest.raises(ValueError, match=msg):
        idx.putmask(np.ones((len(idx) + 1), np.bool_), 1)
    with pytest.raises(ValueError, match=msg):
        idx.putmask(np.ones((len(idx) - 1), np.bool_), 1)
    with pytest.raises(ValueError, match=msg):
        idx.putmask('foo', 1)

class TestGetIndexer():

    def test_get_indexer(self):
        major_axis = Index(np.arange(4))
        minor_axis = Index(np.arange(2))
        major_codes = np.array([0, 0, 1, 2, 2, 3, 3], dtype=np.intp)
        minor_codes = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.intp)
        index = MultiIndex(levels=[major_axis, minor_axis], codes=[major_codes, minor_codes])
        idx1 = index[:5]
        idx2 = index[[1, 3, 5]]
        r1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, np.array([1, 3, (- 1)], dtype=np.intp))
        r1 = idx2.get_indexer(idx1, method='pad')
        e1 = np.array([(- 1), 0, 0, 1, 1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)
        r2 = idx2.get_indexer(idx1[::(- 1)], method='pad')
        tm.assert_almost_equal(r2, e1[::(- 1)])
        rffill1 = idx2.get_indexer(idx1, method='ffill')
        tm.assert_almost_equal(r1, rffill1)
        r1 = idx2.get_indexer(idx1, method='backfill')
        e1 = np.array([0, 0, 1, 1, 2], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)
        r2 = idx2.get_indexer(idx1[::(- 1)], method='backfill')
        tm.assert_almost_equal(r2, e1[::(- 1)])
        rbfill1 = idx2.get_indexer(idx1, method='bfill')
        tm.assert_almost_equal(r1, rbfill1)
        r1 = idx1.get_indexer(idx2.values)
        rexp1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, rexp1)
        r1 = idx1.get_indexer([1, 2, 3])
        assert (r1 == [(- 1), (- 1), (- 1)]).all()
        idx1 = Index((list(range(10)) + list(range(10))))
        idx2 = Index(list(range(20)))
        msg = 'Reindexing only valid with uniquely valued Index objects'
        with pytest.raises(InvalidIndexError, match=msg):
            idx1.get_indexer(idx2)

    def test_get_indexer_nearest(self):
        midx = MultiIndex.from_tuples([('a', 1), ('b', 2)])
        msg = "method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365"
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(['a'], method='nearest')
        msg = 'tolerance not implemented yet for MultiIndex'
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(['a'], method='pad', tolerance=2)

    def test_get_indexer_categorical_time(self):
        midx = MultiIndex.from_product([Categorical(['a', 'b', 'c']), Categorical(date_range('2012-01-01', periods=3, freq='H'))])
        result = midx.get_indexer(midx)
        tm.assert_numpy_array_equal(result, np.arange(9, dtype=np.intp))

    @pytest.mark.parametrize('index_arr,labels,expected', [([[1, np.nan, 2], [3, 4, 5]], [1, np.nan, 2], np.array([(- 1), (- 1), (- 1)], dtype=np.intp)), ([[1, np.nan, 2], [3, 4, 5]], [(np.nan, 4)], np.array([1], dtype=np.intp)), ([[1, 2, 3], [np.nan, 4, 5]], [(1, np.nan)], np.array([0], dtype=np.intp)), ([[1, 2, 3], [np.nan, 4, 5]], [np.nan, 4, 5], np.array([(- 1), (- 1), (- 1)], dtype=np.intp))])
    def test_get_indexer_with_missing_value(self, index_arr, labels, expected):
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.get_indexer(labels)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_methods(self):
        mult_idx_1 = MultiIndex.from_product([[(- 1), 0, 1], [0, 2, 3, 4]])
        mult_idx_2 = MultiIndex.from_product([[0], [1, 3, 4]])
        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([(- 1), 6, 7], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method='backfill')
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method='bfill')
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method='pad')
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method='ffill')
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    def test_get_indexer_three_or_more_levels(self):
        mult_idx_1 = MultiIndex.from_product([[1, 3], [2, 4, 6], [5, 7]])
        mult_idx_2 = MultiIndex.from_tuples([(1, 1, 8), (1, 5, 9), (1, 6, 7), (2, 1, 6), (2, 7, 7), (2, 7, 8), (3, 6, 8)])
        assert mult_idx_1.is_monotonic
        assert mult_idx_1.is_unique
        assert mult_idx_2.is_monotonic
        assert mult_idx_2.is_unique
        assert (mult_idx_2[0] < mult_idx_1[0])
        assert (mult_idx_1[3] < mult_idx_2[1] < mult_idx_1[4])
        assert (mult_idx_1[5] == mult_idx_2[2])
        assert (mult_idx_1[5] < mult_idx_2[3] < mult_idx_1[6])
        assert (mult_idx_1[5] < mult_idx_2[4] < mult_idx_1[6])
        assert (mult_idx_1[5] < mult_idx_2[5] < mult_idx_1[6])
        assert (mult_idx_1[(- 1)] < mult_idx_2[6])
        indexer_no_fill = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([(- 1), (- 1), 5, (- 1), (- 1), (- 1), (- 1)], dtype=indexer_no_fill.dtype)
        tm.assert_almost_equal(expected, indexer_no_fill)
        indexer_backfilled = mult_idx_1.get_indexer(mult_idx_2, method='backfill')
        expected = np.array([0, 4, 5, 6, 6, 6, (- 1)], dtype=indexer_backfilled.dtype)
        tm.assert_almost_equal(expected, indexer_backfilled)
        indexer_padded = mult_idx_1.get_indexer(mult_idx_2, method='pad')
        expected = np.array([(- 1), 3, 5, 5, 5, 5, 11], dtype=indexer_padded.dtype)
        tm.assert_almost_equal(expected, indexer_padded)
        assert (mult_idx_2[0] < mult_idx_1[0] < mult_idx_2[1])
        assert (mult_idx_2[0] < mult_idx_1[1] < mult_idx_2[1])
        assert (mult_idx_2[0] < mult_idx_1[2] < mult_idx_2[1])
        assert (mult_idx_2[0] < mult_idx_1[3] < mult_idx_2[1])
        assert (mult_idx_2[1] < mult_idx_1[4] < mult_idx_2[2])
        assert (mult_idx_2[2] == mult_idx_1[5])
        assert (mult_idx_2[5] < mult_idx_1[6] < mult_idx_2[6])
        assert (mult_idx_2[5] < mult_idx_1[7] < mult_idx_2[6])
        assert (mult_idx_2[5] < mult_idx_1[8] < mult_idx_2[6])
        assert (mult_idx_2[5] < mult_idx_1[9] < mult_idx_2[6])
        assert (mult_idx_2[5] < mult_idx_1[10] < mult_idx_2[6])
        assert (mult_idx_2[5] < mult_idx_1[11] < mult_idx_2[6])
        indexer = mult_idx_2.get_indexer(mult_idx_1)
        expected = np.array([(- 1), (- 1), (- 1), (- 1), (- 1), 2, (- 1), (- 1), (- 1), (- 1), (- 1), (- 1)], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)
        backfill_indexer = mult_idx_2.get_indexer(mult_idx_1, method='bfill')
        expected = np.array([1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)
        pad_indexer = mult_idx_2.get_indexer(mult_idx_1, method='pad')
        expected = np.array([0, 0, 0, 0, 1, 2, 5, 5, 5, 5, 5, 5], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    def test_get_indexer_crossing_levels(self):
        mult_idx_1 = MultiIndex.from_product(([[1, 2]] * 4))
        mult_idx_2 = MultiIndex.from_tuples([(1, 3, 2, 2), (2, 3, 2, 2)])
        assert (mult_idx_1[7] < mult_idx_2[0] < mult_idx_1[8])
        assert (mult_idx_1[(- 1)] < mult_idx_2[1])
        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([(- 1), (- 1)], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method='bfill')
        expected = np.array([8, (- 1)], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method='ffill')
        expected = np.array([7, 15], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

def test_getitem(idx):
    assert (idx[2] == ('bar', 'one'))
    result = idx[2:5]
    expected = idx[[2, 3, 4]]
    assert result.equals(expected)
    result = idx[[True, False, True, False, True, True]]
    result2 = idx[np.array([True, False, True, False, True, True])]
    expected = idx[[0, 2, 4, 5]]
    assert result.equals(expected)
    assert result2.equals(expected)

def test_getitem_group_select(idx):
    (sorted_idx, _) = idx.sortlevel(0)
    assert (sorted_idx.get_loc('baz') == slice(3, 4))
    assert (sorted_idx.get_loc('foo') == slice(0, 2))

@pytest.mark.parametrize('ind1', [([True] * 5), Index(([True] * 5))])
@pytest.mark.parametrize('ind2', [[True, False, True, False, False], Index([True, False, True, False, False])])
def test_getitem_bool_index_all(ind1, ind2):
    idx = MultiIndex.from_tuples([(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)])
    tm.assert_index_equal(idx[ind1], idx)
    expected = MultiIndex.from_tuples([(10, 1), (30, 3)])
    tm.assert_index_equal(idx[ind2], expected)

@pytest.mark.parametrize('ind1', [[True], Index([True])])
@pytest.mark.parametrize('ind2', [[False], Index([False])])
def test_getitem_bool_index_single(ind1, ind2):
    idx = MultiIndex.from_tuples([(10, 1)])
    tm.assert_index_equal(idx[ind1], idx)
    expected = MultiIndex(levels=[np.array([], dtype=np.int64), np.array([], dtype=np.int64)], codes=[[], []])
    tm.assert_index_equal(idx[ind2], expected)

class TestGetLoc():

    def test_get_loc(self, idx):
        assert (idx.get_loc(('foo', 'two')) == 1)
        assert (idx.get_loc(('baz', 'two')) == 3)
        with pytest.raises(KeyError, match='^10$'):
            idx.get_loc(('bar', 'two'))
        with pytest.raises(KeyError, match="^'quux'$"):
            idx.get_loc('quux')
        msg = 'only the default get_loc method is currently supported for MultiIndex'
        with pytest.raises(NotImplementedError, match=msg):
            idx.get_loc('foo', method='nearest')
        index = MultiIndex(levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))], codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array([0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])])
        with pytest.raises(KeyError, match='^\\(1, 1\\)$'):
            index.get_loc((1, 1))
        assert (index.get_loc((2, 0)) == slice(3, 5))

    def test_get_loc_duplicates(self):
        index = Index([2, 2, 2, 2])
        result = index.get_loc(2)
        expected = slice(0, 4)
        assert (result == expected)
        index = Index(['c', 'a', 'a', 'b', 'b'])
        rs = index.get_loc('c')
        xp = 0
        assert (rs == xp)
        with pytest.raises(KeyError, match='2'):
            index.get_loc(2)

    def test_get_loc_level(self):
        index = MultiIndex(levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))], codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array([0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])])
        (loc, new_index) = index.get_loc_level((0, 1))
        expected = slice(1, 2)
        exp_index = index[expected].droplevel(0).droplevel(0)
        assert (loc == expected)
        assert new_index.equals(exp_index)
        (loc, new_index) = index.get_loc_level((0, 1, 0))
        expected = 1
        assert (loc == expected)
        assert (new_index is None)
        with pytest.raises(KeyError, match='^\\(2, 2\\)$'):
            index.get_loc_level((2, 2))
        with pytest.raises(KeyError, match='^2$'):
            index.drop(2).get_loc_level(2)
        with pytest.raises(KeyError, match='^2$'):
            index.drop(1, level=2).get_loc_level(2, level=2)
        index = MultiIndex(levels=[[2000], list(range(4))], codes=[np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])])
        (result, new_index) = index.get_loc_level((2000, slice(None, None)))
        expected = slice(None, None)
        assert (result == expected)
        assert new_index.equals(index.droplevel(0))

    @pytest.mark.parametrize('dtype1', [int, float, bool, str])
    @pytest.mark.parametrize('dtype2', [int, float, bool, str])
    def test_get_loc_multiple_dtypes(self, dtype1, dtype2):
        levels = [np.array([0, 1]).astype(dtype1), np.array([0, 1]).astype(dtype2)]
        idx = MultiIndex.from_product(levels)
        assert (idx.get_loc(idx[2]) == 2)

    @pytest.mark.parametrize('level', [0, 1])
    @pytest.mark.parametrize('dtypes', [[int, float], [float, int]])
    def test_get_loc_implicit_cast(self, level, dtypes):
        levels = [['a', 'b'], ['c', 'd']]
        key = ['b', 'd']
        (lev_dtype, key_dtype) = dtypes
        levels[level] = np.array([0, 1], dtype=lev_dtype)
        key[level] = key_dtype(1)
        idx = MultiIndex.from_product(levels)
        assert (idx.get_loc(tuple(key)) == 3)

    def test_get_loc_cast_bool(self):
        levels = [[False, True], np.arange(2, dtype='int64')]
        idx = MultiIndex.from_product(levels)
        assert (idx.get_loc((0, 1)) == 1)
        assert (idx.get_loc((1, 0)) == 2)
        with pytest.raises(KeyError, match='^\\(False, True\\)$'):
            idx.get_loc((False, True))
        with pytest.raises(KeyError, match='^\\(True, False\\)$'):
            idx.get_loc((True, False))

    @pytest.mark.parametrize('level', [0, 1])
    def test_get_loc_nan(self, level, nulls_fixture):
        levels = [['a', 'b'], ['c', 'd']]
        key = ['b', 'd']
        levels[level] = np.array([0, nulls_fixture], dtype=type(nulls_fixture))
        key[level] = nulls_fixture
        idx = MultiIndex.from_product(levels)
        assert (idx.get_loc(tuple(key)) == 3)

    def test_get_loc_missing_nan(self):
        idx = MultiIndex.from_arrays([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(idx.get_loc(1), slice)
        with pytest.raises(KeyError, match='^3$'):
            idx.get_loc(3)
        with pytest.raises(KeyError, match='^nan$'):
            idx.get_loc(np.nan)
        with pytest.raises(TypeError, match="unhashable type: 'list'"):
            idx.get_loc([np.nan])

    def test_get_loc_with_values_including_missing_values(self):
        idx = MultiIndex.from_product(([[np.nan, 1]] * 2))
        expected = slice(0, 2, None)
        assert (idx.get_loc(np.nan) == expected)
        idx = MultiIndex.from_arrays([[np.nan, 1, 2, np.nan]])
        expected = np.array([True, False, False, True])
        tm.assert_numpy_array_equal(idx.get_loc(np.nan), expected)
        idx = MultiIndex.from_product(([[np.nan, 1]] * 3))
        expected = slice(2, 4, None)
        assert (idx.get_loc((np.nan, 1)) == expected)

    def test_get_loc_duplicates2(self):
        index = MultiIndex(levels=[['D', 'B', 'C'], [0, 26, 27, 37, 57, 67, 75, 82]], codes=[[0, 0, 0, 1, 2, 2, 2, 2, 2, 2], [1, 3, 4, 6, 0, 2, 2, 3, 5, 7]], names=['tag', 'day'])
        assert (index.get_loc('D') == slice(0, 3))

    def test_get_loc_past_lexsort_depth(self):
        idx = MultiIndex(levels=[['a'], [0, 7], [1]], codes=[[0, 0], [1, 0], [0, 0]], names=['x', 'y', 'z'], sortorder=0)
        key = ('a', 7)
        with tm.assert_produces_warning(PerformanceWarning):
            result = idx.get_loc(key)
        assert (result == slice(0, 1, None))

    def test_multiindex_get_loc_list_raises(self):
        idx = MultiIndex.from_tuples([('a', 1), ('b', 2)])
        msg = 'unhashable type'
        with pytest.raises(TypeError, match=msg):
            idx.get_loc([])

class TestWhere():

    def test_where(self):
        i = MultiIndex.from_tuples([('A', 1), ('A', 2)])
        msg = '\\.where is not supported for MultiIndex operations'
        with pytest.raises(NotImplementedError, match=msg):
            i.where(True)

    @pytest.mark.parametrize('klass', [list, tuple, np.array, pd.Series])
    def test_where_array_like(self, klass):
        i = MultiIndex.from_tuples([('A', 1), ('A', 2)])
        cond = [False, True]
        msg = '\\.where is not supported for MultiIndex operations'
        with pytest.raises(NotImplementedError, match=msg):
            i.where(klass(cond))

class TestContains():

    def test_contains_top_level(self):
        midx = MultiIndex.from_product([['A', 'B'], [1, 2]])
        assert ('A' in midx)
        assert ('A' not in midx._engine)

    def test_contains_with_nat(self):
        mi = MultiIndex(levels=[['C'], date_range('2012-01-01', periods=5)], codes=[[0, 0, 0, 0, 0, 0], [(- 1), 0, 1, 2, 3, 4]], names=[None, 'B'])
        assert (('C', pd.Timestamp('2012-01-01')) in mi)
        for val in mi.values:
            assert (val in mi)

    def test_contains(self, idx):
        assert (('foo', 'two') in idx)
        assert (('bar', 'two') not in idx)
        assert (None not in idx)

    def test_contains_with_missing_value(self):
        idx = MultiIndex.from_arrays([[1, np.nan, 2]])
        assert (np.nan in idx)
        idx = MultiIndex.from_arrays([[1, 2], [np.nan, 3]])
        assert (np.nan not in idx)
        assert ((1, np.nan) in idx)

    def test_multiindex_contains_dropped(self):
        idx = MultiIndex.from_product([[1, 2], [3, 4]])
        assert (2 in idx)
        idx = idx.drop(2)
        assert (2 in idx.levels[0])
        assert (2 not in idx)
        idx = MultiIndex.from_product([['a', 'b'], ['c', 'd']])
        assert ('a' in idx)
        idx = idx.drop('a')
        assert ('a' in idx.levels[0])
        assert ('a' not in idx)

    def test_contains_td64_level(self):
        tx = pd.timedelta_range('09:30:00', '16:00:00', freq='30 min')
        idx = MultiIndex.from_arrays([tx, np.arange(len(tx))])
        assert (tx[0] in idx)
        assert ('element_not_exit' not in idx)
        assert ('0 day 09:30:00' in idx)

    @pytest.mark.slow
    def test_large_mi_contains(self):
        result = MultiIndex.from_arrays([range((10 ** 6)), range((10 ** 6))])
        assert (not (((10 ** 6), 0) in result))

def test_timestamp_multiindex_indexer():
    idx = MultiIndex.from_product([date_range('2019-01-01T00:15:33', periods=100, freq='H', name='date'), ['x'], [3]])
    df = pd.DataFrame({'foo': np.arange(len(idx))}, idx)
    result = df.loc[(pd.IndexSlice['2019-1-2':, 'x', :], 'foo')]
    qidx = MultiIndex.from_product([date_range(start='2019-01-02T00:15:33', end='2019-01-05T03:15:33', freq='H', name='date'), ['x'], [3]])
    should_be = pd.Series(data=np.arange(24, (len(qidx) + 24)), index=qidx, name='foo')
    tm.assert_series_equal(result, should_be)

@pytest.mark.parametrize('index_arr,expected,target,algo', [([[np.nan, 'a', 'b'], ['c', 'd', 'e']], 0, np.nan, 'left'), ([[np.nan, 'a', 'b'], ['c', 'd', 'e']], 1, (np.nan, 'c'), 'right'), ([['a', 'b', 'c'], ['d', np.nan, 'd']], 1, ('b', np.nan), 'left')])
def test_get_slice_bound_with_missing_value(index_arr, expected, target, algo):
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.get_slice_bound(target, side=algo, kind='loc')
    assert (result == expected)

@pytest.mark.parametrize('index_arr,expected,start_idx,end_idx', [([[np.nan, 1, 2], [3, 4, 5]], slice(0, 2, None), np.nan, 1), ([[np.nan, 1, 2], [3, 4, 5]], slice(0, 3, None), np.nan, (2, 5)), ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), 3), ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), (3, 5))])
def test_slice_indexer_with_missing_value(index_arr, expected, start_idx, end_idx):
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.slice_indexer(start=start_idx, end=end_idx)
    assert (result == expected)

def test_pyint_engine():
    N = 5
    keys = [tuple(arr) for arr in [(([0] * 10) * N), (([1] * 10) * N), (([2] * 10) * N), (([np.nan] * N) + (([2] * 9) * N)), (([0] * N) + (([2] * 9) * N)), ((([np.nan] * N) + (([2] * 8) * N)) + ([0] * N))]]
    for idx in range(len(keys)):
        index = MultiIndex.from_tuples(keys)
        assert (index.get_loc(keys[idx]) == idx)
        expected = np.arange((idx + 1), dtype=np.intp)
        result = index.get_indexer([keys[i] for i in expected])
        tm.assert_numpy_array_equal(result, expected)
    idces = range(len(keys))
    expected = np.array(([(- 1)] + list(idces)), dtype=np.intp)
    missing = tuple((([0, 1] * 5) * N))
    result = index.get_indexer(([missing] + [keys[i] for i in idces]))
    tm.assert_numpy_array_equal(result, expected)
