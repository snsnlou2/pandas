
import numpy as np
import pytest
import pandas as pd
from pandas import Index, MultiIndex, Series
import pandas._testing as tm

def test_equals(idx):
    assert idx.equals(idx)
    assert idx.equals(idx.copy())
    assert idx.equals(idx.astype(object))
    assert idx.equals(idx.to_flat_index())
    assert idx.equals(idx.to_flat_index().astype('category'))
    assert (not idx.equals(list(idx)))
    assert (not idx.equals(np.array(idx)))
    same_values = Index(idx, dtype=object)
    assert idx.equals(same_values)
    assert same_values.equals(idx)
    if (idx.nlevels == 1):
        assert (not idx.equals(Series(idx)))

def test_equals_op(idx):
    index_a = idx
    n = len(index_a)
    index_b = index_a[0:(- 1)]
    index_c = index_a[0:(- 1)].append(index_a[(- 2):(- 1)])
    index_d = index_a[0:1]
    with pytest.raises(ValueError, match='Lengths must match'):
        (index_a == index_b)
    expected1 = np.array(([True] * n))
    expected2 = np.array((([True] * (n - 1)) + [False]))
    tm.assert_numpy_array_equal((index_a == index_a), expected1)
    tm.assert_numpy_array_equal((index_a == index_c), expected2)
    array_a = np.array(index_a)
    array_b = np.array(index_a[0:(- 1)])
    array_c = np.array(index_a[0:(- 1)].append(index_a[(- 2):(- 1)]))
    array_d = np.array(index_a[0:1])
    with pytest.raises(ValueError, match='Lengths must match'):
        (index_a == array_b)
    tm.assert_numpy_array_equal((index_a == array_a), expected1)
    tm.assert_numpy_array_equal((index_a == array_c), expected2)
    series_a = Series(array_a)
    series_b = Series(array_b)
    series_c = Series(array_c)
    series_d = Series(array_d)
    with pytest.raises(ValueError, match='Lengths must match'):
        (index_a == series_b)
    tm.assert_numpy_array_equal((index_a == series_a), expected1)
    tm.assert_numpy_array_equal((index_a == series_c), expected2)
    with pytest.raises(ValueError, match='Lengths must match'):
        (index_a == index_d)
    with pytest.raises(ValueError, match='Lengths must match'):
        (index_a == series_d)
    with pytest.raises(ValueError, match='Lengths must match'):
        (index_a == array_d)
    msg = 'Can only compare identically-labeled Series objects'
    with pytest.raises(ValueError, match=msg):
        (series_a == series_d)
    with pytest.raises(ValueError, match='Lengths must match'):
        (series_a == array_d)
    if (not isinstance(index_a, MultiIndex)):
        expected3 = np.array((([False] * (len(index_a) - 2)) + [True, False]))
        item = index_a[(- 2)]
        tm.assert_numpy_array_equal((index_a == item), expected3)
        tm.assert_series_equal((series_a == item), Series(expected3))

def test_compare_tuple():
    mi = MultiIndex.from_product(([[1, 2]] * 2))
    all_false = np.array([False, False, False, False])
    result = (mi == mi[0])
    expected = np.array([True, False, False, False])
    tm.assert_numpy_array_equal(result, expected)
    result = (mi != mi[0])
    tm.assert_numpy_array_equal(result, (~ expected))
    result = (mi < mi[0])
    tm.assert_numpy_array_equal(result, all_false)
    result = (mi <= mi[0])
    tm.assert_numpy_array_equal(result, expected)
    result = (mi > mi[0])
    tm.assert_numpy_array_equal(result, (~ expected))
    result = (mi >= mi[0])
    tm.assert_numpy_array_equal(result, (~ all_false))

def test_compare_tuple_strs():
    mi = MultiIndex.from_tuples([('a', 'b'), ('b', 'c'), ('c', 'a')])
    result = (mi == ('c', 'a'))
    expected = np.array([False, False, True])
    tm.assert_numpy_array_equal(result, expected)
    result = (mi == ('c',))
    expected = np.array([False, False, False])
    tm.assert_numpy_array_equal(result, expected)

def test_equals_multi(idx):
    assert idx.equals(idx)
    assert (not idx.equals(idx.values))
    assert idx.equals(Index(idx.values))
    assert idx.equal_levels(idx)
    assert (not idx.equals(idx[:(- 1)]))
    assert (not idx.equals(idx[(- 1)]))
    index = MultiIndex(levels=[Index(list(range(4))), Index(list(range(4))), Index(list(range(4)))], codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array([0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])])
    index2 = MultiIndex(levels=index.levels[:(- 1)], codes=index.codes[:(- 1)])
    assert (not index.equals(index2))
    assert (not index.equal_levels(index2))
    major_axis = Index(list(range(4)))
    minor_axis = Index(list(range(2)))
    major_codes = np.array([0, 0, 1, 2, 2, 3])
    minor_codes = np.array([0, 1, 0, 0, 1, 0])
    index = MultiIndex(levels=[major_axis, minor_axis], codes=[major_codes, minor_codes])
    assert (not idx.equals(index))
    assert (not idx.equal_levels(index))
    major_axis = Index(['foo', 'bar', 'baz', 'qux'])
    minor_axis = Index(['one', 'two'])
    major_codes = np.array([0, 0, 2, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index = MultiIndex(levels=[major_axis, minor_axis], codes=[major_codes, minor_codes])
    assert (not idx.equals(index))

def test_identical(idx):
    mi = idx.copy()
    mi2 = idx.copy()
    assert mi.identical(mi2)
    mi = mi.set_names(['new1', 'new2'])
    assert mi.equals(mi2)
    assert (not mi.identical(mi2))
    mi2 = mi2.set_names(['new1', 'new2'])
    assert mi.identical(mi2)
    with tm.assert_produces_warning(FutureWarning):
        mi3 = Index(mi.tolist(), names=mi.names)
    msg = "Unexpected keyword arguments {'names'}"
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(FutureWarning):
            Index(mi.tolist(), names=mi.names, tupleize_cols=False)
    mi4 = Index(mi.tolist(), tupleize_cols=False)
    assert mi.identical(mi3)
    assert (not mi.identical(mi4))
    assert mi.equals(mi4)

def test_equals_operator(idx):
    assert (idx == idx).all()

def test_equals_missing_values():
    i = MultiIndex.from_tuples([(0, pd.NaT), (0, pd.Timestamp('20130101'))])
    result = i[0:1].equals(i[0])
    assert (not result)
    result = i[1:2].equals(i[1])
    assert (not result)

def test_equals_missing_values_differently_sorted():
    mi1 = pd.MultiIndex.from_tuples([(81.0, np.nan), (np.nan, np.nan)])
    mi2 = pd.MultiIndex.from_tuples([(np.nan, np.nan), (81.0, np.nan)])
    assert (not mi1.equals(mi2))
    mi2 = pd.MultiIndex.from_tuples([(81.0, np.nan), (np.nan, np.nan)])
    assert mi1.equals(mi2)

def test_is_():
    mi = MultiIndex.from_tuples(zip(range(10), range(10)))
    assert mi.is_(mi)
    assert mi.is_(mi.view())
    assert mi.is_(mi.view().view().view().view())
    mi2 = mi.view()
    mi2.names = ['A', 'B']
    assert mi2.is_(mi)
    assert mi.is_(mi2)
    assert (not mi.is_(mi.set_names(['C', 'D'])))
    mi2 = mi.view()
    mi2.set_names(['E', 'F'], inplace=True)
    assert mi.is_(mi2)
    mi3 = mi2.set_levels([list(range(10)), list(range(10))])
    assert (not mi3.is_(mi2))
    assert mi2.is_(mi)
    mi4 = mi3.view()
    with tm.assert_produces_warning(FutureWarning):
        mi4.set_levels([list(range(10)), list(range(10))], inplace=True)
    assert (not mi4.is_(mi3))
    mi5 = mi.view()
    with tm.assert_produces_warning(FutureWarning):
        mi5.set_levels(mi5.levels, inplace=True)
    assert (not mi5.is_(mi))

def test_is_all_dates(idx):
    assert (not idx._is_all_dates)

def test_is_numeric(idx):
    assert (not idx.is_numeric())

def test_multiindex_compare():
    midx = MultiIndex.from_product([[0, 1]])
    expected = Series([True, True])
    result = Series((midx == midx))
    tm.assert_series_equal(result, expected)
    expected = Series([False, False])
    result = Series((midx > midx))
    tm.assert_series_equal(result, expected)
