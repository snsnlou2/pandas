
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm

@pytest.mark.parametrize('ordered', [True, False])
@pytest.mark.parametrize('categories', [['b', 'a', 'c'], ['a', 'b', 'c', 'd']])
def test_factorize(categories, ordered):
    cat = pd.Categorical(['b', 'b', 'a', 'c', None], categories=categories, ordered=ordered)
    (codes, uniques) = pd.factorize(cat)
    expected_codes = np.array([0, 0, 1, 2, (- 1)], dtype=np.intp)
    expected_uniques = pd.Categorical(['b', 'a', 'c'], categories=categories, ordered=ordered)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)

def test_factorized_sort():
    cat = pd.Categorical(['b', 'b', None, 'a'])
    (codes, uniques) = pd.factorize(cat, sort=True)
    expected_codes = np.array([1, 1, (- 1), 0], dtype=np.intp)
    expected_uniques = pd.Categorical(['a', 'b'])
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)

def test_factorized_sort_ordered():
    cat = pd.Categorical(['b', 'b', None, 'a'], categories=['c', 'b', 'a'], ordered=True)
    (codes, uniques) = pd.factorize(cat, sort=True)
    expected_codes = np.array([0, 0, (- 1), 1], dtype=np.intp)
    expected_uniques = pd.Categorical(['b', 'a'], categories=['c', 'b', 'a'], ordered=True)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)

def test_isin_cats():
    cat = pd.Categorical(['a', 'b', np.nan])
    result = cat.isin(['a', np.nan])
    expected = np.array([True, False, True], dtype=bool)
    tm.assert_numpy_array_equal(expected, result)
    result = cat.isin(['a', 'c'])
    expected = np.array([True, False, False], dtype=bool)
    tm.assert_numpy_array_equal(expected, result)

@pytest.mark.parametrize('empty', [[], pd.Series(dtype=object), np.array([])])
def test_isin_empty(empty):
    s = pd.Categorical(['a', 'b'])
    expected = np.array([False, False], dtype=bool)
    result = s.isin(empty)
    tm.assert_numpy_array_equal(expected, result)

def test_diff():
    s = pd.Series([1, 2, 3], dtype='category')
    with tm.assert_produces_warning(FutureWarning):
        result = s.diff()
    expected = pd.Series([np.nan, 1, 1])
    tm.assert_series_equal(result, expected)
    expected = expected.to_frame(name='A')
    df = s.to_frame(name='A')
    with tm.assert_produces_warning(FutureWarning):
        result = df.diff()
    tm.assert_frame_equal(result, expected)
