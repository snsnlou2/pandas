
import numpy as np
import pytest
import pandas as pd
from pandas import Categorical
import pandas._testing as tm

@pytest.mark.parametrize('to_replace,value,expected,flip_categories', [(1, 2, [2, 2, 3], False), (1, 4, [4, 2, 3], False), (4, 1, [1, 2, 3], False), (5, 6, [1, 2, 3], False), ([1], 2, [2, 2, 3], False), ([1, 2], 3, [3, 3, 3], False), ([1, 2], 4, [4, 4, 3], False), ((1, 2, 4), 5, [5, 5, 3], False), ((5, 6), 2, [1, 2, 3], False), ([1], [2], [2, 2, 3], True), ([1, 4], [5, 2], [5, 2, 3], True), (3, '4', [1, 2, '4'], False), ([1, 2, '3'], '5', ['5', '5', 3], True)])
def test_replace(to_replace, value, expected, flip_categories):
    stays_categorical = ((not isinstance(value, list)) or (len(pd.unique(value)) == 1))
    s = pd.Series([1, 2, 3], dtype='category')
    result = s.replace(to_replace, value)
    expected = pd.Series(expected, dtype='category')
    s.replace(to_replace, value, inplace=True)
    if flip_categories:
        expected = expected.cat.set_categories(expected.cat.categories[::(- 1)])
    if (not stays_categorical):
        expected = pd.Series(np.asarray(expected))
    tm.assert_series_equal(expected, result, check_category_order=False)
    tm.assert_series_equal(expected, s, check_category_order=False)

@pytest.mark.parametrize('to_replace, value, result, expected_error_msg', [('b', 'c', ['a', 'c'], 'Categorical.categories are different'), ('c', 'd', ['a', 'b'], None), ('a', 'a', ['a', 'b'], None), ('b', None, ['a', None], 'Categorical.categories length are different')])
def test_replace2(to_replace, value, result, expected_error_msg):
    cat = Categorical(['a', 'b'])
    expected = Categorical(result)
    result = cat.replace(to_replace, value)
    tm.assert_categorical_equal(result, expected)
    if (to_replace == 'b'):
        with pytest.raises(AssertionError, match=expected_error_msg):
            tm.assert_categorical_equal(cat, expected)
    cat.replace(to_replace, value, inplace=True)
    tm.assert_categorical_equal(cat, expected)
