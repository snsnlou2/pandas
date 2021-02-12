
import numpy as np
import pytest
from pandas import Categorical, Index, MultiIndex, NaT
import pandas._testing as tm

def test_index_equal_levels_mismatch():
    msg = "Index are different\n\nIndex levels are different\n\\[left\\]:  1, Int64Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: 2, MultiIndex\\(\\[\\('A', 1\\),\n            \\('A', 2\\),\n            \\('B', 3\\),\n            \\('B', 4\\)\\],\n           \\)"
    idx1 = Index([1, 2, 3])
    idx2 = MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 3), ('B', 4)])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=False)

def test_index_equal_values_mismatch(check_exact):
    msg = "MultiIndex level \\[1\\] are different\n\nMultiIndex level \\[1\\] values are different \\(25\\.0 %\\)\n\\[left\\]:  Int64Index\\(\\[2, 2, 3, 4\\], dtype='int64'\\)\n\\[right\\]: Int64Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"
    idx1 = MultiIndex.from_tuples([('A', 2), ('A', 2), ('B', 3), ('B', 4)])
    idx2 = MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 3), ('B', 4)])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)

def test_index_equal_length_mismatch(check_exact):
    msg = "Index are different\n\nIndex length are different\n\\[left\\]:  3, Int64Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: 4, Int64Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3, 4])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)

def test_index_equal_class_mismatch(check_exact):
    msg = "Index are different\n\nIndex classes are different\n\\[left\\]:  Int64Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: Float64Index\\(\\[1\\.0, 2\\.0, 3\\.0\\], dtype='float64'\\)"
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3.0])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)

def test_index_equal_values_close(check_exact):
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0000000001])
    if check_exact:
        msg = "Index are different\n\nIndex values are different \\(33\\.33333 %\\)\n\\[left\\]:  Float64Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)\n\\[right\\]: Float64Index\\(\\[1.0, 2.0, 3.0000000001\\], dtype='float64'\\)"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_exact=check_exact)
    else:
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)

def test_index_equal_values_less_close(check_exact, rtol):
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0001])
    kwargs = {'check_exact': check_exact, 'rtol': rtol}
    if (check_exact or (rtol < 0.0005)):
        msg = "Index are different\n\nIndex values are different \\(33\\.33333 %\\)\n\\[left\\]:  Float64Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)\n\\[right\\]: Float64Index\\(\\[1.0, 2.0, 3.0001\\], dtype='float64'\\)"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, **kwargs)
    else:
        tm.assert_index_equal(idx1, idx2, **kwargs)

def test_index_equal_values_too_far(check_exact, rtol):
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 4])
    kwargs = {'check_exact': check_exact, 'rtol': rtol}
    msg = "Index are different\n\nIndex values are different \\(33\\.33333 %\\)\n\\[left\\]:  Int64Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: Int64Index\\(\\[1, 2, 4\\], dtype='int64'\\)"
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)

@pytest.mark.parametrize('check_order', [True, False])
def test_index_equal_value_oder_mismatch(check_exact, rtol, check_order):
    idx1 = Index([1, 2, 3])
    idx2 = Index([3, 2, 1])
    msg = "Index are different\n\nIndex values are different \\(66\\.66667 %\\)\n\\[left\\]:  Int64Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: Int64Index\\(\\[3, 2, 1\\], dtype='int64'\\)"
    if check_order:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=True)
    else:
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=False)

def test_index_equal_level_values_mismatch(check_exact, rtol):
    idx1 = MultiIndex.from_tuples([('A', 2), ('A', 2), ('B', 3), ('B', 4)])
    idx2 = MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 3), ('B', 4)])
    kwargs = {'check_exact': check_exact, 'rtol': rtol}
    msg = "MultiIndex level \\[1\\] are different\n\nMultiIndex level \\[1\\] values are different \\(25\\.0 %\\)\n\\[left\\]:  Int64Index\\(\\[2, 2, 3, 4\\], dtype='int64'\\)\n\\[right\\]: Int64Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)

@pytest.mark.parametrize('name1,name2', [(None, 'x'), ('x', 'x'), (np.nan, np.nan), (NaT, NaT), (np.nan, NaT)])
def test_index_equal_names(name1, name2):
    idx1 = Index([1, 2, 3], name=name1)
    idx2 = Index([1, 2, 3], name=name2)
    if ((name1 == name2) or (name1 is name2)):
        tm.assert_index_equal(idx1, idx2)
    else:
        name1 = ("'x'" if (name1 == 'x') else name1)
        name2 = ("'x'" if (name2 == 'x') else name2)
        msg = f'''Index are different

Attribute "names" are different
\[left\]:  \[{name1}\]
\[right\]: \[{name2}\]'''
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2)

def test_index_equal_category_mismatch(check_categorical):
    msg = 'Index are different\n\nAttribute "dtype" are different\n\\[left\\]:  CategoricalDtype\\(categories=\\[\'a\', \'b\'\\], ordered=False\\)\n\\[right\\]: CategoricalDtype\\(categories=\\[\'a\', \'b\', \'c\'\\], ordered=False\\)'
    idx1 = Index(Categorical(['a', 'b']))
    idx2 = Index(Categorical(['a', 'b'], categories=['a', 'b', 'c']))
    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
    else:
        tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
