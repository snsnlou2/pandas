
from copy import copy, deepcopy
import pytest
from pandas import MultiIndex
import pandas._testing as tm

def assert_multiindex_copied(copy, original):
    tm.assert_copy(copy.levels, original.levels)
    tm.assert_almost_equal(copy.codes, original.codes)
    tm.assert_almost_equal(copy.codes, original.codes)
    assert (copy.codes is not original.codes)
    assert (copy.names == original.names)
    assert (copy.names is not original.names)
    assert (copy.sortorder == original.sortorder)

def test_copy(idx):
    i_copy = idx.copy()
    assert_multiindex_copied(i_copy, idx)

def test_shallow_copy(idx):
    i_copy = idx._shallow_copy()
    assert_multiindex_copied(i_copy, idx)

def test_view(idx):
    i_view = idx.view()
    assert_multiindex_copied(i_view, idx)

@pytest.mark.parametrize('func', [copy, deepcopy])
def test_copy_and_deepcopy(func):
    idx = MultiIndex(levels=[['foo', 'bar'], ['fizz', 'buzz']], codes=[[0, 0, 0, 1], [0, 0, 1, 1]], names=['first', 'second'])
    idx_copy = func(idx)
    assert (idx_copy is not idx)
    assert idx_copy.equals(idx)

@pytest.mark.parametrize('deep', [True, False])
def test_copy_method(deep):
    idx = MultiIndex(levels=[['foo', 'bar'], ['fizz', 'buzz']], codes=[[0, 0, 0, 1], [0, 0, 1, 1]], names=['first', 'second'])
    idx_copy = idx.copy(deep=deep)
    assert idx_copy.equals(idx)

@pytest.mark.parametrize('deep', [True, False])
@pytest.mark.parametrize('kwarg, value', [('names', ['third', 'fourth'])])
def test_copy_method_kwargs(deep, kwarg, value):
    idx = MultiIndex(levels=[['foo', 'bar'], ['fizz', 'buzz']], codes=[[0, 0, 0, 1], [0, 0, 1, 1]], names=['first', 'second'])
    idx_copy = idx.copy(**{kwarg: value, 'deep': deep})
    if (kwarg == 'names'):
        assert (getattr(idx_copy, kwarg) == value)
    else:
        assert ([list(i) for i in getattr(idx_copy, kwarg)] == value)

@pytest.mark.parametrize('deep', [True, False])
@pytest.mark.parametrize('param_name, param_value', [('levels', [['foo2', 'bar2'], ['fizz2', 'buzz2']]), ('codes', [[1, 0, 0, 0], [1, 1, 0, 0]])])
def test_copy_deprecated_parameters(deep, param_name, param_value):
    idx = MultiIndex(levels=[['foo', 'bar'], ['fizz', 'buzz']], codes=[[0, 0, 0, 1], [0, 0, 1, 1]], names=['first', 'second'])
    with tm.assert_produces_warning(FutureWarning):
        idx_copy = idx.copy(deep=deep, **{param_name: param_value})
    assert ([list(i) for i in getattr(idx_copy, param_name)] == param_value)
