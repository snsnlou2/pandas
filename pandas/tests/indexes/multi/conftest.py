
import numpy as np
import pytest
import pandas as pd
from pandas import Index, MultiIndex

@pytest.fixture
def idx():
    major_axis = Index(['foo', 'bar', 'baz', 'qux'])
    minor_axis = Index(['one', 'two'])
    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ['first', 'second']
    mi = MultiIndex(levels=[major_axis, minor_axis], codes=[major_codes, minor_codes], names=index_names, verify_integrity=False)
    return mi

@pytest.fixture
def idx_dup():
    major_axis = Index(['foo', 'bar', 'baz', 'qux'])
    minor_axis = Index(['one', 'two'])
    major_codes = np.array([0, 0, 1, 0, 1, 1])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ['first', 'second']
    mi = MultiIndex(levels=[major_axis, minor_axis], codes=[major_codes, minor_codes], names=index_names, verify_integrity=False)
    return mi

@pytest.fixture
def index_names():
    return ['first', 'second']

@pytest.fixture
def compat_props():
    return ['shape', 'ndim', 'size']

@pytest.fixture
def narrow_multi_index():
    '\n    Return a MultiIndex that is narrower than the display (<80 characters).\n    '
    n = 1000
    ci = pd.CategoricalIndex((list(('a' * n)) + (['abc'] * n)))
    dti = pd.date_range('2000-01-01', freq='s', periods=(n * 2))
    return MultiIndex.from_arrays([ci, (ci.codes + 9), dti], names=['a', 'b', 'dti'])

@pytest.fixture
def wide_multi_index():
    '\n    Return a MultiIndex that is wider than the display (>80 characters).\n    '
    n = 1000
    ci = pd.CategoricalIndex((list(('a' * n)) + (['abc'] * n)))
    dti = pd.date_range('2000-01-01', freq='s', periods=(n * 2))
    levels = [ci, (ci.codes + 9), dti, dti, dti]
    names = ['a', 'b', 'dti_1', 'dti_2', 'dti_3']
    return MultiIndex.from_arrays(levels, names=names)
