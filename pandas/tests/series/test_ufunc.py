
from collections import deque
import string
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
UNARY_UFUNCS = [np.positive, np.floor, np.exp]
BINARY_UFUNCS = [np.add, np.logaddexp]
SPARSE = [True, False]
SPARSE_IDS = ['sparse', 'dense']
SHUFFLE = [True, False]

@pytest.fixture
def arrays_for_binary_ufunc():
    '\n    A pair of random, length-100 integer-dtype arrays, that are mostly 0.\n    '
    a1 = np.random.randint(0, 10, 100, dtype='int64')
    a2 = np.random.randint(0, 10, 100, dtype='int64')
    a1[::3] = 0
    a2[::4] = 0
    return (a1, a2)

@pytest.mark.parametrize('ufunc', UNARY_UFUNCS)
@pytest.mark.parametrize('sparse', SPARSE, ids=SPARSE_IDS)
def test_unary_ufunc(ufunc, sparse):
    array = np.random.randint(0, 10, 10, dtype='int64')
    array[::2] = 0
    if sparse:
        array = SparseArray(array, dtype=pd.SparseDtype('int64', 0))
    index = list(string.ascii_letters[:10])
    name = 'name'
    series = pd.Series(array, index=index, name=name)
    result = ufunc(series)
    expected = pd.Series(ufunc(array), index=index, name=name)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ufunc', BINARY_UFUNCS)
@pytest.mark.parametrize('sparse', SPARSE, ids=SPARSE_IDS)
@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_array(flip, sparse, ufunc, arrays_for_binary_ufunc):
    (a1, a2) = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    name = 'name'
    series = pd.Series(a1, name=name)
    other = a2
    array_args = (a1, a2)
    series_args = (series, other)
    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)
    expected = pd.Series(ufunc(*array_args), name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ufunc', BINARY_UFUNCS)
@pytest.mark.parametrize('sparse', SPARSE, ids=SPARSE_IDS)
@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_index(flip, sparse, ufunc, arrays_for_binary_ufunc):
    (a1, a2) = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    name = 'name'
    series = pd.Series(a1, name=name)
    other = pd.Index(a2, name=name).astype('int64')
    array_args = (a1, a2)
    series_args = (series, other)
    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)
    expected = pd.Series(ufunc(*array_args), name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ufunc', BINARY_UFUNCS)
@pytest.mark.parametrize('sparse', SPARSE, ids=SPARSE_IDS)
@pytest.mark.parametrize('shuffle', [True, False], ids=['unaligned', 'aligned'])
@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_series(flip, shuffle, sparse, ufunc, arrays_for_binary_ufunc):
    (a1, a2) = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    name = 'name'
    series = pd.Series(a1, name=name)
    other = pd.Series(a2, name=name)
    idx = np.random.permutation(len(a1))
    if shuffle:
        other = other.take(idx)
        if flip:
            index = other.align(series)[0].index
        else:
            index = series.align(other)[0].index
    else:
        index = series.index
    array_args = (a1, a2)
    series_args = (series, other)
    if flip:
        array_args = tuple(reversed(array_args))
        series_args = tuple(reversed(series_args))
    expected = pd.Series(ufunc(*array_args), index=index, name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ufunc', BINARY_UFUNCS)
@pytest.mark.parametrize('sparse', SPARSE, ids=SPARSE_IDS)
@pytest.mark.parametrize('flip', [True, False])
def test_binary_ufunc_scalar(ufunc, sparse, flip, arrays_for_binary_ufunc):
    (array, _) = arrays_for_binary_ufunc
    if sparse:
        array = SparseArray(array)
    other = 2
    series = pd.Series(array, name='name')
    series_args = (series, other)
    array_args = (array, other)
    if flip:
        series_args = tuple(reversed(series_args))
        array_args = tuple(reversed(array_args))
    expected = pd.Series(ufunc(*array_args), name='name')
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ufunc', [np.divmod])
@pytest.mark.parametrize('sparse', SPARSE, ids=SPARSE_IDS)
@pytest.mark.parametrize('shuffle', SHUFFLE)
@pytest.mark.filterwarnings('ignore:divide by zero:RuntimeWarning')
def test_multiple_output_binary_ufuncs(ufunc, sparse, shuffle, arrays_for_binary_ufunc):
    if (sparse and (ufunc is np.divmod)):
        pytest.skip('sparse divmod not implemented.')
    (a1, a2) = arrays_for_binary_ufunc
    a1[(a1 == 0)] = 1
    a2[(a2 == 0)] = 1
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    s1 = pd.Series(a1)
    s2 = pd.Series(a2)
    if shuffle:
        s2 = s2.sample(frac=1)
    expected = ufunc(a1, a2)
    assert isinstance(expected, tuple)
    result = ufunc(s1, s2)
    assert isinstance(result, tuple)
    tm.assert_series_equal(result[0], pd.Series(expected[0]))
    tm.assert_series_equal(result[1], pd.Series(expected[1]))

@pytest.mark.parametrize('sparse', SPARSE, ids=SPARSE_IDS)
def test_multiple_output_ufunc(sparse, arrays_for_binary_ufunc):
    (array, _) = arrays_for_binary_ufunc
    if sparse:
        array = SparseArray(array)
    series = pd.Series(array, name='name')
    result = np.modf(series)
    expected = np.modf(array)
    assert isinstance(result, tuple)
    assert isinstance(expected, tuple)
    tm.assert_series_equal(result[0], pd.Series(expected[0], name='name'))
    tm.assert_series_equal(result[1], pd.Series(expected[1], name='name'))

@pytest.mark.parametrize('sparse', SPARSE, ids=SPARSE_IDS)
@pytest.mark.parametrize('ufunc', BINARY_UFUNCS)
def test_binary_ufunc_drops_series_name(ufunc, sparse, arrays_for_binary_ufunc):
    (a1, a2) = arrays_for_binary_ufunc
    s1 = pd.Series(a1, name='a')
    s2 = pd.Series(a2, name='b')
    result = ufunc(s1, s2)
    assert (result.name is None)

def test_object_series_ok():

    class Dummy():

        def __init__(self, value):
            self.value = value

        def __add__(self, other):
            return (self.value + other.value)
    arr = np.array([Dummy(0), Dummy(1)])
    ser = pd.Series(arr)
    tm.assert_series_equal(np.add(ser, ser), pd.Series(np.add(ser, arr)))
    tm.assert_series_equal(np.add(ser, Dummy(1)), pd.Series(np.add(ser, Dummy(1))))

@pytest.mark.parametrize('values', [pd.array([1, 3, 2], dtype='int64'), pd.array([1, 10, 0], dtype='Sparse[int]'), pd.to_datetime(['2000', '2010', '2001']), pd.to_datetime(['2000', '2010', '2001']).tz_localize('CET'), pd.to_datetime(['2000', '2010', '2001']).to_period(freq='D')])
def test_reduce(values):
    a = pd.Series(values)
    assert (np.maximum.reduce(a) == values[1])

@pytest.mark.parametrize('type_', [list, deque, tuple])
def test_binary_ufunc_other_types(type_):
    a = pd.Series([1, 2, 3], name='name')
    b = type_([3, 4, 5])
    result = np.add(a, b)
    expected = pd.Series(np.add(a.to_numpy(), b), name='name')
    tm.assert_series_equal(result, expected)

def test_object_dtype_ok():

    class Thing():

        def __init__(self, value):
            self.value = value

        def __add__(self, other):
            other = getattr(other, 'value', other)
            return type(self)((self.value + other))

        def __eq__(self, other) -> bool:
            return ((type(other) is Thing) and (self.value == other.value))

        def __repr__(self) -> str:
            return f'Thing({self.value})'
    s = pd.Series([Thing(1), Thing(2)])
    result = np.add(s, Thing(1))
    expected = pd.Series([Thing(2), Thing(3)])
    tm.assert_series_equal(result, expected)

def test_outer():
    s = pd.Series([1, 2, 3])
    o = np.array([1, 2, 3])
    with pytest.raises(NotImplementedError, match=tm.EMPTY_STRING_PATTERN):
        np.subtract.outer(s, o)
