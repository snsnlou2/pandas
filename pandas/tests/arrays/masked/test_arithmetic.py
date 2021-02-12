
from typing import Any, List
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_EA_INT_DTYPES]
scalars = ([2] * len(arrays))
arrays += [pd.array([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES]
scalars += [0.2, 0.2]
arrays += [pd.array([True, False, True, None], dtype='boolean')]
scalars += [False]

@pytest.fixture(params=zip(arrays, scalars), ids=[a.dtype.name for a in arrays])
def data(request):
    return request.param

def check_skip(data, op_name):
    if (isinstance(data.dtype, pd.BooleanDtype) and ('sub' in op_name)):
        pytest.skip('subtract not implemented for boolean')

def test_array_scalar_like_equivalence(data, all_arithmetic_operators):
    (data, scalar) = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)
    scalar_array = pd.array(([scalar] * len(data)), dtype=data.dtype)
    for scalar in [scalar, data.dtype.type(scalar)]:
        result = op(data, scalar)
        expected = op(data, scalar_array)
        tm.assert_extension_array_equal(result, expected)

def test_array_NA(data, all_arithmetic_operators):
    if ('truediv' in all_arithmetic_operators):
        pytest.skip('division with pd.NA raises')
    (data, _) = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)
    scalar = pd.NA
    scalar_array = pd.array(([pd.NA] * len(data)), dtype=data.dtype)
    result = op(data, scalar)
    expected = op(data, scalar_array)
    tm.assert_extension_array_equal(result, expected)

def test_numpy_array_equivalence(data, all_arithmetic_operators):
    (data, scalar) = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)
    numpy_array = np.array(([scalar] * len(data)), dtype=data.dtype.numpy_dtype)
    pd_array = pd.array(numpy_array, dtype=data.dtype)
    result = op(data, numpy_array)
    expected = op(data, pd_array)
    if isinstance(expected, ExtensionArray):
        tm.assert_extension_array_equal(result, expected)
    else:
        tm.assert_numpy_array_equal(result, expected)

def test_frame(data, all_arithmetic_operators):
    (data, scalar) = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)
    df = pd.DataFrame({'A': data})
    result = op(df, scalar)
    expected = pd.DataFrame({'A': op(data, scalar)})
    tm.assert_frame_equal(result, expected)

def test_series(data, all_arithmetic_operators):
    (data, scalar) = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)
    s = pd.Series(data)
    result = op(s, scalar)
    expected = pd.Series(op(data, scalar))
    tm.assert_series_equal(result, expected)
    other = np.array(([scalar] * len(data)), dtype=data.dtype.numpy_dtype)
    result = op(s, other)
    expected = pd.Series(op(data, other))
    tm.assert_series_equal(result, expected)
    other = pd.array(([scalar] * len(data)), dtype=data.dtype)
    result = op(s, other)
    expected = pd.Series(op(data, other))
    tm.assert_series_equal(result, expected)
    other = pd.Series(([scalar] * len(data)), dtype=data.dtype)
    result = op(s, other)
    expected = pd.Series(op(data, other.array))
    tm.assert_series_equal(result, expected)

def test_error_invalid_object(data, all_arithmetic_operators):
    (data, _) = data
    op = all_arithmetic_operators
    opa = getattr(data, op)
    result = opa(pd.DataFrame({'A': data}))
    assert (result is NotImplemented)
    msg = 'can only perform ops with 1-d structures'
    with pytest.raises(NotImplementedError, match=msg):
        opa(np.arange(len(data)).reshape((- 1), len(data)))

def test_error_len_mismatch(data, all_arithmetic_operators):
    (data, scalar) = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    other = ([scalar] * (len(data) - 1))
    for other in [other, np.array(other)]:
        with pytest.raises(ValueError, match='Lengths must match'):
            op(data, other)
        s = pd.Series(data)
        with pytest.raises(ValueError, match='Lengths must match'):
            op(s, other)
