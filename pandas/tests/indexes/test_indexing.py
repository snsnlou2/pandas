
'\ntest_indexing tests the following Index methods:\n    __getitem__\n    get_loc\n    get_value\n    __contains__\n    take\n    where\n    get_indexer\n    slice_locs\n    asof_locs\n\nThe corresponding tests.indexes.[index_type].test_indexing files\ncontain tests for the corresponding methods specific to those Index subclasses.\n'
import numpy as np
import pytest
from pandas import DatetimeIndex, Float64Index, Index, Int64Index, PeriodIndex, TimedeltaIndex, UInt64Index
import pandas._testing as tm

class TestTake():

    def test_take_invalid_kwargs(self, index):
        indices = [1, 2]
        msg = "take\\(\\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            index.take(indices, foo=2)
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            index.take(indices, out=indices)
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            index.take(indices, mode='clip')

    def test_take(self, index):
        indexer = [4, 3, 0, 2]
        if (len(index) < 5):
            return
        result = index.take(indexer)
        expected = index[indexer]
        assert result.equals(expected)
        if (not isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex))):
            msg = "'(.*Index)' object has no attribute 'freq'"
            with pytest.raises(AttributeError, match=msg):
                index.freq

    def test_take_minus1_without_fill(self, index):
        if (len(index) == 0):
            return
        result = index.take([0, 0, (- 1)])
        expected = index.take([0, 0, (len(index) - 1)])
        tm.assert_index_equal(result, expected)

class TestContains():

    @pytest.mark.parametrize('index,val', [(Index([0, 1, 2]), 2), (Index([0, 1, '2']), '2'), (Index([0, 1, 2, np.inf, 4]), 4), (Index([0, 1, 2, np.nan, 4]), 4), (Index([0, 1, 2, np.inf]), np.inf), (Index([0, 1, 2, np.nan]), np.nan)])
    def test_index_contains(self, index, val):
        assert (val in index)

    @pytest.mark.parametrize('index,val', [(Index([0, 1, 2]), '2'), (Index([0, 1, '2']), 2), (Index([0, 1, 2, np.inf]), 4), (Index([0, 1, 2, np.nan]), 4), (Index([0, 1, 2, np.inf]), np.nan), (Index([0, 1, 2, np.nan]), np.inf), (Int64Index([0, 1, 2]), np.inf), (Int64Index([0, 1, 2]), np.nan), (UInt64Index([0, 1, 2]), np.inf), (UInt64Index([0, 1, 2]), np.nan)])
    def test_index_not_contains(self, index, val):
        assert (val not in index)

    @pytest.mark.parametrize('index,val', [(Index([0, 1, '2']), 0), (Index([0, 1, '2']), '2')])
    def test_mixed_index_contains(self, index, val):
        assert (val in index)

    @pytest.mark.parametrize('index,val', [(Index([0, 1, '2']), '1'), (Index([0, 1, '2']), 2)])
    def test_mixed_index_not_contains(self, index, val):
        assert (val not in index)

    def test_contains_with_float_index(self):
        integer_index = Int64Index([0, 1, 2, 3])
        uinteger_index = UInt64Index([0, 1, 2, 3])
        float_index = Float64Index([0.1, 1.1, 2.2, 3.3])
        for index in (integer_index, uinteger_index):
            assert (1.1 not in index)
            assert (1.0 in index)
            assert (1 in index)
        assert (1.1 in float_index)
        assert (1.0 not in float_index)
        assert (1 not in float_index)

@pytest.mark.parametrize('idx', [Index([1, 2, 3]), Index([0.1, 0.2, 0.3]), Index(['a', 'b', 'c'])])
def test_getitem_deprecated_float(idx):
    with tm.assert_produces_warning(FutureWarning):
        result = idx[1.0]
    expected = idx[1]
    assert (result == expected)
