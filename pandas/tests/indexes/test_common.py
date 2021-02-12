
'\nCollection of tests asserting things that should be true for\nany index subclass. Makes use of the `indices` fixture defined\nin pandas/tests/indexes/conftest.py.\n'
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.core.dtypes.common import is_period_dtype, needs_i8_conversion
import pandas as pd
from pandas import CategoricalIndex, DatetimeIndex, Int64Index, MultiIndex, PeriodIndex, RangeIndex, TimedeltaIndex, UInt64Index
import pandas._testing as tm

class TestCommon():

    def test_droplevel(self, index):
        if isinstance(index, MultiIndex):
            return
        assert index.droplevel([]).equals(index)
        for level in (index.name, [index.name]):
            if (isinstance(index.name, tuple) and (level is index.name)):
                continue
            msg = 'Cannot remove 1 levels from an index with 1 levels: at least one level must be left.'
            with pytest.raises(ValueError, match=msg):
                index.droplevel(level)
        for level in ('wrong', ['wrong']):
            with pytest.raises(KeyError, match="'Requested level \\(wrong\\) does not match index name \\(None\\)'"):
                index.droplevel(level)

    def test_constructor_non_hashable_name(self, index):
        if isinstance(index, MultiIndex):
            pytest.skip('multiindex handled in test_multi.py')
        message = 'Index.name must be a hashable type'
        renamed = [['1']]
        with pytest.raises(TypeError, match=message):
            index.rename(name=renamed)
        with pytest.raises(TypeError, match=message):
            index.set_names(names=renamed)

    def test_constructor_unwraps_index(self, index):
        if isinstance(index, pd.MultiIndex):
            raise pytest.skip('MultiIndex has no ._data')
        a = index
        b = type(a)(a)
        tm.assert_equal(a._data, b._data)

    @pytest.mark.parametrize('itm', [101, 'no_int'])
    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_getitem_error(self, index, itm):
        msg = ('index 101 is out of bounds for axis 0 with size [\\d]+|' + re.escape('only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices'))
        with pytest.raises(IndexError, match=msg):
            index[itm]

    def test_to_flat_index(self, index):
        if isinstance(index, MultiIndex):
            pytest.skip('Separate expectation for MultiIndex')
        result = index.to_flat_index()
        tm.assert_index_equal(result, index)

    def test_set_name_methods(self, index):
        new_name = 'This is the new name for this index'
        if isinstance(index, MultiIndex):
            pytest.skip('Skip check for MultiIndex')
        original_name = index.name
        new_ind = index.set_names([new_name])
        assert (new_ind.name == new_name)
        assert (index.name == original_name)
        res = index.rename(new_name, inplace=True)
        assert (res is None)
        assert (index.name == new_name)
        assert (index.names == [new_name])
        with pytest.raises(ValueError, match='Level must be None'):
            index.set_names('a', level=0)
        name = ('A', 'B')
        index.rename(name, inplace=True)
        assert (index.name == name)
        assert (index.names == [name])

    def test_copy_and_deepcopy(self, index):
        from copy import copy, deepcopy
        if isinstance(index, MultiIndex):
            pytest.skip('Skip check for MultiIndex')
        for func in (copy, deepcopy):
            idx_copy = func(index)
            assert (idx_copy is not index)
            assert idx_copy.equals(index)
        new_copy = index.copy(deep=True, name='banana')
        assert (new_copy.name == 'banana')

    def test_unique(self, index):
        if isinstance(index, (MultiIndex, CategoricalIndex)):
            pytest.skip('Skip check for MultiIndex/CategoricalIndex')
        expected = index.drop_duplicates()
        for level in (0, index.name, None):
            result = index.unique(level=level)
            tm.assert_index_equal(result, expected)
        msg = 'Too many levels: Index has only 1 level, not 4'
        with pytest.raises(IndexError, match=msg):
            index.unique(level=3)
        msg = f'Requested level \(wrong\) does not match index name \({re.escape(index.name.__repr__())}\)'
        with pytest.raises(KeyError, match=msg):
            index.unique(level='wrong')

    def test_get_unique_index(self, index):
        if ((not len(index)) or isinstance(index, MultiIndex)):
            pytest.skip('Skip check for empty Index and MultiIndex')
        idx = index[([0] * 5)]
        idx_unique = index[[0]]
        assert (idx_unique.is_unique is True)
        try:
            assert (idx_unique.hasnans is False)
        except NotImplementedError:
            pass
        for dropna in [False, True]:
            result = idx._get_unique_index(dropna=dropna)
            tm.assert_index_equal(result, idx_unique)
        if (not index._can_hold_na):
            pytest.skip('Skip na-check if index cannot hold na')
        if is_period_dtype(index.dtype):
            vals = index[([0] * 5)]._data
            vals[0] = pd.NaT
        elif needs_i8_conversion(index.dtype):
            vals = index.asi8[([0] * 5)]
            vals[0] = iNaT
        else:
            vals = index.values[([0] * 5)]
            vals[0] = np.nan
        vals_unique = vals[:2]
        if (index.dtype.kind in ['m', 'M']):
            vals = type(index._data)._simple_new(vals, dtype=index.dtype)
            vals_unique = type(index._data)._simple_new(vals_unique, dtype=index.dtype)
        idx_nan = index._shallow_copy(vals)
        idx_unique_nan = index._shallow_copy(vals_unique)
        assert (idx_unique_nan.is_unique is True)
        assert (idx_nan.dtype == index.dtype)
        assert (idx_unique_nan.dtype == index.dtype)
        for (dropna, expected) in zip([False, True], [idx_unique_nan, idx_unique]):
            for i in [idx_nan, idx_unique_nan]:
                result = i._get_unique_index(dropna=dropna)
                tm.assert_index_equal(result, expected)

    def test_view(self, index):
        assert (index.view().name == index.name)

    def test_searchsorted_monotonic(self, index):
        if isinstance(index, (MultiIndex, pd.IntervalIndex)):
            pytest.skip('Skip check for MultiIndex/IntervalIndex')
        if index.empty:
            pytest.skip('Skip check for empty Index')
        value = index[0]
        (expected_left, expected_right) = (0, (index == value).argmin())
        if (expected_right == 0):
            expected_right = len(index)
        if index.is_monotonic_increasing:
            ssm_left = index._searchsorted_monotonic(value, side='left')
            assert (expected_left == ssm_left)
            ssm_right = index._searchsorted_monotonic(value, side='right')
            assert (expected_right == ssm_right)
            ss_left = index.searchsorted(value, side='left')
            assert (expected_left == ss_left)
            ss_right = index.searchsorted(value, side='right')
            assert (expected_right == ss_right)
        elif index.is_monotonic_decreasing:
            ssm_left = index._searchsorted_monotonic(value, side='left')
            assert (expected_left == ssm_left)
            ssm_right = index._searchsorted_monotonic(value, side='right')
            assert (expected_right == ssm_right)
        else:
            msg = 'index must be monotonic increasing or decreasing'
            with pytest.raises(ValueError, match=msg):
                index._searchsorted_monotonic(value, side='left')

    def test_pickle(self, index):
        (original_name, index.name) = (index.name, 'foo')
        unpickled = tm.round_trip_pickle(index)
        assert index.equals(unpickled)
        index.name = original_name

    def test_drop_duplicates(self, index, keep):
        if isinstance(index, MultiIndex):
            pytest.skip('MultiIndex is tested separately')
        if isinstance(index, RangeIndex):
            pytest.skip('RangeIndex is tested in test_drop_duplicates_no_duplicates as it cannot hold duplicates')
        if (len(index) == 0):
            pytest.skip('empty index is tested in test_drop_duplicates_no_duplicates as it cannot hold duplicates')
        holder = type(index)
        unique_values = list(set(index))
        unique_idx = holder(unique_values)
        n = len(unique_idx)
        duplicated_selection = np.random.choice(n, int((n * 1.5)))
        idx = holder(unique_idx.values[duplicated_selection])
        expected_duplicated = pd.Series(duplicated_selection).duplicated(keep=keep).values
        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected_duplicated)
        expected_dropped = holder(pd.Series(idx).drop_duplicates(keep=keep))
        tm.assert_index_equal(idx.drop_duplicates(keep=keep), expected_dropped)

    def test_drop_duplicates_no_duplicates(self, index):
        if isinstance(index, MultiIndex):
            pytest.skip('MultiIndex is tested separately')
        if isinstance(index, RangeIndex):
            unique_idx = index
        else:
            holder = type(index)
            unique_values = list(set(index))
            unique_idx = holder(unique_values)
        expected_duplicated = np.array(([False] * len(unique_idx)), dtype='bool')
        tm.assert_numpy_array_equal(unique_idx.duplicated(), expected_duplicated)
        result_dropped = unique_idx.drop_duplicates()
        tm.assert_index_equal(result_dropped, unique_idx)
        assert (result_dropped is not unique_idx)

    def test_drop_duplicates_inplace(self, index):
        msg = 'drop_duplicates\\(\\) got an unexpected keyword argument'
        with pytest.raises(TypeError, match=msg):
            index.drop_duplicates(inplace=True)

    def test_has_duplicates(self, index):
        holder = type(index)
        if ((not len(index)) or isinstance(index, (MultiIndex, RangeIndex))):
            pytest.skip('Skip check for empty Index, MultiIndex, and RangeIndex')
        idx = holder(([index[0]] * 5))
        assert (idx.is_unique is False)
        assert (idx.has_duplicates is True)

    @pytest.mark.parametrize('dtype', ['int64', 'uint64', 'float64', 'category', 'datetime64[ns]', 'timedelta64[ns]'])
    def test_astype_preserves_name(self, index, dtype):
        if isinstance(index, MultiIndex):
            index.names = [('idx' + str(i)) for i in range(index.nlevels)]
        else:
            index.name = 'idx'
        warn = None
        if (dtype in ['int64', 'uint64']):
            if needs_i8_conversion(index.dtype):
                warn = FutureWarning
        try:
            with tm.assert_produces_warning(warn, check_stacklevel=False):
                result = index.astype(dtype)
        except (ValueError, TypeError, NotImplementedError, SystemError):
            return
        if isinstance(index, MultiIndex):
            assert (result.names == index.names)
        else:
            assert (result.name == index.name)

    def test_ravel_deprecation(self, index):
        with tm.assert_produces_warning(FutureWarning):
            index.ravel()

    @pytest.mark.xfail(reason='GH38630', strict=False)
    def test_asi8_deprecation(self, index):
        if isinstance(index, (Int64Index, UInt64Index, DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            warn = None
        else:
            warn = FutureWarning
        with tm.assert_produces_warning(warn):
            index.asi8

@pytest.mark.parametrize('na_position', [None, 'middle'])
def test_sort_values_invalid_na_position(index_with_missing, na_position):
    if isinstance(index_with_missing, (CategoricalIndex, MultiIndex)):
        pytest.xfail('missing value sorting order not defined for index type')
    if (na_position not in ['first', 'last']):
        with pytest.raises(ValueError, match=f'invalid na_position: {na_position}'):
            index_with_missing.sort_values(na_position=na_position)

@pytest.mark.parametrize('na_position', ['first', 'last'])
def test_sort_values_with_missing(index_with_missing, na_position):
    if isinstance(index_with_missing, (CategoricalIndex, MultiIndex)):
        pytest.xfail('missing value sorting order not defined for index type')
    missing_count = np.sum(index_with_missing.isna())
    not_na_vals = index_with_missing[index_with_missing.notna()].values
    sorted_values = np.sort(not_na_vals)
    if (na_position == 'first'):
        sorted_values = np.concatenate([([None] * missing_count), sorted_values])
    else:
        sorted_values = np.concatenate([sorted_values, ([None] * missing_count)])
    expected = type(index_with_missing)(sorted_values)
    result = index_with_missing.sort_values(na_position=na_position)
    tm.assert_index_equal(result, expected)
