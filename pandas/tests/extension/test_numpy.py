
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.numpy_ import PandasArray, PandasDtype
from . import base

@pytest.fixture(params=['float', 'object'])
def dtype(request):
    return PandasDtype(np.dtype(request.param))

@pytest.fixture
def allow_in_pandas(monkeypatch):
    "\n    A monkeypatch to tells pandas to let us in.\n\n    By default, passing a PandasArray to an index / series / frame\n    constructor will unbox that PandasArray to an ndarray, and treat\n    it as a non-EA column. We don't want people using EAs without\n    reason.\n\n    The mechanism for this is a check against ABCPandasArray\n    in each constructor.\n\n    But, for testing, we need to allow them in pandas. So we patch\n    the _typ of PandasArray, so that we evade the ABCPandasArray\n    check.\n    "
    with monkeypatch.context() as m:
        m.setattr(PandasArray, '_typ', 'extension')
        (yield)

@pytest.fixture
def data(allow_in_pandas, dtype):
    if (dtype.numpy_dtype == 'object'):
        return pd.Series([(i,) for i in range(100)]).array
    return PandasArray(np.arange(1, 101, dtype=dtype._dtype))

@pytest.fixture
def data_missing(allow_in_pandas, dtype):
    if (dtype.numpy_dtype == 'object'):
        return PandasArray(np.array([np.nan, (1,)], dtype=object))
    return PandasArray(np.array([np.nan, 1.0]))

@pytest.fixture
def na_value():
    return np.nan

@pytest.fixture
def na_cmp():

    def cmp(a, b):
        return (np.isnan(a) and np.isnan(b))
    return cmp

@pytest.fixture
def data_for_sorting(allow_in_pandas, dtype):
    'Length-3 array with a known sort order.\n\n    This should be three items [B, C, A] with\n    A < B < C\n    '
    if (dtype.numpy_dtype == 'object'):
        return PandasArray(np.array([(), (2,), (3,), (1,)], dtype=object)[1:])
    return PandasArray(np.array([1, 2, 0]))

@pytest.fixture
def data_missing_for_sorting(allow_in_pandas, dtype):
    'Length-3 array with a known sort order.\n\n    This should be three items [B, NA, A] with\n    A < B and NA missing.\n    '
    if (dtype.numpy_dtype == 'object'):
        return PandasArray(np.array([(1,), np.nan, (0,)], dtype=object))
    return PandasArray(np.array([1, np.nan, 0]))

@pytest.fixture
def data_for_grouping(allow_in_pandas, dtype):
    'Data for factorization, grouping, and unique tests.\n\n    Expected to be like [B, B, NA, NA, A, A, B, C]\n\n    Where A < B < C and NA is missing\n    '
    if (dtype.numpy_dtype == 'object'):
        (a, b, c) = ((1,), (2,), (3,))
    else:
        (a, b, c) = np.arange(3)
    return PandasArray(np.array([b, b, np.nan, np.nan, a, a, b, c], dtype=dtype.numpy_dtype))

@pytest.fixture
def skip_numpy_object(dtype):
    "\n    Tests for PandasArray with nested data. Users typically won't create\n    these objects via `pd.array`, but they can show up through `.array`\n    on a Series with nested data. Many of the base tests fail, as they aren't\n    appropriate for nested data.\n\n    This fixture allows these tests to be skipped when used as a usefixtures\n    marker to either an individual test or a test class.\n    "
    if (dtype == 'object'):
        raise pytest.skip('Skipping for object dtype.')
skip_nested = pytest.mark.usefixtures('skip_numpy_object')

class BaseNumPyTests():
    pass

class TestCasting(BaseNumPyTests, base.BaseCastingTests):

    @skip_nested
    def test_astype_str(self, data):
        super().test_astype_str(data)

    @skip_nested
    def test_astype_string(self, data):
        super().test_astype_string(data)

class TestConstructors(BaseNumPyTests, base.BaseConstructorsTests):

    @pytest.mark.skip(reason="We don't register our dtype")
    def test_from_dtype(self, data):
        pass

    @skip_nested
    def test_array_from_scalars(self, data):
        super().test_array_from_scalars(data)

    @skip_nested
    def test_series_constructor_scalar_with_index(self, data, dtype):
        super().test_series_constructor_scalar_with_index(data, dtype)

class TestDtype(BaseNumPyTests, base.BaseDtypeTests):

    @pytest.mark.skip(reason='Incorrect expected.')
    def test_check_dtype(self, data):
        pass

class TestGetitem(BaseNumPyTests, base.BaseGetitemTests):

    @skip_nested
    def test_getitem_scalar(self, data):
        super().test_getitem_scalar(data)

    @skip_nested
    def test_take_series(self, data):
        super().test_take_series(data)

    def test_loc_iloc_frame_single_dtype(self, data, request):
        npdtype = data.dtype.numpy_dtype
        if (npdtype == object):
            mark = pytest.mark.xfail(reason="GH#33125 astype doesn't recognize data.dtype")
            request.node.add_marker(mark)
        super().test_loc_iloc_frame_single_dtype(data)

class TestGroupby(BaseNumPyTests, base.BaseGroupbyTests):

    @skip_nested
    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op, request):
        super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)

class TestInterface(BaseNumPyTests, base.BaseInterfaceTests):

    @skip_nested
    def test_array_interface(self, data):
        super().test_array_interface(data)

class TestMethods(BaseNumPyTests, base.BaseMethodsTests):

    @pytest.mark.skip(reason='TODO: remove?')
    def test_value_counts(self, all_data, dropna):
        pass

    @pytest.mark.xfail(reason='not working. will be covered by #32028')
    def test_value_counts_with_normalize(self, data):
        return super().test_value_counts_with_normalize(data)

    @pytest.mark.skip(reason='Incorrect expected')
    def test_combine_le(self, data_repeated):
        super().test_combine_le(data_repeated)

    @skip_nested
    def test_combine_add(self, data_repeated):
        super().test_combine_add(data_repeated)

    @skip_nested
    def test_shift_fill_value(self, data):
        super().test_shift_fill_value(data)

    @skip_nested
    @pytest.mark.parametrize('box', [pd.Series, (lambda x: x)])
    @pytest.mark.parametrize('method', [(lambda x: x.unique()), pd.unique])
    def test_unique(self, data, box, method):
        super().test_unique(data, box, method)

    @skip_nested
    def test_fillna_copy_frame(self, data_missing):
        super().test_fillna_copy_frame(data_missing)

    @skip_nested
    def test_fillna_copy_series(self, data_missing):
        super().test_fillna_copy_series(data_missing)

    @skip_nested
    def test_hash_pandas_object_works(self, data, as_frame):
        super().test_hash_pandas_object_works(data, as_frame)

    @skip_nested
    def test_searchsorted(self, data_for_sorting, as_series):
        super().test_searchsorted(data_for_sorting, as_series)

    @skip_nested
    def test_where_series(self, data, na_value, as_frame):
        super().test_where_series(data, na_value, as_frame)

    @skip_nested
    @pytest.mark.parametrize('repeats', [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data, repeats, as_series, use_numpy):
        super().test_repeat(data, repeats, as_series, use_numpy)

    @pytest.mark.xfail(reason='PandasArray.diff may fail on dtype')
    def test_diff(self, data, periods):
        return super().test_diff(data, periods)

    @skip_nested
    @pytest.mark.parametrize('box', [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        super().test_equals(data, na_value, as_series, box)

@skip_nested
class TestArithmetics(BaseNumPyTests, base.BaseArithmeticOpsTests):
    divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    def test_divmod_series_array(self, data):
        s = pd.Series(data)
        self._check_divmod_op(s, divmod, data, exc=None)

    @pytest.mark.skip('We implement ops')
    def test_error(self, data, all_arithmetic_operators):
        pass

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        super().test_arith_series_with_array(data, all_arithmetic_operators)

class TestPrinting(BaseNumPyTests, base.BasePrintingTests):
    pass

@skip_nested
class TestNumericReduce(BaseNumPyTests, base.BaseNumericReduceTests):

    def check_reduce(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        expected = getattr(s.astype(s.dtype._dtype), op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

@skip_nested
class TestBooleanReduce(BaseNumPyTests, base.BaseBooleanReduceTests):
    pass

class TestMissing(BaseNumPyTests, base.BaseMissingTests):

    @skip_nested
    def test_fillna_scalar(self, data_missing):
        super().test_fillna_scalar(data_missing)

    @skip_nested
    def test_fillna_series_method(self, data_missing, fillna_method):
        super().test_fillna_series_method(data_missing, fillna_method)

    @skip_nested
    def test_fillna_series(self, data_missing):
        super().test_fillna_series(data_missing)

    @skip_nested
    def test_fillna_frame(self, data_missing):
        super().test_fillna_frame(data_missing)

    @pytest.mark.skip('Invalid test')
    def test_fillna_fill_other(self, data):
        super().test_fillna_fill_other(data_missing)

class TestReshaping(BaseNumPyTests, base.BaseReshapingTests):

    @pytest.mark.skip('Incorrect parent test')
    def test_concat_mixed_dtypes(self, data):
        super().test_concat_mixed_dtypes(data)

    @pytest.mark.xfail(reason='GH#33125 PandasArray.astype does not recognize PandasDtype')
    def test_concat(self, data, in_frame):
        super().test_concat(data, in_frame)

    @pytest.mark.xfail(reason='GH#33125 PandasArray.astype does not recognize PandasDtype')
    def test_concat_all_na_block(self, data_missing, in_frame):
        super().test_concat_all_na_block(data_missing, in_frame)

    @skip_nested
    def test_merge(self, data, na_value):
        super().test_merge(data, na_value)

    @skip_nested
    def test_merge_on_extension_array(self, data):
        super().test_merge_on_extension_array(data)

    @skip_nested
    def test_merge_on_extension_array_duplicates(self, data):
        super().test_merge_on_extension_array_duplicates(data)

    @skip_nested
    def test_transpose_frame(self, data):
        super().test_transpose_frame(data)

class TestSetitem(BaseNumPyTests, base.BaseSetitemTests):

    @skip_nested
    def test_setitem_scalar_series(self, data, box_in_series):
        super().test_setitem_scalar_series(data, box_in_series)

    @skip_nested
    def test_setitem_sequence(self, data, box_in_series):
        super().test_setitem_sequence(data, box_in_series)

    @skip_nested
    def test_setitem_sequence_mismatched_length_raises(self, data, as_array):
        super().test_setitem_sequence_mismatched_length_raises(data, as_array)

    @skip_nested
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        super().test_setitem_sequence_broadcasts(data, box_in_series)

    @skip_nested
    def test_setitem_loc_scalar_mixed(self, data):
        super().test_setitem_loc_scalar_mixed(data)

    @skip_nested
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        super().test_setitem_loc_scalar_multiple_homogoneous(data)

    @skip_nested
    def test_setitem_iloc_scalar_mixed(self, data):
        super().test_setitem_iloc_scalar_mixed(data)

    @skip_nested
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        super().test_setitem_iloc_scalar_multiple_homogoneous(data)

    @skip_nested
    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_broadcast(self, data, setter):
        super().test_setitem_mask_broadcast(data, setter)

    @skip_nested
    def test_setitem_scalar_key_sequence_raise(self, data):
        super().test_setitem_scalar_key_sequence_raise(data)

    @skip_nested
    @pytest.mark.parametrize('mask', [np.array([True, True, True, False, False]), pd.array([True, True, True, False, False], dtype='boolean')], ids=['numpy-array', 'boolean-array'])
    def test_setitem_mask(self, data, mask, box_in_series):
        super().test_setitem_mask(data, mask, box_in_series)

    @skip_nested
    def test_setitem_mask_raises(self, data, box_in_series):
        super().test_setitem_mask_raises(data, box_in_series)

    @skip_nested
    @pytest.mark.parametrize('idx', [[0, 1, 2], pd.array([0, 1, 2], dtype='Int64'), np.array([0, 1, 2])], ids=['list', 'integer-array', 'numpy-array'])
    def test_setitem_integer_array(self, data, idx, box_in_series):
        super().test_setitem_integer_array(data, idx, box_in_series)

    @skip_nested
    @pytest.mark.parametrize('idx, box_in_series', [([0, 1, 2, pd.NA], False), pytest.param([0, 1, 2, pd.NA], True, marks=pytest.mark.xfail), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False)], ids=['list-False', 'list-True', 'integer-array-False', 'integer-array-True'])
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)

    @skip_nested
    def test_setitem_slice(self, data, box_in_series):
        super().test_setitem_slice(data, box_in_series)

    @skip_nested
    def test_setitem_loc_iloc_slice(self, data):
        super().test_setitem_loc_iloc_slice(data)

@skip_nested
class TestParsing(BaseNumPyTests, base.BaseParsingTests):
    pass
