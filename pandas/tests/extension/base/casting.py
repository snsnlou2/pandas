
import numpy as np
import pytest
import pandas as pd
from pandas.core.internals import ObjectBlock
from .base import BaseExtensionTests

class BaseCastingTests(BaseExtensionTests):
    'Casting to and from ExtensionDtypes'

    def test_astype_object_series(self, all_data):
        ser = pd.Series(all_data, name='A')
        result = ser.astype(object)
        assert isinstance(result._mgr.blocks[0], ObjectBlock)

    def test_astype_object_frame(self, all_data):
        df = pd.DataFrame({'A': all_data})
        result = df.astype(object)
        blk = result._data.blocks[0]
        assert isinstance(blk, ObjectBlock), type(blk)

    def test_tolist(self, data):
        result = pd.Series(data).tolist()
        expected = list(data)
        assert (result == expected)

    def test_astype_str(self, data):
        result = pd.Series(data[:5]).astype(str)
        expected = pd.Series([str(x) for x in data[:5]], dtype=str)
        self.assert_series_equal(result, expected)

    def test_astype_string(self, data):
        result = pd.Series(data[:5]).astype('string')
        expected = pd.Series([str(x) for x in data[:5]], dtype='string')
        self.assert_series_equal(result, expected)

    def test_to_numpy(self, data):
        expected = np.asarray(data)
        result = data.to_numpy()
        self.assert_equal(result, expected)
        result = pd.Series(data).to_numpy()
        self.assert_equal(result, expected)

    def test_astype_empty_dataframe(self, dtype):
        df = pd.DataFrame()
        result = df.astype(dtype)
        self.assert_frame_equal(result, df)

    @pytest.mark.parametrize('copy', [True, False])
    def test_astype_own_type(self, data, copy):
        result = data.astype(data.dtype, copy=copy)
        assert ((result is data) is (not copy))
        self.assert_extension_array_equal(result, data)
