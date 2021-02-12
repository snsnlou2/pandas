
import numpy as np
from pandas import DataFrame, MultiIndex, Series
import pandas._testing as tm

class TestMultiIndexInsertion():

    def test_setitem_mixed_depth(self):
        arrays = [['a', 'top', 'top', 'routine1', 'routine1', 'routine2'], ['', 'OD', 'OD', 'result1', 'result2', 'result1'], ['', 'wx', 'wy', '', '', '']]
        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.random.randn(4, 6), columns=index)
        result = df.copy()
        expected = df.copy()
        result['b'] = [1, 2, 3, 4]
        expected[('b', '', '')] = [1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    def test_dataframe_insert_column_all_na(self):
        mix = MultiIndex.from_tuples([('1a', '2a'), ('1a', '2b'), ('1a', '2c')])
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mix)
        s = Series({(1, 1): 1, (1, 2): 2})
        df['new'] = s
        assert df['new'].isna().all()
