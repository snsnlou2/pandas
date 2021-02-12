
import numpy as np
from pandas import DataFrame, MultiIndex
import pandas._testing as tm
from pandas.core.arrays import PandasArray

class TestToDictOfBlocks():

    def test_copy_blocks(self, float_frame):
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]
        blocks = df._to_dict_of_blocks(copy=True)
        for (dtype, _df) in blocks.items():
            if (column in _df):
                _df.loc[:, column] = (_df[column] + 1)
        assert (not _df[column].equals(df[column]))

    def test_no_copy_blocks(self, float_frame):
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]
        blocks = df._to_dict_of_blocks(copy=False)
        for (dtype, _df) in blocks.items():
            if (column in _df):
                _df.loc[:, column] = (_df[column] + 1)
        assert _df[column].equals(df[column])

def test_to_dict_of_blocks_item_cache():
    df = DataFrame({'a': [1, 2, 3, 4], 'b': ['a', 'b', 'c', 'd']})
    df['c'] = PandasArray(np.array([1, 2, None, 3], dtype=object))
    mgr = df._mgr
    assert (len(mgr.blocks) == 3)
    ser = df['b']
    df._to_dict_of_blocks()
    ser.values[0] = 'foo'
    assert (df.loc[(0, 'b')] == 'foo')
    assert (df['b'] is ser)

def test_set_change_dtype_slice():
    cols = MultiIndex.from_tuples([('1st', 'a'), ('2nd', 'b'), ('3rd', 'c')])
    df = DataFrame([[1.0, 2, 3], [4.0, 5, 6]], columns=cols)
    df['2nd'] = (df['2nd'] * 2.0)
    blocks = df._to_dict_of_blocks()
    assert (sorted(blocks.keys()) == ['float64', 'int64'])
    tm.assert_frame_equal(blocks['float64'], DataFrame([[1.0, 4.0], [4.0, 10.0]], columns=cols[:2]))
    tm.assert_frame_equal(blocks['int64'], DataFrame([[3], [6]], columns=cols[2:]))
