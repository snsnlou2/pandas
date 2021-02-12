
import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series
import pandas._testing as tm
import pandas.core.common as com

def test_detect_chained_assignment():
    a = [12, 23]
    b = [123, None]
    c = [1234, 2345]
    d = [12345, 23456]
    tuples = [('eyes', 'left'), ('eyes', 'right'), ('ears', 'left'), ('ears', 'right')]
    events = {('eyes', 'left'): a, ('eyes', 'right'): b, ('ears', 'left'): c, ('ears', 'right'): d}
    multiind = MultiIndex.from_tuples(tuples, names=['part', 'side'])
    zed = DataFrame(events, index=['a', 'b'], columns=multiind)
    msg = 'A value is trying to be set on a copy of a slice from a DataFrame'
    with pytest.raises(com.SettingWithCopyError, match=msg):
        zed['eyes']['right'].fillna(value=555, inplace=True)

def test_cache_updating():
    a = np.random.rand(10, 3)
    df = DataFrame(a, columns=['x', 'y', 'z'])
    tuples = [(i, j) for i in range(5) for j in range(2)]
    index = MultiIndex.from_tuples(tuples)
    df.index = index
    df.loc[0]['z'].iloc[0] = 1.0
    result = df.loc[((0, 0), 'z')]
    assert (result == 1)
    df.loc[((0, 0), 'z')] = 2
    result = df.loc[((0, 0), 'z')]
    assert (result == 2)

@pytest.mark.arm_slow
def test_indexer_caching():
    n = 1000001
    arrays = (range(n), range(n))
    index = MultiIndex.from_tuples(zip(*arrays))
    s = Series(np.zeros(n), index=index)
    str(s)
    expected = Series(np.ones(n), index=index)
    s = Series(np.zeros(n), index=index)
    s[(s == 0)] = 1
    tm.assert_series_equal(s, expected)
