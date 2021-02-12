
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import pandas._testing as tm

def test_pipe():
    random_state = np.random.RandomState(1234567890)
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': random_state.randn(8), 'C': random_state.randn(8)})

    def f(dfgb):
        return (dfgb.B.max() - dfgb.C.min().min())

    def square(srs):
        return (srs ** 2)
    result = df.groupby('A').pipe(f).pipe(square)
    index = Index(['bar', 'foo'], dtype='object', name='A')
    expected = pd.Series([8.99110003361, 8.17516964785], name='B', index=index)
    tm.assert_series_equal(expected, result)

def test_pipe_args():
    df = DataFrame({'group': ['A', 'A', 'B', 'B', 'C'], 'x': [1.0, 2.0, 3.0, 2.0, 5.0], 'y': [10.0, 100.0, 1000.0, (- 100.0), (- 1000.0)]})

    def f(dfgb, arg1):
        return dfgb.filter((lambda grp: (grp.y.mean() > arg1)), dropna=False).groupby(dfgb.grouper)

    def g(dfgb, arg2):
        return ((dfgb.sum() / dfgb.sum().sum()) + arg2)

    def h(df, arg3):
        return ((df.x + df.y) - arg3)
    result = df.groupby('group').pipe(f, 0).pipe(g, 10).pipe(h, 100)
    index = Index(['A', 'B', 'C'], name='group')
    expected = pd.Series([(- 79.5160891089), (- 78.4839108911), (- 80)], index=index)
    tm.assert_series_equal(expected, result)
    ser = pd.Series([1, 1, 2, 2, 3, 3])
    result = ser.groupby(ser).pipe((lambda grp: (grp.sum() * grp.count())))
    expected = pd.Series([4, 8, 12], index=pd.Int64Index([1, 2, 3]))
    tm.assert_series_equal(result, expected)
