
import pytest
from pandas import DataFrame
import pandas._testing as tm

class TestCopy():

    @pytest.mark.parametrize('attr', ['index', 'columns'])
    def test_copy_index_name_checking(self, float_frame, attr):
        ind = getattr(float_frame, attr)
        ind.name = None
        cp = float_frame.copy()
        getattr(cp, attr).name = 'foo'
        assert (getattr(float_frame, attr).name is None)

    def test_copy_cache(self):
        df = DataFrame({'a': [1]})
        df['x'] = [0]
        df['a']
        df.copy()
        df['a'].values[0] = (- 1)
        tm.assert_frame_equal(df, DataFrame({'a': [(- 1)], 'x': [0]}))
        df['y'] = [0]
        assert (df['a'].values[0] == (- 1))
        tm.assert_frame_equal(df, DataFrame({'a': [(- 1)], 'x': [0], 'y': [0]}))

    def test_copy(self, float_frame, float_string_frame):
        cop = float_frame.copy()
        cop['E'] = cop['A']
        assert ('E' not in float_frame)
        copy = float_string_frame.copy()
        assert (copy._mgr is not float_string_frame._mgr)
