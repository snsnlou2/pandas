
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

class TestDataFrameFilter():

    def test_filter(self, float_frame, float_string_frame):
        filtered = float_frame.filter(['A', 'B', 'E'])
        assert (len(filtered.columns) == 2)
        assert ('E' not in filtered)
        filtered = float_frame.filter(['A', 'B', 'E'], axis='columns')
        assert (len(filtered.columns) == 2)
        assert ('E' not in filtered)
        idx = float_frame.index[0:4]
        filtered = float_frame.filter(idx, axis='index')
        expected = float_frame.reindex(index=idx)
        tm.assert_frame_equal(filtered, expected)
        fcopy = float_frame.copy()
        fcopy['AA'] = 1
        filtered = fcopy.filter(like='A')
        assert (len(filtered.columns) == 2)
        assert ('AA' in filtered)
        df = DataFrame(0.0, index=[0, 1, 2], columns=[0, 1, '_A', '_B'])
        filtered = df.filter(like='_')
        assert (len(filtered.columns) == 2)
        df = DataFrame(0.0, index=[0, 1, 2], columns=['A1', 1, 'B', 2, 'C'])
        expected = DataFrame(0.0, index=[0, 1, 2], columns=pd.Index([1, 2], dtype=object))
        filtered = df.filter(regex='^[0-9]+$')
        tm.assert_frame_equal(filtered, expected)
        expected = DataFrame(0.0, index=[0, 1, 2], columns=[0, '0', 1, '1'])
        filtered = expected.filter(regex='^[0-9]+$')
        tm.assert_frame_equal(filtered, expected)
        with pytest.raises(TypeError, match='Must pass'):
            float_frame.filter()
        with pytest.raises(TypeError, match='Must pass'):
            float_frame.filter(items=None)
        with pytest.raises(TypeError, match='Must pass'):
            float_frame.filter(axis=1)
        with pytest.raises(TypeError, match='mutually exclusive'):
            float_frame.filter(items=['one', 'three'], regex='e$', like='bbi')
        with pytest.raises(TypeError, match='mutually exclusive'):
            float_frame.filter(items=['one', 'three'], regex='e$', axis=1)
        with pytest.raises(TypeError, match='mutually exclusive'):
            float_frame.filter(items=['one', 'three'], regex='e$')
        with pytest.raises(TypeError, match='mutually exclusive'):
            float_frame.filter(items=['one', 'three'], like='bbi', axis=0)
        with pytest.raises(TypeError, match='mutually exclusive'):
            float_frame.filter(items=['one', 'three'], like='bbi')
        filtered = float_string_frame.filter(like='foo')
        assert ('foo' in filtered)
        df = float_frame.rename(columns={'B': '∂'})
        filtered = df.filter(like='C')
        assert ('C' in filtered)

    def test_filter_regex_search(self, float_frame):
        fcopy = float_frame.copy()
        fcopy['AA'] = 1
        filtered = fcopy.filter(regex='[A]+')
        assert (len(filtered.columns) == 2)
        assert ('AA' in filtered)
        df = DataFrame({'aBBa': [1, 2], 'BBaBB': [1, 2], 'aCCa': [1, 2], 'aCCaBB': [1, 2]})
        result = df.filter(regex='BB')
        exp = df[[x for x in df.columns if ('BB' in x)]]
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize('name,expected', [('a', DataFrame({'a': [1, 2]})), ('a', DataFrame({'a': [1, 2]})), ('あ', DataFrame({'あ': [3, 4]}))])
    def test_filter_unicode(self, name, expected):
        df = DataFrame({'a': [1, 2], 'あ': [3, 4]})
        tm.assert_frame_equal(df.filter(like=name), expected)
        tm.assert_frame_equal(df.filter(regex=name), expected)

    @pytest.mark.parametrize('name', ['a', 'a'])
    def test_filter_bytestring(self, name):
        df = DataFrame({b'a': [1, 2], b'b': [3, 4]})
        expected = DataFrame({b'a': [1, 2]})
        tm.assert_frame_equal(df.filter(like=name), expected)
        tm.assert_frame_equal(df.filter(regex=name), expected)

    def test_filter_corner(self):
        empty = DataFrame()
        result = empty.filter([])
        tm.assert_frame_equal(result, empty)
        result = empty.filter(like='foo')
        tm.assert_frame_equal(result, empty)

    def test_filter_regex_non_string(self):
        df = DataFrame(np.random.random((3, 2)), columns=['STRING', 123])
        result = df.filter(regex='STRING')
        expected = df[['STRING']]
        tm.assert_frame_equal(result, expected)
