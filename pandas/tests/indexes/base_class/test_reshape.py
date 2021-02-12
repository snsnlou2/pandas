
'\nTests for ndarray-like method on the base Index class\n'
import pytest
from pandas import Index
import pandas._testing as tm

class TestReshape():

    def test_repeat(self):
        repeats = 2
        index = Index([1, 2, 3])
        expected = Index([1, 1, 2, 2, 3, 3])
        result = index.repeat(repeats)
        tm.assert_index_equal(result, expected)

    def test_insert(self):
        result = Index(['b', 'c', 'd'])
        tm.assert_index_equal(Index(['a', 'b', 'c', 'd']), result.insert(0, 'a'))
        tm.assert_index_equal(Index(['b', 'c', 'e', 'd']), result.insert((- 1), 'e'))
        tm.assert_index_equal(result.insert(1, 'z'), result.insert((- 2), 'z'))
        null_index = Index([])
        tm.assert_index_equal(Index(['a']), null_index.insert(0, 'a'))

    @pytest.mark.parametrize('pos,expected', [(0, Index(['b', 'c', 'd'], name='index')), ((- 1), Index(['a', 'b', 'c'], name='index'))])
    def test_delete(self, pos, expected):
        index = Index(['a', 'b', 'c', 'd'], name='index')
        result = index.delete(pos)
        tm.assert_index_equal(result, expected)
        assert (result.name == expected.name)

    def test_append_multiple(self):
        index = Index(['a', 'b', 'c', 'd', 'e', 'f'])
        foos = [index[:2], index[2:4], index[4:]]
        result = foos[0].append(foos[1:])
        tm.assert_index_equal(result, index)
        result = index.append([])
        tm.assert_index_equal(result, index)
