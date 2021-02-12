
'\nTests that work on both the Python and C engines but do not have a\nspecific classification into the other test modules.\n'
import codecs
import csv
from datetime import datetime
from inspect import signature
from io import BytesIO, StringIO
import os
import platform
from urllib.error import URLError
import numpy as np
import pytest
from pandas._libs.tslib import Timestamp
from pandas.compat import is_platform_linux
from pandas.errors import DtypeWarning, EmptyDataError, ParserError
import pandas.util._test_decorators as td
from pandas import DataFrame, Index, MultiIndex, Series, compat, concat, option_context
import pandas._testing as tm
from pandas.io.parsers import CParserWrapper, TextFileReader, TextParser

def test_override_set_noconvert_columns():

    class MyTextFileReader(TextFileReader):

        def __init__(self):
            self._currow = 0
            self.squeeze = False

    class MyCParserWrapper(CParserWrapper):

        def _set_noconvert_columns(self):
            if (self.usecols_dtype == 'integer'):
                self.usecols = list(self.usecols)
                self.usecols.reverse()
            return CParserWrapper._set_noconvert_columns(self)
    data = 'a,b,c,d,e\n0,1,20140101,0900,4\n0,1,20140102,1000,4'
    parse_dates = [[1, 2]]
    cols = {'a': [0, 0], 'c_d': [Timestamp('2014-01-01 09:00:00'), Timestamp('2014-01-02 10:00:00')]}
    expected = DataFrame(cols, columns=['c_d', 'a'])
    parser = MyTextFileReader()
    parser.options = {'usecols': [0, 2, 3], 'parse_dates': parse_dates, 'delimiter': ','}
    parser._engine = MyCParserWrapper(StringIO(data), **parser.options)
    result = parser.read()
    tm.assert_frame_equal(result, expected)

def test_empty_decimal_marker(all_parsers):
    data = 'A|B|C\n1|2,334|5\n10|13|10.\n'
    msg = 'Only length-1 decimal markers supported'
    parser = all_parsers
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), decimal='')

def test_bad_stream_exception(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, 'sauron.SHIFT_JIS.csv')
    codec = codecs.lookup('utf-8')
    utf8 = codecs.lookup('utf-8')
    parser = all_parsers
    msg = "'utf-8' codec can't decode byte"
    with open(path, 'rb') as handle, codecs.StreamRecoder(handle, utf8.encode, utf8.decode, codec.streamreader, codec.streamwriter) as stream:
        with pytest.raises(UnicodeDecodeError, match=msg):
            parser.read_csv(stream)

def test_read_csv_local(all_parsers, csv1):
    prefix = ('file:///' if compat.is_platform_windows() else 'file://')
    parser = all_parsers
    fname = (prefix + str(os.path.abspath(csv1)))
    result = parser.read_csv(fname, index_col=0, parse_dates=True)
    expected = DataFrame([[0.980269, 3.685731, (- 0.364216805298), (- 1.159738)], [1.047916, (- 0.041232), (- 0.16181208307), 0.212549], [0.498581, 0.731168, (- 0.537677223318), 1.34627], [1.120202, 1.567621, 0.00364077397681, 0.675253], [(- 0.487094), 0.571455, (- 1.6116394093), 0.103469], [0.836649, 0.246462, 0.588542635376, 1.062782], [(- 0.157161), 1.340307, 1.1957779562, (- 1.097007)]], columns=['A', 'B', 'C', 'D'], index=Index([datetime(2000, 1, 3), datetime(2000, 1, 4), datetime(2000, 1, 5), datetime(2000, 1, 6), datetime(2000, 1, 7), datetime(2000, 1, 10), datetime(2000, 1, 11)], name='index'))
    tm.assert_frame_equal(result, expected)

def test_1000_sep(all_parsers):
    parser = all_parsers
    data = 'A|B|C\n1|2,334|5\n10|13|10.\n'
    expected = DataFrame({'A': [1, 10], 'B': [2334, 13], 'C': [5, 10.0]})
    result = parser.read_csv(StringIO(data), sep='|', thousands=',')
    tm.assert_frame_equal(result, expected)

def test_squeeze(all_parsers):
    data = 'a,1\nb,2\nc,3\n'
    parser = all_parsers
    index = Index(['a', 'b', 'c'], name=0)
    expected = Series([1, 2, 3], name=1, index=index)
    result = parser.read_csv(StringIO(data), index_col=0, header=None, squeeze=True)
    tm.assert_series_equal(result, expected)
    assert (not result._is_view)

def test_malformed(all_parsers):
    parser = all_parsers
    data = 'ignore\nA,B,C\n1,2,3 # comment\n1,2,3,4,5\n2,3,4\n'
    msg = 'Expected 3 fields in line 4, saw 5'
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), header=1, comment='#')

@pytest.mark.parametrize('nrows', [5, 3, None])
def test_malformed_chunks(all_parsers, nrows):
    data = 'ignore\nA,B,C\nskip\n1,2,3\n3,5,10 # comment\n1,2,3,4,5\n2,3,4\n'
    parser = all_parsers
    msg = 'Expected 3 fields in line 6, saw 5'
    with parser.read_csv(StringIO(data), header=1, comment='#', iterator=True, chunksize=1, skiprows=[2]) as reader:
        with pytest.raises(ParserError, match=msg):
            reader.read(nrows)

def test_unnamed_columns(all_parsers):
    data = 'A,B,C,,\n1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15\n'
    parser = all_parsers
    expected = DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=np.int64, columns=['A', 'B', 'C', 'Unnamed: 3', 'Unnamed: 4'])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)

def test_csv_mixed_type(all_parsers):
    data = 'A,B,C\na,1,2\nb,3,4\nc,4,5\n'
    parser = all_parsers
    expected = DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 3, 4], 'C': [2, 4, 5]})
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)

def test_read_csv_low_memory_no_rows_with_index(all_parsers):
    parser = all_parsers
    if (not parser.low_memory):
        pytest.skip('This is a low-memory specific test')
    data = 'A,B,C\n1,1,1,2\n2,2,3,4\n3,3,4,5\n'
    result = parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
    expected = DataFrame(columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)

def test_read_csv_dataframe(all_parsers, csv1):
    parser = all_parsers
    result = parser.read_csv(csv1, index_col=0, parse_dates=True)
    expected = DataFrame([[0.980269, 3.685731, (- 0.364216805298), (- 1.159738)], [1.047916, (- 0.041232), (- 0.16181208307), 0.212549], [0.498581, 0.731168, (- 0.537677223318), 1.34627], [1.120202, 1.567621, 0.00364077397681, 0.675253], [(- 0.487094), 0.571455, (- 1.6116394093), 0.103469], [0.836649, 0.246462, 0.588542635376, 1.062782], [(- 0.157161), 1.340307, 1.1957779562, (- 1.097007)]], columns=['A', 'B', 'C', 'D'], index=Index([datetime(2000, 1, 3), datetime(2000, 1, 4), datetime(2000, 1, 5), datetime(2000, 1, 6), datetime(2000, 1, 7), datetime(2000, 1, 10), datetime(2000, 1, 11)], name='index'))
    tm.assert_frame_equal(result, expected)

def test_read_csv_no_index_name(all_parsers, csv_dir_path):
    parser = all_parsers
    csv2 = os.path.join(csv_dir_path, 'test2.csv')
    result = parser.read_csv(csv2, index_col=0, parse_dates=True)
    expected = DataFrame([[0.980269, 3.685731, (- 0.364216805298), (- 1.159738), 'foo'], [1.047916, (- 0.041232), (- 0.16181208307), 0.212549, 'bar'], [0.498581, 0.731168, (- 0.537677223318), 1.34627, 'baz'], [1.120202, 1.567621, 0.00364077397681, 0.675253, 'qux'], [(- 0.487094), 0.571455, (- 1.6116394093), 0.103469, 'foo2']], columns=['A', 'B', 'C', 'D', 'E'], index=Index([datetime(2000, 1, 3), datetime(2000, 1, 4), datetime(2000, 1, 5), datetime(2000, 1, 6), datetime(2000, 1, 7)]))
    tm.assert_frame_equal(result, expected)

def test_read_csv_wrong_num_columns(all_parsers):
    data = 'A,B,C,D,E,F\n1,2,3,4,5,6\n6,7,8,9,10,11,12\n11,12,13,14,15,16\n'
    parser = all_parsers
    msg = 'Expected 6 fields in line 3, saw 7'
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data))

def test_read_duplicate_index_explicit(all_parsers):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo,12,13,14,15\nbar,12,13,14,15\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=0)
    expected = DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], columns=['A', 'B', 'C', 'D'], index=Index(['foo', 'bar', 'baz', 'qux', 'foo', 'bar'], name='index'))
    tm.assert_frame_equal(result, expected)

def test_read_duplicate_index_implicit(all_parsers):
    data = 'A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo,12,13,14,15\nbar,12,13,14,15\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], columns=['A', 'B', 'C', 'D'], index=Index(['foo', 'bar', 'baz', 'qux', 'foo', 'bar']))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data,kwargs,expected', [('A,B\nTrue,1\nFalse,2\nTrue,3', {}, DataFrame([[True, 1], [False, 2], [True, 3]], columns=['A', 'B'])), ('A,B\nYES,1\nno,2\nyes,3\nNo,3\nYes,3', {'true_values': ['yes', 'Yes', 'YES'], 'false_values': ['no', 'NO', 'No']}, DataFrame([[True, 1], [False, 2], [True, 3], [False, 3], [True, 3]], columns=['A', 'B'])), ('A,B\nTRUE,1\nFALSE,2\nTRUE,3', {}, DataFrame([[True, 1], [False, 2], [True, 3]], columns=['A', 'B'])), ('A,B\nfoo,bar\nbar,foo', {'true_values': ['foo'], 'false_values': ['bar']}, DataFrame([[True, False], [False, True]], columns=['A', 'B']))])
def test_parse_bool(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)

def test_int_conversion(all_parsers):
    data = 'A,B\n1.0,1\n2.0,2\n3.0,3\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=['A', 'B'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('nrows', [3, 3.0])
def test_read_nrows(all_parsers, nrows):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    expected = DataFrame([['foo', 2, 3, 4, 5], ['bar', 7, 8, 9, 10], ['baz', 12, 13, 14, 15]], columns=['index', 'A', 'B', 'C', 'D'])
    parser = all_parsers
    result = parser.read_csv(StringIO(data), nrows=nrows)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('nrows', [1.2, 'foo', (- 1)])
def test_read_nrows_bad(all_parsers, nrows):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    msg = "'nrows' must be an integer >=0"
    parser = all_parsers
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), nrows=nrows)

@pytest.mark.parametrize('index_col', [0, 'index'])
def test_read_chunksize_with_index(all_parsers, index_col):
    parser = all_parsers
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    expected = DataFrame([['foo', 2, 3, 4, 5], ['bar', 7, 8, 9, 10], ['baz', 12, 13, 14, 15], ['qux', 12, 13, 14, 15], ['foo2', 12, 13, 14, 15], ['bar2', 12, 13, 14, 15]], columns=['index', 'A', 'B', 'C', 'D'])
    expected = expected.set_index('index')
    with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
        chunks = list(reader)
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])

@pytest.mark.parametrize('chunksize', [1.3, 'foo', 0])
def test_read_chunksize_bad(all_parsers, chunksize):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    msg = "'chunksize' must be an integer >=1"
    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), chunksize=chunksize) as _:
            pass

@pytest.mark.parametrize('chunksize', [2, 8])
def test_read_chunksize_and_nrows(all_parsers, chunksize):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    kwargs = {'index_col': 0, 'nrows': 5}
    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=chunksize, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), expected)

def test_read_chunksize_and_nrows_changing_size(all_parsers):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    kwargs = {'index_col': 0, 'nrows': 5}
    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=8, **kwargs) as reader:
        tm.assert_frame_equal(reader.get_chunk(size=2), expected.iloc[:2])
        tm.assert_frame_equal(reader.get_chunk(size=4), expected.iloc[2:5])
        with pytest.raises(StopIteration, match=''):
            reader.get_chunk(size=3)

def test_get_chunk_passed_chunksize(all_parsers):
    parser = all_parsers
    data = 'A,B,C\n1,2,3\n4,5,6\n7,8,9\n1,2,3'
    with parser.read_csv(StringIO(data), chunksize=2) as reader:
        result = reader.get_chunk()
    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('kwargs', [{}, {'index_col': 0}])
def test_read_chunksize_compat(all_parsers, kwargs):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), result)

def test_read_chunksize_jagged_names(all_parsers):
    parser = all_parsers
    data = '\n'.join(((['0'] * 7) + [','.join((['0'] * 10))]))
    expected = DataFrame((([([0] + ([np.nan] * 9))] * 7) + [([0] * 10)]))
    with parser.read_csv(StringIO(data), names=range(10), chunksize=4) as reader:
        result = concat(reader)
    tm.assert_frame_equal(result, expected)

def test_read_data_list(all_parsers):
    parser = all_parsers
    kwargs = {'index_col': 0}
    data = 'A,B,C\nfoo,1,2,3\nbar,4,5,6'
    data_list = [['A', 'B', 'C'], ['foo', '1', '2', '3'], ['bar', '4', '5', '6']]
    expected = parser.read_csv(StringIO(data), **kwargs)
    with TextParser(data_list, chunksize=2, **kwargs) as parser:
        result = parser.read()
    tm.assert_frame_equal(result, expected)

def test_iterator(all_parsers):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    kwargs = {'index_col': 0}
    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), iterator=True, **kwargs) as reader:
        first_chunk = reader.read(3)
        tm.assert_frame_equal(first_chunk, expected[:3])
        last_chunk = reader.read(5)
    tm.assert_frame_equal(last_chunk, expected[3:])

def test_iterator2(all_parsers):
    parser = all_parsers
    data = 'A,B,C\nfoo,1,2,3\nbar,4,5,6\nbaz,7,8,9\n'
    with parser.read_csv(StringIO(data), iterator=True) as reader:
        result = list(reader)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['foo', 'bar', 'baz'], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result[0], expected)

def test_reader_list(all_parsers):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    kwargs = {'index_col': 0}
    lines = list(csv.reader(StringIO(data)))
    with TextParser(lines, chunksize=2, **kwargs) as reader:
        chunks = list(reader)
    expected = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])

def test_reader_list_skiprows(all_parsers):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    kwargs = {'index_col': 0}
    lines = list(csv.reader(StringIO(data)))
    with TextParser(lines, chunksize=2, skiprows=[1], **kwargs) as reader:
        chunks = list(reader)
    expected = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(chunks[0], expected[1:3])

def test_iterator_stop_on_chunksize(all_parsers):
    parser = all_parsers
    data = 'A,B,C\nfoo,1,2,3\nbar,4,5,6\nbaz,7,8,9\n'
    with parser.read_csv(StringIO(data), chunksize=1) as reader:
        result = list(reader)
    assert (len(result) == 3)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['foo', 'bar', 'baz'], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(concat(result), expected)

@pytest.mark.parametrize('kwargs', [{'iterator': True, 'chunksize': 1}, {'iterator': True}, {'chunksize': 1}])
def test_iterator_skipfooter_errors(all_parsers, kwargs):
    msg = "'skipfooter' not supported for iteration"
    parser = all_parsers
    data = 'a\n1\n2'
    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), skipfooter=1, **kwargs) as _:
            pass

def test_nrows_skipfooter_errors(all_parsers):
    msg = "'skipfooter' not supported with 'nrows'"
    data = 'a\n1\n2\n3\n4\n5\n6'
    parser = all_parsers
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=1, nrows=5)

@pytest.mark.parametrize('data,kwargs,expected', [('foo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n', {'index_col': 0, 'names': ['index', 'A', 'B', 'C', 'D']}, DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], index=Index(['foo', 'bar', 'baz', 'qux', 'foo2', 'bar2'], name='index'), columns=['A', 'B', 'C', 'D'])), ('foo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n', {'index_col': [0, 1], 'names': ['index1', 'index2', 'A', 'B', 'C', 'D']}, DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], index=MultiIndex.from_tuples([('foo', 'one'), ('foo', 'two'), ('foo', 'three'), ('bar', 'one'), ('bar', 'two')], names=['index1', 'index2']), columns=['A', 'B', 'C', 'D']))])
def test_pass_names_with_index(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('index_col', [[0, 1], [1, 0]])
def test_multi_index_no_level_names(all_parsers, index_col):
    data = 'index1,index2,A,B,C,D\nfoo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n'
    headless_data = '\n'.join(data.split('\n')[1:])
    names = ['A', 'B', 'C', 'D']
    parser = all_parsers
    result = parser.read_csv(StringIO(headless_data), index_col=index_col, header=None, names=names)
    expected = parser.read_csv(StringIO(data), index_col=index_col)
    expected.index.names = ([None] * 2)
    tm.assert_frame_equal(result, expected)

def test_multi_index_no_level_names_implicit(all_parsers):
    parser = all_parsers
    data = 'A,B,C,D\nfoo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], columns=['A', 'B', 'C', 'D'], index=MultiIndex.from_tuples([('foo', 'one'), ('foo', 'two'), ('foo', 'three'), ('bar', 'one'), ('bar', 'two')]))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data,expected,header', [('a,b', DataFrame(columns=['a', 'b']), [0]), ('a,b\nc,d', DataFrame(columns=MultiIndex.from_tuples([('a', 'c'), ('b', 'd')])), [0, 1])])
@pytest.mark.parametrize('round_trip', [True, False])
def test_multi_index_blank_df(all_parsers, data, expected, header, round_trip):
    parser = all_parsers
    data = (expected.to_csv(index=False) if round_trip else data)
    result = parser.read_csv(StringIO(data), header=header)
    tm.assert_frame_equal(result, expected)

def test_no_unnamed_index(all_parsers):
    parser = all_parsers
    data = ' id c0 c1 c2\n0 1 0 a b\n1 2 0 c d\n2 2 2 e f\n'
    result = parser.read_csv(StringIO(data), sep=' ')
    expected = DataFrame([[0, 1, 0, 'a', 'b'], [1, 2, 0, 'c', 'd'], [2, 2, 2, 'e', 'f']], columns=['Unnamed: 0', 'id', 'c0', 'c1', 'c2'])
    tm.assert_frame_equal(result, expected)

def test_read_csv_parse_simple_list(all_parsers):
    parser = all_parsers
    data = 'foo\nbar baz\nqux foo\nfoo\nbar'
    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame(['foo', 'bar baz', 'qux foo', 'foo', 'bar'])
    tm.assert_frame_equal(result, expected)

@tm.network
def test_url(all_parsers, csv_dir_path):
    parser = all_parsers
    kwargs = {'sep': '\t'}
    url = 'https://raw.github.com/pandas-dev/pandas/master/pandas/tests/io/parser/data/salaries.csv'
    url_result = parser.read_csv(url, **kwargs)
    local_path = os.path.join(csv_dir_path, 'salaries.csv')
    local_result = parser.read_csv(local_path, **kwargs)
    tm.assert_frame_equal(url_result, local_result)

@pytest.mark.slow
def test_local_file(all_parsers, csv_dir_path):
    parser = all_parsers
    kwargs = {'sep': '\t'}
    local_path = os.path.join(csv_dir_path, 'salaries.csv')
    local_result = parser.read_csv(local_path, **kwargs)
    url = ('file://localhost/' + local_path)
    try:
        url_result = parser.read_csv(url, **kwargs)
        tm.assert_frame_equal(url_result, local_result)
    except URLError:
        pytest.skip(('Failing on: ' + ' '.join(platform.uname())))

def test_path_path_lib(all_parsers):
    parser = all_parsers
    df = tm.makeDataFrame()
    result = tm.round_trip_pathlib(df.to_csv, (lambda p: parser.read_csv(p, index_col=0)))
    tm.assert_frame_equal(df, result)

def test_path_local_path(all_parsers):
    parser = all_parsers
    df = tm.makeDataFrame()
    result = tm.round_trip_localpath(df.to_csv, (lambda p: parser.read_csv(p, index_col=0)))
    tm.assert_frame_equal(df, result)

def test_nonexistent_path(all_parsers):
    parser = all_parsers
    path = f'{tm.rands(10)}.csv'
    msg = '\\[Errno 2\\]'
    with pytest.raises(FileNotFoundError, match=msg) as e:
        parser.read_csv(path)
    assert (path == e.value.filename)

@td.skip_if_windows
def test_no_permission(all_parsers):
    parser = all_parsers
    msg = '\\[Errno 13\\]'
    with tm.ensure_clean() as path:
        os.chmod(path, 0)
        try:
            with open(path):
                pass
            pytest.skip('Running as sudo.')
        except PermissionError:
            pass
        with pytest.raises(PermissionError, match=msg) as e:
            parser.read_csv(path)
        assert (path == e.value.filename)

def test_missing_trailing_delimiters(all_parsers):
    parser = all_parsers
    data = 'A,B,C,D\n1,2,3,4\n1,3,3,\n1,4,5'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[1, 2, 3, 4], [1, 3, 3, np.nan], [1, 4, 5, np.nan]], columns=['A', 'B', 'C', 'D'])
    tm.assert_frame_equal(result, expected)

def test_skip_initial_space(all_parsers):
    data = '"09-Apr-2012", "01:10:18.300", 2456026.548822908, 12849, 1.00361,  1.12551, 330.65659, 0355626618.16711,  73.48821, 314.11625,  1917.09447,   179.71425,  80.000, 240.000, -350,  70.06056, 344.98370, 1,   1, -0.689265, -0.692787,  0.212036,    14.7674,   41.605,   -9999.0,   -9999.0,   -9999.0,   -9999.0,   -9999.0,  -9999.0, 000, 012, 128'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), names=list(range(33)), header=None, na_values=['-9999.0'], skipinitialspace=True)
    expected = DataFrame([['09-Apr-2012', '01:10:18.300', 2456026.548822908, 12849, 1.00361, 1.12551, 330.65659, 355626618.16711, 73.48821, 314.11625, 1917.09447, 179.71425, 80.0, 240.0, (- 350), 70.06056, 344.9837, 1, 1, (- 0.689265), (- 0.692787), 0.212036, 14.7674, 41.605, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 12, 128]])
    tm.assert_frame_equal(result, expected)

def test_trailing_delimiters(all_parsers):
    data = 'A,B,C\n1,2,3,\n4,5,6,\n7,8,9,'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=False)
    expected = DataFrame({'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]})
    tm.assert_frame_equal(result, expected)

def test_escapechar(all_parsers):
    data = 'SEARCH_TERM,ACTUAL_URL\n"bra tv bord","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"\n"tv pÃ¥ hjul","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"\n"SLAGBORD, \\"Bergslagen\\", IKEA:s 1700-tals series","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), escapechar='\\', quotechar='"', encoding='utf-8')
    assert (result['SEARCH_TERM'][2] == 'SLAGBORD, "Bergslagen", IKEA:s 1700-tals series')
    tm.assert_index_equal(result.columns, Index(['SEARCH_TERM', 'ACTUAL_URL']))

def test_int64_min_issues(all_parsers):
    parser = all_parsers
    data = 'A,B\n0,0\n0,'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame({'A': [0, 0], 'B': [0, np.nan]})
    tm.assert_frame_equal(result, expected)

def test_parse_integers_above_fp_precision(all_parsers):
    data = 'Numbers\n17007000002000191\n17007000002000191\n17007000002000191\n17007000002000191\n17007000002000192\n17007000002000192\n17007000002000192\n17007000002000192\n17007000002000192\n17007000002000194'
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame({'Numbers': [17007000002000191, 17007000002000191, 17007000002000191, 17007000002000191, 17007000002000192, 17007000002000192, 17007000002000192, 17007000002000192, 17007000002000192, 17007000002000194]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.xfail(reason='GH38630, sometimes gives ResourceWarning', strict=False)
def test_chunks_have_consistent_numerical_type(all_parsers):
    parser = all_parsers
    integers = [str(i) for i in range(499999)]
    data = ('a\n' + '\n'.join(((integers + ['1.0', '2.0']) + integers)))
    with tm.assert_produces_warning(None):
        result = parser.read_csv(StringIO(data))
    assert (type(result.a[0]) is np.float64)
    assert (result.a.dtype == float)

def test_warn_if_chunks_have_mismatched_type(all_parsers):
    warning_type = None
    parser = all_parsers
    integers = [str(i) for i in range(499999)]
    data = ('a\n' + '\n'.join(((integers + ['a', 'b']) + integers)))
    if ((parser.engine == 'c') and parser.low_memory):
        warning_type = DtypeWarning
    with tm.assert_produces_warning(warning_type):
        df = parser.read_csv(StringIO(data))
    assert (df.a.dtype == object)

@pytest.mark.parametrize('sep', [' ', '\\s+'])
def test_integer_overflow_bug(all_parsers, sep):
    data = '65248E10 11\n55555E55 22\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), header=None, sep=sep)
    expected = DataFrame([[652480000000000.0, 11], [5.5555e+59, 22]])
    tm.assert_frame_equal(result, expected)

def test_catch_too_many_names(all_parsers):
    data = '1,2,3\n4,,6\n7,8,9\n10,11,12\n'
    parser = all_parsers
    msg = ('Too many columns specified: expected 4 and found 3' if (parser.engine == 'c') else 'Number of passed names did not match number of header fields in the file')
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=0, names=['a', 'b', 'c', 'd'])

def test_ignore_leading_whitespace(all_parsers):
    parser = all_parsers
    data = ' a b c\n 1 2 3\n 4 5 6\n 7 8 9'
    result = parser.read_csv(StringIO(data), sep='\\s+')
    expected = DataFrame({'a': [1, 4, 7], 'b': [2, 5, 8], 'c': [3, 6, 9]})
    tm.assert_frame_equal(result, expected)

def test_chunk_begins_with_newline_whitespace(all_parsers):
    parser = all_parsers
    data = '\n hello\nworld\n'
    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([' hello', 'world'])
    tm.assert_frame_equal(result, expected)

def test_empty_with_index(all_parsers):
    data = 'x,y'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=0)
    expected = DataFrame(columns=['y'], index=Index([], name='x'))
    tm.assert_frame_equal(result, expected)

def test_empty_with_multi_index(all_parsers):
    data = 'x,y,z'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=['x', 'y'])
    expected = DataFrame(columns=['z'], index=MultiIndex.from_arrays(([[]] * 2), names=['x', 'y']))
    tm.assert_frame_equal(result, expected)

def test_empty_with_reversed_multi_index(all_parsers):
    data = 'x,y,z'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=[1, 0])
    expected = DataFrame(columns=['z'], index=MultiIndex.from_arrays(([[]] * 2), names=['y', 'x']))
    tm.assert_frame_equal(result, expected)

def test_float_parser(all_parsers):
    parser = all_parsers
    data = '45e-1,4.5,45.,inf,-inf'
    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([[float(s) for s in data.split(',')]])
    tm.assert_frame_equal(result, expected)

def test_scientific_no_exponent(all_parsers_all_precisions):
    df = DataFrame.from_dict({'w': ['2e'], 'x': ['3E'], 'y': ['42e'], 'z': ['632E']})
    data = df.to_csv(index=False)
    (parser, precision) = all_parsers_all_precisions
    df_roundtrip = parser.read_csv(StringIO(data), float_precision=precision)
    tm.assert_frame_equal(df_roundtrip, df)

@pytest.mark.parametrize('conv', [None, np.int64, np.uint64])
def test_int64_overflow(all_parsers, conv):
    data = 'ID\n00013007854817840016671868\n00013007854817840016749251\n00013007854817840016754630\n00013007854817840016781876\n00013007854817840017028824\n00013007854817840017963235\n00013007854817840018860166'
    parser = all_parsers
    if (conv is None):
        result = parser.read_csv(StringIO(data))
        expected = DataFrame(['00013007854817840016671868', '00013007854817840016749251', '00013007854817840016754630', '00013007854817840016781876', '00013007854817840017028824', '00013007854817840017963235', '00013007854817840018860166'], columns=['ID'])
        tm.assert_frame_equal(result, expected)
    else:
        msg = '(Python int too large to convert to C long)|(long too big to convert)|(int too big to convert)'
        with pytest.raises(OverflowError, match=msg):
            parser.read_csv(StringIO(data), converters={'ID': conv})

@pytest.mark.parametrize('val', [np.iinfo(np.uint64).max, np.iinfo(np.int64).max, np.iinfo(np.int64).min])
def test_int64_uint64_range(all_parsers, val):
    parser = all_parsers
    result = parser.read_csv(StringIO(str(val)), header=None)
    expected = DataFrame([val])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('val', [(np.iinfo(np.uint64).max + 1), (np.iinfo(np.int64).min - 1)])
def test_outside_int64_uint64_range(all_parsers, val):
    parser = all_parsers
    result = parser.read_csv(StringIO(str(val)), header=None)
    expected = DataFrame([str(val)])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('exp_data', [[str((- 1)), str((2 ** 63))], [str((2 ** 63)), str((- 1))]])
def test_numeric_range_too_wide(all_parsers, exp_data):
    parser = all_parsers
    data = '\n'.join(exp_data)
    expected = DataFrame(exp_data)
    result = parser.read_csv(StringIO(data), header=None)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('neg_exp', [(- 617), (- 100000), (- 99999999999999999)])
def test_very_negative_exponent(all_parsers_all_precisions, neg_exp):
    (parser, precision) = all_parsers_all_precisions
    data = f'''data
10E{neg_exp}'''
    result = parser.read_csv(StringIO(data), float_precision=precision)
    expected = DataFrame({'data': [0.0]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('exp', [999999999999999999, (- 999999999999999999)])
def test_too_many_exponent_digits(all_parsers_all_precisions, exp, request):
    (parser, precision) = all_parsers_all_precisions
    data = f'''data
10E{exp}'''
    result = parser.read_csv(StringIO(data), float_precision=precision)
    if (precision == 'round_trip'):
        if ((exp == 999999999999999999) and is_platform_linux()):
            mark = pytest.mark.xfail(reason='GH38794, on Linux gives object result')
            request.node.add_marker(mark)
        value = (np.inf if (exp > 0) else 0.0)
        expected = DataFrame({'data': [value]})
    else:
        expected = DataFrame({'data': [f'10E{exp}']})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('iterator', [True, False])
def test_empty_with_nrows_chunksize(all_parsers, iterator):
    parser = all_parsers
    expected = DataFrame(columns=['foo', 'bar'])
    nrows = 10
    data = StringIO('foo,bar\n')
    if iterator:
        with parser.read_csv(data, chunksize=nrows) as reader:
            result = next(iter(reader))
    else:
        result = parser.read_csv(data, nrows=nrows)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data,kwargs,expected,msg', [('a,b,c\n4,5,6\n ', {}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n#comment', {'comment': '#'}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n\r', {}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6#comment', {'comment': '#'}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\nskipme', {'skiprows': [2]}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n#comment', {'comment': '#', 'skip_blank_lines': False}, DataFrame([[4, 5, 6]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n ', {'skip_blank_lines': False}, DataFrame([['4', 5, 6], [' ', None, None]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n\r', {'skip_blank_lines': False}, DataFrame([[4, 5, 6], [None, None, None]], columns=['a', 'b', 'c']), None), ('a,b,c\n4,5,6\n\\', {'escapechar': '\\'}, None, '(EOF following escape character)|(unexpected end of data)'), ('a,b,c\n4,5,6\n"\\', {'escapechar': '\\'}, None, '(EOF inside string starting at row 2)|(unexpected end of data)'), ('a,b,c\n4,5,6\n"', {'escapechar': '\\'}, None, '(EOF inside string starting at row 2)|(unexpected end of data)')], ids=['whitespace-line', 'eat-line-comment', 'eat-crnl-nop', 'eat-comment', 'skip-line', 'eat-line-comment', 'in-field', 'eat-crnl', 'escaped-char', 'escape-in-quoted-field', 'in-quoted-field'])
def test_eof_states(all_parsers, data, kwargs, expected, msg):
    parser = all_parsers
    if (expected is None):
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('usecols', [None, [0, 1], ['a', 'b']])
def test_uneven_lines_with_usecols(all_parsers, usecols):
    parser = all_parsers
    data = 'a,b,c\n0,1,2\n3,4,5,6,7\n8,9,10'
    if (usecols is None):
        msg = 'Expected \\d+ fields in line \\d+, saw \\d+'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
    else:
        expected = DataFrame({'a': [0, 3, 8], 'b': [1, 4, 9]})
        result = parser.read_csv(StringIO(data), usecols=usecols)
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data,kwargs,expected', [('', {}, None), ('', {'usecols': ['X']}, None), (',,', {'names': ['Dummy', 'X', 'Dummy_2'], 'usecols': ['X']}, DataFrame(columns=['X'], index=[0], dtype=np.float64)), ('', {'names': ['Dummy', 'X', 'Dummy_2'], 'usecols': ['X']}, DataFrame(columns=['X']))])
def test_read_empty_with_usecols(all_parsers, data, kwargs, expected):
    parser = all_parsers
    if (expected is None):
        msg = 'No columns to parse from file'
        with pytest.raises(EmptyDataError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('kwargs,expected', [({'header': None, 'delim_whitespace': True, 'skiprows': [0, 1, 2, 3, 5, 6], 'skip_blank_lines': True}, DataFrame([[1.0, 2.0, 4.0], [5.1, np.nan, 10.0]])), ({'delim_whitespace': True, 'skiprows': [1, 2, 3, 5, 6], 'skip_blank_lines': True}, DataFrame({'A': [1.0, 5.1], 'B': [2.0, np.nan], 'C': [4.0, 10]}))])
def test_trailing_spaces(all_parsers, kwargs, expected):
    data = 'A B C  \nrandom line with trailing spaces    \nskip\n1,2,3\n1,2.,4.\nrandom line with trailing tabs\t\t\t\n   \n5.1,NaN,10.0\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data.replace(',', '  ')), **kwargs)
    tm.assert_frame_equal(result, expected)

def test_raise_on_sep_with_delim_whitespace(all_parsers):
    data = 'a b c\n1 2 3'
    parser = all_parsers
    with pytest.raises(ValueError, match='you can only specify one'):
        parser.read_csv(StringIO(data), sep='\\s', delim_whitespace=True)

@pytest.mark.parametrize('delim_whitespace', [True, False])
def test_single_char_leading_whitespace(all_parsers, delim_whitespace):
    parser = all_parsers
    data = 'MyColumn\na\nb\na\nb\n'
    expected = DataFrame({'MyColumn': list('abab')})
    result = parser.read_csv(StringIO(data), skipinitialspace=True, delim_whitespace=delim_whitespace)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('sep,skip_blank_lines,exp_data', [(',', True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [(- 70.0), 0.4, 1.0]]), ('\\s+', True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [(- 70.0), 0.4, 1.0]]), (',', False, [[1.0, 2.0, 4.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [5.0, np.nan, 10.0], [np.nan, np.nan, np.nan], [(- 70.0), 0.4, 1.0]])])
def test_empty_lines(all_parsers, sep, skip_blank_lines, exp_data):
    parser = all_parsers
    data = 'A,B,C\n1,2.,4.\n\n\n5.,NaN,10.0\n\n-70,.4,1\n'
    if (sep == '\\s+'):
        data = data.replace(',', '  ')
    result = parser.read_csv(StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines)
    expected = DataFrame(exp_data, columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)

def test_whitespace_lines(all_parsers):
    parser = all_parsers
    data = '\n\n\t  \t\t\n\t\nA,B,C\n\t    1,2.,4.\n5.,NaN,10.0\n'
    expected = DataFrame([[1, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data,expected', [('   A   B   C   D\na   1   2   3   4\nb   1   2   3   4\nc   1   2   3   4\n', DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], columns=['A', 'B', 'C', 'D'], index=['a', 'b', 'c'])), ('    a b c\n1 2 3 \n4 5  6\n 7 8 9', DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c']))])
def test_whitespace_regex_separator(all_parsers, data, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), sep='\\s+')
    tm.assert_frame_equal(result, expected)

def test_verbose_read(all_parsers, capsys):
    parser = all_parsers
    data = 'a,b,c,d\none,1,2,3\none,1,2,3\n,1,2,3\none,1,2,3\n,1,2,3\n,1,2,3\none,1,2,3\ntwo,1,2,3'
    parser.read_csv(StringIO(data), verbose=True)
    captured = capsys.readouterr()
    if (parser.engine == 'c'):
        assert ('Tokenization took:' in captured.out)
        assert ('Parser memory cleanup took:' in captured.out)
    else:
        assert (captured.out == 'Filled 3 NA values in column a\n')

def test_verbose_read2(all_parsers, capsys):
    parser = all_parsers
    data = 'a,b,c,d\none,1,2,3\ntwo,1,2,3\nthree,1,2,3\nfour,1,2,3\nfive,1,2,3\n,1,2,3\nseven,1,2,3\neight,1,2,3'
    parser.read_csv(StringIO(data), verbose=True, index_col=0)
    captured = capsys.readouterr()
    if (parser.engine == 'c'):
        assert ('Tokenization took:' in captured.out)
        assert ('Parser memory cleanup took:' in captured.out)
    else:
        assert (captured.out == 'Filled 1 NA values in column a\n')

def test_iteration_open_handle(all_parsers):
    parser = all_parsers
    kwargs = {'squeeze': True, 'header': None}
    with tm.ensure_clean() as path:
        with open(path, 'w') as f:
            f.write('AAA\nBBB\nCCC\nDDD\nEEE\nFFF\nGGG')
        with open(path) as f:
            for line in f:
                if ('CCC' in line):
                    break
            result = parser.read_csv(f, **kwargs)
            expected = Series(['DDD', 'EEE', 'FFF', 'GGG'], name=0)
            tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('data,thousands,decimal', [('A|B|C\n1|2,334.01|5\n10|13|10.\n', ',', '.'), ('A|B|C\n1|2.334,01|5\n10|13|10,\n', '.', ',')])
def test_1000_sep_with_decimal(all_parsers, data, thousands, decimal):
    parser = all_parsers
    expected = DataFrame({'A': [1, 10], 'B': [2334.01, 13], 'C': [5, 10.0]})
    result = parser.read_csv(StringIO(data), sep='|', thousands=thousands, decimal=decimal)
    tm.assert_frame_equal(result, expected)

def test_euro_decimal_format(all_parsers):
    parser = all_parsers
    data = 'Id;Number1;Number2;Text1;Text2;Number3\n1;1521,1541;187101,9543;ABC;poi;4,738797819\n2;121,12;14897,76;DEF;uyt;0,377320872\n3;878,158;108013,434;GHI;rez;2,735694704'
    result = parser.read_csv(StringIO(data), sep=';', decimal=',')
    expected = DataFrame([[1, 1521.1541, 187101.9543, 'ABC', 'poi', 4.738797819], [2, 121.12, 14897.76, 'DEF', 'uyt', 0.377320872], [3, 878.158, 108013.434, 'GHI', 'rez', 2.735694704]], columns=['Id', 'Number1', 'Number2', 'Text1', 'Text2', 'Number3'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('na_filter', [True, False])
def test_inf_parsing(all_parsers, na_filter):
    parser = all_parsers
    data = ',A\na,inf\nb,-inf\nc,+Inf\nd,-Inf\ne,INF\nf,-INF\ng,+INf\nh,-INf\ni,inF\nj,-inF'
    expected = DataFrame({'A': ([float('inf'), float('-inf')] * 5)}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('na_filter', [True, False])
def test_infinity_parsing(all_parsers, na_filter):
    parser = all_parsers
    data = ',A\na,Infinity\nb,-Infinity\nc,+Infinity\n'
    expected = DataFrame({'A': [float('infinity'), float('-infinity'), float('+infinity')]}, index=['a', 'b', 'c'])
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('nrows', [0, 1, 2, 3, 4, 5])
def test_raise_on_no_columns(all_parsers, nrows):
    parser = all_parsers
    data = ('\n' * nrows)
    msg = 'No columns to parse from file'
    with pytest.raises(EmptyDataError, match=msg):
        parser.read_csv(StringIO(data))

@td.check_file_leaks
def test_memory_map(all_parsers, csv_dir_path):
    mmap_file = os.path.join(csv_dir_path, 'test_mmap.csv')
    parser = all_parsers
    expected = DataFrame({'a': [1, 2, 3], 'b': ['one', 'two', 'three'], 'c': ['I', 'II', 'III']})
    result = parser.read_csv(mmap_file, memory_map=True)
    tm.assert_frame_equal(result, expected)

def test_null_byte_char(all_parsers):
    data = '\x00,foo'
    names = ['a', 'b']
    parser = all_parsers
    if (parser.engine == 'c'):
        expected = DataFrame([[np.nan, 'foo']], columns=names)
        out = parser.read_csv(StringIO(data), names=names)
        tm.assert_frame_equal(out, expected)
    else:
        msg = 'NULL byte detected'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), names=names)

def test_temporary_file(all_parsers):
    parser = all_parsers
    data = '0 0'
    with tm.ensure_clean(mode='w+', return_filelike=True) as new_file:
        new_file.write(data)
        new_file.flush()
        new_file.seek(0)
        result = parser.read_csv(new_file, sep='\\s+', header=None)
        expected = DataFrame([[0, 0]])
        tm.assert_frame_equal(result, expected)

def test_internal_eof_byte(all_parsers):
    parser = all_parsers
    data = 'a,b\n1\x1a,2'
    expected = DataFrame([['1\x1a', 2]], columns=['a', 'b'])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)

def test_internal_eof_byte_to_file(all_parsers):
    parser = all_parsers
    data = b'c1,c2\r\n"test \x1a    test", test\r\n'
    expected = DataFrame([['test \x1a    test', ' test']], columns=['c1', 'c2'])
    path = f'__{tm.rands(10)}__.csv'
    with tm.ensure_clean(path) as path:
        with open(path, 'wb') as f:
            f.write(data)
        result = parser.read_csv(path)
        tm.assert_frame_equal(result, expected)

def test_sub_character(all_parsers, csv_dir_path):
    filename = os.path.join(csv_dir_path, 'sub_char.csv')
    expected = DataFrame([[1, 2, 3]], columns=['a', '\x1ab', 'c'])
    parser = all_parsers
    result = parser.read_csv(filename)
    tm.assert_frame_equal(result, expected)

def test_file_handle_string_io(all_parsers):
    parser = all_parsers
    data = 'a,b\n1,2'
    fh = StringIO(data)
    parser.read_csv(fh)
    assert (not fh.closed)

def test_file_handles_with_open(all_parsers, csv1):
    parser = all_parsers
    for mode in ['r', 'rb']:
        with open(csv1, mode) as f:
            parser.read_csv(f)
            assert (not f.closed)

def test_invalid_file_buffer_class(all_parsers):

    class InvalidBuffer():
        pass
    parser = all_parsers
    msg = 'Invalid file path or buffer object type'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(InvalidBuffer())

def test_invalid_file_buffer_mock(all_parsers):
    parser = all_parsers
    msg = 'Invalid file path or buffer object type'

    class Foo():
        pass
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(Foo())

def test_valid_file_buffer_seems_invalid(all_parsers):

    class NoSeekTellBuffer(StringIO):

        def tell(self):
            raise AttributeError('No tell method')

        def seek(self, pos, whence=0):
            raise AttributeError('No seek method')
    data = 'a\n1'
    parser = all_parsers
    expected = DataFrame({'a': [1]})
    result = parser.read_csv(NoSeekTellBuffer(data))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('kwargs', [{}, {'error_bad_lines': True}])
@pytest.mark.parametrize('warn_kwargs', [{}, {'warn_bad_lines': True}, {'warn_bad_lines': False}])
def test_error_bad_lines(all_parsers, kwargs, warn_kwargs):
    parser = all_parsers
    kwargs.update(**warn_kwargs)
    data = 'a\n1\n1,2,3\n4\n5,6,7'
    msg = 'Expected 1 fields in line 3, saw 3'
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)

def test_warn_bad_lines(all_parsers, capsys):
    parser = all_parsers
    data = 'a\n1\n1,2,3\n4\n5,6,7'
    expected = DataFrame({'a': [1, 4]})
    result = parser.read_csv(StringIO(data), error_bad_lines=False, warn_bad_lines=True)
    tm.assert_frame_equal(result, expected)
    captured = capsys.readouterr()
    assert ('Skipping line 3' in captured.err)
    assert ('Skipping line 5' in captured.err)

def test_suppress_error_output(all_parsers, capsys):
    parser = all_parsers
    data = 'a\n1\n1,2,3\n4\n5,6,7'
    expected = DataFrame({'a': [1, 4]})
    result = parser.read_csv(StringIO(data), error_bad_lines=False, warn_bad_lines=False)
    tm.assert_frame_equal(result, expected)
    captured = capsys.readouterr()
    assert (captured.err == '')

@pytest.mark.parametrize('filename', ['sé-es-vé.csv', 'ru-sй.csv', '中文文件名.csv'])
def test_filename_with_special_chars(all_parsers, filename):
    parser = all_parsers
    df = DataFrame({'a': [1, 2, 3]})
    with tm.ensure_clean(filename) as path:
        df.to_csv(path, index=False)
        result = parser.read_csv(path)
        tm.assert_frame_equal(result, df)

def test_read_csv_memory_growth_chunksize(all_parsers):
    parser = all_parsers
    with tm.ensure_clean() as path:
        with open(path, 'w') as f:
            for i in range(1000):
                f.write((str(i) + '\n'))
        with parser.read_csv(path, chunksize=20) as result:
            for _ in result:
                pass

def test_read_csv_raises_on_header_prefix(all_parsers):
    parser = all_parsers
    msg = 'Argument prefix must be None if argument header is not None'
    s = StringIO('0,1\n2,3')
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(s, header=0, prefix='_X')

def test_unexpected_keyword_parameter_exception(all_parsers):
    parser = all_parsers
    msg = "{}\\(\\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg.format('read_csv')):
        parser.read_csv('foo.csv', foo=1)
    with pytest.raises(TypeError, match=msg.format('read_table')):
        parser.read_table('foo.tsv', foo=1)

def test_read_table_same_signature_as_read_csv(all_parsers):
    parser = all_parsers
    table_sign = signature(parser.read_table)
    csv_sign = signature(parser.read_csv)
    assert (table_sign.parameters.keys() == csv_sign.parameters.keys())
    assert (table_sign.return_annotation == csv_sign.return_annotation)
    for (key, csv_param) in csv_sign.parameters.items():
        table_param = table_sign.parameters[key]
        if (key == 'sep'):
            assert (csv_param.default == ',')
            assert (table_param.default == '\t')
            assert (table_param.annotation == csv_param.annotation)
            assert (table_param.kind == csv_param.kind)
            continue
        else:
            assert (table_param == csv_param)

def test_read_table_equivalency_to_read_csv(all_parsers):
    parser = all_parsers
    data = 'a\tb\n1\t2\n3\t4'
    expected = parser.read_csv(StringIO(data), sep='\t')
    result = parser.read_table(StringIO(data))
    tm.assert_frame_equal(result, expected)

def test_first_row_bom(all_parsers):
    parser = all_parsers
    data = '\ufeff"Head1"\t"Head2"\t"Head3"'
    result = parser.read_csv(StringIO(data), delimiter='\t')
    expected = DataFrame(columns=['Head1', 'Head2', 'Head3'])
    tm.assert_frame_equal(result, expected)

def test_first_row_bom_unquoted(all_parsers):
    parser = all_parsers
    data = '\ufeffHead1\tHead2\tHead3'
    result = parser.read_csv(StringIO(data), delimiter='\t')
    expected = DataFrame(columns=['Head1', 'Head2', 'Head3'])
    tm.assert_frame_equal(result, expected)

def test_integer_precision(all_parsers):
    s = '1,1;0;0;0;1;1;3844;3844;3844;1;1;1;1;1;1;0;0;1;1;0;0,,,4321583677327450765\n5,1;0;0;0;1;1;843;843;843;1;1;1;1;1;1;0;0;1;1;0;0,64.0,;,4321113141090630389'
    parser = all_parsers
    result = parser.read_csv(StringIO(s), header=None)[4]
    expected = Series([4321583677327450765, 4321113141090630389], name=4)
    tm.assert_series_equal(result, expected)

def test_file_descriptor_leak(all_parsers):
    parser = all_parsers
    with tm.ensure_clean() as path:

        def test():
            with pytest.raises(EmptyDataError, match='No columns to parse from file'):
                parser.read_csv(path)
        td.check_file_leaks(test)()

@pytest.mark.parametrize('nrows', range(1, 6))
def test_blank_lines_between_header_and_data_rows(all_parsers, nrows):
    ref = DataFrame([[np.nan, np.nan], [np.nan, np.nan], [1, 2], [np.nan, np.nan], [3, 4]], columns=list('ab'))
    csv = '\nheader\n\na,b\n\n\n1,2\n\n3,4'
    parser = all_parsers
    df = parser.read_csv(StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False)
    tm.assert_frame_equal(df, ref[:nrows])

def test_no_header_two_extra_columns(all_parsers):
    column_names = ['one', 'two', 'three']
    ref = DataFrame([['foo', 'bar', 'baz']], columns=column_names)
    stream = StringIO('foo,bar,baz,bam,blah')
    parser = all_parsers
    df = parser.read_csv(stream, header=None, names=column_names, index_col=False)
    tm.assert_frame_equal(df, ref)

def test_read_csv_names_not_accepting_sets(all_parsers):
    data = '    1,2,3\n    4,5,6\n'
    parser = all_parsers
    with pytest.raises(ValueError, match='Names should be an ordered collection.'):
        parser.read_csv(StringIO(data), names=set('QAZ'))

def test_read_csv_with_use_inf_as_na(all_parsers):
    parser = all_parsers
    data = '1.0\nNaN\n3.0'
    with option_context('use_inf_as_na', True):
        result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([1.0, np.nan, 3.0])
    tm.assert_frame_equal(result, expected)

def test_read_table_delim_whitespace_default_sep(all_parsers):
    f = StringIO('a  b  c\n1 -2 -3\n4  5   6')
    parser = all_parsers
    result = parser.read_table(f, delim_whitespace=True)
    expected = DataFrame({'a': [1, 4], 'b': [(- 2), 5], 'c': [(- 3), 6]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('delimiter', [',', '\t'])
def test_read_csv_delim_whitespace_non_default_sep(all_parsers, delimiter):
    f = StringIO('a  b  c\n1 -2 -3\n4  5   6')
    parser = all_parsers
    msg = 'Specified a delimiter with both sep and delim_whitespace=True; you can only specify one.'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(f, delim_whitespace=True, sep=delimiter)
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(f, delim_whitespace=True, delimiter=delimiter)

@pytest.mark.parametrize('delimiter', [',', '\t'])
def test_read_table_delim_whitespace_non_default_sep(all_parsers, delimiter):
    f = StringIO('a  b  c\n1 -2 -3\n4  5   6')
    parser = all_parsers
    msg = 'Specified a delimiter with both sep and delim_whitespace=True; you can only specify one.'
    with pytest.raises(ValueError, match=msg):
        parser.read_table(f, delim_whitespace=True, sep=delimiter)
    with pytest.raises(ValueError, match=msg):
        parser.read_table(f, delim_whitespace=True, delimiter=delimiter)

def test_dict_keys_as_names(all_parsers):
    data = '1,2'
    keys = {'a': int, 'b': int}.keys()
    parser = all_parsers
    result = parser.read_csv(StringIO(data), names=keys)
    expected = DataFrame({'a': [1], 'b': [2]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('io_class', [StringIO, BytesIO])
@pytest.mark.parametrize('encoding', [None, 'utf-8'])
def test_read_csv_file_handle(all_parsers, io_class, encoding):
    '\n    Test whether read_csv does not close user-provided file handles.\n\n    GH 36980\n    '
    parser = all_parsers
    expected = DataFrame({'a': [1], 'b': [2]})
    content = 'a,b\n1,2'
    if (io_class == BytesIO):
        content = content.encode('utf-8')
    handle = io_class(content)
    tm.assert_frame_equal(parser.read_csv(handle, encoding=encoding), expected)
    assert (not handle.closed)

def test_memory_map_file_handle_silent_fallback(all_parsers, compression):
    '\n    Do not fail for buffers with memory_map=True (cannot memory map BytesIO).\n\n    GH 37621\n    '
    parser = all_parsers
    expected = DataFrame({'a': [1], 'b': [2]})
    handle = BytesIO()
    expected.to_csv(handle, index=False, compression=compression, mode='wb')
    handle.seek(0)
    tm.assert_frame_equal(parser.read_csv(handle, memory_map=True, compression=compression), expected)

def test_memory_map_compression(all_parsers, compression):
    '\n    Support memory map for compressed files.\n\n    GH 37621\n    '
    parser = all_parsers
    expected = DataFrame({'a': [1], 'b': [2]})
    with tm.ensure_clean() as path:
        expected.to_csv(path, index=False, compression=compression)
        tm.assert_frame_equal(parser.read_csv(path, memory_map=True, compression=compression), expected)

def test_context_manager(all_parsers, datapath):
    parser = all_parsers
    path = datapath('io', 'data', 'csv', 'iris.csv')
    reader = parser.read_csv(path, chunksize=1)
    assert (not reader._engine.handles.handle.closed)
    try:
        with reader:
            next(reader)
            assert False
    except AssertionError:
        assert reader._engine.handles.handle.closed

def test_context_manageri_user_provided(all_parsers, datapath):
    parser = all_parsers
    with open(datapath('io', 'data', 'csv', 'iris.csv'), mode='r') as path:
        reader = parser.read_csv(path, chunksize=1)
        assert (not reader._engine.handles.handle.closed)
        try:
            with reader:
                next(reader)
                assert False
        except AssertionError:
            assert (not reader._engine.handles.handle.closed)
