
'\nTests that apply specifically to the Python parser. Unless specifically\nstated as a Python-specific issue, the goal is to eventually move as many of\nthese tests out of this module as soon as the C parser can accept further\narguments when parsing.\n'
import csv
from io import BytesIO, StringIO
import pytest
from pandas.errors import ParserError
from pandas import DataFrame, Index, MultiIndex
import pandas._testing as tm

def test_default_separator(python_parser_only):
    data = 'aob\n1o2\n3o4'
    parser = python_parser_only
    expected = DataFrame({'a': [1, 3], 'b': [2, 4]})
    result = parser.read_csv(StringIO(data), sep=None)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('skipfooter', ['foo', 1.5, True])
def test_invalid_skipfooter_non_int(python_parser_only, skipfooter):
    data = 'a\n1\n2'
    parser = python_parser_only
    msg = 'skipfooter must be an integer'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=skipfooter)

def test_invalid_skipfooter_negative(python_parser_only):
    data = 'a\n1\n2'
    parser = python_parser_only
    msg = 'skipfooter cannot be negative'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=(- 1))

@pytest.mark.parametrize('kwargs', [{'sep': None}, {'delimiter': '|'}])
def test_sniff_delimiter(python_parser_only, kwargs):
    data = 'index|A|B|C\nfoo|1|2|3\nbar|4|5|6\nbaz|7|8|9\n'
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=Index(['foo', 'bar', 'baz'], name='index'))
    tm.assert_frame_equal(result, expected)

def test_sniff_delimiter_comment(python_parser_only):
    data = '# comment line\nindex|A|B|C\n# comment line\nfoo|1|2|3 # ignore | this\nbar|4|5|6\nbaz|7|8|9\n'
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, sep=None, comment='#')
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=Index(['foo', 'bar', 'baz'], name='index'))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('encoding', [None, 'utf-8'])
def test_sniff_delimiter_encoding(python_parser_only, encoding):
    parser = python_parser_only
    data = 'ignore this\nignore this too\nindex|A|B|C\nfoo|1|2|3\nbar|4|5|6\nbaz|7|8|9\n'
    if (encoding is not None):
        from io import TextIOWrapper
        data = data.encode(encoding)
        data = BytesIO(data)
        data = TextIOWrapper(data, encoding=encoding)
    else:
        data = StringIO(data)
    result = parser.read_csv(data, index_col=0, sep=None, skiprows=2, encoding=encoding)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=Index(['foo', 'bar', 'baz'], name='index'))
    tm.assert_frame_equal(result, expected)

def test_single_line(python_parser_only):
    parser = python_parser_only
    result = parser.read_csv(StringIO('1,2'), names=['a', 'b'], header=None, sep=None)
    expected = DataFrame({'a': [1], 'b': [2]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('kwargs', [{'skipfooter': 2}, {'nrows': 3}])
def test_skipfooter(python_parser_only, kwargs):
    data = 'A,B,C\n1,2,3\n4,5,6\n7,8,9\nwant to skip this\nalso also skip this\n'
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), **kwargs)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('compression,klass', [('gzip', 'GzipFile'), ('bz2', 'BZ2File')])
def test_decompression_regex_sep(python_parser_only, csv1, compression, klass):
    parser = python_parser_only
    with open(csv1, 'rb') as f:
        data = f.read()
    data = data.replace(b',', b'::')
    expected = parser.read_csv(csv1)
    module = pytest.importorskip(compression)
    klass = getattr(module, klass)
    with tm.ensure_clean() as path:
        tmp = klass(path, mode='wb')
        tmp.write(data)
        tmp.close()
        result = parser.read_csv(path, sep='::', compression=compression)
        tm.assert_frame_equal(result, expected)

def test_read_csv_buglet_4x_multi_index(python_parser_only):
    data = '                      A       B       C       D        E\none two three   four\na   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640\na   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744\nx   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838'
    parser = python_parser_only
    expected = DataFrame([[(- 0.5109), (- 2.3358), (- 0.4645), 0.05076, 0.364], [0.4473, 1.4152, 0.2834, 1.00661, 0.1744], [(- 0.6662), (- 0.5243), (- 0.358), 0.89145, 2.5838]], columns=['A', 'B', 'C', 'D', 'E'], index=MultiIndex.from_tuples([('a', 'b', 10.0032, 5), ('a', 'q', 20, 4), ('x', 'q', 30, 3)], names=['one', 'two', 'three', 'four']))
    result = parser.read_csv(StringIO(data), sep='\\s+')
    tm.assert_frame_equal(result, expected)

def test_read_csv_buglet_4x_multi_index2(python_parser_only):
    data = '      A B C\na b c\n1 3 7 0 3 6\n3 1 4 1 5 9'
    parser = python_parser_only
    expected = DataFrame.from_records([(1, 3, 7, 0, 3, 6), (3, 1, 4, 1, 5, 9)], columns=list('abcABC'), index=list('abc'))
    result = parser.read_csv(StringIO(data), sep='\\s+')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('add_footer', [True, False])
def test_skipfooter_with_decimal(python_parser_only, add_footer):
    data = '1#2\n3#4'
    parser = python_parser_only
    expected = DataFrame({'a': [1.2, 3.4]})
    if add_footer:
        kwargs = {'skipfooter': 1}
        data += '\nFooter'
    else:
        kwargs = {}
    result = parser.read_csv(StringIO(data), names=['a'], decimal='#', **kwargs)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('sep', ['::', '#####', '!!!', '123', '#1!c5', '%!c!d', '@@#4:2', '_!pd#_'])
@pytest.mark.parametrize('encoding', ['utf-16', 'utf-16-be', 'utf-16-le', 'utf-32', 'cp037'])
def test_encoding_non_utf8_multichar_sep(python_parser_only, sep, encoding):
    expected = DataFrame({'a': [1], 'b': [2]})
    parser = python_parser_only
    data = (('1' + sep) + '2')
    encoded_data = data.encode(encoding)
    result = parser.read_csv(BytesIO(encoded_data), sep=sep, names=['a', 'b'], encoding=encoding)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('quoting', [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
def test_multi_char_sep_quotes(python_parser_only, quoting):
    kwargs = {'sep': ',,'}
    parser = python_parser_only
    data = 'a,,b\n1,,a\n2,,"2,,b"'
    if (quoting == csv.QUOTE_NONE):
        msg = 'Expected 2 fields in line 3, saw 3'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)
    else:
        msg = 'ignored when a multi-char delimiter is used'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)

def test_none_delimiter(python_parser_only, capsys):
    parser = python_parser_only
    data = 'a,b,c\n0,1,2\n3,4,5,6\n7,8,9'
    expected = DataFrame({'a': [0, 7], 'b': [1, 8], 'c': [2, 9]})
    result = parser.read_csv(StringIO(data), header=0, sep=None, warn_bad_lines=True, error_bad_lines=False)
    tm.assert_frame_equal(result, expected)
    captured = capsys.readouterr()
    assert ('Skipping line 3' in captured.err)

@pytest.mark.parametrize('data', ['a\n1\n"b"a', 'a,b,c\ncat,foo,bar\ndog,foo,"baz'])
@pytest.mark.parametrize('skipfooter', [0, 1])
def test_skipfooter_bad_row(python_parser_only, data, skipfooter):
    parser = python_parser_only
    if skipfooter:
        msg = 'parsing errors in the skipped footer rows'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)
    else:
        msg = 'unexpected end of data|expected after'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)

def test_malformed_skipfooter(python_parser_only):
    parser = python_parser_only
    data = 'ignore\nA,B,C\n1,2,3 # comment\n1,2,3,4,5\n2,3,4\nfooter\n'
    msg = 'Expected 3 fields in line 4, saw 5'
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), header=1, comment='#', skipfooter=1)
