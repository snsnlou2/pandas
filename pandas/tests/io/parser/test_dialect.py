
'\nTests that dialects are properly handled during parsing\nfor all of the parsers defined in parsers.py\n'
import csv
from io import StringIO
import pytest
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm

@pytest.fixture
def custom_dialect():
    dialect_name = 'weird'
    dialect_kwargs = {'doublequote': False, 'escapechar': '~', 'delimiter': ':', 'skipinitialspace': False, 'quotechar': '~', 'quoting': 3}
    return (dialect_name, dialect_kwargs)

def test_dialect(all_parsers):
    parser = all_parsers
    data = 'label1,label2,label3\nindex1,"a,c,e\nindex2,b,d,f\n'
    dia = csv.excel()
    dia.quoting = csv.QUOTE_NONE
    df = parser.read_csv(StringIO(data), dialect=dia)
    data = 'label1,label2,label3\nindex1,a,c,e\nindex2,b,d,f\n'
    exp = parser.read_csv(StringIO(data))
    exp.replace('a', '"a', inplace=True)
    tm.assert_frame_equal(df, exp)

def test_dialect_str(all_parsers):
    dialect_name = 'mydialect'
    parser = all_parsers
    data = 'fruit:vegetable\napple:broccoli\npear:tomato\n'
    exp = DataFrame({'fruit': ['apple', 'pear'], 'vegetable': ['broccoli', 'tomato']})
    with tm.with_csv_dialect(dialect_name, delimiter=':'):
        df = parser.read_csv(StringIO(data), dialect=dialect_name)
        tm.assert_frame_equal(df, exp)

def test_invalid_dialect(all_parsers):

    class InvalidDialect():
        pass
    data = 'a\n1'
    parser = all_parsers
    msg = 'Invalid dialect'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dialect=InvalidDialect)

@pytest.mark.parametrize('arg', [None, 'doublequote', 'escapechar', 'skipinitialspace', 'quotechar', 'quoting'])
@pytest.mark.parametrize('value', ['dialect', 'default', 'other'])
def test_dialect_conflict_except_delimiter(all_parsers, custom_dialect, arg, value):
    (dialect_name, dialect_kwargs) = custom_dialect
    parser = all_parsers
    expected = DataFrame({'a': [1], 'b': [2]})
    data = 'a:b\n1:2'
    warning_klass = None
    kwds = {}
    if (arg is not None):
        if ('value' == 'dialect'):
            kwds[arg] = dialect_kwargs[arg]
        elif ('value' == 'default'):
            from pandas.io.parsers import _parser_defaults
            kwds[arg] = _parser_defaults[arg]
        else:
            warning_klass = ParserWarning
            kwds[arg] = 'blah'
    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        with tm.assert_produces_warning(warning_klass):
            result = parser.read_csv(StringIO(data), dialect=dialect_name, **kwds)
            tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('kwargs,warning_klass', [({'sep': ','}, None), ({'sep': '.'}, ParserWarning), ({'delimiter': ':'}, None), ({'delimiter': None}, None), ({'delimiter': ','}, ParserWarning), ({'delimiter': '.'}, ParserWarning)], ids=['sep-override-true', 'sep-override-false', 'delimiter-no-conflict', 'delimiter-default-arg', 'delimiter-conflict', 'delimiter-conflict2'])
def test_dialect_conflict_delimiter(all_parsers, custom_dialect, kwargs, warning_klass):
    (dialect_name, dialect_kwargs) = custom_dialect
    parser = all_parsers
    expected = DataFrame({'a': [1], 'b': [2]})
    data = 'a:b\n1:2'
    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        with tm.assert_produces_warning(warning_klass):
            result = parser.read_csv(StringIO(data), dialect=dialect_name, **kwargs)
            tm.assert_frame_equal(result, expected)
