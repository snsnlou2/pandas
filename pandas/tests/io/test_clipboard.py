
from textwrap import dedent
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, get_option, read_clipboard
import pandas._testing as tm
from pandas.io.clipboard import clipboard_get, clipboard_set

def build_kwargs(sep, excel):
    kwargs = {}
    if (excel != 'default'):
        kwargs['excel'] = excel
    if (sep != 'default'):
        kwargs['sep'] = sep
    return kwargs

@pytest.fixture(params=['delims', 'utf8', 'utf16', 'string', 'long', 'nonascii', 'colwidth', 'mixed', 'float', 'int'])
def df(request):
    data_type = request.param
    if (data_type == 'delims'):
        return DataFrame({'a': ['"a,\t"b|c', 'd\tef´'], 'b': ["hi'j", "k''lm"]})
    elif (data_type == 'utf8'):
        return DataFrame({'a': ['µasd', 'Ωœ∑´'], 'b': ['øπ∆˚¬', 'œ∑´®']})
    elif (data_type == 'utf16'):
        return DataFrame({'a': ['👍👍', '👍👍'], 'b': ['abc', 'def']})
    elif (data_type == 'string'):
        return tm.makeCustomDataframe(5, 3, c_idx_type='s', r_idx_type='i', c_idx_names=[None], r_idx_names=[None])
    elif (data_type == 'long'):
        max_rows = get_option('display.max_rows')
        return tm.makeCustomDataframe((max_rows + 1), 3, data_gen_f=(lambda *args: np.random.randint(2)), c_idx_type='s', r_idx_type='i', c_idx_names=[None], r_idx_names=[None])
    elif (data_type == 'nonascii'):
        return DataFrame({'en': 'in English'.split(), 'es': 'en español'.split()})
    elif (data_type == 'colwidth'):
        _cw = (get_option('display.max_colwidth') + 1)
        return tm.makeCustomDataframe(5, 3, data_gen_f=(lambda *args: ('x' * _cw)), c_idx_type='s', r_idx_type='i', c_idx_names=[None], r_idx_names=[None])
    elif (data_type == 'mixed'):
        return DataFrame({'a': (np.arange(1.0, 6.0) + 0.01), 'b': np.arange(1, 6).astype(np.int64), 'c': list('abcde')})
    elif (data_type == 'float'):
        return tm.makeCustomDataframe(5, 3, data_gen_f=(lambda r, c: (float(r) + 0.01)), c_idx_type='s', r_idx_type='i', c_idx_names=[None], r_idx_names=[None])
    elif (data_type == 'int'):
        return tm.makeCustomDataframe(5, 3, data_gen_f=(lambda *args: np.random.randint(2)), c_idx_type='s', r_idx_type='i', c_idx_names=[None], r_idx_names=[None])
    else:
        raise ValueError

@pytest.fixture
def mock_clipboard(monkeypatch, request):
    'Fixture mocking clipboard IO.\n\n    This mocks pandas.io.clipboard.clipboard_get and\n    pandas.io.clipboard.clipboard_set.\n\n    This uses a local dict for storing data. The dictionary\n    key used is the test ID, available with ``request.node.name``.\n\n    This returns the local dictionary, for direct manipulation by\n    tests.\n    '
    _mock_data = {}

    def _mock_set(data):
        _mock_data[request.node.name] = data

    def _mock_get():
        return _mock_data[request.node.name]
    monkeypatch.setattr('pandas.io.clipboard.clipboard_set', _mock_set)
    monkeypatch.setattr('pandas.io.clipboard.clipboard_get', _mock_get)
    (yield _mock_data)

@pytest.mark.clipboard
def test_mock_clipboard(mock_clipboard):
    import pandas.io.clipboard
    pandas.io.clipboard.clipboard_set('abc')
    assert ('abc' in set(mock_clipboard.values()))
    result = pandas.io.clipboard.clipboard_get()
    assert (result == 'abc')

@pytest.mark.single
@pytest.mark.clipboard
@pytest.mark.usefixtures('mock_clipboard')
class TestClipboard():

    def check_round_trip_frame(self, data, excel=None, sep=None, encoding=None):
        data.to_clipboard(excel=excel, sep=sep, encoding=encoding)
        result = read_clipboard(sep=(sep or '\t'), index_col=0, encoding=encoding)
        tm.assert_frame_equal(data, result)

    def test_round_trip_frame(self, df):
        self.check_round_trip_frame(df)

    @pytest.mark.parametrize('sep', ['\t', ',', '|'])
    def test_round_trip_frame_sep(self, df, sep):
        self.check_round_trip_frame(df, sep=sep)

    def test_round_trip_frame_string(self, df):
        df.to_clipboard(excel=False, sep=None)
        result = read_clipboard()
        assert (df.to_string() == result.to_string())
        assert (df.shape == result.shape)

    def test_excel_sep_warning(self, df):
        with tm.assert_produces_warning():
            df.to_clipboard(excel=True, sep='\\t')

    def test_copy_delim_warning(self, df):
        with tm.assert_produces_warning():
            df.to_clipboard(excel=False, sep='\t')

    @pytest.mark.parametrize('sep', ['\t', None, 'default'])
    @pytest.mark.parametrize('excel', [True, None, 'default'])
    def test_clipboard_copy_tabs_default(self, sep, excel, df, request, mock_clipboard):
        kwargs = build_kwargs(sep, excel)
        df.to_clipboard(**kwargs)
        assert (mock_clipboard[request.node.name] == df.to_csv(sep='\t'))

    @pytest.mark.parametrize('sep', [None, 'default'])
    @pytest.mark.parametrize('excel', [False])
    def test_clipboard_copy_strings(self, sep, excel, df):
        kwargs = build_kwargs(sep, excel)
        df.to_clipboard(**kwargs)
        result = read_clipboard(sep='\\s+')
        assert (result.to_string() == df.to_string())
        assert (df.shape == result.shape)

    def test_read_clipboard_infer_excel(self, request, mock_clipboard):
        clip_kwargs = {'engine': 'python'}
        text = dedent('\n            John James\tCharlie Mingus\n            1\t2\n            4\tHarry Carney\n            '.strip())
        mock_clipboard[request.node.name] = text
        df = pd.read_clipboard(**clip_kwargs)
        assert (df.iloc[1][1] == 'Harry Carney')
        text = dedent('\n            a\t b\n            1  2\n            3  4\n            '.strip())
        mock_clipboard[request.node.name] = text
        res = pd.read_clipboard(**clip_kwargs)
        text = dedent('\n            a  b\n            1  2\n            3  4\n            '.strip())
        mock_clipboard[request.node.name] = text
        exp = pd.read_clipboard(**clip_kwargs)
        tm.assert_frame_equal(res, exp)

    def test_invalid_encoding(self, df):
        msg = 'clipboard only supports utf-8 encoding'
        with pytest.raises(ValueError, match=msg):
            df.to_clipboard(encoding='ascii')
        with pytest.raises(NotImplementedError, match=msg):
            pd.read_clipboard(encoding='ascii')

    @pytest.mark.parametrize('enc', ['UTF-8', 'utf-8', 'utf8'])
    def test_round_trip_valid_encodings(self, enc, df):
        self.check_round_trip_frame(df, encoding=enc)

@pytest.mark.single
@pytest.mark.clipboard
@pytest.mark.parametrize('data', ['👍...', 'Ωœ∑´...', 'abcd...'])
def test_raw_roundtrip(data):
    clipboard_set(data)
    assert (data == clipboard_get())
