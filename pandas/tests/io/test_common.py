
'\nTests for the pandas.io.common functionalities\n'
from io import StringIO
import mmap
import os
from pathlib import Path
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom

class CustomFSPath():
    'For testing fspath on unknown objects'

    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return self.path
path_types = [str, CustomFSPath, Path]
try:
    from py.path import local as LocalPath
    path_types.append(LocalPath)
except ImportError:
    pass
HERE = os.path.abspath(os.path.dirname(__file__))

@pytest.mark.filterwarnings("ignore:can't resolve package:ImportWarning")
class TestCommonIOCapabilities():
    data1 = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'

    def test_expand_user(self):
        filename = '~/sometest'
        expanded_name = icom._expand_user(filename)
        assert (expanded_name != filename)
        assert os.path.isabs(expanded_name)
        assert (os.path.expanduser(filename) == expanded_name)

    def test_expand_user_normal_path(self):
        filename = '/somefolder/sometest'
        expanded_name = icom._expand_user(filename)
        assert (expanded_name == filename)
        assert (os.path.expanduser(filename) == expanded_name)

    def test_stringify_path_pathlib(self):
        rel_path = icom.stringify_path(Path('.'))
        assert (rel_path == '.')
        redundant_path = icom.stringify_path(Path('foo//bar'))
        assert (redundant_path == os.path.join('foo', 'bar'))

    @td.skip_if_no('py.path')
    def test_stringify_path_localpath(self):
        path = os.path.join('foo', 'bar')
        abs_path = os.path.abspath(path)
        lpath = LocalPath(path)
        assert (icom.stringify_path(lpath) == abs_path)

    def test_stringify_path_fspath(self):
        p = CustomFSPath('foo/bar.csv')
        result = icom.stringify_path(p)
        assert (result == 'foo/bar.csv')

    def test_stringify_file_and_path_like(self):
        fsspec = pytest.importorskip('fsspec')
        with tm.ensure_clean() as path:
            with fsspec.open(f'file://{path}', mode='wb') as fsspec_obj:
                assert (fsspec_obj == icom.stringify_path(fsspec_obj))

    @pytest.mark.parametrize('extension,expected', [('', None), ('.gz', 'gzip'), ('.bz2', 'bz2'), ('.zip', 'zip'), ('.xz', 'xz'), ('.GZ', 'gzip'), ('.BZ2', 'bz2'), ('.ZIP', 'zip'), ('.XZ', 'xz')])
    @pytest.mark.parametrize('path_type', path_types)
    def test_infer_compression_from_path(self, extension, expected, path_type):
        path = path_type(('foo/bar.csv' + extension))
        compression = icom.infer_compression(path, compression='infer')
        assert (compression == expected)

    @pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type):
        filename = path_type('~/sometest')
        with icom.get_handle(filename, 'w') as handles:
            assert os.path.isabs(handles.handle.name)
            assert (os.path.expanduser(filename) == handles.handle.name)

    def test_get_handle_with_buffer(self):
        input_buffer = StringIO()
        with icom.get_handle(input_buffer, 'r') as handles:
            assert (handles.handle == input_buffer)
        assert (not input_buffer.closed)
        input_buffer.close()

    def test_iterator(self):
        with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
            result = pd.concat(reader, ignore_index=True)
        expected = pd.read_csv(StringIO(self.data1))
        tm.assert_frame_equal(result, expected)
        with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
            first = next(it)
            tm.assert_frame_equal(first, expected.iloc[[0]])
            tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [(pd.read_csv, 'os', FileNotFoundError, 'csv'), (pd.read_fwf, 'os', FileNotFoundError, 'txt'), (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'), (pd.read_feather, 'pyarrow', IOError, 'feather'), (pd.read_hdf, 'tables', FileNotFoundError, 'h5'), (pd.read_stata, 'os', FileNotFoundError, 'dta'), (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'), (pd.read_json, 'os', ValueError, 'json'), (pd.read_pickle, 'os', FileNotFoundError, 'pickle')])
    def test_read_non_existent(self, reader, module, error_class, fn_ext):
        pytest.importorskip(module)
        path = os.path.join(HERE, 'data', ('does_not_exist.' + fn_ext))
        msg1 = f"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = f"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = 'Expected object or value'
        msg4 = 'path_or_buf needs to be a string file path or file-like'
        msg5 = f"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: '.+does_not_exist\.{fn_ext}'"
        msg6 = f"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = f"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        msg8 = f'Failed to open local file.+does_not_exist\.{fn_ext}'
        with pytest.raises(error_class, match=f'({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})'):
            reader(path)

    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [(pd.read_csv, 'os', FileNotFoundError, 'csv'), (pd.read_table, 'os', FileNotFoundError, 'csv'), (pd.read_fwf, 'os', FileNotFoundError, 'txt'), (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'), (pd.read_feather, 'pyarrow', IOError, 'feather'), (pd.read_hdf, 'tables', FileNotFoundError, 'h5'), (pd.read_stata, 'os', FileNotFoundError, 'dta'), (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'), (pd.read_json, 'os', ValueError, 'json'), (pd.read_pickle, 'os', FileNotFoundError, 'pickle')])
    def test_read_expands_user_home_dir(self, reader, module, error_class, fn_ext, monkeypatch):
        pytest.importorskip(module)
        path = os.path.join('~', ('does_not_exist.' + fn_ext))
        monkeypatch.setattr(icom, '_expand_user', (lambda x: os.path.join('foo', x)))
        msg1 = f"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = f"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Unexpected character found when decoding 'false'"
        msg4 = 'path_or_buf needs to be a string file path or file-like'
        msg5 = f"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: '.+does_not_exist\.{fn_ext}'"
        msg6 = f"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = f"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        msg8 = f'Failed to open local file.+does_not_exist\.{fn_ext}'
        with pytest.raises(error_class, match=f'({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})'):
            reader(path)

    @pytest.mark.parametrize('reader, module, path', [(pd.read_csv, 'os', ('io', 'data', 'csv', 'iris.csv')), (pd.read_table, 'os', ('io', 'data', 'csv', 'iris.csv')), (pd.read_fwf, 'os', ('io', 'data', 'fixed_width', 'fixed_width_format.txt')), (pd.read_excel, 'xlrd', ('io', 'data', 'excel', 'test1.xlsx')), (pd.read_feather, 'pyarrow', ('io', 'data', 'feather', 'feather-0_3_1.feather')), (pd.read_hdf, 'tables', ('io', 'data', 'legacy_hdf', 'datetimetz_object.h5')), (pd.read_stata, 'os', ('io', 'data', 'stata', 'stata10_115.dta')), (pd.read_sas, 'os', ('io', 'sas', 'data', 'test1.sas7bdat')), (pd.read_json, 'os', ('io', 'json', 'data', 'tsframe_v012.json')), (pd.read_pickle, 'os', ('io', 'data', 'pickle', 'categorical.0.25.0.pickle'))])
    def test_read_fspath_all(self, reader, module, path, datapath):
        pytest.importorskip(module)
        path = datapath(*path)
        mypath = CustomFSPath(path)
        result = reader(mypath)
        expected = reader(path)
        if path.endswith('.pickle'):
            tm.assert_categorical_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('writer_name, writer_kwargs, module', [('to_csv', {}, 'os'), ('to_excel', {'engine': 'xlwt'}, 'xlwt'), ('to_feather', {}, 'pyarrow'), ('to_html', {}, 'os'), ('to_json', {}, 'os'), ('to_latex', {}, 'os'), ('to_pickle', {}, 'os'), ('to_stata', {'time_stamp': pd.to_datetime('2019-01-01 00:00')}, 'os')])
    def test_write_fspath_all(self, writer_name, writer_kwargs, module):
        p1 = tm.ensure_clean('string')
        p2 = tm.ensure_clean('fspath')
        df = pd.DataFrame({'A': [1, 2]})
        with p1 as string, p2 as fspath:
            pytest.importorskip(module)
            mypath = CustomFSPath(fspath)
            writer = getattr(df, writer_name)
            writer(string, **writer_kwargs)
            with open(string, 'rb') as f:
                expected = f.read()
            writer(mypath, **writer_kwargs)
            with open(fspath, 'rb') as f:
                result = f.read()
            assert (result == expected)

    def test_write_fspath_hdf5(self):
        pytest.importorskip('tables')
        df = pd.DataFrame({'A': [1, 2]})
        p1 = tm.ensure_clean('string')
        p2 = tm.ensure_clean('fspath')
        with p1 as string, p2 as fspath:
            mypath = CustomFSPath(fspath)
            df.to_hdf(mypath, key='bar')
            df.to_hdf(string, key='bar')
            result = pd.read_hdf(fspath, key='bar')
            expected = pd.read_hdf(string, key='bar')
        tm.assert_frame_equal(result, expected)

@pytest.fixture
def mmap_file(datapath):
    return datapath('io', 'data', 'csv', 'test_mmap.csv')

class TestMMapWrapper():

    def test_constructor_bad_file(self, mmap_file):
        non_file = StringIO('I am not a file')
        non_file.fileno = (lambda : (- 1))
        if is_platform_windows():
            msg = 'The parameter is incorrect'
            err = OSError
        else:
            msg = '[Errno 22]'
            err = mmap.error
        with pytest.raises(err, match=msg):
            icom._MMapWrapper(non_file)
        target = open(mmap_file)
        target.close()
        msg = 'I/O operation on closed file'
        with pytest.raises(ValueError, match=msg):
            icom._MMapWrapper(target)

    def test_get_attr(self, mmap_file):
        with open(mmap_file) as target:
            wrapper = icom._MMapWrapper(target)
        attrs = dir(wrapper.mmap)
        attrs = [attr for attr in attrs if (not attr.startswith('__'))]
        attrs.append('__next__')
        for attr in attrs:
            assert hasattr(wrapper, attr)
        assert (not hasattr(wrapper, 'foo'))

    def test_next(self, mmap_file):
        with open(mmap_file) as target:
            wrapper = icom._MMapWrapper(target)
            lines = target.readlines()
        for line in lines:
            next_line = next(wrapper)
            assert (next_line.strip() == line.strip())
        with pytest.raises(StopIteration, match='^$'):
            next(wrapper)

    def test_unknown_engine(self):
        with tm.ensure_clean() as path:
            df = tm.makeDataFrame()
            df.to_csv(path)
            with pytest.raises(ValueError, match='Unknown engine'):
                pd.read_csv(path, engine='pyt')

    def test_binary_mode(self):
        "\n        'encoding' shouldn't be passed to 'open' in binary mode.\n\n        GH 35058\n        "
        with tm.ensure_clean() as path:
            df = tm.makeDataFrame()
            df.to_csv(path, mode='w+b')
            tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))

    @pytest.mark.parametrize('encoding', ['utf-16', 'utf-32'])
    @pytest.mark.parametrize('compression_', ['bz2', 'xz'])
    def test_warning_missing_utf_bom(self, encoding, compression_):
        '\n        bz2 and xz do not write the byte order mark (BOM) for utf-16/32.\n\n        https://stackoverflow.com/questions/55171439\n\n        GH 35681\n        '
        df = tm.makeDataFrame()
        with tm.ensure_clean() as path:
            with tm.assert_produces_warning(UnicodeWarning):
                df.to_csv(path, compression=compression_, encoding=encoding)
            msg = 'UTF-\\d+ stream does not start with BOM'
            with pytest.raises(UnicodeError, match=msg):
                pd.read_csv(path, compression=compression_, encoding=encoding)

def test_is_fsspec_url():
    assert icom.is_fsspec_url('gcs://pandas/somethingelse.com')
    assert icom.is_fsspec_url('gs://pandas/somethingelse.com')
    assert (not icom.is_fsspec_url('http://pandas/somethingelse.com'))
    assert (not icom.is_fsspec_url('random:pandas/somethingelse.com'))
    assert (not icom.is_fsspec_url('/local/path'))
    assert (not icom.is_fsspec_url('relative/local/path'))
