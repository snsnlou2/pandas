
from io import BytesIO
import os
import numpy as np
import pytest
from pandas import DataFrame, date_range, read_csv, read_excel, read_json, read_parquet
import pandas._testing as tm
from pandas.util import _test_decorators as td

@pytest.fixture
def gcs_buffer(monkeypatch):
    'Emulate GCS using a binary buffer.'
    from fsspec import AbstractFileSystem, registry
    registry.target.clear()
    gcs_buffer = BytesIO()
    gcs_buffer.close = (lambda : True)

    class MockGCSFileSystem(AbstractFileSystem):

        def open(*args, **kwargs):
            gcs_buffer.seek(0)
            return gcs_buffer

        def ls(self, path, **kwargs):
            return [{'name': path, 'type': 'file'}]
    monkeypatch.setattr('gcsfs.GCSFileSystem', MockGCSFileSystem)
    return gcs_buffer

@td.skip_if_no('gcsfs')
@pytest.mark.parametrize('format', ['csv', 'json', 'parquet', 'excel', 'markdown'])
def test_to_read_gcs(gcs_buffer, format):
    '\n    Test that many to/read functions support GCS.\n\n    GH 33987\n    '
    from fsspec import registry
    registry.target.clear()
    df1 = DataFrame({'int': [1, 3], 'float': [2.0, np.nan], 'str': ['t', 's'], 'dt': date_range('2018-06-18', periods=2)})
    path = f'gs://test/test.{format}'
    if (format == 'csv'):
        df1.to_csv(path, index=True)
        df2 = read_csv(path, parse_dates=['dt'], index_col=0)
    elif (format == 'excel'):
        path = 'gs://test/test.xls'
        df1.to_excel(path)
        df2 = read_excel(path, parse_dates=['dt'], index_col=0)
    elif (format == 'json'):
        df1.to_json(path)
        df2 = read_json(path, convert_dates=['dt'])
    elif (format == 'parquet'):
        pytest.importorskip('pyarrow')
        df1.to_parquet(path)
        df2 = read_parquet(path)
    elif (format == 'markdown'):
        pytest.importorskip('tabulate')
        df1.to_markdown(path)
        df2 = df1
    tm.assert_frame_equal(df1, df2)

@td.skip_if_no('gcsfs')
@pytest.mark.parametrize('encoding', ['utf-8', 'cp1251'])
def test_to_csv_compression_encoding_gcs(gcs_buffer, compression_only, encoding):
    '\n    Compression and encoding should with GCS.\n\n    GH 35677 (to_csv, compression), GH 26124 (to_csv, encoding), and\n    GH 32392 (read_csv, encoding)\n    '
    from fsspec import registry
    registry.target.clear()
    df = tm.makeDataFrame()
    compression = {'method': compression_only}
    if (compression_only == 'gzip'):
        compression['mtime'] = 1
    buffer = BytesIO()
    df.to_csv(buffer, compression=compression, encoding=encoding, mode='wb')
    path_gcs = 'gs://test/test.csv'
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    assert (gcs_buffer.getvalue() == buffer.getvalue())
    read_df = read_csv(path_gcs, index_col=0, compression=compression_only, encoding=encoding)
    tm.assert_frame_equal(df, read_df)
    if (compression_only == 'gzip'):
        compression_only = 'gz'
    compression['method'] = 'infer'
    path_gcs += f'.{compression_only}'
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    assert (gcs_buffer.getvalue() == buffer.getvalue())
    read_df = read_csv(path_gcs, index_col=0, compression='infer', encoding=encoding)
    tm.assert_frame_equal(df, read_df)

@td.skip_if_no('fastparquet')
@td.skip_if_no('gcsfs')
def test_to_parquet_gcs_new_file(monkeypatch, tmpdir):
    'Regression test for writing to a not-yet-existent GCS Parquet file.'
    from fsspec import AbstractFileSystem, registry
    registry.target.clear()
    df1 = DataFrame({'int': [1, 3], 'float': [2.0, np.nan], 'str': ['t', 's'], 'dt': date_range('2018-06-18', periods=2)})

    class MockGCSFileSystem(AbstractFileSystem):

        def open(self, path, mode='r', *args):
            if ('w' not in mode):
                raise FileNotFoundError
            return open(os.path.join(tmpdir, 'test.parquet'), mode)
    monkeypatch.setattr('gcsfs.GCSFileSystem', MockGCSFileSystem)
    df1.to_parquet('gs://test/test.csv', index=True, engine='fastparquet', compression=None)

@td.skip_if_installed('gcsfs')
def test_gcs_not_present_exception():
    with tm.external_error_raised(ImportError):
        read_csv('gs://test/test.csv')
