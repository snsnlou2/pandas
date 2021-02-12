
' feather-format compat '
from typing import AnyStr
from pandas._typing import FilePathOrBuffer, StorageOptions
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas import DataFrame, Int64Index, RangeIndex
from pandas.core import generic
from pandas.io.common import get_handle

@doc(storage_options=generic._shared_docs['storage_options'])
def to_feather(df, path, storage_options=None, **kwargs):
    '\n    Write a DataFrame to the binary Feather format.\n\n    Parameters\n    ----------\n    df : DataFrame\n    path : string file path, or file-like object\n    {storage_options}\n\n        .. versionadded:: 1.2.0\n\n    **kwargs :\n        Additional keywords passed to `pyarrow.feather.write_feather`.\n\n        .. versionadded:: 1.1.0\n    '
    import_optional_dependency('pyarrow')
    from pyarrow import feather
    if (not isinstance(df, DataFrame)):
        raise ValueError('feather only support IO with DataFrames')
    valid_types = {'string', 'unicode'}
    if (not isinstance(df.index, Int64Index)):
        typ = type(df.index)
        raise ValueError(f'feather does not support serializing {typ} for the index; you can .reset_index() to make the index into column(s)')
    if (not df.index.equals(RangeIndex.from_range(range(len(df))))):
        raise ValueError('feather does not support serializing a non-default index for the index; you can .reset_index() to make the index into column(s)')
    if (df.index.name is not None):
        raise ValueError('feather does not serialize index meta-data on a default index')
    if (df.columns.inferred_type not in valid_types):
        raise ValueError('feather must have string column names')
    with get_handle(path, 'wb', storage_options=storage_options, is_text=False) as handles:
        feather.write_feather(df, handles.handle, **kwargs)

@doc(storage_options=generic._shared_docs['storage_options'])
def read_feather(path, columns=None, use_threads=True, storage_options=None):
    '\n    Load a feather-format object from the file path.\n\n    Parameters\n    ----------\n    path : str, path object or file-like object\n        Any valid string path is acceptable. The string could be a URL. Valid\n        URL schemes include http, ftp, s3, and file. For file URLs, a host is\n        expected. A local file could be:\n        ``file://localhost/path/to/table.feather``.\n\n        If you want to pass in a path object, pandas accepts any\n        ``os.PathLike``.\n\n        By file-like object, we refer to objects with a ``read()`` method,\n        such as a file handle (e.g. via builtin ``open`` function)\n        or ``StringIO``.\n    columns : sequence, default None\n        If not provided, all columns are read.\n\n        .. versionadded:: 0.24.0\n    use_threads : bool, default True\n        Whether to parallelize reading using multiple threads.\n\n       .. versionadded:: 0.24.0\n    {storage_options}\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    type of object stored in file\n    '
    import_optional_dependency('pyarrow')
    from pyarrow import feather
    with get_handle(path, 'rb', storage_options=storage_options, is_text=False) as handles:
        return feather.read_feather(handles.handle, columns=columns, use_threads=bool(use_threads))
