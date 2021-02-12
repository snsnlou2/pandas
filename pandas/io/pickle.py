
' pickle compat '
import pickle
from typing import Any
import warnings
from pandas._typing import CompressionOptions, FilePathOrBuffer, StorageOptions
from pandas.compat import pickle_compat as pc
from pandas.util._decorators import doc
from pandas.core import generic
from pandas.io.common import get_handle

@doc(storage_options=generic._shared_docs['storage_options'])
def to_pickle(obj, filepath_or_buffer, compression='infer', protocol=pickle.HIGHEST_PROTOCOL, storage_options=None):
    '\n    Pickle (serialize) object to file.\n\n    Parameters\n    ----------\n    obj : any object\n        Any python object.\n    filepath_or_buffer : str, path object or file-like object\n        File path, URL, or buffer where the pickled object will be stored.\n\n        .. versionchanged:: 1.0.0\n           Accept URL. URL has to be of S3 or GCS.\n\n    compression : {{\'infer\', \'gzip\', \'bz2\', \'zip\', \'xz\', None}}, default \'infer\'\n        If \'infer\' and \'path_or_url\' is path-like, then detect compression from\n        the following extensions: \'.gz\', \'.bz2\', \'.zip\', or \'.xz\' (otherwise no\n        compression) If \'infer\' and \'path_or_url\' is not path-like, then use\n        None (= no decompression).\n    protocol : int\n        Int which indicates which protocol should be used by the pickler,\n        default HIGHEST_PROTOCOL (see [1], paragraph 12.1.2). The possible\n        values for this parameter depend on the version of Python. For Python\n        2.x, possible values are 0, 1, 2. For Python>=3.0, 3 is a valid value.\n        For Python >= 3.4, 4 is a valid value. A negative value for the\n        protocol parameter is equivalent to setting its value to\n        HIGHEST_PROTOCOL.\n\n    {storage_options}\n\n        .. versionadded:: 1.2.0\n\n        .. [1] https://docs.python.org/3/library/pickle.html\n\n    See Also\n    --------\n    read_pickle : Load pickled pandas object (or any object) from file.\n    DataFrame.to_hdf : Write DataFrame to an HDF5 file.\n    DataFrame.to_sql : Write DataFrame to a SQL database.\n    DataFrame.to_parquet : Write a DataFrame to the binary parquet format.\n\n    Examples\n    --------\n    >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})\n    >>> original_df\n       foo  bar\n    0    0    5\n    1    1    6\n    2    2    7\n    3    3    8\n    4    4    9\n    >>> pd.to_pickle(original_df, "./dummy.pkl")\n\n    >>> unpickled_df = pd.read_pickle("./dummy.pkl")\n    >>> unpickled_df\n       foo  bar\n    0    0    5\n    1    1    6\n    2    2    7\n    3    3    8\n    4    4    9\n\n    >>> import os\n    >>> os.remove("./dummy.pkl")\n    '
    if (protocol < 0):
        protocol = pickle.HIGHEST_PROTOCOL
    with get_handle(filepath_or_buffer, 'wb', compression=compression, is_text=False, storage_options=storage_options) as handles:
        pickle.dump(obj, handles.handle, protocol=protocol)

@doc(storage_options=generic._shared_docs['storage_options'])
def read_pickle(filepath_or_buffer, compression='infer', storage_options=None):
    '\n    Load pickled pandas object (or any object) from file.\n\n    .. warning::\n\n       Loading pickled data received from untrusted sources can be\n       unsafe. See `here <https://docs.python.org/3/library/pickle.html>`__.\n\n    Parameters\n    ----------\n    filepath_or_buffer : str, path object or file-like object\n        File path, URL, or buffer where the pickled object will be loaded from.\n\n        .. versionchanged:: 1.0.0\n           Accept URL. URL is not limited to S3 and GCS.\n\n    compression : {{\'infer\', \'gzip\', \'bz2\', \'zip\', \'xz\', None}}, default \'infer\'\n        If \'infer\' and \'path_or_url\' is path-like, then detect compression from\n        the following extensions: \'.gz\', \'.bz2\', \'.zip\', or \'.xz\' (otherwise no\n        compression) If \'infer\' and \'path_or_url\' is not path-like, then use\n        None (= no decompression).\n\n    {storage_options}\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    unpickled : same type as object stored in file\n\n    See Also\n    --------\n    DataFrame.to_pickle : Pickle (serialize) DataFrame object to file.\n    Series.to_pickle : Pickle (serialize) Series object to file.\n    read_hdf : Read HDF5 file into a DataFrame.\n    read_sql : Read SQL query or database table into a DataFrame.\n    read_parquet : Load a parquet object, returning a DataFrame.\n\n    Notes\n    -----\n    read_pickle is only guaranteed to be backwards compatible to pandas 0.20.3.\n\n    Examples\n    --------\n    >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})\n    >>> original_df\n       foo  bar\n    0    0    5\n    1    1    6\n    2    2    7\n    3    3    8\n    4    4    9\n    >>> pd.to_pickle(original_df, "./dummy.pkl")\n\n    >>> unpickled_df = pd.read_pickle("./dummy.pkl")\n    >>> unpickled_df\n       foo  bar\n    0    0    5\n    1    1    6\n    2    2    7\n    3    3    8\n    4    4    9\n\n    >>> import os\n    >>> os.remove("./dummy.pkl")\n    '
    excs_to_catch = (AttributeError, ImportError, ModuleNotFoundError, TypeError)
    with get_handle(filepath_or_buffer, 'rb', compression=compression, is_text=False, storage_options=storage_options) as handles:
        try:
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('ignore', Warning)
                    return pickle.load(handles.handle)
            except excs_to_catch:
                return pc.load(handles.handle, encoding=None)
        except UnicodeDecodeError:
            return pc.load(handles.handle, encoding='latin-1')
