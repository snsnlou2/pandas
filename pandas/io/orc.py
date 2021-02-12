
' orc compat '
import distutils
from typing import TYPE_CHECKING, List, Optional
from pandas._typing import FilePathOrBuffer
from pandas.io.common import get_handle
if TYPE_CHECKING:
    from pandas import DataFrame

def read_orc(path, columns=None, **kwargs):
    '\n    Load an ORC object from the file path, returning a DataFrame.\n\n    .. versionadded:: 1.0.0\n\n    Parameters\n    ----------\n    path : str, path object or file-like object\n        Any valid string path is acceptable. The string could be a URL. Valid\n        URL schemes include http, ftp, s3, and file. For file URLs, a host is\n        expected. A local file could be:\n        ``file://localhost/path/to/table.orc``.\n\n        If you want to pass in a path object, pandas accepts any\n        ``os.PathLike``.\n\n        By file-like object, we refer to objects with a ``read()`` method,\n        such as a file handle (e.g. via builtin ``open`` function)\n        or ``StringIO``.\n    columns : list, default None\n        If not None, only these columns will be read from the file.\n    **kwargs\n        Any additional kwargs are passed to pyarrow.\n\n    Returns\n    -------\n    DataFrame\n    '
    import pyarrow
    if (distutils.version.LooseVersion(pyarrow.__version__) < '0.13.0'):
        raise ImportError('pyarrow must be >= 0.13.0 for read_orc')
    with get_handle(path, 'rb', is_text=False) as handles:
        orc_file = pyarrow.orc.ORCFile(handles.handle)
        return orc_file.read(columns=columns, **kwargs).to_pandas()
