
from contextlib import contextmanager
import os
from shutil import rmtree
import tempfile
from pandas.io.common import get_handle

@contextmanager
def decompress_file(path, compression):
    "\n    Open a compressed file and return a file object.\n\n    Parameters\n    ----------\n    path : str\n        The path where the file is read from.\n\n    compression : {'gzip', 'bz2', 'zip', 'xz', None}\n        Name of the decompression to use\n\n    Returns\n    -------\n    file object\n    "
    with get_handle(path, 'rb', compression=compression, is_text=False) as handle:
        (yield handle.handle)

@contextmanager
def set_timezone(tz):
    "\n    Context manager for temporarily setting a timezone.\n\n    Parameters\n    ----------\n    tz : str\n        A string representing a valid timezone.\n\n    Examples\n    --------\n    >>> from datetime import datetime\n    >>> from dateutil.tz import tzlocal\n    >>> tzlocal().tzname(datetime.now())\n    'IST'\n\n    >>> with set_timezone('US/Eastern'):\n    ...     tzlocal().tzname(datetime.now())\n    ...\n    'EDT'\n    "
    import os
    import time

    def setTZ(tz):
        if (tz is None):
            try:
                del os.environ['TZ']
            except KeyError:
                pass
        else:
            os.environ['TZ'] = tz
            time.tzset()
    orig_tz = os.environ.get('TZ')
    setTZ(tz)
    try:
        (yield)
    finally:
        setTZ(orig_tz)

@contextmanager
def ensure_clean(filename=None, return_filelike=False, **kwargs):
    '\n    Gets a temporary path and agrees to remove on close.\n\n    Parameters\n    ----------\n    filename : str (optional)\n        if None, creates a temporary file which is then removed when out of\n        scope. if passed, creates temporary file with filename as ending.\n    return_filelike : bool (default False)\n        if True, returns a file-like which is *always* cleaned. Necessary for\n        savefig and other functions which want to append extensions.\n    **kwargs\n        Additional keywords passed in for creating a temporary file.\n        :meth:`tempFile.TemporaryFile` is used when `return_filelike` is ``True``.\n        :meth:`tempfile.mkstemp` is used when `return_filelike` is ``False``.\n        Note that the `filename` parameter will be passed in as the `suffix`\n        argument to either function.\n\n    See Also\n    --------\n    tempfile.TemporaryFile\n    tempfile.mkstemp\n    '
    filename = (filename or '')
    fd = None
    kwargs['suffix'] = filename
    if return_filelike:
        f = tempfile.TemporaryFile(**kwargs)
        try:
            (yield f)
        finally:
            f.close()
    else:
        if len(os.path.dirname(filename)):
            raise ValueError("Can't pass a qualified name to ensure_clean()")
        try:
            (fd, filename) = tempfile.mkstemp(**kwargs)
        except UnicodeEncodeError:
            import pytest
            pytest.skip('no unicode file names on this system')
        try:
            (yield filename)
        finally:
            try:
                os.close(fd)
            except OSError:
                print(f"Couldn't close file descriptor: {fd} (file: {filename})")
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except OSError as e:
                print(f'Exception on removing file: {e}')

@contextmanager
def ensure_clean_dir():
    '\n    Get a temporary directory path and agrees to remove on close.\n\n    Yields\n    ------\n    Temporary directory path\n    '
    directory_name = tempfile.mkdtemp(suffix='')
    try:
        (yield directory_name)
    finally:
        try:
            rmtree(directory_name)
        except OSError:
            pass

@contextmanager
def ensure_safe_environment_variables():
    '\n    Get a context manager to safely set environment variables\n\n    All changes will be undone on close, hence environment variables set\n    within this contextmanager will neither persist nor change global state.\n    '
    saved_environ = dict(os.environ)
    try:
        (yield)
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)

@contextmanager
def with_csv_dialect(name, **kwargs):
    "\n    Context manager to temporarily register a CSV dialect for parsing CSV.\n\n    Parameters\n    ----------\n    name : str\n        The name of the dialect.\n    kwargs : mapping\n        The parameters for the dialect.\n\n    Raises\n    ------\n    ValueError : the name of the dialect conflicts with a builtin one.\n\n    See Also\n    --------\n    csv : Python's CSV library.\n    "
    import csv
    _BUILTIN_DIALECTS = {'excel', 'excel-tab', 'unix'}
    if (name in _BUILTIN_DIALECTS):
        raise ValueError('Cannot override builtin dialect.')
    csv.register_dialect(name, **kwargs)
    (yield)
    csv.unregister_dialect(name)

@contextmanager
def use_numexpr(use, min_elements=None):
    from pandas.core.computation import expressions as expr
    if (min_elements is None):
        min_elements = expr._MIN_ELEMENTS
    olduse = expr.USE_NUMEXPR
    oldmin = expr._MIN_ELEMENTS
    expr.set_use_numexpr(use)
    expr._MIN_ELEMENTS = min_elements
    (yield)
    expr._MIN_ELEMENTS = oldmin
    expr.set_use_numexpr(olduse)
