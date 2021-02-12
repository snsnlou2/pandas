
'Common IO api utilities'
import bz2
from collections import abc
import dataclasses
import gzip
from io import BufferedIOBase, BytesIO, RawIOBase, StringIO, TextIOWrapper
import mmap
import os
from typing import IO, Any, AnyStr, Dict, List, Mapping, Optional, Tuple, Union, cast
from urllib.parse import urljoin, urlparse as parse_url, uses_netloc, uses_params, uses_relative
import warnings
import zipfile
from pandas._typing import Buffer, CompressionDict, CompressionOptions, FileOrBuffer, FilePathOrBuffer, StorageOptions
from pandas.compat import get_lzma_file, import_lzma
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import is_file_like
lzma = import_lzma()
_VALID_URLS = set(((uses_relative + uses_netloc) + uses_params))
_VALID_URLS.discard('')

@dataclasses.dataclass
class IOArgs():
    '\n    Return value of io/common.py:_get_filepath_or_buffer.\n\n    Note (copy&past from io/parsers):\n    filepath_or_buffer can be Union[FilePathOrBuffer, s3fs.S3File, gcsfs.GCSFile]\n    though mypy handling of conditional imports is difficult.\n    See https://github.com/python/mypy/issues/1297\n    '
    should_close = False

@dataclasses.dataclass
class IOHandles():
    '\n    Return value of io/common.py:get_handle\n\n    Can be used as a context manager.\n\n    This is used to easily close created buffers and to handle corner cases when\n    TextIOWrapper is inserted.\n\n    handle: The file handle to be used.\n    created_handles: All file handles that are created by get_handle\n    is_wrapped: Whether a TextIOWrapper needs to be detached.\n    '
    created_handles = dataclasses.field(default_factory=list)
    is_wrapped = False
    is_mmap = False

    def close(self):
        '\n        Close all created buffers.\n\n        Note: If a TextIOWrapper was inserted, it is flushed and detached to\n        avoid closing the potentially user-created buffer.\n        '
        if self.is_wrapped:
            assert isinstance(self.handle, TextIOWrapper)
            self.handle.flush()
            self.handle.detach()
            self.created_handles.remove(self.handle)
        try:
            for handle in self.created_handles:
                handle.close()
        except (OSError, ValueError):
            pass
        self.created_handles = []
        self.is_wrapped = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def is_url(url):
    '\n    Check to see if a URL has a valid protocol.\n\n    Parameters\n    ----------\n    url : str or unicode\n\n    Returns\n    -------\n    isurl : bool\n        If `url` has a valid protocol return True otherwise False.\n    '
    if (not isinstance(url, str)):
        return False
    return (parse_url(url).scheme in _VALID_URLS)

def _expand_user(filepath_or_buffer):
    "\n    Return the argument with an initial component of ~ or ~user\n    replaced by that user's home directory.\n\n    Parameters\n    ----------\n    filepath_or_buffer : object to be converted if possible\n\n    Returns\n    -------\n    expanded_filepath_or_buffer : an expanded filepath or the\n                                  input if not expandable\n    "
    if isinstance(filepath_or_buffer, str):
        return os.path.expanduser(filepath_or_buffer)
    return filepath_or_buffer

def validate_header_arg(header):
    if isinstance(header, bool):
        raise TypeError('Passing a bool to header is invalid. Use header=None for no header or header=int or list-like of ints to specify the row(s) making up the column names')

def stringify_path(filepath_or_buffer, convert_file_like=False):
    "\n    Attempt to convert a path-like object to a string.\n\n    Parameters\n    ----------\n    filepath_or_buffer : object to be converted\n\n    Returns\n    -------\n    str_filepath_or_buffer : maybe a string version of the object\n\n    Notes\n    -----\n    Objects supporting the fspath protocol (python 3.6+) are coerced\n    according to its __fspath__ method.\n\n    Any other object is passed through unchanged, which includes bytes,\n    strings, buffers, or anything else that's not even path-like.\n    "
    if ((not convert_file_like) and is_file_like(filepath_or_buffer)):
        return cast(FileOrBuffer[AnyStr], filepath_or_buffer)
    if isinstance(filepath_or_buffer, os.PathLike):
        filepath_or_buffer = filepath_or_buffer.__fspath__()
    return _expand_user(filepath_or_buffer)

def urlopen(*args, **kwargs):
    '\n    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of\n    the stdlib.\n    '
    import urllib.request
    return urllib.request.urlopen(*args, **kwargs)

def is_fsspec_url(url):
    '\n    Returns true if the given URL looks like\n    something fsspec can handle\n    '
    return (isinstance(url, str) and ('://' in url) and (not url.startswith(('http://', 'https://'))))

def _get_filepath_or_buffer(filepath_or_buffer, encoding='utf-8', compression=None, mode='r', storage_options=None):
    '\n    If the filepath_or_buffer is a url, translate and return the buffer.\n    Otherwise passthrough.\n\n    Parameters\n    ----------\n    filepath_or_buffer : a url, filepath (str, py.path.local or pathlib.Path),\n                         or buffer\n    compression : {{\'gzip\', \'bz2\', \'zip\', \'xz\', None}}, optional\n    encoding : the encoding to use to decode bytes, default is \'utf-8\'\n    mode : str, optional\n\n    storage_options : dict, optional\n        Extra options that make sense for a particular storage connection, e.g.\n        host, port, username, password, etc., if using a URL that will\n        be parsed by ``fsspec``, e.g., starting "s3://", "gcs://". An error\n        will be raised if providing this argument with a local path or\n        a file-like buffer. See the fsspec and backend storage implementation\n        docs for the set of allowed keys and values\n\n        .. versionadded:: 1.2.0\n\n    ..versionchange:: 1.2.0\n\n      Returns the dataclass IOArgs.\n    '
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    (compression_method, compression) = get_compression_method(compression)
    compression_method = infer_compression(filepath_or_buffer, compression_method)
    if (compression_method and hasattr(filepath_or_buffer, 'write') and ('b' not in mode)):
        warnings.warn('compression has no effect when passing a non-binary object as input.', RuntimeWarning, stacklevel=2)
        compression_method = None
    compression = dict(compression, method=compression_method)
    if (encoding is not None):
        encoding = encoding.replace('_', '-').lower()
    if (('w' in mode) and (compression_method in ['bz2', 'xz']) and (encoding in ['utf-16', 'utf-32'])):
        warnings.warn(f'{compression} will not write the byte order mark for {encoding}', UnicodeWarning)
    fsspec_mode = mode
    if (('t' not in fsspec_mode) and ('b' not in fsspec_mode)):
        fsspec_mode += 'b'
    if (isinstance(filepath_or_buffer, str) and is_url(filepath_or_buffer)):
        storage_options = (storage_options or {})
        import urllib.request
        req_info = urllib.request.Request(filepath_or_buffer, headers=storage_options)
        req = urlopen(req_info)
        content_encoding = req.headers.get('Content-Encoding', None)
        if (content_encoding == 'gzip'):
            compression = {'method': 'gzip'}
        reader = BytesIO(req.read())
        req.close()
        return IOArgs(filepath_or_buffer=reader, encoding=encoding, compression=compression, should_close=True, mode=fsspec_mode)
    if is_fsspec_url(filepath_or_buffer):
        assert isinstance(filepath_or_buffer, str)
        if filepath_or_buffer.startswith('s3a://'):
            filepath_or_buffer = filepath_or_buffer.replace('s3a://', 's3://')
        if filepath_or_buffer.startswith('s3n://'):
            filepath_or_buffer = filepath_or_buffer.replace('s3n://', 's3://')
        fsspec = import_optional_dependency('fsspec')
        err_types_to_retry_with_anon: List[Any] = []
        try:
            import_optional_dependency('botocore')
            from botocore.exceptions import ClientError, NoCredentialsError
            err_types_to_retry_with_anon = [ClientError, NoCredentialsError, PermissionError]
        except ImportError:
            pass
        try:
            file_obj = fsspec.open(filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})).open()
        except tuple(err_types_to_retry_with_anon):
            if (storage_options is None):
                storage_options = {'anon': True}
            else:
                storage_options = dict(storage_options)
                storage_options['anon'] = True
            file_obj = fsspec.open(filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})).open()
        return IOArgs(filepath_or_buffer=file_obj, encoding=encoding, compression=compression, should_close=True, mode=fsspec_mode)
    elif storage_options:
        raise ValueError('storage_options passed with file object or non-fsspec file path')
    if isinstance(filepath_or_buffer, (str, bytes, mmap.mmap)):
        return IOArgs(filepath_or_buffer=_expand_user(filepath_or_buffer), encoding=encoding, compression=compression, should_close=False, mode=mode)
    if (not is_file_like(filepath_or_buffer)):
        msg = f'Invalid file path or buffer object type: {type(filepath_or_buffer)}'
        raise ValueError(msg)
    return IOArgs(filepath_or_buffer=filepath_or_buffer, encoding=encoding, compression=compression, should_close=False, mode=mode)

def file_path_to_url(path):
    '\n    converts an absolute native path to a FILE URL.\n\n    Parameters\n    ----------\n    path : a path in native format\n\n    Returns\n    -------\n    a valid FILE URL\n    '
    from urllib.request import pathname2url
    return urljoin('file:', pathname2url(path))
_compression_to_extension = {'gzip': '.gz', 'bz2': '.bz2', 'zip': '.zip', 'xz': '.xz'}

def get_compression_method(compression):
    "\n    Simplifies a compression argument to a compression method string and\n    a mapping containing additional arguments.\n\n    Parameters\n    ----------\n    compression : str or mapping\n        If string, specifies the compression method. If mapping, value at key\n        'method' specifies compression method.\n\n    Returns\n    -------\n    tuple of ({compression method}, Optional[str]\n              {compression arguments}, Dict[str, Any])\n\n    Raises\n    ------\n    ValueError on mapping missing 'method' key\n    "
    compression_method: Optional[str]
    if isinstance(compression, Mapping):
        compression_args = dict(compression)
        try:
            compression_method = compression_args.pop('method')
        except KeyError as err:
            raise ValueError("If mapping, compression must have key 'method'") from err
    else:
        compression_args = {}
        compression_method = compression
    return (compression_method, compression_args)

def infer_compression(filepath_or_buffer, compression):
    "\n    Get the compression method for filepath_or_buffer. If compression='infer',\n    the inferred compression method is returned. Otherwise, the input\n    compression method is returned unchanged, unless it's invalid, in which\n    case an error is raised.\n\n    Parameters\n    ----------\n    filepath_or_buffer : str or file handle\n        File path or object.\n    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}\n        If 'infer' and `filepath_or_buffer` is path-like, then detect\n        compression from the following extensions: '.gz', '.bz2', '.zip',\n        or '.xz' (otherwise no compression).\n\n    Returns\n    -------\n    string or None\n\n    Raises\n    ------\n    ValueError on invalid compression specified.\n    "
    if (compression is None):
        return None
    if (compression == 'infer'):
        filepath_or_buffer = stringify_path(filepath_or_buffer, convert_file_like=True)
        if (not isinstance(filepath_or_buffer, str)):
            return None
        for (compression, extension) in _compression_to_extension.items():
            if filepath_or_buffer.lower().endswith(extension):
                return compression
        return None
    if (compression in _compression_to_extension):
        return compression
    msg = f'Unrecognized compression type: {compression}'
    valid = (['infer', None] + sorted(_compression_to_extension))
    msg += f'''
Valid compression types are {valid}'''
    raise ValueError(msg)

def get_handle(path_or_buf, mode, encoding=None, compression=None, memory_map=False, is_text=True, errors=None, storage_options=None):
    '\n    Get file handle for given path/buffer and mode.\n\n    Parameters\n    ----------\n    path_or_buf : str or file handle\n        File path or object.\n    mode : str\n        Mode to open path_or_buf with.\n    encoding : str or None\n        Encoding to use.\n    compression : str or dict, default None\n        If string, specifies compression mode. If dict, value at key \'method\'\n        specifies compression mode. Compression mode must be one of {\'infer\',\n        \'gzip\', \'bz2\', \'zip\', \'xz\', None}. If compression mode is \'infer\'\n        and `filepath_or_buffer` is path-like, then detect compression from\n        the following extensions: \'.gz\', \'.bz2\', \'.zip\', or \'.xz\' (otherwise\n        no compression). If dict and compression mode is one of\n        {\'zip\', \'gzip\', \'bz2\'}, or inferred as one of the above,\n        other entries passed as additional compression options.\n\n        .. versionchanged:: 1.0.0\n\n           May now be a dict with key \'method\' as compression mode\n           and other keys as compression options if compression\n           mode is \'zip\'.\n\n        .. versionchanged:: 1.1.0\n\n           Passing compression options as keys in dict is now\n           supported for compression modes \'gzip\' and \'bz2\' as well as \'zip\'.\n\n    memory_map : boolean, default False\n        See parsers._parser_params for more information.\n    is_text : boolean, default True\n        Whether the type of the content passed to the file/buffer is string or\n        bytes. This is not the same as `"b" not in mode`. If a string content is\n        passed to a binary file/buffer, a wrapper is inserted.\n    errors : str, default \'strict\'\n        Specifies how encoding and decoding errors are to be handled.\n        See the errors argument for :func:`open` for a full list\n        of options.\n    storage_options: StorageOptions = None\n        Passed to _get_filepath_or_buffer\n\n    .. versionchanged:: 1.2.0\n\n    Returns the dataclass IOHandles\n    '
    if (encoding is None):
        encoding = 'utf-8'
    if (_is_binary_mode(path_or_buf, mode) and ('b' not in mode)):
        mode += 'b'
    ioargs = _get_filepath_or_buffer(path_or_buf, encoding=encoding, compression=compression, mode=mode, storage_options=storage_options)
    handle = ioargs.filepath_or_buffer
    handles: List[Buffer]
    (handle, memory_map, handles) = _maybe_memory_map(handle, memory_map, ioargs.encoding, ioargs.mode, errors)
    is_path = isinstance(handle, str)
    compression_args = dict(ioargs.compression)
    compression = compression_args.pop('method')
    if compression:
        ioargs.mode = ioargs.mode.replace('t', '')
        if (compression == 'gzip'):
            if is_path:
                assert isinstance(handle, str)
                handle = gzip.GzipFile(filename=handle, mode=ioargs.mode, **compression_args)
            else:
                handle = gzip.GzipFile(fileobj=handle, mode=ioargs.mode, **compression_args)
        elif (compression == 'bz2'):
            handle = bz2.BZ2File(handle, mode=ioargs.mode, **compression_args)
        elif (compression == 'zip'):
            handle = _BytesZipFile(handle, ioargs.mode, **compression_args)
            if (handle.mode == 'r'):
                handles.append(handle)
                zip_names = handle.namelist()
                if (len(zip_names) == 1):
                    handle = handle.open(zip_names.pop())
                elif (len(zip_names) == 0):
                    raise ValueError(f'Zero files found in ZIP file {path_or_buf}')
                else:
                    raise ValueError(f'Multiple files found in ZIP file. Only one file per ZIP: {zip_names}')
        elif (compression == 'xz'):
            handle = get_lzma_file(lzma)(handle, ioargs.mode)
        else:
            msg = f'Unrecognized compression type: {compression}'
            raise ValueError(msg)
        assert (not isinstance(handle, str))
        handles.append(handle)
    elif isinstance(handle, str):
        if (ioargs.encoding and ('b' not in ioargs.mode)):
            handle = open(handle, ioargs.mode, encoding=ioargs.encoding, errors=errors, newline='')
        else:
            handle = open(handle, ioargs.mode)
        handles.append(handle)
    is_wrapped = False
    if (is_text and (compression or _is_binary_mode(handle, ioargs.mode))):
        handle = TextIOWrapper(handle, encoding=ioargs.encoding, errors=errors, newline='')
        handles.append(handle)
        is_wrapped = (not (isinstance(ioargs.filepath_or_buffer, str) or ioargs.should_close))
    handles.reverse()
    if ioargs.should_close:
        assert (not isinstance(ioargs.filepath_or_buffer, str))
        handles.append(ioargs.filepath_or_buffer)
    assert (not isinstance(handle, str))
    return IOHandles(handle=handle, created_handles=handles, is_wrapped=is_wrapped, is_mmap=memory_map, compression=ioargs.compression)

class _BytesZipFile(zipfile.ZipFile, BytesIO):
    '\n    Wrapper for standard library class ZipFile and allow the returned file-like\n    handle to accept byte strings via `write` method.\n\n    BytesIO provides attributes of file-like object and ZipFile.writestr writes\n    bytes strings into a member of the archive.\n    '

    def __init__(self, file, mode, archive_name=None, **kwargs):
        mode = mode.replace('b', '')
        self.archive_name = archive_name
        self.multiple_write_buffer: Optional[Union[(StringIO, BytesIO)]] = None
        kwargs_zip: Dict[(str, Any)] = {'compression': zipfile.ZIP_DEFLATED}
        kwargs_zip.update(kwargs)
        super().__init__(file, mode, **kwargs_zip)

    def write(self, data):
        if (self.multiple_write_buffer is None):
            self.multiple_write_buffer = (BytesIO() if isinstance(data, bytes) else StringIO())
        self.multiple_write_buffer.write(data)

    def flush(self):
        if ((self.multiple_write_buffer is None) or self.multiple_write_buffer.closed):
            return
        archive_name = (self.archive_name or self.filename or 'zip')
        with self.multiple_write_buffer:
            super().writestr(archive_name, self.multiple_write_buffer.getvalue())

    def close(self):
        self.flush()
        super().close()

    @property
    def closed(self):
        return (self.fp is None)

class _MMapWrapper(abc.Iterator):
    "\n    Wrapper for the Python's mmap class so that it can be properly read in\n    by Python's csv.reader class.\n\n    Parameters\n    ----------\n    f : file object\n        File object to be mapped onto memory. Must support the 'fileno'\n        method or have an equivalent attribute\n\n    "

    def __init__(self, f):
        self.attributes = {}
        for attribute in ('seekable', 'readable', 'writeable'):
            if (not hasattr(f, attribute)):
                continue
            self.attributes[attribute] = getattr(f, attribute)()
        self.mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __getattr__(self, name):
        if (name in self.attributes):
            return (lambda : self.attributes[name])
        return getattr(self.mmap, name)

    def __iter__(self):
        return self

    def __next__(self):
        newbytes = self.mmap.readline()
        newline = newbytes.decode('utf-8')
        if (newline == ''):
            raise StopIteration
        return newline

def _maybe_memory_map(handle, memory_map, encoding, mode, errors):
    'Try to memory map file/buffer.'
    handles: List[Buffer] = []
    memory_map &= (hasattr(handle, 'fileno') or isinstance(handle, str))
    if (not memory_map):
        return (handle, memory_map, handles)
    if isinstance(handle, str):
        if (encoding and ('b' not in mode)):
            handle = open(handle, mode, encoding=encoding, errors=errors, newline='')
        else:
            handle = open(handle, mode)
        handles.append(handle)
    try:
        wrapped = cast(mmap.mmap, _MMapWrapper(handle))
        handle.close()
        handles.remove(handle)
        handles.append(wrapped)
        handle = wrapped
    except Exception:
        memory_map = False
    return (handle, memory_map, handles)

def file_exists(filepath_or_buffer):
    'Test whether file exists.'
    exists = False
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    if (not isinstance(filepath_or_buffer, str)):
        return exists
    try:
        exists = os.path.exists(filepath_or_buffer)
    except (TypeError, ValueError):
        pass
    return exists

def _is_binary_mode(handle, mode):
    'Whether the handle is opened in binary mode'
    binary_classes = [BufferedIOBase, RawIOBase]
    return (isinstance(handle, tuple(binary_classes)) or ('b' in getattr(handle, 'mode', mode)))
