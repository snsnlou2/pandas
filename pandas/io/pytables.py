
'\nHigh level interface to PyTables for reading and writing pandas data structures\nto disk\n'
from contextlib import suppress
import copy
from datetime import date, tzinfo
import itertools
import os
import re
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import warnings
import numpy as np
from pandas._config import config, get_option
from pandas._libs import lib, writers as libwriters
from pandas._libs.tslibs import timezones
from pandas._typing import ArrayLike, DtypeArg, FrameOrSeries, FrameOrSeriesUnion, Label, Shape
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import ensure_object, is_categorical_dtype, is_complex_dtype, is_datetime64_dtype, is_datetime64tz_dtype, is_extension_array_dtype, is_list_like, is_string_dtype, is_timedelta64_dtype, needs_i8_conversion
from pandas.core.dtypes.missing import array_equivalent
from pandas import DataFrame, DatetimeIndex, Index, Int64Index, MultiIndex, PeriodIndex, Series, TimedeltaIndex, concat, isna
from pandas.core.arrays import Categorical, DatetimeArray, PeriodArray
import pandas.core.common as com
from pandas.core.computation.pytables import PyTablesExpr, maybe_expression
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.io.common import stringify_path
from pandas.io.formats.printing import adjoin, pprint_thing
if TYPE_CHECKING:
    from tables import Col, File, Node
_version = '0.15.2'
_default_encoding = 'UTF-8'

def _ensure_decoded(s):
    ' if we have bytes, decode them to unicode '
    if isinstance(s, np.bytes_):
        s = s.decode('UTF-8')
    return s

def _ensure_encoding(encoding):
    if (encoding is None):
        encoding = _default_encoding
    return encoding

def _ensure_str(name):
    '\n    Ensure that an index / column name is a str (python 3); otherwise they\n    may be np.string dtype. Non-string dtypes are passed through unchanged.\n\n    https://github.com/pandas-dev/pandas/issues/13492\n    '
    if isinstance(name, str):
        name = str(name)
    return name
Term = PyTablesExpr

def _ensure_term(where, scope_level):
    '\n    Ensure that the where is a Term or a list of Term.\n\n    This makes sure that we are capturing the scope of variables that are\n    passed create the terms here with a frame_level=2 (we are 2 levels down)\n    '
    level = (scope_level + 1)
    if isinstance(where, (list, tuple)):
        where = [(Term(term, scope_level=(level + 1)) if maybe_expression(term) else term) for term in where if (term is not None)]
    elif maybe_expression(where):
        where = Term(where, scope_level=level)
    return (where if ((where is None) or len(where)) else None)

class PossibleDataLossError(Exception):
    pass

class ClosedFileError(Exception):
    pass

class IncompatibilityWarning(Warning):
    pass
incompatibility_doc = '\nwhere criteria is being ignored as this version [%s] is too old (or\nnot-defined), read the file in and write it out to a new file to upgrade (with\nthe copy_to method)\n'

class AttributeConflictWarning(Warning):
    pass
attribute_conflict_doc = '\nthe [%s] attribute of the existing index is [%s] which conflicts with the new\n[%s], resetting the attribute to None\n'

class DuplicateWarning(Warning):
    pass
duplicate_doc = '\nduplicate entries in table, taking most recently appended\n'
performance_doc = '\nyour performance may suffer as PyTables will pickle object types that it cannot\nmap directly to c-types [inferred_type->%s,key->%s] [items->%s]\n'
_FORMAT_MAP = {'f': 'fixed', 'fixed': 'fixed', 't': 'table', 'table': 'table'}
_AXES_MAP = {DataFrame: [0]}
dropna_doc = '\n: boolean\n    drop ALL nan rows when appending to a table\n'
format_doc = "\n: format\n    default format writing format, if None, then\n    put will default to 'fixed' and append will default to 'table'\n"
with config.config_prefix('io.hdf'):
    config.register_option('dropna_table', False, dropna_doc, validator=config.is_bool)
    config.register_option('default_format', None, format_doc, validator=config.is_one_of_factory(['fixed', 'table', None]))
_table_mod = None
_table_file_open_policy_is_strict = False

def _tables():
    global _table_mod
    global _table_file_open_policy_is_strict
    if (_table_mod is None):
        import tables
        _table_mod = tables
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (tables.file._FILE_OPEN_POLICY == 'strict')
    return _table_mod

def to_hdf(path_or_buf, key, value, mode='a', complevel=None, complib=None, append=False, format=None, index=True, min_itemsize=None, nan_rep=None, dropna=None, data_columns=None, errors='strict', encoding='UTF-8'):
    ' store this object, close it if we opened it '
    if append:
        f = (lambda store: store.append(key, value, format=format, index=index, min_itemsize=min_itemsize, nan_rep=nan_rep, dropna=dropna, data_columns=data_columns, errors=errors, encoding=encoding))
    else:
        f = (lambda store: store.put(key, value, format=format, index=index, min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=data_columns, errors=errors, encoding=encoding, dropna=dropna))
    path_or_buf = stringify_path(path_or_buf)
    if isinstance(path_or_buf, str):
        with HDFStore(path_or_buf, mode=mode, complevel=complevel, complib=complib) as store:
            f(store)
    else:
        f(path_or_buf)

def read_hdf(path_or_buf, key=None, mode='r', errors='strict', where=None, start=None, stop=None, columns=None, iterator=False, chunksize=None, **kwargs):
    '\n    Read from the store, close it if we opened it.\n\n    Retrieve pandas object stored in file, optionally based on where\n    criteria.\n\n    .. warning::\n\n       Pandas uses PyTables for reading and writing HDF5 files, which allows\n       serializing object-dtype data with pickle when using the "fixed" format.\n       Loading pickled data received from untrusted sources can be unsafe.\n\n       See: https://docs.python.org/3/library/pickle.html for more.\n\n    Parameters\n    ----------\n    path_or_buf : str, path object, pandas.HDFStore or file-like object\n        Any valid string path is acceptable. The string could be a URL. Valid\n        URL schemes include http, ftp, s3, and file. For file URLs, a host is\n        expected. A local file could be: ``file://localhost/path/to/table.h5``.\n\n        If you want to pass in a path object, pandas accepts any\n        ``os.PathLike``.\n\n        Alternatively, pandas accepts an open :class:`pandas.HDFStore` object.\n\n        By file-like object, we refer to objects with a ``read()`` method,\n        such as a file handle (e.g. via builtin ``open`` function)\n        or ``StringIO``.\n    key : object, optional\n        The group identifier in the store. Can be omitted if the HDF file\n        contains a single pandas object.\n    mode : {\'r\', \'r+\', \'a\'}, default \'r\'\n        Mode to use when opening the file. Ignored if path_or_buf is a\n        :class:`pandas.HDFStore`. Default is \'r\'.\n    errors : str, default \'strict\'\n        Specifies how encoding and decoding errors are to be handled.\n        See the errors argument for :func:`open` for a full list\n        of options.\n    where : list, optional\n        A list of Term (or convertible) objects.\n    start : int, optional\n        Row number to start selection.\n    stop  : int, optional\n        Row number to stop selection.\n    columns : list, optional\n        A list of columns names to return.\n    iterator : bool, optional\n        Return an iterator object.\n    chunksize : int, optional\n        Number of rows to include in an iteration when using an iterator.\n    **kwargs\n        Additional keyword arguments passed to HDFStore.\n\n    Returns\n    -------\n    item : object\n        The selected object. Return type depends on the object stored.\n\n    See Also\n    --------\n    DataFrame.to_hdf : Write a HDF file from a DataFrame.\n    HDFStore : Low-level access to HDF files.\n\n    Examples\n    --------\n    >>> df = pd.DataFrame([[1, 1.0, \'a\']], columns=[\'x\', \'y\', \'z\'])\n    >>> df.to_hdf(\'./store.h5\', \'data\')\n    >>> reread = pd.read_hdf(\'./store.h5\')\n    '
    if (mode not in ['r', 'r+', 'a']):
        raise ValueError(f'mode {mode} is not allowed while performing a read. Allowed modes are r, r+ and a.')
    if (where is not None):
        where = _ensure_term(where, scope_level=1)
    if isinstance(path_or_buf, HDFStore):
        if (not path_or_buf.is_open):
            raise OSError('The HDFStore must be open for reading.')
        store = path_or_buf
        auto_close = False
    else:
        path_or_buf = stringify_path(path_or_buf)
        if (not isinstance(path_or_buf, str)):
            raise NotImplementedError('Support for generic buffers has not been implemented.')
        try:
            exists = os.path.exists(path_or_buf)
        except (TypeError, ValueError):
            exists = False
        if (not exists):
            raise FileNotFoundError(f'File {path_or_buf} does not exist')
        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        auto_close = True
    try:
        if (key is None):
            groups = store.groups()
            if (len(groups) == 0):
                raise ValueError('Dataset(s) incompatible with Pandas data types, not table, or no datasets found in HDF5 file.')
            candidate_only_group = groups[0]
            for group_to_check in groups[1:]:
                if (not _is_metadata_of(group_to_check, candidate_only_group)):
                    raise ValueError('key must be provided when HDF5 file contains multiple datasets.')
            key = candidate_only_group._v_pathname
        return store.select(key, where=where, start=start, stop=stop, columns=columns, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
    except (ValueError, TypeError, KeyError):
        if (not isinstance(path_or_buf, HDFStore)):
            with suppress(AttributeError):
                store.close()
        raise

def _is_metadata_of(group, parent_group):
    'Check if a given group is a metadata group for a given parent_group.'
    if (group._v_depth <= parent_group._v_depth):
        return False
    current = group
    while (current._v_depth > 1):
        parent = current._v_parent
        if ((parent == parent_group) and (current._v_name == 'meta')):
            return True
        current = current._v_parent
    return False

class HDFStore():
    '\n    Dict-like IO interface for storing pandas objects in PyTables.\n\n    Either Fixed or Table format.\n\n    .. warning::\n\n       Pandas uses PyTables for reading and writing HDF5 files, which allows\n       serializing object-dtype data with pickle when using the "fixed" format.\n       Loading pickled data received from untrusted sources can be unsafe.\n\n       See: https://docs.python.org/3/library/pickle.html for more.\n\n    Parameters\n    ----------\n    path : str\n        File path to HDF5 file.\n    mode : {\'a\', \'w\', \'r\', \'r+\'}, default \'a\'\n\n        ``\'r\'``\n            Read-only; no data can be modified.\n        ``\'w\'``\n            Write; a new file is created (an existing file with the same\n            name would be deleted).\n        ``\'a\'``\n            Append; an existing file is opened for reading and writing,\n            and if the file does not exist it is created.\n        ``\'r+\'``\n            It is similar to ``\'a\'``, but the file must already exist.\n    complevel : int, 0-9, default None\n        Specifies a compression level for data.\n        A value of 0 or None disables compression.\n    complib : {\'zlib\', \'lzo\', \'bzip2\', \'blosc\'}, default \'zlib\'\n        Specifies the compression library to be used.\n        As of v0.20.2 these additional compressors for Blosc are supported\n        (default if no compressor specified: \'blosc:blosclz\'):\n        {\'blosc:blosclz\', \'blosc:lz4\', \'blosc:lz4hc\', \'blosc:snappy\',\n         \'blosc:zlib\', \'blosc:zstd\'}.\n        Specifying a compression library which is not available issues\n        a ValueError.\n    fletcher32 : bool, default False\n        If applying compression use the fletcher32 checksum.\n    **kwargs\n        These parameters will be passed to the PyTables open_file method.\n\n    Examples\n    --------\n    >>> bar = pd.DataFrame(np.random.randn(10, 4))\n    >>> store = pd.HDFStore(\'test.h5\')\n    >>> store[\'foo\'] = bar   # write to HDF5\n    >>> bar = store[\'foo\']   # retrieve\n    >>> store.close()\n\n    **Create or load HDF5 file in-memory**\n\n    When passing the `driver` option to the PyTables open_file method through\n    **kwargs, the HDF5 file is loaded or created in-memory and will only be\n    written when closed:\n\n    >>> bar = pd.DataFrame(np.random.randn(10, 4))\n    >>> store = pd.HDFStore(\'test.h5\', driver=\'H5FD_CORE\')\n    >>> store[\'foo\'] = bar\n    >>> store.close()   # only now, data is written to disk\n    '

    def __init__(self, path, mode='a', complevel=None, complib=None, fletcher32=False, **kwargs):
        if ('format' in kwargs):
            raise ValueError('format is not a defined argument for HDFStore')
        tables = import_optional_dependency('tables')
        if ((complib is not None) and (complib not in tables.filters.all_complibs)):
            raise ValueError(f'complib only supports {tables.filters.all_complibs} compression.')
        if ((complib is None) and (complevel is not None)):
            complib = tables.filters.default_complib
        self._path = stringify_path(path)
        if (mode is None):
            mode = 'a'
        self._mode = mode
        self._handle = None
        self._complevel = (complevel if complevel else 0)
        self._complib = complib
        self._fletcher32 = fletcher32
        self._filters = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self):
        return self._path

    @property
    def root(self):
        ' return the root node '
        self._check_if_open()
        assert (self._handle is not None)
        return self._handle.root

    @property
    def filename(self):
        return self._path

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        return self.remove(key)

    def __getattr__(self, name):
        ' allow attribute access to get stores '
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __contains__(self, key):
        "\n        check for existence of this key\n        can match the exact pathname or the pathnm w/o the leading '/'\n        "
        node = self.get_node(key)
        if (node is not None):
            name = node._v_pathname
            if ((name == key) or (name[1:] == key)):
                return True
        return False

    def __len__(self):
        return len(self.groups())

    def __repr__(self):
        pstr = pprint_thing(self._path)
        return f'''{type(self)}
File path: {pstr}
'''

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def keys(self, include='pandas'):
        "\n        Return a list of keys corresponding to objects stored in HDFStore.\n\n        Parameters\n        ----------\n\n        include : str, default 'pandas'\n                When kind equals 'pandas' return pandas objects.\n                When kind equals 'native' return native HDF5 Table objects.\n\n                .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        list\n            List of ABSOLUTE path-names (e.g. have the leading '/').\n\n        Raises\n        ------\n        raises ValueError if kind has an illegal value\n        "
        if (include == 'pandas'):
            return [n._v_pathname for n in self.groups()]
        elif (include == 'native'):
            assert (self._handle is not None)
            return [n._v_pathname for n in self._handle.walk_nodes('/', classname='Table')]
        raise ValueError(f"`include` should be either 'pandas' or 'native' but is '{include}'")

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        '\n        iterate on key->group\n        '
        for g in self.groups():
            (yield (g._v_pathname, g))
    iteritems = items

    def open(self, mode='a', **kwargs):
        "\n        Open the file in the specified mode\n\n        Parameters\n        ----------\n        mode : {'a', 'w', 'r', 'r+'}, default 'a'\n            See HDFStore docstring or tables.open_file for info about modes\n        **kwargs\n            These parameters will be passed to the PyTables open_file method.\n        "
        tables = _tables()
        if (self._mode != mode):
            if ((self._mode in ['a', 'w']) and (mode in ['r', 'r+'])):
                pass
            elif (mode in ['w']):
                if self.is_open:
                    raise PossibleDataLossError(f'Re-opening the file [{self._path}] with mode [{self._mode}] will delete the current file!')
            self._mode = mode
        if self.is_open:
            self.close()
        if (self._complevel and (self._complevel > 0)):
            self._filters = _tables().Filters(self._complevel, self._complib, fletcher32=self._fletcher32)
        if (_table_file_open_policy_is_strict and self.is_open):
            msg = 'Cannot open HDF5 file, which is already opened, even in read-only mode.'
            raise ValueError(msg)
        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def close(self):
        '\n        Close the PyTables file handle\n        '
        if (self._handle is not None):
            self._handle.close()
        self._handle = None

    @property
    def is_open(self):
        '\n        return a boolean indicating whether the file is open\n        '
        if (self._handle is None):
            return False
        return bool(self._handle.isopen)

    def flush(self, fsync=False):
        '\n        Force all buffered modifications to be written to disk.\n\n        Parameters\n        ----------\n        fsync : bool (default False)\n          call ``os.fsync()`` on the file handle to force writing to disk.\n\n        Notes\n        -----\n        Without ``fsync=True``, flushing may not guarantee that the OS writes\n        to disk. With fsync, the operation will block until the OS claims the\n        file has been written; however, other caching layers may still\n        interfere.\n        '
        if (self._handle is not None):
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def get(self, key):
        '\n        Retrieve pandas object stored in file.\n\n        Parameters\n        ----------\n        key : str\n\n        Returns\n        -------\n        object\n            Same type as object stored in file.\n        '
        with patch_pickle():
            group = self.get_node(key)
            if (group is None):
                raise KeyError(f'No object named {key} in the file')
            return self._read_group(group)

    def select(self, key, where=None, start=None, stop=None, columns=None, iterator=False, chunksize=None, auto_close=False):
        '\n        Retrieve pandas object stored in file, optionally based on where criteria.\n\n        .. warning::\n\n           Pandas uses PyTables for reading and writing HDF5 files, which allows\n           serializing object-dtype data with pickle when using the "fixed" format.\n           Loading pickled data received from untrusted sources can be unsafe.\n\n           See: https://docs.python.org/3/library/pickle.html for more.\n\n        Parameters\n        ----------\n        key : str\n            Object being retrieved from file.\n        where : list or None\n            List of Term (or convertible) objects, optional.\n        start : int or None\n            Row number to start selection.\n        stop : int, default None\n            Row number to stop selection.\n        columns : list or None\n            A list of columns that if not None, will limit the return columns.\n        iterator : bool or False\n            Returns an iterator.\n        chunksize : int or None\n            Number or rows to include in iteration, return an iterator.\n        auto_close : bool or False\n            Should automatically close the store when finished.\n\n        Returns\n        -------\n        object\n            Retrieved object from file.\n        '
        group = self.get_node(key)
        if (group is None):
            raise KeyError(f'No object named {key} in the file')
        where = _ensure_term(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        def func(_start, _stop, _where):
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)
        it = TableIterator(self, s, func, where=where, nrows=s.nrows, start=start, stop=stop, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
        return it.get_result()

    def select_as_coordinates(self, key, where=None, start=None, stop=None):
        '\n        return the selection as an Index\n\n        .. warning::\n\n           Pandas uses PyTables for reading and writing HDF5 files, which allows\n           serializing object-dtype data with pickle when using the "fixed" format.\n           Loading pickled data received from untrusted sources can be unsafe.\n\n           See: https://docs.python.org/3/library/pickle.html for more.\n\n\n        Parameters\n        ----------\n        key : str\n        where : list of Term (or convertible) objects, optional\n        start : integer (defaults to None), row number to start selection\n        stop  : integer (defaults to None), row number to stop selection\n        '
        where = _ensure_term(where, scope_level=1)
        tbl = self.get_storer(key)
        if (not isinstance(tbl, Table)):
            raise TypeError('can only read_coordinates with a table')
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def select_column(self, key, column, start=None, stop=None):
        '\n        return a single column from the table. This is generally only useful to\n        select an indexable\n\n        .. warning::\n\n           Pandas uses PyTables for reading and writing HDF5 files, which allows\n           serializing object-dtype data with pickle when using the "fixed" format.\n           Loading pickled data received from untrusted sources can be unsafe.\n\n           See: https://docs.python.org/3/library/pickle.html for more.\n\n        Parameters\n        ----------\n        key : str\n        column : str\n            The column of interest.\n        start : int or None, default None\n        stop : int or None, default None\n\n        Raises\n        ------\n        raises KeyError if the column is not found (or key is not a valid\n            store)\n        raises ValueError if the column can not be extracted individually (it\n            is part of a data block)\n\n        '
        tbl = self.get_storer(key)
        if (not isinstance(tbl, Table)):
            raise TypeError('can only read_column with a table')
        return tbl.read_column(column=column, start=start, stop=stop)

    def select_as_multiple(self, keys, where=None, selector=None, columns=None, start=None, stop=None, iterator=False, chunksize=None, auto_close=False):
        '\n        Retrieve pandas objects from multiple tables.\n\n        .. warning::\n\n           Pandas uses PyTables for reading and writing HDF5 files, which allows\n           serializing object-dtype data with pickle when using the "fixed" format.\n           Loading pickled data received from untrusted sources can be unsafe.\n\n           See: https://docs.python.org/3/library/pickle.html for more.\n\n        Parameters\n        ----------\n        keys : a list of the tables\n        selector : the table to apply the where criteria (defaults to keys[0]\n            if not supplied)\n        columns : the columns I want back\n        start : integer (defaults to None), row number to start selection\n        stop  : integer (defaults to None), row number to stop selection\n        iterator : boolean, return an iterator, default False\n        chunksize : nrows to include in iteration, return an iterator\n        auto_close : bool, default False\n            Should automatically close the store when finished.\n\n        Raises\n        ------\n        raises KeyError if keys or selector is not found or keys is empty\n        raises TypeError if keys is not a list or tuple\n        raises ValueError if the tables are not ALL THE SAME DIMENSIONS\n        '
        where = _ensure_term(where, scope_level=1)
        if (isinstance(keys, (list, tuple)) and (len(keys) == 1)):
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(key=keys, where=where, columns=columns, start=start, stop=stop, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
        if (not isinstance(keys, (list, tuple))):
            raise TypeError('keys must be a list/tuple')
        if (not len(keys)):
            raise ValueError('keys must have a non-zero length')
        if (selector is None):
            selector = keys[0]
        tbls = [self.get_storer(k) for k in keys]
        s = self.get_storer(selector)
        nrows = None
        for (t, k) in itertools.chain([(s, selector)], zip(tbls, keys)):
            if (t is None):
                raise KeyError(f'Invalid table [{k}]')
            if (not t.is_table):
                raise TypeError(f'object [{t.pathname}] is not a table, and cannot be used in all select as multiple')
            if (nrows is None):
                nrows = t.nrows
            elif (t.nrows != nrows):
                raise ValueError('all tables must have exactly the same nrows!')
        _tbls = [x for x in tbls if isinstance(x, Table)]
        axis = list({t.non_index_axes[0][0] for t in _tbls})[0]

        def func(_start, _stop, _where):
            objs = [t.read(where=_where, columns=columns, start=_start, stop=_stop) for t in tbls]
            return concat(objs, axis=axis, verify_integrity=False)._consolidate()
        it = TableIterator(self, s, func, where=where, nrows=nrows, start=start, stop=stop, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
        return it.get_result(coordinates=True)

    def put(self, key, value, format=None, index=True, append=False, complib=None, complevel=None, min_itemsize=None, nan_rep=None, data_columns=None, encoding=None, errors='strict', track_times=True, dropna=False):
        "\n        Store object in HDFStore.\n\n        Parameters\n        ----------\n        key : str\n        value : {Series, DataFrame}\n        format : 'fixed(f)|table(t)', default is 'fixed'\n            Format to use when storing object in HDFStore. Value can be one of:\n\n            ``'fixed'``\n                Fixed format.  Fast writing/reading. Not-appendable, nor searchable.\n            ``'table'``\n                Table format.  Write as a PyTables Table structure which may perform\n                worse but allow more flexible operations like searching / selecting\n                subsets of the data.\n        append : bool, default False\n            This will force Table format, append the input data to the existing.\n        data_columns : list, default None\n            List of columns to create as data columns, or True to use all columns.\n            See `here\n            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.\n        encoding : str, default None\n            Provide an encoding for strings.\n        track_times : bool, default True\n            Parameter is propagated to 'create_table' method of 'PyTables'.\n            If set to False it enables to have the same h5 files (same hashes)\n            independent on creation time.\n\n            .. versionadded:: 1.1.0\n        "
        if (format is None):
            format = (get_option('io.hdf.default_format') or 'fixed')
        format = self._validate_format(format)
        self._write_to_group(key, value, format=format, index=index, append=append, complib=complib, complevel=complevel, min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=data_columns, encoding=encoding, errors=errors, track_times=track_times, dropna=dropna)

    def remove(self, key, where=None, start=None, stop=None):
        '\n        Remove pandas object partially by specifying the where condition\n\n        Parameters\n        ----------\n        key : string\n            Node to remove or delete rows from\n        where : list of Term (or convertible) objects, optional\n        start : integer (defaults to None), row number to start selection\n        stop  : integer (defaults to None), row number to stop selection\n\n        Returns\n        -------\n        number of rows removed (or None if not a Table)\n\n        Raises\n        ------\n        raises KeyError if key is not a valid store\n\n        '
        where = _ensure_term(where, scope_level=1)
        try:
            s = self.get_storer(key)
        except KeyError:
            raise
        except AssertionError:
            raise
        except Exception as err:
            if (where is not None):
                raise ValueError('trying to remove a node with a non-None where clause!') from err
            node = self.get_node(key)
            if (node is not None):
                node._f_remove(recursive=True)
                return None
        if com.all_none(where, start, stop):
            s.group._f_remove(recursive=True)
        else:
            if (not s.is_table):
                raise ValueError('can only remove with where on objects written as tables')
            return s.delete(where=where, start=start, stop=stop)

    def append(self, key, value, format=None, axes=None, index=True, append=True, complib=None, complevel=None, columns=None, min_itemsize=None, nan_rep=None, chunksize=None, expectedrows=None, dropna=None, data_columns=None, encoding=None, errors='strict'):
        "\n        Append to Table in file. Node must already exist and be Table\n        format.\n\n        Parameters\n        ----------\n        key : str\n        value : {Series, DataFrame}\n        format : 'table' is the default\n            Format to use when storing object in HDFStore.  Value can be one of:\n\n            ``'table'``\n                Table format. Write as a PyTables Table structure which may perform\n                worse but allow more flexible operations like searching / selecting\n                subsets of the data.\n        append       : bool, default True\n            Append the input data to the existing.\n        data_columns : list of columns, or True, default None\n            List of columns to create as indexed data columns for on-disk\n            queries, or True to use all columns. By default only the axes\n            of the object are indexed. See `here\n            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.\n        min_itemsize : dict of columns that specify minimum str sizes\n        nan_rep      : str to use as str nan representation\n        chunksize    : size to chunk the writing\n        expectedrows : expected TOTAL row size of this table\n        encoding     : default None, provide an encoding for str\n        dropna : bool, default False\n            Do not write an ALL nan row to the store settable\n            by the option 'io.hdf.dropna_table'.\n\n        Notes\n        -----\n        Does *not* check if data being appended overlaps with existing\n        data in the table, so be careful\n        "
        if (columns is not None):
            raise TypeError('columns is not a supported keyword in append, try data_columns')
        if (dropna is None):
            dropna = get_option('io.hdf.dropna_table')
        if (format is None):
            format = (get_option('io.hdf.default_format') or 'table')
        format = self._validate_format(format)
        self._write_to_group(key, value, format=format, axes=axes, index=index, append=append, complib=complib, complevel=complevel, min_itemsize=min_itemsize, nan_rep=nan_rep, chunksize=chunksize, expectedrows=expectedrows, dropna=dropna, data_columns=data_columns, encoding=encoding, errors=errors)

    def append_to_multiple(self, d, value, selector, data_columns=None, axes=None, dropna=False, **kwargs):
        '\n        Append to multiple tables\n\n        Parameters\n        ----------\n        d : a dict of table_name to table_columns, None is acceptable as the\n            values of one node (this will get all the remaining columns)\n        value : a pandas object\n        selector : a string that designates the indexable table; all of its\n            columns will be designed as data_columns, unless data_columns is\n            passed, in which case these are used\n        data_columns : list of columns to create as data columns, or True to\n            use all columns\n        dropna : if evaluates to True, drop rows from all tables if any single\n                 row in each table has all NaN. Default False.\n\n        Notes\n        -----\n        axes parameter is currently not accepted\n\n        '
        if (axes is not None):
            raise TypeError('axes is currently not accepted as a parameter to append_to_multiple; you can create the tables independently instead')
        if (not isinstance(d, dict)):
            raise ValueError('append_to_multiple must have a dictionary specified as the way to split the value')
        if (selector not in d):
            raise ValueError('append_to_multiple requires a selector that is in passed dict')
        axis = list((set(range(value.ndim)) - set(_AXES_MAP[type(value)])))[0]
        remain_key = None
        remain_values: List = []
        for (k, v) in d.items():
            if (v is None):
                if (remain_key is not None):
                    raise ValueError('append_to_multiple can only have one value in d that is None')
                remain_key = k
            else:
                remain_values.extend(v)
        if (remain_key is not None):
            ordered = value.axes[axis]
            ordd = ordered.difference(Index(remain_values))
            ordd = sorted(ordered.get_indexer(ordd))
            d[remain_key] = ordered.take(ordd)
        if (data_columns is None):
            data_columns = d[selector]
        if dropna:
            idxs = (value[cols].dropna(how='all').index for cols in d.values())
            valid_index = next(idxs)
            for index in idxs:
                valid_index = valid_index.intersection(index)
            value = value.loc[valid_index]
        min_itemsize = kwargs.pop('min_itemsize', None)
        for (k, v) in d.items():
            dc = (data_columns if (k == selector) else None)
            val = value.reindex(v, axis=axis)
            filtered = ({key: value for (key, value) in min_itemsize.items() if (key in v)} if (min_itemsize is not None) else None)
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)

    def create_table_index(self, key, columns=None, optlevel=None, kind=None):
        '\n        Create a pytables index on the table.\n\n        Parameters\n        ----------\n        key : str\n        columns : None, bool, or listlike[str]\n            Indicate which columns to create an index on.\n\n            * False : Do not create any indexes.\n            * True : Create indexes on all columns.\n            * None : Create indexes on all columns.\n            * listlike : Create indexes on the given columns.\n\n        optlevel : int or None, default None\n            Optimization level, if None, pytables defaults to 6.\n        kind : str or None, default None\n            Kind of index, if None, pytables defaults to "medium".\n\n        Raises\n        ------\n        TypeError: raises if the node is not a table\n        '
        _tables()
        s = self.get_storer(key)
        if (s is None):
            return
        if (not isinstance(s, Table)):
            raise TypeError('cannot create table index on a Fixed format store')
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def groups(self):
        '\n        Return a list of all the top-level nodes.\n\n        Each node returned is not a pandas storage object.\n\n        Returns\n        -------\n        list\n            List of objects.\n        '
        _tables()
        self._check_if_open()
        assert (self._handle is not None)
        assert (_table_mod is not None)
        return [g for g in self._handle.walk_groups() if ((not isinstance(g, _table_mod.link.Link)) and (getattr(g._v_attrs, 'pandas_type', None) or getattr(g, 'table', None) or (isinstance(g, _table_mod.table.Table) and (g._v_name != 'table'))))]

    def walk(self, where='/'):
        '\n        Walk the pytables group hierarchy for pandas objects.\n\n        This generator will yield the group path, subgroups and pandas object\n        names for each group.\n\n        Any non-pandas PyTables objects that are not a group will be ignored.\n\n        The `where` group itself is listed first (preorder), then each of its\n        child groups (following an alphanumerical order) is also traversed,\n        following the same procedure.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        where : str, default "/"\n            Group where to start walking.\n\n        Yields\n        ------\n        path : str\n            Full path to a group (without trailing \'/\').\n        groups : list\n            Names (strings) of the groups contained in `path`.\n        leaves : list\n            Names (strings) of the pandas objects contained in `path`.\n        '
        _tables()
        self._check_if_open()
        assert (self._handle is not None)
        assert (_table_mod is not None)
        for g in self._handle.walk_groups(where):
            if (getattr(g._v_attrs, 'pandas_type', None) is not None):
                continue
            groups = []
            leaves = []
            for child in g._v_children.values():
                pandas_type = getattr(child._v_attrs, 'pandas_type', None)
                if (pandas_type is None):
                    if isinstance(child, _table_mod.group.Group):
                        groups.append(child._v_name)
                else:
                    leaves.append(child._v_name)
            (yield (g._v_pathname.rstrip('/'), groups, leaves))

    def get_node(self, key):
        ' return the node with the key or None if it does not exist '
        self._check_if_open()
        if (not key.startswith('/')):
            key = ('/' + key)
        assert (self._handle is not None)
        assert (_table_mod is not None)
        try:
            node = self._handle.get_node(self.root, key)
        except _table_mod.exceptions.NoSuchNodeError:
            return None
        assert isinstance(node, _table_mod.Node), type(node)
        return node

    def get_storer(self, key):
        ' return the storer object for a key, raise if not in the file '
        group = self.get_node(key)
        if (group is None):
            raise KeyError(f'No object named {key} in the file')
        s = self._create_storer(group)
        s.infer_axes()
        return s

    def copy(self, file, mode='w', propindexes=True, keys=None, complib=None, complevel=None, fletcher32=False, overwrite=True):
        '\n        Copy the existing store to a new file, updating in place.\n\n        Parameters\n        ----------\n        propindexes : bool, default True\n            Restore indexes in copied file.\n        keys : list, optional\n            List of keys to include in the copy (defaults to all).\n        overwrite : bool, default True\n            Whether to overwrite (remove and replace) existing nodes in the new store.\n        mode, complib, complevel, fletcher32 same as in HDFStore.__init__\n\n        Returns\n        -------\n        open file handle of the new store\n        '
        new_store = HDFStore(file, mode=mode, complib=complib, complevel=complevel, fletcher32=fletcher32)
        if (keys is None):
            keys = list(self.keys())
        if (not isinstance(keys, (tuple, list))):
            keys = [keys]
        for k in keys:
            s = self.get_storer(k)
            if (s is not None):
                if (k in new_store):
                    if overwrite:
                        new_store.remove(k)
                data = self.select(k)
                if isinstance(s, Table):
                    index: Union[(bool, List[str])] = False
                    if propindexes:
                        index = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(k, data, index=index, data_columns=getattr(s, 'data_columns', None), encoding=s.encoding)
                else:
                    new_store.put(k, data, encoding=s.encoding)
        return new_store

    def info(self):
        '\n        Print detailed information on the store.\n\n        Returns\n        -------\n        str\n        '
        path = pprint_thing(self._path)
        output = f'''{type(self)}
File path: {path}
'''
        if self.is_open:
            lkeys = sorted(self.keys())
            if len(lkeys):
                keys = []
                values = []
                for k in lkeys:
                    try:
                        s = self.get_storer(k)
                        if (s is not None):
                            keys.append(pprint_thing((s.pathname or k)))
                            values.append(pprint_thing((s or 'invalid_HDFStore node')))
                    except AssertionError:
                        raise
                    except Exception as detail:
                        keys.append(k)
                        dstr = pprint_thing(detail)
                        values.append(f'[invalid_HDFStore node: {dstr}]')
                output += adjoin(12, keys, values)
            else:
                output += 'Empty'
        else:
            output += 'File is CLOSED'
        return output

    def _check_if_open(self):
        if (not self.is_open):
            raise ClosedFileError(f'{self._path} file is not open!')

    def _validate_format(self, format):
        ' validate / deprecate formats '
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f'invalid HDFStore format specified [{format}]') from err
        return format

    def _create_storer(self, group, format=None, value=None, encoding='UTF-8', errors='strict'):
        ' return a suitable class to operate '
        cls: Union[(Type['GenericFixed'], Type['Table'])]
        if ((value is not None) and (not isinstance(value, (Series, DataFrame)))):
            raise TypeError('value must be None, Series, or DataFrame')

        def error(t):
            return TypeError(f'cannot properly create the storer for: [{t}] [group->{group},value->{type(value)},format->{format}')
        pt = _ensure_decoded(getattr(group._v_attrs, 'pandas_type', None))
        tt = _ensure_decoded(getattr(group._v_attrs, 'table_type', None))
        if (pt is None):
            if (value is None):
                _tables()
                assert (_table_mod is not None)
                if (getattr(group, 'table', None) or isinstance(group, _table_mod.table.Table)):
                    pt = 'frame_table'
                    tt = 'generic_table'
                else:
                    raise TypeError('cannot create a storer if the object is not existing nor a value are passed')
            else:
                if isinstance(value, Series):
                    pt = 'series'
                else:
                    pt = 'frame'
                if (format == 'table'):
                    pt += '_table'
        if ('table' not in pt):
            _STORER_MAP = {'series': SeriesFixed, 'frame': FrameFixed}
            try:
                cls = _STORER_MAP[pt]
            except KeyError as err:
                raise error('_STORER_MAP') from err
            return cls(self, group, encoding=encoding, errors=errors)
        if (tt is None):
            if (value is not None):
                if (pt == 'series_table'):
                    index = getattr(value, 'index', None)
                    if (index is not None):
                        if (index.nlevels == 1):
                            tt = 'appendable_series'
                        elif (index.nlevels > 1):
                            tt = 'appendable_multiseries'
                elif (pt == 'frame_table'):
                    index = getattr(value, 'index', None)
                    if (index is not None):
                        if (index.nlevels == 1):
                            tt = 'appendable_frame'
                        elif (index.nlevels > 1):
                            tt = 'appendable_multiframe'
        _TABLE_MAP = {'generic_table': GenericTable, 'appendable_series': AppendableSeriesTable, 'appendable_multiseries': AppendableMultiSeriesTable, 'appendable_frame': AppendableFrameTable, 'appendable_multiframe': AppendableMultiFrameTable, 'worm': WORMTable}
        try:
            cls = _TABLE_MAP[tt]
        except KeyError as err:
            raise error('_TABLE_MAP') from err
        return cls(self, group, encoding=encoding, errors=errors)

    def _write_to_group(self, key, value, format, axes=None, index=True, append=False, complib=None, complevel=None, fletcher32=None, min_itemsize=None, chunksize=None, expectedrows=None, dropna=False, nan_rep=None, data_columns=None, encoding=None, errors='strict', track_times=True):
        if (getattr(value, 'empty', None) and ((format == 'table') or append)):
            return
        group = self._identify_group(key, append)
        s = self._create_storer(group, format, value, encoding=encoding, errors=errors)
        if append:
            if ((not s.is_table) or (s.is_table and (format == 'fixed') and s.is_exists)):
                raise ValueError('Can only append to Tables')
            if (not s.is_exists):
                s.set_object_info()
        else:
            s.set_object_info()
        if ((not s.is_table) and complib):
            raise ValueError('Compression not supported on Fixed format stores')
        s.write(obj=value, axes=axes, append=append, complib=complib, complevel=complevel, fletcher32=fletcher32, min_itemsize=min_itemsize, chunksize=chunksize, expectedrows=expectedrows, dropna=dropna, nan_rep=nan_rep, data_columns=data_columns, track_times=track_times)
        if (isinstance(s, Table) and index):
            s.create_index(columns=index)

    def _read_group(self, group):
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def _identify_group(self, key, append):
        'Identify HDF5 group based on key, delete/create group if needed.'
        group = self.get_node(key)
        assert (self._handle is not None)
        if ((group is not None) and (not append)):
            self._handle.remove_node(group, recursive=True)
            group = None
        if (group is None):
            group = self._create_nodes_and_group(key)
        return group

    def _create_nodes_and_group(self, key):
        'Create nodes from key and return group name.'
        assert (self._handle is not None)
        paths = key.split('/')
        path = '/'
        for p in paths:
            if (not len(p)):
                continue
            new_path = path
            if (not path.endswith('/')):
                new_path += '/'
            new_path += p
            group = self.get_node(new_path)
            if (group is None):
                group = self._handle.create_group(path, p)
            path = new_path
        return group

class TableIterator():
    '\n    Define the iteration interface on a table\n\n    Parameters\n    ----------\n    store : HDFStore\n    s     : the referred storer\n    func  : the function to execute the query\n    where : the where of the query\n    nrows : the rows to iterate on\n    start : the passed start value (default is None)\n    stop  : the passed stop value (default is None)\n    iterator : bool, default False\n        Whether to use the default iterator.\n    chunksize : the passed chunking value (default is 100000)\n    auto_close : bool, default False\n        Whether to automatically close the store at the end of iteration.\n    '

    def __init__(self, store, s, func, where, nrows, start=None, stop=None, iterator=False, chunksize=None, auto_close=False):
        self.store = store
        self.s = s
        self.func = func
        self.where = where
        if self.s.is_table:
            if (nrows is None):
                nrows = 0
            if (start is None):
                start = 0
            if (stop is None):
                stop = nrows
            stop = min(nrows, stop)
        self.nrows = nrows
        self.start = start
        self.stop = stop
        self.coordinates = None
        if (iterator or (chunksize is not None)):
            if (chunksize is None):
                chunksize = 100000
            self.chunksize = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close = auto_close

    def __iter__(self):
        current = self.start
        if (self.coordinates is None):
            raise ValueError('Cannot iterate until get_result is called.')
        while (current < self.stop):
            stop = min((current + self.chunksize), self.stop)
            value = self.func(None, None, self.coordinates[current:stop])
            current = stop
            if ((value is None) or (not len(value))):
                continue
            (yield value)
        self.close()

    def close(self):
        if self.auto_close:
            self.store.close()

    def get_result(self, coordinates=False):
        if (self.chunksize is not None):
            if (not isinstance(self.s, Table)):
                raise TypeError('can only use an iterator or chunksize on a table')
            self.coordinates = self.s.read_coordinates(where=self.where)
            return self
        if coordinates:
            if (not isinstance(self.s, Table)):
                raise TypeError('can only read_coordinates on a table')
            where = self.s.read_coordinates(where=self.where, start=self.start, stop=self.stop)
        else:
            where = self.where
        results = self.func(self.start, self.stop, where)
        self.close()
        return results

class IndexCol():
    '\n    an index column description class\n\n    Parameters\n    ----------\n    axis   : axis which I reference\n    values : the ndarray like converted values\n    kind   : a string description of this type\n    typ    : the pytables type\n    pos    : the position in the pytables\n\n    '
    is_an_indexable = True
    is_data_indexable = True
    _info_fields = ['freq', 'tz', 'index_name']

    def __init__(self, name, values=None, kind=None, typ=None, cname=None, axis=None, pos=None, freq=None, tz=None, index_name=None, ordered=None, table=None, meta=None, metadata=None):
        if (not isinstance(name, str)):
            raise ValueError('`name` must be a str.')
        self.values = values
        self.kind = kind
        self.typ = typ
        self.name = name
        self.cname = (cname or name)
        self.axis = axis
        self.pos = pos
        self.freq = freq
        self.tz = tz
        self.index_name = index_name
        self.ordered = ordered
        self.table = table
        self.meta = meta
        self.metadata = metadata
        if (pos is not None):
            self.set_pos(pos)
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def itemsize(self):
        return self.typ.itemsize

    @property
    def kind_attr(self):
        return f'{self.name}_kind'

    def set_pos(self, pos):
        ' set the position of this column in the Table '
        self.pos = pos
        if ((pos is not None) and (self.typ is not None)):
            self.typ._v_pos = pos

    def __repr__(self):
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.axis, self.pos, self.kind)))
        return ','.join((f'{key}->{value}' for (key, value) in zip(['name', 'cname', 'axis', 'pos', 'kind'], temp)))

    def __eq__(self, other):
        ' compare 2 col items '
        return all(((getattr(self, a, None) == getattr(other, a, None)) for a in ['name', 'cname', 'axis', 'pos']))

    def __ne__(self, other):
        return (not self.__eq__(other))

    @property
    def is_indexed(self):
        ' return whether I am an indexed column '
        if (not hasattr(self.table, 'cols')):
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(self, values, nan_rep, encoding, errors):
        '\n        Convert the data from this selection to the appropriate pandas type.\n        '
        assert isinstance(values, np.ndarray), type(values)
        if (values.dtype.fields is not None):
            values = values[self.cname]
        val_kind = _ensure_decoded(self.kind)
        values = _maybe_convert(values, val_kind, encoding, errors)
        kwargs = {}
        kwargs['name'] = _ensure_decoded(self.index_name)
        if (self.freq is not None):
            kwargs['freq'] = _ensure_decoded(self.freq)
        factory: Union[(Type[Index], Type[DatetimeIndex])] = Index
        if (is_datetime64_dtype(values.dtype) or is_datetime64tz_dtype(values.dtype)):
            factory = DatetimeIndex
        try:
            new_pd_index = factory(values, **kwargs)
        except ValueError:
            if ('freq' in kwargs):
                kwargs['freq'] = None
            new_pd_index = factory(values, **kwargs)
        new_pd_index = _set_tz(new_pd_index, self.tz)
        return (new_pd_index, new_pd_index)

    def take_data(self):
        ' return the values'
        return self.values

    @property
    def attrs(self):
        return self.table._v_attrs

    @property
    def description(self):
        return self.table.description

    @property
    def col(self):
        ' return my current col description '
        return getattr(self.description, self.cname, None)

    @property
    def cvalues(self):
        ' return my cython values '
        return self.values

    def __iter__(self):
        return iter(self.values)

    def maybe_set_size(self, min_itemsize=None):
        '\n        maybe set a string col itemsize:\n            min_itemsize can be an integer or a dict with this columns name\n            with an integer size\n        '
        if (_ensure_decoded(self.kind) == 'string'):
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)
            if ((min_itemsize is not None) and (self.typ.itemsize < min_itemsize)):
                self.typ = _tables().StringCol(itemsize=min_itemsize, pos=self.pos)

    def validate_names(self):
        pass

    def validate_and_set(self, handler, append):
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def validate_col(self, itemsize=None):
        ' validate this column: return the compared against itemsize '
        if (_ensure_decoded(self.kind) == 'string'):
            c = self.col
            if (c is not None):
                if (itemsize is None):
                    itemsize = self.itemsize
                if (c.itemsize < itemsize):
                    raise ValueError(f'''Trying to store a string with len [{itemsize}] in [{self.cname}] column but
this column has a limit of [{c.itemsize}]!
Consider using min_itemsize to preset the sizes on these columns''')
                return c.itemsize
        return None

    def validate_attr(self, append):
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if ((existing_kind is not None) and (existing_kind != self.kind)):
                raise TypeError(f'incompatible kind in col [{existing_kind} - {self.kind}]')

    def update_info(self, info):
        '\n        set/update the info for this indexable with the key/value\n        if there is a conflict raise/warn as needed\n        '
        for key in self._info_fields:
            value = getattr(self, key, None)
            idx = info.setdefault(self.name, {})
            existing_value = idx.get(key)
            if ((key in idx) and (value is not None) and (existing_value != value)):
                if (key in ['freq', 'index_name']):
                    ws = (attribute_conflict_doc % (key, existing_value, value))
                    warnings.warn(ws, AttributeConflictWarning, stacklevel=6)
                    idx[key] = None
                    setattr(self, key, None)
                else:
                    raise ValueError(f'invalid info for [{self.name}] for [{key}], existing_value [{existing_value}] conflicts with new value [{value}]')
            elif ((value is not None) or (existing_value is not None)):
                idx[key] = value

    def set_info(self, info):
        ' set my state from the passed info '
        idx = info.get(self.name)
        if (idx is not None):
            self.__dict__.update(idx)

    def set_attr(self):
        ' set the kind for this column '
        setattr(self.attrs, self.kind_attr, self.kind)

    def validate_metadata(self, handler):
        ' validate that kind=category does not change the categories '
        if (self.meta == 'category'):
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if ((new_metadata is not None) and (cur_metadata is not None) and (not array_equivalent(new_metadata, cur_metadata))):
                raise ValueError('cannot append a categorical with different categories to the existing')

    def write_metadata(self, handler):
        ' set the meta data '
        if (self.metadata is not None):
            handler.write_metadata(self.cname, self.metadata)

class GenericIndexCol(IndexCol):
    ' an index which is not represented in the data of the table '

    @property
    def is_indexed(self):
        return False

    def convert(self, values, nan_rep, encoding, errors):
        '\n        Convert the data from this selection to the appropriate pandas type.\n\n        Parameters\n        ----------\n        values : np.ndarray\n        nan_rep : str\n        encoding : str\n        errors : str\n        '
        assert isinstance(values, np.ndarray), type(values)
        values = Int64Index(np.arange(len(values)))
        return (values, values)

    def set_attr(self):
        pass

class DataCol(IndexCol):
    '\n    a data holding column, by definition this is not indexable\n\n    Parameters\n    ----------\n    data   : the actual data\n    cname  : the column name in the table to hold the data (typically\n                values)\n    meta   : a string description of the metadata\n    metadata : the actual metadata\n    '
    is_an_indexable = False
    is_data_indexable = False
    _info_fields = ['tz', 'ordered']

    def __init__(self, name, values=None, kind=None, typ=None, cname=None, pos=None, tz=None, ordered=None, table=None, meta=None, metadata=None, dtype=None, data=None):
        super().__init__(name=name, values=values, kind=kind, typ=typ, pos=pos, cname=cname, tz=tz, ordered=ordered, table=table, meta=meta, metadata=metadata)
        self.dtype = dtype
        self.data = data

    @property
    def dtype_attr(self):
        return f'{self.name}_dtype'

    @property
    def meta_attr(self):
        return f'{self.name}_meta'

    def __repr__(self):
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.dtype, self.kind, self.shape)))
        return ','.join((f'{key}->{value}' for (key, value) in zip(['name', 'cname', 'dtype', 'kind', 'shape'], temp)))

    def __eq__(self, other):
        ' compare 2 col items '
        return all(((getattr(self, a, None) == getattr(other, a, None)) for a in ['name', 'cname', 'dtype', 'pos']))

    def set_data(self, data):
        assert (data is not None)
        assert (self.dtype is None)
        (data, dtype_name) = _get_data_and_dtype_name(data)
        self.data = data
        self.dtype = dtype_name
        self.kind = _dtype_to_kind(dtype_name)

    def take_data(self):
        ' return the data '
        return self.data

    @classmethod
    def _get_atom(cls, values):
        '\n        Get an appropriately typed and shaped pytables.Col object for values.\n        '
        dtype = values.dtype
        itemsize = dtype.itemsize
        shape = values.shape
        if (values.ndim == 1):
            shape = (1, values.size)
        if isinstance(values, Categorical):
            codes = values.codes
            atom = cls.get_atom_data(shape, kind=codes.dtype.name)
        elif (is_datetime64_dtype(dtype) or is_datetime64tz_dtype(dtype)):
            atom = cls.get_atom_datetime64(shape)
        elif is_timedelta64_dtype(dtype):
            atom = cls.get_atom_timedelta64(shape)
        elif is_complex_dtype(dtype):
            atom = _tables().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif is_string_dtype(dtype):
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)
        return atom

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        return _tables().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def get_atom_coltype(cls, kind):
        ' return the PyTables column class for this column '
        if kind.startswith('uint'):
            k4 = kind[4:]
            col_name = f'UInt{k4}Col'
        elif kind.startswith('period'):
            col_name = 'Int64Col'
        else:
            kcap = kind.capitalize()
            col_name = f'{kcap}Col'
        return getattr(_tables(), col_name)

    @classmethod
    def get_atom_data(cls, shape, kind):
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def get_atom_datetime64(cls, shape):
        return _tables().Int64Col(shape=shape[0])

    @classmethod
    def get_atom_timedelta64(cls, shape):
        return _tables().Int64Col(shape=shape[0])

    @property
    def shape(self):
        return getattr(self.data, 'shape', None)

    @property
    def cvalues(self):
        ' return my cython values '
        return self.data

    def validate_attr(self, append):
        'validate that we have the same order as the existing & same dtype'
        if append:
            existing_fields = getattr(self.attrs, self.kind_attr, None)
            if ((existing_fields is not None) and (existing_fields != list(self.values))):
                raise ValueError('appended items do not match existing items in table!')
            existing_dtype = getattr(self.attrs, self.dtype_attr, None)
            if ((existing_dtype is not None) and (existing_dtype != self.dtype)):
                raise ValueError('appended items dtype do not match existing items dtype in table!')

    def convert(self, values, nan_rep, encoding, errors):
        '\n        Convert the data from this selection to the appropriate pandas type.\n\n        Parameters\n        ----------\n        values : np.ndarray\n        nan_rep :\n        encoding : str\n        errors : str\n\n        Returns\n        -------\n        index : listlike to become an Index\n        data : ndarraylike to become a column\n        '
        assert isinstance(values, np.ndarray), type(values)
        if (values.dtype.fields is not None):
            values = values[self.cname]
        assert (self.typ is not None)
        if (self.dtype is None):
            (converted, dtype_name) = _get_data_and_dtype_name(values)
            kind = _dtype_to_kind(dtype_name)
        else:
            converted = values
            dtype_name = self.dtype
            kind = self.kind
        assert isinstance(converted, np.ndarray)
        meta = _ensure_decoded(self.meta)
        metadata = self.metadata
        ordered = self.ordered
        tz = self.tz
        assert (dtype_name is not None)
        dtype = _ensure_decoded(dtype_name)
        if (dtype == 'datetime64'):
            converted = _set_tz(converted, tz, coerce=True)
        elif (dtype == 'timedelta64'):
            converted = np.asarray(converted, dtype='m8[ns]')
        elif (dtype == 'date'):
            try:
                converted = np.asarray([date.fromordinal(v) for v in converted], dtype=object)
            except ValueError:
                converted = np.asarray([date.fromtimestamp(v) for v in converted], dtype=object)
        elif (meta == 'category'):
            categories = metadata
            codes = converted.ravel()
            if (categories is None):
                categories = Index([], dtype=np.float64)
            else:
                mask = isna(categories)
                if mask.any():
                    categories = categories[(~ mask)]
                    codes[(codes != (- 1))] -= mask.astype(int).cumsum()._values
            converted = Categorical.from_codes(codes, categories=categories, ordered=ordered)
        else:
            try:
                converted = converted.astype(dtype, copy=False)
            except TypeError:
                converted = converted.astype('O', copy=False)
        if (_ensure_decoded(kind) == 'string'):
            converted = _unconvert_string_array(converted, nan_rep=nan_rep, encoding=encoding, errors=errors)
        return (self.values, converted)

    def set_attr(self):
        ' set the data for this column '
        setattr(self.attrs, self.kind_attr, self.values)
        setattr(self.attrs, self.meta_attr, self.meta)
        assert (self.dtype is not None)
        setattr(self.attrs, self.dtype_attr, self.dtype)

class DataIndexableCol(DataCol):
    ' represent a data column that can be indexed '
    is_data_indexable = True

    def validate_names(self):
        if (not Index(self.values).is_object()):
            raise ValueError('cannot have non-object label DataIndexableCol')

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        return _tables().StringCol(itemsize=itemsize)

    @classmethod
    def get_atom_data(cls, shape, kind):
        return cls.get_atom_coltype(kind=kind)()

    @classmethod
    def get_atom_datetime64(cls, shape):
        return _tables().Int64Col()

    @classmethod
    def get_atom_timedelta64(cls, shape):
        return _tables().Int64Col()

class GenericDataIndexableCol(DataIndexableCol):
    ' represent a generic pytables data column '
    pass

class Fixed():
    '\n    represent an object in my store\n    facilitate read/write of various types of objects\n    this is an abstract base class\n\n    Parameters\n    ----------\n    parent : HDFStore\n    group : Node\n        The group node where the table resides.\n    '
    format_type = 'fixed'
    is_table = False

    def __init__(self, parent, group, encoding='UTF-8', errors='strict'):
        assert isinstance(parent, HDFStore), type(parent)
        assert (_table_mod is not None)
        assert isinstance(group, _table_mod.Node), type(group)
        self.parent = parent
        self.group = group
        self.encoding = _ensure_encoding(encoding)
        self.errors = errors

    @property
    def is_old_version(self):
        return ((self.version[0] <= 0) and (self.version[1] <= 10) and (self.version[2] < 1))

    @property
    def version(self):
        ' compute and set our version '
        version = _ensure_decoded(getattr(self.group._v_attrs, 'pandas_version', None))
        try:
            version = tuple((int(x) for x in version.split('.')))
            if (len(version) == 2):
                version = (version + (0,))
        except AttributeError:
            version = (0, 0, 0)
        return version

    @property
    def pandas_type(self):
        return _ensure_decoded(getattr(self.group._v_attrs, 'pandas_type', None))

    def __repr__(self):
        ' return a pretty representation of myself '
        self.infer_axes()
        s = self.shape
        if (s is not None):
            if isinstance(s, (list, tuple)):
                jshape = ','.join((pprint_thing(x) for x in s))
                s = f'[{jshape}]'
            return f'{self.pandas_type:12.12} (shape->{s})'
        return self.pandas_type

    def set_object_info(self):
        ' set my pandas type & version '
        self.attrs.pandas_type = str(self.pandas_kind)
        self.attrs.pandas_version = str(_version)

    def copy(self):
        new_self = copy.copy(self)
        return new_self

    @property
    def shape(self):
        return self.nrows

    @property
    def pathname(self):
        return self.group._v_pathname

    @property
    def _handle(self):
        return self.parent._handle

    @property
    def _filters(self):
        return self.parent._filters

    @property
    def _complevel(self):
        return self.parent._complevel

    @property
    def _fletcher32(self):
        return self.parent._fletcher32

    @property
    def attrs(self):
        return self.group._v_attrs

    def set_attrs(self):
        ' set our object attributes '
        pass

    def get_attrs(self):
        ' get our object attributes '
        pass

    @property
    def storable(self):
        ' return my storable '
        return self.group

    @property
    def is_exists(self):
        return False

    @property
    def nrows(self):
        return getattr(self.storable, 'nrows', None)

    def validate(self, other):
        ' validate against an existing storable '
        if (other is None):
            return
        return True

    def validate_version(self, where=None):
        ' are we trying to operate on an old version? '
        return True

    def infer_axes(self):
        '\n        infer the axes of my storer\n        return a boolean indicating if we have a valid storer or not\n        '
        s = self.storable
        if (s is None):
            return False
        self.get_attrs()
        return True

    def read(self, where=None, columns=None, start=None, stop=None):
        raise NotImplementedError('cannot read on an abstract storer: subclasses should implement')

    def write(self, **kwargs):
        raise NotImplementedError('cannot write on an abstract storer: subclasses should implement')

    def delete(self, where=None, start=None, stop=None):
        '\n        support fully deleting the node in its entirety (only) - where\n        specification must be None\n        '
        if com.all_none(where, start, stop):
            self._handle.remove_node(self.group, recursive=True)
            return None
        raise TypeError('cannot delete on an abstract storer')

class GenericFixed(Fixed):
    ' a generified fixed version '
    _index_type_map = {DatetimeIndex: 'datetime', PeriodIndex: 'period'}
    _reverse_index_map = {v: k for (k, v) in _index_type_map.items()}
    attributes = []

    def _class_to_alias(self, cls):
        return self._index_type_map.get(cls, '')

    def _alias_to_class(self, alias):
        if isinstance(alias, type):
            return alias
        return self._reverse_index_map.get(alias, Index)

    def _get_index_factory(self, attrs):
        index_class = self._alias_to_class(_ensure_decoded(getattr(attrs, 'index_class', '')))
        factory: Callable
        if (index_class == DatetimeIndex):

            def f(values, freq=None, tz=None):
                dta = DatetimeArray._simple_new(values.values, freq=freq)
                result = DatetimeIndex._simple_new(dta, name=None)
                if (tz is not None):
                    result = result.tz_localize('UTC').tz_convert(tz)
                return result
            factory = f
        elif (index_class == PeriodIndex):

            def f(values, freq=None, tz=None):
                parr = PeriodArray._simple_new(values, freq=freq)
                return PeriodIndex._simple_new(parr, name=None)
            factory = f
        else:
            factory = index_class
        kwargs = {}
        if ('freq' in attrs):
            kwargs['freq'] = attrs['freq']
            if (index_class is Index):
                factory = TimedeltaIndex
        if ('tz' in attrs):
            if isinstance(attrs['tz'], bytes):
                kwargs['tz'] = attrs['tz'].decode('utf-8')
            else:
                kwargs['tz'] = attrs['tz']
            assert (index_class is DatetimeIndex)
        return (factory, kwargs)

    def validate_read(self, columns, where):
        '\n        raise if any keywords are passed which are not-None\n        '
        if (columns is not None):
            raise TypeError('cannot pass a column specification when reading a Fixed format store. this store must be selected in its entirety')
        if (where is not None):
            raise TypeError('cannot pass a where specification when reading from a Fixed format store. this store must be selected in its entirety')

    @property
    def is_exists(self):
        return True

    def set_attrs(self):
        ' set our object attributes '
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors

    def get_attrs(self):
        ' retrieve our attributes '
        self.encoding = _ensure_encoding(getattr(self.attrs, 'encoding', None))
        self.errors = _ensure_decoded(getattr(self.attrs, 'errors', 'strict'))
        for n in self.attributes:
            setattr(self, n, _ensure_decoded(getattr(self.attrs, n, None)))

    def write(self, obj, **kwargs):
        self.set_attrs()

    def read_array(self, key, start=None, stop=None):
        ' read an array for the specified node (off of group '
        import tables
        node = getattr(self.group, key)
        attrs = node._v_attrs
        transposed = getattr(attrs, 'transposed', False)
        if isinstance(node, tables.VLArray):
            ret = node[0][start:stop]
        else:
            dtype = _ensure_decoded(getattr(attrs, 'value_type', None))
            shape = getattr(attrs, 'shape', None)
            if (shape is not None):
                ret = np.empty(shape, dtype=dtype)
            else:
                ret = node[start:stop]
            if (dtype == 'datetime64'):
                tz = getattr(attrs, 'tz', None)
                ret = _set_tz(ret, tz, coerce=True)
            elif (dtype == 'timedelta64'):
                ret = np.asarray(ret, dtype='m8[ns]')
        if transposed:
            return ret.T
        else:
            return ret

    def read_index(self, key, start=None, stop=None):
        variety = _ensure_decoded(getattr(self.attrs, f'{key}_variety'))
        if (variety == 'multi'):
            return self.read_multi_index(key, start=start, stop=stop)
        elif (variety == 'regular'):
            node = getattr(self.group, key)
            index = self.read_index_node(node, start=start, stop=stop)
            return index
        else:
            raise TypeError(f'unrecognized index variety: {variety}')

    def write_index(self, key, index):
        if isinstance(index, MultiIndex):
            setattr(self.attrs, f'{key}_variety', 'multi')
            self.write_multi_index(key, index)
        else:
            setattr(self.attrs, f'{key}_variety', 'regular')
            converted = _convert_index('index', index, self.encoding, self.errors)
            self.write_array(key, converted.values)
            node = getattr(self.group, key)
            node._v_attrs.kind = converted.kind
            node._v_attrs.name = index.name
            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                node._v_attrs.index_class = self._class_to_alias(type(index))
            if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
                node._v_attrs.freq = index.freq
            if (isinstance(index, DatetimeIndex) and (index.tz is not None)):
                node._v_attrs.tz = _get_tz(index.tz)

    def write_multi_index(self, key, index):
        setattr(self.attrs, f'{key}_nlevels', index.nlevels)
        for (i, (lev, level_codes, name)) in enumerate(zip(index.levels, index.codes, index.names)):
            if is_extension_array_dtype(lev):
                raise NotImplementedError('Saving a MultiIndex with an extension dtype is not supported.')
            level_key = f'{key}_level{i}'
            conv_level = _convert_index(level_key, lev, self.encoding, self.errors)
            self.write_array(level_key, conv_level.values)
            node = getattr(self.group, level_key)
            node._v_attrs.kind = conv_level.kind
            node._v_attrs.name = name
            setattr(node._v_attrs, f'{key}_name{name}', name)
            label_key = f'{key}_label{i}'
            self.write_array(label_key, level_codes)

    def read_multi_index(self, key, start=None, stop=None):
        nlevels = getattr(self.attrs, f'{key}_nlevels')
        levels = []
        codes = []
        names: List[Label] = []
        for i in range(nlevels):
            level_key = f'{key}_level{i}'
            node = getattr(self.group, level_key)
            lev = self.read_index_node(node, start=start, stop=stop)
            levels.append(lev)
            names.append(lev.name)
            label_key = f'{key}_label{i}'
            level_codes = self.read_array(label_key, start=start, stop=stop)
            codes.append(level_codes)
        return MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=True)

    def read_index_node(self, node, start=None, stop=None):
        data = node[start:stop]
        if (('shape' in node._v_attrs) and (np.prod(node._v_attrs.shape) == 0)):
            data = np.empty(node._v_attrs.shape, dtype=node._v_attrs.value_type)
        kind = _ensure_decoded(node._v_attrs.kind)
        name = None
        if ('name' in node._v_attrs):
            name = _ensure_str(node._v_attrs.name)
            name = _ensure_decoded(name)
        attrs = node._v_attrs
        (factory, kwargs) = self._get_index_factory(attrs)
        if (kind == 'date'):
            index = factory(_unconvert_index(data, kind, encoding=self.encoding, errors=self.errors), dtype=object, **kwargs)
        else:
            index = factory(_unconvert_index(data, kind, encoding=self.encoding, errors=self.errors), **kwargs)
        index.name = name
        return index

    def write_array_empty(self, key, value):
        ' write a 0-len array '
        arr = np.empty(((1,) * value.ndim))
        self._handle.create_array(self.group, key, arr)
        node = getattr(self.group, key)
        node._v_attrs.value_type = str(value.dtype)
        node._v_attrs.shape = value.shape

    def write_array(self, key, obj, items=None):
        value = extract_array(obj, extract_numpy=True)
        if (key in self.group):
            self._handle.remove_node(self.group, key)
        empty_array = (value.size == 0)
        transposed = False
        if is_categorical_dtype(value.dtype):
            raise NotImplementedError('Cannot store a category dtype in a HDF5 dataset that uses format="fixed". Use format="table".')
        if (not empty_array):
            if hasattr(value, 'T'):
                value = value.T
                transposed = True
        atom = None
        if (self._filters is not None):
            with suppress(ValueError):
                atom = _tables().Atom.from_dtype(value.dtype)
        if (atom is not None):
            if (not empty_array):
                ca = self._handle.create_carray(self.group, key, atom, value.shape, filters=self._filters)
                ca[:] = value
            else:
                self.write_array_empty(key, value)
        elif (value.dtype.type == np.object_):
            inferred_type = lib.infer_dtype(value, skipna=False)
            if empty_array:
                pass
            elif (inferred_type == 'string'):
                pass
            else:
                ws = (performance_doc % (inferred_type, key, items))
                warnings.warn(ws, PerformanceWarning, stacklevel=7)
            vlarr = self._handle.create_vlarray(self.group, key, _tables().ObjectAtom())
            vlarr.append(value)
        elif is_datetime64_dtype(value.dtype):
            self._handle.create_array(self.group, key, value.view('i8'))
            getattr(self.group, key)._v_attrs.value_type = 'datetime64'
        elif is_datetime64tz_dtype(value.dtype):
            self._handle.create_array(self.group, key, value.asi8)
            node = getattr(self.group, key)
            node._v_attrs.tz = _get_tz(value.tz)
            node._v_attrs.value_type = 'datetime64'
        elif is_timedelta64_dtype(value.dtype):
            self._handle.create_array(self.group, key, value.view('i8'))
            getattr(self.group, key)._v_attrs.value_type = 'timedelta64'
        elif empty_array:
            self.write_array_empty(key, value)
        else:
            self._handle.create_array(self.group, key, value)
        getattr(self.group, key)._v_attrs.transposed = transposed

class SeriesFixed(GenericFixed):
    pandas_kind = 'series'
    attributes = ['name']

    @property
    def shape(self):
        try:
            return (len(self.group.values),)
        except (TypeError, AttributeError):
            return None

    def read(self, where=None, columns=None, start=None, stop=None):
        self.validate_read(columns, where)
        index = self.read_index('index', start=start, stop=stop)
        values = self.read_array('values', start=start, stop=stop)
        return Series(values, index=index, name=self.name)

    def write(self, obj, **kwargs):
        super().write(obj, **kwargs)
        self.write_index('index', obj.index)
        self.write_array('values', obj)
        self.attrs.name = obj.name

class BlockManagerFixed(GenericFixed):
    attributes = ['ndim', 'nblocks']

    @property
    def shape(self):
        try:
            ndim = self.ndim
            items = 0
            for i in range(self.nblocks):
                node = getattr(self.group, f'block{i}_items')
                shape = getattr(node, 'shape', None)
                if (shape is not None):
                    items += shape[0]
            node = self.group.block0_values
            shape = getattr(node, 'shape', None)
            if (shape is not None):
                shape = list(shape[0:(ndim - 1)])
            else:
                shape = []
            shape.append(items)
            return shape
        except AttributeError:
            return None

    def read(self, where=None, columns=None, start=None, stop=None):
        self.validate_read(columns, where)
        select_axis = self.obj_type()._get_block_manager_axis(0)
        axes = []
        for i in range(self.ndim):
            (_start, _stop) = ((start, stop) if (i == select_axis) else (None, None))
            ax = self.read_index(f'axis{i}', start=_start, stop=_stop)
            axes.append(ax)
        items = axes[0]
        dfs = []
        for i in range(self.nblocks):
            blk_items = self.read_index(f'block{i}_items')
            values = self.read_array(f'block{i}_values', start=_start, stop=_stop)
            columns = items[items.get_indexer(blk_items)]
            df = DataFrame(values.T, columns=columns, index=axes[1])
            dfs.append(df)
        if (len(dfs) > 0):
            out = concat(dfs, axis=1)
            out = out.reindex(columns=items, copy=False)
            return out
        return DataFrame(columns=axes[0], index=axes[1])

    def write(self, obj, **kwargs):
        super().write(obj, **kwargs)
        data = obj._mgr
        if (not data.is_consolidated()):
            data = data.consolidate()
        self.attrs.ndim = data.ndim
        for (i, ax) in enumerate(data.axes):
            if ((i == 0) and (not ax.is_unique)):
                raise ValueError('Columns index has to be unique for fixed format')
            self.write_index(f'axis{i}', ax)
        self.attrs.nblocks = len(data.blocks)
        for (i, blk) in enumerate(data.blocks):
            blk_items = data.items.take(blk.mgr_locs)
            self.write_array(f'block{i}_values', blk.values, items=blk_items)
            self.write_index(f'block{i}_items', blk_items)

class FrameFixed(BlockManagerFixed):
    pandas_kind = 'frame'
    obj_type = DataFrame

class Table(Fixed):
    '\n    represent a table:\n        facilitate read/write of various types of tables\n\n    Attrs in Table Node\n    -------------------\n    These are attributes that are store in the main table node, they are\n    necessary to recreate these tables when read back in.\n\n    index_axes    : a list of tuples of the (original indexing axis and\n        index column)\n    non_index_axes: a list of tuples of the (original index axis and\n        columns on a non-indexing axis)\n    values_axes   : a list of the columns which comprise the data of this\n        table\n    data_columns  : a list of the columns that we are allowing indexing\n        (these become single columns in values_axes), or True to force all\n        columns\n    nan_rep       : the string to use for nan representations for string\n        objects\n    levels        : the names of levels\n    metadata      : the names of the metadata columns\n    '
    pandas_kind = 'wide_table'
    format_type = 'table'
    levels = 1
    is_table = True

    def __init__(self, parent, group, encoding=None, errors='strict', index_axes=None, non_index_axes=None, values_axes=None, data_columns=None, info=None, nan_rep=None):
        super().__init__(parent, group, encoding=encoding, errors=errors)
        self.index_axes = (index_axes or [])
        self.non_index_axes = (non_index_axes or [])
        self.values_axes = (values_axes or [])
        self.data_columns = (data_columns or [])
        self.info = (info or {})
        self.nan_rep = nan_rep

    @property
    def table_type_short(self):
        return self.table_type.split('_')[0]

    def __repr__(self):
        ' return a pretty representation of myself '
        self.infer_axes()
        jdc = (','.join(self.data_columns) if len(self.data_columns) else '')
        dc = f',dc->[{jdc}]'
        ver = ''
        if self.is_old_version:
            jver = '.'.join((str(x) for x in self.version))
            ver = f'[{jver}]'
        jindex_axes = ','.join((a.name for a in self.index_axes))
        return f'{self.pandas_type:12.12}{ver} (typ->{self.table_type_short},nrows->{self.nrows},ncols->{self.ncols},indexers->[{jindex_axes}]{dc})'

    def __getitem__(self, c):
        ' return the axis for c '
        for a in self.axes:
            if (c == a.name):
                return a
        return None

    def validate(self, other):
        ' validate against an existing table '
        if (other is None):
            return
        if (other.table_type != self.table_type):
            raise TypeError(f'incompatible table_type with existing [{other.table_type} - {self.table_type}]')
        for c in ['index_axes', 'non_index_axes', 'values_axes']:
            sv = getattr(self, c, None)
            ov = getattr(other, c, None)
            if (sv != ov):
                for (i, sax) in enumerate(sv):
                    oax = ov[i]
                    if (sax != oax):
                        raise ValueError(f'invalid combination of [{c}] on appending data [{sax}] vs current table [{oax}]')
                raise Exception(f'invalid combination of [{c}] on appending data [{sv}] vs current table [{ov}]')

    @property
    def is_multi_index(self):
        'the levels attribute is 1 or a list in the case of a multi-index'
        return isinstance(self.levels, list)

    def validate_multiindex(self, obj):
        '\n        validate that we can store the multi-index; reset and return the\n        new object\n        '
        levels = [(l if (l is not None) else f'level_{i}') for (i, l) in enumerate(obj.index.names)]
        try:
            reset_obj = obj.reset_index()
        except ValueError as err:
            raise ValueError('duplicate names/columns in the multi-index when storing as a table') from err
        assert isinstance(reset_obj, DataFrame)
        return (reset_obj, levels)

    @property
    def nrows_expected(self):
        ' based on our axes, compute the expected nrows '
        return np.prod([i.cvalues.shape[0] for i in self.index_axes])

    @property
    def is_exists(self):
        ' has this table been created '
        return ('table' in self.group)

    @property
    def storable(self):
        return getattr(self.group, 'table', None)

    @property
    def table(self):
        ' return the table group (this is my storable) '
        return self.storable

    @property
    def dtype(self):
        return self.table.dtype

    @property
    def description(self):
        return self.table.description

    @property
    def axes(self):
        return itertools.chain(self.index_axes, self.values_axes)

    @property
    def ncols(self):
        ' the number of total columns in the values axes '
        return sum((len(a.values) for a in self.values_axes))

    @property
    def is_transposed(self):
        return False

    @property
    def data_orientation(self):
        'return a tuple of my permutated axes, non_indexable at the front'
        return tuple(itertools.chain([int(a[0]) for a in self.non_index_axes], [int(a.axis) for a in self.index_axes]))

    def queryables(self):
        ' return a dict of the kinds allowable columns for this object '
        axis_names = {0: 'index', 1: 'columns'}
        d1 = [(a.cname, a) for a in self.index_axes]
        d2 = [(axis_names[axis], None) for (axis, values) in self.non_index_axes]
        d3 = [(v.cname, v) for v in self.values_axes if (v.name in set(self.data_columns))]
        return dict(((d1 + d2) + d3))

    def index_cols(self):
        ' return a list of my index cols '
        return [(i.axis, i.cname) for i in self.index_axes]

    def values_cols(self):
        ' return a list of my values cols '
        return [i.cname for i in self.values_axes]

    def _get_metadata_path(self, key):
        ' return the metadata pathname for this key '
        group = self.group._v_pathname
        return f'{group}/meta/{key}/meta'

    def write_metadata(self, key, values):
        '\n        Write out a metadata array to the key as a fixed-format Series.\n\n        Parameters\n        ----------\n        key : str\n        values : ndarray\n        '
        values = Series(values)
        self.parent.put(self._get_metadata_path(key), values, format='table', encoding=self.encoding, errors=self.errors, nan_rep=self.nan_rep)

    def read_metadata(self, key):
        ' return the meta data array for this key '
        if (getattr(getattr(self.group, 'meta', None), key, None) is not None):
            return self.parent.select(self._get_metadata_path(key))
        return None

    def set_attrs(self):
        ' set our table type & indexables '
        self.attrs.table_type = str(self.table_type)
        self.attrs.index_cols = self.index_cols()
        self.attrs.values_cols = self.values_cols()
        self.attrs.non_index_axes = self.non_index_axes
        self.attrs.data_columns = self.data_columns
        self.attrs.nan_rep = self.nan_rep
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors
        self.attrs.levels = self.levels
        self.attrs.info = self.info

    def get_attrs(self):
        ' retrieve our attributes '
        self.non_index_axes = (getattr(self.attrs, 'non_index_axes', None) or [])
        self.data_columns = (getattr(self.attrs, 'data_columns', None) or [])
        self.info = (getattr(self.attrs, 'info', None) or {})
        self.nan_rep = getattr(self.attrs, 'nan_rep', None)
        self.encoding = _ensure_encoding(getattr(self.attrs, 'encoding', None))
        self.errors = _ensure_decoded(getattr(self.attrs, 'errors', 'strict'))
        self.levels: List[Label] = (getattr(self.attrs, 'levels', None) or [])
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if (not a.is_an_indexable)]

    def validate_version(self, where=None):
        ' are we trying to operate on an old version? '
        if (where is not None):
            if ((self.version[0] <= 0) and (self.version[1] <= 10) and (self.version[2] < 1)):
                ws = (incompatibility_doc % '.'.join([str(x) for x in self.version]))
                warnings.warn(ws, IncompatibilityWarning)

    def validate_min_itemsize(self, min_itemsize):
        "\n        validate the min_itemsize doesn't contain items that are not in the\n        axes this needs data_columns to be defined\n        "
        if (min_itemsize is None):
            return
        if (not isinstance(min_itemsize, dict)):
            return
        q = self.queryables()
        for (k, v) in min_itemsize.items():
            if (k == 'values'):
                continue
            if (k not in q):
                raise ValueError(f'min_itemsize has the key [{k}] which is not an axis or data_column')

    @cache_readonly
    def indexables(self):
        " create/cache the indexables if they don't exist "
        _indexables = []
        desc = self.description
        table_attrs = self.table.attrs
        for (i, (axis, name)) in enumerate(self.attrs.index_cols):
            atom = getattr(desc, name)
            md = self.read_metadata(name)
            meta = ('category' if (md is not None) else None)
            kind_attr = f'{name}_kind'
            kind = getattr(table_attrs, kind_attr, None)
            index_col = IndexCol(name=name, axis=axis, pos=i, kind=kind, typ=atom, table=self.table, meta=meta, metadata=md)
            _indexables.append(index_col)
        dc = set(self.data_columns)
        base_pos = len(_indexables)

        def f(i, c):
            assert isinstance(c, str)
            klass = DataCol
            if (c in dc):
                klass = DataIndexableCol
            atom = getattr(desc, c)
            adj_name = _maybe_adjust_name(c, self.version)
            values = getattr(table_attrs, f'{adj_name}_kind', None)
            dtype = getattr(table_attrs, f'{adj_name}_dtype', None)
            kind = _dtype_to_kind(dtype)
            md = self.read_metadata(c)
            meta = getattr(table_attrs, f'{adj_name}_meta', None)
            obj = klass(name=adj_name, cname=c, values=values, kind=kind, pos=(base_pos + i), typ=atom, table=self.table, meta=meta, metadata=md, dtype=dtype)
            return obj
        _indexables.extend([f(i, c) for (i, c) in enumerate(self.attrs.values_cols)])
        return _indexables

    def create_index(self, columns=None, optlevel=None, kind=None):
        '\n        Create a pytables index on the specified columns.\n\n        Parameters\n        ----------\n        columns : None, bool, or listlike[str]\n            Indicate which columns to create an index on.\n\n            * False : Do not create any indexes.\n            * True : Create indexes on all columns.\n            * None : Create indexes on all columns.\n            * listlike : Create indexes on the given columns.\n\n        optlevel : int or None, default None\n            Optimization level, if None, pytables defaults to 6.\n        kind : str or None, default None\n            Kind of index, if None, pytables defaults to "medium".\n\n        Raises\n        ------\n        TypeError if trying to create an index on a complex-type column.\n\n        Notes\n        -----\n        Cannot index Time64Col or ComplexCol.\n        Pytables must be >= 3.0.\n        '
        if (not self.infer_axes()):
            return
        if (columns is False):
            return
        if ((columns is None) or (columns is True)):
            columns = [a.cname for a in self.axes if a.is_data_indexable]
        if (not isinstance(columns, (tuple, list))):
            columns = [columns]
        kw = {}
        if (optlevel is not None):
            kw['optlevel'] = optlevel
        if (kind is not None):
            kw['kind'] = kind
        table = self.table
        for c in columns:
            v = getattr(table.cols, c, None)
            if (v is not None):
                if v.is_indexed:
                    index = v.index
                    cur_optlevel = index.optlevel
                    cur_kind = index.kind
                    if ((kind is not None) and (cur_kind != kind)):
                        v.remove_index()
                    else:
                        kw['kind'] = cur_kind
                    if ((optlevel is not None) and (cur_optlevel != optlevel)):
                        v.remove_index()
                    else:
                        kw['optlevel'] = cur_optlevel
                if (not v.is_indexed):
                    if v.type.startswith('complex'):
                        raise TypeError('Columns containing complex values can be stored but cannot be indexed when using table format. Either use fixed format, set index=False, or do not include the columns containing complex values to data_columns when initializing the table.')
                    v.create_index(**kw)
            elif (c in self.non_index_axes[0][1]):
                raise AttributeError(f'''column {c} is not a data_column.
In order to read column {c} you must reload the dataframe 
into HDFStore and include {c} with the data_columns argument.''')

    def _read_axes(self, where, start=None, stop=None):
        '\n        Create the axes sniffed from the table.\n\n        Parameters\n        ----------\n        where : ???\n        start : int or None, default None\n        stop : int or None, default None\n\n        Returns\n        -------\n        List[Tuple[index_values, column_values]]\n        '
        selection = Selection(self, where=where, start=start, stop=stop)
        values = selection.select()
        results = []
        for a in self.axes:
            a.set_info(self.info)
            res = a.convert(values, nan_rep=self.nan_rep, encoding=self.encoding, errors=self.errors)
            results.append(res)
        return results

    @classmethod
    def get_object(cls, obj, transposed):
        ' return the data for this obj '
        return obj

    def validate_data_columns(self, data_columns, min_itemsize, non_index_axes):
        '\n        take the input data_columns and min_itemize and create a data\n        columns spec\n        '
        if (not len(non_index_axes)):
            return []
        (axis, axis_labels) = non_index_axes[0]
        info = self.info.get(axis, {})
        if ((info.get('type') == 'MultiIndex') and data_columns):
            raise ValueError(f'cannot use a multi-index on axis [{axis}] with data_columns {data_columns}')
        if (data_columns is True):
            data_columns = list(axis_labels)
        elif (data_columns is None):
            data_columns = []
        if isinstance(min_itemsize, dict):
            existing_data_columns = set(data_columns)
            data_columns = list(data_columns)
            data_columns.extend([k for k in min_itemsize.keys() if ((k != 'values') and (k not in existing_data_columns))])
        return [c for c in data_columns if (c in axis_labels)]

    def _create_axes(self, axes, obj, validate=True, nan_rep=None, data_columns=None, min_itemsize=None):
        '\n        Create and return the axes.\n\n        Parameters\n        ----------\n        axes: list or None\n            The names or numbers of the axes to create.\n        obj : DataFrame\n            The object to create axes on.\n        validate: bool, default True\n            Whether to validate the obj against an existing object already written.\n        nan_rep :\n            A value to use for string column nan_rep.\n        data_columns : List[str], True, or None, default None\n            Specify the columns that we want to create to allow indexing on.\n\n            * True : Use all available columns.\n            * None : Use no columns.\n            * List[str] : Use the specified columns.\n\n        min_itemsize: Dict[str, int] or None, default None\n            The min itemsize for a column in bytes.\n        '
        if (not isinstance(obj, DataFrame)):
            group = self.group._v_name
            raise TypeError(f'cannot properly create the storer for: [group->{group},value->{type(obj)}]')
        if (axes is None):
            axes = [0]
        axes = [obj._get_axis_number(a) for a in axes]
        if self.infer_axes():
            table_exists = True
            axes = [a.axis for a in self.index_axes]
            data_columns = list(self.data_columns)
            nan_rep = self.nan_rep
        else:
            table_exists = False
        new_info = self.info
        assert (self.ndim == 2)
        if (len(axes) != (self.ndim - 1)):
            raise ValueError('currently only support ndim-1 indexers in an AppendableTable')
        new_non_index_axes: List = []
        if (nan_rep is None):
            nan_rep = 'nan'
        idx = [x for x in [0, 1] if (x not in axes)][0]
        a = obj.axes[idx]
        append_axis = list(a)
        if table_exists:
            indexer = len(new_non_index_axes)
            exist_axis = self.non_index_axes[indexer][1]
            if (not array_equivalent(np.array(append_axis), np.array(exist_axis))):
                if array_equivalent(np.array(sorted(append_axis)), np.array(sorted(exist_axis))):
                    append_axis = exist_axis
        info = new_info.setdefault(idx, {})
        info['names'] = list(a.names)
        info['type'] = type(a).__name__
        new_non_index_axes.append((idx, append_axis))
        idx = axes[0]
        a = obj.axes[idx]
        axis_name = obj._get_axis_name(idx)
        new_index = _convert_index(axis_name, a, self.encoding, self.errors)
        new_index.axis = idx
        new_index.set_pos(0)
        new_index.update_info(new_info)
        new_index.maybe_set_size(min_itemsize)
        new_index_axes = [new_index]
        j = len(new_index_axes)
        assert (j == 1)
        assert (len(new_non_index_axes) == 1)
        for a in new_non_index_axes:
            obj = _reindex_axis(obj, a[0], a[1])

        def get_blk_items(mgr, blocks):
            return [mgr.items.take(blk.mgr_locs) for blk in blocks]
        transposed = (new_index.axis == 1)
        data_columns = self.validate_data_columns(data_columns, min_itemsize, new_non_index_axes)
        block_obj = self.get_object(obj, transposed)._consolidate()
        (blocks, blk_items) = self._get_blocks_and_items(block_obj, table_exists, new_non_index_axes, self.values_axes, data_columns)
        vaxes = []
        for (i, (b, b_items)) in enumerate(zip(blocks, blk_items)):
            klass = DataCol
            name = None
            if (data_columns and (len(b_items) == 1) and (b_items[0] in data_columns)):
                klass = DataIndexableCol
                name = b_items[0]
                if (not ((name is None) or isinstance(name, str))):
                    raise ValueError('cannot have non-object label DataIndexableCol')
            existing_col: Optional[DataCol]
            if (table_exists and validate):
                try:
                    existing_col = self.values_axes[i]
                except (IndexError, KeyError) as err:
                    raise ValueError(f'Incompatible appended table [{blocks}]with existing table [{self.values_axes}]') from err
            else:
                existing_col = None
            new_name = (name or f'values_block_{i}')
            data_converted = _maybe_convert_for_string_atom(new_name, b, existing_col=existing_col, min_itemsize=min_itemsize, nan_rep=nan_rep, encoding=self.encoding, errors=self.errors)
            adj_name = _maybe_adjust_name(new_name, self.version)
            typ = klass._get_atom(data_converted)
            kind = _dtype_to_kind(data_converted.dtype.name)
            tz = (_get_tz(data_converted.tz) if hasattr(data_converted, 'tz') else None)
            meta = metadata = ordered = None
            if is_categorical_dtype(data_converted.dtype):
                ordered = data_converted.ordered
                meta = 'category'
                metadata = np.array(data_converted.categories, copy=False).ravel()
            (data, dtype_name) = _get_data_and_dtype_name(data_converted)
            col = klass(name=adj_name, cname=new_name, values=list(b_items), typ=typ, pos=j, kind=kind, tz=tz, ordered=ordered, meta=meta, metadata=metadata, dtype=dtype_name, data=data)
            col.update_info(new_info)
            vaxes.append(col)
            j += 1
        dcs = [col.name for col in vaxes if col.is_data_indexable]
        new_table = type(self)(parent=self.parent, group=self.group, encoding=self.encoding, errors=self.errors, index_axes=new_index_axes, non_index_axes=new_non_index_axes, values_axes=vaxes, data_columns=dcs, info=new_info, nan_rep=nan_rep)
        if hasattr(self, 'levels'):
            new_table.levels = self.levels
        new_table.validate_min_itemsize(min_itemsize)
        if (validate and table_exists):
            new_table.validate(self)
        return new_table

    @staticmethod
    def _get_blocks_and_items(block_obj, table_exists, new_non_index_axes, values_axes, data_columns):

        def get_blk_items(mgr, blocks):
            return [mgr.items.take(blk.mgr_locs) for blk in blocks]
        blocks = block_obj._mgr.blocks
        blk_items = get_blk_items(block_obj._mgr, blocks)
        if len(data_columns):
            (axis, axis_labels) = new_non_index_axes[0]
            new_labels = Index(axis_labels).difference(Index(data_columns))
            mgr = block_obj.reindex(new_labels, axis=axis)._mgr
            blocks = list(mgr.blocks)
            blk_items = get_blk_items(mgr, blocks)
            for c in data_columns:
                mgr = block_obj.reindex([c], axis=axis)._mgr
                blocks.extend(mgr.blocks)
                blk_items.extend(get_blk_items(mgr, mgr.blocks))
        if table_exists:
            by_items = {tuple(b_items.tolist()): (b, b_items) for (b, b_items) in zip(blocks, blk_items)}
            new_blocks = []
            new_blk_items = []
            for ea in values_axes:
                items = tuple(ea.values)
                try:
                    (b, b_items) = by_items.pop(items)
                    new_blocks.append(b)
                    new_blk_items.append(b_items)
                except (IndexError, KeyError) as err:
                    jitems = ','.join((pprint_thing(item) for item in items))
                    raise ValueError(f'cannot match existing table structure for [{jitems}] on appending data') from err
            blocks = new_blocks
            blk_items = new_blk_items
        return (blocks, blk_items)

    def process_axes(self, obj, selection, columns=None):
        ' process axes filters '
        if (columns is not None):
            columns = list(columns)
        if ((columns is not None) and self.is_multi_index):
            assert isinstance(self.levels, list)
            for n in self.levels:
                if (n not in columns):
                    columns.insert(0, n)
        for (axis, labels) in self.non_index_axes:
            obj = _reindex_axis(obj, axis, labels, columns)
        if (selection.filter is not None):
            for (field, op, filt) in selection.filter.format():

                def process_filter(field, filt):
                    for axis_name in obj._AXIS_ORDERS:
                        axis_number = obj._get_axis_number(axis_name)
                        axis_values = obj._get_axis(axis_name)
                        assert (axis_number is not None)
                        if (field == axis_name):
                            if self.is_multi_index:
                                filt = filt.union(Index(self.levels))
                            takers = op(axis_values, filt)
                            return obj.loc(axis=axis_number)[takers]
                        elif (field in axis_values):
                            values = ensure_index(getattr(obj, field).values)
                            filt = ensure_index(filt)
                            if isinstance(obj, DataFrame):
                                axis_number = (1 - axis_number)
                            takers = op(values, filt)
                            return obj.loc(axis=axis_number)[takers]
                    raise ValueError(f'cannot find the field [{field}] for filtering!')
                obj = process_filter(field, filt)
        return obj

    def create_description(self, complib, complevel, fletcher32, expectedrows):
        ' create the description of the table from the axes & values '
        if (expectedrows is None):
            expectedrows = max(self.nrows_expected, 10000)
        d = {'name': 'table', 'expectedrows': expectedrows}
        d['description'] = {a.cname: a.typ for a in self.axes}
        if complib:
            if (complevel is None):
                complevel = (self._complevel or 9)
            filters = _tables().Filters(complevel=complevel, complib=complib, fletcher32=(fletcher32 or self._fletcher32))
            d['filters'] = filters
        elif (self._filters is not None):
            d['filters'] = self._filters
        return d

    def read_coordinates(self, where=None, start=None, stop=None):
        '\n        select coordinates (row numbers) from a table; return the\n        coordinates object\n        '
        self.validate_version(where)
        if (not self.infer_axes()):
            return False
        selection = Selection(self, where=where, start=start, stop=stop)
        coords = selection.select_coords()
        if (selection.filter is not None):
            for (field, op, filt) in selection.filter.format():
                data = self.read_column(field, start=coords.min(), stop=(coords.max() + 1))
                coords = coords[op(data.iloc[(coords - coords.min())], filt).values]
        return Index(coords)

    def read_column(self, column, where=None, start=None, stop=None):
        '\n        return a single column from the table, generally only indexables\n        are interesting\n        '
        self.validate_version()
        if (not self.infer_axes()):
            return False
        if (where is not None):
            raise TypeError('read_column does not currently accept a where clause')
        for a in self.axes:
            if (column == a.name):
                if (not a.is_data_indexable):
                    raise ValueError(f'column [{column}] can not be extracted individually; it is not data indexable')
                c = getattr(self.table.cols, column)
                a.set_info(self.info)
                col_values = a.convert(c[start:stop], nan_rep=self.nan_rep, encoding=self.encoding, errors=self.errors)
                return Series(_set_tz(col_values[1], a.tz), name=column)
        raise KeyError(f'column [{column}] not found in the table')

class WORMTable(Table):
    '\n    a write-once read-many table: this format DOES NOT ALLOW appending to a\n    table. writing is a one-time operation the data are stored in a format\n    that allows for searching the data on disk\n    '
    table_type = 'worm'

    def read(self, where=None, columns=None, start=None, stop=None):
        '\n        read the indices and the indexing array, calculate offset rows and return\n        '
        raise NotImplementedError('WORMTable needs to implement read')

    def write(self, **kwargs):
        '\n        write in a format that we can search later on (but cannot append\n        to): write out the indices and the values using _write_array\n        (e.g. a CArray) create an indexing table so that we can search\n        '
        raise NotImplementedError('WORMTable needs to implement write')

class AppendableTable(Table):
    ' support the new appendable table formats '
    table_type = 'appendable'

    def write(self, obj, axes=None, append=False, complib=None, complevel=None, fletcher32=None, min_itemsize=None, chunksize=None, expectedrows=None, dropna=False, nan_rep=None, data_columns=None, track_times=True):
        if ((not append) and self.is_exists):
            self._handle.remove_node(self.group, 'table')
        table = self._create_axes(axes=axes, obj=obj, validate=append, min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=data_columns)
        for a in table.axes:
            a.validate_names()
        if (not table.is_exists):
            options = table.create_description(complib=complib, complevel=complevel, fletcher32=fletcher32, expectedrows=expectedrows)
            table.set_attrs()
            options['track_times'] = track_times
            table._handle.create_table(table.group, **options)
        table.attrs.info = table.info
        for a in table.axes:
            a.validate_and_set(table, append)
        table.write_data(chunksize, dropna=dropna)

    def write_data(self, chunksize, dropna=False):
        '\n        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk\n        '
        names = self.dtype.names
        nrows = self.nrows_expected
        masks = []
        if dropna:
            for a in self.values_axes:
                mask = isna(a.data).all(axis=0)
                if isinstance(mask, np.ndarray):
                    masks.append(mask.astype('u1', copy=False))
        if len(masks):
            mask = masks[0]
            for m in masks[1:]:
                mask = (mask & m)
            mask = mask.ravel()
        else:
            mask = None
        indexes = [a.cvalues for a in self.index_axes]
        nindexes = len(indexes)
        assert (nindexes == 1), nindexes
        values = [a.take_data() for a in self.values_axes]
        values = [v.transpose(np.roll(np.arange(v.ndim), (v.ndim - 1))) for v in values]
        bvalues = []
        for (i, v) in enumerate(values):
            new_shape = ((nrows,) + self.dtype[names[(nindexes + i)]].shape)
            bvalues.append(values[i].reshape(new_shape))
        if (chunksize is None):
            chunksize = 100000
        rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
        chunks = ((nrows // chunksize) + 1)
        for i in range(chunks):
            start_i = (i * chunksize)
            end_i = min(((i + 1) * chunksize), nrows)
            if (start_i >= end_i):
                break
            self.write_data_chunk(rows, indexes=[a[start_i:end_i] for a in indexes], mask=(mask[start_i:end_i] if (mask is not None) else None), values=[v[start_i:end_i] for v in bvalues])

    def write_data_chunk(self, rows, indexes, mask, values):
        '\n        Parameters\n        ----------\n        rows : an empty memory space where we are putting the chunk\n        indexes : an array of the indexes\n        mask : an array of the masks\n        values : an array of the values\n        '
        for v in values:
            if (not np.prod(v.shape)):
                return
        nrows = indexes[0].shape[0]
        if (nrows != len(rows)):
            rows = np.empty(nrows, dtype=self.dtype)
        names = self.dtype.names
        nindexes = len(indexes)
        for (i, idx) in enumerate(indexes):
            rows[names[i]] = idx
        for (i, v) in enumerate(values):
            rows[names[(i + nindexes)]] = v
        if (mask is not None):
            m = (~ mask.ravel().astype(bool, copy=False))
            if (not m.all()):
                rows = rows[m]
        if len(rows):
            self.table.append(rows)
            self.table.flush()

    def delete(self, where=None, start=None, stop=None):
        if ((where is None) or (not len(where))):
            if ((start is None) and (stop is None)):
                nrows = self.nrows
                self._handle.remove_node(self.group, recursive=True)
            else:
                if (stop is None):
                    stop = self.nrows
                nrows = self.table.remove_rows(start=start, stop=stop)
                self.table.flush()
            return nrows
        if (not self.infer_axes()):
            return None
        table = self.table
        selection = Selection(self, where, start=start, stop=stop)
        values = selection.select_coords()
        sorted_series = Series(values).sort_values()
        ln = len(sorted_series)
        if ln:
            diff = sorted_series.diff()
            groups = list(diff[(diff > 1)].index)
            if (not len(groups)):
                groups = [0]
            if (groups[(- 1)] != ln):
                groups.append(ln)
            if (groups[0] != 0):
                groups.insert(0, 0)
            pg = groups.pop()
            for g in reversed(groups):
                rows = sorted_series.take(range(g, pg))
                table.remove_rows(start=rows[rows.index[0]], stop=(rows[rows.index[(- 1)]] + 1))
                pg = g
            self.table.flush()
        return ln

class AppendableFrameTable(AppendableTable):
    ' support the new appendable table formats '
    pandas_kind = 'frame_table'
    table_type = 'appendable_frame'
    ndim = 2
    obj_type = DataFrame

    @property
    def is_transposed(self):
        return (self.index_axes[0].axis == 1)

    @classmethod
    def get_object(cls, obj, transposed):
        ' these are written transposed '
        if transposed:
            obj = obj.T
        return obj

    def read(self, where=None, columns=None, start=None, stop=None):
        self.validate_version(where)
        if (not self.infer_axes()):
            return None
        result = self._read_axes(where=where, start=start, stop=stop)
        info = (self.info.get(self.non_index_axes[0][0], {}) if len(self.non_index_axes) else {})
        inds = [i for (i, ax) in enumerate(self.axes) if (ax is self.index_axes[0])]
        assert (len(inds) == 1)
        ind = inds[0]
        index = result[ind][0]
        frames = []
        for (i, a) in enumerate(self.axes):
            if (a not in self.values_axes):
                continue
            (index_vals, cvalues) = result[i]
            if (info.get('type') == 'MultiIndex'):
                cols = MultiIndex.from_tuples(index_vals)
            else:
                cols = Index(index_vals)
            names = info.get('names')
            if (names is not None):
                cols.set_names(names, inplace=True)
            if self.is_transposed:
                values = cvalues
                index_ = cols
                cols_ = Index(index, name=getattr(index, 'name', None))
            else:
                values = cvalues.T
                index_ = Index(index, name=getattr(index, 'name', None))
                cols_ = cols
            if ((values.ndim == 1) and isinstance(values, np.ndarray)):
                values = values.reshape((1, values.shape[0]))
            if isinstance(values, np.ndarray):
                df = DataFrame(values.T, columns=cols_, index=index_)
            elif isinstance(values, Index):
                df = DataFrame(values, columns=cols_, index=index_)
            else:
                df = DataFrame([values], columns=cols_, index=index_)
            assert (df.dtypes == values.dtype).all(), (df.dtypes, values.dtype)
            frames.append(df)
        if (len(frames) == 1):
            df = frames[0]
        else:
            df = concat(frames, axis=1)
        selection = Selection(self, where=where, start=start, stop=stop)
        df = self.process_axes(df, selection=selection, columns=columns)
        return df

class AppendableSeriesTable(AppendableFrameTable):
    ' support the new appendable table formats '
    pandas_kind = 'series_table'
    table_type = 'appendable_series'
    ndim = 2
    obj_type = Series

    @property
    def is_transposed(self):
        return False

    @classmethod
    def get_object(cls, obj, transposed):
        return obj

    def write(self, obj, data_columns=None, **kwargs):
        ' we are going to write this as a frame table '
        if (not isinstance(obj, DataFrame)):
            name = (obj.name or 'values')
            obj = obj.to_frame(name)
        return super().write(obj=obj, data_columns=obj.columns.tolist(), **kwargs)

    def read(self, where=None, columns=None, start=None, stop=None):
        is_multi_index = self.is_multi_index
        if ((columns is not None) and is_multi_index):
            assert isinstance(self.levels, list)
            for n in self.levels:
                if (n not in columns):
                    columns.insert(0, n)
        s = super().read(where=where, columns=columns, start=start, stop=stop)
        if is_multi_index:
            s.set_index(self.levels, inplace=True)
        s = s.iloc[:, 0]
        if (s.name == 'values'):
            s.name = None
        return s

class AppendableMultiSeriesTable(AppendableSeriesTable):
    ' support the new appendable table formats '
    pandas_kind = 'series_table'
    table_type = 'appendable_multiseries'

    def write(self, obj, **kwargs):
        ' we are going to write this as a frame table '
        name = (obj.name or 'values')
        (newobj, self.levels) = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)
        cols = list(self.levels)
        cols.append(name)
        newobj.columns = Index(cols)
        return super().write(obj=newobj, **kwargs)

class GenericTable(AppendableFrameTable):
    ' a table that read/writes the generic pytables table format '
    pandas_kind = 'frame_table'
    table_type = 'generic_table'
    ndim = 2
    obj_type = DataFrame

    @property
    def pandas_type(self):
        return self.pandas_kind

    @property
    def storable(self):
        return (getattr(self.group, 'table', None) or self.group)

    def get_attrs(self):
        ' retrieve our attributes '
        self.non_index_axes = []
        self.nan_rep = None
        self.levels = []
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if (not a.is_an_indexable)]
        self.data_columns = [a.name for a in self.values_axes]

    @cache_readonly
    def indexables(self):
        ' create the indexables from the table description '
        d = self.description
        md = self.read_metadata('index')
        meta = ('category' if (md is not None) else None)
        index_col = GenericIndexCol(name='index', axis=0, table=self.table, meta=meta, metadata=md)
        _indexables: List[Union[(GenericIndexCol, GenericDataIndexableCol)]] = [index_col]
        for (i, n) in enumerate(d._v_names):
            assert isinstance(n, str)
            atom = getattr(d, n)
            md = self.read_metadata(n)
            meta = ('category' if (md is not None) else None)
            dc = GenericDataIndexableCol(name=n, pos=i, values=[n], typ=atom, table=self.table, meta=meta, metadata=md)
            _indexables.append(dc)
        return _indexables

    def write(self, **kwargs):
        raise NotImplementedError('cannot write on an generic table')

class AppendableMultiFrameTable(AppendableFrameTable):
    ' a frame with a multi-index '
    table_type = 'appendable_multiframe'
    obj_type = DataFrame
    ndim = 2
    _re_levels = re.compile('^level_\\d+$')

    @property
    def table_type_short(self):
        return 'appendable_multi'

    def write(self, obj, data_columns=None, **kwargs):
        if (data_columns is None):
            data_columns = []
        elif (data_columns is True):
            data_columns = obj.columns.tolist()
        (obj, self.levels) = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)
        for n in self.levels:
            if (n not in data_columns):
                data_columns.insert(0, n)
        return super().write(obj=obj, data_columns=data_columns, **kwargs)

    def read(self, where=None, columns=None, start=None, stop=None):
        df = super().read(where=where, columns=columns, start=start, stop=stop)
        df = df.set_index(self.levels)
        df.index = df.index.set_names([(None if self._re_levels.search(name) else name) for name in df.index.names])
        return df

def _reindex_axis(obj, axis, labels, other=None):
    ax = obj._get_axis(axis)
    labels = ensure_index(labels)
    if (other is not None):
        other = ensure_index(other)
    if (((other is None) or labels.equals(other)) and labels.equals(ax)):
        return obj
    labels = ensure_index(labels.unique())
    if (other is not None):
        labels = ensure_index(other.unique()).intersection(labels, sort=False)
    if (not labels.equals(ax)):
        slicer: List[Union[(slice, Index)]] = ([slice(None, None)] * obj.ndim)
        slicer[axis] = labels
        obj = obj.loc[tuple(slicer)]
    return obj

def _get_tz(tz):
    ' for a tz-aware type, return an encoded zone '
    zone = timezones.get_timezone(tz)
    return zone

def _set_tz(values, tz, coerce=False):
    '\n    coerce the values to a DatetimeIndex if tz is set\n    preserve the input shape if possible\n\n    Parameters\n    ----------\n    values : ndarray or Index\n    tz : str or tzinfo\n    coerce : if we do not have a passed timezone, coerce to M8[ns] ndarray\n    '
    if isinstance(values, DatetimeIndex):
        assert ((values.tz is None) or (values.tz == tz))
    if (tz is not None):
        if isinstance(values, DatetimeIndex):
            name = values.name
            values = values.asi8
        else:
            name = None
            values = values.ravel()
        tz = _ensure_decoded(tz)
        values = DatetimeIndex(values, name=name)
        values = values.tz_localize('UTC').tz_convert(tz)
    elif coerce:
        values = np.asarray(values, dtype='M8[ns]')
    return values

def _convert_index(name, index, encoding, errors):
    assert isinstance(name, str)
    index_name = index.name
    (converted, dtype_name) = _get_data_and_dtype_name(index)
    kind = _dtype_to_kind(dtype_name)
    atom = DataIndexableCol._get_atom(converted)
    if (isinstance(index, Int64Index) or needs_i8_conversion(index.dtype)):
        return IndexCol(name, values=converted, kind=kind, typ=atom, freq=getattr(index, 'freq', None), tz=getattr(index, 'tz', None), index_name=index_name)
    if isinstance(index, MultiIndex):
        raise TypeError('MultiIndex not supported here!')
    inferred_type = lib.infer_dtype(index, skipna=False)
    values = np.asarray(index)
    if (inferred_type == 'date'):
        converted = np.asarray([v.toordinal() for v in values], dtype=np.int32)
        return IndexCol(name, converted, 'date', _tables().Time32Col(), index_name=index_name)
    elif (inferred_type == 'string'):
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(name, converted, 'string', _tables().StringCol(itemsize), index_name=index_name)
    elif (inferred_type in ['integer', 'floating']):
        return IndexCol(name, values=converted, kind=kind, typ=atom, index_name=index_name)
    else:
        assert (isinstance(converted, np.ndarray) and (converted.dtype == object))
        assert (kind == 'object'), kind
        atom = _tables().ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)

def _unconvert_index(data, kind, encoding, errors):
    index: Union[(Index, np.ndarray)]
    if (kind == 'datetime64'):
        index = DatetimeIndex(data)
    elif (kind == 'timedelta64'):
        index = TimedeltaIndex(data)
    elif (kind == 'date'):
        try:
            index = np.asarray([date.fromordinal(v) for v in data], dtype=object)
        except ValueError:
            index = np.asarray([date.fromtimestamp(v) for v in data], dtype=object)
    elif (kind in ('integer', 'float')):
        index = np.asarray(data)
    elif (kind in 'string'):
        index = _unconvert_string_array(data, nan_rep=None, encoding=encoding, errors=errors)
    elif (kind == 'object'):
        index = np.asarray(data[0])
    else:
        raise ValueError(f'unrecognized index type {kind}')
    return index

def _maybe_convert_for_string_atom(name, block, existing_col, min_itemsize, nan_rep, encoding, errors):
    if (not block.is_object):
        return block.values
    dtype_name = block.dtype.name
    inferred_type = lib.infer_dtype(block.values, skipna=False)
    if (inferred_type == 'date'):
        raise TypeError('[date] is not implemented as a table column')
    elif (inferred_type == 'datetime'):
        raise TypeError('too many timezones in this block, create separate data columns')
    elif (not ((inferred_type == 'string') or (dtype_name == 'object'))):
        return block.values
    block = block.fillna(nan_rep, downcast=False)
    if isinstance(block, list):
        block = block[0]
    data = block.values
    inferred_type = lib.infer_dtype(data, skipna=False)
    if (inferred_type != 'string'):
        for i in range(len(block.shape[0])):
            col = block.iget(i)
            inferred_type = lib.infer_dtype(col, skipna=False)
            if (inferred_type != 'string'):
                iloc = block.mgr_locs.indexer[i]
                raise TypeError(f'''Cannot serialize the column [{iloc}] because
its data contents are [{inferred_type}] object dtype''')
    data_converted = _convert_string_array(data, encoding, errors).reshape(data.shape)
    assert (data_converted.shape == block.shape), (data_converted.shape, block.shape)
    itemsize = data_converted.itemsize
    if isinstance(min_itemsize, dict):
        min_itemsize = int((min_itemsize.get(name) or min_itemsize.get('values') or 0))
    itemsize = max((min_itemsize or 0), itemsize)
    if (existing_col is not None):
        eci = existing_col.validate_col(itemsize)
        if (eci > itemsize):
            itemsize = eci
    data_converted = data_converted.astype(f'|S{itemsize}', copy=False)
    return data_converted

def _convert_string_array(data, encoding, errors):
    '\n    Take a string-like that is object dtype and coerce to a fixed size string type.\n\n    Parameters\n    ----------\n    data : np.ndarray[object]\n    encoding : str\n    errors : str\n        Handler for encoding errors.\n\n    Returns\n    -------\n    np.ndarray[fixed-length-string]\n    '
    if len(data):
        data = Series(data.ravel()).str.encode(encoding, errors)._values.reshape(data.shape)
    ensured = ensure_object(data.ravel())
    itemsize = max(1, libwriters.max_len_string_array(ensured))
    data = np.asarray(data, dtype=f'S{itemsize}')
    return data

def _unconvert_string_array(data, nan_rep, encoding, errors):
    '\n    Inverse of _convert_string_array.\n\n    Parameters\n    ----------\n    data : np.ndarray[fixed-length-string]\n    nan_rep : the storage repr of NaN\n    encoding : str\n    errors : str\n        Handler for encoding errors.\n\n    Returns\n    -------\n    np.ndarray[object]\n        Decoded data.\n    '
    shape = data.shape
    data = np.asarray(data.ravel(), dtype=object)
    if len(data):
        itemsize = libwriters.max_len_string_array(ensure_object(data))
        dtype = f'U{itemsize}'
        if isinstance(data[0], bytes):
            data = Series(data).str.decode(encoding, errors=errors)._values
        else:
            data = data.astype(dtype, copy=False).astype(object, copy=False)
    if (nan_rep is None):
        nan_rep = 'nan'
    data = libwriters.string_array_replace_from_nan_rep(data, nan_rep)
    return data.reshape(shape)

def _maybe_convert(values, val_kind, encoding, errors):
    assert isinstance(val_kind, str), type(val_kind)
    if _need_convert(val_kind):
        conv = _get_converter(val_kind, encoding, errors)
        values = conv(values)
    return values

def _get_converter(kind, encoding, errors):
    if (kind == 'datetime64'):
        return (lambda x: np.asarray(x, dtype='M8[ns]'))
    elif (kind == 'string'):
        return (lambda x: _unconvert_string_array(x, nan_rep=None, encoding=encoding, errors=errors))
    else:
        raise ValueError(f'invalid kind {kind}')

def _need_convert(kind):
    if (kind in ('datetime64', 'string')):
        return True
    return False

def _maybe_adjust_name(name, version):
    '\n    Prior to 0.10.1, we named values blocks like: values_block_0 an the\n    name values_0, adjust the given name if necessary.\n\n    Parameters\n    ----------\n    name : str\n    version : Tuple[int, int, int]\n\n    Returns\n    -------\n    str\n    '
    if (isinstance(version, str) or (len(version) < 3)):
        raise ValueError('Version is incorrect, expected sequence of 3 integers.')
    if ((version[0] == 0) and (version[1] <= 10) and (version[2] == 0)):
        m = re.search('values_block_(\\d+)', name)
        if m:
            grp = m.groups()[0]
            name = f'values_{grp}'
    return name

def _dtype_to_kind(dtype_str):
    '\n    Find the "kind" string describing the given dtype name.\n    '
    dtype_str = _ensure_decoded(dtype_str)
    if (dtype_str.startswith('string') or dtype_str.startswith('bytes')):
        kind = 'string'
    elif dtype_str.startswith('float'):
        kind = 'float'
    elif dtype_str.startswith('complex'):
        kind = 'complex'
    elif (dtype_str.startswith('int') or dtype_str.startswith('uint')):
        kind = 'integer'
    elif dtype_str.startswith('datetime64'):
        kind = 'datetime64'
    elif dtype_str.startswith('timedelta'):
        kind = 'timedelta64'
    elif dtype_str.startswith('bool'):
        kind = 'bool'
    elif dtype_str.startswith('category'):
        kind = 'category'
    elif dtype_str.startswith('period'):
        kind = 'integer'
    elif (dtype_str == 'object'):
        kind = 'object'
    else:
        raise ValueError(f'cannot interpret dtype of [{dtype_str}]')
    return kind

def _get_data_and_dtype_name(data):
    '\n    Convert the passed data into a storable form and a dtype string.\n    '
    if isinstance(data, Categorical):
        data = data.codes
    dtype_name = data.dtype.name.split('[')[0]
    if (data.dtype.kind in ['m', 'M']):
        data = np.asarray(data.view('i8'))
    elif isinstance(data, PeriodIndex):
        data = data.asi8
    data = np.asarray(data)
    return (data, dtype_name)

class Selection():
    '\n    Carries out a selection operation on a tables.Table object.\n\n    Parameters\n    ----------\n    table : a Table object\n    where : list of Terms (or convertible to)\n    start, stop: indices to start and/or stop selection\n\n    '

    def __init__(self, table, where=None, start=None, stop=None):
        self.table = table
        self.where = where
        self.start = start
        self.stop = stop
        self.condition = None
        self.filter = None
        self.terms = None
        self.coordinates = None
        if is_list_like(where):
            with suppress(ValueError):
                inferred = lib.infer_dtype(where, skipna=False)
                if ((inferred == 'integer') or (inferred == 'boolean')):
                    where = np.asarray(where)
                    if (where.dtype == np.bool_):
                        (start, stop) = (self.start, self.stop)
                        if (start is None):
                            start = 0
                        if (stop is None):
                            stop = self.table.nrows
                        self.coordinates = np.arange(start, stop)[where]
                    elif issubclass(where.dtype.type, np.integer):
                        if (((self.start is not None) and (where < self.start).any()) or ((self.stop is not None) and (where >= self.stop).any())):
                            raise ValueError('where must have index locations >= start and < stop')
                        self.coordinates = where
        if (self.coordinates is None):
            self.terms = self.generate(where)
            if (self.terms is not None):
                (self.condition, self.filter) = self.terms.evaluate()

    def generate(self, where):
        ' where can be a : dict,list,tuple,string '
        if (where is None):
            return None
        q = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            qkeys = ','.join(q.keys())
            msg = dedent(f'''                The passed where expression: {where}
                            contains an invalid variable reference
                            all of the variable references must be a reference to
                            an axis (e.g. 'index' or 'columns'), or a data_column
                            The currently defined references are: {qkeys}
                ''')
            raise ValueError(msg) from err

    def select(self):
        '\n        generate the selection\n        '
        if (self.condition is not None):
            return self.table.table.read_where(self.condition.format(), start=self.start, stop=self.stop)
        elif (self.coordinates is not None):
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self):
        '\n        generate the selection\n        '
        (start, stop) = (self.start, self.stop)
        nrows = self.table.nrows
        if (start is None):
            start = 0
        elif (start < 0):
            start += nrows
        if (stop is None):
            stop = nrows
        elif (stop < 0):
            stop += nrows
        if (self.condition is not None):
            return self.table.table.get_where_list(self.condition.format(), start=start, stop=stop, sort=True)
        elif (self.coordinates is not None):
            return self.coordinates
        return np.arange(start, stop)
