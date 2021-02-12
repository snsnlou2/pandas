
from abc import ABC, abstractmethod
from collections import abc
import functools
from io import StringIO
from itertools import islice
from typing import Any, Callable, Mapping, Optional, Tuple, Type, Union
import numpy as np
import pandas._libs.json as json
from pandas._libs.tslibs import iNaT
from pandas._typing import CompressionOptions, DtypeArg, FrameOrSeriesUnion, IndexLabel, JSONSerializable, StorageOptions
from pandas.errors import AbstractMethodError
from pandas.util._decorators import deprecate_kwarg, deprecate_nonkeyword_arguments, doc
from pandas.core.dtypes.common import ensure_str, is_period_dtype
from pandas import DataFrame, MultiIndex, Series, isna, notna, to_datetime
from pandas.core import generic
from pandas.core.construction import create_series_with_explicit_dtype
from pandas.core.generic import NDFrame
from pandas.core.reshape.concat import concat
from pandas.io.common import IOHandles, file_exists, get_handle, is_fsspec_url, is_url, stringify_path
from pandas.io.json._normalize import convert_to_line_delimits
from pandas.io.json._table_schema import build_table_schema, parse_table_schema
from pandas.io.parsers import validate_integer
loads = json.loads
dumps = json.dumps
TABLE_SCHEMA_VERSION = '0.20.0'

def to_json(path_or_buf, obj, orient=None, date_format='epoch', double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=True, indent=0, storage_options=None):
    if ((not index) and (orient not in ['split', 'table'])):
        raise ValueError("'index=False' is only valid when 'orient' is 'split' or 'table'")
    if (lines and (orient != 'records')):
        raise ValueError("'lines' keyword only valid when 'orient' is records")
    if ((orient == 'table') and isinstance(obj, Series)):
        obj = obj.to_frame(name=(obj.name or 'values'))
    writer: Type['Writer']
    if ((orient == 'table') and isinstance(obj, DataFrame)):
        writer = JSONTableWriter
    elif isinstance(obj, Series):
        writer = SeriesWriter
    elif isinstance(obj, DataFrame):
        writer = FrameWriter
    else:
        raise NotImplementedError("'obj' should be a Series or a DataFrame")
    s = writer(obj, orient=orient, date_format=date_format, double_precision=double_precision, ensure_ascii=force_ascii, date_unit=date_unit, default_handler=default_handler, index=index, indent=indent).write()
    if lines:
        s = convert_to_line_delimits(s)
    if (path_or_buf is not None):
        with get_handle(path_or_buf, 'wt', compression=compression, storage_options=storage_options) as handles:
            handles.handle.write(s)
    else:
        return s

class Writer(ABC):

    def __init__(self, obj, orient, date_format, double_precision, ensure_ascii, date_unit, index, default_handler=None, indent=0):
        self.obj = obj
        if (orient is None):
            orient = self._default_orient
        self.orient = orient
        self.date_format = date_format
        self.double_precision = double_precision
        self.ensure_ascii = ensure_ascii
        self.date_unit = date_unit
        self.default_handler = default_handler
        self.index = index
        self.indent = indent
        self.is_copy = None
        self._format_axes()

    def _format_axes(self):
        raise AbstractMethodError(self)

    def write(self):
        iso_dates = (self.date_format == 'iso')
        return dumps(self.obj_to_write, orient=self.orient, double_precision=self.double_precision, ensure_ascii=self.ensure_ascii, date_unit=self.date_unit, iso_dates=iso_dates, default_handler=self.default_handler, indent=self.indent)

    @property
    @abstractmethod
    def obj_to_write(self):
        'Object to write in JSON format.'
        pass

class SeriesWriter(Writer):
    _default_orient = 'index'

    @property
    def obj_to_write(self):
        if ((not self.index) and (self.orient == 'split')):
            return {'name': self.obj.name, 'data': self.obj.values}
        else:
            return self.obj

    def _format_axes(self):
        if ((not self.obj.index.is_unique) and (self.orient == 'index')):
            raise ValueError(f"Series index must be unique for orient='{self.orient}'")

class FrameWriter(Writer):
    _default_orient = 'columns'

    @property
    def obj_to_write(self):
        if ((not self.index) and (self.orient == 'split')):
            obj_to_write = self.obj.to_dict(orient='split')
            del obj_to_write['index']
        else:
            obj_to_write = self.obj
        return obj_to_write

    def _format_axes(self):
        '\n        Try to format axes if they are datelike.\n        '
        if ((not self.obj.index.is_unique) and (self.orient in ('index', 'columns'))):
            raise ValueError(f"DataFrame index must be unique for orient='{self.orient}'.")
        if ((not self.obj.columns.is_unique) and (self.orient in ('index', 'columns', 'records'))):
            raise ValueError(f"DataFrame columns must be unique for orient='{self.orient}'.")

class JSONTableWriter(FrameWriter):
    _default_orient = 'records'

    def __init__(self, obj, orient, date_format, double_precision, ensure_ascii, date_unit, index, default_handler=None, indent=0):
        "\n        Adds a `schema` attribute with the Table Schema, resets\n        the index (can't do in caller, because the schema inference needs\n        to know what the index is, forces orient to records, and forces\n        date_format to 'iso'.\n        "
        super().__init__(obj, orient, date_format, double_precision, ensure_ascii, date_unit, index, default_handler=default_handler, indent=indent)
        if (date_format != 'iso'):
            msg = f"Trying to write with `orient='table'` and `date_format='{date_format}'`. Table Schema requires dates to be formatted with `date_format='iso'`"
            raise ValueError(msg)
        self.schema = build_table_schema(obj, index=self.index)
        if ((obj.ndim == 2) and isinstance(obj.columns, MultiIndex)):
            raise NotImplementedError("orient='table' is not supported for MultiIndex columns")
        if (((obj.ndim == 1) and (obj.name in set(obj.index.names))) or len(obj.columns.intersection(obj.index.names))):
            msg = 'Overlapping names between the index and columns'
            raise ValueError(msg)
        obj = obj.copy()
        timedeltas = obj.select_dtypes(include=['timedelta']).columns
        if len(timedeltas):
            obj[timedeltas] = obj[timedeltas].applymap((lambda x: x.isoformat()))
        if is_period_dtype(obj.index.dtype):
            obj.index = obj.index.to_timestamp()
        if (not self.index):
            self.obj = obj.reset_index(drop=True)
        else:
            self.obj = obj.reset_index(drop=False)
        self.date_format = 'iso'
        self.orient = 'records'
        self.index = index

    @property
    def obj_to_write(self):
        return {'schema': self.schema, 'data': self.obj}

@doc(storage_options=generic._shared_docs['storage_options'])
@deprecate_kwarg(old_arg_name='numpy', new_arg_name=None)
@deprecate_nonkeyword_arguments(version='2.0', allowed_args=['path_or_buf'], stacklevel=3)
def read_json(path_or_buf=None, orient=None, typ='frame', dtype=None, convert_axes=None, convert_dates=True, keep_default_dates=True, numpy=False, precise_float=False, date_unit=None, encoding=None, lines=False, chunksize=None, compression='infer', nrows=None, storage_options=None):
    '\n    Convert a JSON string to pandas object.\n\n    Parameters\n    ----------\n    path_or_buf : a valid JSON str, path object or file-like object\n        Any valid string path is acceptable. The string could be a URL. Valid\n        URL schemes include http, ftp, s3, and file. For file URLs, a host is\n        expected. A local file could be:\n        ``file://localhost/path/to/table.json``.\n\n        If you want to pass in a path object, pandas accepts any\n        ``os.PathLike``.\n\n        By file-like object, we refer to objects with a ``read()`` method,\n        such as a file handle (e.g. via builtin ``open`` function)\n        or ``StringIO``.\n    orient : str\n        Indication of expected JSON string format.\n        Compatible JSON strings can be produced by ``to_json()`` with a\n        corresponding orient value.\n        The set of possible orients is:\n\n        - ``\'split\'`` : dict like\n          ``{{index -> [index], columns -> [columns], data -> [values]}}``\n        - ``\'records\'`` : list like\n          ``[{{column -> value}}, ... , {{column -> value}}]``\n        - ``\'index\'`` : dict like ``{{index -> {{column -> value}}}}``\n        - ``\'columns\'`` : dict like ``{{column -> {{index -> value}}}}``\n        - ``\'values\'`` : just the values array\n\n        The allowed and default values depend on the value\n        of the `typ` parameter.\n\n        * when ``typ == \'series\'``,\n\n          - allowed orients are ``{{\'split\',\'records\',\'index\'}}``\n          - default is ``\'index\'``\n          - The Series index must be unique for orient ``\'index\'``.\n\n        * when ``typ == \'frame\'``,\n\n          - allowed orients are ``{{\'split\',\'records\',\'index\',\n            \'columns\',\'values\', \'table\'}}``\n          - default is ``\'columns\'``\n          - The DataFrame index must be unique for orients ``\'index\'`` and\n            ``\'columns\'``.\n          - The DataFrame columns must be unique for orients ``\'index\'``,\n            ``\'columns\'``, and ``\'records\'``.\n\n    typ : {{\'frame\', \'series\'}}, default \'frame\'\n        The type of object to recover.\n\n    dtype : bool or dict, default None\n        If True, infer dtypes; if a dict of column to dtype, then use those;\n        if False, then don\'t infer dtypes at all, applies only to the data.\n\n        For all ``orient`` values except ``\'table\'``, default is True.\n\n        .. versionchanged:: 0.25.0\n\n           Not applicable for ``orient=\'table\'``.\n\n    convert_axes : bool, default None\n        Try to convert the axes to the proper dtypes.\n\n        For all ``orient`` values except ``\'table\'``, default is True.\n\n        .. versionchanged:: 0.25.0\n\n           Not applicable for ``orient=\'table\'``.\n\n    convert_dates : bool or list of str, default True\n        If True then default datelike columns may be converted (depending on\n        keep_default_dates).\n        If False, no dates will be converted.\n        If a list of column names, then those columns will be converted and\n        default datelike columns may also be converted (depending on\n        keep_default_dates).\n\n    keep_default_dates : bool, default True\n        If parsing dates (convert_dates is not False), then try to parse the\n        default datelike columns.\n        A column label is datelike if\n\n        * it ends with ``\'_at\'``,\n\n        * it ends with ``\'_time\'``,\n\n        * it begins with ``\'timestamp\'``,\n\n        * it is ``\'modified\'``, or\n\n        * it is ``\'date\'``.\n\n    numpy : bool, default False\n        Direct decoding to numpy arrays. Supports numeric data only, but\n        non-numeric column and index labels are supported. Note also that the\n        JSON ordering MUST be the same for each term if numpy=True.\n\n        .. deprecated:: 1.0.0\n\n    precise_float : bool, default False\n        Set to enable usage of higher precision (strtod) function when\n        decoding string to double values. Default (False) is to use fast but\n        less precise builtin functionality.\n\n    date_unit : str, default None\n        The timestamp unit to detect if converting dates. The default behaviour\n        is to try and detect the correct precision, but if this is not desired\n        then pass one of \'s\', \'ms\', \'us\' or \'ns\' to force parsing only seconds,\n        milliseconds, microseconds or nanoseconds respectively.\n\n    encoding : str, default is \'utf-8\'\n        The encoding to use to decode py3 bytes.\n\n    lines : bool, default False\n        Read the file as a json object per line.\n\n    chunksize : int, optional\n        Return JsonReader object for iteration.\n        See the `line-delimited json docs\n        <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json>`_\n        for more information on ``chunksize``.\n        This can only be passed if `lines=True`.\n        If this is None, the file will be read into memory all at once.\n\n        .. versionchanged:: 1.2\n\n           ``JsonReader`` is a context manager.\n\n    compression : {{\'infer\', \'gzip\', \'bz2\', \'zip\', \'xz\', None}}, default \'infer\'\n        For on-the-fly decompression of on-disk data. If \'infer\', then use\n        gzip, bz2, zip or xz if path_or_buf is a string ending in\n        \'.gz\', \'.bz2\', \'.zip\', or \'xz\', respectively, and no decompression\n        otherwise. If using \'zip\', the ZIP file must contain only one data\n        file to be read in. Set to None for no decompression.\n\n    nrows : int, optional\n        The number of lines from the line-delimited jsonfile that has to be read.\n        This can only be passed if `lines=True`.\n        If this is None, all the rows will be returned.\n\n        .. versionadded:: 1.1\n\n    {storage_options}\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    Series or DataFrame\n        The type returned depends on the value of `typ`.\n\n    See Also\n    --------\n    DataFrame.to_json : Convert a DataFrame to a JSON string.\n    Series.to_json : Convert a Series to a JSON string.\n\n    Notes\n    -----\n    Specific to ``orient=\'table\'``, if a :class:`DataFrame` with a literal\n    :class:`Index` name of `index` gets written with :func:`to_json`, the\n    subsequent read operation will incorrectly set the :class:`Index` name to\n    ``None``. This is because `index` is also used by :func:`DataFrame.to_json`\n    to denote a missing :class:`Index` name, and the subsequent\n    :func:`read_json` operation cannot distinguish between the two. The same\n    limitation is encountered with a :class:`MultiIndex` and any names\n    beginning with ``\'level_\'``.\n\n    Examples\n    --------\n    >>> df = pd.DataFrame([[\'a\', \'b\'], [\'c\', \'d\']],\n    ...                   index=[\'row 1\', \'row 2\'],\n    ...                   columns=[\'col 1\', \'col 2\'])\n\n    Encoding/decoding a Dataframe using ``\'split\'`` formatted JSON:\n\n    >>> df.to_json(orient=\'split\')\n    \'{{"columns":["col 1","col 2"],\n      "index":["row 1","row 2"],\n      "data":[["a","b"],["c","d"]]}}\'\n    >>> pd.read_json(_, orient=\'split\')\n          col 1 col 2\n    row 1     a     b\n    row 2     c     d\n\n    Encoding/decoding a Dataframe using ``\'index\'`` formatted JSON:\n\n    >>> df.to_json(orient=\'index\')\n    \'{{"row 1":{{"col 1":"a","col 2":"b"}},"row 2":{{"col 1":"c","col 2":"d"}}}}\'\n    >>> pd.read_json(_, orient=\'index\')\n          col 1 col 2\n    row 1     a     b\n    row 2     c     d\n\n    Encoding/decoding a Dataframe using ``\'records\'`` formatted JSON.\n    Note that index labels are not preserved with this encoding.\n\n    >>> df.to_json(orient=\'records\')\n    \'[{{"col 1":"a","col 2":"b"}},{{"col 1":"c","col 2":"d"}}]\'\n    >>> pd.read_json(_, orient=\'records\')\n      col 1 col 2\n    0     a     b\n    1     c     d\n\n    Encoding with Table Schema\n\n    >>> df.to_json(orient=\'table\')\n    \'{{"schema": {{"fields": [{{"name": "index", "type": "string"}},\n                            {{"name": "col 1", "type": "string"}},\n                            {{"name": "col 2", "type": "string"}}],\n                    "primaryKey": "index",\n                    "pandas_version": "0.20.0"}},\n        "data": [{{"index": "row 1", "col 1": "a", "col 2": "b"}},\n                {{"index": "row 2", "col 1": "c", "col 2": "d"}}]}}\'\n    '
    if ((orient == 'table') and dtype):
        raise ValueError("cannot pass both dtype and orient='table'")
    if ((orient == 'table') and convert_axes):
        raise ValueError("cannot pass both convert_axes and orient='table'")
    if ((dtype is None) and (orient != 'table')):
        dtype = True
    if ((convert_axes is None) and (orient != 'table')):
        convert_axes = True
    json_reader = JsonReader(path_or_buf, orient=orient, typ=typ, dtype=dtype, convert_axes=convert_axes, convert_dates=convert_dates, keep_default_dates=keep_default_dates, numpy=numpy, precise_float=precise_float, date_unit=date_unit, encoding=encoding, lines=lines, chunksize=chunksize, compression=compression, nrows=nrows, storage_options=storage_options)
    if chunksize:
        return json_reader
    with json_reader:
        return json_reader.read()

class JsonReader(abc.Iterator):
    '\n    JsonReader provides an interface for reading in a JSON file.\n\n    If initialized with ``lines=True`` and ``chunksize``, can be iterated over\n    ``chunksize`` lines at a time. Otherwise, calling ``read`` reads in the\n    whole document.\n    '

    def __init__(self, filepath_or_buffer, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, numpy, precise_float, date_unit, encoding, lines, chunksize, compression, nrows, storage_options=None):
        self.orient = orient
        self.typ = typ
        self.dtype = dtype
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.keep_default_dates = keep_default_dates
        self.numpy = numpy
        self.precise_float = precise_float
        self.date_unit = date_unit
        self.encoding = encoding
        self.compression = compression
        self.storage_options = storage_options
        self.lines = lines
        self.chunksize = chunksize
        self.nrows_seen = 0
        self.nrows = nrows
        self.handles: Optional[IOHandles] = None
        if (self.chunksize is not None):
            self.chunksize = validate_integer('chunksize', self.chunksize, 1)
            if (not self.lines):
                raise ValueError('chunksize can only be passed if lines=True')
        if (self.nrows is not None):
            self.nrows = validate_integer('nrows', self.nrows, 0)
            if (not self.lines):
                raise ValueError('nrows can only be passed if lines=True')
        data = self._get_data_from_filepath(filepath_or_buffer)
        self.data = self._preprocess_data(data)

    def _preprocess_data(self, data):
        '\n        At this point, the data either has a `read` attribute (e.g. a file\n        object or a StringIO) or is a string that is a JSON document.\n\n        If self.chunksize, we prepare the data for the `__next__` method.\n        Otherwise, we read it into memory for the `read` method.\n        '
        if (hasattr(data, 'read') and (not (self.chunksize or self.nrows))):
            data = data.read()
            self.close()
        if ((not hasattr(data, 'read')) and (self.chunksize or self.nrows)):
            data = StringIO(data)
        return data

    def _get_data_from_filepath(self, filepath_or_buffer):
        '\n        The function read_json accepts three input types:\n            1. filepath (string-like)\n            2. file-like object (e.g. open file object, StringIO)\n            3. JSON string\n\n        This method turns (1) into (2) to simplify the rest of the processing.\n        It returns input types (2) and (3) unchanged.\n        '
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        if ((not isinstance(filepath_or_buffer, str)) or is_url(filepath_or_buffer) or is_fsspec_url(filepath_or_buffer) or file_exists(filepath_or_buffer)):
            self.handles = get_handle(filepath_or_buffer, 'r', encoding=self.encoding, compression=self.compression, storage_options=self.storage_options)
            filepath_or_buffer = self.handles.handle
        return filepath_or_buffer

    def _combine_lines(self, lines):
        '\n        Combines a list of JSON objects into one JSON object.\n        '
        return f"[{','.join((line for line in (line.strip() for line in lines) if line))}]"

    def read(self):
        '\n        Read the whole JSON input into a pandas object.\n        '
        if self.lines:
            if self.chunksize:
                obj = concat(self)
            elif self.nrows:
                lines = list(islice(self.data, self.nrows))
                lines_json = self._combine_lines(lines)
                obj = self._get_object_parser(lines_json)
            else:
                data = ensure_str(self.data)
                data_lines = data.split('\n')
                obj = self._get_object_parser(self._combine_lines(data_lines))
        else:
            obj = self._get_object_parser(self.data)
        self.close()
        return obj

    def _get_object_parser(self, json):
        '\n        Parses a json document into a pandas object.\n        '
        typ = self.typ
        dtype = self.dtype
        kwargs = {'orient': self.orient, 'dtype': self.dtype, 'convert_axes': self.convert_axes, 'convert_dates': self.convert_dates, 'keep_default_dates': self.keep_default_dates, 'numpy': self.numpy, 'precise_float': self.precise_float, 'date_unit': self.date_unit}
        obj = None
        if (typ == 'frame'):
            obj = FrameParser(json, **kwargs).parse()
        if ((typ == 'series') or (obj is None)):
            if (not isinstance(dtype, bool)):
                kwargs['dtype'] = dtype
            obj = SeriesParser(json, **kwargs).parse()
        return obj

    def close(self):
        '\n        If we opened a stream earlier, in _get_data_from_filepath, we should\n        close it.\n\n        If an open stream or file was passed, we leave it open.\n        '
        if (self.handles is not None):
            self.handles.close()

    def __next__(self):
        if self.nrows:
            if (self.nrows_seen >= self.nrows):
                self.close()
                raise StopIteration
        lines = list(islice(self.data, self.chunksize))
        if lines:
            lines_json = self._combine_lines(lines)
            obj = self._get_object_parser(lines_json)
            obj.index = range(self.nrows_seen, (self.nrows_seen + len(obj)))
            self.nrows_seen += len(obj)
            return obj
        self.close()
        raise StopIteration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class Parser():
    _STAMP_UNITS = ('s', 'ms', 'us', 'ns')
    _MIN_STAMPS = {'s': 31536000, 'ms': 31536000000, 'us': 31536000000000, 'ns': 31536000000000000}

    def __init__(self, json, orient, dtype=None, convert_axes=True, convert_dates=True, keep_default_dates=False, numpy=False, precise_float=False, date_unit=None):
        self.json = json
        if (orient is None):
            orient = self._default_orient
        self.orient = orient
        self.dtype = dtype
        if (orient == 'split'):
            numpy = False
        if (date_unit is not None):
            date_unit = date_unit.lower()
            if (date_unit not in self._STAMP_UNITS):
                raise ValueError(f'date_unit must be one of {self._STAMP_UNITS}')
            self.min_stamp = self._MIN_STAMPS[date_unit]
        else:
            self.min_stamp = self._MIN_STAMPS['s']
        self.numpy = numpy
        self.precise_float = precise_float
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.date_unit = date_unit
        self.keep_default_dates = keep_default_dates
        self.obj: Optional[FrameOrSeriesUnion] = None

    def check_keys_split(self, decoded):
        "\n        Checks that dict has only the appropriate keys for orient='split'.\n        "
        bad_keys = set(decoded.keys()).difference(set(self._split_keys))
        if bad_keys:
            bad_keys_joined = ', '.join(bad_keys)
            raise ValueError(f'JSON data had unexpected key(s): {bad_keys_joined}')

    def parse(self):
        numpy = self.numpy
        if numpy:
            self._parse_numpy()
        else:
            self._parse_no_numpy()
        if (self.obj is None):
            return None
        if self.convert_axes:
            self._convert_axes()
        self._try_convert_types()
        return self.obj

    def _parse_numpy(self):
        raise AbstractMethodError(self)

    def _parse_no_numpy(self):
        raise AbstractMethodError(self)

    def _convert_axes(self):
        '\n        Try to convert axes.\n        '
        obj = self.obj
        assert (obj is not None)
        for axis_name in obj._AXIS_ORDERS:
            (new_axis, result) = self._try_convert_data(name=axis_name, data=obj._get_axis(axis_name), use_dtypes=False, convert_dates=True)
            if result:
                setattr(self.obj, axis_name, new_axis)

    def _try_convert_types(self):
        raise AbstractMethodError(self)

    def _try_convert_data(self, name, data, use_dtypes=True, convert_dates=True):
        '\n        Try to parse a ndarray like into a column by inferring dtype.\n        '
        if use_dtypes:
            if (not self.dtype):
                if all(notna(data)):
                    return (data, False)
                return (data.fillna(np.nan), True)
            elif (self.dtype is True):
                pass
            else:
                dtype = (self.dtype.get(name) if isinstance(self.dtype, dict) else self.dtype)
                if (dtype is not None):
                    try:
                        dtype = np.dtype(dtype)
                        return (data.astype(dtype), True)
                    except (TypeError, ValueError):
                        return (data, False)
        if convert_dates:
            (new_data, result) = self._try_convert_to_date(data)
            if result:
                return (new_data, True)
        result = False
        if (data.dtype == 'object'):
            try:
                data = data.astype('float64')
                result = True
            except (TypeError, ValueError):
                pass
        if (data.dtype.kind == 'f'):
            if (data.dtype != 'float64'):
                try:
                    data = data.astype('float64')
                    result = True
                except (TypeError, ValueError):
                    pass
        if (len(data) and ((data.dtype == 'float') or (data.dtype == 'object'))):
            try:
                new_data = data.astype('int64')
                if (new_data == data).all():
                    data = new_data
                    result = True
            except (TypeError, ValueError, OverflowError):
                pass
        if (data.dtype == 'int'):
            try:
                data = data.astype('int64')
                result = True
            except (TypeError, ValueError):
                pass
        return (data, result)

    def _try_convert_to_date(self, data):
        '\n        Try to parse a ndarray like into a date column.\n\n        Try to coerce object in epoch/iso formats and integer/float in epoch\n        formats. Return a boolean if parsing was successful.\n        '
        if (not len(data)):
            return (data, False)
        new_data = data
        if (new_data.dtype == 'object'):
            try:
                new_data = data.astype('int64')
            except (TypeError, ValueError, OverflowError):
                pass
        if issubclass(new_data.dtype.type, np.number):
            in_range = ((isna(new_data._values) | (new_data > self.min_stamp)) | (new_data._values == iNaT))
            if (not in_range.all()):
                return (data, False)
        date_units = ((self.date_unit,) if self.date_unit else self._STAMP_UNITS)
        for date_unit in date_units:
            try:
                new_data = to_datetime(new_data, errors='raise', unit=date_unit)
            except (ValueError, OverflowError, TypeError):
                continue
            return (new_data, True)
        return (data, False)

    def _try_convert_dates(self):
        raise AbstractMethodError(self)

class SeriesParser(Parser):
    _default_orient = 'index'
    _split_keys = ('name', 'index', 'data')

    def _parse_no_numpy(self):
        data = loads(self.json, precise_float=self.precise_float)
        if (self.orient == 'split'):
            decoded = {str(k): v for (k, v) in data.items()}
            self.check_keys_split(decoded)
            self.obj = create_series_with_explicit_dtype(**decoded)
        else:
            self.obj = create_series_with_explicit_dtype(data, dtype_if_empty=object)

    def _parse_numpy(self):
        load_kwargs = {'dtype': None, 'numpy': True, 'precise_float': self.precise_float}
        if (self.orient in ['columns', 'index']):
            load_kwargs['labelled'] = True
        loads_ = functools.partial(loads, **load_kwargs)
        data = loads_(self.json)
        if (self.orient == 'split'):
            decoded = {str(k): v for (k, v) in data.items()}
            self.check_keys_split(decoded)
            self.obj = create_series_with_explicit_dtype(**decoded)
        elif (self.orient in ['columns', 'index']):
            self.obj = create_series_with_explicit_dtype(*data, dtype_if_empty=object)
        else:
            self.obj = create_series_with_explicit_dtype(data, dtype_if_empty=object)

    def _try_convert_types(self):
        if (self.obj is None):
            return
        (obj, result) = self._try_convert_data('data', self.obj, convert_dates=self.convert_dates)
        if result:
            self.obj = obj

class FrameParser(Parser):
    _default_orient = 'columns'
    _split_keys = ('columns', 'index', 'data')

    def _parse_numpy(self):
        json = self.json
        orient = self.orient
        if (orient == 'columns'):
            args = loads(json, dtype=None, numpy=True, labelled=True, precise_float=self.precise_float)
            if len(args):
                args = (args[0].T, args[2], args[1])
            self.obj = DataFrame(*args)
        elif (orient == 'split'):
            decoded = loads(json, dtype=None, numpy=True, precise_float=self.precise_float)
            decoded = {str(k): v for (k, v) in decoded.items()}
            self.check_keys_split(decoded)
            self.obj = DataFrame(**decoded)
        elif (orient == 'values'):
            self.obj = DataFrame(loads(json, dtype=None, numpy=True, precise_float=self.precise_float))
        else:
            self.obj = DataFrame(*loads(json, dtype=None, numpy=True, labelled=True, precise_float=self.precise_float))

    def _parse_no_numpy(self):
        json = self.json
        orient = self.orient
        if (orient == 'columns'):
            self.obj = DataFrame(loads(json, precise_float=self.precise_float), dtype=None)
        elif (orient == 'split'):
            decoded = {str(k): v for (k, v) in loads(json, precise_float=self.precise_float).items()}
            self.check_keys_split(decoded)
            self.obj = DataFrame(dtype=None, **decoded)
        elif (orient == 'index'):
            self.obj = DataFrame.from_dict(loads(json, precise_float=self.precise_float), dtype=None, orient='index')
        elif (orient == 'table'):
            self.obj = parse_table_schema(json, precise_float=self.precise_float)
        else:
            self.obj = DataFrame(loads(json, precise_float=self.precise_float), dtype=None)

    def _process_converter(self, f, filt=None):
        '\n        Take a conversion function and possibly recreate the frame.\n        '
        if (filt is None):
            filt = (lambda col, c: True)
        obj = self.obj
        assert (obj is not None)
        needs_new_obj = False
        new_obj = {}
        for (i, (col, c)) in enumerate(obj.items()):
            if filt(col, c):
                (new_data, result) = f(col, c)
                if result:
                    c = new_data
                    needs_new_obj = True
            new_obj[i] = c
        if needs_new_obj:
            new_frame = DataFrame(new_obj, index=obj.index)
            new_frame.columns = obj.columns
            self.obj = new_frame

    def _try_convert_types(self):
        if (self.obj is None):
            return
        if self.convert_dates:
            self._try_convert_dates()
        self._process_converter((lambda col, c: self._try_convert_data(col, c, convert_dates=False)))

    def _try_convert_dates(self):
        if (self.obj is None):
            return
        convert_dates = self.convert_dates
        if (convert_dates is True):
            convert_dates = []
        convert_dates = set(convert_dates)

        def is_ok(col) -> bool:
            '\n            Return if this col is ok to try for a date parse.\n            '
            if (not isinstance(col, str)):
                return False
            col_lower = col.lower()
            if (col_lower.endswith('_at') or col_lower.endswith('_time') or (col_lower == 'modified') or (col_lower == 'date') or (col_lower == 'datetime') or col_lower.startswith('timestamp')):
                return True
            return False
        self._process_converter((lambda col, c: self._try_convert_to_date(c)), (lambda col, c: ((self.keep_default_dates and is_ok(col)) or (col in convert_dates))))
