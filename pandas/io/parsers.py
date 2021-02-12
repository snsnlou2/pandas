
'\nModule contains tools for processing files into DataFrames or other objects\n'
from collections import abc, defaultdict
import csv
import datetime
from io import StringIO
import itertools
import re
import sys
from textwrap import fill
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Type, cast
import warnings
import numpy as np
import pandas._libs.lib as lib
import pandas._libs.ops as libops
import pandas._libs.parsers as parsers
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas._typing import DtypeArg, FilePathOrBuffer, StorageOptions, Union
from pandas.errors import AbstractMethodError, EmptyDataError, ParserError, ParserWarning
from pandas.util._decorators import Appender
from pandas.core.dtypes.cast import astype_nansafe
from pandas.core.dtypes.common import ensure_object, ensure_str, is_bool_dtype, is_categorical_dtype, is_dict_like, is_dtype_equal, is_extension_array_dtype, is_file_like, is_float, is_integer, is_integer_dtype, is_list_like, is_object_dtype, is_scalar, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.missing import isna
from pandas.core import algorithms, generic
from pandas.core.arrays import Categorical
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import Index, MultiIndex, RangeIndex, ensure_index_from_sequences
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import IOHandles, get_handle, validate_header_arg
from pandas.io.date_converters import generic_parser
_BOM = '\ufeff'
_doc_read_csv_and_table = (("\n{summary}\n\nAlso supports optionally iterating or breaking of the file\ninto chunks.\n\nAdditional help can be found in the online docs for\n`IO Tools <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.\n\nParameters\n----------\nfilepath_or_buffer : str, path object or file-like object\n    Any valid string path is acceptable. The string could be a URL. Valid\n    URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is\n    expected. A local file could be: file://localhost/path/to/table.csv.\n\n    If you want to pass in a path object, pandas accepts any ``os.PathLike``.\n\n    By file-like object, we refer to objects with a ``read()`` method, such as\n    a file handle (e.g. via builtin ``open`` function) or ``StringIO``.\nsep : str, default {_default_sep}\n    Delimiter to use. If sep is None, the C engine cannot automatically detect\n    the separator, but the Python parsing engine can, meaning the latter will\n    be used and automatically detect the separator by Python's builtin sniffer\n    tool, ``csv.Sniffer``. In addition, separators longer than 1 character and\n    different from ``'\\s+'`` will be interpreted as regular expressions and\n    will also force the use of the Python parsing engine. Note that regex\n    delimiters are prone to ignoring quoted data. Regex example: ``'\\r\\t'``.\ndelimiter : str, default ``None``\n    Alias for sep.\nheader : int, list of int, default 'infer'\n    Row number(s) to use as the column names, and the start of the\n    data.  Default behavior is to infer the column names: if no names\n    are passed the behavior is identical to ``header=0`` and column\n    names are inferred from the first line of the file, if column\n    names are passed explicitly then the behavior is identical to\n    ``header=None``. Explicitly pass ``header=0`` to be able to\n    replace existing names. The header can be a list of integers that\n    specify row locations for a multi-index on the columns\n    e.g. [0,1,3]. Intervening rows that are not specified will be\n    skipped (e.g. 2 in this example is skipped). Note that this\n    parameter ignores commented lines and empty lines if\n    ``skip_blank_lines=True``, so ``header=0`` denotes the first line of\n    data rather than the first line of the file.\nnames : array-like, optional\n    List of column names to use. If the file contains a header row,\n    then you should explicitly pass ``header=0`` to override the column names.\n    Duplicates in this list are not allowed.\nindex_col : int, str, sequence of int / str, or False, default ``None``\n  Column(s) to use as the row labels of the ``DataFrame``, either given as\n  string name or column index. If a sequence of int / str is given, a\n  MultiIndex is used.\n\n  Note: ``index_col=False`` can be used to force pandas to *not* use the first\n  column as the index, e.g. when you have a malformed file with delimiters at\n  the end of each line.\nusecols : list-like or callable, optional\n    Return a subset of the columns. If list-like, all elements must either\n    be positional (i.e. integer indices into the document columns) or strings\n    that correspond to column names provided either by the user in `names` or\n    inferred from the document header row(s). For example, a valid list-like\n    `usecols` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.\n    Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.\n    To instantiate a DataFrame from ``data`` with element order preserved use\n    ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]`` for columns\n    in ``['foo', 'bar']`` order or\n    ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``\n    for ``['bar', 'foo']`` order.\n\n    If callable, the callable function will be evaluated against the column\n    names, returning names where the callable function evaluates to True. An\n    example of a valid callable argument would be ``lambda x: x.upper() in\n    ['AAA', 'BBB', 'DDD']``. Using this parameter results in much faster\n    parsing time and lower memory usage.\nsqueeze : bool, default False\n    If the parsed data only contains one column then return a Series.\nprefix : str, optional\n    Prefix to add to column numbers when no header, e.g. 'X' for X0, X1, ...\nmangle_dupe_cols : bool, default True\n    Duplicate columns will be specified as 'X', 'X.1', ...'X.N', rather than\n    'X'...'X'. Passing in False will cause data to be overwritten if there\n    are duplicate names in the columns.\ndtype : Type name or dict of column -> type, optional\n    Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32,\n    'c': 'Int64'}}\n    Use `str` or `object` together with suitable `na_values` settings\n    to preserve and not interpret dtype.\n    If converters are specified, they will be applied INSTEAD\n    of dtype conversion.\nengine : {{'c', 'python'}}, optional\n    Parser engine to use. The C engine is faster while the python engine is\n    currently more feature-complete.\nconverters : dict, optional\n    Dict of functions for converting values in certain columns. Keys can either\n    be integers or column labels.\ntrue_values : list, optional\n    Values to consider as True.\nfalse_values : list, optional\n    Values to consider as False.\nskipinitialspace : bool, default False\n    Skip spaces after delimiter.\nskiprows : list-like, int or callable, optional\n    Line numbers to skip (0-indexed) or number of lines to skip (int)\n    at the start of the file.\n\n    If callable, the callable function will be evaluated against the row\n    indices, returning True if the row should be skipped and False otherwise.\n    An example of a valid callable argument would be ``lambda x: x in [0, 2]``.\nskipfooter : int, default 0\n    Number of lines at bottom of file to skip (Unsupported with engine='c').\nnrows : int, optional\n    Number of rows of file to read. Useful for reading pieces of large files.\nna_values : scalar, str, list-like, or dict, optional\n    Additional strings to recognize as NA/NaN. If dict passed, specific\n    per-column NA values.  By default the following values are interpreted as\n    NaN: '" + fill("', '".join(sorted(STR_NA_VALUES)), 70, subsequent_indent='    ')) + '\'.\nkeep_default_na : bool, default True\n    Whether or not to include the default NaN values when parsing the data.\n    Depending on whether `na_values` is passed in, the behavior is as follows:\n\n    * If `keep_default_na` is True, and `na_values` are specified, `na_values`\n      is appended to the default NaN values used for parsing.\n    * If `keep_default_na` is True, and `na_values` are not specified, only\n      the default NaN values are used for parsing.\n    * If `keep_default_na` is False, and `na_values` are specified, only\n      the NaN values specified `na_values` are used for parsing.\n    * If `keep_default_na` is False, and `na_values` are not specified, no\n      strings will be parsed as NaN.\n\n    Note that if `na_filter` is passed in as False, the `keep_default_na` and\n    `na_values` parameters will be ignored.\nna_filter : bool, default True\n    Detect missing value markers (empty strings and the value of na_values). In\n    data without any NAs, passing na_filter=False can improve the performance\n    of reading a large file.\nverbose : bool, default False\n    Indicate number of NA values placed in non-numeric columns.\nskip_blank_lines : bool, default True\n    If True, skip over blank lines rather than interpreting as NaN values.\nparse_dates : bool or list of int or names or list of lists or dict, default False\n    The behavior is as follows:\n\n    * boolean. If True -> try parsing the index.\n    * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3\n      each as a separate date column.\n    * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as\n      a single date column.\n    * dict, e.g. {{\'foo\' : [1, 3]}} -> parse columns 1, 3 as date and call\n      result \'foo\'\n\n    If a column or index cannot be represented as an array of datetimes,\n    say because of an unparsable value or a mixture of timezones, the column\n    or index will be returned unaltered as an object data type. For\n    non-standard datetime parsing, use ``pd.to_datetime`` after\n    ``pd.read_csv``. To parse an index or column with a mixture of timezones,\n    specify ``date_parser`` to be a partially-applied\n    :func:`pandas.to_datetime` with ``utc=True``. See\n    :ref:`io.csv.mixed_timezones` for more.\n\n    Note: A fast-path exists for iso8601-formatted dates.\ninfer_datetime_format : bool, default False\n    If True and `parse_dates` is enabled, pandas will attempt to infer the\n    format of the datetime strings in the columns, and if it can be inferred,\n    switch to a faster method of parsing them. In some cases this can increase\n    the parsing speed by 5-10x.\nkeep_date_col : bool, default False\n    If True and `parse_dates` specifies combining multiple columns then\n    keep the original columns.\ndate_parser : function, optional\n    Function to use for converting a sequence of string columns to an array of\n    datetime instances. The default uses ``dateutil.parser.parser`` to do the\n    conversion. Pandas will try to call `date_parser` in three different ways,\n    advancing to the next if an exception occurs: 1) Pass one or more arrays\n    (as defined by `parse_dates`) as arguments; 2) concatenate (row-wise) the\n    string values from the columns defined by `parse_dates` into a single array\n    and pass that; and 3) call `date_parser` once for each row using one or\n    more strings (corresponding to the columns defined by `parse_dates`) as\n    arguments.\ndayfirst : bool, default False\n    DD/MM format dates, international and European format.\ncache_dates : bool, default True\n    If True, use a cache of unique, converted dates to apply the datetime\n    conversion. May produce significant speed-up when parsing duplicate\n    date strings, especially ones with timezone offsets.\n\n    .. versionadded:: 0.25.0\niterator : bool, default False\n    Return TextFileReader object for iteration or getting chunks with\n    ``get_chunk()``.\n\n    .. versionchanged:: 1.2\n\n       ``TextFileReader`` is a context manager.\nchunksize : int, optional\n    Return TextFileReader object for iteration.\n    See the `IO Tools docs\n    <https://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking>`_\n    for more information on ``iterator`` and ``chunksize``.\n\n    .. versionchanged:: 1.2\n\n       ``TextFileReader`` is a context manager.\ncompression : {{\'infer\', \'gzip\', \'bz2\', \'zip\', \'xz\', None}}, default \'infer\'\n    For on-the-fly decompression of on-disk data. If \'infer\' and\n    `filepath_or_buffer` is path-like, then detect compression from the\n    following extensions: \'.gz\', \'.bz2\', \'.zip\', or \'.xz\' (otherwise no\n    decompression). If using \'zip\', the ZIP file must contain only one data\n    file to be read in. Set to None for no decompression.\nthousands : str, optional\n    Thousands separator.\ndecimal : str, default \'.\'\n    Character to recognize as decimal point (e.g. use \',\' for European data).\nlineterminator : str (length 1), optional\n    Character to break file into lines. Only valid with C parser.\nquotechar : str (length 1), optional\n    The character used to denote the start and end of a quoted item. Quoted\n    items can include the delimiter and it will be ignored.\nquoting : int or csv.QUOTE_* instance, default 0\n    Control field quoting behavior per ``csv.QUOTE_*`` constants. Use one of\n    QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or QUOTE_NONE (3).\ndoublequote : bool, default ``True``\n   When quotechar is specified and quoting is not ``QUOTE_NONE``, indicate\n   whether or not to interpret two consecutive quotechar elements INSIDE a\n   field as a single ``quotechar`` element.\nescapechar : str (length 1), optional\n    One-character string used to escape other characters.\ncomment : str, optional\n    Indicates remainder of line should not be parsed. If found at the beginning\n    of a line, the line will be ignored altogether. This parameter must be a\n    single character. Like empty lines (as long as ``skip_blank_lines=True``),\n    fully commented lines are ignored by the parameter `header` but not by\n    `skiprows`. For example, if ``comment=\'#\'``, parsing\n    ``#empty\\na,b,c\\n1,2,3`` with ``header=0`` will result in \'a,b,c\' being\n    treated as the header.\nencoding : str, optional\n    Encoding to use for UTF when reading/writing (ex. \'utf-8\'). `List of Python\n    standard encodings\n    <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .\ndialect : str or csv.Dialect, optional\n    If provided, this parameter will override values (default or not) for the\n    following parameters: `delimiter`, `doublequote`, `escapechar`,\n    `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to\n    override values, a ParserWarning will be issued. See csv.Dialect\n    documentation for more details.\nerror_bad_lines : bool, default True\n    Lines with too many fields (e.g. a csv line with too many commas) will by\n    default cause an exception to be raised, and no DataFrame will be returned.\n    If False, then these "bad lines" will dropped from the DataFrame that is\n    returned.\nwarn_bad_lines : bool, default True\n    If error_bad_lines is False, and warn_bad_lines is True, a warning for each\n    "bad line" will be output.\ndelim_whitespace : bool, default False\n    Specifies whether or not whitespace (e.g. ``\' \'`` or ``\'\t\'``) will be\n    used as the sep. Equivalent to setting ``sep=\'\\s+\'``. If this option\n    is set to True, nothing should be passed in for the ``delimiter``\n    parameter.\nlow_memory : bool, default True\n    Internally process the file in chunks, resulting in lower memory use\n    while parsing, but possibly mixed type inference.  To ensure no mixed\n    types either set False, or specify the type with the `dtype` parameter.\n    Note that the entire file is read into a single DataFrame regardless,\n    use the `chunksize` or `iterator` parameter to return the data in chunks.\n    (Only valid with C parser).\nmemory_map : bool, default False\n    If a filepath is provided for `filepath_or_buffer`, map the file object\n    directly onto memory and access the data directly from there. Using this\n    option can improve performance because there is no longer any I/O overhead.\nfloat_precision : str, optional\n    Specifies which converter the C engine should use for floating-point\n    values. The options are ``None`` or \'high\' for the ordinary converter,\n    \'legacy\' for the original lower precision pandas converter, and\n    \'round_trip\' for the round-trip converter.\n\n    .. versionchanged:: 1.2\n\n{storage_options}\n\n    .. versionadded:: 1.2\n\nReturns\n-------\nDataFrame or TextParser\n    A comma-separated values (csv) file is returned as two-dimensional\n    data structure with labeled axes.\n\nSee Also\n--------\nDataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.\nread_csv : Read a comma-separated values (csv) file into DataFrame.\nread_fwf : Read a table of fixed-width formatted lines into DataFrame.\n\nExamples\n--------\n>>> pd.{func_name}(\'data.csv\')  # doctest: +SKIP\n')

def validate_integer(name, val, min_val=0):
    "\n    Checks whether the 'name' parameter for parsing is either\n    an integer OR float that can SAFELY be cast to an integer\n    without losing accuracy. Raises a ValueError if that is\n    not the case.\n\n    Parameters\n    ----------\n    name : string\n        Parameter name (used for error reporting)\n    val : int or float\n        The value to check\n    min_val : int\n        Minimum allowed value (val < min_val will result in a ValueError)\n    "
    msg = f"'{name:s}' must be an integer >={min_val:d}"
    if (val is not None):
        if is_float(val):
            if (int(val) != val):
                raise ValueError(msg)
            val = int(val)
        elif (not (is_integer(val) and (val >= min_val))):
            raise ValueError(msg)
    return val

def _validate_names(names):
    '\n    Raise ValueError if the `names` parameter contains duplicates or has an\n    invalid data type.\n\n    Parameters\n    ----------\n    names : array-like or None\n        An array containing a list of the names used for the output DataFrame.\n\n    Raises\n    ------\n    ValueError\n        If names are not unique or are not ordered (e.g. set).\n    '
    if (names is not None):
        if (len(names) != len(set(names))):
            raise ValueError('Duplicate names are not allowed.')
        if (not (is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView))):
            raise ValueError('Names should be an ordered collection.')

def _read(filepath_or_buffer, kwds):
    'Generic reader of line files.'
    if (kwds.get('date_parser', None) is not None):
        if isinstance(kwds['parse_dates'], bool):
            kwds['parse_dates'] = True
    iterator = kwds.get('iterator', False)
    chunksize = validate_integer('chunksize', kwds.get('chunksize', None), 1)
    nrows = kwds.get('nrows', None)
    _validate_names(kwds.get('names', None))
    parser = TextFileReader(filepath_or_buffer, **kwds)
    if (chunksize or iterator):
        return parser
    with parser:
        return parser.read(nrows)
_parser_defaults = {'delimiter': None, 'escapechar': None, 'quotechar': '"', 'quoting': csv.QUOTE_MINIMAL, 'doublequote': True, 'skipinitialspace': False, 'lineterminator': None, 'header': 'infer', 'index_col': None, 'names': None, 'prefix': None, 'skiprows': None, 'skipfooter': 0, 'nrows': None, 'na_values': None, 'keep_default_na': True, 'true_values': None, 'false_values': None, 'converters': None, 'dtype': None, 'cache_dates': True, 'thousands': None, 'comment': None, 'decimal': '.', 'parse_dates': False, 'keep_date_col': False, 'dayfirst': False, 'date_parser': None, 'usecols': None, 'chunksize': None, 'verbose': False, 'encoding': None, 'squeeze': False, 'compression': None, 'mangle_dupe_cols': True, 'infer_datetime_format': False, 'skip_blank_lines': True}
_c_parser_defaults = {'delim_whitespace': False, 'na_filter': True, 'low_memory': True, 'memory_map': False, 'error_bad_lines': True, 'warn_bad_lines': True, 'float_precision': None}
_fwf_defaults = {'colspecs': 'infer', 'infer_nrows': 100, 'widths': None}
_c_unsupported = {'skipfooter'}
_python_unsupported = {'low_memory', 'float_precision'}
_deprecated_defaults = {}
_deprecated_args = set()

@Appender(_doc_read_csv_and_table.format(func_name='read_csv', summary='Read a comma-separated values (csv) file into DataFrame.', _default_sep="','", storage_options=generic._shared_docs['storage_options']))
def read_csv(filepath_or_buffer, sep=lib.no_default, delimiter=None, header='infer', names=None, index_col=None, usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None, quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True, escapechar=None, comment=None, encoding=None, dialect=None, error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False, low_memory=_c_parser_defaults['low_memory'], memory_map=False, float_precision=None, storage_options=None):
    kwds = locals()
    del kwds['filepath_or_buffer']
    del kwds['sep']
    kwds_defaults = _refine_defaults_read(dialect, delimiter, delim_whitespace, engine, sep, defaults={'delimiter': ','})
    kwds.update(kwds_defaults)
    return _read(filepath_or_buffer, kwds)

@Appender(_doc_read_csv_and_table.format(func_name='read_table', summary='Read general delimited file into DataFrame.', _default_sep="'\\\\t' (tab-stop)", storage_options=generic._shared_docs['storage_options']))
def read_table(filepath_or_buffer, sep=lib.no_default, delimiter=None, header='infer', names=None, index_col=None, usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None, quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True, escapechar=None, comment=None, encoding=None, dialect=None, error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False, low_memory=_c_parser_defaults['low_memory'], memory_map=False, float_precision=None):
    kwds = locals()
    del kwds['filepath_or_buffer']
    del kwds['sep']
    kwds_defaults = _refine_defaults_read(dialect, delimiter, delim_whitespace, engine, sep, defaults={'delimiter': '\t'})
    kwds.update(kwds_defaults)
    return _read(filepath_or_buffer, kwds)

def read_fwf(filepath_or_buffer, colspecs='infer', widths=None, infer_nrows=100, **kwds):
    "\n    Read a table of fixed-width formatted lines into DataFrame.\n\n    Also supports optionally iterating or breaking of the file\n    into chunks.\n\n    Additional help can be found in the `online docs for IO Tools\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.\n\n    Parameters\n    ----------\n    filepath_or_buffer : str, path object or file-like object\n        Any valid string path is acceptable. The string could be a URL. Valid\n        URL schemes include http, ftp, s3, and file. For file URLs, a host is\n        expected. A local file could be:\n        ``file://localhost/path/to/table.csv``.\n\n        If you want to pass in a path object, pandas accepts any\n        ``os.PathLike``.\n\n        By file-like object, we refer to objects with a ``read()`` method,\n        such as a file handle (e.g. via builtin ``open`` function)\n        or ``StringIO``.\n    colspecs : list of tuple (int, int) or 'infer'. optional\n        A list of tuples giving the extents of the fixed-width\n        fields of each line as half-open intervals (i.e.,  [from, to[ ).\n        String value 'infer' can be used to instruct the parser to try\n        detecting the column specifications from the first 100 rows of\n        the data which are not being skipped via skiprows (default='infer').\n    widths : list of int, optional\n        A list of field widths which can be used instead of 'colspecs' if\n        the intervals are contiguous.\n    infer_nrows : int, default 100\n        The number of rows to consider when letting the parser determine the\n        `colspecs`.\n\n        .. versionadded:: 0.24.0\n    **kwds : optional\n        Optional keyword arguments can be passed to ``TextFileReader``.\n\n    Returns\n    -------\n    DataFrame or TextParser\n        A comma-separated values (csv) file is returned as two-dimensional\n        data structure with labeled axes.\n\n    See Also\n    --------\n    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.\n    read_csv : Read a comma-separated values (csv) file into DataFrame.\n\n    Examples\n    --------\n    >>> pd.read_fwf('data.csv')  # doctest: +SKIP\n    "
    if ((colspecs is None) and (widths is None)):
        raise ValueError('Must specify either colspecs or widths')
    elif ((colspecs not in (None, 'infer')) and (widths is not None)):
        raise ValueError("You must specify only one of 'widths' and 'colspecs'")
    if (widths is not None):
        (colspecs, col) = ([], 0)
        for w in widths:
            colspecs.append((col, (col + w)))
            col += w
    kwds['colspecs'] = colspecs
    kwds['infer_nrows'] = infer_nrows
    kwds['engine'] = 'python-fwf'
    return _read(filepath_or_buffer, kwds)

class TextFileReader(abc.Iterator):
    '\n\n    Passed dialect overrides any of the related parser options\n\n    '

    def __init__(self, f, engine=None, **kwds):
        self.f = f
        if (engine is not None):
            engine_specified = True
        else:
            engine = 'python'
            engine_specified = False
        self.engine = engine
        self._engine_specified = kwds.get('engine_specified', engine_specified)
        _validate_skipfooter(kwds)
        dialect = _extract_dialect(kwds)
        if (dialect is not None):
            kwds = _merge_with_dialect_properties(dialect, kwds)
        if (kwds.get('header', 'infer') == 'infer'):
            kwds['header'] = (0 if (kwds.get('names') is None) else None)
        self.orig_options = kwds
        self._currow = 0
        options = self._get_options_with_defaults(engine)
        options['storage_options'] = kwds.get('storage_options', None)
        self.chunksize = options.pop('chunksize', None)
        self.nrows = options.pop('nrows', None)
        self.squeeze = options.pop('squeeze', False)
        self._check_file_or_buffer(f, engine)
        (self.options, self.engine) = self._clean_options(options, engine)
        if ('has_index_names' in kwds):
            self.options['has_index_names'] = kwds['has_index_names']
        self._engine = self._make_engine(self.engine)

    def close(self):
        self._engine.close()

    def _get_options_with_defaults(self, engine):
        kwds = self.orig_options
        options = {}
        for (argname, default) in _parser_defaults.items():
            value = kwds.get(argname, default)
            if ((argname == 'mangle_dupe_cols') and (not value)):
                raise ValueError('Setting mangle_dupe_cols=False is not supported yet')
            else:
                options[argname] = value
        for (argname, default) in _c_parser_defaults.items():
            if (argname in kwds):
                value = kwds[argname]
                if ((engine != 'c') and (value != default)):
                    if (('python' in engine) and (argname not in _python_unsupported)):
                        pass
                    elif (value == _deprecated_defaults.get(argname, default)):
                        pass
                    else:
                        raise ValueError(f'The {repr(argname)} option is not supported with the {repr(engine)} engine')
            else:
                value = _deprecated_defaults.get(argname, default)
            options[argname] = value
        if (engine == 'python-fwf'):
            for (argname, default) in _fwf_defaults.items():
                options[argname] = kwds.get(argname, default)
        return options

    def _check_file_or_buffer(self, f, engine):
        if (is_file_like(f) and (engine != 'c') and (not hasattr(f, '__next__'))):
            raise ValueError("The 'python' engine cannot iterate through this file buffer.")

    def _clean_options(self, options, engine):
        result = options.copy()
        fallback_reason = None
        if (engine == 'c'):
            if (options['skipfooter'] > 0):
                fallback_reason = "the 'c' engine does not support skipfooter"
                engine = 'python'
        sep = options['delimiter']
        delim_whitespace = options['delim_whitespace']
        if ((sep is None) and (not delim_whitespace)):
            if (engine == 'c'):
                fallback_reason = "the 'c' engine does not support sep=None with delim_whitespace=False"
                engine = 'python'
        elif ((sep is not None) and (len(sep) > 1)):
            if ((engine == 'c') and (sep == '\\s+')):
                result['delim_whitespace'] = True
                del result['delimiter']
            elif (engine not in ('python', 'python-fwf')):
                fallback_reason = "the 'c' engine does not support regex separators (separators > 1 char and different from '\\s+' are interpreted as regex)"
                engine = 'python'
        elif delim_whitespace:
            if ('python' in engine):
                result['delimiter'] = '\\s+'
        elif (sep is not None):
            encodeable = True
            encoding = (sys.getfilesystemencoding() or 'utf-8')
            try:
                if (len(sep.encode(encoding)) > 1):
                    encodeable = False
            except UnicodeDecodeError:
                encodeable = False
            if ((not encodeable) and (engine not in ('python', 'python-fwf'))):
                fallback_reason = f"the separator encoded in {encoding} is > 1 char long, and the 'c' engine does not support such separators"
                engine = 'python'
        quotechar = options['quotechar']
        if ((quotechar is not None) and isinstance(quotechar, (str, bytes))):
            if ((len(quotechar) == 1) and (ord(quotechar) > 127) and (engine not in ('python', 'python-fwf'))):
                fallback_reason = "ord(quotechar) > 127, meaning the quotechar is larger than one byte, and the 'c' engine does not support such quotechars"
                engine = 'python'
        if (fallback_reason and self._engine_specified):
            raise ValueError(fallback_reason)
        if (engine == 'c'):
            for arg in _c_unsupported:
                del result[arg]
        if ('python' in engine):
            for arg in _python_unsupported:
                if (fallback_reason and (result[arg] != _c_parser_defaults[arg])):
                    raise ValueError(f"Falling back to the 'python' engine because {fallback_reason}, but this causes {repr(arg)} to be ignored as it is not supported by the 'python' engine.")
                del result[arg]
        if fallback_reason:
            warnings.warn(f"Falling back to the 'python' engine because {fallback_reason}; you can avoid this warning by specifying engine='python'.", ParserWarning, stacklevel=5)
        index_col = options['index_col']
        names = options['names']
        converters = options['converters']
        na_values = options['na_values']
        skiprows = options['skiprows']
        validate_header_arg(options['header'])
        for arg in _deprecated_args:
            parser_default = _c_parser_defaults[arg]
            depr_default = _deprecated_defaults[arg]
            if (result.get(arg, depr_default) != depr_default):
                msg = f'''The {arg} argument has been deprecated and will be removed in a future version.

'''
                warnings.warn(msg, FutureWarning, stacklevel=2)
            else:
                result[arg] = parser_default
        if (index_col is True):
            raise ValueError("The value of index_col couldn't be 'True'")
        if _is_index_col(index_col):
            if (not isinstance(index_col, (list, tuple, np.ndarray))):
                index_col = [index_col]
        result['index_col'] = index_col
        names = (list(names) if (names is not None) else names)
        if (converters is not None):
            if (not isinstance(converters, dict)):
                raise TypeError(f'Type converters must be a dict or subclass, input was a {type(converters).__name__}')
        else:
            converters = {}
        keep_default_na = options['keep_default_na']
        (na_values, na_fvalues) = _clean_na_values(na_values, keep_default_na)
        if (engine != 'c'):
            if is_integer(skiprows):
                skiprows = list(range(skiprows))
            if (skiprows is None):
                skiprows = set()
            elif (not callable(skiprows)):
                skiprows = set(skiprows)
        result['names'] = names
        result['converters'] = converters
        result['na_values'] = na_values
        result['na_fvalues'] = na_fvalues
        result['skiprows'] = skiprows
        return (result, engine)

    def __next__(self):
        try:
            return self.get_chunk()
        except StopIteration:
            self.close()
            raise

    def _make_engine(self, engine='c'):
        mapping: Dict[(str, Type[ParserBase])] = {'c': CParserWrapper, 'python': PythonParser, 'python-fwf': FixedWidthFieldParser}
        if (engine not in mapping):
            raise ValueError(f'Unknown engine: {engine} (valid options are {mapping.keys()})')
        return mapping[engine](self.f, **self.options)

    def _failover_to_python(self):
        raise AbstractMethodError(self)

    def read(self, nrows=None):
        nrows = validate_integer('nrows', nrows)
        (index, columns, col_dict) = self._engine.read(nrows)
        if (index is None):
            if col_dict:
                new_rows = len(next(iter(col_dict.values())))
                index = RangeIndex(self._currow, (self._currow + new_rows))
            else:
                new_rows = 0
        else:
            new_rows = len(index)
        df = DataFrame(col_dict, columns=columns, index=index)
        self._currow += new_rows
        if (self.squeeze and (len(df.columns) == 1)):
            return df[df.columns[0]].copy()
        return df

    def get_chunk(self, size=None):
        if (size is None):
            size = self.chunksize
        if (self.nrows is not None):
            if (self._currow >= self.nrows):
                raise StopIteration
            size = min(size, (self.nrows - self._currow))
        return self.read(nrows=size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def _is_index_col(col):
    return ((col is not None) and (col is not False))

def _is_potential_multi_index(columns, index_col=None):
    '\n    Check whether or not the `columns` parameter\n    could be converted into a MultiIndex.\n\n    Parameters\n    ----------\n    columns : array-like\n        Object which may or may not be convertible into a MultiIndex\n    index_col : None, bool or list, optional\n        Column or columns to use as the (possibly hierarchical) index\n\n    Returns\n    -------\n    boolean : Whether or not columns could become a MultiIndex\n    '
    if ((index_col is None) or isinstance(index_col, bool)):
        index_col = []
    return (len(columns) and (not isinstance(columns, MultiIndex)) and all((isinstance(c, tuple) for c in columns if (c not in list(index_col)))))

def _evaluate_usecols(usecols, names):
    "\n    Check whether or not the 'usecols' parameter\n    is a callable.  If so, enumerates the 'names'\n    parameter and returns a set of indices for\n    each entry in 'names' that evaluates to True.\n    If not a callable, returns 'usecols'.\n    "
    if callable(usecols):
        return {i for (i, name) in enumerate(names) if usecols(name)}
    return usecols

def _validate_usecols_names(usecols, names):
    '\n    Validates that all usecols are present in a given\n    list of names. If not, raise a ValueError that\n    shows what usecols are missing.\n\n    Parameters\n    ----------\n    usecols : iterable of usecols\n        The columns to validate are present in names.\n    names : iterable of names\n        The column names to check against.\n\n    Returns\n    -------\n    usecols : iterable of usecols\n        The `usecols` parameter if the validation succeeds.\n\n    Raises\n    ------\n    ValueError : Columns were missing. Error message will list them.\n    '
    missing = [c for c in usecols if (c not in names)]
    if (len(missing) > 0):
        raise ValueError(f'Usecols do not match columns, columns expected but not found: {missing}')
    return usecols

def _validate_skipfooter_arg(skipfooter):
    "\n    Validate the 'skipfooter' parameter.\n\n    Checks whether 'skipfooter' is a non-negative integer.\n    Raises a ValueError if that is not the case.\n\n    Parameters\n    ----------\n    skipfooter : non-negative integer\n        The number of rows to skip at the end of the file.\n\n    Returns\n    -------\n    validated_skipfooter : non-negative integer\n        The original input if the validation succeeds.\n\n    Raises\n    ------\n    ValueError : 'skipfooter' was not a non-negative integer.\n    "
    if (not is_integer(skipfooter)):
        raise ValueError('skipfooter must be an integer')
    if (skipfooter < 0):
        raise ValueError('skipfooter cannot be negative')
    return skipfooter

def _validate_usecols_arg(usecols):
    "\n    Validate the 'usecols' parameter.\n\n    Checks whether or not the 'usecols' parameter contains all integers\n    (column selection by index), strings (column by name) or is a callable.\n    Raises a ValueError if that is not the case.\n\n    Parameters\n    ----------\n    usecols : list-like, callable, or None\n        List of columns to use when parsing or a callable that can be used\n        to filter a list of table columns.\n\n    Returns\n    -------\n    usecols_tuple : tuple\n        A tuple of (verified_usecols, usecols_dtype).\n\n        'verified_usecols' is either a set if an array-like is passed in or\n        'usecols' if a callable or None is passed in.\n\n        'usecols_dtype` is the inferred dtype of 'usecols' if an array-like\n        is passed in or None if a callable or None is passed in.\n    "
    msg = "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
    if (usecols is not None):
        if callable(usecols):
            return (usecols, None)
        if (not is_list_like(usecols)):
            raise ValueError(msg)
        usecols_dtype = lib.infer_dtype(usecols, skipna=False)
        if (usecols_dtype not in ('empty', 'integer', 'string')):
            raise ValueError(msg)
        usecols = set(usecols)
        return (usecols, usecols_dtype)
    return (usecols, None)

def _validate_parse_dates_arg(parse_dates):
    "\n    Check whether or not the 'parse_dates' parameter\n    is a non-boolean scalar. Raises a ValueError if\n    that is the case.\n    "
    msg = "Only booleans, lists, and dictionaries are accepted for the 'parse_dates' parameter"
    if (parse_dates is not None):
        if is_scalar(parse_dates):
            if (not lib.is_bool(parse_dates)):
                raise TypeError(msg)
        elif (not isinstance(parse_dates, (list, dict))):
            raise TypeError(msg)
    return parse_dates

class ParserBase():

    def __init__(self, kwds):
        self.names = kwds.get('names')
        self.orig_names: Optional[List] = None
        self.prefix = kwds.pop('prefix', None)
        self.index_col = kwds.get('index_col', None)
        self.unnamed_cols: Set = set()
        self.index_names: Optional[List] = None
        self.col_names = None
        self.parse_dates = _validate_parse_dates_arg(kwds.pop('parse_dates', False))
        self.date_parser = kwds.pop('date_parser', None)
        self.dayfirst = kwds.pop('dayfirst', False)
        self.keep_date_col = kwds.pop('keep_date_col', False)
        self.na_values = kwds.get('na_values')
        self.na_fvalues = kwds.get('na_fvalues')
        self.na_filter = kwds.get('na_filter', False)
        self.keep_default_na = kwds.get('keep_default_na', True)
        self.true_values = kwds.get('true_values')
        self.false_values = kwds.get('false_values')
        self.mangle_dupe_cols = kwds.get('mangle_dupe_cols', True)
        self.infer_datetime_format = kwds.pop('infer_datetime_format', False)
        self.cache_dates = kwds.pop('cache_dates', True)
        self._date_conv = _make_date_converter(date_parser=self.date_parser, dayfirst=self.dayfirst, infer_datetime_format=self.infer_datetime_format, cache_dates=self.cache_dates)
        self.header = kwds.get('header')
        if isinstance(self.header, (list, tuple, np.ndarray)):
            if (not all(map(is_integer, self.header))):
                raise ValueError('header must be integer or list of integers')
            if any(((i < 0) for i in self.header)):
                raise ValueError('cannot specify multi-index header with negative integers')
            if kwds.get('usecols'):
                raise ValueError('cannot specify usecols when specifying a multi-index header')
            if kwds.get('names'):
                raise ValueError('cannot specify names when specifying a multi-index header')
            if (self.index_col is not None):
                is_sequence = isinstance(self.index_col, (list, tuple, np.ndarray))
                if (not ((is_sequence and all(map(is_integer, self.index_col))) or is_integer(self.index_col))):
                    raise ValueError('index_col must only contain row numbers when specifying a multi-index header')
        elif (self.header is not None):
            if (self.prefix is not None):
                raise ValueError('Argument prefix must be None if argument header is not None')
            elif (not is_integer(self.header)):
                raise ValueError('header must be integer or list of integers')
            elif (self.header < 0):
                raise ValueError('Passing negative integer to header is invalid. For no header, use header=None instead')
        self._name_processed = False
        self._first_chunk = True
        self.handles: Optional[IOHandles] = None

    def _open_handles(self, src, kwds):
        '\n        Let the readers open IOHanldes after they are done with their potential raises.\n        '
        self.handles = get_handle(src, 'r', encoding=kwds.get('encoding', None), compression=kwds.get('compression', None), memory_map=kwds.get('memory_map', False), storage_options=kwds.get('storage_options', None))

    def _validate_parse_dates_presence(self, columns):
        '\n        Check if parse_dates are in columns.\n\n        If user has provided names for parse_dates, check if those columns\n        are available.\n\n        Parameters\n        ----------\n        columns : list\n            List of names of the dataframe.\n\n        Raises\n        ------\n        ValueError\n            If column to parse_date is not in dataframe.\n\n        '
        cols_needed: Iterable
        if is_dict_like(self.parse_dates):
            cols_needed = itertools.chain(*self.parse_dates.values())
        elif is_list_like(self.parse_dates):
            cols_needed = itertools.chain.from_iterable(((col if is_list_like(col) else [col]) for col in self.parse_dates))
        else:
            cols_needed = []
        missing_cols = ', '.join(sorted({col for col in cols_needed if (isinstance(col, str) and (col not in columns))}))
        if missing_cols:
            raise ValueError(f"Missing column provided to 'parse_dates': '{missing_cols}'")

    def close(self):
        if (self.handles is not None):
            self.handles.close()

    @property
    def _has_complex_date_col(self):
        return (isinstance(self.parse_dates, dict) or (isinstance(self.parse_dates, list) and (len(self.parse_dates) > 0) and isinstance(self.parse_dates[0], list)))

    def _should_parse_dates(self, i):
        if isinstance(self.parse_dates, bool):
            return self.parse_dates
        else:
            if (self.index_names is not None):
                name = self.index_names[i]
            else:
                name = None
            j = (i if (self.index_col is None) else self.index_col[i])
            if is_scalar(self.parse_dates):
                return ((j == self.parse_dates) or ((name is not None) and (name == self.parse_dates)))
            else:
                return ((j in self.parse_dates) or ((name is not None) and (name in self.parse_dates)))

    def _extract_multi_indexer_columns(self, header, index_names, col_names, passed_names=False):
        '\n        extract and return the names, index_names, col_names\n        header is a list-of-lists returned from the parsers\n        '
        if (len(header) < 2):
            return (header[0], index_names, col_names, passed_names)
        ic = self.index_col
        if (ic is None):
            ic = []
        if (not isinstance(ic, (list, tuple, np.ndarray))):
            ic = [ic]
        sic = set(ic)
        index_names = header.pop((- 1))
        (index_names, _, _) = _clean_index_names(index_names, self.index_col, self.unnamed_cols)
        field_count = len(header[0])

        def extract(r):
            return tuple((r[i] for i in range(field_count) if (i not in sic)))
        columns = list(zip(*(extract(r) for r in header)))
        names = (ic + columns)
        for n in range(len(columns[0])):
            if all(((ensure_str(col[n]) in self.unnamed_cols) for col in columns)):
                header = ','.join((str(x) for x in self.header))
                raise ParserError(f'Passed header=[{header}] are too many rows for this multi_index of columns')
        if len(ic):
            col_names = [(r[0] if ((r[0] is not None) and (r[0] not in self.unnamed_cols)) else None) for r in header]
        else:
            col_names = ([None] * len(header))
        passed_names = True
        return (names, index_names, col_names, passed_names)

    def _maybe_dedup_names(self, names):
        if self.mangle_dupe_cols:
            names = list(names)
            counts = defaultdict(int)
            is_potential_mi = _is_potential_multi_index(names, self.index_col)
            for (i, col) in enumerate(names):
                cur_count = counts[col]
                while (cur_count > 0):
                    counts[col] = (cur_count + 1)
                    if is_potential_mi:
                        col = (col[:(- 1)] + (f'{col[(- 1)]}.{cur_count}',))
                    else:
                        col = f'{col}.{cur_count}'
                    cur_count = counts[col]
                names[i] = col
                counts[col] = (cur_count + 1)
        return names

    def _maybe_make_multi_index_columns(self, columns, col_names=None):
        if _is_potential_multi_index(columns):
            columns = MultiIndex.from_tuples(columns, names=col_names)
        return columns

    def _make_index(self, data, alldata, columns, indexnamerow=False):
        if ((not _is_index_col(self.index_col)) or (not self.index_col)):
            index = None
        elif (not self._has_complex_date_col):
            index = self._get_simple_index(alldata, columns)
            index = self._agg_index(index)
        elif self._has_complex_date_col:
            if (not self._name_processed):
                (self.index_names, _, self.index_col) = _clean_index_names(list(columns), self.index_col, self.unnamed_cols)
                self._name_processed = True
            index = self._get_complex_date_index(data, columns)
            index = self._agg_index(index, try_parse_dates=False)
        if indexnamerow:
            coffset = (len(indexnamerow) - len(columns))
            index = index.set_names(indexnamerow[:coffset])
        columns = self._maybe_make_multi_index_columns(columns, self.col_names)
        return (index, columns)
    _implicit_index = False

    def _get_simple_index(self, data, columns):

        def ix(col):
            if (not isinstance(col, str)):
                return col
            raise ValueError(f'Index {col} invalid')
        to_remove = []
        index = []
        for idx in self.index_col:
            i = ix(idx)
            to_remove.append(i)
            index.append(data[i])
        for i in sorted(to_remove, reverse=True):
            data.pop(i)
            if (not self._implicit_index):
                columns.pop(i)
        return index

    def _get_complex_date_index(self, data, col_names):

        def _get_name(icol):
            if isinstance(icol, str):
                return icol
            if (col_names is None):
                raise ValueError(f'Must supply column order to use {icol!s} as index')
            for (i, c) in enumerate(col_names):
                if (i == icol):
                    return c
        to_remove = []
        index = []
        for idx in self.index_col:
            name = _get_name(idx)
            to_remove.append(name)
            index.append(data[name])
        for c in sorted(to_remove, reverse=True):
            data.pop(c)
            col_names.remove(c)
        return index

    def _agg_index(self, index, try_parse_dates=True):
        arrays = []
        for (i, arr) in enumerate(index):
            if (try_parse_dates and self._should_parse_dates(i)):
                arr = self._date_conv(arr)
            if self.na_filter:
                col_na_values = self.na_values
                col_na_fvalues = self.na_fvalues
            else:
                col_na_values = set()
                col_na_fvalues = set()
            if isinstance(self.na_values, dict):
                col_name = self.index_names[i]
                if (col_name is not None):
                    (col_na_values, col_na_fvalues) = _get_na_values(col_name, self.na_values, self.na_fvalues, self.keep_default_na)
            (arr, _) = self._infer_types(arr, (col_na_values | col_na_fvalues))
            arrays.append(arr)
        names = self.index_names
        index = ensure_index_from_sequences(arrays, names)
        return index

    def _convert_to_ndarrays(self, dct, na_values, na_fvalues, verbose=False, converters=None, dtypes=None):
        result = {}
        for (c, values) in dct.items():
            conv_f = (None if (converters is None) else converters.get(c, None))
            if isinstance(dtypes, dict):
                cast_type = dtypes.get(c, None)
            else:
                cast_type = dtypes
            if self.na_filter:
                (col_na_values, col_na_fvalues) = _get_na_values(c, na_values, na_fvalues, self.keep_default_na)
            else:
                (col_na_values, col_na_fvalues) = (set(), set())
            if (conv_f is not None):
                if (cast_type is not None):
                    warnings.warn(f'Both a converter and dtype were specified for column {c} - only the converter will be used', ParserWarning, stacklevel=7)
                try:
                    values = lib.map_infer(values, conv_f)
                except ValueError:
                    mask = algorithms.isin(values, list(na_values)).view(np.uint8)
                    values = lib.map_infer_mask(values, conv_f, mask)
                (cvals, na_count) = self._infer_types(values, (set(col_na_values) | col_na_fvalues), try_num_bool=False)
            else:
                is_ea = is_extension_array_dtype(cast_type)
                is_str_or_ea_dtype = (is_ea or is_string_dtype(cast_type))
                try_num_bool = (not (cast_type and is_str_or_ea_dtype))
                (cvals, na_count) = self._infer_types(values, (set(col_na_values) | col_na_fvalues), try_num_bool)
                if (cast_type and ((not is_dtype_equal(cvals, cast_type)) or is_extension_array_dtype(cast_type))):
                    if ((not is_ea) and (na_count > 0)):
                        try:
                            if is_bool_dtype(cast_type):
                                raise ValueError(f'Bool column has NA values in column {c}')
                        except (AttributeError, TypeError):
                            pass
                    cast_type = pandas_dtype(cast_type)
                    cvals = self._cast_types(cvals, cast_type, c)
            result[c] = cvals
            if (verbose and na_count):
                print(f'Filled {na_count} NA values in column {c!s}')
        return result

    def _infer_types(self, values, na_values, try_num_bool=True):
        '\n        Infer types of values, possibly casting\n\n        Parameters\n        ----------\n        values : ndarray\n        na_values : set\n        try_num_bool : bool, default try\n           try to cast values to numeric (first preference) or boolean\n\n        Returns\n        -------\n        converted : ndarray\n        na_count : int\n        '
        na_count = 0
        if issubclass(values.dtype.type, (np.number, np.bool_)):
            mask = algorithms.isin(values, list(na_values))
            na_count = mask.sum()
            if (na_count > 0):
                if is_integer_dtype(values):
                    values = values.astype(np.float64)
                np.putmask(values, mask, np.nan)
            return (values, na_count)
        if (try_num_bool and is_object_dtype(values.dtype)):
            try:
                result = lib.maybe_convert_numeric(values, na_values, False)
            except (ValueError, TypeError):
                result = values
                na_count = parsers.sanitize_objects(result, na_values, False)
            else:
                na_count = isna(result).sum()
        else:
            result = values
            if (values.dtype == np.object_):
                na_count = parsers.sanitize_objects(values, na_values, False)
        if ((result.dtype == np.object_) and try_num_bool):
            result = libops.maybe_convert_bool(np.asarray(values), true_values=self.true_values, false_values=self.false_values)
        return (result, na_count)

    def _cast_types(self, values, cast_type, column):
        '\n        Cast values to specified type\n\n        Parameters\n        ----------\n        values : ndarray\n        cast_type : string or np.dtype\n           dtype to cast values to\n        column : string\n            column name - used only for error reporting\n\n        Returns\n        -------\n        converted : ndarray\n        '
        if is_categorical_dtype(cast_type):
            known_cats = (isinstance(cast_type, CategoricalDtype) and (cast_type.categories is not None))
            if ((not is_object_dtype(values)) and (not known_cats)):
                values = astype_nansafe(values, str)
            cats = Index(values).unique().dropna()
            values = Categorical._from_inferred_categories(cats, cats.get_indexer(values), cast_type, true_values=self.true_values)
        elif is_extension_array_dtype(cast_type):
            cast_type = pandas_dtype(cast_type)
            array_type = cast_type.construct_array_type()
            try:
                return array_type._from_sequence_of_strings(values, dtype=cast_type)
            except NotImplementedError as err:
                raise NotImplementedError(f'Extension Array: {array_type} must implement _from_sequence_of_strings in order to be used in parser methods') from err
        else:
            try:
                values = astype_nansafe(values, cast_type, copy=True, skipna=True)
            except ValueError as err:
                raise ValueError(f'Unable to convert column {column} to type {cast_type}') from err
        return values

    def _do_date_conversions(self, names, data):
        if (self.parse_dates is not None):
            (data, names) = _process_date_conversion(data, self._date_conv, self.parse_dates, self.index_col, self.index_names, names, keep_date_col=self.keep_date_col)
        return (names, data)

class CParserWrapper(ParserBase):

    def __init__(self, src, **kwds):
        self.kwds = kwds
        kwds = kwds.copy()
        ParserBase.__init__(self, kwds)
        kwds['allow_leading_cols'] = (self.index_col is not False)
        (self.usecols, self.usecols_dtype) = _validate_usecols_arg(kwds['usecols'])
        kwds['usecols'] = self.usecols
        self._open_handles(src, kwds)
        assert (self.handles is not None)
        for key in ('storage_options', 'encoding', 'memory_map', 'compression'):
            kwds.pop(key, None)
        if (self.handles.is_mmap and hasattr(self.handles.handle, 'mmap')):
            self.handles.handle = self.handles.handle.mmap
        try:
            self._reader = parsers.TextReader(self.handles.handle, **kwds)
        except Exception:
            self.handles.close()
            raise
        self.unnamed_cols = self._reader.unnamed_cols
        passed_names = (self.names is None)
        if (self._reader.header is None):
            self.names = None
        elif (len(self._reader.header) > 1):
            (self.names, self.index_names, self.col_names, passed_names) = self._extract_multi_indexer_columns(self._reader.header, self.index_names, self.col_names, passed_names)
        else:
            self.names = list(self._reader.header[0])
        if (self.names is None):
            if self.prefix:
                self.names = [f'{self.prefix}{i}' for i in range(self._reader.table_width)]
            else:
                self.names = list(range(self._reader.table_width))
        self.orig_names = self.names[:]
        if self.usecols:
            usecols = _evaluate_usecols(self.usecols, self.orig_names)
            assert (self.orig_names is not None)
            if ((self.usecols_dtype == 'string') and (not set(usecols).issubset(self.orig_names))):
                _validate_usecols_names(usecols, self.orig_names)
            if (len(self.names) > len(usecols)):
                self.names = [n for (i, n) in enumerate(self.names) if ((i in usecols) or (n in usecols))]
            if (len(self.names) < len(usecols)):
                _validate_usecols_names(usecols, self.names)
        self._validate_parse_dates_presence(self.names)
        self._set_noconvert_columns()
        self.orig_names = self.names
        if (not self._has_complex_date_col):
            if ((self._reader.leading_cols == 0) and _is_index_col(self.index_col)):
                self._name_processed = True
                (index_names, self.names, self.index_col) = _clean_index_names(self.names, self.index_col, self.unnamed_cols)
                if (self.index_names is None):
                    self.index_names = index_names
            if ((self._reader.header is None) and (not passed_names)):
                self.index_names = ([None] * len(self.index_names))
        self._implicit_index = (self._reader.leading_cols > 0)

    def close(self):
        super().close()
        try:
            self._reader.close()
        except ValueError:
            pass

    def _set_noconvert_columns(self):
        '\n        Set the columns that should not undergo dtype conversions.\n\n        Currently, any column that is involved with date parsing will not\n        undergo such conversions.\n        '
        names = self.orig_names
        if (self.usecols_dtype == 'integer'):
            usecols = list(self.usecols)
            usecols.sort()
        elif (callable(self.usecols) or (self.usecols_dtype not in ('empty', None))):
            usecols = self.names[:]
        else:
            usecols = None

        def _set(x):
            if ((usecols is not None) and is_integer(x)):
                x = usecols[x]
            if (not is_integer(x)):
                assert (names is not None)
                x = names.index(x)
            self._reader.set_noconvert(x)
        if isinstance(self.parse_dates, list):
            for val in self.parse_dates:
                if isinstance(val, list):
                    for k in val:
                        _set(k)
                else:
                    _set(val)
        elif isinstance(self.parse_dates, dict):
            for val in self.parse_dates.values():
                if isinstance(val, list):
                    for k in val:
                        _set(k)
                else:
                    _set(val)
        elif self.parse_dates:
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    _set(k)
            elif (self.index_col is not None):
                _set(self.index_col)

    def set_error_bad_lines(self, status):
        self._reader.set_error_bad_lines(int(status))

    def read(self, nrows=None):
        try:
            data = self._reader.read(nrows)
        except StopIteration:
            if self._first_chunk:
                self._first_chunk = False
                names = self._maybe_dedup_names(self.orig_names)
                (index, columns, col_dict) = _get_empty_meta(names, self.index_col, self.index_names, dtype=self.kwds.get('dtype'))
                columns = self._maybe_make_multi_index_columns(columns, self.col_names)
                if (self.usecols is not None):
                    columns = self._filter_usecols(columns)
                col_dict = {k: v for (k, v) in col_dict.items() if (k in columns)}
                return (index, columns, col_dict)
            else:
                self.close()
                raise
        self._first_chunk = False
        names = self.names
        if self._reader.leading_cols:
            if self._has_complex_date_col:
                raise NotImplementedError('file structure not yet supported')
            arrays = []
            for i in range(self._reader.leading_cols):
                if (self.index_col is None):
                    values = data.pop(i)
                else:
                    values = data.pop(self.index_col[i])
                values = self._maybe_parse_dates(values, i, try_parse_dates=True)
                arrays.append(values)
            index = ensure_index_from_sequences(arrays)
            if (self.usecols is not None):
                names = self._filter_usecols(names)
            names = self._maybe_dedup_names(names)
            data = sorted(data.items())
            data = {k: v for (k, (i, v)) in zip(names, data)}
            (names, data) = self._do_date_conversions(names, data)
        else:
            data = sorted(data.items())
            assert (self.orig_names is not None)
            names = list(self.orig_names)
            names = self._maybe_dedup_names(names)
            if (self.usecols is not None):
                names = self._filter_usecols(names)
            alldata = [x[1] for x in data]
            data = {k: v for (k, (i, v)) in zip(names, data)}
            (names, data) = self._do_date_conversions(names, data)
            (index, names) = self._make_index(data, alldata, names)
        names = self._maybe_make_multi_index_columns(names, self.col_names)
        return (index, names, data)

    def _filter_usecols(self, names):
        usecols = _evaluate_usecols(self.usecols, names)
        if ((usecols is not None) and (len(names) != len(usecols))):
            names = [name for (i, name) in enumerate(names) if ((i in usecols) or (name in usecols))]
        return names

    def _get_index_names(self):
        names = list(self._reader.header[0])
        idx_names = None
        if ((self._reader.leading_cols == 0) and (self.index_col is not None)):
            (idx_names, names, self.index_col) = _clean_index_names(names, self.index_col, self.unnamed_cols)
        return (names, idx_names)

    def _maybe_parse_dates(self, values, index, try_parse_dates=True):
        if (try_parse_dates and self._should_parse_dates(index)):
            values = self._date_conv(values)
        return values

def TextParser(*args, **kwds):
    "\n    Converts lists of lists/tuples into DataFrames with proper type inference\n    and optional (e.g. string to datetime) conversion. Also enables iterating\n    lazily over chunks of large files\n\n    Parameters\n    ----------\n    data : file-like object or list\n    delimiter : separator character to use\n    dialect : str or csv.Dialect instance, optional\n        Ignored if delimiter is longer than 1 character\n    names : sequence, default\n    header : int, default 0\n        Row to use to parse column labels. Defaults to the first row. Prior\n        rows will be discarded\n    index_col : int or list, optional\n        Column or columns to use as the (possibly hierarchical) index\n    has_index_names: bool, default False\n        True if the cols defined in index_col have an index name and are\n        not in the header.\n    na_values : scalar, str, list-like, or dict, optional\n        Additional strings to recognize as NA/NaN.\n    keep_default_na : bool, default True\n    thousands : str, optional\n        Thousands separator\n    comment : str, optional\n        Comment out remainder of line\n    parse_dates : bool, default False\n    keep_date_col : bool, default False\n    date_parser : function, optional\n    skiprows : list of integers\n        Row numbers to skip\n    skipfooter : int\n        Number of line at bottom of file to skip\n    converters : dict, optional\n        Dict of functions for converting values in certain columns. Keys can\n        either be integers or column labels, values are functions that take one\n        input argument, the cell (not column) content, and return the\n        transformed content.\n    encoding : str, optional\n        Encoding to use for UTF when reading/writing (ex. 'utf-8')\n    squeeze : bool, default False\n        returns Series if only one column.\n    infer_datetime_format: bool, default False\n        If True and `parse_dates` is True for a column, try to infer the\n        datetime format based on the first datetime string. If the format\n        can be inferred, there often will be a large parsing speed-up.\n    float_precision : str, optional\n        Specifies which converter the C engine should use for floating-point\n        values. The options are `None` or `high` for the ordinary converter,\n        `legacy` for the original lower precision pandas converter, and\n        `round_trip` for the round-trip converter.\n\n        .. versionchanged:: 1.2\n    "
    kwds['engine'] = 'python'
    return TextFileReader(*args, **kwds)

def count_empty_vals(vals):
    return sum((1 for v in vals if ((v == '') or (v is None))))

class PythonParser(ParserBase):

    def __init__(self, f, **kwds):
        '\n        Workhorse function for processing nested list into DataFrame\n        '
        ParserBase.__init__(self, kwds)
        self.data: Optional[Iterator[str]] = None
        self.buf: List = []
        self.pos = 0
        self.line_pos = 0
        self.skiprows = kwds['skiprows']
        if callable(self.skiprows):
            self.skipfunc = self.skiprows
        else:
            self.skipfunc = (lambda x: (x in self.skiprows))
        self.skipfooter = _validate_skipfooter_arg(kwds['skipfooter'])
        self.delimiter = kwds['delimiter']
        self.quotechar = kwds['quotechar']
        if isinstance(self.quotechar, str):
            self.quotechar = str(self.quotechar)
        self.escapechar = kwds['escapechar']
        self.doublequote = kwds['doublequote']
        self.skipinitialspace = kwds['skipinitialspace']
        self.lineterminator = kwds['lineterminator']
        self.quoting = kwds['quoting']
        (self.usecols, _) = _validate_usecols_arg(kwds['usecols'])
        self.skip_blank_lines = kwds['skip_blank_lines']
        self.warn_bad_lines = kwds['warn_bad_lines']
        self.error_bad_lines = kwds['error_bad_lines']
        self.names_passed = (kwds['names'] or None)
        self.has_index_names = False
        if ('has_index_names' in kwds):
            self.has_index_names = kwds['has_index_names']
        self.verbose = kwds['verbose']
        self.converters = kwds['converters']
        self.dtype = kwds['dtype']
        self.thousands = kwds['thousands']
        self.decimal = kwds['decimal']
        self.comment = kwds['comment']
        if isinstance(f, list):
            self.data = cast(Iterator[str], f)
        else:
            self._open_handles(f, kwds)
            assert (self.handles is not None)
            assert hasattr(self.handles.handle, 'readline')
            self._make_reader(self.handles.handle)
        self._col_indices = None
        try:
            (self.columns, self.num_original_columns, self.unnamed_cols) = self._infer_columns()
        except (TypeError, ValueError):
            self.close()
            raise
        if (len(self.columns) > 1):
            (self.columns, self.index_names, self.col_names, _) = self._extract_multi_indexer_columns(self.columns, self.index_names, self.col_names)
            self.num_original_columns = len(self.columns)
        else:
            self.columns = self.columns[0]
        self.orig_names = list(self.columns)
        if (not self._has_complex_date_col):
            (index_names, self.orig_names, self.columns) = self._get_index_name(self.columns)
            self._name_processed = True
            if (self.index_names is None):
                self.index_names = index_names
        self._validate_parse_dates_presence(self.columns)
        if self.parse_dates:
            self._no_thousands_columns = self._set_no_thousands_columns()
        else:
            self._no_thousands_columns = None
        if (len(self.decimal) != 1):
            raise ValueError('Only length-1 decimal markers supported')
        if (self.thousands is None):
            self.nonnum = re.compile(f'[^-^0-9^{self.decimal}]+')
        else:
            self.nonnum = re.compile(f'[^-^0-9^{self.thousands}^{self.decimal}]+')

    def _set_no_thousands_columns(self):
        noconvert_columns = set()

        def _set(x):
            if is_integer(x):
                noconvert_columns.add(x)
            else:
                noconvert_columns.add(self.columns.index(x))
        if isinstance(self.parse_dates, list):
            for val in self.parse_dates:
                if isinstance(val, list):
                    for k in val:
                        _set(k)
                else:
                    _set(val)
        elif isinstance(self.parse_dates, dict):
            for val in self.parse_dates.values():
                if isinstance(val, list):
                    for k in val:
                        _set(k)
                else:
                    _set(val)
        elif self.parse_dates:
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    _set(k)
            elif (self.index_col is not None):
                _set(self.index_col)
        return noconvert_columns

    def _make_reader(self, f):
        sep = self.delimiter
        if ((sep is None) or (len(sep) == 1)):
            if self.lineterminator:
                raise ValueError('Custom line terminators not supported in python parser (yet)')

            class MyDialect(csv.Dialect):
                delimiter = self.delimiter
                quotechar = self.quotechar
                escapechar = self.escapechar
                doublequote = self.doublequote
                skipinitialspace = self.skipinitialspace
                quoting = self.quoting
                lineterminator = '\n'
            dia = MyDialect
            if (sep is not None):
                dia.delimiter = sep
            else:
                line = f.readline()
                lines = self._check_comments([[line]])[0]
                while (self.skipfunc(self.pos) or (not lines)):
                    self.pos += 1
                    line = f.readline()
                    lines = self._check_comments([[line]])[0]
                line = lines[0]
                self.pos += 1
                self.line_pos += 1
                sniffed = csv.Sniffer().sniff(line)
                dia.delimiter = sniffed.delimiter
                line_rdr = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))
            reader = csv.reader(f, dialect=dia, strict=True)
        else:

            def _read():
                line = f.readline()
                pat = re.compile(sep)
                (yield pat.split(line.strip()))
                for line in f:
                    (yield pat.split(line.strip()))
            reader = _read()
        self.data = reader

    def read(self, rows=None):
        try:
            content = self._get_lines(rows)
        except StopIteration:
            if self._first_chunk:
                content = []
            else:
                self.close()
                raise
        self._first_chunk = False
        columns = list(self.orig_names)
        if (not len(content)):
            names = self._maybe_dedup_names(self.orig_names)
            (index, columns, col_dict) = _get_empty_meta(names, self.index_col, self.index_names, self.dtype)
            columns = self._maybe_make_multi_index_columns(columns, self.col_names)
            return (index, columns, col_dict)
        count_empty_content_vals = count_empty_vals(content[0])
        indexnamerow = None
        if (self.has_index_names and (count_empty_content_vals == len(columns))):
            indexnamerow = content[0]
            content = content[1:]
        alldata = self._rows_to_cols(content)
        (data, columns) = self._exclude_implicit_index(alldata)
        (columns, data) = self._do_date_conversions(columns, data)
        data = self._convert_data(data)
        (index, columns) = self._make_index(data, alldata, columns, indexnamerow)
        return (index, columns, data)

    def _exclude_implicit_index(self, alldata):
        names = self._maybe_dedup_names(self.orig_names)
        offset = 0
        if self._implicit_index:
            offset = len(self.index_col)
        if ((self._col_indices is not None) and (len(names) != len(self._col_indices))):
            names = [names[i] for i in sorted(self._col_indices)]
        return ({name: alldata[(i + offset)] for (i, name) in enumerate(names)}, names)

    def get_chunk(self, size=None):
        if (size is None):
            size = self.chunksize
        return self.read(rows=size)

    def _convert_data(self, data):

        def _clean_mapping(mapping):
            'converts col numbers to names'
            clean = {}
            for (col, v) in mapping.items():
                if (isinstance(col, int) and (col not in self.orig_names)):
                    col = self.orig_names[col]
                clean[col] = v
            return clean
        clean_conv = _clean_mapping(self.converters)
        if (not isinstance(self.dtype, dict)):
            clean_dtypes = self.dtype
        else:
            clean_dtypes = _clean_mapping(self.dtype)
        clean_na_values = {}
        clean_na_fvalues = {}
        if isinstance(self.na_values, dict):
            for col in self.na_values:
                na_value = self.na_values[col]
                na_fvalue = self.na_fvalues[col]
                if (isinstance(col, int) and (col not in self.orig_names)):
                    col = self.orig_names[col]
                clean_na_values[col] = na_value
                clean_na_fvalues[col] = na_fvalue
        else:
            clean_na_values = self.na_values
            clean_na_fvalues = self.na_fvalues
        return self._convert_to_ndarrays(data, clean_na_values, clean_na_fvalues, self.verbose, clean_conv, clean_dtypes)

    def _infer_columns(self):
        names = self.names
        num_original_columns = 0
        clear_buffer = True
        unnamed_cols = set()
        if (self.header is not None):
            header = self.header
            if isinstance(header, (list, tuple, np.ndarray)):
                have_mi_columns = (len(header) > 1)
                if have_mi_columns:
                    header = (list(header) + [(header[(- 1)] + 1)])
            else:
                have_mi_columns = False
                header = [header]
            columns = []
            for (level, hr) in enumerate(header):
                try:
                    line = self._buffered_line()
                    while (self.line_pos <= hr):
                        line = self._next_line()
                except StopIteration as err:
                    if (self.line_pos < hr):
                        raise ValueError(f'Passed header={hr} but only {(self.line_pos + 1)} lines in file') from err
                    if (have_mi_columns and (hr > 0)):
                        if clear_buffer:
                            self._clear_buffer()
                        columns.append(([None] * len(columns[(- 1)])))
                        return (columns, num_original_columns, unnamed_cols)
                    if (not self.names):
                        raise EmptyDataError('No columns to parse from file') from err
                    line = self.names[:]
                this_columns = []
                this_unnamed_cols = []
                for (i, c) in enumerate(line):
                    if (c == ''):
                        if have_mi_columns:
                            col_name = f'Unnamed: {i}_level_{level}'
                        else:
                            col_name = f'Unnamed: {i}'
                        this_unnamed_cols.append(i)
                        this_columns.append(col_name)
                    else:
                        this_columns.append(c)
                if ((not have_mi_columns) and self.mangle_dupe_cols):
                    counts = defaultdict(int)
                    for (i, col) in enumerate(this_columns):
                        cur_count = counts[col]
                        while (cur_count > 0):
                            counts[col] = (cur_count + 1)
                            col = f'{col}.{cur_count}'
                            cur_count = counts[col]
                        this_columns[i] = col
                        counts[col] = (cur_count + 1)
                elif have_mi_columns:
                    if (hr == header[(- 1)]):
                        lc = len(this_columns)
                        ic = (len(self.index_col) if (self.index_col is not None) else 0)
                        unnamed_count = len(this_unnamed_cols)
                        if ((lc != unnamed_count) and ((lc - ic) > unnamed_count)):
                            clear_buffer = False
                            this_columns = ([None] * lc)
                            self.buf = [self.buf[(- 1)]]
                columns.append(this_columns)
                unnamed_cols.update({this_columns[i] for i in this_unnamed_cols})
                if (len(columns) == 1):
                    num_original_columns = len(this_columns)
            if clear_buffer:
                self._clear_buffer()
            if (names is not None):
                if (len(names) > len(columns[0])):
                    raise ValueError('Number of passed names did not match number of header fields in the file')
                if (len(columns) > 1):
                    raise TypeError('Cannot pass names with multi-index columns')
                if (self.usecols is not None):
                    self._handle_usecols(columns, names)
                else:
                    self._col_indices = None
                    num_original_columns = len(names)
                columns = [names]
            else:
                columns = self._handle_usecols(columns, columns[0])
        else:
            try:
                line = self._buffered_line()
            except StopIteration as err:
                if (not names):
                    raise EmptyDataError('No columns to parse from file') from err
                line = names[:]
            ncols = len(line)
            num_original_columns = ncols
            if (not names):
                if self.prefix:
                    columns = [[f'{self.prefix}{i}' for i in range(ncols)]]
                else:
                    columns = [list(range(ncols))]
                columns = self._handle_usecols(columns, columns[0])
            elif ((self.usecols is None) or (len(names) >= num_original_columns)):
                columns = self._handle_usecols([names], names)
                num_original_columns = len(names)
            else:
                if ((not callable(self.usecols)) and (len(names) != len(self.usecols))):
                    raise ValueError('Number of passed names did not match number of header fields in the file')
                self._handle_usecols([names], names)
                columns = [names]
                num_original_columns = ncols
        return (columns, num_original_columns, unnamed_cols)

    def _handle_usecols(self, columns, usecols_key):
        '\n        Sets self._col_indices\n\n        usecols_key is used if there are string usecols.\n        '
        if (self.usecols is not None):
            if callable(self.usecols):
                col_indices = _evaluate_usecols(self.usecols, usecols_key)
            elif any((isinstance(u, str) for u in self.usecols)):
                if (len(columns) > 1):
                    raise ValueError('If using multiple headers, usecols must be integers.')
                col_indices = []
                for col in self.usecols:
                    if isinstance(col, str):
                        try:
                            col_indices.append(usecols_key.index(col))
                        except ValueError:
                            _validate_usecols_names(self.usecols, usecols_key)
                    else:
                        col_indices.append(col)
            else:
                col_indices = self.usecols
            columns = [[n for (i, n) in enumerate(column) if (i in col_indices)] for column in columns]
            self._col_indices = col_indices
        return columns

    def _buffered_line(self):
        '\n        Return a line from buffer, filling buffer if required.\n        '
        if (len(self.buf) > 0):
            return self.buf[0]
        else:
            return self._next_line()

    def _check_for_bom(self, first_row):
        '\n        Checks whether the file begins with the BOM character.\n        If it does, remove it. In addition, if there is quoting\n        in the field subsequent to the BOM, remove it as well\n        because it technically takes place at the beginning of\n        the name, not the middle of it.\n        '
        if (not first_row):
            return first_row
        if (not isinstance(first_row[0], str)):
            return first_row
        if (not first_row[0]):
            return first_row
        first_elt = first_row[0][0]
        if (first_elt != _BOM):
            return first_row
        first_row_bom = first_row[0]
        if ((len(first_row_bom) > 1) and (first_row_bom[1] == self.quotechar)):
            start = 2
            quote = first_row_bom[1]
            end = (first_row_bom[2:].index(quote) + 2)
            new_row = first_row_bom[start:end]
            if (len(first_row_bom) > (end + 1)):
                new_row += first_row_bom[(end + 1):]
        else:
            new_row = first_row_bom[1:]
        return ([new_row] + first_row[1:])

    def _is_line_empty(self, line):
        '\n        Check if a line is empty or not.\n\n        Parameters\n        ----------\n        line : str, array-like\n            The line of data to check.\n\n        Returns\n        -------\n        boolean : Whether or not the line is empty.\n        '
        return ((not line) or all(((not x) for x in line)))

    def _next_line(self):
        if isinstance(self.data, list):
            while self.skipfunc(self.pos):
                self.pos += 1
            while True:
                try:
                    line = self._check_comments([self.data[self.pos]])[0]
                    self.pos += 1
                    if ((not self.skip_blank_lines) and (self._is_line_empty(self.data[(self.pos - 1)]) or line)):
                        break
                    elif self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                except IndexError:
                    raise StopIteration
        else:
            while self.skipfunc(self.pos):
                self.pos += 1
                assert (self.data is not None)
                next(self.data)
            while True:
                orig_line = self._next_iter_line(row_num=(self.pos + 1))
                self.pos += 1
                if (orig_line is not None):
                    line = self._check_comments([orig_line])[0]
                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                    elif (self._is_line_empty(orig_line) or line):
                        break
        if (self.pos == 1):
            line = self._check_for_bom(line)
        self.line_pos += 1
        self.buf.append(line)
        return line

    def _alert_malformed(self, msg, row_num):
        '\n        Alert a user about a malformed row.\n\n        If `self.error_bad_lines` is True, the alert will be `ParserError`.\n        If `self.warn_bad_lines` is True, the alert will be printed out.\n\n        Parameters\n        ----------\n        msg : The error message to display.\n        row_num : The row number where the parsing error occurred.\n                  Because this row number is displayed, we 1-index,\n                  even though we 0-index internally.\n        '
        if self.error_bad_lines:
            raise ParserError(msg)
        elif self.warn_bad_lines:
            base = f'Skipping line {row_num}: '
            sys.stderr.write(((base + msg) + '\n'))

    def _next_iter_line(self, row_num):
        '\n        Wrapper around iterating through `self.data` (CSV source).\n\n        When a CSV error is raised, we check for specific\n        error messages that allow us to customize the\n        error message displayed to the user.\n\n        Parameters\n        ----------\n        row_num : The row number of the line being parsed.\n        '
        try:
            assert (self.data is not None)
            return next(self.data)
        except csv.Error as e:
            if (self.warn_bad_lines or self.error_bad_lines):
                msg = str(e)
                if (('NULL byte' in msg) or ('line contains NUL' in msg)):
                    msg = "NULL byte detected. This byte cannot be processed in Python's native csv library at the moment, so please pass in engine='c' instead"
                if (self.skipfooter > 0):
                    reason = "Error could possibly be due to parsing errors in the skipped footer rows (the skipfooter keyword is only applied after Python's csv library has parsed all rows)."
                    msg += ('. ' + reason)
                self._alert_malformed(msg, row_num)
            return None

    def _check_comments(self, lines):
        if (self.comment is None):
            return lines
        ret = []
        for line in lines:
            rl = []
            for x in line:
                if ((not isinstance(x, str)) or (self.comment not in x) or (x in self.na_values)):
                    rl.append(x)
                else:
                    x = x[:x.find(self.comment)]
                    if (len(x) > 0):
                        rl.append(x)
                    break
            ret.append(rl)
        return ret

    def _remove_empty_lines(self, lines):
        '\n        Iterate through the lines and remove any that are\n        either empty or contain only one whitespace value\n\n        Parameters\n        ----------\n        lines : array-like\n            The array of lines that we are to filter.\n\n        Returns\n        -------\n        filtered_lines : array-like\n            The same array of lines with the "empty" ones removed.\n        '
        ret = []
        for line in lines:
            if ((len(line) > 1) or ((len(line) == 1) and ((not isinstance(line[0], str)) or line[0].strip()))):
                ret.append(line)
        return ret

    def _check_thousands(self, lines):
        if (self.thousands is None):
            return lines
        return self._search_replace_num_columns(lines=lines, search=self.thousands, replace='')

    def _search_replace_num_columns(self, lines, search, replace):
        ret = []
        for line in lines:
            rl = []
            for (i, x) in enumerate(line):
                if ((not isinstance(x, str)) or (search not in x) or (self._no_thousands_columns and (i in self._no_thousands_columns)) or self.nonnum.search(x.strip())):
                    rl.append(x)
                else:
                    rl.append(x.replace(search, replace))
            ret.append(rl)
        return ret

    def _check_decimal(self, lines):
        if (self.decimal == _parser_defaults['decimal']):
            return lines
        return self._search_replace_num_columns(lines=lines, search=self.decimal, replace='.')

    def _clear_buffer(self):
        self.buf = []
    _implicit_index = False

    def _get_index_name(self, columns):
        '\n        Try several cases to get lines:\n\n        0) There are headers on row 0 and row 1 and their\n        total summed lengths equals the length of the next line.\n        Treat row 0 as columns and row 1 as indices\n        1) Look for implicit index: there are more columns\n        on row 1 than row 0. If this is true, assume that row\n        1 lists index columns and row 0 lists normal columns.\n        2) Get index from the columns if it was listed.\n        '
        orig_names = list(columns)
        columns = list(columns)
        try:
            line = self._next_line()
        except StopIteration:
            line = None
        try:
            next_line = self._next_line()
        except StopIteration:
            next_line = None
        implicit_first_cols = 0
        if (line is not None):
            if (self.index_col is not False):
                implicit_first_cols = (len(line) - self.num_original_columns)
            if (next_line is not None):
                if (len(next_line) == (len(line) + self.num_original_columns)):
                    self.index_col = list(range(len(line)))
                    self.buf = self.buf[1:]
                    for c in reversed(line):
                        columns.insert(0, c)
                    orig_names = list(columns)
                    self.num_original_columns = len(columns)
                    return (line, orig_names, columns)
        if (implicit_first_cols > 0):
            self._implicit_index = True
            if (self.index_col is None):
                self.index_col = list(range(implicit_first_cols))
            index_name = None
        else:
            (index_name, columns_, self.index_col) = _clean_index_names(columns, self.index_col, self.unnamed_cols)
        return (index_name, orig_names, columns)

    def _rows_to_cols(self, content):
        col_len = self.num_original_columns
        if self._implicit_index:
            col_len += len(self.index_col)
        max_len = max((len(row) for row in content))
        if ((max_len > col_len) and (self.index_col is not False) and (self.usecols is None)):
            footers = (self.skipfooter if self.skipfooter else 0)
            bad_lines = []
            iter_content = enumerate(content)
            content_len = len(content)
            content = []
            for (i, l) in iter_content:
                actual_len = len(l)
                if (actual_len > col_len):
                    if (self.error_bad_lines or self.warn_bad_lines):
                        row_num = (self.pos - ((content_len - i) + footers))
                        bad_lines.append((row_num, actual_len))
                        if self.error_bad_lines:
                            break
                else:
                    content.append(l)
            for (row_num, actual_len) in bad_lines:
                msg = f'Expected {col_len} fields in line {(row_num + 1)}, saw {actual_len}'
                if (self.delimiter and (len(self.delimiter) > 1) and (self.quoting != csv.QUOTE_NONE)):
                    reason = 'Error could possibly be due to quotes being ignored when a multi-char delimiter is used.'
                    msg += ('. ' + reason)
                self._alert_malformed(msg, (row_num + 1))
        zipped_content = list(lib.to_object_array(content, min_width=col_len).T)
        if self.usecols:
            if self._implicit_index:
                zipped_content = [a for (i, a) in enumerate(zipped_content) if ((i < len(self.index_col)) or ((i - len(self.index_col)) in self._col_indices))]
            else:
                zipped_content = [a for (i, a) in enumerate(zipped_content) if (i in self._col_indices)]
        return zipped_content

    def _get_lines(self, rows=None):
        lines = self.buf
        new_rows = None
        if (rows is not None):
            if (len(self.buf) >= rows):
                (new_rows, self.buf) = (self.buf[:rows], self.buf[rows:])
            else:
                rows -= len(self.buf)
        if (new_rows is None):
            if isinstance(self.data, list):
                if (self.pos > len(self.data)):
                    raise StopIteration
                if (rows is None):
                    new_rows = self.data[self.pos:]
                    new_pos = len(self.data)
                else:
                    new_rows = self.data[self.pos:(self.pos + rows)]
                    new_pos = (self.pos + rows)
                if self.skiprows:
                    new_rows = [row for (i, row) in enumerate(new_rows) if (not self.skipfunc((i + self.pos)))]
                lines.extend(new_rows)
                self.pos = new_pos
            else:
                new_rows = []
                try:
                    if (rows is not None):
                        for _ in range(rows):
                            assert (self.data is not None)
                            new_rows.append(next(self.data))
                        lines.extend(new_rows)
                    else:
                        rows = 0
                        while True:
                            new_row = self._next_iter_line(row_num=((self.pos + rows) + 1))
                            rows += 1
                            if (new_row is not None):
                                new_rows.append(new_row)
                except StopIteration:
                    if self.skiprows:
                        new_rows = [row for (i, row) in enumerate(new_rows) if (not self.skipfunc((i + self.pos)))]
                    lines.extend(new_rows)
                    if (len(lines) == 0):
                        raise
                self.pos += len(new_rows)
            self.buf = []
        else:
            lines = new_rows
        if self.skipfooter:
            lines = lines[:(- self.skipfooter)]
        lines = self._check_comments(lines)
        if self.skip_blank_lines:
            lines = self._remove_empty_lines(lines)
        lines = self._check_thousands(lines)
        return self._check_decimal(lines)

def _make_date_converter(date_parser=None, dayfirst=False, infer_datetime_format=False, cache_dates=True):

    def converter(*date_cols):
        if (date_parser is None):
            strs = parsing.concat_date_cols(date_cols)
            try:
                return tools.to_datetime(ensure_object(strs), utc=None, dayfirst=dayfirst, errors='ignore', infer_datetime_format=infer_datetime_format, cache=cache_dates).to_numpy()
            except ValueError:
                return tools.to_datetime(parsing.try_parse_dates(strs, dayfirst=dayfirst), cache=cache_dates)
        else:
            try:
                result = tools.to_datetime(date_parser(*date_cols), errors='ignore', cache=cache_dates)
                if isinstance(result, datetime.datetime):
                    raise Exception('scalar parser')
                return result
            except Exception:
                try:
                    return tools.to_datetime(parsing.try_parse_dates(parsing.concat_date_cols(date_cols), parser=date_parser, dayfirst=dayfirst), errors='ignore')
                except Exception:
                    return generic_parser(date_parser, *date_cols)
    return converter

def _process_date_conversion(data_dict, converter, parse_spec, index_col, index_names, columns, keep_date_col=False):

    def _isindex(colspec):
        return ((isinstance(index_col, list) and (colspec in index_col)) or (isinstance(index_names, list) and (colspec in index_names)))
    new_cols = []
    new_data = {}
    orig_names = columns
    columns = list(columns)
    date_cols = set()
    if ((parse_spec is None) or isinstance(parse_spec, bool)):
        return (data_dict, columns)
    if isinstance(parse_spec, list):
        for colspec in parse_spec:
            if is_scalar(colspec):
                if (isinstance(colspec, int) and (colspec not in data_dict)):
                    colspec = orig_names[colspec]
                if _isindex(colspec):
                    continue
                data_dict[colspec] = converter(data_dict[colspec])
            else:
                (new_name, col, old_names) = _try_convert_dates(converter, colspec, data_dict, orig_names)
                if (new_name in data_dict):
                    raise ValueError(f'New date column already in dict {new_name}')
                new_data[new_name] = col
                new_cols.append(new_name)
                date_cols.update(old_names)
    elif isinstance(parse_spec, dict):
        for (new_name, colspec) in parse_spec.items():
            if (new_name in data_dict):
                raise ValueError(f'Date column {new_name} already in dict')
            (_, col, old_names) = _try_convert_dates(converter, colspec, data_dict, orig_names)
            new_data[new_name] = col
            new_cols.append(new_name)
            date_cols.update(old_names)
    data_dict.update(new_data)
    new_cols.extend(columns)
    if (not keep_date_col):
        for c in list(date_cols):
            data_dict.pop(c)
            new_cols.remove(c)
    return (data_dict, new_cols)

def _try_convert_dates(parser, colspec, data_dict, columns):
    colset = set(columns)
    colnames = []
    for c in colspec:
        if (c in colset):
            colnames.append(c)
        elif (isinstance(c, int) and (c not in columns)):
            colnames.append(columns[c])
        else:
            colnames.append(c)
    new_name = '_'.join((str(x) for x in colnames))
    to_parse = [data_dict[c] for c in colnames if (c in data_dict)]
    new_col = parser(*to_parse)
    return (new_name, new_col, colnames)

def _clean_na_values(na_values, keep_default_na=True):
    if (na_values is None):
        if keep_default_na:
            na_values = STR_NA_VALUES
        else:
            na_values = set()
        na_fvalues = set()
    elif isinstance(na_values, dict):
        old_na_values = na_values.copy()
        na_values = {}
        for (k, v) in old_na_values.items():
            if (not is_list_like(v)):
                v = [v]
            if keep_default_na:
                v = (set(v) | STR_NA_VALUES)
            na_values[k] = v
        na_fvalues = {k: _floatify_na_values(v) for (k, v) in na_values.items()}
    else:
        if (not is_list_like(na_values)):
            na_values = [na_values]
        na_values = _stringify_na_values(na_values)
        if keep_default_na:
            na_values = (na_values | STR_NA_VALUES)
        na_fvalues = _floatify_na_values(na_values)
    return (na_values, na_fvalues)

def _clean_index_names(columns, index_col, unnamed_cols):
    if (not _is_index_col(index_col)):
        return (None, columns, index_col)
    columns = list(columns)
    if (not columns):
        return (([None] * len(index_col)), columns, index_col)
    cp_cols = list(columns)
    index_names = []
    index_col = list(index_col)
    for (i, c) in enumerate(index_col):
        if isinstance(c, str):
            index_names.append(c)
            for (j, name) in enumerate(cp_cols):
                if (name == c):
                    index_col[i] = j
                    columns.remove(name)
                    break
        else:
            name = cp_cols[c]
            columns.remove(name)
            index_names.append(name)
    for (i, name) in enumerate(index_names):
        if (isinstance(name, str) and (name in unnamed_cols)):
            index_names[i] = None
    return (index_names, columns, index_col)

def _get_empty_meta(columns, index_col, index_names, dtype=None):
    columns = list(columns)
    if (not is_dict_like(dtype)):
        default_dtype = (dtype or object)
        dtype = defaultdict((lambda : default_dtype))
    else:
        dtype = cast(dict, dtype)
        dtype = defaultdict((lambda : object), {(columns[k] if is_integer(k) else k): v for (k, v) in dtype.items()})
    if (((index_col is None) or (index_col is False)) or (index_names is None)):
        index = Index([])
    else:
        data = [Series([], dtype=dtype[name]) for name in index_names]
        index = ensure_index_from_sequences(data, names=index_names)
        index_col.sort()
        for (i, n) in enumerate(index_col):
            columns.pop((n - i))
    col_dict = {col_name: Series([], dtype=dtype[col_name]) for col_name in columns}
    return (index, columns, col_dict)

def _floatify_na_values(na_values):
    result = set()
    for v in na_values:
        try:
            v = float(v)
            if (not np.isnan(v)):
                result.add(v)
        except (TypeError, ValueError, OverflowError):
            pass
    return result

def _stringify_na_values(na_values):
    ' return a stringified and numeric for these values '
    result = []
    for x in na_values:
        result.append(str(x))
        result.append(x)
        try:
            v = float(x)
            if (v == int(v)):
                v = int(v)
                result.append(f'{v}.0')
                result.append(str(v))
            result.append(v)
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            result.append(int(x))
        except (TypeError, ValueError, OverflowError):
            pass
    return set(result)

def _get_na_values(col, na_values, na_fvalues, keep_default_na):
    '\n    Get the NaN values for a given column.\n\n    Parameters\n    ----------\n    col : str\n        The name of the column.\n    na_values : array-like, dict\n        The object listing the NaN values as strings.\n    na_fvalues : array-like, dict\n        The object listing the NaN values as floats.\n    keep_default_na : bool\n        If `na_values` is a dict, and the column is not mapped in the\n        dictionary, whether to return the default NaN values or the empty set.\n\n    Returns\n    -------\n    nan_tuple : A length-two tuple composed of\n\n        1) na_values : the string NaN values for that column.\n        2) na_fvalues : the float NaN values for that column.\n    '
    if isinstance(na_values, dict):
        if (col in na_values):
            return (na_values[col], na_fvalues[col])
        else:
            if keep_default_na:
                return (STR_NA_VALUES, set())
            return (set(), set())
    else:
        return (na_values, na_fvalues)

def _get_col_names(colspec, columns):
    colset = set(columns)
    colnames = []
    for c in colspec:
        if (c in colset):
            colnames.append(c)
        elif isinstance(c, int):
            colnames.append(columns[c])
    return colnames

class FixedWidthReader(abc.Iterator):
    '\n    A reader of fixed-width lines.\n    '

    def __init__(self, f, colspecs, delimiter, comment, skiprows=None, infer_nrows=100):
        self.f = f
        self.buffer = None
        self.delimiter = (('\r\n' + delimiter) if delimiter else '\n\r\t ')
        self.comment = comment
        if (colspecs == 'infer'):
            self.colspecs = self.detect_colspecs(infer_nrows=infer_nrows, skiprows=skiprows)
        else:
            self.colspecs = colspecs
        if (not isinstance(self.colspecs, (tuple, list))):
            raise TypeError(f'column specifications must be a list or tuple, input was a {type(colspecs).__name__}')
        for colspec in self.colspecs:
            if (not (isinstance(colspec, (tuple, list)) and (len(colspec) == 2) and isinstance(colspec[0], (int, np.integer, type(None))) and isinstance(colspec[1], (int, np.integer, type(None))))):
                raise TypeError('Each column specification must be 2 element tuple or list of integers')

    def get_rows(self, infer_nrows, skiprows=None):
        "\n        Read rows from self.f, skipping as specified.\n\n        We distinguish buffer_rows (the first <= infer_nrows\n        lines) from the rows returned to detect_colspecs\n        because it's simpler to leave the other locations\n        with skiprows logic alone than to modify them to\n        deal with the fact we skipped some rows here as\n        well.\n\n        Parameters\n        ----------\n        infer_nrows : int\n            Number of rows to read from self.f, not counting\n            rows that are skipped.\n        skiprows: set, optional\n            Indices of rows to skip.\n\n        Returns\n        -------\n        detect_rows : list of str\n            A list containing the rows to read.\n\n        "
        if (skiprows is None):
            skiprows = set()
        buffer_rows = []
        detect_rows = []
        for (i, row) in enumerate(self.f):
            if (i not in skiprows):
                detect_rows.append(row)
            buffer_rows.append(row)
            if (len(detect_rows) >= infer_nrows):
                break
        self.buffer = iter(buffer_rows)
        return detect_rows

    def detect_colspecs(self, infer_nrows=100, skiprows=None):
        delimiters = ''.join((f'\{x}' for x in self.delimiter))
        pattern = re.compile(f'([^{delimiters}]+)')
        rows = self.get_rows(infer_nrows, skiprows)
        if (not rows):
            raise EmptyDataError('No rows from which to infer column width')
        max_len = max(map(len, rows))
        mask = np.zeros((max_len + 1), dtype=int)
        if (self.comment is not None):
            rows = [row.partition(self.comment)[0] for row in rows]
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start():m.end()] = 1
        shifted = np.roll(mask, 1)
        shifted[0] = 0
        edges = np.where(((mask ^ shifted) == 1))[0]
        edge_pairs = list(zip(edges[::2], edges[1::2]))
        return edge_pairs

    def __next__(self):
        if (self.buffer is not None):
            try:
                line = next(self.buffer)
            except StopIteration:
                self.buffer = None
                line = next(self.f)
        else:
            line = next(self.f)
        return [line[fromm:to].strip(self.delimiter) for (fromm, to) in self.colspecs]

class FixedWidthFieldParser(PythonParser):
    '\n    Specialization that Converts fixed-width fields into DataFrames.\n    See PythonParser for details.\n    '

    def __init__(self, f, **kwds):
        self.colspecs = kwds.pop('colspecs')
        self.infer_nrows = kwds.pop('infer_nrows')
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f):
        self.data = FixedWidthReader(f, self.colspecs, self.delimiter, self.comment, self.skiprows, self.infer_nrows)

    def _remove_empty_lines(self, lines):
        '\n        Returns the list of lines without the empty ones. With fixed-width\n        fields, empty lines become arrays of empty strings.\n\n        See PythonParser._remove_empty_lines.\n        '
        return [line for line in lines if any((((not isinstance(e, str)) or e.strip()) for e in line))]

def _refine_defaults_read(dialect, delimiter, delim_whitespace, engine, sep, defaults):
    "Validate/refine default values of input parameters of read_csv, read_table.\n\n    Parameters\n    ----------\n    dialect : str or csv.Dialect\n        If provided, this parameter will override values (default or not) for the\n        following parameters: `delimiter`, `doublequote`, `escapechar`,\n        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to\n        override values, a ParserWarning will be issued. See csv.Dialect\n        documentation for more details.\n    delimiter : str or object\n        Alias for sep.\n    delim_whitespace : bool\n        Specifies whether or not whitespace (e.g. ``' '`` or ``'\t'``) will be\n        used as the sep. Equivalent to setting ``sep='\\s+'``. If this option\n        is set to True, nothing should be passed in for the ``delimiter``\n        parameter.\n    engine : {{'c', 'python'}}\n        Parser engine to use. The C engine is faster while the python engine is\n        currently more feature-complete.\n    sep : str or object\n        A delimiter provided by the user (str) or a sentinel value, i.e.\n        pandas._libs.lib.no_default.\n    defaults: dict\n        Default values of input parameters.\n\n    Returns\n    -------\n    kwds : dict\n        Input parameters with correct values.\n\n    Raises\n    ------\n    ValueError : If a delimiter was specified with ``sep`` (or ``delimiter``) and\n        ``delim_whitespace=True``.\n    "
    delim_default = defaults['delimiter']
    kwds: Dict[(str, Any)] = {}
    if (dialect is not None):
        kwds['sep_override'] = ((delimiter is None) and ((sep is lib.no_default) or (sep == delim_default)))
    if (delimiter is None):
        delimiter = sep
    if (delim_whitespace and (delimiter is not lib.no_default)):
        raise ValueError('Specified a delimiter with both sep and delim_whitespace=True; you can only specify one.')
    if (delimiter is lib.no_default):
        kwds['delimiter'] = delim_default
    else:
        kwds['delimiter'] = delimiter
    if (engine is not None):
        kwds['engine_specified'] = True
    else:
        kwds['engine'] = 'c'
        kwds['engine_specified'] = False
    return kwds

def _extract_dialect(kwds):
    '\n    Extract concrete csv dialect instance.\n\n    Returns\n    -------\n    csv.Dialect or None\n    '
    if (kwds.get('dialect') is None):
        return None
    dialect = kwds['dialect']
    if (dialect in csv.list_dialects()):
        dialect = csv.get_dialect(dialect)
    _validate_dialect(dialect)
    return dialect
MANDATORY_DIALECT_ATTRS = ('delimiter', 'doublequote', 'escapechar', 'skipinitialspace', 'quotechar', 'quoting')

def _validate_dialect(dialect):
    '\n    Validate csv dialect instance.\n\n    Raises\n    ------\n    ValueError\n        If incorrect dialect is provided.\n    '
    for param in MANDATORY_DIALECT_ATTRS:
        if (not hasattr(dialect, param)):
            raise ValueError(f'Invalid dialect {dialect} provided')

def _merge_with_dialect_properties(dialect, defaults):
    '\n    Merge default kwargs in TextFileReader with dialect parameters.\n\n    Parameters\n    ----------\n    dialect : csv.Dialect\n        Concrete csv dialect. See csv.Dialect documentation for more details.\n    defaults : dict\n        Keyword arguments passed to TextFileReader.\n\n    Returns\n    -------\n    kwds : dict\n        Updated keyword arguments, merged with dialect parameters.\n    '
    kwds = defaults.copy()
    for param in MANDATORY_DIALECT_ATTRS:
        dialect_val = getattr(dialect, param)
        parser_default = _parser_defaults[param]
        provided = kwds.get(param, parser_default)
        conflict_msgs = []
        if ((provided != parser_default) and (provided != dialect_val)):
            msg = f"Conflicting values for '{param}': '{provided}' was provided, but the dialect specifies '{dialect_val}'. Using the dialect-specified value."
            if (not ((param == 'delimiter') and kwds.pop('sep_override', False))):
                conflict_msgs.append(msg)
        if conflict_msgs:
            warnings.warn('\n\n'.join(conflict_msgs), ParserWarning, stacklevel=2)
        kwds[param] = dialect_val
    return kwds

def _validate_skipfooter(kwds):
    '\n    Check whether skipfooter is compatible with other kwargs in TextFileReader.\n\n    Parameters\n    ----------\n    kwds : dict\n        Keyword arguments passed to TextFileReader.\n\n    Raises\n    ------\n    ValueError\n        If skipfooter is not compatible with other parameters.\n    '
    if kwds.get('skipfooter'):
        if (kwds.get('iterator') or kwds.get('chunksize')):
            raise ValueError("'skipfooter' not supported for iteration")
        if kwds.get('nrows'):
            raise ValueError("'skipfooter' not supported with 'nrows'")
