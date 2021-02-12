
import abc
import datetime
from distutils.version import LooseVersion
import inspect
from io import BufferedIOBase, BytesIO, RawIOBase
import os
from textwrap import fill
from typing import IO, Any, Dict, Mapping, Optional, Union, cast
import warnings
import zipfile
from pandas._config import config
from pandas._libs.parsers import STR_NA_VALUES
from pandas._typing import Buffer, DtypeArg, FilePathOrBuffer, StorageOptions
from pandas.compat._optional import import_optional_dependency
from pandas.errors import EmptyDataError
from pandas.util._decorators import Appender, deprecate_nonkeyword_arguments, doc
from pandas.core.dtypes.common import is_bool, is_float, is_integer, is_list_like
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import IOHandles, get_handle, stringify_path, validate_header_arg
from pandas.io.excel._util import fill_mi_header, get_default_writer, get_writer, maybe_convert_usecols, pop_header_name
from pandas.io.parsers import TextParser
_read_excel_doc = (('\nRead an Excel file into a pandas DataFrame.\n\nSupports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt` file extensions\nread from a local filesystem or URL. Supports an option to read\na single sheet or a list of sheets.\n\nParameters\n----------\nio : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object\n    Any valid string path is acceptable. The string could be a URL. Valid\n    URL schemes include http, ftp, s3, and file. For file URLs, a host is\n    expected. A local file could be: ``file://localhost/path/to/table.xlsx``.\n\n    If you want to pass in a path object, pandas accepts any ``os.PathLike``.\n\n    By file-like object, we refer to objects with a ``read()`` method,\n    such as a file handle (e.g. via builtin ``open`` function)\n    or ``StringIO``.\nsheet_name : str, int, list, or None, default 0\n    Strings are used for sheet names. Integers are used in zero-indexed\n    sheet positions. Lists of strings/integers are used to request\n    multiple sheets. Specify None to get all sheets.\n\n    Available cases:\n\n    * Defaults to ``0``: 1st sheet as a `DataFrame`\n    * ``1``: 2nd sheet as a `DataFrame`\n    * ``"Sheet1"``: Load sheet with name "Sheet1"\n    * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5"\n      as a dict of `DataFrame`\n    * None: All sheets.\n\nheader : int, list of int, default 0\n    Row (0-indexed) to use for the column labels of the parsed\n    DataFrame. If a list of integers is passed those row positions will\n    be combined into a ``MultiIndex``. Use None if there is no header.\nnames : array-like, default None\n    List of column names to use. If file contains no header row,\n    then you should explicitly pass header=None.\nindex_col : int, list of int, default None\n    Column (0-indexed) to use as the row labels of the DataFrame.\n    Pass None if there is no such column.  If a list is passed,\n    those columns will be combined into a ``MultiIndex``.  If a\n    subset of data is selected with ``usecols``, index_col\n    is based on the subset.\nusecols : int, str, list-like, or callable default None\n    * If None, then parse all columns.\n    * If str, then indicates comma separated list of Excel column letters\n      and column ranges (e.g. "A:E" or "A,C,E:F"). Ranges are inclusive of\n      both sides.\n    * If list of int, then indicates list of column numbers to be parsed.\n    * If list of string, then indicates list of column names to be parsed.\n\n      .. versionadded:: 0.24.0\n\n    * If callable, then evaluate each column name against it and parse the\n      column if the callable returns ``True``.\n\n    Returns a subset of the columns according to behavior above.\n\n      .. versionadded:: 0.24.0\n\nsqueeze : bool, default False\n    If the parsed data only contains one column then return a Series.\ndtype : Type name or dict of column -> type, default None\n    Data type for data or columns. E.g. {\'a\': np.float64, \'b\': np.int32}\n    Use `object` to preserve data as stored in Excel and not interpret dtype.\n    If converters are specified, they will be applied INSTEAD\n    of dtype conversion.\nengine : str, default None\n    If io is not a buffer or path, this must be set to identify io.\n    Supported engines: "xlrd", "openpyxl", "odf", "pyxlsb".\n    Engine compatibility :\n\n    - "xlrd" supports old-style Excel files (.xls).\n    - "openpyxl" supports newer Excel file formats.\n    - "odf" supports OpenDocument file formats (.odf, .ods, .odt).\n    - "pyxlsb" supports Binary Excel files.\n\n    .. versionchanged:: 1.2.0\n        The engine `xlrd <https://xlrd.readthedocs.io/en/latest/>`_\n        now only supports old-style ``.xls`` files.\n        When ``engine=None``, the following logic will be\n        used to determine the engine:\n\n       - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),\n         then `odf <https://pypi.org/project/odfpy/>`_ will be used.\n       - Otherwise if ``path_or_buffer`` is an xls format,\n         ``xlrd`` will be used.\n       - Otherwise if `openpyxl <https://pypi.org/project/openpyxl/>`_ is installed,\n         then ``openpyxl`` will be used.\n       - Otherwise if ``xlrd >= 2.0`` is installed, a ``ValueError`` will be raised.\n       - Otherwise ``xlrd`` will be used and a ``FutureWarning`` will be raised. This\n         case will raise a ``ValueError`` in a future version of pandas.\n\nconverters : dict, default None\n    Dict of functions for converting values in certain columns. Keys can\n    either be integers or column labels, values are functions that take one\n    input argument, the Excel cell content, and return the transformed\n    content.\ntrue_values : list, default None\n    Values to consider as True.\nfalse_values : list, default None\n    Values to consider as False.\nskiprows : list-like, int, or callable, optional\n    Line numbers to skip (0-indexed) or number of lines to skip (int) at the\n    start of the file. If callable, the callable function will be evaluated\n    against the row indices, returning True if the row should be skipped and\n    False otherwise. An example of a valid callable argument would be ``lambda\n    x: x in [0, 2]``.\nnrows : int, default None\n    Number of rows to parse.\nna_values : scalar, str, list-like, or dict, default None\n    Additional strings to recognize as NA/NaN. If dict passed, specific\n    per-column NA values. By default the following values are interpreted\n    as NaN: \'' + fill("', '".join(sorted(STR_NA_VALUES)), 70, subsequent_indent='    ')) + '\'.\nkeep_default_na : bool, default True\n    Whether or not to include the default NaN values when parsing the data.\n    Depending on whether `na_values` is passed in, the behavior is as follows:\n\n    * If `keep_default_na` is True, and `na_values` are specified, `na_values`\n      is appended to the default NaN values used for parsing.\n    * If `keep_default_na` is True, and `na_values` are not specified, only\n      the default NaN values are used for parsing.\n    * If `keep_default_na` is False, and `na_values` are specified, only\n      the NaN values specified `na_values` are used for parsing.\n    * If `keep_default_na` is False, and `na_values` are not specified, no\n      strings will be parsed as NaN.\n\n    Note that if `na_filter` is passed in as False, the `keep_default_na` and\n    `na_values` parameters will be ignored.\nna_filter : bool, default True\n    Detect missing value markers (empty strings and the value of na_values). In\n    data without any NAs, passing na_filter=False can improve the performance\n    of reading a large file.\nverbose : bool, default False\n    Indicate number of NA values placed in non-numeric columns.\nparse_dates : bool, list-like, or dict, default False\n    The behavior is as follows:\n\n    * bool. If True -> try parsing the index.\n    * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3\n      each as a separate date column.\n    * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as\n      a single date column.\n    * dict, e.g. {\'foo\' : [1, 3]} -> parse columns 1, 3 as date and call\n      result \'foo\'\n\n    If a column or index contains an unparseable date, the entire column or\n    index will be returned unaltered as an object data type. If you don`t want to\n    parse some cells as date just change their type in Excel to "Text".\n    For non-standard datetime parsing, use ``pd.to_datetime`` after ``pd.read_excel``.\n\n    Note: A fast-path exists for iso8601-formatted dates.\ndate_parser : function, optional\n    Function to use for converting a sequence of string columns to an array of\n    datetime instances. The default uses ``dateutil.parser.parser`` to do the\n    conversion. Pandas will try to call `date_parser` in three different ways,\n    advancing to the next if an exception occurs: 1) Pass one or more arrays\n    (as defined by `parse_dates`) as arguments; 2) concatenate (row-wise) the\n    string values from the columns defined by `parse_dates` into a single array\n    and pass that; and 3) call `date_parser` once for each row using one or\n    more strings (corresponding to the columns defined by `parse_dates`) as\n    arguments.\nthousands : str, default None\n    Thousands separator for parsing string columns to numeric.  Note that\n    this parameter is only necessary for columns stored as TEXT in Excel,\n    any numeric columns will automatically be parsed, regardless of display\n    format.\ncomment : str, default None\n    Comments out remainder of line. Pass a character or characters to this\n    argument to indicate comments in the input file. Any data between the\n    comment string and the end of the current line is ignored.\nskipfooter : int, default 0\n    Rows at the end to skip (0-indexed).\nconvert_float : bool, default True\n    Convert integral floats to int (i.e., 1.0 --> 1). If False, all numeric\n    data will be read in as floats: Excel stores all numbers as floats\n    internally.\nmangle_dupe_cols : bool, default True\n    Duplicate columns will be specified as \'X\', \'X.1\', ...\'X.N\', rather than\n    \'X\'...\'X\'. Passing in False will cause data to be overwritten if there\n    are duplicate names in the columns.\nstorage_options : dict, optional\n    Extra options that make sense for a particular storage connection, e.g.\n    host, port, username, password, etc., if using a URL that will\n    be parsed by ``fsspec``, e.g., starting "s3://", "gcs://". An error\n    will be raised if providing this argument with a local path or\n    a file-like buffer. See the fsspec and backend storage implementation\n    docs for the set of allowed keys and values.\n\n    .. versionadded:: 1.2.0\n\nReturns\n-------\nDataFrame or dict of DataFrames\n    DataFrame from the passed in Excel file. See notes in sheet_name\n    argument for more information on when a dict of DataFrames is returned.\n\nSee Also\n--------\nDataFrame.to_excel : Write DataFrame to an Excel file.\nDataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.\nread_csv : Read a comma-separated values (csv) file into DataFrame.\nread_fwf : Read a table of fixed-width formatted lines into DataFrame.\n\nExamples\n--------\nThe file can be read using the file name as string or an open file object:\n\n>>> pd.read_excel(\'tmp.xlsx\', index_col=0)  # doctest: +SKIP\n       Name  Value\n0   string1      1\n1   string2      2\n2  #Comment      3\n\n>>> pd.read_excel(open(\'tmp.xlsx\', \'rb\'),\n...               sheet_name=\'Sheet3\')  # doctest: +SKIP\n   Unnamed: 0      Name  Value\n0           0   string1      1\n1           1   string2      2\n2           2  #Comment      3\n\nIndex and header can be specified via the `index_col` and `header` arguments\n\n>>> pd.read_excel(\'tmp.xlsx\', index_col=None, header=None)  # doctest: +SKIP\n     0         1      2\n0  NaN      Name  Value\n1  0.0   string1      1\n2  1.0   string2      2\n3  2.0  #Comment      3\n\nColumn types are inferred but can be explicitly specified\n\n>>> pd.read_excel(\'tmp.xlsx\', index_col=0,\n...               dtype={\'Name\': str, \'Value\': float})  # doctest: +SKIP\n       Name  Value\n0   string1    1.0\n1   string2    2.0\n2  #Comment    3.0\n\nTrue, False, and NA values, and thousands separators have defaults,\nbut can be explicitly specified, too. Supply the values you would like\nas strings or lists of strings!\n\n>>> pd.read_excel(\'tmp.xlsx\', index_col=0,\n...               na_values=[\'string1\', \'string2\'])  # doctest: +SKIP\n       Name  Value\n0       NaN      1\n1       NaN      2\n2  #Comment      3\n\nComment lines in the excel input file can be skipped using the `comment` kwarg\n\n>>> pd.read_excel(\'tmp.xlsx\', index_col=0, comment=\'#\')  # doctest: +SKIP\n      Name  Value\n0  string1    1.0\n1  string2    2.0\n2     None    NaN\n')

@deprecate_nonkeyword_arguments(allowed_args=2, version='2.0')
@Appender(_read_excel_doc)
def read_excel(io, sheet_name=0, header=0, names=None, index_col=None, usecols=None, squeeze=False, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, parse_dates=False, date_parser=None, thousands=None, comment=None, skipfooter=0, convert_float=True, mangle_dupe_cols=True, storage_options=None):
    should_close = False
    if (not isinstance(io, ExcelFile)):
        should_close = True
        io = ExcelFile(io, storage_options=storage_options, engine=engine)
    elif (engine and (engine != io.engine)):
        raise ValueError('Engine should not be specified when passing an ExcelFile - ExcelFile already has the engine set')
    try:
        data = io.parse(sheet_name=sheet_name, header=header, names=names, index_col=index_col, usecols=usecols, squeeze=squeeze, dtype=dtype, converters=converters, true_values=true_values, false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values, keep_default_na=keep_default_na, na_filter=na_filter, verbose=verbose, parse_dates=parse_dates, date_parser=date_parser, thousands=thousands, comment=comment, skipfooter=skipfooter, convert_float=convert_float, mangle_dupe_cols=mangle_dupe_cols)
    finally:
        if should_close:
            io.close()
    return data

class BaseExcelReader(metaclass=abc.ABCMeta):

    def __init__(self, filepath_or_buffer, storage_options=None):
        self.handles = IOHandles(handle=filepath_or_buffer, compression={'method': None})
        if (not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class))):
            self.handles = get_handle(filepath_or_buffer, 'rb', storage_options=storage_options, is_text=False)
        if isinstance(self.handles.handle, self._workbook_class):
            self.book = self.handles.handle
        elif hasattr(self.handles.handle, 'read'):
            self.handles.handle.seek(0)
            self.book = self.load_workbook(self.handles.handle)
        elif isinstance(self.handles.handle, bytes):
            self.book = self.load_workbook(BytesIO(self.handles.handle))
        else:
            raise ValueError('Must explicitly set engine if not passing in buffer or path for io.')

    @property
    @abc.abstractmethod
    def _workbook_class(self):
        pass

    @abc.abstractmethod
    def load_workbook(self, filepath_or_buffer):
        pass

    def close(self):
        self.handles.close()

    @property
    @abc.abstractmethod
    def sheet_names(self):
        pass

    @abc.abstractmethod
    def get_sheet_by_name(self, name):
        pass

    @abc.abstractmethod
    def get_sheet_by_index(self, index):
        pass

    @abc.abstractmethod
    def get_sheet_data(self, sheet, convert_float):
        pass

    def parse(self, sheet_name=0, header=0, names=None, index_col=None, usecols=None, squeeze=False, dtype=None, true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None, verbose=False, parse_dates=False, date_parser=None, thousands=None, comment=None, skipfooter=0, convert_float=True, mangle_dupe_cols=True, **kwds):
        validate_header_arg(header)
        ret_dict = False
        if isinstance(sheet_name, list):
            sheets = sheet_name
            ret_dict = True
        elif (sheet_name is None):
            sheets = self.sheet_names
            ret_dict = True
        else:
            sheets = [sheet_name]
        sheets = list(dict.fromkeys(sheets).keys())
        output = {}
        for asheetname in sheets:
            if verbose:
                print(f'Reading sheet {asheetname}')
            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:
                sheet = self.get_sheet_by_index(asheetname)
            data = self.get_sheet_data(sheet, convert_float)
            usecols = maybe_convert_usecols(usecols)
            if (not data):
                output[asheetname] = DataFrame()
                continue
            if (is_list_like(header) and (len(header) == 1)):
                header = header[0]
            header_names = None
            if ((header is not None) and is_list_like(header)):
                header_names = []
                control_row = ([True] * len(data[0]))
                for row in header:
                    if is_integer(skiprows):
                        row += skiprows
                    (data[row], control_row) = fill_mi_header(data[row], control_row)
                    if (index_col is not None):
                        (header_name, _) = pop_header_name(data[row], index_col)
                        header_names.append(header_name)
            has_index_names = (is_list_like(header) and (len(header) > 1))
            if is_list_like(index_col):
                if (header is None):
                    offset = 0
                elif (not is_list_like(header)):
                    offset = (1 + header)
                else:
                    offset = (1 + max(header))
                if has_index_names:
                    offset += 1
                if (offset < len(data)):
                    for col in index_col:
                        last = data[offset][col]
                        for row in range((offset + 1), len(data)):
                            if ((data[row][col] == '') or (data[row][col] is None)):
                                data[row][col] = last
                            else:
                                last = data[row][col]
            try:
                parser = TextParser(data, names=names, header=header, index_col=index_col, has_index_names=has_index_names, squeeze=squeeze, dtype=dtype, true_values=true_values, false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values, parse_dates=parse_dates, date_parser=date_parser, thousands=thousands, comment=comment, skipfooter=skipfooter, usecols=usecols, mangle_dupe_cols=mangle_dupe_cols, **kwds)
                output[asheetname] = parser.read(nrows=nrows)
                if ((not squeeze) or isinstance(output[asheetname], DataFrame)):
                    if header_names:
                        output[asheetname].columns = output[asheetname].columns.set_names(header_names)
            except EmptyDataError:
                output[asheetname] = DataFrame()
        if ret_dict:
            return output
        else:
            return output[asheetname]

class ExcelWriter(metaclass=abc.ABCMeta):
    '\n    Class for writing DataFrame objects into excel sheets.\n\n    Default is to use xlwt for xls, openpyxl for xlsx, odf for ods.\n    See DataFrame.to_excel for typical usage.\n\n    The writer should be used as a context manager. Otherwise, call `close()` to save\n    and close any opened file handles.\n\n    Parameters\n    ----------\n    path : str or typing.BinaryIO\n        Path to xls or xlsx or ods file.\n    engine : str (optional)\n        Engine to use for writing. If None, defaults to\n        ``io.excel.<extension>.writer``.  NOTE: can only be passed as a keyword\n        argument.\n\n        .. deprecated:: 1.2.0\n\n            As the `xlwt <https://pypi.org/project/xlwt/>`__ package is no longer\n            maintained, the ``xlwt`` engine will be removed in a future\n            version of pandas.\n\n    date_format : str, default None\n        Format string for dates written into Excel files (e.g. \'YYYY-MM-DD\').\n    datetime_format : str, default None\n        Format string for datetime objects written into Excel files.\n        (e.g. \'YYYY-MM-DD HH:MM:SS\').\n    mode : {\'w\', \'a\'}, default \'w\'\n        File mode to use (write or append). Append does not work with fsspec URLs.\n\n        .. versionadded:: 0.24.0\n    storage_options : dict, optional\n        Extra options that make sense for a particular storage connection, e.g.\n        host, port, username, password, etc., if using a URL that will\n        be parsed by ``fsspec``, e.g., starting "s3://", "gcs://".\n\n        .. versionadded:: 1.2.0\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n\n    Notes\n    -----\n    None of the methods and properties are considered public.\n\n    For compatibility with CSV writers, ExcelWriter serializes lists\n    and dicts to strings before writing.\n\n    Examples\n    --------\n    Default usage:\n\n    >>> with ExcelWriter(\'path_to_file.xlsx\') as writer:\n    ...     df.to_excel(writer)\n\n    To write to separate sheets in a single file:\n\n    >>> with ExcelWriter(\'path_to_file.xlsx\') as writer:\n    ...     df1.to_excel(writer, sheet_name=\'Sheet1\')\n    ...     df2.to_excel(writer, sheet_name=\'Sheet2\')\n\n    You can set the date format or datetime format:\n\n    >>> with ExcelWriter(\'path_to_file.xlsx\',\n    ...                   date_format=\'YYYY-MM-DD\',\n    ...                   datetime_format=\'YYYY-MM-DD HH:MM:SS\') as writer:\n    ...     df.to_excel(writer)\n\n    You can also append to an existing Excel file:\n\n    >>> with ExcelWriter(\'path_to_file.xlsx\', mode=\'a\') as writer:\n    ...     df.to_excel(writer, sheet_name=\'Sheet3\')\n\n    You can store Excel file in RAM:\n\n    >>> import io\n    >>> buffer = io.BytesIO()\n    >>> with pd.ExcelWriter(buffer) as writer:\n    ...     df.to_excel(writer)\n\n    You can pack Excel file into zip archive:\n\n    >>> import zipfile\n    >>> with zipfile.ZipFile(\'path_to_file.zip\', \'w\') as zf:\n    ...     with zf.open(\'filename.xlsx\', \'w\') as buffer:\n    ...         with pd.ExcelWriter(buffer) as writer:\n    ...             df.to_excel(writer)\n    '

    def __new__(cls, path, engine=None, **kwargs):
        if (cls is ExcelWriter):
            if ((engine is None) or (isinstance(engine, str) and (engine == 'auto'))):
                if isinstance(path, str):
                    ext = os.path.splitext(path)[(- 1)][1:]
                else:
                    ext = 'xlsx'
                try:
                    engine = config.get_option(f'io.excel.{ext}.writer', silent=True)
                    if (engine == 'auto'):
                        engine = get_default_writer(ext)
                except KeyError as err:
                    raise ValueError(f"No engine for filetype: '{ext}'") from err
            if (engine == 'xlwt'):
                xls_config_engine = config.get_option('io.excel.xls.writer', silent=True)
                if (xls_config_engine != 'xlwt'):
                    warnings.warn("As the xlwt package is no longer maintained, the xlwt engine will be removed in a future version of pandas. This is the only engine in pandas that supports writing in the xls format. Install openpyxl and write to an xlsx file instead. You can set the option io.excel.xls.writer to 'xlwt' to silence this warning. While this option is deprecated and will also raise a warning, it can be globally set and the warning suppressed.", FutureWarning, stacklevel=4)
            cls = get_writer(engine)
        return object.__new__(cls)
    curr_sheet = None
    path = None

    @property
    @abc.abstractmethod
    def supported_extensions(self):
        'Extensions that writer engine supports.'
        pass

    @property
    @abc.abstractmethod
    def engine(self):
        'Name of engine.'
        pass

    @abc.abstractmethod
    def write_cells(self, cells, sheet_name=None, startrow=0, startcol=0, freeze_panes=None):
        '\n        Write given formatted cells into Excel an excel sheet\n\n        Parameters\n        ----------\n        cells : generator\n            cell of formatted data to save to Excel sheet\n        sheet_name : str, default None\n            Name of Excel sheet, if None, then use self.cur_sheet\n        startrow : upper left cell row to dump data frame\n        startcol : upper left cell column to dump data frame\n        freeze_panes: int tuple of length 2\n            contains the bottom-most row and right-most column to freeze\n        '
        pass

    @abc.abstractmethod
    def save(self):
        '\n        Save workbook to disk.\n        '
        pass

    def __init__(self, path, engine=None, date_format=None, datetime_format=None, mode='w', storage_options=None, **engine_kwargs):
        if isinstance(path, str):
            ext = os.path.splitext(path)[(- 1)]
            self.check_extension(ext)
        if ('b' not in mode):
            mode += 'b'
        mode = mode.replace('a', 'r+')
        self.handles = IOHandles(cast(Buffer, path), compression={'copression': None})
        if (not isinstance(path, ExcelWriter)):
            self.handles = get_handle(path, mode, storage_options=storage_options, is_text=False)
        self.sheets: Dict[(str, Any)] = {}
        self.cur_sheet = None
        if (date_format is None):
            self.date_format = 'YYYY-MM-DD'
        else:
            self.date_format = date_format
        if (datetime_format is None):
            self.datetime_format = 'YYYY-MM-DD HH:MM:SS'
        else:
            self.datetime_format = datetime_format
        self.mode = mode

    def __fspath__(self):
        return getattr(self.handles.handle, 'name', '')

    def _get_sheet_name(self, sheet_name):
        if (sheet_name is None):
            sheet_name = self.cur_sheet
        if (sheet_name is None):
            raise ValueError('Must pass explicit sheet_name or set cur_sheet property')
        return sheet_name

    def _value_with_fmt(self, val):
        '\n        Convert numpy types to Python types for the Excel writers.\n\n        Parameters\n        ----------\n        val : object\n            Value to be written into cells\n\n        Returns\n        -------\n        Tuple with the first element being the converted value and the second\n            being an optional format\n        '
        fmt = None
        if is_integer(val):
            val = int(val)
        elif is_float(val):
            val = float(val)
        elif is_bool(val):
            val = bool(val)
        elif isinstance(val, datetime.datetime):
            fmt = self.datetime_format
        elif isinstance(val, datetime.date):
            fmt = self.date_format
        elif isinstance(val, datetime.timedelta):
            val = (val.total_seconds() / float(86400))
            fmt = '0'
        else:
            val = str(val)
        return (val, fmt)

    @classmethod
    def check_extension(cls, ext):
        "\n        checks that path's extension against the Writer's supported\n        extensions.  If it isn't supported, raises UnsupportedFiletypeError.\n        "
        if ext.startswith('.'):
            ext = ext[1:]
        if (not any(((ext in extension) for extension in cls.supported_extensions))):
            raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")
        else:
            return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        'synonym for save, to make it more file-like'
        content = self.save()
        self.handles.close()
        return content
XLS_SIGNATURE = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
ZIP_SIGNATURE = b'PK\x03\x04'
PEEK_SIZE = max(len(XLS_SIGNATURE), len(ZIP_SIGNATURE))

@doc(storage_options=_shared_docs['storage_options'])
def inspect_excel_format(path=None, content=None, storage_options=None):
    '\n    Inspect the path or content of an excel file and get its format.\n\n    At least one of path or content must be not None. If both are not None,\n    content will take precedence.\n\n    Adopted from xlrd: https://github.com/python-excel/xlrd.\n\n    Parameters\n    ----------\n    path : str, optional\n        Path to file to inspect. May be a URL.\n    content : file-like object, optional\n        Content of file to inspect.\n    {storage_options}\n\n    Returns\n    -------\n    str\n        Format of file.\n\n    Raises\n    ------\n    ValueError\n        If resulting stream is empty.\n    BadZipFile\n        If resulting stream does not have an XLS signature and is not a valid zipfile.\n    '
    content_or_path: Union[(None, str, BufferedIOBase, RawIOBase, IO[bytes])]
    if isinstance(content, bytes):
        content_or_path = BytesIO(content)
    else:
        content_or_path = (content or path)
    assert (content_or_path is not None)
    with get_handle(content_or_path, 'rb', storage_options=storage_options, is_text=False) as handle:
        stream = handle.handle
        stream.seek(0)
        buf = stream.read(PEEK_SIZE)
        if (buf is None):
            raise ValueError('stream is empty')
        else:
            assert isinstance(buf, bytes)
            peek = buf
        stream.seek(0)
        if peek.startswith(XLS_SIGNATURE):
            return 'xls'
        elif (not peek.startswith(ZIP_SIGNATURE)):
            raise ValueError('File is not a recognized excel file')
        zf = zipfile.ZipFile(stream)
        component_names = [name.replace('\\', '/').lower() for name in zf.namelist()]
        if ('xl/workbook.xml' in component_names):
            return 'xlsx'
        if ('xl/workbook.bin' in component_names):
            return 'xlsb'
        if ('content.xml' in component_names):
            return 'ods'
        return 'zip'

class ExcelFile():
    '\n    Class for parsing tabular excel sheets into DataFrame objects.\n\n    See read_excel for more documentation.\n\n    Parameters\n    ----------\n    path_or_buffer : str, path object (pathlib.Path or py._path.local.LocalPath),\n        a file-like object, xlrd workbook or openpypl workbook.\n        If a string or path object, expected to be a path to a\n        .xls, .xlsx, .xlsb, .xlsm, .odf, .ods, or .odt file.\n    engine : str, default None\n        If io is not a buffer or path, this must be set to identify io.\n        Supported engines: ``xlrd``, ``openpyxl``, ``odf``, ``pyxlsb``\n        Engine compatibility :\n\n        - ``xlrd`` supports old-style Excel files (.xls).\n        - ``openpyxl`` supports newer Excel file formats.\n        - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).\n        - ``pyxlsb`` supports Binary Excel files.\n\n        .. versionchanged:: 1.2.0\n\n           The engine `xlrd <https://xlrd.readthedocs.io/en/latest/>`_\n           now only supports old-style ``.xls`` files.\n           When ``engine=None``, the following logic will be\n           used to determine the engine:\n\n           - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),\n             then `odf <https://pypi.org/project/odfpy/>`_ will be used.\n           - Otherwise if ``path_or_buffer`` is an xls format,\n             ``xlrd`` will be used.\n           - Otherwise if `openpyxl <https://pypi.org/project/openpyxl/>`_ is installed,\n             then ``openpyxl`` will be used.\n           - Otherwise if ``xlrd >= 2.0`` is installed, a ``ValueError`` will be raised.\n           - Otherwise ``xlrd`` will be used and a ``FutureWarning`` will be raised.\n             This case will raise a ``ValueError`` in a future version of pandas.\n\n           .. warning::\n\n            Please do not report issues when using ``xlrd`` to read ``.xlsx`` files.\n            This is not supported, switch to using ``openpyxl`` instead.\n    '
    from pandas.io.excel._odfreader import ODFReader
    from pandas.io.excel._openpyxl import OpenpyxlReader
    from pandas.io.excel._pyxlsb import PyxlsbReader
    from pandas.io.excel._xlrd import XlrdReader
    _engines = {'xlrd': XlrdReader, 'openpyxl': OpenpyxlReader, 'odf': ODFReader, 'pyxlsb': PyxlsbReader}

    def __init__(self, path_or_buffer, engine=None, storage_options=None):
        if ((engine is not None) and (engine not in self._engines)):
            raise ValueError(f'Unknown engine: {engine}')
        self.io = path_or_buffer
        self._io = stringify_path(path_or_buffer)
        if (import_optional_dependency('xlrd', raise_on_missing=False, on_version='ignore') is None):
            xlrd_version = None
        else:
            import xlrd
            xlrd_version = LooseVersion(xlrd.__version__)
        if ((xlrd_version is not None) and isinstance(path_or_buffer, xlrd.Book)):
            ext = 'xls'
        else:
            ext = inspect_excel_format(content=path_or_buffer, storage_options=storage_options)
        if (engine is None):
            if (ext == 'ods'):
                engine = 'odf'
            elif (ext == 'xls'):
                engine = 'xlrd'
            elif (import_optional_dependency('openpyxl', raise_on_missing=False, on_version='ignore') is not None):
                engine = 'openpyxl'
            else:
                engine = 'xlrd'
        if ((engine == 'xlrd') and (ext != 'xls') and (xlrd_version is not None)):
            if (xlrd_version >= '2'):
                raise ValueError(f'Your version of xlrd is {xlrd_version}. In xlrd >= 2.0, only the xls format is supported. Install openpyxl instead.')
            else:
                caller = inspect.stack()[1]
                if (caller.filename.endswith(os.path.join('pandas', 'io', 'excel', '_base.py')) and (caller.function == 'read_excel')):
                    stacklevel = 4
                else:
                    stacklevel = 2
                warnings.warn(f'Your version of xlrd is {xlrd_version}. In xlrd >= 2.0, only the xls format is supported. As a result, the openpyxl engine will be used if it is installed and the engine argument is not specified. Install openpyxl instead.', FutureWarning, stacklevel=stacklevel)
        assert (engine in self._engines), f'Engine {engine} not recognized'
        self.engine = engine
        self.storage_options = storage_options
        self._reader = self._engines[engine](self._io, storage_options=storage_options)

    def __fspath__(self):
        return self._io

    def parse(self, sheet_name=0, header=0, names=None, index_col=None, usecols=None, squeeze=False, converters=None, true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None, parse_dates=False, date_parser=None, thousands=None, comment=None, skipfooter=0, convert_float=True, mangle_dupe_cols=True, **kwds):
        '\n        Parse specified sheet(s) into a DataFrame.\n\n        Equivalent to read_excel(ExcelFile, ...)  See the read_excel\n        docstring for more info on accepted parameters.\n\n        Returns\n        -------\n        DataFrame or dict of DataFrames\n            DataFrame from the passed in Excel file.\n        '
        return self._reader.parse(sheet_name=sheet_name, header=header, names=names, index_col=index_col, usecols=usecols, squeeze=squeeze, converters=converters, true_values=true_values, false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values, parse_dates=parse_dates, date_parser=date_parser, thousands=thousands, comment=comment, skipfooter=skipfooter, convert_float=convert_float, mangle_dupe_cols=mangle_dupe_cols, **kwds)

    @property
    def book(self):
        return self._reader.book

    @property
    def sheet_names(self):
        return self._reader.sheet_names

    def close(self):
        'close io if necessary'
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass
