
'\nModule contains tools for processing Stata files into DataFrames\n\nThe StataReader below was originally written by Joe Presbrey as part of PyDTA.\nIt has been extended and improved by Skipper Seabold from the Statsmodels\nproject who also developed the StataWriter and was finally added to pandas in\na once again improved version.\n\nYou can find more information on http://presbrey.mit.edu/PyDTA and\nhttps://www.statsmodels.org/devel/\n'
from collections import abc
import datetime
from io import BytesIO
import os
from pathlib import Path
import struct
import sys
from typing import Any, AnyStr, Dict, List, Optional, Sequence, Tuple, Union, cast
import warnings
from dateutil.relativedelta import relativedelta
import numpy as np
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas._typing import Buffer, CompressionOptions, FilePathOrBuffer, Label, StorageOptions
from pandas.util._decorators import Appender, doc
from pandas.core.dtypes.common import ensure_object, is_categorical_dtype, is_datetime64_dtype
from pandas import Categorical, DatetimeIndex, NaT, Timestamp, concat, isna, to_datetime, to_timedelta
from pandas.core import generic
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from pandas.io.common import get_handle
_version_error = 'Version of given Stata file is {version}. pandas supports importing versions 105, 108, 111 (Stata 7SE), 113 (Stata 8/9), 114 (Stata 10/11), 115 (Stata 12), 117 (Stata 13), 118 (Stata 14/15/16),and 119 (Stata 15/16, over 32,767 variables).'
_statafile_processing_params1 = 'convert_dates : bool, default True\n    Convert date variables to DataFrame time values.\nconvert_categoricals : bool, default True\n    Read value labels and convert columns to Categorical/Factor variables.'
_statafile_processing_params2 = 'index_col : str, optional\n    Column to set as index.\nconvert_missing : bool, default False\n    Flag indicating whether to convert missing values to their Stata\n    representations.  If False, missing values are replaced with nan.\n    If True, columns containing missing values are returned with\n    object data types and missing values are represented by\n    StataMissingValue objects.\npreserve_dtypes : bool, default True\n    Preserve Stata datatypes. If False, numeric data are upcast to pandas\n    default types for foreign data (float64 or int64).\ncolumns : list or None\n    Columns to retain.  Columns will be returned in the given order.  None\n    returns all columns.\norder_categoricals : bool, default True\n    Flag indicating whether converted categorical data are ordered.'
_chunksize_params = 'chunksize : int, default None\n    Return StataReader object for iterations, returns chunks with\n    given number of lines.'
_iterator_params = 'iterator : bool, default False\n    Return StataReader object.'
_reader_notes = 'Notes\n-----\nCategorical variables read through an iterator may not have the same\ncategories and dtype. This occurs when  a variable stored in a DTA\nfile is associated to an incomplete set of value labels that only\nlabel a strict subset of the values.'
_read_stata_doc = f'''
Read Stata file into DataFrame.

Parameters
----------
filepath_or_buffer : str, path object or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, and file. For file URLs, a host is
    expected. A local file could be: ``file://localhost/path/to/table.dta``.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method,
    such as a file handle (e.g. via builtin ``open`` function)
    or ``StringIO``.
{_statafile_processing_params1}
{_statafile_processing_params2}
{_chunksize_params}
{_iterator_params}

Returns
-------
DataFrame or StataReader

See Also
--------
io.stata.StataReader : Low-level reader for Stata data files.
DataFrame.to_stata: Export Stata data files.

{_reader_notes}

Examples
--------
Read a Stata dta file:

>>> df = pd.read_stata('filename.dta')

Read a Stata dta file in 10,000 line chunks:

>>> itr = pd.read_stata('filename.dta', chunksize=10000)
>>> for chunk in itr:
...     do_something(chunk)
'''
_read_method_doc = f'''Reads observations from Stata file, converting them into a dataframe

Parameters
----------
nrows : int
    Number of lines to read from data file, if None read whole file.
{_statafile_processing_params1}
{_statafile_processing_params2}

Returns
-------
DataFrame
'''
_stata_reader_doc = f'''Class for reading Stata dta files.

Parameters
----------
path_or_buf : path (string), buffer or path object
    string, path object (pathlib.Path or py._path.local.LocalPath) or object
    implementing a binary read() functions.
{_statafile_processing_params1}
{_statafile_processing_params2}
{_chunksize_params}

{_reader_notes}
'''
_date_formats = ['%tc', '%tC', '%td', '%d', '%tw', '%tm', '%tq', '%th', '%ty']
stata_epoch = datetime.datetime(1960, 1, 1)

def _stata_elapsed_date_to_datetime_vec(dates, fmt):
    '\n    Convert from SIF to datetime. https://www.stata.com/help.cgi?datetime\n\n    Parameters\n    ----------\n    dates : Series\n        The Stata Internal Format date to convert to datetime according to fmt\n    fmt : str\n        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty\n        Returns\n\n    Returns\n    -------\n    converted : Series\n        The converted dates\n\n    Examples\n    --------\n    >>> dates = pd.Series([52])\n    >>> _stata_elapsed_date_to_datetime_vec(dates , "%tw")\n    0   1961-01-01\n    dtype: datetime64[ns]\n\n    Notes\n    -----\n    datetime/c - tc\n        milliseconds since 01jan1960 00:00:00.000, assuming 86,400 s/day\n    datetime/C - tC - NOT IMPLEMENTED\n        milliseconds since 01jan1960 00:00:00.000, adjusted for leap seconds\n    date - td\n        days since 01jan1960 (01jan1960 = 0)\n    weekly date - tw\n        weeks since 1960w1\n        This assumes 52 weeks in a year, then adds 7 * remainder of the weeks.\n        The datetime value is the start of the week in terms of days in the\n        year, not ISO calendar weeks.\n    monthly date - tm\n        months since 1960m1\n    quarterly date - tq\n        quarters since 1960q1\n    half-yearly date - th\n        half-years since 1960h1 yearly\n    date - ty\n        years since 0000\n    '
    (MIN_YEAR, MAX_YEAR) = (Timestamp.min.year, Timestamp.max.year)
    MAX_DAY_DELTA = (Timestamp.max - datetime.datetime(1960, 1, 1)).days
    MIN_DAY_DELTA = (Timestamp.min - datetime.datetime(1960, 1, 1)).days
    MIN_MS_DELTA = (((MIN_DAY_DELTA * 24) * 3600) * 1000)
    MAX_MS_DELTA = (((MAX_DAY_DELTA * 24) * 3600) * 1000)

    def convert_year_month_safe(year, month) -> Series:
        '\n        Convert year and month to datetimes, using pandas vectorized versions\n        when the date range falls within the range supported by pandas.\n        Otherwise it falls back to a slower but more robust method\n        using datetime.\n        '
        if ((year.max() < MAX_YEAR) and (year.min() > MIN_YEAR)):
            return to_datetime(((100 * year) + month), format='%Y%m')
        else:
            index = getattr(year, 'index', None)
            return Series([datetime.datetime(y, m, 1) for (y, m) in zip(year, month)], index=index)

    def convert_year_days_safe(year, days) -> Series:
        '\n        Converts year (e.g. 1999) and days since the start of the year to a\n        datetime or datetime64 Series\n        '
        if ((year.max() < (MAX_YEAR - 1)) and (year.min() > MIN_YEAR)):
            return (to_datetime(year, format='%Y') + to_timedelta(days, unit='d'))
        else:
            index = getattr(year, 'index', None)
            value = [(datetime.datetime(y, 1, 1) + relativedelta(days=int(d))) for (y, d) in zip(year, days)]
            return Series(value, index=index)

    def convert_delta_safe(base, deltas, unit) -> Series:
        '\n        Convert base dates and deltas to datetimes, using pandas vectorized\n        versions if the deltas satisfy restrictions required to be expressed\n        as dates in pandas.\n        '
        index = getattr(deltas, 'index', None)
        if (unit == 'd'):
            if ((deltas.max() > MAX_DAY_DELTA) or (deltas.min() < MIN_DAY_DELTA)):
                values = [(base + relativedelta(days=int(d))) for d in deltas]
                return Series(values, index=index)
        elif (unit == 'ms'):
            if ((deltas.max() > MAX_MS_DELTA) or (deltas.min() < MIN_MS_DELTA)):
                values = [(base + relativedelta(microseconds=(int(d) * 1000))) for d in deltas]
                return Series(values, index=index)
        else:
            raise ValueError('format not understood')
        base = to_datetime(base)
        deltas = to_timedelta(deltas, unit=unit)
        return (base + deltas)
    bad_locs = np.isnan(dates)
    has_bad_values = False
    if bad_locs.any():
        has_bad_values = True
        data_col = Series(dates)
        data_col[bad_locs] = 1.0
    dates = dates.astype(np.int64)
    if fmt.startswith(('%tc', 'tc')):
        base = stata_epoch
        ms = dates
        conv_dates = convert_delta_safe(base, ms, 'ms')
    elif fmt.startswith(('%tC', 'tC')):
        warnings.warn('Encountered %tC format. Leaving in Stata Internal Format.')
        conv_dates = Series(dates, dtype=object)
        if has_bad_values:
            conv_dates[bad_locs] = NaT
        return conv_dates
    elif fmt.startswith(('%td', 'td', '%d', 'd')):
        base = stata_epoch
        days = dates
        conv_dates = convert_delta_safe(base, days, 'd')
    elif fmt.startswith(('%tw', 'tw')):
        year = (stata_epoch.year + (dates // 52))
        days = ((dates % 52) * 7)
        conv_dates = convert_year_days_safe(year, days)
    elif fmt.startswith(('%tm', 'tm')):
        year = (stata_epoch.year + (dates // 12))
        month = ((dates % 12) + 1)
        conv_dates = convert_year_month_safe(year, month)
    elif fmt.startswith(('%tq', 'tq')):
        year = (stata_epoch.year + (dates // 4))
        quarter_month = (((dates % 4) * 3) + 1)
        conv_dates = convert_year_month_safe(year, quarter_month)
    elif fmt.startswith(('%th', 'th')):
        year = (stata_epoch.year + (dates // 2))
        month = (((dates % 2) * 6) + 1)
        conv_dates = convert_year_month_safe(year, month)
    elif fmt.startswith(('%ty', 'ty')):
        year = dates
        first_month = np.ones_like(dates)
        conv_dates = convert_year_month_safe(year, first_month)
    else:
        raise ValueError(f'Date fmt {fmt} not understood')
    if has_bad_values:
        conv_dates[bad_locs] = NaT
    return conv_dates

def _datetime_to_stata_elapsed_vec(dates, fmt):
    '\n    Convert from datetime to SIF. https://www.stata.com/help.cgi?datetime\n\n    Parameters\n    ----------\n    dates : Series\n        Series or array containing datetime.datetime or datetime64[ns] to\n        convert to the Stata Internal Format given by fmt\n    fmt : str\n        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty\n    '
    index = dates.index
    NS_PER_DAY = ((((24 * 3600) * 1000) * 1000) * 1000)
    US_PER_DAY = (NS_PER_DAY / 1000)

    def parse_dates_safe(dates, delta=False, year=False, days=False):
        d = {}
        if is_datetime64_dtype(dates.dtype):
            if delta:
                time_delta = (dates - stata_epoch)
                d['delta'] = (time_delta._values.view(np.int64) // 1000)
            if (days or year):
                date_index = DatetimeIndex(dates)
                d['year'] = date_index._data.year
                d['month'] = date_index._data.month
            if days:
                days_in_ns = (dates.view(np.int64) - to_datetime(d['year'], format='%Y').view(np.int64))
                d['days'] = (days_in_ns // NS_PER_DAY)
        elif (infer_dtype(dates, skipna=False) == 'datetime'):
            if delta:
                delta = (dates._values - stata_epoch)

                def f(x: datetime.timedelta) -> float:
                    return (((US_PER_DAY * x.days) + (1000000 * x.seconds)) + x.microseconds)
                v = np.vectorize(f)
                d['delta'] = v(delta)
            if year:
                year_month = dates.apply((lambda x: ((100 * x.year) + x.month)))
                d['year'] = (year_month._values // 100)
                d['month'] = (year_month._values - (d['year'] * 100))
            if days:

                def g(x: datetime.datetime) -> int:
                    return (x - datetime.datetime(x.year, 1, 1)).days
                v = np.vectorize(g)
                d['days'] = v(dates)
        else:
            raise ValueError('Columns containing dates must contain either datetime64, datetime.datetime or null values.')
        return DataFrame(d, index=index)
    bad_loc = isna(dates)
    index = dates.index
    if bad_loc.any():
        dates = Series(dates)
        if is_datetime64_dtype(dates):
            dates[bad_loc] = to_datetime(stata_epoch)
        else:
            dates[bad_loc] = stata_epoch
    if (fmt in ['%tc', 'tc']):
        d = parse_dates_safe(dates, delta=True)
        conv_dates = (d.delta / 1000)
    elif (fmt in ['%tC', 'tC']):
        warnings.warn('Stata Internal Format tC not supported.')
        conv_dates = dates
    elif (fmt in ['%td', 'td']):
        d = parse_dates_safe(dates, delta=True)
        conv_dates = (d.delta // US_PER_DAY)
    elif (fmt in ['%tw', 'tw']):
        d = parse_dates_safe(dates, year=True, days=True)
        conv_dates = ((52 * (d.year - stata_epoch.year)) + (d.days // 7))
    elif (fmt in ['%tm', 'tm']):
        d = parse_dates_safe(dates, year=True)
        conv_dates = (((12 * (d.year - stata_epoch.year)) + d.month) - 1)
    elif (fmt in ['%tq', 'tq']):
        d = parse_dates_safe(dates, year=True)
        conv_dates = ((4 * (d.year - stata_epoch.year)) + ((d.month - 1) // 3))
    elif (fmt in ['%th', 'th']):
        d = parse_dates_safe(dates, year=True)
        conv_dates = ((2 * (d.year - stata_epoch.year)) + (d.month > 6).astype(int))
    elif (fmt in ['%ty', 'ty']):
        d = parse_dates_safe(dates, year=True)
        conv_dates = d.year
    else:
        raise ValueError(f'Format {fmt} is not a known Stata date format')
    conv_dates = Series(conv_dates, dtype=np.float64)
    missing_value = struct.unpack('<d', b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0]
    conv_dates[bad_loc] = missing_value
    return Series(conv_dates, index=index)
excessive_string_length_error = "\nFixed width strings in Stata .dta files are limited to 244 (or fewer)\ncharacters.  Column '{0}' does not satisfy this restriction. Use the\n'version=117' parameter to write the newer (Stata 13 and later) format.\n"

class PossiblePrecisionLoss(Warning):
    pass
precision_loss_doc = '\nColumn converted from {0} to {1}, and some data are outside of the lossless\nconversion range. This may result in a loss of precision in the saved data.\n'

class ValueLabelTypeMismatch(Warning):
    pass
value_label_mismatch_doc = '\nStata value labels (pandas categories) must be strings. Column {0} contains\nnon-string labels which will be converted to strings.  Please check that the\nStata data file created has not lost information due to duplicate labels.\n'

class InvalidColumnName(Warning):
    pass
invalid_name_doc = '\nNot all pandas column names were valid Stata variable names.\nThe following replacements have been made:\n\n    {0}\n\nIf this is not what you expect, please make sure you have Stata-compliant\ncolumn names in your DataFrame (strings only, max 32 characters, only\nalphanumerics and underscores, no Stata reserved words)\n'

class CategoricalConversionWarning(Warning):
    pass
categorical_conversion_warning = '\nOne or more series with value labels are not fully labeled. Reading this\ndataset with an iterator results in categorical variable with different\ncategories. This occurs since it is not possible to know all possible values\nuntil the entire dataset has been read. To avoid this warning, you can either\nread dataset without an iterator, or manually convert categorical data by\n``convert_categoricals`` to False and then accessing the variable labels\nthrough the value_labels method of the reader.\n'

def _cast_to_stata_types(data):
    '\n    Checks the dtypes of the columns of a pandas DataFrame for\n    compatibility with the data types and ranges supported by Stata, and\n    converts if necessary.\n\n    Parameters\n    ----------\n    data : DataFrame\n        The DataFrame to check and convert\n\n    Notes\n    -----\n    Numeric columns in Stata must be one of int8, int16, int32, float32 or\n    float64, with some additional value restrictions.  int8 and int16 columns\n    are checked for violations of the value restrictions and upcast if needed.\n    int64 data is not usable in Stata, and so it is downcast to int32 whenever\n    the value are in the int32 range, and sidecast to float64 when larger than\n    this range.  If the int64 values are outside of the range of those\n    perfectly representable as float64 values, a warning is raised.\n\n    bool columns are cast to int8.  uint columns are converted to int of the\n    same size if there is no loss in precision, otherwise are upcast to a\n    larger type.  uint64 is currently not supported since it is concerted to\n    object in a DataFrame.\n    '
    ws = ''
    conversion_data = ((np.bool_, np.int8, np.int8), (np.uint8, np.int8, np.int16), (np.uint16, np.int16, np.int32), (np.uint32, np.int32, np.int64))
    float32_max = struct.unpack('<f', b'\xff\xff\xff~')[0]
    float64_max = struct.unpack('<d', b'\xff\xff\xff\xff\xff\xff\xdf\x7f')[0]
    for col in data:
        dtype = data[col].dtype
        for c_data in conversion_data:
            if (dtype == c_data[0]):
                if (data[col].max() <= np.iinfo(c_data[1]).max):
                    dtype = c_data[1]
                else:
                    dtype = c_data[2]
                if (c_data[2] == np.int64):
                    if (data[col].max() >= (2 ** 53)):
                        ws = precision_loss_doc.format('uint64', 'float64')
                data[col] = data[col].astype(dtype)
        if (dtype == np.int8):
            if ((data[col].max() > 100) or (data[col].min() < (- 127))):
                data[col] = data[col].astype(np.int16)
        elif (dtype == np.int16):
            if ((data[col].max() > 32740) or (data[col].min() < (- 32767))):
                data[col] = data[col].astype(np.int32)
        elif (dtype == np.int64):
            if ((data[col].max() <= 2147483620) and (data[col].min() >= (- 2147483647))):
                data[col] = data[col].astype(np.int32)
            else:
                data[col] = data[col].astype(np.float64)
                if ((data[col].max() >= (2 ** 53)) or (data[col].min() <= (- (2 ** 53)))):
                    ws = precision_loss_doc.format('int64', 'float64')
        elif (dtype in (np.float32, np.float64)):
            value = data[col].max()
            if np.isinf(value):
                raise ValueError(f'Column {col} has a maximum value of infinity which is outside the range supported by Stata.')
            if ((dtype == np.float32) and (value > float32_max)):
                data[col] = data[col].astype(np.float64)
            elif (dtype == np.float64):
                if (value > float64_max):
                    raise ValueError(f'Column {col} has a maximum value ({value}) outside the range supported by Stata ({float64_max})')
    if ws:
        warnings.warn(ws, PossiblePrecisionLoss)
    return data

class StataValueLabel():
    '\n    Parse a categorical column and prepare formatted output\n\n    Parameters\n    ----------\n    catarray : Series\n        Categorical Series to encode\n    encoding : {"latin-1", "utf-8"}\n        Encoding to use for value labels.\n    '

    def __init__(self, catarray, encoding='latin-1'):
        if (encoding not in ('latin-1', 'utf-8')):
            raise ValueError('Only latin-1 and utf-8 are supported.')
        self.labname = catarray.name
        self._encoding = encoding
        categories = catarray.cat.categories
        self.value_labels = list(zip(np.arange(len(categories)), categories))
        self.value_labels.sort(key=(lambda x: x[0]))
        self.text_len = 0
        self.txt: List[bytes] = []
        self.n = 0
        offsets: List[int] = []
        values: List[int] = []
        for vl in self.value_labels:
            category = vl[1]
            if (not isinstance(category, str)):
                category = str(category)
                warnings.warn(value_label_mismatch_doc.format(catarray.name), ValueLabelTypeMismatch)
            category = category.encode(encoding)
            offsets.append(self.text_len)
            self.text_len += (len(category) + 1)
            values.append(vl[0])
            self.txt.append(category)
            self.n += 1
        if (self.text_len > 32000):
            raise ValueError('Stata value labels for a single variable must have a combined length less than 32,000 characters.')
        self.off = np.array(offsets, dtype=np.int32)
        self.val = np.array(values, dtype=np.int32)
        self.len = ((((4 + 4) + (4 * self.n)) + (4 * self.n)) + self.text_len)

    def generate_value_label(self, byteorder):
        '\n        Generate the binary representation of the value labels.\n\n        Parameters\n        ----------\n        byteorder : str\n            Byte order of the output\n\n        Returns\n        -------\n        value_label : bytes\n            Bytes containing the formatted value label\n        '
        encoding = self._encoding
        bio = BytesIO()
        null_byte = b'\x00'
        bio.write(struct.pack((byteorder + 'i'), self.len))
        labname = str(self.labname)[:32].encode(encoding)
        lab_len = (32 if (encoding not in ('utf-8', 'utf8')) else 128)
        labname = _pad_bytes(labname, (lab_len + 1))
        bio.write(labname)
        for i in range(3):
            bio.write(struct.pack('c', null_byte))
        bio.write(struct.pack((byteorder + 'i'), self.n))
        bio.write(struct.pack((byteorder + 'i'), self.text_len))
        for offset in self.off:
            bio.write(struct.pack((byteorder + 'i'), offset))
        for value in self.val:
            bio.write(struct.pack((byteorder + 'i'), value))
        for text in self.txt:
            bio.write((text + null_byte))
        bio.seek(0)
        return bio.read()

class StataMissingValue():
    "\n    An observation's missing value.\n\n    Parameters\n    ----------\n    value : {int, float}\n        The Stata missing value code\n\n    Notes\n    -----\n    More information: <https://www.stata.com/help.cgi?missing>\n\n    Integer missing values make the code '.', '.a', ..., '.z' to the ranges\n    101 ... 127 (for int8), 32741 ... 32767  (for int16) and 2147483621 ...\n    2147483647 (for int32).  Missing values for floating point data types are\n    more complex but the pattern is simple to discern from the following table.\n\n    np.float32 missing values (float in Stata)\n    0000007f    .\n    0008007f    .a\n    0010007f    .b\n    ...\n    00c0007f    .x\n    00c8007f    .y\n    00d0007f    .z\n\n    np.float64 missing values (double in Stata)\n    000000000000e07f    .\n    000000000001e07f    .a\n    000000000002e07f    .b\n    ...\n    000000000018e07f    .x\n    000000000019e07f    .y\n    00000000001ae07f    .z\n    "
    MISSING_VALUES = {}
    bases = (101, 32741, 2147483621)
    for b in bases:
        MISSING_VALUES[b] = '.'
        for i in range(1, 27):
            MISSING_VALUES[(i + b)] = ('.' + chr((96 + i)))
    float32_base = b'\x00\x00\x00\x7f'
    increment = struct.unpack('<i', b'\x00\x08\x00\x00')[0]
    for i in range(27):
        key = struct.unpack('<f', float32_base)[0]
        MISSING_VALUES[key] = '.'
        if (i > 0):
            MISSING_VALUES[key] += chr((96 + i))
        int_value = (struct.unpack('<i', struct.pack('<f', key))[0] + increment)
        float32_base = struct.pack('<i', int_value)
    float64_base = b'\x00\x00\x00\x00\x00\x00\xe0\x7f'
    increment = struct.unpack('q', b'\x00\x00\x00\x00\x00\x01\x00\x00')[0]
    for i in range(27):
        key = struct.unpack('<d', float64_base)[0]
        MISSING_VALUES[key] = '.'
        if (i > 0):
            MISSING_VALUES[key] += chr((96 + i))
        int_value = (struct.unpack('q', struct.pack('<d', key))[0] + increment)
        float64_base = struct.pack('q', int_value)
    BASE_MISSING_VALUES = {'int8': 101, 'int16': 32741, 'int32': 2147483621, 'float32': struct.unpack('<f', float32_base)[0], 'float64': struct.unpack('<d', float64_base)[0]}

    def __init__(self, value):
        self._value = value
        value = (int(value) if (value < 2147483648) else float(value))
        self._str = self.MISSING_VALUES[value]

    @property
    def string(self):
        "\n        The Stata representation of the missing value: '.', '.a'..'.z'\n\n        Returns\n        -------\n        str\n            The representation of the missing value.\n        "
        return self._str

    @property
    def value(self):
        '\n        The binary representation of the missing value.\n\n        Returns\n        -------\n        {int, float}\n            The binary representation of the missing value.\n        '
        return self._value

    def __str__(self):
        return self.string

    def __repr__(self):
        return f'{type(self)}({self})'

    def __eq__(self, other):
        return (isinstance(other, type(self)) and (self.string == other.string) and (self.value == other.value))

    @classmethod
    def get_base_missing_value(cls, dtype):
        if (dtype == np.int8):
            value = cls.BASE_MISSING_VALUES['int8']
        elif (dtype == np.int16):
            value = cls.BASE_MISSING_VALUES['int16']
        elif (dtype == np.int32):
            value = cls.BASE_MISSING_VALUES['int32']
        elif (dtype == np.float32):
            value = cls.BASE_MISSING_VALUES['float32']
        elif (dtype == np.float64):
            value = cls.BASE_MISSING_VALUES['float64']
        else:
            raise ValueError('Unsupported dtype')
        return value

class StataParser():

    def __init__(self):
        self.DTYPE_MAP = dict((list(zip(range(1, 245), [np.dtype(('a' + str(i))) for i in range(1, 245)])) + [(251, np.dtype(np.int8)), (252, np.dtype(np.int16)), (253, np.dtype(np.int32)), (254, np.dtype(np.float32)), (255, np.dtype(np.float64))]))
        self.DTYPE_MAP_XML = {32768: np.dtype(np.uint8), 65526: np.dtype(np.float64), 65527: np.dtype(np.float32), 65528: np.dtype(np.int32), 65529: np.dtype(np.int16), 65530: np.dtype(np.int8)}
        self.TYPE_MAP = (list(range(251)) + list('bhlfd'))
        self.TYPE_MAP_XML = {32768: 'Q', 65526: 'd', 65527: 'f', 65528: 'l', 65529: 'h', 65530: 'b'}
        float32_min = b'\xff\xff\xff\xfe'
        float32_max = b'\xff\xff\xff~'
        float64_min = b'\xff\xff\xff\xff\xff\xff\xef\xff'
        float64_max = b'\xff\xff\xff\xff\xff\xff\xdf\x7f'
        self.VALID_RANGE = {'b': ((- 127), 100), 'h': ((- 32767), 32740), 'l': ((- 2147483647), 2147483620), 'f': (np.float32(struct.unpack('<f', float32_min)[0]), np.float32(struct.unpack('<f', float32_max)[0])), 'd': (np.float64(struct.unpack('<d', float64_min)[0]), np.float64(struct.unpack('<d', float64_max)[0]))}
        self.OLD_TYPE_MAPPING = {98: 251, 105: 252, 108: 253, 102: 254, 100: 255}
        self.MISSING_VALUES = {'b': 101, 'h': 32741, 'l': 2147483621, 'f': np.float32(struct.unpack('<f', b'\x00\x00\x00\x7f')[0]), 'd': np.float64(struct.unpack('<d', b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0])}
        self.NUMPY_TYPE_MAP = {'b': 'i1', 'h': 'i2', 'l': 'i4', 'f': 'f4', 'd': 'f8', 'Q': 'u8'}
        self.RESERVED_WORDS = ('aggregate', 'array', 'boolean', 'break', 'byte', 'case', 'catch', 'class', 'colvector', 'complex', 'const', 'continue', 'default', 'delegate', 'delete', 'do', 'double', 'else', 'eltypedef', 'end', 'enum', 'explicit', 'export', 'external', 'float', 'for', 'friend', 'function', 'global', 'goto', 'if', 'inline', 'int', 'local', 'long', 'NULL', 'pragma', 'protected', 'quad', 'rowvector', 'short', 'typedef', 'typename', 'virtual', '_all', '_N', '_skip', '_b', '_pi', 'str#', 'in', '_pred', 'strL', '_coef', '_rc', 'using', '_cons', '_se', 'with', '_n')

class StataReader(StataParser, abc.Iterator):
    __doc__ = _stata_reader_doc

    def __init__(self, path_or_buf, convert_dates=True, convert_categoricals=True, index_col=None, convert_missing=False, preserve_dtypes=True, columns=None, order_categoricals=True, chunksize=None, storage_options=None):
        super().__init__()
        self.col_sizes: List[int] = []
        self._convert_dates = convert_dates
        self._convert_categoricals = convert_categoricals
        self._index_col = index_col
        self._convert_missing = convert_missing
        self._preserve_dtypes = preserve_dtypes
        self._columns = columns
        self._order_categoricals = order_categoricals
        self._encoding = ''
        self._chunksize = chunksize
        self._using_iterator = False
        if (self._chunksize is None):
            self._chunksize = 1
        elif ((not isinstance(chunksize, int)) or (chunksize <= 0)):
            raise ValueError('chunksize must be a positive integer when set.')
        self._has_string_data = False
        self._missing_values = False
        self._can_read_value_labels = False
        self._column_selector_set = False
        self._value_labels_read = False
        self._data_read = False
        self._dtype: Optional[np.dtype] = None
        self._lines_read = 0
        self._native_byteorder = _set_endianness(sys.byteorder)
        with get_handle(path_or_buf, 'rb', storage_options=storage_options, is_text=False) as handles:
            contents = handles.handle.read()
        self.path_or_buf = BytesIO(contents)
        self._read_header()
        self._setup_dtype()

    def __enter__(self):
        ' enter context manager '
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ' exit context manager '
        self.close()

    def close(self):
        ' close the handle if its open '
        self.path_or_buf.close()

    def _set_encoding(self):
        '\n        Set string encoding which depends on file version\n        '
        if (self.format_version < 118):
            self._encoding = 'latin-1'
        else:
            self._encoding = 'utf-8'

    def _read_header(self):
        first_char = self.path_or_buf.read(1)
        if (struct.unpack('c', first_char)[0] == b'<'):
            self._read_new_header()
        else:
            self._read_old_header(first_char)
        self.has_string_data = (len([x for x in self.typlist if (type(x) is int)]) > 0)
        self.col_sizes = [self._calcsize(typ) for typ in self.typlist]

    def _read_new_header(self):
        self.path_or_buf.read(27)
        self.format_version = int(self.path_or_buf.read(3))
        if (self.format_version not in [117, 118, 119]):
            raise ValueError(_version_error.format(version=self.format_version))
        self._set_encoding()
        self.path_or_buf.read(21)
        self.byteorder = (((self.path_or_buf.read(3) == b'MSF') and '>') or '<')
        self.path_or_buf.read(15)
        nvar_type = ('H' if (self.format_version <= 118) else 'I')
        nvar_size = (2 if (self.format_version <= 118) else 4)
        self.nvar = struct.unpack((self.byteorder + nvar_type), self.path_or_buf.read(nvar_size))[0]
        self.path_or_buf.read(7)
        self.nobs = self._get_nobs()
        self.path_or_buf.read(11)
        self._data_label = self._get_data_label()
        self.path_or_buf.read(19)
        self.time_stamp = self._get_time_stamp()
        self.path_or_buf.read(26)
        self.path_or_buf.read(8)
        self.path_or_buf.read(8)
        self._seek_vartypes = (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 16)
        self._seek_varnames = (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 10)
        self._seek_sortlist = (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 10)
        self._seek_formats = (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 9)
        self._seek_value_label_names = (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 19)
        self._seek_variable_labels = self._get_seek_variable_labels()
        self.path_or_buf.read(8)
        self.data_location = (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 6)
        self.seek_strls = (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 7)
        self.seek_value_labels = (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 14)
        (self.typlist, self.dtyplist) = self._get_dtypes(self._seek_vartypes)
        self.path_or_buf.seek(self._seek_varnames)
        self.varlist = self._get_varlist()
        self.path_or_buf.seek(self._seek_sortlist)
        self.srtlist = struct.unpack((self.byteorder + ('h' * (self.nvar + 1))), self.path_or_buf.read((2 * (self.nvar + 1))))[:(- 1)]
        self.path_or_buf.seek(self._seek_formats)
        self.fmtlist = self._get_fmtlist()
        self.path_or_buf.seek(self._seek_value_label_names)
        self.lbllist = self._get_lbllist()
        self.path_or_buf.seek(self._seek_variable_labels)
        self._variable_labels = self._get_variable_labels()

    def _get_dtypes(self, seek_vartypes):
        self.path_or_buf.seek(seek_vartypes)
        raw_typlist = [struct.unpack((self.byteorder + 'H'), self.path_or_buf.read(2))[0] for _ in range(self.nvar)]

        def f(typ: int) -> Union[(int, str)]:
            if (typ <= 2045):
                return typ
            try:
                return self.TYPE_MAP_XML[typ]
            except KeyError as err:
                raise ValueError(f'cannot convert stata types [{typ}]') from err
        typlist = [f(x) for x in raw_typlist]

        def g(typ: int) -> Union[(str, np.dtype)]:
            if (typ <= 2045):
                return str(typ)
            try:
                return self.DTYPE_MAP_XML[typ]
            except KeyError as err:
                raise ValueError(f'cannot convert stata dtype [{typ}]') from err
        dtyplist = [g(x) for x in raw_typlist]
        return (typlist, dtyplist)

    def _get_varlist(self):
        b = (33 if (self.format_version < 118) else 129)
        return [self._decode(self.path_or_buf.read(b)) for _ in range(self.nvar)]

    def _get_fmtlist(self):
        if (self.format_version >= 118):
            b = 57
        elif (self.format_version > 113):
            b = 49
        elif (self.format_version > 104):
            b = 12
        else:
            b = 7
        return [self._decode(self.path_or_buf.read(b)) for _ in range(self.nvar)]

    def _get_lbllist(self):
        if (self.format_version >= 118):
            b = 129
        elif (self.format_version > 108):
            b = 33
        else:
            b = 9
        return [self._decode(self.path_or_buf.read(b)) for _ in range(self.nvar)]

    def _get_variable_labels(self):
        if (self.format_version >= 118):
            vlblist = [self._decode(self.path_or_buf.read(321)) for _ in range(self.nvar)]
        elif (self.format_version > 105):
            vlblist = [self._decode(self.path_or_buf.read(81)) for _ in range(self.nvar)]
        else:
            vlblist = [self._decode(self.path_or_buf.read(32)) for _ in range(self.nvar)]
        return vlblist

    def _get_nobs(self):
        if (self.format_version >= 118):
            return struct.unpack((self.byteorder + 'Q'), self.path_or_buf.read(8))[0]
        else:
            return struct.unpack((self.byteorder + 'I'), self.path_or_buf.read(4))[0]

    def _get_data_label(self):
        if (self.format_version >= 118):
            strlen = struct.unpack((self.byteorder + 'H'), self.path_or_buf.read(2))[0]
            return self._decode(self.path_or_buf.read(strlen))
        elif (self.format_version == 117):
            strlen = struct.unpack('b', self.path_or_buf.read(1))[0]
            return self._decode(self.path_or_buf.read(strlen))
        elif (self.format_version > 105):
            return self._decode(self.path_or_buf.read(81))
        else:
            return self._decode(self.path_or_buf.read(32))

    def _get_time_stamp(self):
        if (self.format_version >= 118):
            strlen = struct.unpack('b', self.path_or_buf.read(1))[0]
            return self.path_or_buf.read(strlen).decode('utf-8')
        elif (self.format_version == 117):
            strlen = struct.unpack('b', self.path_or_buf.read(1))[0]
            return self._decode(self.path_or_buf.read(strlen))
        elif (self.format_version > 104):
            return self._decode(self.path_or_buf.read(18))
        else:
            raise ValueError()

    def _get_seek_variable_labels(self):
        if (self.format_version == 117):
            self.path_or_buf.read(8)
            return (((self._seek_value_label_names + (33 * self.nvar)) + 20) + 17)
        elif (self.format_version >= 118):
            return (struct.unpack((self.byteorder + 'q'), self.path_or_buf.read(8))[0] + 17)
        else:
            raise ValueError()

    def _read_old_header(self, first_char):
        self.format_version = struct.unpack('b', first_char)[0]
        if (self.format_version not in [104, 105, 108, 111, 113, 114, 115]):
            raise ValueError(_version_error.format(version=self.format_version))
        self._set_encoding()
        self.byteorder = (((struct.unpack('b', self.path_or_buf.read(1))[0] == 1) and '>') or '<')
        self.filetype = struct.unpack('b', self.path_or_buf.read(1))[0]
        self.path_or_buf.read(1)
        self.nvar = struct.unpack((self.byteorder + 'H'), self.path_or_buf.read(2))[0]
        self.nobs = self._get_nobs()
        self._data_label = self._get_data_label()
        self.time_stamp = self._get_time_stamp()
        if (self.format_version > 108):
            typlist = [ord(self.path_or_buf.read(1)) for _ in range(self.nvar)]
        else:
            buf = self.path_or_buf.read(self.nvar)
            typlistb = np.frombuffer(buf, dtype=np.uint8)
            typlist = []
            for tp in typlistb:
                if (tp in self.OLD_TYPE_MAPPING):
                    typlist.append(self.OLD_TYPE_MAPPING[tp])
                else:
                    typlist.append((tp - 127))
        try:
            self.typlist = [self.TYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_types = ','.join((str(x) for x in typlist))
            raise ValueError(f'cannot convert stata types [{invalid_types}]') from err
        try:
            self.dtyplist = [self.DTYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_dtypes = ','.join((str(x) for x in typlist))
            raise ValueError(f'cannot convert stata dtypes [{invalid_dtypes}]') from err
        if (self.format_version > 108):
            self.varlist = [self._decode(self.path_or_buf.read(33)) for _ in range(self.nvar)]
        else:
            self.varlist = [self._decode(self.path_or_buf.read(9)) for _ in range(self.nvar)]
        self.srtlist = struct.unpack((self.byteorder + ('h' * (self.nvar + 1))), self.path_or_buf.read((2 * (self.nvar + 1))))[:(- 1)]
        self.fmtlist = self._get_fmtlist()
        self.lbllist = self._get_lbllist()
        self._variable_labels = self._get_variable_labels()
        if (self.format_version > 104):
            while True:
                data_type = struct.unpack((self.byteorder + 'b'), self.path_or_buf.read(1))[0]
                if (self.format_version > 108):
                    data_len = struct.unpack((self.byteorder + 'i'), self.path_or_buf.read(4))[0]
                else:
                    data_len = struct.unpack((self.byteorder + 'h'), self.path_or_buf.read(2))[0]
                if (data_type == 0):
                    break
                self.path_or_buf.read(data_len)
        self.data_location = self.path_or_buf.tell()

    def _setup_dtype(self):
        'Map between numpy and state dtypes'
        if (self._dtype is not None):
            return self._dtype
        dtypes = []
        for (i, typ) in enumerate(self.typlist):
            if (typ in self.NUMPY_TYPE_MAP):
                typ = cast(str, typ)
                dtypes.append((('s' + str(i)), (self.byteorder + self.NUMPY_TYPE_MAP[typ])))
            else:
                dtypes.append((('s' + str(i)), ('S' + str(typ))))
        self._dtype = np.dtype(dtypes)
        return self._dtype

    def _calcsize(self, fmt):
        if isinstance(fmt, int):
            return fmt
        return struct.calcsize((self.byteorder + fmt))

    def _decode(self, s):
        s = s.partition(b'\x00')[0]
        try:
            return s.decode(self._encoding)
        except UnicodeDecodeError:
            encoding = self._encoding
            msg = f'''
One or more strings in the dta file could not be decoded using {encoding}, and
so the fallback encoding of latin-1 is being used.  This can happen when a file
has been incorrectly encoded by Stata or some other software. You should verify
the string values returned are correct.'''
            warnings.warn(msg, UnicodeWarning)
            return s.decode('latin-1')

    def _read_value_labels(self):
        if self._value_labels_read:
            return
        if (self.format_version <= 108):
            self._value_labels_read = True
            self.value_label_dict: Dict[(str, Dict[(Union[(float, int)], str)])] = {}
            return
        if (self.format_version >= 117):
            self.path_or_buf.seek(self.seek_value_labels)
        else:
            assert (self._dtype is not None)
            offset = (self.nobs * self._dtype.itemsize)
            self.path_or_buf.seek((self.data_location + offset))
        self._value_labels_read = True
        self.value_label_dict = {}
        while True:
            if (self.format_version >= 117):
                if (self.path_or_buf.read(5) == b'</val'):
                    break
            slength = self.path_or_buf.read(4)
            if (not slength):
                break
            if (self.format_version <= 117):
                labname = self._decode(self.path_or_buf.read(33))
            else:
                labname = self._decode(self.path_or_buf.read(129))
            self.path_or_buf.read(3)
            n = struct.unpack((self.byteorder + 'I'), self.path_or_buf.read(4))[0]
            txtlen = struct.unpack((self.byteorder + 'I'), self.path_or_buf.read(4))[0]
            off = np.frombuffer(self.path_or_buf.read((4 * n)), dtype=(self.byteorder + 'i4'), count=n)
            val = np.frombuffer(self.path_or_buf.read((4 * n)), dtype=(self.byteorder + 'i4'), count=n)
            ii = np.argsort(off)
            off = off[ii]
            val = val[ii]
            txt = self.path_or_buf.read(txtlen)
            self.value_label_dict[labname] = {}
            for i in range(n):
                end = (off[(i + 1)] if (i < (n - 1)) else txtlen)
                self.value_label_dict[labname][val[i]] = self._decode(txt[off[i]:end])
            if (self.format_version >= 117):
                self.path_or_buf.read(6)
        self._value_labels_read = True

    def _read_strls(self):
        self.path_or_buf.seek(self.seek_strls)
        self.GSO = {'0': ''}
        while True:
            if (self.path_or_buf.read(3) != b'GSO'):
                break
            if (self.format_version == 117):
                v_o = struct.unpack((self.byteorder + 'Q'), self.path_or_buf.read(8))[0]
            else:
                buf = self.path_or_buf.read(12)
                v_size = (2 if (self.format_version == 118) else 3)
                if (self.byteorder == '<'):
                    buf = (buf[0:v_size] + buf[4:(12 - v_size)])
                else:
                    buf = (buf[0:v_size] + buf[(4 + v_size):])
                v_o = struct.unpack('Q', buf)[0]
            typ = struct.unpack('B', self.path_or_buf.read(1))[0]
            length = struct.unpack((self.byteorder + 'I'), self.path_or_buf.read(4))[0]
            va = self.path_or_buf.read(length)
            if (typ == 130):
                decoded_va = va[0:(- 1)].decode(self._encoding)
            else:
                decoded_va = str(va)
            self.GSO[str(v_o)] = decoded_va

    def __next__(self):
        self._using_iterator = True
        return self.read(nrows=self._chunksize)

    def get_chunk(self, size=None):
        '\n        Reads lines from Stata file and returns as dataframe\n\n        Parameters\n        ----------\n        size : int, defaults to None\n            Number of lines to read.  If None, reads whole file.\n\n        Returns\n        -------\n        DataFrame\n        '
        if (size is None):
            size = self._chunksize
        return self.read(nrows=size)

    @Appender(_read_method_doc)
    def read(self, nrows=None, convert_dates=None, convert_categoricals=None, index_col=None, convert_missing=None, preserve_dtypes=None, columns=None, order_categoricals=None):
        if ((self.nobs == 0) and (nrows is None)):
            self._can_read_value_labels = True
            self._data_read = True
            self.close()
            return DataFrame(columns=self.varlist)
        if (convert_dates is None):
            convert_dates = self._convert_dates
        if (convert_categoricals is None):
            convert_categoricals = self._convert_categoricals
        if (convert_missing is None):
            convert_missing = self._convert_missing
        if (preserve_dtypes is None):
            preserve_dtypes = self._preserve_dtypes
        if (columns is None):
            columns = self._columns
        if (order_categoricals is None):
            order_categoricals = self._order_categoricals
        if (index_col is None):
            index_col = self._index_col
        if (nrows is None):
            nrows = self.nobs
        if ((self.format_version >= 117) and (not self._value_labels_read)):
            self._can_read_value_labels = True
            self._read_strls()
        assert (self._dtype is not None)
        dtype = self._dtype
        max_read_len = ((self.nobs - self._lines_read) * dtype.itemsize)
        read_len = (nrows * dtype.itemsize)
        read_len = min(read_len, max_read_len)
        if (read_len <= 0):
            if convert_categoricals:
                self._read_value_labels()
            self.close()
            raise StopIteration
        offset = (self._lines_read * dtype.itemsize)
        self.path_or_buf.seek((self.data_location + offset))
        read_lines = min(nrows, (self.nobs - self._lines_read))
        data = np.frombuffer(self.path_or_buf.read(read_len), dtype=dtype, count=read_lines)
        self._lines_read += read_lines
        if (self._lines_read == self.nobs):
            self._can_read_value_labels = True
            self._data_read = True
        if (self.byteorder != self._native_byteorder):
            data = data.byteswap().newbyteorder()
        if convert_categoricals:
            self._read_value_labels()
        if (len(data) == 0):
            data = DataFrame(columns=self.varlist)
        else:
            data = DataFrame.from_records(data)
            data.columns = self.varlist
        if (index_col is None):
            ix = np.arange((self._lines_read - read_lines), self._lines_read)
            data = data.set_index(ix)
        if (columns is not None):
            try:
                data = self._do_select_columns(data, columns)
            except ValueError:
                self.close()
                raise
        for (col, typ) in zip(data, self.typlist):
            if (type(typ) is int):
                data[col] = data[col].apply(self._decode, convert_dtype=True)
        data = self._insert_strls(data)
        cols_ = np.where([(dtyp is not None) for dtyp in self.dtyplist])[0]
        ix = data.index
        requires_type_conversion = False
        data_formatted = []
        for i in cols_:
            if (self.dtyplist[i] is not None):
                col = data.columns[i]
                dtype = data[col].dtype
                if ((dtype != np.dtype(object)) and (dtype != self.dtyplist[i])):
                    requires_type_conversion = True
                    data_formatted.append((col, Series(data[col], ix, self.dtyplist[i])))
                else:
                    data_formatted.append((col, data[col]))
        if requires_type_conversion:
            data = DataFrame.from_dict(dict(data_formatted))
        del data_formatted
        data = self._do_convert_missing(data, convert_missing)
        if convert_dates:

            def any_startswith(x: str) -> bool:
                return any((x.startswith(fmt) for fmt in _date_formats))
            cols = np.where([any_startswith(x) for x in self.fmtlist])[0]
            for i in cols:
                col = data.columns[i]
                try:
                    data[col] = _stata_elapsed_date_to_datetime_vec(data[col], self.fmtlist[i])
                except ValueError:
                    self.close()
                    raise
        if (convert_categoricals and (self.format_version > 108)):
            data = self._do_convert_categoricals(data, self.value_label_dict, self.lbllist, order_categoricals)
        if (not preserve_dtypes):
            retyped_data = []
            convert = False
            for col in data:
                dtype = data[col].dtype
                if (dtype in (np.dtype(np.float16), np.dtype(np.float32))):
                    dtype = np.dtype(np.float64)
                    convert = True
                elif (dtype in (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32))):
                    dtype = np.dtype(np.int64)
                    convert = True
                retyped_data.append((col, data[col].astype(dtype)))
            if convert:
                data = DataFrame.from_dict(dict(retyped_data))
        if (index_col is not None):
            data = data.set_index(data.pop(index_col))
        return data

    def _do_convert_missing(self, data, convert_missing):
        replacements = {}
        for (i, colname) in enumerate(data):
            fmt = self.typlist[i]
            if (fmt not in self.VALID_RANGE):
                continue
            fmt = cast(str, fmt)
            (nmin, nmax) = self.VALID_RANGE[fmt]
            series = data[colname]
            missing = np.logical_or((series < nmin), (series > nmax))
            if (not missing.any()):
                continue
            if convert_missing:
                missing_loc = np.nonzero(np.asarray(missing))[0]
                (umissing, umissing_loc) = np.unique(series[missing], return_inverse=True)
                replacement = Series(series, dtype=object)
                for (j, um) in enumerate(umissing):
                    missing_value = StataMissingValue(um)
                    loc = missing_loc[(umissing_loc == j)]
                    replacement.iloc[loc] = missing_value
            else:
                dtype = series.dtype
                if (dtype not in (np.float32, np.float64)):
                    dtype = np.float64
                replacement = Series(series, dtype=dtype)
                replacement[missing] = np.nan
            replacements[colname] = replacement
        if replacements:
            columns = data.columns
            replacement_df = DataFrame(replacements)
            replaced = concat([data.drop(replacement_df.columns, 1), replacement_df], 1)
            data = replaced[columns]
        return data

    def _insert_strls(self, data):
        if ((not hasattr(self, 'GSO')) or (len(self.GSO) == 0)):
            return data
        for (i, typ) in enumerate(self.typlist):
            if (typ != 'Q'):
                continue
            data.iloc[:, i] = [self.GSO[str(k)] for k in data.iloc[:, i]]
        return data

    def _do_select_columns(self, data, columns):
        if (not self._column_selector_set):
            column_set = set(columns)
            if (len(column_set) != len(columns)):
                raise ValueError('columns contains duplicate entries')
            unmatched = column_set.difference(data.columns)
            if unmatched:
                joined = ', '.join(list(unmatched))
                raise ValueError(f'The following columns were not found in the Stata data set: {joined}')
            dtyplist = []
            typlist = []
            fmtlist = []
            lbllist = []
            for col in columns:
                i = data.columns.get_loc(col)
                dtyplist.append(self.dtyplist[i])
                typlist.append(self.typlist[i])
                fmtlist.append(self.fmtlist[i])
                lbllist.append(self.lbllist[i])
            self.dtyplist = dtyplist
            self.typlist = typlist
            self.fmtlist = fmtlist
            self.lbllist = lbllist
            self._column_selector_set = True
        return data[columns]

    def _do_convert_categoricals(self, data, value_label_dict, lbllist, order_categoricals):
        '\n        Converts categorical columns to Categorical type.\n        '
        value_labels = list(value_label_dict.keys())
        cat_converted_data = []
        for (col, label) in zip(data, lbllist):
            if (label in value_labels):
                vl = value_label_dict[label]
                keys = np.array(list(vl.keys()))
                column = data[col]
                key_matches = column.isin(keys)
                if (self._using_iterator and key_matches.all()):
                    initial_categories: Optional[np.ndarray] = keys
                else:
                    if self._using_iterator:
                        warnings.warn(categorical_conversion_warning, CategoricalConversionWarning)
                    initial_categories = None
                cat_data = Categorical(column, categories=initial_categories, ordered=order_categoricals)
                if (initial_categories is None):
                    categories = []
                    for category in cat_data.categories:
                        if (category in vl):
                            categories.append(vl[category])
                        else:
                            categories.append(category)
                else:
                    categories = list(vl.values())
                try:
                    cat_data.categories = categories
                except ValueError as err:
                    vc = Series(categories).value_counts()
                    repeated_cats = list(vc.index[(vc > 1)])
                    repeats = ((('-' * 80) + '\n') + '\n'.join(repeated_cats))
                    msg = f'''
Value labels for column {col} are not unique. These cannot be converted to
pandas categoricals.

Either read the file with `convert_categoricals` set to False or use the
low level interface in `StataReader` to separately read the values and the
value_labels.

The repeated labels are:
{repeats}
'''
                    raise ValueError(msg) from err
                cat_series = Series(cat_data, index=data.index)
                cat_converted_data.append((col, cat_series))
            else:
                cat_converted_data.append((col, data[col]))
        data = DataFrame.from_dict(dict(cat_converted_data))
        return data

    @property
    def data_label(self):
        '\n        Return data label of Stata file.\n        '
        return self._data_label

    def variable_labels(self):
        '\n        Return variable labels as a dict, associating each variable name\n        with corresponding label.\n\n        Returns\n        -------\n        dict\n        '
        return dict(zip(self.varlist, self._variable_labels))

    def value_labels(self):
        '\n        Return a dict, associating each variable name a dict, associating\n        each value its corresponding label.\n\n        Returns\n        -------\n        dict\n        '
        if (not self._value_labels_read):
            self._read_value_labels()
        return self.value_label_dict

@Appender(_read_stata_doc)
def read_stata(filepath_or_buffer, convert_dates=True, convert_categoricals=True, index_col=None, convert_missing=False, preserve_dtypes=True, columns=None, order_categoricals=True, chunksize=None, iterator=False, storage_options=None):
    reader = StataReader(filepath_or_buffer, convert_dates=convert_dates, convert_categoricals=convert_categoricals, index_col=index_col, convert_missing=convert_missing, preserve_dtypes=preserve_dtypes, columns=columns, order_categoricals=order_categoricals, chunksize=chunksize, storage_options=storage_options)
    if (iterator or chunksize):
        return reader
    with reader:
        return reader.read()

def _set_endianness(endianness):
    if (endianness.lower() in ['<', 'little']):
        return '<'
    elif (endianness.lower() in ['>', 'big']):
        return '>'
    else:
        raise ValueError(f'Endianness {endianness} not understood')

def _pad_bytes(name, length):
    "\n    Take a char string and pads it with null bytes until it's length chars.\n    "
    if isinstance(name, bytes):
        return (name + (b'\x00' * (length - len(name))))
    return (name + ('\x00' * (length - len(name))))

def _convert_datetime_to_stata_type(fmt):
    '\n    Convert from one of the stata date formats to a type in TYPE_MAP.\n    '
    if (fmt in ['tc', '%tc', 'td', '%td', 'tw', '%tw', 'tm', '%tm', 'tq', '%tq', 'th', '%th', 'ty', '%ty']):
        return np.dtype(np.float64)
    else:
        raise NotImplementedError(f'Format {fmt} not implemented')

def _maybe_convert_to_int_keys(convert_dates, varlist):
    new_dict = {}
    for key in convert_dates:
        if (not convert_dates[key].startswith('%')):
            convert_dates[key] = ('%' + convert_dates[key])
        if (key in varlist):
            new_dict.update({varlist.index(key): convert_dates[key]})
        else:
            if (not isinstance(key, int)):
                raise ValueError('convert_dates key must be a column or an integer')
            new_dict.update({key: convert_dates[key]})
    return new_dict

def _dtype_to_stata_type(dtype, column):
    '\n    Convert dtype types to stata types. Returns the byte of the given ordinal.\n    See TYPE_MAP and comments for an explanation. This is also explained in\n    the dta spec.\n    1 - 244 are strings of this length\n                         Pandas    Stata\n    251 - for int8      byte\n    252 - for int16     int\n    253 - for int32     long\n    254 - for float32   float\n    255 - for double    double\n\n    If there are dates to convert, then dtype will already have the correct\n    type inserted.\n    '
    if (dtype.type == np.object_):
        itemsize = max_len_string_array(ensure_object(column._values))
        return max(itemsize, 1)
    elif (dtype == np.float64):
        return 255
    elif (dtype == np.float32):
        return 254
    elif (dtype == np.int32):
        return 253
    elif (dtype == np.int16):
        return 252
    elif (dtype == np.int8):
        return 251
    else:
        raise NotImplementedError(f'Data type {dtype} not supported.')

def _dtype_to_default_stata_fmt(dtype, column, dta_version=114, force_strl=False):
    '\n    Map numpy dtype to stata\'s default format for this type. Not terribly\n    important since users can change this in Stata. Semantics are\n\n    object  -> "%DDs" where DD is the length of the string.  If not a string,\n                raise ValueError\n    float64 -> "%10.0g"\n    float32 -> "%9.0g"\n    int64   -> "%9.0g"\n    int32   -> "%12.0g"\n    int16   -> "%8.0g"\n    int8    -> "%8.0g"\n    strl    -> "%9s"\n    '
    if (dta_version < 117):
        max_str_len = 244
    else:
        max_str_len = 2045
        if force_strl:
            return '%9s'
    if (dtype.type == np.object_):
        itemsize = max_len_string_array(ensure_object(column._values))
        if (itemsize > max_str_len):
            if (dta_version >= 117):
                return '%9s'
            else:
                raise ValueError(excessive_string_length_error.format(column.name))
        return (('%' + str(max(itemsize, 1))) + 's')
    elif (dtype == np.float64):
        return '%10.0g'
    elif (dtype == np.float32):
        return '%9.0g'
    elif (dtype == np.int32):
        return '%12.0g'
    elif ((dtype == np.int8) or (dtype == np.int16)):
        return '%8.0g'
    else:
        raise NotImplementedError(f'Data type {dtype} not supported.')

@doc(storage_options=generic._shared_docs['storage_options'])
class StataWriter(StataParser):
    '\n    A class for writing Stata binary dta files\n\n    Parameters\n    ----------\n    fname : path (string), buffer or path object\n        string, path object (pathlib.Path or py._path.local.LocalPath) or\n        object implementing a binary write() functions. If using a buffer\n        then the buffer will not be automatically closed after the file\n        is written.\n    data : DataFrame\n        Input to save\n    convert_dates : dict\n        Dictionary mapping columns containing datetime types to stata internal\n        format to use when writing the dates. Options are \'tc\', \'td\', \'tm\',\n        \'tw\', \'th\', \'tq\', \'ty\'. Column can be either an integer or a name.\n        Datetime columns that do not have a conversion type specified will be\n        converted to \'tc\'. Raises NotImplementedError if a datetime column has\n        timezone information\n    write_index : bool\n        Write the index to Stata dataset.\n    byteorder : str\n        Can be ">", "<", "little", or "big". default is `sys.byteorder`\n    time_stamp : datetime\n        A datetime to use as file creation date.  Default is the current time\n    data_label : str\n        A label for the data set.  Must be 80 characters or smaller.\n    variable_labels : dict\n        Dictionary containing columns as keys and variable labels as values.\n        Each label must be 80 characters or smaller.\n    compression : str or dict, default \'infer\'\n        For on-the-fly compression of the output dta. If string, specifies\n        compression mode. If dict, value at key \'method\' specifies compression\n        mode. Compression mode must be one of {{\'infer\', \'gzip\', \'bz2\', \'zip\',\n        \'xz\', None}}. If compression mode is \'infer\' and `fname` is path-like,\n        then detect compression from the following extensions: \'.gz\', \'.bz2\',\n        \'.zip\', or \'.xz\' (otherwise no compression). If dict and compression\n        mode is one of {{\'zip\', \'gzip\', \'bz2\'}}, or inferred as one of the above,\n        other entries passed as additional compression options.\n\n        .. versionadded:: 1.1.0\n\n    {storage_options}\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    writer : StataWriter instance\n        The StataWriter instance has a write_file method, which will\n        write the file to the given `fname`.\n\n    Raises\n    ------\n    NotImplementedError\n        * If datetimes contain timezone information\n    ValueError\n        * Columns listed in convert_dates are neither datetime64[ns]\n          or datetime.datetime\n        * Column dtype is not representable in Stata\n        * Column listed in convert_dates is not in DataFrame\n        * Categorical label contains more than 32,000 characters\n\n    Examples\n    --------\n    >>> data = pd.DataFrame([[1.0, 1]], columns=[\'a\', \'b\'])\n    >>> writer = StataWriter(\'./data_file.dta\', data)\n    >>> writer.write_file()\n\n    Directly write a zip file\n    >>> compression = {{"method": "zip", "archive_name": "data_file.dta"}}\n    >>> writer = StataWriter(\'./data_file.zip\', data, compression=compression)\n    >>> writer.write_file()\n\n    Save a DataFrame with dates\n    >>> from datetime import datetime\n    >>> data = pd.DataFrame([[datetime(2000,1,1)]], columns=[\'date\'])\n    >>> writer = StataWriter(\'./date_data_file.dta\', data, {{\'date\' : \'tw\'}})\n    >>> writer.write_file()\n    '
    _max_string_length = 244
    _encoding = 'latin-1'

    def __init__(self, fname, data, convert_dates=None, write_index=True, byteorder=None, time_stamp=None, data_label=None, variable_labels=None, compression='infer', storage_options=None):
        super().__init__()
        self._convert_dates = ({} if (convert_dates is None) else convert_dates)
        self._write_index = write_index
        self._time_stamp = time_stamp
        self._data_label = data_label
        self._variable_labels = variable_labels
        self._compression = compression
        self._output_file: Optional[Buffer] = None
        self._prepare_pandas(data)
        self.storage_options = storage_options
        if (byteorder is None):
            byteorder = sys.byteorder
        self._byteorder = _set_endianness(byteorder)
        self._fname = fname
        self.type_converters = {253: np.int32, 252: np.int16, 251: np.int8}
        self._converted_names: Dict[(Label, str)] = {}

    def _write(self, to_write):
        '\n        Helper to call encode before writing to file for Python 3 compat.\n        '
        self.handles.handle.write(to_write.encode(self._encoding))

    def _write_bytes(self, value):
        '\n        Helper to assert file is open before writing.\n        '
        self.handles.handle.write(value)

    def _prepare_categoricals(self, data):
        '\n        Check for categorical columns, retain categorical information for\n        Stata file and convert categorical data to int\n        '
        is_cat = [is_categorical_dtype(data[col].dtype) for col in data]
        self._is_col_cat = is_cat
        self._value_labels: List[StataValueLabel] = []
        if (not any(is_cat)):
            return data
        get_base_missing_value = StataMissingValue.get_base_missing_value
        data_formatted = []
        for (col, col_is_cat) in zip(data, is_cat):
            if col_is_cat:
                svl = StataValueLabel(data[col], encoding=self._encoding)
                self._value_labels.append(svl)
                dtype = data[col].cat.codes.dtype
                if (dtype == np.int64):
                    raise ValueError('It is not possible to export int64-based categorical data to Stata.')
                values = data[col].cat.codes._values.copy()
                if (values.max() >= get_base_missing_value(dtype)):
                    if (dtype == np.int8):
                        dtype = np.int16
                    elif (dtype == np.int16):
                        dtype = np.int32
                    else:
                        dtype = np.float64
                    values = np.array(values, dtype=dtype)
                values[(values == (- 1))] = get_base_missing_value(dtype)
                data_formatted.append((col, values))
            else:
                data_formatted.append((col, data[col]))
        return DataFrame.from_dict(dict(data_formatted))

    def _replace_nans(self, data):
        '\n        Checks floating point data columns for nans, and replaces these with\n        the generic Stata for missing value (.)\n        '
        for c in data:
            dtype = data[c].dtype
            if (dtype in (np.float32, np.float64)):
                if (dtype == np.float32):
                    replacement = self.MISSING_VALUES['f']
                else:
                    replacement = self.MISSING_VALUES['d']
                data[c] = data[c].fillna(replacement)
        return data

    def _update_strl_names(self):
        'No-op, forward compatibility'
        pass

    def _validate_variable_name(self, name):
        '\n        Validate variable names for Stata export.\n\n        Parameters\n        ----------\n        name : str\n            Variable name\n\n        Returns\n        -------\n        str\n            The validated name with invalid characters replaced with\n            underscores.\n\n        Notes\n        -----\n        Stata 114 and 117 support ascii characters in a-z, A-Z, 0-9\n        and _.\n        '
        for c in name:
            if (((c < 'A') or (c > 'Z')) and ((c < 'a') or (c > 'z')) and ((c < '0') or (c > '9')) and (c != '_')):
                name = name.replace(c, '_')
        return name

    def _check_column_names(self, data):
        '\n        Checks column names to ensure that they are valid Stata column names.\n        This includes checks for:\n            * Non-string names\n            * Stata keywords\n            * Variables that start with numbers\n            * Variables with names that are too long\n\n        When an illegal variable name is detected, it is converted, and if\n        dates are exported, the variable name is propagated to the date\n        conversion dictionary\n        '
        converted_names: Dict[(Label, str)] = {}
        columns: List[Label] = list(data.columns)
        original_columns = columns[:]
        duplicate_var_id = 0
        for (j, name) in enumerate(columns):
            orig_name = name
            if (not isinstance(name, str)):
                name = str(name)
            name = self._validate_variable_name(name)
            if (name in self.RESERVED_WORDS):
                name = ('_' + name)
            if ('0' <= name[0] <= '9'):
                name = ('_' + name)
            name = name[:min(len(name), 32)]
            if (not (name == orig_name)):
                while (columns.count(name) > 0):
                    name = (('_' + str(duplicate_var_id)) + name)
                    name = name[:min(len(name), 32)]
                    duplicate_var_id += 1
                converted_names[orig_name] = name
            columns[j] = name
        data.columns = Index(columns)
        if self._convert_dates:
            for (c, o) in zip(columns, original_columns):
                if (c != o):
                    self._convert_dates[c] = self._convert_dates[o]
                    del self._convert_dates[o]
        if converted_names:
            conversion_warning = []
            for (orig_name, name) in converted_names.items():
                msg = f'{orig_name}   ->   {name}'
                conversion_warning.append(msg)
            ws = invalid_name_doc.format('\n    '.join(conversion_warning))
            warnings.warn(ws, InvalidColumnName)
        self._converted_names = converted_names
        self._update_strl_names()
        return data

    def _set_formats_and_types(self, dtypes):
        self.fmtlist: List[str] = []
        self.typlist: List[int] = []
        for (col, dtype) in dtypes.items():
            self.fmtlist.append(_dtype_to_default_stata_fmt(dtype, self.data[col]))
            self.typlist.append(_dtype_to_stata_type(dtype, self.data[col]))

    def _prepare_pandas(self, data):
        data = data.copy()
        if self._write_index:
            temp = data.reset_index()
            if isinstance(temp, DataFrame):
                data = temp
        data = self._check_column_names(data)
        data = _cast_to_stata_types(data)
        data = self._replace_nans(data)
        data = self._prepare_categoricals(data)
        (self.nobs, self.nvar) = data.shape
        self.data = data
        self.varlist = data.columns.tolist()
        dtypes = data.dtypes
        for col in data:
            if (col in self._convert_dates):
                continue
            if is_datetime64_dtype(data[col]):
                self._convert_dates[col] = 'tc'
        self._convert_dates = _maybe_convert_to_int_keys(self._convert_dates, self.varlist)
        for key in self._convert_dates:
            new_type = _convert_datetime_to_stata_type(self._convert_dates[key])
            dtypes[key] = np.dtype(new_type)
        self._encode_strings()
        self._set_formats_and_types(dtypes)
        if (self._convert_dates is not None):
            for key in self._convert_dates:
                if isinstance(key, int):
                    self.fmtlist[key] = self._convert_dates[key]

    def _encode_strings(self):
        '\n        Encode strings in dta-specific encoding\n\n        Do not encode columns marked for date conversion or for strL\n        conversion. The strL converter independently handles conversion and\n        also accepts empty string arrays.\n        '
        convert_dates = self._convert_dates
        convert_strl = getattr(self, '_convert_strl', [])
        for (i, col) in enumerate(self.data):
            if ((i in convert_dates) or (col in convert_strl)):
                continue
            column = self.data[col]
            dtype = column.dtype
            if (dtype.type == np.object_):
                inferred_dtype = infer_dtype(column, skipna=True)
                if (not ((inferred_dtype == 'string') or (len(column) == 0))):
                    col = column.name
                    raise ValueError(f'''Column `{col}` cannot be exported.

Only string-like object arrays
containing all strings or a mix of strings and None can be exported.
Object arrays containing only null values are prohibited. Other object
types cannot be exported and must first be converted to one of the
supported types.''')
                encoded = self.data[col].str.encode(self._encoding)
                if (max_len_string_array(ensure_object(encoded._values)) <= self._max_string_length):
                    self.data[col] = encoded

    def write_file(self):
        with get_handle(self._fname, 'wb', compression=self._compression, is_text=False, storage_options=self.storage_options) as self.handles:
            if (self.handles.compression['method'] is not None):
                self._output_file = self.handles.handle
                self.handles.handle = BytesIO()
            try:
                self._write_header(data_label=self._data_label, time_stamp=self._time_stamp)
                self._write_map()
                self._write_variable_types()
                self._write_varnames()
                self._write_sortlist()
                self._write_formats()
                self._write_value_label_names()
                self._write_variable_labels()
                self._write_expansion_fields()
                self._write_characteristics()
                records = self._prepare_data()
                self._write_data(records)
                self._write_strls()
                self._write_value_labels()
                self._write_file_close_tag()
                self._write_map()
            except Exception as exc:
                self._close()
                if isinstance(self._fname, (str, Path)):
                    try:
                        os.unlink(self._fname)
                    except OSError:
                        warnings.warn(f'This save was not successful but {self._fname} could not be deleted.  This file is not valid.', ResourceWarning)
                raise exc
            else:
                self._close()

    def _close(self):
        '\n        Close the file if it was created by the writer.\n\n        If a buffer or file-like object was passed in, for example a GzipFile,\n        then leave this file open for the caller to close.\n        '
        if (self._output_file is not None):
            assert isinstance(self.handles.handle, BytesIO)
            bio = self.handles.handle
            bio.seek(0)
            self.handles.handle = self._output_file
            self.handles.handle.write(bio.read())
            bio.close()

    def _write_map(self):
        'No-op, future compatibility'
        pass

    def _write_file_close_tag(self):
        'No-op, future compatibility'
        pass

    def _write_characteristics(self):
        'No-op, future compatibility'
        pass

    def _write_strls(self):
        'No-op, future compatibility'
        pass

    def _write_expansion_fields(self):
        'Write 5 zeros for expansion fields'
        self._write(_pad_bytes('', 5))

    def _write_value_labels(self):
        for vl in self._value_labels:
            self._write_bytes(vl.generate_value_label(self._byteorder))

    def _write_header(self, data_label=None, time_stamp=None):
        byteorder = self._byteorder
        self._write_bytes(struct.pack('b', 114))
        self._write((((byteorder == '>') and '\x01') or '\x02'))
        self._write('\x01')
        self._write('\x00')
        self._write_bytes(struct.pack((byteorder + 'h'), self.nvar)[:2])
        self._write_bytes(struct.pack((byteorder + 'i'), self.nobs)[:4])
        if (data_label is None):
            self._write_bytes(self._null_terminate_bytes(_pad_bytes('', 80)))
        else:
            self._write_bytes(self._null_terminate_bytes(_pad_bytes(data_label[:80], 80)))
        if (time_stamp is None):
            time_stamp = datetime.datetime.now()
        elif (not isinstance(time_stamp, datetime.datetime)):
            raise ValueError('time_stamp should be datetime type')
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_lookup = {(i + 1): month for (i, month) in enumerate(months)}
        ts = ((time_stamp.strftime('%d ') + month_lookup[time_stamp.month]) + time_stamp.strftime(' %Y %H:%M'))
        self._write_bytes(self._null_terminate_bytes(ts))

    def _write_variable_types(self):
        for typ in self.typlist:
            self._write_bytes(struct.pack('B', typ))

    def _write_varnames(self):
        for name in self.varlist:
            name = self._null_terminate_str(name)
            name = _pad_bytes(name[:32], 33)
            self._write(name)

    def _write_sortlist(self):
        srtlist = _pad_bytes('', (2 * (self.nvar + 1)))
        self._write(srtlist)

    def _write_formats(self):
        for fmt in self.fmtlist:
            self._write(_pad_bytes(fmt, 49))

    def _write_value_label_names(self):
        for i in range(self.nvar):
            if self._is_col_cat[i]:
                name = self.varlist[i]
                name = self._null_terminate_str(name)
                name = _pad_bytes(name[:32], 33)
                self._write(name)
            else:
                self._write(_pad_bytes('', 33))

    def _write_variable_labels(self):
        blank = _pad_bytes('', 81)
        if (self._variable_labels is None):
            for i in range(self.nvar):
                self._write(blank)
            return
        for col in self.data:
            if (col in self._variable_labels):
                label = self._variable_labels[col]
                if (len(label) > 80):
                    raise ValueError('Variable labels must be 80 characters or fewer')
                is_latin1 = all(((ord(c) < 256) for c in label))
                if (not is_latin1):
                    raise ValueError('Variable labels must contain only characters that can be encoded in Latin-1')
                self._write(_pad_bytes(label, 81))
            else:
                self._write(blank)

    def _convert_strls(self, data):
        'No-op, future compatibility'
        return data

    def _prepare_data(self):
        data = self.data
        typlist = self.typlist
        convert_dates = self._convert_dates
        if (self._convert_dates is not None):
            for (i, col) in enumerate(data):
                if (i in convert_dates):
                    data[col] = _datetime_to_stata_elapsed_vec(data[col], self.fmtlist[i])
        data = self._convert_strls(data)
        dtypes = {}
        native_byteorder = (self._byteorder == _set_endianness(sys.byteorder))
        for (i, col) in enumerate(data):
            typ = typlist[i]
            if (typ <= self._max_string_length):
                data[col] = data[col].fillna('').apply(_pad_bytes, args=(typ,))
                stype = f'S{typ}'
                dtypes[col] = stype
                data[col] = data[col].astype(stype)
            else:
                dtype = data[col].dtype
                if (not native_byteorder):
                    dtype = dtype.newbyteorder(self._byteorder)
                dtypes[col] = dtype
        return data.to_records(index=False, column_dtypes=dtypes)

    def _write_data(self, records):
        self._write_bytes(records.tobytes())

    @staticmethod
    def _null_terminate_str(s):
        s += '\x00'
        return s

    def _null_terminate_bytes(self, s):
        return self._null_terminate_str(s).encode(self._encoding)

def _dtype_to_stata_type_117(dtype, column, force_strl):
    '\n    Converts dtype types to stata types. Returns the byte of the given ordinal.\n    See TYPE_MAP and comments for an explanation. This is also explained in\n    the dta spec.\n    1 - 2045 are strings of this length\n                Pandas    Stata\n    32768 - for object    strL\n    65526 - for int8      byte\n    65527 - for int16     int\n    65528 - for int32     long\n    65529 - for float32   float\n    65530 - for double    double\n\n    If there are dates to convert, then dtype will already have the correct\n    type inserted.\n    '
    if force_strl:
        return 32768
    if (dtype.type == np.object_):
        itemsize = max_len_string_array(ensure_object(column._values))
        itemsize = max(itemsize, 1)
        if (itemsize <= 2045):
            return itemsize
        return 32768
    elif (dtype == np.float64):
        return 65526
    elif (dtype == np.float32):
        return 65527
    elif (dtype == np.int32):
        return 65528
    elif (dtype == np.int16):
        return 65529
    elif (dtype == np.int8):
        return 65530
    else:
        raise NotImplementedError(f'Data type {dtype} not supported.')

def _pad_bytes_new(name, length):
    "\n    Takes a bytes instance and pads it with null bytes until it's length chars.\n    "
    if isinstance(name, str):
        name = bytes(name, 'utf-8')
    return (name + (b'\x00' * (length - len(name))))

class StataStrLWriter():
    '\n    Converter for Stata StrLs\n\n    Stata StrLs map 8 byte values to strings which are stored using a\n    dictionary-like format where strings are keyed to two values.\n\n    Parameters\n    ----------\n    df : DataFrame\n        DataFrame to convert\n    columns : Sequence[str]\n        List of columns names to convert to StrL\n    version : int, optional\n        dta version.  Currently supports 117, 118 and 119\n    byteorder : str, optional\n        Can be ">", "<", "little", or "big". default is `sys.byteorder`\n\n    Notes\n    -----\n    Supports creation of the StrL block of a dta file for dta versions\n    117, 118 and 119.  These differ in how the GSO is stored.  118 and\n    119 store the GSO lookup value as a uint32 and a uint64, while 117\n    uses two uint32s. 118 and 119 also encode all strings as unicode\n    which is required by the format.  117 uses \'latin-1\' a fixed width\n    encoding that extends the 7-bit ascii table with an additional 128\n    characters.\n    '

    def __init__(self, df, columns, version=117, byteorder=None):
        if (version not in (117, 118, 119)):
            raise ValueError('Only dta versions 117, 118 and 119 supported')
        self._dta_ver = version
        self.df = df
        self.columns = columns
        self._gso_table = {'': (0, 0)}
        if (byteorder is None):
            byteorder = sys.byteorder
        self._byteorder = _set_endianness(byteorder)
        gso_v_type = 'I'
        gso_o_type = 'Q'
        self._encoding = 'utf-8'
        if (version == 117):
            o_size = 4
            gso_o_type = 'I'
            self._encoding = 'latin-1'
        elif (version == 118):
            o_size = 6
        else:
            o_size = 5
        self._o_offet = (2 ** (8 * (8 - o_size)))
        self._gso_o_type = gso_o_type
        self._gso_v_type = gso_v_type

    def _convert_key(self, key):
        (v, o) = key
        return (v + (self._o_offet * o))

    def generate_table(self):
        '\n        Generates the GSO lookup table for the DataFrame\n\n        Returns\n        -------\n        gso_table : dict\n            Ordered dictionary using the string found as keys\n            and their lookup position (v,o) as values\n        gso_df : DataFrame\n            DataFrame where strl columns have been converted to\n            (v,o) values\n\n        Notes\n        -----\n        Modifies the DataFrame in-place.\n\n        The DataFrame returned encodes the (v,o) values as uint64s. The\n        encoding depends on the dta version, and can be expressed as\n\n        enc = v + o * 2 ** (o_size * 8)\n\n        so that v is stored in the lower bits and o is in the upper\n        bits. o_size is\n\n          * 117: 4\n          * 118: 6\n          * 119: 5\n        '
        gso_table = self._gso_table
        gso_df = self.df
        columns = list(gso_df.columns)
        selected = gso_df[self.columns]
        col_index = [(col, columns.index(col)) for col in self.columns]
        keys = np.empty(selected.shape, dtype=np.uint64)
        for (o, (idx, row)) in enumerate(selected.iterrows()):
            for (j, (col, v)) in enumerate(col_index):
                val = row[col]
                val = ('' if (val is None) else val)
                key = gso_table.get(val, None)
                if (key is None):
                    key = ((v + 1), (o + 1))
                    gso_table[val] = key
                keys[(o, j)] = self._convert_key(key)
        for (i, col) in enumerate(self.columns):
            gso_df[col] = keys[:, i]
        return (gso_table, gso_df)

    def generate_blob(self, gso_table):
        '\n        Generates the binary blob of GSOs that is written to the dta file.\n\n        Parameters\n        ----------\n        gso_table : dict\n            Ordered dictionary (str, vo)\n\n        Returns\n        -------\n        gso : bytes\n            Binary content of dta file to be placed between strl tags\n\n        Notes\n        -----\n        Output format depends on dta version.  117 uses two uint32s to\n        express v and o while 118+ uses a uint32 for v and a uint64 for o.\n        '
        bio = BytesIO()
        gso = bytes('GSO', 'ascii')
        gso_type = struct.pack((self._byteorder + 'B'), 130)
        null = struct.pack((self._byteorder + 'B'), 0)
        v_type = (self._byteorder + self._gso_v_type)
        o_type = (self._byteorder + self._gso_o_type)
        len_type = (self._byteorder + 'I')
        for (strl, vo) in gso_table.items():
            if (vo == (0, 0)):
                continue
            (v, o) = vo
            bio.write(gso)
            bio.write(struct.pack(v_type, v))
            bio.write(struct.pack(o_type, o))
            bio.write(gso_type)
            utf8_string = bytes(strl, 'utf-8')
            bio.write(struct.pack(len_type, (len(utf8_string) + 1)))
            bio.write(utf8_string)
            bio.write(null)
        bio.seek(0)
        return bio.read()

class StataWriter117(StataWriter):
    '\n    A class for writing Stata binary dta files in Stata 13 format (117)\n\n    Parameters\n    ----------\n    fname : path (string), buffer or path object\n        string, path object (pathlib.Path or py._path.local.LocalPath) or\n        object implementing a binary write() functions. If using a buffer\n        then the buffer will not be automatically closed after the file\n        is written.\n    data : DataFrame\n        Input to save\n    convert_dates : dict\n        Dictionary mapping columns containing datetime types to stata internal\n        format to use when writing the dates. Options are \'tc\', \'td\', \'tm\',\n        \'tw\', \'th\', \'tq\', \'ty\'. Column can be either an integer or a name.\n        Datetime columns that do not have a conversion type specified will be\n        converted to \'tc\'. Raises NotImplementedError if a datetime column has\n        timezone information\n    write_index : bool\n        Write the index to Stata dataset.\n    byteorder : str\n        Can be ">", "<", "little", or "big". default is `sys.byteorder`\n    time_stamp : datetime\n        A datetime to use as file creation date.  Default is the current time\n    data_label : str\n        A label for the data set.  Must be 80 characters or smaller.\n    variable_labels : dict\n        Dictionary containing columns as keys and variable labels as values.\n        Each label must be 80 characters or smaller.\n    convert_strl : list\n        List of columns names to convert to Stata StrL format.  Columns with\n        more than 2045 characters are automatically written as StrL.\n        Smaller columns can be converted by including the column name.  Using\n        StrLs can reduce output file size when strings are longer than 8\n        characters, and either frequently repeated or sparse.\n    compression : str or dict, default \'infer\'\n        For on-the-fly compression of the output dta. If string, specifies\n        compression mode. If dict, value at key \'method\' specifies compression\n        mode. Compression mode must be one of {\'infer\', \'gzip\', \'bz2\', \'zip\',\n        \'xz\', None}. If compression mode is \'infer\' and `fname` is path-like,\n        then detect compression from the following extensions: \'.gz\', \'.bz2\',\n        \'.zip\', or \'.xz\' (otherwise no compression). If dict and compression\n        mode is one of {\'zip\', \'gzip\', \'bz2\'}, or inferred as one of the above,\n        other entries passed as additional compression options.\n\n        .. versionadded:: 1.1.0\n\n    Returns\n    -------\n    writer : StataWriter117 instance\n        The StataWriter117 instance has a write_file method, which will\n        write the file to the given `fname`.\n\n    Raises\n    ------\n    NotImplementedError\n        * If datetimes contain timezone information\n    ValueError\n        * Columns listed in convert_dates are neither datetime64[ns]\n          or datetime.datetime\n        * Column dtype is not representable in Stata\n        * Column listed in convert_dates is not in DataFrame\n        * Categorical label contains more than 32,000 characters\n\n    Examples\n    --------\n    >>> from pandas.io.stata import StataWriter117\n    >>> data = pd.DataFrame([[1.0, 1, \'a\']], columns=[\'a\', \'b\', \'c\'])\n    >>> writer = StataWriter117(\'./data_file.dta\', data)\n    >>> writer.write_file()\n\n    Directly write a zip file\n    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}\n    >>> writer = StataWriter117(\'./data_file.zip\', data, compression=compression)\n    >>> writer.write_file()\n\n    Or with long strings stored in strl format\n    >>> data = pd.DataFrame([[\'A relatively long string\'], [\'\'], [\'\']],\n    ...                     columns=[\'strls\'])\n    >>> writer = StataWriter117(\'./data_file_with_long_strings.dta\', data,\n    ...                         convert_strl=[\'strls\'])\n    >>> writer.write_file()\n    '
    _max_string_length = 2045
    _dta_version = 117

    def __init__(self, fname, data, convert_dates=None, write_index=True, byteorder=None, time_stamp=None, data_label=None, variable_labels=None, convert_strl=None, compression='infer', storage_options=None):
        self._convert_strl: List[Label] = []
        if (convert_strl is not None):
            self._convert_strl.extend(convert_strl)
        super().__init__(fname, data, convert_dates, write_index, byteorder=byteorder, time_stamp=time_stamp, data_label=data_label, variable_labels=variable_labels, compression=compression, storage_options=storage_options)
        self._map: Dict[(str, int)] = {}
        self._strl_blob = b''

    @staticmethod
    def _tag(val, tag):
        'Surround val with <tag></tag>'
        if isinstance(val, str):
            val = bytes(val, 'utf-8')
        return ((bytes((('<' + tag) + '>'), 'utf-8') + val) + bytes((('</' + tag) + '>'), 'utf-8'))

    def _update_map(self, tag):
        'Update map location for tag with file position'
        assert (self.handles.handle is not None)
        self._map[tag] = self.handles.handle.tell()

    def _write_header(self, data_label=None, time_stamp=None):
        'Write the file header'
        byteorder = self._byteorder
        self._write_bytes(bytes('<stata_dta>', 'utf-8'))
        bio = BytesIO()
        bio.write(self._tag(bytes(str(self._dta_version), 'utf-8'), 'release'))
        bio.write(self._tag((((byteorder == '>') and 'MSF') or 'LSF'), 'byteorder'))
        nvar_type = ('H' if (self._dta_version <= 118) else 'I')
        bio.write(self._tag(struct.pack((byteorder + nvar_type), self.nvar), 'K'))
        nobs_size = ('I' if (self._dta_version == 117) else 'Q')
        bio.write(self._tag(struct.pack((byteorder + nobs_size), self.nobs), 'N'))
        label = (data_label[:80] if (data_label is not None) else '')
        encoded_label = label.encode(self._encoding)
        label_size = ('B' if (self._dta_version == 117) else 'H')
        label_len = struct.pack((byteorder + label_size), len(encoded_label))
        encoded_label = (label_len + encoded_label)
        bio.write(self._tag(encoded_label, 'label'))
        if (time_stamp is None):
            time_stamp = datetime.datetime.now()
        elif (not isinstance(time_stamp, datetime.datetime)):
            raise ValueError('time_stamp should be datetime type')
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_lookup = {(i + 1): month for (i, month) in enumerate(months)}
        ts = ((time_stamp.strftime('%d ') + month_lookup[time_stamp.month]) + time_stamp.strftime(' %Y %H:%M'))
        stata_ts = (b'\x11' + bytes(ts, 'utf-8'))
        bio.write(self._tag(stata_ts, 'timestamp'))
        bio.seek(0)
        self._write_bytes(self._tag(bio.read(), 'header'))

    def _write_map(self):
        '\n        Called twice during file write. The first populates the values in\n        the map with 0s.  The second call writes the final map locations when\n        all blocks have been written.\n        '
        if (not self._map):
            self._map = {'stata_data': 0, 'map': self.handles.handle.tell(), 'variable_types': 0, 'varnames': 0, 'sortlist': 0, 'formats': 0, 'value_label_names': 0, 'variable_labels': 0, 'characteristics': 0, 'data': 0, 'strls': 0, 'value_labels': 0, 'stata_data_close': 0, 'end-of-file': 0}
        self.handles.handle.seek(self._map['map'])
        bio = BytesIO()
        for val in self._map.values():
            bio.write(struct.pack((self._byteorder + 'Q'), val))
        bio.seek(0)
        self._write_bytes(self._tag(bio.read(), 'map'))

    def _write_variable_types(self):
        self._update_map('variable_types')
        bio = BytesIO()
        for typ in self.typlist:
            bio.write(struct.pack((self._byteorder + 'H'), typ))
        bio.seek(0)
        self._write_bytes(self._tag(bio.read(), 'variable_types'))

    def _write_varnames(self):
        self._update_map('varnames')
        bio = BytesIO()
        vn_len = (32 if (self._dta_version == 117) else 128)
        for name in self.varlist:
            name = self._null_terminate_str(name)
            name = _pad_bytes_new(name[:32].encode(self._encoding), (vn_len + 1))
            bio.write(name)
        bio.seek(0)
        self._write_bytes(self._tag(bio.read(), 'varnames'))

    def _write_sortlist(self):
        self._update_map('sortlist')
        sort_size = (2 if (self._dta_version < 119) else 4)
        self._write_bytes(self._tag(((b'\x00' * sort_size) * (self.nvar + 1)), 'sortlist'))

    def _write_formats(self):
        self._update_map('formats')
        bio = BytesIO()
        fmt_len = (49 if (self._dta_version == 117) else 57)
        for fmt in self.fmtlist:
            bio.write(_pad_bytes_new(fmt.encode(self._encoding), fmt_len))
        bio.seek(0)
        self._write_bytes(self._tag(bio.read(), 'formats'))

    def _write_value_label_names(self):
        self._update_map('value_label_names')
        bio = BytesIO()
        vl_len = (32 if (self._dta_version == 117) else 128)
        for i in range(self.nvar):
            name = ''
            if self._is_col_cat[i]:
                name = self.varlist[i]
            name = self._null_terminate_str(name)
            encoded_name = _pad_bytes_new(name[:32].encode(self._encoding), (vl_len + 1))
            bio.write(encoded_name)
        bio.seek(0)
        self._write_bytes(self._tag(bio.read(), 'value_label_names'))

    def _write_variable_labels(self):
        self._update_map('variable_labels')
        bio = BytesIO()
        vl_len = (80 if (self._dta_version == 117) else 320)
        blank = _pad_bytes_new('', (vl_len + 1))
        if (self._variable_labels is None):
            for _ in range(self.nvar):
                bio.write(blank)
            bio.seek(0)
            self._write_bytes(self._tag(bio.read(), 'variable_labels'))
            return
        for col in self.data:
            if (col in self._variable_labels):
                label = self._variable_labels[col]
                if (len(label) > 80):
                    raise ValueError('Variable labels must be 80 characters or fewer')
                try:
                    encoded = label.encode(self._encoding)
                except UnicodeEncodeError as err:
                    raise ValueError(f'Variable labels must contain only characters that can be encoded in {self._encoding}') from err
                bio.write(_pad_bytes_new(encoded, (vl_len + 1)))
            else:
                bio.write(blank)
        bio.seek(0)
        self._write_bytes(self._tag(bio.read(), 'variable_labels'))

    def _write_characteristics(self):
        self._update_map('characteristics')
        self._write_bytes(self._tag(b'', 'characteristics'))

    def _write_data(self, records):
        self._update_map('data')
        self._write_bytes(b'<data>')
        self._write_bytes(records.tobytes())
        self._write_bytes(b'</data>')

    def _write_strls(self):
        self._update_map('strls')
        self._write_bytes(self._tag(self._strl_blob, 'strls'))

    def _write_expansion_fields(self):
        'No-op in dta 117+'
        pass

    def _write_value_labels(self):
        self._update_map('value_labels')
        bio = BytesIO()
        for vl in self._value_labels:
            lab = vl.generate_value_label(self._byteorder)
            lab = self._tag(lab, 'lbl')
            bio.write(lab)
        bio.seek(0)
        self._write_bytes(self._tag(bio.read(), 'value_labels'))

    def _write_file_close_tag(self):
        self._update_map('stata_data_close')
        self._write_bytes(bytes('</stata_dta>', 'utf-8'))
        self._update_map('end-of-file')

    def _update_strl_names(self):
        '\n        Update column names for conversion to strl if they might have been\n        changed to comply with Stata naming rules\n        '
        for (orig, new) in self._converted_names.items():
            if (orig in self._convert_strl):
                idx = self._convert_strl.index(orig)
                self._convert_strl[idx] = new

    def _convert_strls(self, data):
        '\n        Convert columns to StrLs if either very large or in the\n        convert_strl variable\n        '
        convert_cols = [col for (i, col) in enumerate(data) if ((self.typlist[i] == 32768) or (col in self._convert_strl))]
        if convert_cols:
            ssw = StataStrLWriter(data, convert_cols, version=self._dta_version)
            (tab, new_data) = ssw.generate_table()
            data = new_data
            self._strl_blob = ssw.generate_blob(tab)
        return data

    def _set_formats_and_types(self, dtypes):
        self.typlist = []
        self.fmtlist = []
        for (col, dtype) in dtypes.items():
            force_strl = (col in self._convert_strl)
            fmt = _dtype_to_default_stata_fmt(dtype, self.data[col], dta_version=self._dta_version, force_strl=force_strl)
            self.fmtlist.append(fmt)
            self.typlist.append(_dtype_to_stata_type_117(dtype, self.data[col], force_strl))

class StataWriterUTF8(StataWriter117):
    '\n    Stata binary dta file writing in Stata 15 (118) and 16 (119) formats\n\n    DTA 118 and 119 format files support unicode string data (both fixed\n    and strL) format. Unicode is also supported in value labels, variable\n    labels and the dataset label. Format 119 is automatically used if the\n    file contains more than 32,767 variables.\n\n    .. versionadded:: 1.0.0\n\n    Parameters\n    ----------\n    fname : path (string), buffer or path object\n        string, path object (pathlib.Path or py._path.local.LocalPath) or\n        object implementing a binary write() functions. If using a buffer\n        then the buffer will not be automatically closed after the file\n        is written.\n    data : DataFrame\n        Input to save\n    convert_dates : dict, default None\n        Dictionary mapping columns containing datetime types to stata internal\n        format to use when writing the dates. Options are \'tc\', \'td\', \'tm\',\n        \'tw\', \'th\', \'tq\', \'ty\'. Column can be either an integer or a name.\n        Datetime columns that do not have a conversion type specified will be\n        converted to \'tc\'. Raises NotImplementedError if a datetime column has\n        timezone information\n    write_index : bool, default True\n        Write the index to Stata dataset.\n    byteorder : str, default None\n        Can be ">", "<", "little", or "big". default is `sys.byteorder`\n    time_stamp : datetime, default None\n        A datetime to use as file creation date.  Default is the current time\n    data_label : str, default None\n        A label for the data set.  Must be 80 characters or smaller.\n    variable_labels : dict, default None\n        Dictionary containing columns as keys and variable labels as values.\n        Each label must be 80 characters or smaller.\n    convert_strl : list, default None\n        List of columns names to convert to Stata StrL format.  Columns with\n        more than 2045 characters are automatically written as StrL.\n        Smaller columns can be converted by including the column name.  Using\n        StrLs can reduce output file size when strings are longer than 8\n        characters, and either frequently repeated or sparse.\n    version : int, default None\n        The dta version to use. By default, uses the size of data to determine\n        the version. 118 is used if data.shape[1] <= 32767, and 119 is used\n        for storing larger DataFrames.\n    compression : str or dict, default \'infer\'\n        For on-the-fly compression of the output dta. If string, specifies\n        compression mode. If dict, value at key \'method\' specifies compression\n        mode. Compression mode must be one of {\'infer\', \'gzip\', \'bz2\', \'zip\',\n        \'xz\', None}. If compression mode is \'infer\' and `fname` is path-like,\n        then detect compression from the following extensions: \'.gz\', \'.bz2\',\n        \'.zip\', or \'.xz\' (otherwise no compression). If dict and compression\n        mode is one of {\'zip\', \'gzip\', \'bz2\'}, or inferred as one of the above,\n        other entries passed as additional compression options.\n\n        .. versionadded:: 1.1.0\n\n    Returns\n    -------\n    StataWriterUTF8\n        The instance has a write_file method, which will write the file to the\n        given `fname`.\n\n    Raises\n    ------\n    NotImplementedError\n        * If datetimes contain timezone information\n    ValueError\n        * Columns listed in convert_dates are neither datetime64[ns]\n          or datetime.datetime\n        * Column dtype is not representable in Stata\n        * Column listed in convert_dates is not in DataFrame\n        * Categorical label contains more than 32,000 characters\n\n    Examples\n    --------\n    Using Unicode data and column names\n\n    >>> from pandas.io.stata import StataWriterUTF8\n    >>> data = pd.DataFrame([[1.0, 1, \'ᴬ\']], columns=[\'a\', \'β\', \'ĉ\'])\n    >>> writer = StataWriterUTF8(\'./data_file.dta\', data)\n    >>> writer.write_file()\n\n    Directly write a zip file\n    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}\n    >>> writer = StataWriterUTF8(\'./data_file.zip\', data, compression=compression)\n    >>> writer.write_file()\n\n    Or with long strings stored in strl format\n\n    >>> data = pd.DataFrame([[\'ᴀ relatively long ŝtring\'], [\'\'], [\'\']],\n    ...                     columns=[\'strls\'])\n    >>> writer = StataWriterUTF8(\'./data_file_with_long_strings.dta\', data,\n    ...                          convert_strl=[\'strls\'])\n    >>> writer.write_file()\n    '
    _encoding = 'utf-8'

    def __init__(self, fname, data, convert_dates=None, write_index=True, byteorder=None, time_stamp=None, data_label=None, variable_labels=None, convert_strl=None, version=None, compression='infer', storage_options=None):
        if (version is None):
            version = (118 if (data.shape[1] <= 32767) else 119)
        elif (version not in (118, 119)):
            raise ValueError('version must be either 118 or 119.')
        elif ((version == 118) and (data.shape[1] > 32767)):
            raise ValueError('You must use version 119 for data sets containing more than32,767 variables')
        super().__init__(fname, data, convert_dates=convert_dates, write_index=write_index, byteorder=byteorder, time_stamp=time_stamp, data_label=data_label, variable_labels=variable_labels, convert_strl=convert_strl, compression=compression, storage_options=storage_options)
        self._dta_version = version

    def _validate_variable_name(self, name):
        '\n        Validate variable names for Stata export.\n\n        Parameters\n        ----------\n        name : str\n            Variable name\n\n        Returns\n        -------\n        str\n            The validated name with invalid characters replaced with\n            underscores.\n\n        Notes\n        -----\n        Stata 118+ support most unicode characters. The only limitation is in\n        the ascii range where the characters supported are a-z, A-Z, 0-9 and _.\n        '
        for c in name:
            if (((ord(c) < 128) and ((c < 'A') or (c > 'Z')) and ((c < 'a') or (c > 'z')) and ((c < '0') or (c > '9')) and (c != '_')) or (128 <= ord(c) < 256)):
                name = name.replace(c, '_')
        return name
