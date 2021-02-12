
'\nInternal module for formatting output data in csv, html,\nand latex files. This module also applies to display formatting.\n'
from contextlib import contextmanager
from csv import QUOTE_NONE, QUOTE_NONNUMERIC
import decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast
from unicodedata import east_asian_width
import numpy as np
from pandas._config.config import get_option, set_option
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import NaT, Timedelta, Timestamp, iNaT
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import ArrayLike, ColspaceArgType, ColspaceType, CompressionOptions, FilePathOrBuffer, FloatFormatType, FormattersType, IndexLabel, Label, StorageOptions
from pandas.core.dtypes.common import is_categorical_dtype, is_complex_dtype, is_datetime64_dtype, is_datetime64tz_dtype, is_extension_array_dtype, is_float, is_float_dtype, is_integer, is_integer_dtype, is_list_like, is_numeric_dtype, is_scalar, is_timedelta64_dtype
from pandas.core.dtypes.missing import isna, notna
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexes.api import Index, MultiIndex, PeriodIndex, ensure_index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat
from pandas.io.common import stringify_path
from pandas.io.formats.printing import adjoin, justify, pprint_thing
if TYPE_CHECKING:
    from pandas import Categorical, DataFrame, Series
common_docstring = "\n        Parameters\n        ----------\n        buf : str, Path or StringIO-like, optional, default None\n            Buffer to write to. If None, the output is returned as a string.\n        columns : sequence, optional, default None\n            The subset of columns to write. Writes all columns by default.\n        col_space : %(col_space_type)s, optional\n            %(col_space)s.\n        header : %(header_type)s, optional\n            %(header)s.\n        index : bool, optional, default True\n            Whether to print index (row) labels.\n        na_rep : str, optional, default 'NaN'\n            String representation of ``NaN`` to use.\n        formatters : list, tuple or dict of one-param. functions, optional\n            Formatter functions to apply to columns' elements by position or\n            name.\n            The result of each function must be a unicode string.\n            List/tuple must be of length equal to the number of columns.\n        float_format : one-parameter function, optional, default None\n            Formatter function to apply to columns' elements if they are\n            floats. This function must return a unicode string and will be\n            applied only to the non-``NaN`` elements, with ``NaN`` being\n            handled by ``na_rep``.\n\n            .. versionchanged:: 1.2.0\n\n        sparsify : bool, optional, default True\n            Set to False for a DataFrame with a hierarchical index to print\n            every multiindex key at each row.\n        index_names : bool, optional, default True\n            Prints the names of the indexes.\n        justify : str, default None\n            How to justify the column labels. If None uses the option from\n            the print configuration (controlled by set_option), 'right' out\n            of the box. Valid values are\n\n            * left\n            * right\n            * center\n            * justify\n            * justify-all\n            * start\n            * end\n            * inherit\n            * match-parent\n            * initial\n            * unset.\n        max_rows : int, optional\n            Maximum number of rows to display in the console.\n        min_rows : int, optional\n            The number of rows to display in the console in a truncated repr\n            (when number of rows is above `max_rows`).\n        max_cols : int, optional\n            Maximum number of columns to display in the console.\n        show_dimensions : bool, default False\n            Display DataFrame dimensions (number of rows by number of columns).\n        decimal : str, default '.'\n            Character recognized as decimal separator, e.g. ',' in Europe.\n    "
_VALID_JUSTIFY_PARAMETERS = ('left', 'right', 'center', 'justify', 'justify-all', 'start', 'end', 'inherit', 'match-parent', 'initial', 'unset')
return_docstring = '\n        Returns\n        -------\n        str or None\n            If buf is None, returns the result as a string. Otherwise returns\n            None.\n    '

class CategoricalFormatter():

    def __init__(self, categorical, buf=None, length=True, na_rep='NaN', footer=True):
        self.categorical = categorical
        self.buf = (buf if (buf is not None) else StringIO(''))
        self.na_rep = na_rep
        self.length = length
        self.footer = footer
        self.quoting = QUOTE_NONNUMERIC

    def _get_footer(self):
        footer = ''
        if self.length:
            if footer:
                footer += ', '
            footer += f'Length: {len(self.categorical)}'
        level_info = self.categorical._repr_categories_info()
        if footer:
            footer += '\n'
        footer += level_info
        return str(footer)

    def _get_formatted_values(self):
        return format_array(self.categorical._internal_get_values(), None, float_format=None, na_rep=self.na_rep, quoting=self.quoting)

    def to_string(self):
        categorical = self.categorical
        if (len(categorical) == 0):
            if self.footer:
                return self._get_footer()
            else:
                return ''
        fmt_values = self._get_formatted_values()
        fmt_values = [i.strip() for i in fmt_values]
        values = ', '.join(fmt_values)
        result = [(('[' + values) + ']')]
        if self.footer:
            footer = self._get_footer()
            if footer:
                result.append(footer)
        return str('\n'.join(result))

class SeriesFormatter():

    def __init__(self, series, buf=None, length=True, header=True, index=True, na_rep='NaN', name=False, float_format=None, dtype=True, max_rows=None, min_rows=None):
        self.series = series
        self.buf = (buf if (buf is not None) else StringIO())
        self.name = name
        self.na_rep = na_rep
        self.header = header
        self.length = length
        self.index = index
        self.max_rows = max_rows
        self.min_rows = min_rows
        if (float_format is None):
            float_format = get_option('display.float_format')
        self.float_format = float_format
        self.dtype = dtype
        self.adj = get_adjustment()
        self._chk_truncate()

    def _chk_truncate(self):
        self.tr_row_num: Optional[int]
        min_rows = self.min_rows
        max_rows = self.max_rows
        is_truncated_vertically = (max_rows and (len(self.series) > max_rows))
        series = self.series
        if is_truncated_vertically:
            max_rows = cast(int, max_rows)
            if min_rows:
                max_rows = min(min_rows, max_rows)
            if (max_rows == 1):
                row_num = max_rows
                series = series.iloc[:max_rows]
            else:
                row_num = (max_rows // 2)
                series = concat((series.iloc[:row_num], series.iloc[(- row_num):]))
            self.tr_row_num = row_num
        else:
            self.tr_row_num = None
        self.tr_series = series
        self.is_truncated_vertically = is_truncated_vertically

    def _get_footer(self):
        name = self.series.name
        footer = ''
        if (getattr(self.series.index, 'freq', None) is not None):
            assert isinstance(self.series.index, (DatetimeIndex, PeriodIndex, TimedeltaIndex))
            footer += f'Freq: {self.series.index.freqstr}'
        if ((self.name is not False) and (name is not None)):
            if footer:
                footer += ', '
            series_name = pprint_thing(name, escape_chars=('\t', '\r', '\n'))
            footer += f'Name: {series_name}'
        if ((self.length is True) or ((self.length == 'truncate') and self.is_truncated_vertically)):
            if footer:
                footer += ', '
            footer += f'Length: {len(self.series)}'
        if ((self.dtype is not False) and (self.dtype is not None)):
            dtype_name = getattr(self.tr_series.dtype, 'name', None)
            if dtype_name:
                if footer:
                    footer += ', '
                footer += f'dtype: {pprint_thing(dtype_name)}'
        if is_categorical_dtype(self.tr_series.dtype):
            level_info = self.tr_series._values._repr_categories_info()
            if footer:
                footer += '\n'
            footer += level_info
        return str(footer)

    def _get_formatted_index(self):
        index = self.tr_series.index
        if isinstance(index, MultiIndex):
            have_header = any((name for name in index.names))
            fmt_index = index.format(names=True)
        else:
            have_header = (index.name is not None)
            fmt_index = index.format(name=True)
        return (fmt_index, have_header)

    def _get_formatted_values(self):
        return format_array(self.tr_series._values, None, float_format=self.float_format, na_rep=self.na_rep, leading_space=self.index)

    def to_string(self):
        series = self.tr_series
        footer = self._get_footer()
        if (len(series) == 0):
            return f'{type(self.series).__name__}([], {footer})'
        (fmt_index, have_header) = self._get_formatted_index()
        fmt_values = self._get_formatted_values()
        if self.is_truncated_vertically:
            n_header_rows = 0
            row_num = self.tr_row_num
            row_num = cast(int, row_num)
            width = self.adj.len(fmt_values[(row_num - 1)])
            if (width > 3):
                dot_str = '...'
            else:
                dot_str = '..'
            dot_str = self.adj.justify([dot_str], width, mode='center')[0]
            fmt_values.insert((row_num + n_header_rows), dot_str)
            fmt_index.insert((row_num + 1), '')
        if self.index:
            result = self.adj.adjoin(3, *[fmt_index[1:], fmt_values])
        else:
            result = self.adj.adjoin(3, fmt_values)
        if (self.header and have_header):
            result = ((fmt_index[0] + '\n') + result)
        if footer:
            result += ('\n' + footer)
        return str(''.join(result))

class TextAdjustment():

    def __init__(self):
        self.encoding = get_option('display.encoding')

    def len(self, text):
        return len(text)

    def justify(self, texts, max_len, mode='right'):
        return justify(texts, max_len, mode=mode)

    def adjoin(self, space, *lists, **kwargs):
        return adjoin(space, *lists, strlen=self.len, justfunc=self.justify, **kwargs)

class EastAsianTextAdjustment(TextAdjustment):

    def __init__(self):
        super().__init__()
        if get_option('display.unicode.ambiguous_as_wide'):
            self.ambiguous_width = 2
        else:
            self.ambiguous_width = 1
        self._EAW_MAP = {'Na': 1, 'N': 1, 'W': 2, 'F': 2, 'H': 1}

    def len(self, text):
        '\n        Calculate display width considering unicode East Asian Width\n        '
        if (not isinstance(text, str)):
            return len(text)
        return sum((self._EAW_MAP.get(east_asian_width(c), self.ambiguous_width) for c in text))

    def justify(self, texts, max_len, mode='right'):

        def _get_pad(t):
            return ((max_len - self.len(t)) + len(t))
        if (mode == 'left'):
            return [x.ljust(_get_pad(x)) for x in texts]
        elif (mode == 'center'):
            return [x.center(_get_pad(x)) for x in texts]
        else:
            return [x.rjust(_get_pad(x)) for x in texts]

def get_adjustment():
    use_east_asian_width = get_option('display.unicode.east_asian_width')
    if use_east_asian_width:
        return EastAsianTextAdjustment()
    else:
        return TextAdjustment()

class DataFrameFormatter():
    'Class for processing dataframe formatting options and data.'
    __doc__ = (__doc__ if __doc__ else '')
    __doc__ += (common_docstring + return_docstring)

    def __init__(self, frame, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, justify=None, float_format=None, sparsify=None, index_names=True, max_rows=None, min_rows=None, max_cols=None, show_dimensions=False, decimal='.', bold_rows=False, escape=True):
        self.frame = frame
        self.columns = self._initialize_columns(columns)
        self.col_space = self._initialize_colspace(col_space)
        self.header = header
        self.index = index
        self.na_rep = na_rep
        self.formatters = self._initialize_formatters(formatters)
        self.justify = self._initialize_justify(justify)
        self.float_format = float_format
        self.sparsify = self._initialize_sparsify(sparsify)
        self.show_index_names = index_names
        self.decimal = decimal
        self.bold_rows = bold_rows
        self.escape = escape
        self.max_rows = max_rows
        self.min_rows = min_rows
        self.max_cols = max_cols
        self.show_dimensions = show_dimensions
        self.max_cols_fitted = self._calc_max_cols_fitted()
        self.max_rows_fitted = self._calc_max_rows_fitted()
        self.tr_frame = self.frame
        self.truncate()
        self.adj = get_adjustment()

    def get_strcols(self):
        '\n        Render a DataFrame to a list of columns (as lists of strings).\n        '
        strcols = self._get_strcols_without_index()
        if self.index:
            str_index = self._get_formatted_index(self.tr_frame)
            strcols.insert(0, str_index)
        return strcols

    @property
    def should_show_dimensions(self):
        return ((self.show_dimensions is True) or ((self.show_dimensions == 'truncate') and self.is_truncated))

    @property
    def is_truncated(self):
        return bool((self.is_truncated_horizontally or self.is_truncated_vertically))

    @property
    def is_truncated_horizontally(self):
        return bool((self.max_cols_fitted and (len(self.columns) > self.max_cols_fitted)))

    @property
    def is_truncated_vertically(self):
        return bool((self.max_rows_fitted and (len(self.frame) > self.max_rows_fitted)))

    @property
    def dimensions_info(self):
        return f'''

[{len(self.frame)} rows x {len(self.frame.columns)} columns]'''

    @property
    def has_index_names(self):
        return _has_names(self.frame.index)

    @property
    def has_column_names(self):
        return _has_names(self.frame.columns)

    @property
    def show_row_idx_names(self):
        return all((self.has_index_names, self.index, self.show_index_names))

    @property
    def show_col_idx_names(self):
        return all((self.has_column_names, self.show_index_names, self.header))

    @property
    def max_rows_displayed(self):
        return min((self.max_rows or len(self.frame)), len(self.frame))

    def _initialize_sparsify(self, sparsify):
        if (sparsify is None):
            return get_option('display.multi_sparse')
        return sparsify

    def _initialize_formatters(self, formatters):
        if (formatters is None):
            return {}
        elif ((len(self.frame.columns) == len(formatters)) or isinstance(formatters, dict)):
            return formatters
        else:
            raise ValueError(f'Formatters length({len(formatters)}) should match DataFrame number of columns({len(self.frame.columns)})')

    def _initialize_justify(self, justify):
        if (justify is None):
            return get_option('display.colheader_justify')
        else:
            return justify

    def _initialize_columns(self, columns):
        if (columns is not None):
            cols = ensure_index(columns)
            self.frame = self.frame[cols]
            return cols
        else:
            return self.frame.columns

    def _initialize_colspace(self, col_space):
        result: ColspaceType
        if (col_space is None):
            result = {}
        elif isinstance(col_space, (int, str)):
            result = {'': col_space}
            result.update({column: col_space for column in self.frame.columns})
        elif isinstance(col_space, Mapping):
            for column in col_space.keys():
                if ((column not in self.frame.columns) and (column != '')):
                    raise ValueError(f'Col_space is defined for an unknown column: {column}')
            result = col_space
        else:
            if (len(self.frame.columns) != len(col_space)):
                raise ValueError(f'Col_space length({len(col_space)}) should match DataFrame number of columns({len(self.frame.columns)})')
            result = dict(zip(self.frame.columns, col_space))
        return result

    def _calc_max_cols_fitted(self):
        'Number of columns fitting the screen.'
        if (not self._is_in_terminal()):
            return self.max_cols
        (width, _) = get_terminal_size()
        if self._is_screen_narrow(width):
            return width
        else:
            return self.max_cols

    def _calc_max_rows_fitted(self):
        'Number of rows with data fitting the screen.'
        max_rows: Optional[int]
        if self._is_in_terminal():
            (_, height) = get_terminal_size()
            if (self.max_rows == 0):
                return (height - self._get_number_of_auxillary_rows())
            if self._is_screen_short(height):
                max_rows = height
            else:
                max_rows = self.max_rows
        else:
            max_rows = self.max_rows
        return self._adjust_max_rows(max_rows)

    def _adjust_max_rows(self, max_rows):
        'Adjust max_rows using display logic.\n\n        See description here:\n        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options\n\n        GH #37359\n        '
        if max_rows:
            if ((len(self.frame) > max_rows) and self.min_rows):
                max_rows = min(self.min_rows, max_rows)
        return max_rows

    def _is_in_terminal(self):
        'Check if the output is to be shown in terminal.'
        return bool(((self.max_cols == 0) or (self.max_rows == 0)))

    def _is_screen_narrow(self, max_width):
        return bool(((self.max_cols == 0) and (len(self.frame.columns) > max_width)))

    def _is_screen_short(self, max_height):
        return bool(((self.max_rows == 0) and (len(self.frame) > max_height)))

    def _get_number_of_auxillary_rows(self):
        'Get number of rows occupied by prompt, dots and dimension info.'
        dot_row = 1
        prompt_row = 1
        num_rows = (dot_row + prompt_row)
        if self.show_dimensions:
            num_rows += len(self.dimensions_info.splitlines())
        if self.header:
            num_rows += 1
        return num_rows

    def truncate(self):
        '\n        Check whether the frame should be truncated. If so, slice the frame up.\n        '
        if self.is_truncated_horizontally:
            self._truncate_horizontally()
        if self.is_truncated_vertically:
            self._truncate_vertically()

    def _truncate_horizontally(self):
        'Remove columns, which are not to be displayed and adjust formatters.\n\n        Attributes affected:\n            - tr_frame\n            - formatters\n            - tr_col_num\n        '
        assert (self.max_cols_fitted is not None)
        col_num = (self.max_cols_fitted // 2)
        if (col_num >= 1):
            left = self.tr_frame.iloc[:, :col_num]
            right = self.tr_frame.iloc[:, (- col_num):]
            self.tr_frame = concat((left, right), axis=1)
            if isinstance(self.formatters, (list, tuple)):
                self.formatters = [*self.formatters[:col_num], *self.formatters[(- col_num):]]
        else:
            col_num = cast(int, self.max_cols)
            self.tr_frame = self.tr_frame.iloc[:, :col_num]
        self.tr_col_num = col_num

    def _truncate_vertically(self):
        'Remove rows, which are not to be displayed.\n\n        Attributes affected:\n            - tr_frame\n            - tr_row_num\n        '
        assert (self.max_rows_fitted is not None)
        row_num = (self.max_rows_fitted // 2)
        if (row_num >= 1):
            head = self.tr_frame.iloc[:row_num, :]
            tail = self.tr_frame.iloc[(- row_num):, :]
            self.tr_frame = concat((head, tail))
        else:
            row_num = cast(int, self.max_rows)
            self.tr_frame = self.tr_frame.iloc[:row_num, :]
        self.tr_row_num = row_num

    def _get_strcols_without_index(self):
        strcols: List[List[str]] = []
        if ((not is_list_like(self.header)) and (not self.header)):
            for (i, c) in enumerate(self.tr_frame):
                fmt_values = self.format_col(i)
                fmt_values = _make_fixed_width(strings=fmt_values, justify=self.justify, minimum=int(self.col_space.get(c, 0)), adj=self.adj)
                strcols.append(fmt_values)
            return strcols
        if is_list_like(self.header):
            self.header = cast(List[str], self.header)
            if (len(self.header) != len(self.columns)):
                raise ValueError(f'Writing {len(self.columns)} cols but got {len(self.header)} aliases')
            str_columns = [[label] for label in self.header]
        else:
            str_columns = self._get_formatted_column_labels(self.tr_frame)
        if self.show_row_idx_names:
            for x in str_columns:
                x.append('')
        for (i, c) in enumerate(self.tr_frame):
            cheader = str_columns[i]
            header_colwidth = max(int(self.col_space.get(c, 0)), *(self.adj.len(x) for x in cheader))
            fmt_values = self.format_col(i)
            fmt_values = _make_fixed_width(fmt_values, self.justify, minimum=header_colwidth, adj=self.adj)
            max_len = max(max((self.adj.len(x) for x in fmt_values)), header_colwidth)
            cheader = self.adj.justify(cheader, max_len, mode=self.justify)
            strcols.append((cheader + fmt_values))
        return strcols

    def format_col(self, i):
        frame = self.tr_frame
        formatter = self._get_formatter(i)
        return format_array(frame.iloc[:, i]._values, formatter, float_format=self.float_format, na_rep=self.na_rep, space=self.col_space.get(frame.columns[i]), decimal=self.decimal, leading_space=self.index)

    def _get_formatter(self, i):
        if isinstance(self.formatters, (list, tuple)):
            if is_integer(i):
                i = cast(int, i)
                return self.formatters[i]
            else:
                return None
        else:
            if (is_integer(i) and (i not in self.columns)):
                i = self.columns[i]
            return self.formatters.get(i, None)

    def _get_formatted_column_labels(self, frame):
        from pandas.core.indexes.multi import sparsify_labels
        columns = frame.columns
        if isinstance(columns, MultiIndex):
            fmt_columns = columns.format(sparsify=False, adjoin=False)
            fmt_columns = list(zip(*fmt_columns))
            dtypes = self.frame.dtypes._values
            restrict_formatting = any((level.is_floating for level in columns.levels))
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))

            def space_format(x, y):
                if ((y not in self.formatters) and need_leadsp[x] and (not restrict_formatting)):
                    return (' ' + y)
                return y
            str_columns = list(zip(*[[space_format(x, y) for y in x] for x in fmt_columns]))
            if (self.sparsify and len(str_columns)):
                str_columns = sparsify_labels(str_columns)
            str_columns = [list(x) for x in zip(*str_columns)]
        else:
            fmt_columns = columns.format()
            dtypes = self.frame.dtypes
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))
            str_columns = [[((' ' + x) if ((not self._get_formatter(i)) and need_leadsp[x]) else x)] for (i, (col, x)) in enumerate(zip(columns, fmt_columns))]
        return str_columns

    def _get_formatted_index(self, frame):
        col_space = {k: cast(int, v) for (k, v) in self.col_space.items()}
        index = frame.index
        columns = frame.columns
        fmt = self._get_formatter('__index__')
        if isinstance(index, MultiIndex):
            fmt_index = index.format(sparsify=self.sparsify, adjoin=False, names=self.show_row_idx_names, formatter=fmt)
        else:
            fmt_index = [index.format(name=self.show_row_idx_names, formatter=fmt)]
        fmt_index = [tuple(_make_fixed_width(list(x), justify='left', minimum=col_space.get('', 0), adj=self.adj)) for x in fmt_index]
        adjoined = self.adj.adjoin(1, *fmt_index).split('\n')
        if self.show_col_idx_names:
            col_header = [str(x) for x in self._get_column_name_list()]
        else:
            col_header = ([''] * columns.nlevels)
        if self.header:
            return (col_header + adjoined)
        else:
            return adjoined

    def _get_column_name_list(self):
        names: List[str] = []
        columns = self.frame.columns
        if isinstance(columns, MultiIndex):
            names.extend((('' if (name is None) else name) for name in columns.names))
        else:
            names.append(('' if (columns.name is None) else columns.name))
        return names

class DataFrameRenderer():
    'Class for creating dataframe output in multiple formats.\n\n    Called in pandas.core.generic.NDFrame:\n        - to_csv\n        - to_latex\n\n    Called in pandas.core.frame.DataFrame:\n        - to_html\n        - to_string\n\n    Parameters\n    ----------\n    fmt : DataFrameFormatter\n        Formatter with the formating options.\n    '

    def __init__(self, fmt):
        self.fmt = fmt

    def to_latex(self, buf=None, column_format=None, longtable=False, encoding=None, multicolumn=False, multicolumn_format=None, multirow=False, caption=None, label=None, position=None):
        '\n        Render a DataFrame to a LaTeX tabular/longtable environment output.\n        '
        from pandas.io.formats.latex import LatexFormatter
        latex_formatter = LatexFormatter(self.fmt, longtable=longtable, column_format=column_format, multicolumn=multicolumn, multicolumn_format=multicolumn_format, multirow=multirow, caption=caption, label=label, position=position)
        string = latex_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_html(self, buf=None, encoding=None, classes=None, notebook=False, border=None, table_id=None, render_links=False):
        '\n        Render a DataFrame to a html table.\n\n        Parameters\n        ----------\n        buf : str, Path or StringIO-like, optional, default None\n            Buffer to write to. If None, the output is returned as a string.\n        encoding : str, default “utf-8”\n            Set character encoding.\n        classes : str or list-like\n            classes to include in the `class` attribute of the opening\n            ``<table>`` tag, in addition to the default "dataframe".\n        notebook : {True, False}, optional, default False\n            Whether the generated HTML is for IPython Notebook.\n        border : int\n            A ``border=border`` attribute is included in the opening\n            ``<table>`` tag. Default ``pd.options.display.html.border``.\n        table_id : str, optional\n            A css id is included in the opening `<table>` tag if specified.\n        render_links : bool, default False\n            Convert URLs to HTML links.\n        '
        from pandas.io.formats.html import HTMLFormatter, NotebookFormatter
        Klass = (NotebookFormatter if notebook else HTMLFormatter)
        html_formatter = Klass(self.fmt, classes=classes, border=border, table_id=table_id, render_links=render_links)
        string = html_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_string(self, buf=None, encoding=None, line_width=None):
        '\n        Render a DataFrame to a console-friendly tabular output.\n\n        Parameters\n        ----------\n        buf : str, Path or StringIO-like, optional, default None\n            Buffer to write to. If None, the output is returned as a string.\n        encoding: str, default “utf-8”\n            Set character encoding.\n        line_width : int, optional\n            Width to wrap a line in characters.\n        '
        from pandas.io.formats.string import StringFormatter
        string_formatter = StringFormatter(self.fmt, line_width=line_width)
        string = string_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_csv(self, path_or_buf=None, encoding=None, sep=',', columns=None, index_label=None, mode='w', compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, errors='strict', storage_options=None):
        '\n        Render dataframe as comma-separated file.\n        '
        from pandas.io.formats.csvs import CSVFormatter
        if (path_or_buf is None):
            created_buffer = True
            path_or_buf = StringIO()
        else:
            created_buffer = False
        csv_formatter = CSVFormatter(path_or_buf=path_or_buf, line_terminator=line_terminator, sep=sep, encoding=encoding, errors=errors, compression=compression, quoting=quoting, cols=columns, index_label=index_label, mode=mode, chunksize=chunksize, quotechar=quotechar, date_format=date_format, doublequote=doublequote, escapechar=escapechar, storage_options=storage_options, formatter=self.fmt)
        csv_formatter.save()
        if created_buffer:
            assert isinstance(path_or_buf, StringIO)
            content = path_or_buf.getvalue()
            path_or_buf.close()
            return content
        return None

def save_to_buffer(string, buf=None, encoding=None):
    '\n    Perform serialization. Write to buf or return as string if buf is None.\n    '
    with get_buffer(buf, encoding=encoding) as f:
        f.write(string)
        if (buf is None):
            return f.getvalue()
        return None

@contextmanager
def get_buffer(buf, encoding=None):
    '\n    Context manager to open, yield and close buffer for filenames or Path-like\n    objects, otherwise yield buf unchanged.\n    '
    if (buf is not None):
        buf = stringify_path(buf)
    else:
        buf = StringIO()
    if (encoding is None):
        encoding = 'utf-8'
    elif (not isinstance(buf, str)):
        raise ValueError('buf is not a file name and encoding is specified.')
    if hasattr(buf, 'write'):
        (yield buf)
    elif isinstance(buf, str):
        with open(buf, 'w', encoding=encoding, newline='') as f:
            (yield f)
    else:
        raise TypeError('buf is not a file name and it has no write method')

def format_array(values, formatter, float_format=None, na_rep='NaN', digits=None, space=None, justify='right', decimal='.', leading_space=True, quoting=None):
    "\n    Format an array for printing.\n\n    Parameters\n    ----------\n    values\n    formatter\n    float_format\n    na_rep\n    digits\n    space\n    justify\n    decimal\n    leading_space : bool, optional, default True\n        Whether the array should be formatted with a leading space.\n        When an array as a column of a Series or DataFrame, we do want\n        the leading space to pad between columns.\n\n        When formatting an Index subclass\n        (e.g. IntervalIndex._format_native_types), we don't want the\n        leading space since it should be left-aligned.\n\n    Returns\n    -------\n    List[str]\n    "
    fmt_klass: Type[GenericArrayFormatter]
    if is_datetime64_dtype(values.dtype):
        fmt_klass = Datetime64Formatter
    elif is_datetime64tz_dtype(values.dtype):
        fmt_klass = Datetime64TZFormatter
    elif is_timedelta64_dtype(values.dtype):
        fmt_klass = Timedelta64Formatter
    elif is_extension_array_dtype(values.dtype):
        fmt_klass = ExtensionArrayFormatter
    elif (is_float_dtype(values.dtype) or is_complex_dtype(values.dtype)):
        fmt_klass = FloatArrayFormatter
    elif is_integer_dtype(values.dtype):
        fmt_klass = IntArrayFormatter
    else:
        fmt_klass = GenericArrayFormatter
    if (space is None):
        space = get_option('display.column_space')
    if (float_format is None):
        float_format = get_option('display.float_format')
    if (digits is None):
        digits = get_option('display.precision')
    fmt_obj = fmt_klass(values, digits=digits, na_rep=na_rep, float_format=float_format, formatter=formatter, space=space, justify=justify, decimal=decimal, leading_space=leading_space, quoting=quoting)
    return fmt_obj.get_result()

class GenericArrayFormatter():

    def __init__(self, values, digits=7, formatter=None, na_rep='NaN', space=12, float_format=None, justify='right', decimal='.', quoting=None, fixed_width=True, leading_space=True):
        self.values = values
        self.digits = digits
        self.na_rep = na_rep
        self.space = space
        self.formatter = formatter
        self.float_format = float_format
        self.justify = justify
        self.decimal = decimal
        self.quoting = quoting
        self.fixed_width = fixed_width
        self.leading_space = leading_space

    def get_result(self):
        fmt_values = self._format_strings()
        return _make_fixed_width(fmt_values, self.justify)

    def _format_strings(self):
        if (self.float_format is None):
            float_format = get_option('display.float_format')
            if (float_format is None):
                precision = get_option('display.precision')
                float_format = (lambda x: f'{x: .{precision:d}f}')
        else:
            float_format = self.float_format
        if (self.formatter is not None):
            formatter = self.formatter
        else:
            quote_strings = ((self.quoting is not None) and (self.quoting != QUOTE_NONE))
            formatter = partial(pprint_thing, escape_chars=('\t', '\r', '\n'), quote_strings=quote_strings)

        def _format(x):
            if ((self.na_rep is not None) and is_scalar(x) and isna(x)):
                try:
                    if (x is None):
                        return 'None'
                    elif (x is NA):
                        return str(NA)
                    elif ((x is NaT) or np.isnat(x)):
                        return 'NaT'
                except (TypeError, ValueError):
                    pass
                return self.na_rep
            elif isinstance(x, PandasObject):
                return str(x)
            else:
                return str(formatter(x))
        vals = extract_array(self.values, extract_numpy=True)
        is_float_type = (lib.map_infer(vals, is_float) & np.all(notna(vals), axis=tuple(range(1, len(vals.shape)))))
        leading_space = self.leading_space
        if (leading_space is None):
            leading_space = is_float_type.any()
        fmt_values = []
        for (i, v) in enumerate(vals):
            if ((not is_float_type[i]) and leading_space):
                fmt_values.append(f' {_format(v)}')
            elif is_float_type[i]:
                fmt_values.append(_trim_zeros_single_float(float_format(v)))
            else:
                if (leading_space is False):
                    tpl = '{v}'
                else:
                    tpl = ' {v}'
                fmt_values.append(tpl.format(v=_format(v)))
        return fmt_values

class FloatArrayFormatter(GenericArrayFormatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if ((self.float_format is not None) and (self.formatter is None)):
            self.fixed_width = False
            if callable(self.float_format):
                self.formatter = self.float_format
                self.float_format = None

    def _value_formatter(self, float_format=None, threshold=None):
        'Returns a function to be applied on each value to format it'
        if (float_format is None):
            float_format = self.float_format
        if float_format:

            def base_formatter(v):
                assert (float_format is not None)
                return (float_format(value=v) if notna(v) else self.na_rep)
        else:

            def base_formatter(v):
                return (str(v) if notna(v) else self.na_rep)
        if (self.decimal != '.'):

            def decimal_formatter(v):
                return base_formatter(v).replace('.', self.decimal, 1)
        else:
            decimal_formatter = base_formatter
        if (threshold is None):
            return decimal_formatter

        def formatter(value):
            if notna(value):
                if (abs(value) > threshold):
                    return decimal_formatter(value)
                else:
                    return decimal_formatter(0.0)
            else:
                return self.na_rep
        return formatter

    def get_result_as_array(self):
        '\n        Returns the float values converted into strings using\n        the parameters given at initialisation, as a numpy array\n        '

        def format_with_na_rep(values: ArrayLike, formatter: Callable, na_rep: str):
            mask = isna(values)
            formatted = np.array([(formatter(val) if (not m) else na_rep) for (val, m) in zip(values.ravel(), mask.ravel())]).reshape(values.shape)
            return formatted
        if (self.formatter is not None):
            return format_with_na_rep(self.values, self.formatter, self.na_rep)
        if self.fixed_width:
            threshold = get_option('display.chop_threshold')
        else:
            threshold = None

        def format_values_with(float_format):
            formatter = self._value_formatter(float_format, threshold)
            if (self.justify == 'left'):
                na_rep = (' ' + self.na_rep)
            else:
                na_rep = self.na_rep
            values = self.values
            is_complex = is_complex_dtype(values)
            values = format_with_na_rep(values, formatter, na_rep)
            if self.fixed_width:
                if is_complex:
                    result = _trim_zeros_complex(values, self.decimal)
                else:
                    result = _trim_zeros_float(values, self.decimal)
                return np.asarray(result, dtype='object')
            return values
        float_format: Optional[FloatFormatType]
        if (self.float_format is None):
            if self.fixed_width:
                if (self.leading_space is True):
                    fmt_str = '{value: .{digits:d}f}'
                else:
                    fmt_str = '{value:.{digits:d}f}'
                float_format = partial(fmt_str.format, digits=self.digits)
            else:
                float_format = self.float_format
        else:
            float_format = (lambda value: (self.float_format % value))
        formatted_values = format_values_with(float_format)
        if (not self.fixed_width):
            return formatted_values
        if (len(formatted_values) > 0):
            maxlen = max((len(x) for x in formatted_values))
            too_long = (maxlen > (self.digits + 6))
        else:
            too_long = False
        with np.errstate(invalid='ignore'):
            abs_vals = np.abs(self.values)
            has_large_values = (abs_vals > 1000000.0).any()
            has_small_values = ((abs_vals < (10 ** (- self.digits))) & (abs_vals > 0)).any()
        if (has_small_values or (too_long and has_large_values)):
            if (self.leading_space is True):
                fmt_str = '{value: .{digits:d}e}'
            else:
                fmt_str = '{value:.{digits:d}e}'
            float_format = partial(fmt_str.format, digits=self.digits)
            formatted_values = format_values_with(float_format)
        return formatted_values

    def _format_strings(self):
        return list(self.get_result_as_array())

class IntArrayFormatter(GenericArrayFormatter):

    def _format_strings(self):
        if (self.leading_space is False):
            formatter_str = (lambda x: f'{x:d}'.format(x=x))
        else:
            formatter_str = (lambda x: f'{x: d}'.format(x=x))
        formatter = (self.formatter or formatter_str)
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values

class Datetime64Formatter(GenericArrayFormatter):

    def __init__(self, values, nat_rep='NaT', date_format=None, **kwargs):
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep
        self.date_format = date_format

    def _format_strings(self):
        ' we by definition have DO NOT have a TZ '
        values = self.values
        if (not isinstance(values, DatetimeIndex)):
            values = DatetimeIndex(values)
        if ((self.formatter is not None) and callable(self.formatter)):
            return [self.formatter(x) for x in values]
        fmt_values = values._data._format_native_types(na_rep=self.nat_rep, date_format=self.date_format)
        return fmt_values.tolist()

class ExtensionArrayFormatter(GenericArrayFormatter):

    def _format_strings(self):
        values = extract_array(self.values, extract_numpy=True)
        formatter = self.formatter
        if (formatter is None):
            formatter = values._formatter(boxed=True)
        if is_categorical_dtype(values.dtype):
            array = values._internal_get_values()
        else:
            array = np.asarray(values)
        fmt_values = format_array(array, formatter, float_format=self.float_format, na_rep=self.na_rep, digits=self.digits, space=self.space, justify=self.justify, decimal=self.decimal, leading_space=self.leading_space, quoting=self.quoting)
        return fmt_values

def format_percentiles(percentiles):
    "\n    Outputs rounded and formatted percentiles.\n\n    Parameters\n    ----------\n    percentiles : list-like, containing floats from interval [0,1]\n\n    Returns\n    -------\n    formatted : list of strings\n\n    Notes\n    -----\n    Rounding precision is chosen so that: (1) if any two elements of\n    ``percentiles`` differ, they remain different after rounding\n    (2) no entry is *rounded* to 0% or 100%.\n    Any non-integer is always rounded to at least 1 decimal place.\n\n    Examples\n    --------\n    Keeps all entries different after rounding:\n\n    >>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])\n    ['1.999%', '2.001%', '50%', '66.667%', '99.99%']\n\n    No element is rounded to 0% or 100% (unless already equal to it).\n    Duplicates are allowed:\n\n    >>> format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])\n    ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']\n    "
    percentiles = np.asarray(percentiles)
    with np.errstate(invalid='ignore'):
        if ((not is_numeric_dtype(percentiles)) or (not np.all((percentiles >= 0))) or (not np.all((percentiles <= 1)))):
            raise ValueError('percentiles should all be in the interval [0,1]')
    percentiles = (100 * percentiles)
    int_idx = np.isclose(percentiles.astype(int), percentiles)
    if np.all(int_idx):
        out = percentiles.astype(int).astype(str)
        return [(i + '%') for i in out]
    unique_pcts = np.unique(percentiles)
    to_begin = (unique_pcts[0] if (unique_pcts[0] > 0) else None)
    to_end = ((100 - unique_pcts[(- 1)]) if (unique_pcts[(- 1)] < 100) else None)
    prec = (- np.floor(np.log10(np.min(np.ediff1d(unique_pcts, to_begin=to_begin, to_end=to_end)))).astype(int))
    prec = max(1, prec)
    out = np.empty_like(percentiles, dtype=object)
    out[int_idx] = percentiles[int_idx].astype(int).astype(str)
    out[(~ int_idx)] = percentiles[(~ int_idx)].round(prec).astype(str)
    return [(i + '%') for i in out]

def is_dates_only(values):
    if (not isinstance(values, Index)):
        values = values.ravel()
    values = DatetimeIndex(values)
    if (values.tz is not None):
        return False
    values_int = values.asi8
    consider_values = (values_int != iNaT)
    one_day_nanos = (86400 * 1000000000.0)
    even_days = (np.logical_and(consider_values, ((values_int % int(one_day_nanos)) != 0)).sum() == 0)
    if even_days:
        return True
    return False

def _format_datetime64(x, nat_rep='NaT'):
    if (x is NaT):
        return nat_rep
    return str(x)

def _format_datetime64_dateonly(x, nat_rep='NaT', date_format=None):
    if (x is NaT):
        return nat_rep
    if date_format:
        return x.strftime(date_format)
    else:
        return x._date_repr

def get_format_datetime64(is_dates_only, nat_rep='NaT', date_format=None):
    if is_dates_only:
        return (lambda x: _format_datetime64_dateonly(x, nat_rep=nat_rep, date_format=date_format))
    else:
        return (lambda x: _format_datetime64(x, nat_rep=nat_rep))

def get_format_datetime64_from_values(values, date_format):
    ' given values and a date_format, return a string format '
    if (isinstance(values, np.ndarray) and (values.ndim > 1)):
        values = values.ravel()
    ido = is_dates_only(values)
    if ido:
        return (date_format or '%Y-%m-%d')
    return date_format

class Datetime64TZFormatter(Datetime64Formatter):

    def _format_strings(self):
        ' we by definition have a TZ '
        values = self.values.astype(object)
        ido = is_dates_only(values)
        formatter = (self.formatter or get_format_datetime64(ido, date_format=self.date_format))
        fmt_values = [formatter(x) for x in values]
        return fmt_values

class Timedelta64Formatter(GenericArrayFormatter):

    def __init__(self, values, nat_rep='NaT', box=False, **kwargs):
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep
        self.box = box

    def _format_strings(self):
        formatter = (self.formatter or get_format_timedelta64(self.values, nat_rep=self.nat_rep, box=self.box))
        return [formatter(x) for x in self.values]

def get_format_timedelta64(values, nat_rep='NaT', box=False):
    '\n    Return a formatter function for a range of timedeltas.\n    These will all have the same format argument\n\n    If box, then show the return in quotes\n    '
    values_int = values.view(np.int64)
    consider_values = (values_int != iNaT)
    one_day_nanos = (86400 * 1000000000.0)
    even_days = (np.logical_and(consider_values, ((values_int % one_day_nanos) != 0)).sum() == 0)
    if even_days:
        format = None
    else:
        format = 'long'

    def _formatter(x):
        if ((x is None) or (is_scalar(x) and isna(x))):
            return nat_rep
        if (not isinstance(x, Timedelta)):
            x = Timedelta(x)
        result = x._repr_base(format=format)
        if box:
            result = f"'{result}'"
        return result
    return _formatter

def _make_fixed_width(strings, justify='right', minimum=None, adj=None):
    if ((len(strings) == 0) or (justify == 'all')):
        return strings
    if (adj is None):
        adjustment = get_adjustment()
    else:
        adjustment = adj
    max_len = max((adjustment.len(x) for x in strings))
    if (minimum is not None):
        max_len = max(minimum, max_len)
    conf_max = get_option('display.max_colwidth')
    if ((conf_max is not None) and (max_len > conf_max)):
        max_len = conf_max

    def just(x: str) -> str:
        if (conf_max is not None):
            if ((conf_max > 3) & (adjustment.len(x) > max_len)):
                x = (x[:(max_len - 3)] + '...')
        return x
    strings = [just(x) for x in strings]
    result = adjustment.justify(strings, max_len, mode=justify)
    return result

def _trim_zeros_complex(str_complexes, decimal='.'):
    '\n    Separates the real and imaginary parts from the complex number, and\n    executes the _trim_zeros_float method on each of those.\n    '
    trimmed = [''.join(_trim_zeros_float(re.split('([j+-])', x), decimal)) for x in str_complexes]
    lengths = [len(s) for s in trimmed]
    max_length = max(lengths)
    padded = [(((((s[:(- (((k - 1) // 2) + 1))] + (((max_length - k) // 2) * '0')) + s[(- (((k - 1) // 2) + 1)):(- ((k - 1) // 2))]) + s[(- ((k - 1) // 2)):(- 1)]) + (((max_length - k) // 2) * '0')) + s[(- 1)]) for (s, k) in zip(trimmed, lengths)]
    return padded

def _trim_zeros_single_float(str_float):
    '\n    Trims trailing zeros after a decimal point,\n    leaving just one if necessary.\n    '
    str_float = str_float.rstrip('0')
    if str_float.endswith('.'):
        str_float += '0'
    return str_float

def _trim_zeros_float(str_floats, decimal='.'):
    '\n    Trims the maximum number of trailing zeros equally from\n    all numbers containing decimals, leaving just one if\n    necessary.\n    '
    trimmed = str_floats
    number_regex = re.compile(f'^\s*[\+-]?[0-9]+\{decimal}[0-9]*$')

    def is_number_with_decimal(x):
        return (re.match(number_regex, x) is not None)

    def should_trim(values: Union[(np.ndarray, List[str])]) -> bool:
        '\n        Determine if an array of strings should be trimmed.\n\n        Returns True if all numbers containing decimals (defined by the\n        above regular expression) within the array end in a zero, otherwise\n        returns False.\n        '
        numbers = [x for x in values if is_number_with_decimal(x)]
        return ((len(numbers) > 0) and all((x.endswith('0') for x in numbers)))
    while should_trim(trimmed):
        trimmed = [(x[:(- 1)] if is_number_with_decimal(x) else x) for x in trimmed]
    result = [((x + '0') if (is_number_with_decimal(x) and x.endswith(decimal)) else x) for x in trimmed]
    return result

def _has_names(index):
    if isinstance(index, MultiIndex):
        return com.any_not_none(*index.names)
    else:
        return (index.name is not None)

class EngFormatter():
    '\n    Formats float values according to engineering format.\n\n    Based on matplotlib.ticker.EngFormatter\n    '
    ENG_PREFIXES = {(- 24): 'y', (- 21): 'z', (- 18): 'a', (- 15): 'f', (- 12): 'p', (- 9): 'n', (- 6): 'u', (- 3): 'm', 0: '', 3: 'k', 6: 'M', 9: 'G', 12: 'T', 15: 'P', 18: 'E', 21: 'Z', 24: 'Y'}

    def __init__(self, accuracy=None, use_eng_prefix=False):
        self.accuracy = accuracy
        self.use_eng_prefix = use_eng_prefix

    def __call__(self, num):
        '\n        Formats a number in engineering notation, appending a letter\n        representing the power of 1000 of the original number. Some examples:\n\n        >>> format_eng(0)       # for self.accuracy = 0\n        \' 0\'\n\n        >>> format_eng(1000000) # for self.accuracy = 1,\n                                #     self.use_eng_prefix = True\n        \' 1.0M\'\n\n        >>> format_eng("-1e-6") # for self.accuracy = 2\n                                #     self.use_eng_prefix = False\n        \'-1.00E-06\'\n\n        @param num: the value to represent\n        @type num: either a numeric value or a string that can be converted to\n                   a numeric value (as per decimal.Decimal constructor)\n\n        @return: engineering formatted string\n        '
        dnum = decimal.Decimal(str(num))
        if decimal.Decimal.is_nan(dnum):
            return 'NaN'
        if decimal.Decimal.is_infinite(dnum):
            return 'inf'
        sign = 1
        if (dnum < 0):
            sign = (- 1)
            dnum = (- dnum)
        if (dnum != 0):
            pow10 = decimal.Decimal(int((math.floor((dnum.log10() / 3)) * 3)))
        else:
            pow10 = decimal.Decimal(0)
        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
        int_pow10 = int(pow10)
        if self.use_eng_prefix:
            prefix = self.ENG_PREFIXES[int_pow10]
        elif (int_pow10 < 0):
            prefix = f'E-{(- int_pow10):02d}'
        else:
            prefix = f'E+{int_pow10:02d}'
        mant = ((sign * dnum) / (10 ** pow10))
        if (self.accuracy is None):
            format_str = '{mant: g}{prefix}'
        else:
            format_str = f'{{mant: .{self.accuracy:d}f}}{{prefix}}'
        formatted = format_str.format(mant=mant, prefix=prefix)
        return formatted

def set_eng_float_format(accuracy=3, use_eng_prefix=False):
    '\n    Alter default behavior on how float is formatted in DataFrame.\n    Format float in engineering format. By accuracy, we mean the number of\n    decimal digits after the floating point.\n\n    See also EngFormatter.\n    '
    set_option('display.float_format', EngFormatter(accuracy, use_eng_prefix))
    set_option('display.column_space', max(12, (accuracy + 9)))

def get_level_lengths(levels, sentinel=''):
    '\n    For each index in each level the function returns lengths of indexes.\n\n    Parameters\n    ----------\n    levels : list of lists\n        List of values on for level.\n    sentinel : string, optional\n        Value which states that no new index starts on there.\n\n    Returns\n    -------\n    Returns list of maps. For each level returns map of indexes (key is index\n    in row and value is length of index).\n    '
    if (len(levels) == 0):
        return []
    control = ([True] * len(levels[0]))
    result = []
    for level in levels:
        last_index = 0
        lengths = {}
        for (i, key) in enumerate(level):
            if (control[i] and (key == sentinel)):
                pass
            else:
                control[i] = False
                lengths[last_index] = (i - last_index)
                last_index = i
        lengths[last_index] = (len(level) - last_index)
        result.append(lengths)
    return result

def buffer_put_lines(buf, lines):
    '\n    Appends lines to a buffer.\n\n    Parameters\n    ----------\n    buf\n        The buffer to write to\n    lines\n        The lines to append.\n    '
    if any((isinstance(x, str) for x in lines)):
        lines = [str(x) for x in lines]
    buf.write('\n'.join(lines))
