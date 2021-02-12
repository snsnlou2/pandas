
from abc import ABC, abstractmethod
import sys
from typing import IO, TYPE_CHECKING, Iterable, Iterator, List, Mapping, Optional, Sequence, Union
from pandas._config import get_option
from pandas._typing import Dtype, FrameOrSeriesUnion
from pandas.core.indexes.api import Index
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

def _put_str(s, space):
    '\n    Make string of specified length, padding to the right if necessary.\n\n    Parameters\n    ----------\n    s : Union[str, Dtype]\n        String to be formatted.\n    space : int\n        Length to force string to be of.\n\n    Returns\n    -------\n    str\n        String coerced to given length.\n\n    Examples\n    --------\n    >>> pd.io.formats.info._put_str("panda", 6)\n    \'panda \'\n    >>> pd.io.formats.info._put_str("panda", 4)\n    \'pand\'\n    '
    return str(s)[:space].ljust(space)

def _sizeof_fmt(num, size_qualifier):
    "\n    Return size in human readable format.\n\n    Parameters\n    ----------\n    num : int\n        Size in bytes.\n    size_qualifier : str\n        Either empty, or '+' (if lower bound).\n\n    Returns\n    -------\n    str\n        Size in human readable format.\n\n    Examples\n    --------\n    >>> _sizeof_fmt(23028, '')\n    '22.5 KB'\n\n    >>> _sizeof_fmt(23028, '+')\n    '22.5+ KB'\n    "
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if (num < 1024.0):
            return f'{num:3.1f}{size_qualifier} {x}'
        num /= 1024.0
    return f'{num:3.1f}{size_qualifier} PB'

def _initialize_memory_usage(memory_usage=None):
    'Get memory usage based on inputs and display options.'
    if (memory_usage is None):
        memory_usage = get_option('display.memory_usage')
    return memory_usage

class BaseInfo(ABC):
    '\n    Base class for DataFrameInfo and SeriesInfo.\n\n    Parameters\n    ----------\n    data : DataFrame or Series\n        Either dataframe or series.\n    memory_usage : bool or str, optional\n        If "deep", introspect the data deeply by interrogating object dtypes\n        for system-level memory consumption, and include it in the returned\n        values.\n    '

    @property
    @abstractmethod
    def dtypes(self):
        "\n        Dtypes.\n\n        Returns\n        -------\n        dtypes : sequence\n            Dtype of each of the DataFrame's columns (or one series column).\n        "

    @property
    @abstractmethod
    def dtype_counts(self):
        'Mapping dtype - number of counts.'

    @property
    @abstractmethod
    def non_null_counts(self):
        'Sequence of non-null counts for all columns or column (if series).'

    @property
    @abstractmethod
    def memory_usage_bytes(self):
        "\n        Memory usage in bytes.\n\n        Returns\n        -------\n        memory_usage_bytes : int\n            Object's total memory usage in bytes.\n        "

    @property
    def memory_usage_string(self):
        'Memory usage in a form of human readable string.'
        return f'''{_sizeof_fmt(self.memory_usage_bytes, self.size_qualifier)}
'''

    @property
    def size_qualifier(self):
        size_qualifier = ''
        if self.memory_usage:
            if (self.memory_usage != 'deep'):
                if (('object' in self.dtype_counts) or self.data.index._is_memory_usage_qualified()):
                    size_qualifier = '+'
        return size_qualifier

    @abstractmethod
    def render(self, *, buf, max_cols, verbose, show_counts):
        '\n        Print a concise summary of a %(klass)s.\n\n        This method prints information about a %(klass)s including\n        the index dtype%(type_sub)s, non-null values and memory usage.\n        %(version_added_sub)s\n        Parameters\n        ----------\n        data : %(klass)s\n            %(klass)s to print information about.\n        verbose : bool, optional\n            Whether to print the full summary. By default, the setting in\n            ``pandas.options.display.max_info_columns`` is followed.\n        buf : writable buffer, defaults to sys.stdout\n            Where to send the output. By default, the output is printed to\n            sys.stdout. Pass a writable buffer if you need to further process\n            the output.\n        %(max_cols_sub)s\n        memory_usage : bool, str, optional\n            Specifies whether total memory usage of the %(klass)s\n            elements (including the index) should be displayed. By default,\n            this follows the ``pandas.options.display.memory_usage`` setting.\n\n            True always show memory usage. False never shows memory usage.\n            A value of \'deep\' is equivalent to "True with deep introspection".\n            Memory usage is shown in human-readable units (base-2\n            representation). Without deep introspection a memory estimation is\n            made based in column dtype and number of rows assuming values\n            consume the same memory amount for corresponding dtypes. With deep\n            memory introspection, a real memory usage calculation is performed\n            at the cost of computational resources.\n        %(show_counts_sub)s\n\n        Returns\n        -------\n        None\n            This method prints a summary of a %(klass)s and returns None.\n\n        See Also\n        --------\n        %(see_also_sub)s\n\n        Examples\n        --------\n        %(examples_sub)s\n        '

class DataFrameInfo(BaseInfo):
    '\n    Class storing dataframe-specific info.\n    '

    def __init__(self, data, memory_usage=None):
        self.data: 'DataFrame' = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    @property
    def dtype_counts(self):
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self):
        "\n        Dtypes.\n\n        Returns\n        -------\n        dtypes\n            Dtype of each of the DataFrame's columns.\n        "
        return self.data.dtypes

    @property
    def ids(self):
        "\n        Column names.\n\n        Returns\n        -------\n        ids : Index\n            DataFrame's column names.\n        "
        return self.data.columns

    @property
    def col_count(self):
        'Number of columns to be summarized.'
        return len(self.ids)

    @property
    def non_null_counts(self):
        'Sequence of non-null counts for all columns or column (if series).'
        return self.data.count()

    @property
    def memory_usage_bytes(self):
        if (self.memory_usage == 'deep'):
            deep = True
        else:
            deep = False
        return self.data.memory_usage(index=True, deep=deep).sum()

    def render(self, *, buf, max_cols, verbose, show_counts):
        printer = DataFrameInfoPrinter(info=self, max_cols=max_cols, verbose=verbose, show_counts=show_counts)
        printer.to_buffer(buf)

class InfoPrinterAbstract():
    '\n    Class for printing dataframe or series info.\n    '

    def to_buffer(self, buf=None):
        'Save dataframe info into buffer.'
        table_builder = self._create_table_builder()
        lines = table_builder.get_lines()
        if (buf is None):
            buf = sys.stdout
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self):
        'Create instance of table builder.'

class DataFrameInfoPrinter(InfoPrinterAbstract):
    '\n    Class for printing dataframe info.\n\n    Parameters\n    ----------\n    info : DataFrameInfo\n        Instance of DataFrameInfo.\n    max_cols : int, optional\n        When to switch from the verbose to the truncated output.\n    verbose : bool, optional\n        Whether to print the full summary.\n    show_counts : bool, optional\n        Whether to show the non-null counts.\n    '

    def __init__(self, info, max_cols=None, verbose=None, show_counts=None):
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.max_cols = self._initialize_max_cols(max_cols)
        self.show_counts = self._initialize_show_counts(show_counts)

    @property
    def max_rows(self):
        'Maximum info rows to be displayed.'
        return get_option('display.max_info_rows', (len(self.data) + 1))

    @property
    def exceeds_info_cols(self):
        'Check if number of columns to be summarized does not exceed maximum.'
        return bool((self.col_count > self.max_cols))

    @property
    def exceeds_info_rows(self):
        'Check if number of rows to be summarized does not exceed maximum.'
        return bool((len(self.data) > self.max_rows))

    @property
    def col_count(self):
        'Number of columns to be summarized.'
        return self.info.col_count

    def _initialize_max_cols(self, max_cols):
        if (max_cols is None):
            return get_option('display.max_info_columns', (self.col_count + 1))
        return max_cols

    def _initialize_show_counts(self, show_counts):
        if (show_counts is None):
            return bool(((not self.exceeds_info_cols) and (not self.exceeds_info_rows)))
        else:
            return show_counts

    def _create_table_builder(self):
        '\n        Create instance of table builder based on verbosity and display settings.\n        '
        if self.verbose:
            return DataFrameTableBuilderVerbose(info=self.info, with_counts=self.show_counts)
        elif (self.verbose is False):
            return DataFrameTableBuilderNonVerbose(info=self.info)
        elif self.exceeds_info_cols:
            return DataFrameTableBuilderNonVerbose(info=self.info)
        else:
            return DataFrameTableBuilderVerbose(info=self.info, with_counts=self.show_counts)

class TableBuilderAbstract(ABC):
    '\n    Abstract builder for info table.\n    '

    @abstractmethod
    def get_lines(self):
        'Product in a form of list of lines (strings).'

    @property
    def data(self):
        return self.info.data

    @property
    def dtypes(self):
        "Dtypes of each of the DataFrame's columns."
        return self.info.dtypes

    @property
    def dtype_counts(self):
        'Mapping dtype - number of counts.'
        return self.info.dtype_counts

    @property
    def display_memory_usage(self):
        'Whether to display memory usage.'
        return bool(self.info.memory_usage)

    @property
    def memory_usage_string(self):
        'Memory usage string with proper size qualifier.'
        return self.info.memory_usage_string

    @property
    def non_null_counts(self):
        return self.info.non_null_counts

    def add_object_type_line(self):
        'Add line with string representation of dataframe to the table.'
        self._lines.append(str(type(self.data)))

    def add_index_range_line(self):
        'Add line with range of indices to the table.'
        self._lines.append(self.data.index._summary())

    def add_dtypes_line(self):
        'Add summary line with dtypes present in dataframe.'
        collected_dtypes = [f'{key}({val:d})' for (key, val) in sorted(self.dtype_counts.items())]
        self._lines.append(f"dtypes: {', '.join(collected_dtypes)}")

class DataFrameTableBuilder(TableBuilderAbstract):
    '\n    Abstract builder for dataframe info table.\n\n    Parameters\n    ----------\n    info : DataFrameInfo.\n        Instance of DataFrameInfo.\n    '

    def __init__(self, *, info):
        self.info: DataFrameInfo = info

    def get_lines(self):
        self._lines = []
        if (self.col_count == 0):
            self._fill_empty_info()
        else:
            self._fill_non_empty_info()
        return self._lines

    def _fill_empty_info(self):
        'Add lines to the info table, pertaining to empty dataframe.'
        self.add_object_type_line()
        self.add_index_range_line()
        self._lines.append(f'Empty {type(self.data).__name__}')

    @abstractmethod
    def _fill_non_empty_info(self):
        'Add lines to the info table, pertaining to non-empty dataframe.'

    @property
    def data(self):
        'DataFrame.'
        return self.info.data

    @property
    def ids(self):
        'Dataframe columns.'
        return self.info.ids

    @property
    def col_count(self):
        'Number of dataframe columns to be summarized.'
        return self.info.col_count

    def add_memory_usage_line(self):
        'Add line containing memory usage.'
        self._lines.append(f'memory usage: {self.memory_usage_string}')

class DataFrameTableBuilderNonVerbose(DataFrameTableBuilder):
    '\n    Dataframe info table builder for non-verbose output.\n    '

    def _fill_non_empty_info(self):
        'Add lines to the info table, pertaining to non-empty dataframe.'
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_columns_summary_line(self):
        self._lines.append(self.ids._summary(name='Columns'))

class TableBuilderVerboseMixin(TableBuilderAbstract):
    '\n    Mixin for verbose info output.\n    '
    SPACING = (' ' * 2)

    @property
    @abstractmethod
    def headers(self):
        'Headers names of the columns in verbose table.'

    @property
    def header_column_widths(self):
        'Widths of header columns (only titles).'
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self):
        'Get widths of columns containing both headers and actual content.'
        body_column_widths = self._get_body_column_widths()
        return [max(*widths) for widths in zip(self.header_column_widths, body_column_widths)]

    def _get_body_column_widths(self):
        'Get widths of table content columns.'
        strcols: Sequence[Sequence[str]] = list(zip(*self.strrows))
        return [max((len(x) for x in col)) for col in strcols]

    def _gen_rows(self):
        '\n        Generator function yielding rows content.\n\n        Each element represents a row comprising a sequence of strings.\n        '
        if self.with_counts:
            return self._gen_rows_with_counts()
        else:
            return self._gen_rows_without_counts()

    @abstractmethod
    def _gen_rows_with_counts(self):
        'Iterator with string representation of body data with counts.'

    @abstractmethod
    def _gen_rows_without_counts(self):
        'Iterator with string representation of body data without counts.'

    def add_header_line(self):
        header_line = self.SPACING.join([_put_str(header, col_width) for (header, col_width) in zip(self.headers, self.gross_column_widths)])
        self._lines.append(header_line)

    def add_separator_line(self):
        separator_line = self.SPACING.join([_put_str(('-' * header_colwidth), gross_colwidth) for (header_colwidth, gross_colwidth) in zip(self.header_column_widths, self.gross_column_widths)])
        self._lines.append(separator_line)

    def add_body_lines(self):
        for row in self.strrows:
            body_line = self.SPACING.join([_put_str(col, gross_colwidth) for (col, gross_colwidth) in zip(row, self.gross_column_widths)])
            self._lines.append(body_line)

    def _gen_non_null_counts(self):
        'Iterator with string representation of non-null counts.'
        for count in self.non_null_counts:
            (yield f'{count} non-null')

    def _gen_dtypes(self):
        'Iterator with string representation of column dtypes.'
        for dtype in self.dtypes:
            (yield pprint_thing(dtype))

class DataFrameTableBuilderVerbose(DataFrameTableBuilder, TableBuilderVerboseMixin):
    '\n    Dataframe info table builder for verbose output.\n    '

    def __init__(self, *, info, with_counts):
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self):
        'Add lines to the info table, pertaining to non-empty dataframe.'
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    @property
    def headers(self):
        'Headers names of the columns in verbose table.'
        if self.with_counts:
            return [' # ', 'Column', 'Non-Null Count', 'Dtype']
        return [' # ', 'Column', 'Dtype']

    def add_columns_summary_line(self):
        self._lines.append(f'Data columns (total {self.col_count} columns):')

    def _gen_rows_without_counts(self):
        'Iterator with string representation of body data without counts.'
        (yield from zip(self._gen_line_numbers(), self._gen_columns(), self._gen_dtypes()))

    def _gen_rows_with_counts(self):
        'Iterator with string representation of body data with counts.'
        (yield from zip(self._gen_line_numbers(), self._gen_columns(), self._gen_non_null_counts(), self._gen_dtypes()))

    def _gen_line_numbers(self):
        'Iterator with string representation of column numbers.'
        for (i, _) in enumerate(self.ids):
            (yield f' {i}')

    def _gen_columns(self):
        'Iterator with string representation of column names.'
        for col in self.ids:
            (yield pprint_thing(col))

def _get_dataframe_dtype_counts(df):
    '\n    Create mapping between datatypes and their number of occurences.\n    '
    return df.dtypes.value_counts().groupby((lambda x: x.name)).sum()
