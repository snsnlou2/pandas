
'\nModule for formatting output data in Latex.\n'
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.io.formats.format import DataFrameFormatter

def _split_into_full_short_caption(caption):
    'Extract full and short captions from caption string/tuple.\n\n    Parameters\n    ----------\n    caption : str or tuple, optional\n        Either table caption string or tuple (full_caption, short_caption).\n        If string is provided, then it is treated as table full caption,\n        while short_caption is considered an empty string.\n\n    Returns\n    -------\n    full_caption, short_caption : tuple\n        Tuple of full_caption, short_caption strings.\n    '
    if caption:
        if isinstance(caption, str):
            full_caption = caption
            short_caption = ''
        else:
            try:
                (full_caption, short_caption) = caption
            except ValueError as err:
                msg = 'caption must be either a string or a tuple of two strings'
                raise ValueError(msg) from err
    else:
        full_caption = ''
        short_caption = ''
    return (full_caption, short_caption)

class RowStringConverter(ABC):
    'Converter for dataframe rows into LaTeX strings.\n\n    Parameters\n    ----------\n    formatter : `DataFrameFormatter`\n        Instance of `DataFrameFormatter`.\n    multicolumn: bool, optional\n        Whether to use \\multicolumn macro.\n    multicolumn_format: str, optional\n        Multicolumn format.\n    multirow: bool, optional\n        Whether to use \\multirow macro.\n\n    '

    def __init__(self, formatter, multicolumn=False, multicolumn_format=None, multirow=False):
        self.fmt = formatter
        self.frame = self.fmt.frame
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.clinebuf: List[List[int]] = []
        self.strcols = self._get_strcols()
        self.strrows = list(zip(*self.strcols))

    def get_strrow(self, row_num):
        'Get string representation of the row.'
        row = self.strrows[row_num]
        is_multicol = ((row_num < self.column_levels) and self.fmt.header and self.multicolumn)
        is_multirow = ((row_num >= self.header_levels) and self.fmt.index and self.multirow and (self.index_levels > 1))
        is_cline_maybe_required = (is_multirow and (row_num < (len(self.strrows) - 1)))
        crow = self._preprocess_row(row)
        if is_multicol:
            crow = self._format_multicolumn(crow)
        if is_multirow:
            crow = self._format_multirow(crow, row_num)
        lst = []
        lst.append(' & '.join(crow))
        lst.append(' \\\\')
        if is_cline_maybe_required:
            cline = self._compose_cline(row_num, len(self.strcols))
            lst.append(cline)
        return ''.join(lst)

    @property
    def _header_row_num(self):
        'Number of rows in header.'
        return (self.header_levels if self.fmt.header else 0)

    @property
    def index_levels(self):
        'Integer number of levels in index.'
        return self.frame.index.nlevels

    @property
    def column_levels(self):
        return self.frame.columns.nlevels

    @property
    def header_levels(self):
        nlevels = self.column_levels
        if (self.fmt.has_index_names and self.fmt.show_index_names):
            nlevels += 1
        return nlevels

    def _get_strcols(self):
        'String representation of the columns.'
        if self.fmt.frame.empty:
            strcols = [[self._empty_info_line]]
        else:
            strcols = self.fmt.get_strcols()
        if (self.fmt.index and isinstance(self.frame.index, ABCMultiIndex)):
            out = self.frame.index.format(adjoin=False, sparsify=self.fmt.sparsify, names=self.fmt.has_index_names, na_rep=self.fmt.na_rep)

            def pad_empties(x):
                for pad in reversed(x):
                    if pad:
                        break
                return ([x[0]] + [(i if i else (' ' * len(pad))) for i in x[1:]])
            gen = (pad_empties(i) for i in out)
            clevels = self.frame.columns.nlevels
            out = [(([(' ' * len(i[(- 1)]))] * clevels) + i) for i in gen]
            cnames = self.frame.columns.names
            if any(cnames):
                new_names = [(i if i else '{}') for i in cnames]
                out[(self.frame.index.nlevels - 1)][:clevels] = new_names
            strcols = (out + strcols[1:])
        return strcols

    @property
    def _empty_info_line(self):
        return f'''Empty {type(self.frame).__name__}
Columns: {self.frame.columns}
Index: {self.frame.index}'''

    def _preprocess_row(self, row):
        'Preprocess elements of the row.'
        if self.fmt.escape:
            crow = _escape_symbols(row)
        else:
            crow = [(x if x else '{}') for x in row]
        if (self.fmt.bold_rows and self.fmt.index):
            crow = _convert_to_bold(crow, self.index_levels)
        return crow

    def _format_multicolumn(self, row):
        '\n        Combine columns belonging to a group to a single multicolumn entry\n        according to self.multicolumn_format\n\n        e.g.:\n        a &  &  & b & c &\n        will become\n        \\multicolumn{3}{l}{a} & b & \\multicolumn{2}{l}{c}\n        '
        row2 = row[:self.index_levels]
        ncol = 1
        coltext = ''

        def append_col():
            if (ncol > 1):
                row2.append(f'\multicolumn{{{ncol:d}}}{{{self.multicolumn_format}}}{{{coltext.strip()}}}')
            else:
                row2.append(coltext)
        for c in row[self.index_levels:]:
            if c.strip():
                if coltext:
                    append_col()
                coltext = c
                ncol = 1
            else:
                ncol += 1
        if coltext:
            append_col()
        return row2

    def _format_multirow(self, row, i):
        '\n        Check following rows, whether row should be a multirow\n\n        e.g.:     becomes:\n        a & 0 &   \\multirow{2}{*}{a} & 0 &\n          & 1 &     & 1 &\n        b & 0 &   \\cline{1-2}\n                  b & 0 &\n        '
        for j in range(self.index_levels):
            if row[j].strip():
                nrow = 1
                for r in self.strrows[(i + 1):]:
                    if (not r[j].strip()):
                        nrow += 1
                    else:
                        break
                if (nrow > 1):
                    row[j] = f'\multirow{{{nrow:d}}}{{*}}{{{row[j].strip()}}}'
                    self.clinebuf.append([((i + nrow) - 1), (j + 1)])
        return row

    def _compose_cline(self, i, icol):
        '\n        Create clines after multirow-blocks are finished.\n        '
        lst = []
        for cl in self.clinebuf:
            if (cl[0] == i):
                lst.append(f'''
\cline{{{cl[1]:d}-{icol:d}}}''')
                self.clinebuf = [x for x in self.clinebuf if (x[0] != i)]
        return ''.join(lst)

class RowStringIterator(RowStringConverter):
    'Iterator over rows of the header or the body of the table.'

    @abstractmethod
    def __iter__(self):
        'Iterate over LaTeX string representations of rows.'

class RowHeaderIterator(RowStringIterator):
    'Iterator for the table header rows.'

    def __iter__(self):
        for row_num in range(len(self.strrows)):
            if (row_num < self._header_row_num):
                (yield self.get_strrow(row_num))

class RowBodyIterator(RowStringIterator):
    'Iterator for the table body rows.'

    def __iter__(self):
        for row_num in range(len(self.strrows)):
            if (row_num >= self._header_row_num):
                (yield self.get_strrow(row_num))

class TableBuilderAbstract(ABC):
    "\n    Abstract table builder producing string representation of LaTeX table.\n\n    Parameters\n    ----------\n    formatter : `DataFrameFormatter`\n        Instance of `DataFrameFormatter`.\n    column_format: str, optional\n        Column format, for example, 'rcl' for three columns.\n    multicolumn: bool, optional\n        Use multicolumn to enhance MultiIndex columns.\n    multicolumn_format: str, optional\n        The alignment for multicolumns, similar to column_format.\n    multirow: bool, optional\n        Use multirow to enhance MultiIndex rows.\n    caption: str, optional\n        Table caption.\n    short_caption: str, optional\n        Table short caption.\n    label: str, optional\n        LaTeX label.\n    position: str, optional\n        Float placement specifier, for example, 'htb'.\n    "

    def __init__(self, formatter, column_format=None, multicolumn=False, multicolumn_format=None, multirow=False, caption=None, short_caption=None, label=None, position=None):
        self.fmt = formatter
        self.column_format = column_format
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.caption = caption
        self.short_caption = short_caption
        self.label = label
        self.position = position

    def get_result(self):
        'String representation of LaTeX table.'
        elements = [self.env_begin, self.top_separator, self.header, self.middle_separator, self.env_body, self.bottom_separator, self.env_end]
        result = '\n'.join([item for item in elements if item])
        trailing_newline = '\n'
        result += trailing_newline
        return result

    @property
    @abstractmethod
    def env_begin(self):
        'Beginning of the environment.'

    @property
    @abstractmethod
    def top_separator(self):
        'Top level separator.'

    @property
    @abstractmethod
    def header(self):
        'Header lines.'

    @property
    @abstractmethod
    def middle_separator(self):
        'Middle level separator.'

    @property
    @abstractmethod
    def env_body(self):
        'Environment body.'

    @property
    @abstractmethod
    def bottom_separator(self):
        'Bottom level separator.'

    @property
    @abstractmethod
    def env_end(self):
        'End of the environment.'

class GenericTableBuilder(TableBuilderAbstract):
    'Table builder producing string representation of LaTeX table.'

    @property
    def header(self):
        iterator = self._create_row_iterator(over='header')
        return '\n'.join(list(iterator))

    @property
    def top_separator(self):
        return '\\toprule'

    @property
    def middle_separator(self):
        return ('\\midrule' if self._is_separator_required() else '')

    @property
    def env_body(self):
        iterator = self._create_row_iterator(over='body')
        return '\n'.join(list(iterator))

    def _is_separator_required(self):
        return bool((self.header and self.env_body))

    @property
    def _position_macro(self):
        'Position macro, extracted from self.position, like [h].'
        return (f'[{self.position}]' if self.position else '')

    @property
    def _caption_macro(self):
        'Caption macro, extracted from self.caption.\n\n        With short caption:\n            \\caption[short_caption]{caption_string}.\n\n        Without short caption:\n            \\caption{caption_string}.\n        '
        if self.caption:
            return ''.join(['\\caption', (f'[{self.short_caption}]' if self.short_caption else ''), f'{{{self.caption}}}'])
        return ''

    @property
    def _label_macro(self):
        'Label macro, extracted from self.label, like \\label{ref}.'
        return (f'\label{{{self.label}}}' if self.label else '')

    def _create_row_iterator(self, over):
        "Create iterator over header or body of the table.\n\n        Parameters\n        ----------\n        over : {'body', 'header'}\n            Over what to iterate.\n\n        Returns\n        -------\n        RowStringIterator\n            Iterator over body or header.\n        "
        iterator_kind = self._select_iterator(over)
        return iterator_kind(formatter=self.fmt, multicolumn=self.multicolumn, multicolumn_format=self.multicolumn_format, multirow=self.multirow)

    def _select_iterator(self, over):
        'Select proper iterator over table rows.'
        if (over == 'header'):
            return RowHeaderIterator
        elif (over == 'body'):
            return RowBodyIterator
        else:
            msg = f"'over' must be either 'header' or 'body', but {over} was provided"
            raise ValueError(msg)

class LongTableBuilder(GenericTableBuilder):
    'Concrete table builder for longtable.\n\n    >>> from pandas import DataFrame\n    >>> from pandas.io.formats import format as fmt\n    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})\n    >>> formatter = fmt.DataFrameFormatter(df)\n    >>> builder = LongTableBuilder(formatter, caption=\'a long table\',\n    ...                            label=\'tab:long\', column_format=\'lrl\')\n    >>> table = builder.get_result()\n    >>> print(table)\n    \\begin{longtable}{lrl}\n    \\caption{a long table}\n    \\label{tab:long}\\\\\n    \\toprule\n    {} &  a &   b \\\\\n    \\midrule\n    \\endfirsthead\n    \\caption[]{a long table} \\\\\n    \\toprule\n    {} &  a &   b \\\\\n    \\midrule\n    \\endhead\n    \\midrule\n    \\multicolumn{3}{r}{{Continued on next page}} \\\\\n    \\midrule\n    \\endfoot\n    <BLANKLINE>\n    \\bottomrule\n    \\endlastfoot\n    0 &  1 &  b1 \\\\\n    1 &  2 &  b2 \\\\\n    \\end{longtable}\n    <BLANKLINE>\n    '

    @property
    def env_begin(self):
        first_row = f'\begin{{longtable}}{self._position_macro}{{{self.column_format}}}'
        elements = [first_row, f'{self._caption_and_label()}']
        return '\n'.join([item for item in elements if item])

    def _caption_and_label(self):
        if (self.caption or self.label):
            double_backslash = '\\\\'
            elements = [f'{self._caption_macro}', f'{self._label_macro}']
            caption_and_label = '\n'.join([item for item in elements if item])
            caption_and_label += double_backslash
            return caption_and_label
        else:
            return ''

    @property
    def middle_separator(self):
        iterator = self._create_row_iterator(over='header')
        elements = ['\\midrule', '\\endfirsthead', (f'\caption[]{{{self.caption}}} \\' if self.caption else ''), self.top_separator, self.header, '\\midrule', '\\endhead', '\\midrule', f'\multicolumn{{{len(iterator.strcols)}}}{{r}}{{{{Continued on next page}}}} \\', '\\midrule', '\\endfoot\n', '\\bottomrule', '\\endlastfoot']
        if self._is_separator_required():
            return '\n'.join(elements)
        return ''

    @property
    def bottom_separator(self):
        return ''

    @property
    def env_end(self):
        return '\\end{longtable}'

class RegularTableBuilder(GenericTableBuilder):
    'Concrete table builder for regular table.\n\n    >>> from pandas import DataFrame\n    >>> from pandas.io.formats import format as fmt\n    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})\n    >>> formatter = fmt.DataFrameFormatter(df)\n    >>> builder = RegularTableBuilder(formatter, caption=\'caption\', label=\'lab\',\n    ...                               column_format=\'lrc\')\n    >>> table = builder.get_result()\n    >>> print(table)\n    \\begin{table}\n    \\centering\n    \\caption{caption}\n    \\label{lab}\n    \\begin{tabular}{lrc}\n    \\toprule\n    {} &  a &   b \\\\\n    \\midrule\n    0 &  1 &  b1 \\\\\n    1 &  2 &  b2 \\\\\n    \\bottomrule\n    \\end{tabular}\n    \\end{table}\n    <BLANKLINE>\n    '

    @property
    def env_begin(self):
        elements = [f'\begin{{table}}{self._position_macro}', '\\centering', f'{self._caption_macro}', f'{self._label_macro}', f'\begin{{tabular}}{{{self.column_format}}}']
        return '\n'.join([item for item in elements if item])

    @property
    def bottom_separator(self):
        return '\\bottomrule'

    @property
    def env_end(self):
        return '\n'.join(['\\end{tabular}', '\\end{table}'])

class TabularBuilder(GenericTableBuilder):
    'Concrete table builder for tabular environment.\n\n    >>> from pandas import DataFrame\n    >>> from pandas.io.formats import format as fmt\n    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})\n    >>> formatter = fmt.DataFrameFormatter(df)\n    >>> builder = TabularBuilder(formatter, column_format=\'lrc\')\n    >>> table = builder.get_result()\n    >>> print(table)\n    \\begin{tabular}{lrc}\n    \\toprule\n    {} &  a &   b \\\\\n    \\midrule\n    0 &  1 &  b1 \\\\\n    1 &  2 &  b2 \\\\\n    \\bottomrule\n    \\end{tabular}\n    <BLANKLINE>\n    '

    @property
    def env_begin(self):
        return f'\begin{{tabular}}{{{self.column_format}}}'

    @property
    def bottom_separator(self):
        return '\\bottomrule'

    @property
    def env_end(self):
        return '\\end{tabular}'

class LatexFormatter():
    "\n    Used to render a DataFrame to a LaTeX tabular/longtable environment output.\n\n    Parameters\n    ----------\n    formatter : `DataFrameFormatter`\n    longtable : bool, default False\n        Use longtable environment.\n    column_format : str, default None\n        The columns format as specified in `LaTeX table format\n        <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g 'rcl' for 3 columns\n    multicolumn : bool, default False\n        Use \\multicolumn to enhance MultiIndex columns.\n    multicolumn_format : str, default 'l'\n        The alignment for multicolumns, similar to `column_format`\n    multirow : bool, default False\n        Use \\multirow to enhance MultiIndex rows.\n    caption : str or tuple, optional\n        Tuple (full_caption, short_caption),\n        which results in \\caption[short_caption]{full_caption};\n        if a single string is passed, no short caption will be set.\n    label : str, optional\n        The LaTeX label to be placed inside ``\\label{}`` in the output.\n    position : str, optional\n        The LaTeX positional argument for tables, to be placed after\n        ``\\begin{}`` in the output.\n\n    See Also\n    --------\n    HTMLFormatter\n    "

    def __init__(self, formatter, longtable=False, column_format=None, multicolumn=False, multicolumn_format=None, multirow=False, caption=None, label=None, position=None):
        self.fmt = formatter
        self.frame = self.fmt.frame
        self.longtable = longtable
        self.column_format = column_format
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        (self.caption, self.short_caption) = _split_into_full_short_caption(caption)
        self.label = label
        self.position = position

    def to_string(self):
        '\n        Render a DataFrame to a LaTeX tabular, longtable, or table/tabular\n        environment output.\n        '
        return self.builder.get_result()

    @property
    def builder(self):
        'Concrete table builder.\n\n        Returns\n        -------\n        TableBuilder\n        '
        builder = self._select_builder()
        return builder(formatter=self.fmt, column_format=self.column_format, multicolumn=self.multicolumn, multicolumn_format=self.multicolumn_format, multirow=self.multirow, caption=self.caption, short_caption=self.short_caption, label=self.label, position=self.position)

    def _select_builder(self):
        'Select proper table builder.'
        if self.longtable:
            return LongTableBuilder
        if any([self.caption, self.label, self.position]):
            return RegularTableBuilder
        return TabularBuilder

    @property
    def column_format(self):
        'Column format.'
        return self._column_format

    @column_format.setter
    def column_format(self, input_column_format):
        'Setter for column format.'
        if (input_column_format is None):
            self._column_format = (self._get_index_format() + self._get_column_format_based_on_dtypes())
        elif (not isinstance(input_column_format, str)):
            raise ValueError(f'column_format must be str or unicode, not {type(input_column_format)}')
        else:
            self._column_format = input_column_format

    def _get_column_format_based_on_dtypes(self):
        'Get column format based on data type.\n\n        Right alignment for numbers and left - for strings.\n        '

        def get_col_type(dtype):
            if issubclass(dtype.type, np.number):
                return 'r'
            return 'l'
        dtypes = self.frame.dtypes._values
        return ''.join(map(get_col_type, dtypes))

    def _get_index_format(self):
        'Get index column format.'
        return (('l' * self.frame.index.nlevels) if self.fmt.index else '')

def _escape_symbols(row):
    'Carry out string replacements for special symbols.\n\n    Parameters\n    ----------\n    row : list\n        List of string, that may contain special symbols.\n\n    Returns\n    -------\n    list\n        list of strings with the special symbols replaced.\n    '
    return [(x.replace('\\', '\\textbackslash ').replace('_', '\\_').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('{', '\\{').replace('}', '\\}').replace('~', '\\textasciitilde ').replace('^', '\\textasciicircum ').replace('&', '\\&') if (x and (x != '{}')) else '{}') for x in row]

def _convert_to_bold(crow, ilevels):
    'Convert elements in ``crow`` to bold.'
    return [(f'\textbf{{{x}}}' if ((j < ilevels) and (x.strip() not in ['', '{}'])) else x) for (j, x) in enumerate(crow)]
if (__name__ == '__main__'):
    import doctest
    doctest.testmod()
