
'\nModule for formatting output data in HTML.\n'
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union, cast
from pandas._config import get_option
from pandas._libs import lib
from pandas import MultiIndex, option_context
from pandas.io.common import is_url
from pandas.io.formats.format import DataFrameFormatter, get_level_lengths
from pandas.io.formats.printing import pprint_thing

class HTMLFormatter():
    '\n    Internal class for formatting output data in html.\n    This class is intended for shared functionality between\n    DataFrame.to_html() and DataFrame._repr_html_().\n    Any logic in common with other output formatting methods\n    should ideally be inherited from classes in format.py\n    and this class responsible for only producing html markup.\n    '
    indent_delta = 2

    def __init__(self, formatter, classes=None, border=None, table_id=None, render_links=False):
        self.fmt = formatter
        self.classes = classes
        self.frame = self.fmt.frame
        self.columns = self.fmt.tr_frame.columns
        self.elements: List[str] = []
        self.bold_rows = self.fmt.bold_rows
        self.escape = self.fmt.escape
        self.show_dimensions = self.fmt.show_dimensions
        if (border is None):
            border = cast(int, get_option('display.html.border'))
        self.border = border
        self.table_id = table_id
        self.render_links = render_links
        self.col_space = {column: (f'{value}px' if isinstance(value, int) else value) for (column, value) in self.fmt.col_space.items()}

    def to_string(self):
        lines = self.render()
        if any((isinstance(x, str) for x in lines)):
            lines = [str(x) for x in lines]
        return '\n'.join(lines)

    def render(self):
        self._write_table()
        if self.should_show_dimensions:
            by = chr(215)
            self.write(f'<p>{len(self.frame)} rows {by} {len(self.frame.columns)} columns</p>')
        return self.elements

    @property
    def should_show_dimensions(self):
        return self.fmt.should_show_dimensions

    @property
    def show_row_idx_names(self):
        return self.fmt.show_row_idx_names

    @property
    def show_col_idx_names(self):
        return self.fmt.show_col_idx_names

    @property
    def row_levels(self):
        if self.fmt.index:
            return self.frame.index.nlevels
        elif self.show_col_idx_names:
            return 1
        return 0

    def _get_columns_formatted_values(self):
        return self.columns

    @property
    def is_truncated(self):
        return self.fmt.is_truncated

    @property
    def ncols(self):
        return len(self.fmt.tr_frame.columns)

    def write(self, s, indent=0):
        rs = pprint_thing(s)
        self.elements.append(((' ' * indent) + rs))

    def write_th(self, s, header=False, indent=0, tags=None):
        '\n        Method for writing a formatted <th> cell.\n\n        If col_space is set on the formatter then that is used for\n        the value of min-width.\n\n        Parameters\n        ----------\n        s : object\n            The data to be written inside the cell.\n        header : bool, default False\n            Set to True if the <th> is for use inside <thead>.  This will\n            cause min-width to be set if there is one.\n        indent : int, default 0\n            The indentation level of the cell.\n        tags : str, default None\n            Tags to include in the cell.\n\n        Returns\n        -------\n        A written <th> cell.\n        '
        col_space = self.col_space.get(s, None)
        if (header and (col_space is not None)):
            tags = (tags or '')
            tags += f'style="min-width: {col_space};"'
        self._write_cell(s, kind='th', indent=indent, tags=tags)

    def write_td(self, s, indent=0, tags=None):
        self._write_cell(s, kind='td', indent=indent, tags=tags)

    def _write_cell(self, s, kind='td', indent=0, tags=None):
        if (tags is not None):
            start_tag = f'<{kind} {tags}>'
        else:
            start_tag = f'<{kind}>'
        if self.escape:
            esc = {'&': '&amp;', '<': '&lt;', '>': '&gt;'}
        else:
            esc = {}
        rs = pprint_thing(s, escape_chars=esc).strip()
        if (self.render_links and is_url(rs)):
            rs_unescaped = pprint_thing(s, escape_chars={}).strip()
            start_tag += f'<a href="{rs_unescaped}" target="_blank">'
            end_a = '</a>'
        else:
            end_a = ''
        self.write(f'{start_tag}{rs}{end_a}</{kind}>', indent)

    def write_tr(self, line, indent=0, indent_delta=0, header=False, align=None, tags=None, nindex_levels=0):
        if (tags is None):
            tags = {}
        if (align is None):
            self.write('<tr>', indent)
        else:
            self.write(f'<tr style="text-align: {align};">', indent)
        indent += indent_delta
        for (i, s) in enumerate(line):
            val_tag = tags.get(i, None)
            if (header or (self.bold_rows and (i < nindex_levels))):
                self.write_th(s, indent=indent, header=header, tags=val_tag)
            else:
                self.write_td(s, indent, tags=val_tag)
        indent -= indent_delta
        self.write('</tr>', indent)

    def _write_table(self, indent=0):
        _classes = ['dataframe']
        use_mathjax = get_option('display.html.use_mathjax')
        if (not use_mathjax):
            _classes.append('tex2jax_ignore')
        if (self.classes is not None):
            if isinstance(self.classes, str):
                self.classes = self.classes.split()
            if (not isinstance(self.classes, (list, tuple))):
                raise TypeError(f'classes must be a string, list, or tuple, not {type(self.classes)}')
            _classes.extend(self.classes)
        if (self.table_id is None):
            id_section = ''
        else:
            id_section = f' id="{self.table_id}"'
        self.write(f"""<table border="{self.border}" class="{' '.join(_classes)}"{id_section}>""", indent)
        if (self.fmt.header or self.show_row_idx_names):
            self._write_header((indent + self.indent_delta))
        self._write_body((indent + self.indent_delta))
        self.write('</table>', indent)

    def _write_col_header(self, indent):
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        if isinstance(self.columns, MultiIndex):
            template = 'colspan="{span:d}" halign="left"'
            if self.fmt.sparsify:
                sentinel = lib.no_default
            else:
                sentinel = False
            levels = self.columns.format(sparsify=sentinel, adjoin=False, names=False)
            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = (len(level_lengths) - 1)
            for (lnum, (records, values)) in enumerate(zip(level_lengths, levels)):
                if is_truncated_horizontally:
                    ins_col = self.fmt.tr_col_num
                    if self.fmt.sparsify:
                        recs_new = {}
                        for (tag, span) in list(records.items()):
                            if (tag >= ins_col):
                                recs_new[(tag + 1)] = span
                            elif ((tag + span) > ins_col):
                                recs_new[tag] = (span + 1)
                                if (lnum == inner_lvl):
                                    values = ((values[:ins_col] + ('...',)) + values[ins_col:])
                                else:
                                    values = ((values[:ins_col] + (values[(ins_col - 1)],)) + values[ins_col:])
                            else:
                                recs_new[tag] = span
                            if ((tag + span) == ins_col):
                                recs_new[ins_col] = 1
                                values = ((values[:ins_col] + ('...',)) + values[ins_col:])
                        records = recs_new
                        inner_lvl = (len(level_lengths) - 1)
                        if (lnum == inner_lvl):
                            records[ins_col] = 1
                    else:
                        recs_new = {}
                        for (tag, span) in list(records.items()):
                            if (tag >= ins_col):
                                recs_new[(tag + 1)] = span
                            else:
                                recs_new[tag] = span
                        recs_new[ins_col] = 1
                        records = recs_new
                        values = ((values[:ins_col] + ['...']) + values[ins_col:])
                row = ([''] * (self.row_levels - 1))
                if (self.fmt.index or self.show_col_idx_names):
                    if self.fmt.show_index_names:
                        name = self.columns.names[lnum]
                        row.append(pprint_thing((name or '')))
                    else:
                        row.append('')
                tags = {}
                j = len(row)
                for (i, v) in enumerate(values):
                    if (i in records):
                        if (records[i] > 1):
                            tags[j] = template.format(span=records[i])
                    else:
                        continue
                    j += 1
                    row.append(v)
                self.write_tr(row, indent, self.indent_delta, tags=tags, header=True)
        else:
            row = ([''] * (self.row_levels - 1))
            if (self.fmt.index or self.show_col_idx_names):
                if self.fmt.show_index_names:
                    row.append((self.columns.name or ''))
                else:
                    row.append('')
            row.extend(self._get_columns_formatted_values())
            align = self.fmt.justify
            if is_truncated_horizontally:
                ins_col = (self.row_levels + self.fmt.tr_col_num)
                row.insert(ins_col, '...')
            self.write_tr(row, indent, self.indent_delta, header=True, align=align)

    def _write_row_header(self, indent):
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        row = ([(x if (x is not None) else '') for x in self.frame.index.names] + ([''] * (self.ncols + (1 if is_truncated_horizontally else 0))))
        self.write_tr(row, indent, self.indent_delta, header=True)

    def _write_header(self, indent):
        self.write('<thead>', indent)
        if self.fmt.header:
            self._write_col_header((indent + self.indent_delta))
        if self.show_row_idx_names:
            self._write_row_header((indent + self.indent_delta))
        self.write('</thead>', indent)

    def _get_formatted_values(self):
        with option_context('display.max_colwidth', None):
            fmt_values = {i: self.fmt.format_col(i) for i in range(self.ncols)}
        return fmt_values

    def _write_body(self, indent):
        self.write('<tbody>', indent)
        fmt_values = self._get_formatted_values()
        if (self.fmt.index and isinstance(self.frame.index, MultiIndex)):
            self._write_hierarchical_rows(fmt_values, (indent + self.indent_delta))
        else:
            self._write_regular_rows(fmt_values, (indent + self.indent_delta))
        self.write('</tbody>', indent)

    def _write_regular_rows(self, fmt_values, indent):
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        is_truncated_vertically = self.fmt.is_truncated_vertically
        nrows = len(self.fmt.tr_frame)
        if self.fmt.index:
            fmt = self.fmt._get_formatter('__index__')
            if (fmt is not None):
                index_values = self.fmt.tr_frame.index.map(fmt)
            else:
                index_values = self.fmt.tr_frame.index.format()
        row: List[str] = []
        for i in range(nrows):
            if (is_truncated_vertically and (i == self.fmt.tr_row_num)):
                str_sep_row = (['...'] * len(row))
                self.write_tr(str_sep_row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels)
            row = []
            if self.fmt.index:
                row.append(index_values[i])
            elif self.show_col_idx_names:
                row.append('')
            row.extend((fmt_values[j][i] for j in range(self.ncols)))
            if is_truncated_horizontally:
                dot_col_ix = (self.fmt.tr_col_num + self.row_levels)
                row.insert(dot_col_ix, '...')
            self.write_tr(row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels)

    def _write_hierarchical_rows(self, fmt_values, indent):
        template = 'rowspan="{span}" valign="top"'
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        is_truncated_vertically = self.fmt.is_truncated_vertically
        frame = self.fmt.tr_frame
        nrows = len(frame)
        assert isinstance(frame.index, MultiIndex)
        idx_values = frame.index.format(sparsify=False, adjoin=False, names=False)
        idx_values = list(zip(*idx_values))
        if self.fmt.sparsify:
            sentinel = lib.no_default
            levels = frame.index.format(sparsify=sentinel, adjoin=False, names=False)
            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = (len(level_lengths) - 1)
            if is_truncated_vertically:
                ins_row = self.fmt.tr_row_num
                inserted = False
                for (lnum, records) in enumerate(level_lengths):
                    rec_new = {}
                    for (tag, span) in list(records.items()):
                        if (tag >= ins_row):
                            rec_new[(tag + 1)] = span
                        elif ((tag + span) > ins_row):
                            rec_new[tag] = (span + 1)
                            if (not inserted):
                                dot_row = list(idx_values[(ins_row - 1)])
                                dot_row[(- 1)] = '...'
                                idx_values.insert(ins_row, tuple(dot_row))
                                inserted = True
                            else:
                                dot_row = list(idx_values[ins_row])
                                dot_row[(inner_lvl - lnum)] = '...'
                                idx_values[ins_row] = tuple(dot_row)
                        else:
                            rec_new[tag] = span
                        if ((tag + span) == ins_row):
                            rec_new[ins_row] = 1
                            if (lnum == 0):
                                idx_values.insert(ins_row, tuple((['...'] * len(level_lengths))))
                            elif inserted:
                                dot_row = list(idx_values[ins_row])
                                dot_row[(inner_lvl - lnum)] = '...'
                                idx_values[ins_row] = tuple(dot_row)
                    level_lengths[lnum] = rec_new
                level_lengths[inner_lvl][ins_row] = 1
                for ix_col in range(len(fmt_values)):
                    fmt_values[ix_col].insert(ins_row, '...')
                nrows += 1
            for i in range(nrows):
                row = []
                tags = {}
                sparse_offset = 0
                j = 0
                for (records, v) in zip(level_lengths, idx_values[i]):
                    if (i in records):
                        if (records[i] > 1):
                            tags[j] = template.format(span=records[i])
                    else:
                        sparse_offset += 1
                        continue
                    j += 1
                    row.append(v)
                row.extend((fmt_values[j][i] for j in range(self.ncols)))
                if is_truncated_horizontally:
                    row.insert(((self.row_levels - sparse_offset) + self.fmt.tr_col_num), '...')
                self.write_tr(row, indent, self.indent_delta, tags=tags, nindex_levels=(len(levels) - sparse_offset))
        else:
            row = []
            for i in range(len(frame)):
                if (is_truncated_vertically and (i == self.fmt.tr_row_num)):
                    str_sep_row = (['...'] * len(row))
                    self.write_tr(str_sep_row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels)
                idx_values = list(zip(*frame.index.format(sparsify=False, adjoin=False, names=False)))
                row = []
                row.extend(idx_values[i])
                row.extend((fmt_values[j][i] for j in range(self.ncols)))
                if is_truncated_horizontally:
                    row.insert((self.row_levels + self.fmt.tr_col_num), '...')
                self.write_tr(row, indent, self.indent_delta, tags=None, nindex_levels=frame.index.nlevels)

class NotebookFormatter(HTMLFormatter):
    '\n    Internal class for formatting output data in html for display in Jupyter\n    Notebooks. This class is intended for functionality specific to\n    DataFrame._repr_html_() and DataFrame.to_html(notebook=True)\n    '

    def _get_formatted_values(self):
        return {i: self.fmt.format_col(i) for i in range(self.ncols)}

    def _get_columns_formatted_values(self):
        return self.columns.format()

    def write_style(self):
        template_first = '            <style scoped>'
        template_last = '            </style>'
        template_select = '                .dataframe %s {\n                    %s: %s;\n                }'
        element_props = [('tbody tr th:only-of-type', 'vertical-align', 'middle'), ('tbody tr th', 'vertical-align', 'top')]
        if isinstance(self.columns, MultiIndex):
            element_props.append(('thead tr th', 'text-align', 'left'))
            if self.show_row_idx_names:
                element_props.append(('thead tr:last-of-type th', 'text-align', 'right'))
        else:
            element_props.append(('thead th', 'text-align', 'right'))
        template_mid = '\n\n'.join(map((lambda t: (template_select % t)), element_props))
        template = dedent('\n'.join((template_first, template_mid, template_last)))
        self.write(template)

    def render(self):
        self.write('<div>')
        self.write_style()
        super().render()
        self.write('</div>')
        return self.elements
