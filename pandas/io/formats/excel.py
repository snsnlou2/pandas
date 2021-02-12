
'\nUtilities for conversion to writer-agnostic Excel representation.\n'
from functools import reduce
import itertools
import re
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Union, cast
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas._typing import Label, StorageOptions
from pandas.util._decorators import doc
from pandas.core.dtypes import missing
from pandas.core.dtypes.common import is_float, is_scalar
from pandas import DataFrame, Index, MultiIndex, PeriodIndex
from pandas.core import generic
import pandas.core.common as com
from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import CSSResolver, CSSWarning
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.printing import pprint_thing

class ExcelCell():
    __fields__ = ('row', 'col', 'val', 'style', 'mergestart', 'mergeend')
    __slots__ = __fields__

    def __init__(self, row, col, val, style=None, mergestart=None, mergeend=None):
        self.row = row
        self.col = col
        self.val = val
        self.style = style
        self.mergestart = mergestart
        self.mergeend = mergeend

class CSSToExcelConverter():
    '\n    A callable for converting CSS declarations to ExcelWriter styles\n\n    Supports parts of CSS 2.2, with minimal CSS 3.0 support (e.g. text-shadow),\n    focusing on font styling, backgrounds, borders and alignment.\n\n    Operates by first computing CSS styles in a fairly generic\n    way (see :meth:`compute_css`) then determining Excel style\n    properties from CSS properties (see :meth:`build_xlstyle`).\n\n    Parameters\n    ----------\n    inherited : str, optional\n        CSS declarations understood to be the containing scope for the\n        CSS processed by :meth:`__call__`.\n    '
    NAMED_COLORS = CSS4_COLORS
    VERTICAL_MAP = {'top': 'top', 'text-top': 'top', 'middle': 'center', 'baseline': 'bottom', 'bottom': 'bottom', 'text-bottom': 'bottom'}
    BOLD_MAP = {'bold': True, 'bolder': True, '600': True, '700': True, '800': True, '900': True, 'normal': False, 'lighter': False, '100': False, '200': False, '300': False, '400': False, '500': False}
    ITALIC_MAP = {'normal': False, 'italic': True, 'oblique': True}
    FAMILY_MAP = {'serif': 1, 'sans-serif': 2, 'cursive': 4, 'fantasy': 5}

    def __init__(self, inherited=None):
        if (inherited is not None):
            self.inherited = self.compute_css(inherited)
        else:
            self.inherited = None
    compute_css = CSSResolver()

    def __call__(self, declarations_str):
        '\n        Convert CSS declarations to ExcelWriter style.\n\n        Parameters\n        ----------\n        declarations_str : str\n            List of CSS declarations.\n            e.g. "font-weight: bold; background: blue"\n\n        Returns\n        -------\n        xlstyle : dict\n            A style as interpreted by ExcelWriter when found in\n            ExcelCell.style.\n        '
        properties = self.compute_css(declarations_str, self.inherited)
        return self.build_xlstyle(properties)

    def build_xlstyle(self, props):
        out = {'alignment': self.build_alignment(props), 'border': self.build_border(props), 'fill': self.build_fill(props), 'font': self.build_font(props), 'number_format': self.build_number_format(props)}

        def remove_none(d: Dict[(str, str)]) -> None:
            'Remove key where value is None, through nested dicts'
            for (k, v) in list(d.items()):
                if (v is None):
                    del d[k]
                elif isinstance(v, dict):
                    remove_none(v)
                    if (not v):
                        del d[k]
        remove_none(out)
        return out

    def build_alignment(self, props):
        return {'horizontal': props.get('text-align'), 'vertical': self._get_vertical_alignment(props), 'wrap_text': self._get_is_wrap_text(props)}

    def _get_vertical_alignment(self, props):
        vertical_align = props.get('vertical-align')
        if vertical_align:
            return self.VERTICAL_MAP.get(vertical_align)
        return None

    def _get_is_wrap_text(self, props):
        if (props.get('white-space') is None):
            return None
        return bool((props['white-space'] not in ('nowrap', 'pre', 'pre-line')))

    def build_border(self, props):
        return {side: {'style': self._border_style(props.get(f'border-{side}-style'), props.get(f'border-{side}-width')), 'color': self.color_to_excel(props.get(f'border-{side}-color'))} for side in ['top', 'right', 'bottom', 'left']}

    def _border_style(self, style, width):
        if ((width is None) and (style is None)):
            return None
        if ((style == 'none') or (style == 'hidden')):
            return None
        width_name = self._get_width_name(width)
        if (width_name is None):
            return None
        if (style in (None, 'groove', 'ridge', 'inset', 'outset', 'solid')):
            return width_name
        if (style == 'double'):
            return 'double'
        if (style == 'dotted'):
            if (width_name in ('hair', 'thin')):
                return 'dotted'
            return 'mediumDashDotDot'
        if (style == 'dashed'):
            if (width_name in ('hair', 'thin')):
                return 'dashed'
            return 'mediumDashed'

    def _get_width_name(self, width_input):
        width = self._width_to_float(width_input)
        if (width < 1e-05):
            return None
        elif (width < 1.3):
            return 'thin'
        elif (width < 2.8):
            return 'medium'
        return 'thick'

    def _width_to_float(self, width):
        if (width is None):
            width = '2pt'
        return self._pt_to_float(width)

    def _pt_to_float(self, pt_string):
        assert pt_string.endswith('pt')
        return float(pt_string.rstrip('pt'))

    def build_fill(self, props):
        fill_color = props.get('background-color')
        if (fill_color not in (None, 'transparent', 'none')):
            return {'fgColor': self.color_to_excel(fill_color), 'patternType': 'solid'}

    def build_number_format(self, props):
        return {'format_code': props.get('number-format')}

    def build_font(self, props):
        font_names = self._get_font_names(props)
        decoration = self._get_decoration(props)
        return {'name': (font_names[0] if font_names else None), 'family': self._select_font_family(font_names), 'size': self._get_font_size(props), 'bold': self._get_is_bold(props), 'italic': self._get_is_italic(props), 'underline': ('single' if ('underline' in decoration) else None), 'strike': (('line-through' in decoration) or None), 'color': self.color_to_excel(props.get('color')), 'shadow': self._get_shadow(props)}

    def _get_is_bold(self, props):
        weight = props.get('font-weight')
        if weight:
            return self.BOLD_MAP.get(weight)
        return None

    def _get_is_italic(self, props):
        font_style = props.get('font-style')
        if font_style:
            return self.ITALIC_MAP.get(font_style)
        return None

    def _get_decoration(self, props):
        decoration = props.get('text-decoration')
        if (decoration is not None):
            return decoration.split()
        else:
            return ()

    def _get_underline(self, decoration):
        if ('underline' in decoration):
            return 'single'
        return None

    def _get_shadow(self, props):
        if ('text-shadow' in props):
            return bool(re.search('^[^#(]*[1-9]', props['text-shadow']))
        return None

    def _get_font_names(self, props):
        font_names_tmp = re.findall('(?x)\n            (\n            "(?:[^"]|\\\\")+"\n            |\n            \'(?:[^\']|\\\\\')+\'\n            |\n            [^\'",]+\n            )(?=,|\\s*$)\n        ', props.get('font-family', ''))
        font_names = []
        for name in font_names_tmp:
            if (name[:1] == '"'):
                name = name[1:(- 1)].replace('\\"', '"')
            elif (name[:1] == "'"):
                name = name[1:(- 1)].replace("\\'", "'")
            else:
                name = name.strip()
            if name:
                font_names.append(name)
        return font_names

    def _get_font_size(self, props):
        size = props.get('font-size')
        if (size is None):
            return size
        return self._pt_to_float(size)

    def _select_font_family(self, font_names):
        family = None
        for name in font_names:
            family = self.FAMILY_MAP.get(name)
            if family:
                break
        return family

    def color_to_excel(self, val):
        if (val is None):
            return None
        if self._is_hex_color(val):
            return self._convert_hex_to_excel(val)
        try:
            return self.NAMED_COLORS[val]
        except KeyError:
            warnings.warn(f'Unhandled color format: {repr(val)}', CSSWarning)
        return None

    def _is_hex_color(self, color_string):
        return bool(color_string.startswith('#'))

    def _convert_hex_to_excel(self, color_string):
        code = color_string.lstrip('#')
        if self._is_shorthand_color(color_string):
            return (((code[0] * 2) + (code[1] * 2)) + (code[2] * 2)).upper()
        else:
            return code.upper()

    def _is_shorthand_color(self, color_string):
        'Check if color code is shorthand.\n\n        #FFF is a shorthand as opposed to full #FFFFFF.\n        '
        code = color_string.lstrip('#')
        if (len(code) == 3):
            return True
        elif (len(code) == 6):
            return False
        else:
            raise ValueError(f'Unexpected color {color_string}')

class ExcelFormatter():
    "\n    Class for formatting a DataFrame to a list of ExcelCells,\n\n    Parameters\n    ----------\n    df : DataFrame or Styler\n    na_rep: na representation\n    float_format : string, default None\n            Format string for floating point numbers\n    cols : sequence, optional\n        Columns to write\n    header : boolean or sequence of str, default True\n        Write out column names. If a list of string is given it is\n        assumed to be aliases for the column names\n    index : boolean, default True\n        output row names (index)\n    index_label : string or sequence, default None\n            Column label for index column(s) if desired. If None is given, and\n            `header` and `index` are True, then the index names are used. A\n            sequence should be given if the DataFrame uses MultiIndex.\n    merge_cells : boolean, default False\n            Format MultiIndex and Hierarchical Rows as merged cells.\n    inf_rep : string, default `'inf'`\n        representation for np.inf values (which aren't representable in Excel)\n        A `'-'` sign will be added in front of -inf.\n    style_converter : callable, optional\n        This translates Styler styles (CSS) into ExcelWriter styles.\n        Defaults to ``CSSToExcelConverter()``.\n        It should have signature css_declarations string -> excel style.\n        This is only called for body cells.\n    "
    max_rows = (2 ** 20)
    max_cols = (2 ** 14)

    def __init__(self, df, na_rep='', float_format=None, cols=None, header=True, index=True, index_label=None, merge_cells=False, inf_rep='inf', style_converter=None):
        self.rowcounter = 0
        self.na_rep = na_rep
        if (not isinstance(df, DataFrame)):
            self.styler = df
            df = df.data
            if (style_converter is None):
                style_converter = CSSToExcelConverter()
            self.style_converter = style_converter
        else:
            self.styler = None
        self.df = df
        if (cols is not None):
            if (not len(Index(cols).intersection(df.columns))):
                raise KeyError('passes columns are not ALL present dataframe')
            if (len(Index(cols).intersection(df.columns)) != len(cols)):
                raise KeyError("Not all names specified in 'columns' are found")
            self.df = df.reindex(columns=cols)
        self.columns = self.df.columns
        self.float_format = float_format
        self.index = index
        self.index_label = index_label
        self.header = header
        self.merge_cells = merge_cells
        self.inf_rep = inf_rep

    @property
    def header_style(self):
        return {'font': {'bold': True}, 'borders': {'top': 'thin', 'right': 'thin', 'bottom': 'thin', 'left': 'thin'}, 'alignment': {'horizontal': 'center', 'vertical': 'top'}}

    def _format_value(self, val):
        if (is_scalar(val) and missing.isna(val)):
            val = self.na_rep
        elif is_float(val):
            if missing.isposinf_scalar(val):
                val = self.inf_rep
            elif missing.isneginf_scalar(val):
                val = f'-{self.inf_rep}'
            elif (self.float_format is not None):
                val = float((self.float_format % val))
        if (getattr(val, 'tzinfo', None) is not None):
            raise ValueError('Excel does not support datetimes with timezones. Please ensure that datetimes are timezone unaware before writing to Excel.')
        return val

    def _format_header_mi(self):
        if (self.columns.nlevels > 1):
            if (not self.index):
                raise NotImplementedError("Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.")
        if (not (self._has_aliases or self.header)):
            return
        columns = self.columns
        level_strs = columns.format(sparsify=self.merge_cells, adjoin=False, names=False)
        level_lengths = get_level_lengths(level_strs)
        coloffset = 0
        lnum = 0
        if (self.index and isinstance(self.df.index, MultiIndex)):
            coloffset = (len(self.df.index[0]) - 1)
        if self.merge_cells:
            for (lnum, name) in enumerate(columns.names):
                (yield ExcelCell(row=lnum, col=coloffset, val=name, style=self.header_style))
            for (lnum, (spans, levels, level_codes)) in enumerate(zip(level_lengths, columns.levels, columns.codes)):
                values = levels.take(level_codes)
                for (i, span_val) in spans.items():
                    spans_multiple_cells = (span_val > 1)
                    (yield ExcelCell(row=lnum, col=((coloffset + i) + 1), val=values[i], style=self.header_style, mergestart=(lnum if spans_multiple_cells else None), mergeend=(((coloffset + i) + span_val) if spans_multiple_cells else None)))
        else:
            for (i, values) in enumerate(zip(*level_strs)):
                v = '.'.join(map(pprint_thing, values))
                (yield ExcelCell(lnum, ((coloffset + i) + 1), v, self.header_style))
        self.rowcounter = lnum

    def _format_header_regular(self):
        if (self._has_aliases or self.header):
            coloffset = 0
            if self.index:
                coloffset = 1
                if isinstance(self.df.index, MultiIndex):
                    coloffset = len(self.df.index[0])
            colnames = self.columns
            if self._has_aliases:
                self.header = cast(Sequence, self.header)
                if (len(self.header) != len(self.columns)):
                    raise ValueError(f'Writing {len(self.columns)} cols but got {len(self.header)} aliases')
                else:
                    colnames = self.header
            for (colindex, colname) in enumerate(colnames):
                (yield ExcelCell(self.rowcounter, (colindex + coloffset), colname, self.header_style))

    def _format_header(self):
        if isinstance(self.columns, MultiIndex):
            gen = self._format_header_mi()
        else:
            gen = self._format_header_regular()
        gen2 = ()
        if self.df.index.names:
            row = ([(x if (x is not None) else '') for x in self.df.index.names] + ([''] * len(self.columns)))
            if reduce((lambda x, y: (x and y)), map((lambda x: (x != '')), row)):
                gen2 = (ExcelCell(self.rowcounter, colindex, val, self.header_style) for (colindex, val) in enumerate(row))
                self.rowcounter += 1
        return itertools.chain(gen, gen2)

    def _format_body(self):
        if isinstance(self.df.index, MultiIndex):
            return self._format_hierarchical_rows()
        else:
            return self._format_regular_rows()

    def _format_regular_rows(self):
        if (self._has_aliases or self.header):
            self.rowcounter += 1
        if self.index:
            if (self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index))):
                index_label = self.index_label[0]
            elif (self.index_label and isinstance(self.index_label, str)):
                index_label = self.index_label
            else:
                index_label = self.df.index.names[0]
            if isinstance(self.columns, MultiIndex):
                self.rowcounter += 1
            if (index_label and (self.header is not False)):
                (yield ExcelCell((self.rowcounter - 1), 0, index_label, self.header_style))
            index_values = self.df.index
            if isinstance(self.df.index, PeriodIndex):
                index_values = self.df.index.to_timestamp()
            for (idx, idxval) in enumerate(index_values):
                (yield ExcelCell((self.rowcounter + idx), 0, idxval, self.header_style))
            coloffset = 1
        else:
            coloffset = 0
        (yield from self._generate_body(coloffset))

    def _format_hierarchical_rows(self):
        if (self._has_aliases or self.header):
            self.rowcounter += 1
        gcolidx = 0
        if self.index:
            index_labels = self.df.index.names
            if (self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index))):
                index_labels = self.index_label
            if (isinstance(self.columns, MultiIndex) and self.merge_cells):
                self.rowcounter += 1
            if (com.any_not_none(*index_labels) and (self.header is not False)):
                for (cidx, name) in enumerate(index_labels):
                    (yield ExcelCell((self.rowcounter - 1), cidx, name, self.header_style))
            if self.merge_cells:
                level_strs = self.df.index.format(sparsify=True, adjoin=False, names=False)
                level_lengths = get_level_lengths(level_strs)
                for (spans, levels, level_codes) in zip(level_lengths, self.df.index.levels, self.df.index.codes):
                    values = levels.take(level_codes, allow_fill=levels._can_hold_na, fill_value=levels._na_value)
                    for (i, span_val) in spans.items():
                        spans_multiple_cells = (span_val > 1)
                        (yield ExcelCell(row=(self.rowcounter + i), col=gcolidx, val=values[i], style=self.header_style, mergestart=((((self.rowcounter + i) + span_val) - 1) if spans_multiple_cells else None), mergeend=(gcolidx if spans_multiple_cells else None)))
                    gcolidx += 1
            else:
                for indexcolvals in zip(*self.df.index):
                    for (idx, indexcolval) in enumerate(indexcolvals):
                        (yield ExcelCell(row=(self.rowcounter + idx), col=gcolidx, val=indexcolval, style=self.header_style))
                    gcolidx += 1
        (yield from self._generate_body(gcolidx))

    @property
    def _has_aliases(self):
        'Whether the aliases for column names are present.'
        return is_list_like(self.header)

    def _generate_body(self, coloffset):
        if (self.styler is None):
            styles = None
        else:
            styles = self.styler._compute().ctx
            if (not styles):
                styles = None
        xlstyle = None
        for colidx in range(len(self.columns)):
            series = self.df.iloc[:, colidx]
            for (i, val) in enumerate(series):
                if (styles is not None):
                    xlstyle = self.style_converter(';'.join(styles[(i, colidx)]))
                (yield ExcelCell((self.rowcounter + i), (colidx + coloffset), val, xlstyle))

    def get_formatted_cells(self):
        for cell in itertools.chain(self._format_header(), self._format_body()):
            cell.val = self._format_value(cell.val)
            (yield cell)

    @doc(storage_options=generic._shared_docs['storage_options'])
    def write(self, writer, sheet_name='Sheet1', startrow=0, startcol=0, freeze_panes=None, engine=None, storage_options=None):
        "\n        writer : path-like, file-like, or ExcelWriter object\n            File path or existing ExcelWriter\n        sheet_name : string, default 'Sheet1'\n            Name of sheet which will contain DataFrame\n        startrow :\n            upper left cell row to dump data frame\n        startcol :\n            upper left cell column to dump data frame\n        freeze_panes : tuple of integer (length 2), default None\n            Specifies the one-based bottommost row and rightmost column that\n            is to be frozen\n        engine : string, default None\n            write engine to use if writer is a path - you can also set this\n            via the options ``io.excel.xlsx.writer``, ``io.excel.xls.writer``,\n            and ``io.excel.xlsm.writer``.\n\n            .. deprecated:: 1.2.0\n\n                As the `xlwt <https://pypi.org/project/xlwt/>`__ package is no longer\n                maintained, the ``xlwt`` engine will be removed in a future\n                version of pandas.\n\n        {storage_options}\n\n            .. versionadded:: 1.2.0\n        "
        from pandas.io.excel import ExcelWriter
        (num_rows, num_cols) = self.df.shape
        if ((num_rows > self.max_rows) or (num_cols > self.max_cols)):
            raise ValueError(f'This sheet is too large! Your sheet size is: {num_rows}, {num_cols} Max sheet size is: {self.max_rows}, {self.max_cols}')
        formatted_cells = self.get_formatted_cells()
        if isinstance(writer, ExcelWriter):
            need_save = False
        else:
            writer = ExcelWriter(writer, engine=engine, storage_options=storage_options)
            need_save = True
        try:
            writer.write_cells(formatted_cells, sheet_name, startrow=startrow, startcol=startcol, freeze_panes=freeze_panes)
        finally:
            if need_save:
                writer.close()
