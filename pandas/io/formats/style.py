
'\nModule for applying conditional formatting to DataFrames and Series.\n'
from collections import defaultdict
from contextlib import contextmanager
import copy
from functools import partial
from itertools import product
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas._typing import Axis, FrameOrSeries, FrameOrSeriesUnion, Label
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.core.dtypes.common import is_float
import pandas as pd
from pandas.api.types import is_dict_like, is_list_like
from pandas.core import generic
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.indexing import maybe_numeric_slice, non_reducing_slice
jinja2 = import_optional_dependency('jinja2', extra='DataFrame.style requires jinja2.')
try:
    from matplotlib import colors
    import matplotlib.pyplot as plt
    has_mpl = True
except ImportError:
    has_mpl = False
    no_mpl_message = '{0} requires matplotlib.'

@contextmanager
def _mpl(func):
    if has_mpl:
        (yield (plt, colors))
    else:
        raise ImportError(no_mpl_message.format(func.__name__))

class Styler():
    "\n    Helps style a DataFrame or Series according to the data with HTML and CSS.\n\n    Parameters\n    ----------\n    data : Series or DataFrame\n        Data to be styled - either a Series or DataFrame.\n    precision : int\n        Precision to round floats to, defaults to pd.options.display.precision.\n    table_styles : list-like, default None\n        List of {selector: (attr, value)} dicts; see Notes.\n    uuid : str, default None\n        A unique identifier to avoid CSS collisions; generated automatically.\n    caption : str, default None\n        Caption to attach to the table.\n    table_attributes : str, default None\n        Items that show up in the opening ``<table>`` tag\n        in addition to automatic (by default) id.\n    cell_ids : bool, default True\n        If True, each cell will have an ``id`` attribute in their HTML tag.\n        The ``id`` takes the form ``T_<uuid>_row<num_row>_col<num_col>``\n        where ``<uuid>`` is the unique identifier, ``<num_row>`` is the row\n        number and ``<num_col>`` is the column number.\n    na_rep : str, optional\n        Representation for missing values.\n        If ``na_rep`` is None, no special formatting is applied.\n\n        .. versionadded:: 1.0.0\n\n    uuid_len : int, default 5\n        If ``uuid`` is not specified, the length of the ``uuid`` to randomly generate\n        expressed in hex characters, in range [0, 32].\n\n        .. versionadded:: 1.2.0\n\n    Attributes\n    ----------\n    env : Jinja2 jinja2.Environment\n    template : Jinja2 Template\n    loader : Jinja2 Loader\n\n    See Also\n    --------\n    DataFrame.style : Return a Styler object containing methods for building\n        a styled HTML representation for the DataFrame.\n\n    Notes\n    -----\n    Most styling will be done by passing style functions into\n    ``Styler.apply`` or ``Styler.applymap``. Style functions should\n    return values with strings containing CSS ``'attr: value'`` that will\n    be applied to the indicated cells.\n\n    If using in the Jupyter notebook, Styler has defined a ``_repr_html_``\n    to automatically render itself. Otherwise call Styler.render to get\n    the generated HTML.\n\n    CSS classes are attached to the generated HTML\n\n    * Index and Column names include ``index_name`` and ``level<k>``\n      where `k` is its level in a MultiIndex\n    * Index label cells include\n\n      * ``row_heading``\n      * ``row<n>`` where `n` is the numeric position of the row\n      * ``level<k>`` where `k` is the level in a MultiIndex\n\n    * Column label cells include\n      * ``col_heading``\n      * ``col<n>`` where `n` is the numeric position of the column\n      * ``level<k>`` where `k` is the level in a MultiIndex\n\n    * Blank cells include ``blank``\n    * Data cells include ``data``\n    "
    loader = jinja2.PackageLoader('pandas', 'io/formats/templates')
    env = jinja2.Environment(loader=loader, trim_blocks=True)
    template = env.get_template('html.tpl')

    def __init__(self, data, precision=None, table_styles=None, uuid=None, caption=None, table_attributes=None, cell_ids=True, na_rep=None, uuid_len=5):
        self.ctx: DefaultDict[(Tuple[(int, int)], List[str])] = defaultdict(list)
        self._todo: List[Tuple[(Callable, Tuple, Dict)]] = []
        if (not isinstance(data, (pd.Series, pd.DataFrame))):
            raise TypeError('``data`` must be a Series or DataFrame')
        if (data.ndim == 1):
            data = data.to_frame()
        if ((not data.index.is_unique) or (not data.columns.is_unique)):
            raise ValueError('style is not supported for non-unique indices.')
        self.data = data
        self.index = data.index
        self.columns = data.columns
        if ((not isinstance(uuid_len, int)) or (not (uuid_len >= 0))):
            raise TypeError('``uuid_len`` must be an integer in range [0, 32].')
        self.uuid_len = min(32, uuid_len)
        self.uuid = ((uuid or uuid4().hex[:self.uuid_len]) + '_')
        self.table_styles = table_styles
        self.caption = caption
        if (precision is None):
            precision = get_option('display.precision')
        self.precision = precision
        self.table_attributes = table_attributes
        self.hidden_index = False
        self.hidden_columns: Sequence[int] = []
        self.cell_ids = cell_ids
        self.na_rep = na_rep
        self.cell_context: Dict[(str, Any)] = {}

        def default_display_func(x):
            if ((self.na_rep is not None) and pd.isna(x)):
                return self.na_rep
            elif is_float(x):
                display_format = f'{x:.{self.precision}f}'
                return display_format
            else:
                return x
        self._display_funcs: DefaultDict[(Tuple[(int, int)], Callable[([Any], str)])] = defaultdict((lambda : default_display_func))

    def _repr_html_(self):
        '\n        Hooks into Jupyter notebook rich display system.\n        '
        return self.render()

    @doc(NDFrame.to_excel, klass='Styler', storage_options=generic._shared_docs['storage_options'])
    def to_excel(self, excel_writer, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None):
        from pandas.io.formats.excel import ExcelFormatter
        formatter = ExcelFormatter(self, na_rep=na_rep, cols=columns, header=header, float_format=float_format, index=index, index_label=index_label, merge_cells=merge_cells, inf_rep=inf_rep)
        formatter.write(excel_writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol, freeze_panes=freeze_panes, engine=engine)

    def _translate(self):
        '\n        Convert the DataFrame in `self.data` and the attrs from `_build_styles`\n        into a dictionary of {head, body, uuid, cellstyle}.\n        '
        table_styles = (self.table_styles or [])
        caption = self.caption
        ctx = self.ctx
        precision = self.precision
        hidden_index = self.hidden_index
        hidden_columns = self.hidden_columns
        uuid = self.uuid
        ROW_HEADING_CLASS = 'row_heading'
        COL_HEADING_CLASS = 'col_heading'
        INDEX_NAME_CLASS = 'index_name'
        DATA_CLASS = 'data'
        BLANK_CLASS = 'blank'
        BLANK_VALUE = ''

        def format_attr(pair):
            return f"{pair['key']}={pair['value']}"
        idx_lengths = _get_level_lengths(self.index)
        col_lengths = _get_level_lengths(self.columns, hidden_columns)
        cell_context = self.cell_context
        n_rlvls = self.data.index.nlevels
        n_clvls = self.data.columns.nlevels
        rlabels = self.data.index.tolist()
        clabels = self.data.columns.tolist()
        if (n_rlvls == 1):
            rlabels = [[x] for x in rlabels]
        if (n_clvls == 1):
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels))
        cellstyle_map = defaultdict(list)
        head = []
        for r in range(n_clvls):
            row_es = ([{'type': 'th', 'value': BLANK_VALUE, 'display_value': BLANK_VALUE, 'is_visible': (not hidden_index), 'class': ' '.join([BLANK_CLASS])}] * (n_rlvls - 1))
            name = self.data.columns.names[r]
            cs = [(BLANK_CLASS if (name is None) else INDEX_NAME_CLASS), f'level{r}']
            name = (BLANK_VALUE if (name is None) else name)
            row_es.append({'type': 'th', 'value': name, 'display_value': name, 'class': ' '.join(cs), 'is_visible': (not hidden_index)})
            if clabels:
                for (c, value) in enumerate(clabels[r]):
                    cs = [COL_HEADING_CLASS, f'level{r}', f'col{c}']
                    cs.extend(cell_context.get('col_headings', {}).get(r, {}).get(c, []))
                    es = {'type': 'th', 'value': value, 'display_value': value, 'class': ' '.join(cs), 'is_visible': _is_visible(c, r, col_lengths)}
                    colspan = col_lengths.get((r, c), 0)
                    if (colspan > 1):
                        es['attributes'] = [format_attr({'key': 'colspan', 'value': f'"{colspan}"'})]
                    row_es.append(es)
                head.append(row_es)
        if (self.data.index.names and com.any_not_none(*self.data.index.names) and (not hidden_index)):
            index_header_row = []
            for (c, name) in enumerate(self.data.index.names):
                cs = [INDEX_NAME_CLASS, f'level{c}']
                name = ('' if (name is None) else name)
                index_header_row.append({'type': 'th', 'value': name, 'class': ' '.join(cs)})
            index_header_row.extend(([{'type': 'th', 'value': BLANK_VALUE, 'class': ' '.join([BLANK_CLASS])}] * (len(clabels[0]) - len(hidden_columns))))
            head.append(index_header_row)
        body = []
        for (r, idx) in enumerate(self.data.index):
            row_es = []
            for (c, value) in enumerate(rlabels[r]):
                rid = [ROW_HEADING_CLASS, f'level{c}', f'row{r}']
                es = {'type': 'th', 'is_visible': (_is_visible(r, c, idx_lengths) and (not hidden_index)), 'value': value, 'display_value': value, 'id': '_'.join(rid[1:]), 'class': ' '.join(rid)}
                rowspan = idx_lengths.get((c, r), 0)
                if (rowspan > 1):
                    es['attributes'] = [format_attr({'key': 'rowspan', 'value': f'"{rowspan}"'})]
                row_es.append(es)
            for (c, col) in enumerate(self.data.columns):
                cs = [DATA_CLASS, f'row{r}', f'col{c}']
                cs.extend(cell_context.get('data', {}).get(r, {}).get(c, []))
                formatter = self._display_funcs[(r, c)]
                value = self.data.iloc[(r, c)]
                row_dict = {'type': 'td', 'value': value, 'class': ' '.join(cs), 'display_value': formatter(value), 'is_visible': (c not in hidden_columns)}
                props = []
                if (self.cell_ids or ((r, c) in ctx)):
                    row_dict['id'] = '_'.join(cs[1:])
                    for x in ctx[(r, c)]:
                        if x.count(':'):
                            props.append(tuple(x.split(':')))
                        else:
                            props.append(('', ''))
                row_es.append(row_dict)
                cellstyle_map[tuple(props)].append(f'row{r}_col{c}')
            body.append(row_es)
        cellstyle = [{'props': list(props), 'selectors': selectors} for (props, selectors) in cellstyle_map.items()]
        table_attr = self.table_attributes
        use_mathjax = get_option('display.html.use_mathjax')
        if (not use_mathjax):
            table_attr = (table_attr or '')
            if ('class="' in table_attr):
                table_attr = table_attr.replace('class="', 'class="tex2jax_ignore ')
            else:
                table_attr += ' class="tex2jax_ignore"'
        return {'head': head, 'cellstyle': cellstyle, 'body': body, 'uuid': uuid, 'precision': precision, 'table_styles': table_styles, 'caption': caption, 'table_attributes': table_attr}

    def format(self, formatter, subset=None, na_rep=None):
        '\n        Format the text display value of cells.\n\n        Parameters\n        ----------\n        formatter : str, callable, dict or None\n            If ``formatter`` is None, the default formatter is used.\n        subset : IndexSlice\n            An argument to ``DataFrame.loc`` that restricts which elements\n            ``formatter`` is applied to.\n        na_rep : str, optional\n            Representation for missing values.\n            If ``na_rep`` is None, no special formatting is applied.\n\n            .. versionadded:: 1.0.0\n\n        Returns\n        -------\n        self : Styler\n\n        Notes\n        -----\n        ``formatter`` is either an ``a`` or a dict ``{column name: a}`` where\n        ``a`` is one of\n\n        - str: this will be wrapped in: ``a.format(x)``\n        - callable: called with the value of an individual cell\n\n        The default display value for numeric values is the "general" (``g``)\n        format with ``pd.options.display.precision`` precision.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(np.random.randn(4, 2), columns=[\'a\', \'b\'])\n        >>> df.style.format("{:.2%}")\n        >>> df[\'c\'] = [\'a\', \'b\', \'c\', \'d\']\n        >>> df.style.format({\'c\': str.upper})\n        '
        if (formatter is None):
            assert (self._display_funcs.default_factory is not None)
            formatter = self._display_funcs.default_factory()
        if (subset is None):
            row_locs = range(len(self.data))
            col_locs = range(len(self.data.columns))
        else:
            subset = non_reducing_slice(subset)
            if (len(subset) == 1):
                subset = (subset, self.data.columns)
            sub_df = self.data.loc[subset]
            row_locs = self.data.index.get_indexer_for(sub_df.index)
            col_locs = self.data.columns.get_indexer_for(sub_df.columns)
        if is_dict_like(formatter):
            for (col, col_formatter) in formatter.items():
                col_formatter = _maybe_wrap_formatter(col_formatter, na_rep)
                col_num = self.data.columns.get_indexer_for([col])[0]
                for row_num in row_locs:
                    self._display_funcs[(row_num, col_num)] = col_formatter
        else:
            formatter = _maybe_wrap_formatter(formatter, na_rep)
            locs = product(*(row_locs, col_locs))
            for (i, j) in locs:
                self._display_funcs[(i, j)] = formatter
        return self

    def set_td_classes(self, classes):
        '\n        Add string based CSS class names to data cells that will appear within the\n        `Styler` HTML result. These classes are added within specified `<td>` elements.\n\n        Parameters\n        ----------\n        classes : DataFrame\n            DataFrame containing strings that will be translated to CSS classes,\n            mapped by identical column and index values that must exist on the\n            underlying `Styler` data. None, NaN values, and empty strings will\n            be ignored and not affect the rendered HTML.\n\n        Returns\n        -------\n        self : Styler\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])\n        >>> classes = pd.DataFrame([\n        ...     ["min-val red", "", "blue"],\n        ...     ["red", None, "blue max-val"]\n        ... ], index=df.index, columns=df.columns)\n        >>> df.style.set_td_classes(classes)\n\n        Using `MultiIndex` columns and a `classes` `DataFrame` as a subset of the\n        underlying,\n\n        >>> df = pd.DataFrame([[1,2],[3,4]], index=["a", "b"],\n        ...     columns=[["level0", "level0"], ["level1a", "level1b"]])\n        >>> classes = pd.DataFrame(["min-val"], index=["a"],\n        ...     columns=[["level0"],["level1a"]])\n        >>> df.style.set_td_classes(classes)\n\n        Form of the output with new additional css classes,\n\n        >>> df = pd.DataFrame([[1]])\n        >>> css = pd.DataFrame(["other-class"])\n        >>> s = Styler(df, uuid="_", cell_ids=False).set_td_classes(css)\n        >>> s.hide_index().render()\n        \'<style  type="text/css" ></style>\'\n        \'<table id="T__" >\'\n        \'  <thead>\'\n        \'    <tr><th class="col_heading level0 col0" >0</th></tr>\'\n        \'  </thead>\'\n        \'  <tbody>\'\n        \'    <tr><td  class="data row0 col0 other-class" >1</td></tr>\'\n        \'  </tbody>\'\n        \'</table>\'\n        '
        classes = classes.reindex_like(self.data)
        mask = (classes.isna() | classes.eq(''))
        self.cell_context['data'] = {r: {c: [str(classes.iloc[(r, c)])]} for (r, rn) in enumerate(classes.index) for (c, cn) in enumerate(classes.columns) if (not mask.iloc[(r, c)])}
        return self

    def render(self, **kwargs):
        "\n        Render the built up styles to HTML.\n\n        Parameters\n        ----------\n        **kwargs\n            Any additional keyword arguments are passed\n            through to ``self.template.render``.\n            This is useful when you need to provide\n            additional variables for a custom template.\n\n        Returns\n        -------\n        rendered : str\n            The rendered HTML.\n\n        Notes\n        -----\n        ``Styler`` objects have defined the ``_repr_html_`` method\n        which automatically calls ``self.render()`` when it's the\n        last item in a Notebook cell. When calling ``Styler.render()``\n        directly, wrap the result in ``IPython.display.HTML`` to view\n        the rendered HTML in the notebook.\n\n        Pandas uses the following keys in render. Arguments passed\n        in ``**kwargs`` take precedence, so think carefully if you want\n        to override them:\n\n        * head\n        * cellstyle\n        * body\n        * uuid\n        * precision\n        * table_styles\n        * caption\n        * table_attributes\n        "
        self._compute()
        d = self._translate()
        trimmed = [x for x in d['cellstyle'] if any((any(y) for y in x['props']))]
        d['cellstyle'] = trimmed
        d.update(kwargs)
        return self.template.render(**d)

    def _update_ctx(self, attrs):
        "\n        Update the state of the Styler.\n\n        Collects a mapping of {index_label: ['<property>: <value>']}.\n\n        Parameters\n        ----------\n        attrs : DataFrame\n            should contain strings of '<property>: <value>;<prop2>: <val2>'\n            Whitespace shouldn't matter and the final trailing ';' shouldn't\n            matter.\n        "
        coli = {k: i for (i, k) in enumerate(self.columns)}
        rowi = {k: i for (i, k) in enumerate(self.index)}
        for jj in range(len(attrs.columns)):
            cn = attrs.columns[jj]
            j = coli[cn]
            for (rn, c) in attrs[[cn]].itertuples():
                if (not c):
                    continue
                c = c.rstrip(';')
                if (not c):
                    continue
                i = rowi[rn]
                for pair in c.split(';'):
                    self.ctx[(i, j)].append(pair)

    def _copy(self, deepcopy=False):
        styler = Styler(self.data, precision=self.precision, caption=self.caption, uuid=self.uuid, table_styles=self.table_styles, na_rep=self.na_rep)
        if deepcopy:
            styler.ctx = copy.deepcopy(self.ctx)
            styler._todo = copy.deepcopy(self._todo)
        else:
            styler.ctx = self.ctx
            styler._todo = self._todo
        return styler

    def __copy__(self):
        '\n        Deep copy by default.\n        '
        return self._copy(deepcopy=False)

    def __deepcopy__(self, memo):
        return self._copy(deepcopy=True)

    def clear(self):
        '\n        Reset the styler, removing any previously applied styles.\n\n        Returns None.\n        '
        self.ctx.clear()
        self.cell_context = {}
        self._todo = []

    def _compute(self):
        '\n        Execute the style functions built up in `self._todo`.\n\n        Relies on the conventions that all style functions go through\n        .apply or .applymap. The append styles to apply as tuples of\n\n        (application method, *args, **kwargs)\n        '
        r = self
        for (func, args, kwargs) in self._todo:
            r = func(self)(*args, **kwargs)
        return r

    def _apply(self, func, axis=0, subset=None, **kwargs):
        subset = (slice(None) if (subset is None) else subset)
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]
        if (axis is not None):
            result = data.apply(func, axis=axis, result_type='expand', **kwargs)
            result.columns = data.columns
        else:
            result = func(data, **kwargs)
            if (not isinstance(result, pd.DataFrame)):
                raise TypeError(f'Function {repr(func)} must return a DataFrame when passed to `Styler.apply` with axis=None')
            if (not (result.index.equals(data.index) and result.columns.equals(data.columns))):
                raise ValueError(f'Result of {repr(func)} must have identical index and columns as the input')
        result_shape = result.shape
        expected_shape = self.data.loc[subset].shape
        if (result_shape != expected_shape):
            raise ValueError(f'''Function {repr(func)} returned the wrong shape.
Result has shape: {result.shape}
Expected shape:   {expected_shape}''')
        self._update_ctx(result)
        return self

    def apply(self, func, axis=0, subset=None, **kwargs):
        "\n        Apply a function column-wise, row-wise, or table-wise.\n\n        Updates the HTML representation with the result.\n\n        Parameters\n        ----------\n        func : function\n            ``func`` should take a Series or DataFrame (depending\n            on ``axis``), and return an object with the same shape.\n            Must return a DataFrame with identical index and\n            column labels when ``axis=None``.\n        axis : {0 or 'index', 1 or 'columns', None}, default 0\n            Apply to each column (``axis=0`` or ``'index'``), to each row\n            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once\n            with ``axis=None``.\n        subset : IndexSlice\n            A valid indexer to limit ``data`` to *before* applying the\n            function. Consider using a pandas.IndexSlice.\n        **kwargs : dict\n            Pass along to ``func``.\n\n        Returns\n        -------\n        self : Styler\n\n        Notes\n        -----\n        The output shape of ``func`` should match the input, i.e. if\n        ``x`` is the input row, column, or table (depending on ``axis``),\n        then ``func(x).shape == x.shape`` should be true.\n\n        This is similar to ``DataFrame.apply``, except that ``axis=None``\n        applies the function to the entire DataFrame at once,\n        rather than column-wise or row-wise.\n\n        Examples\n        --------\n        >>> def highlight_max(x):\n        ...     return ['background-color: yellow' if v == x.max() else ''\n                        for v in x]\n        ...\n        >>> df = pd.DataFrame(np.random.randn(5, 2))\n        >>> df.style.apply(highlight_max)\n        "
        self._todo.append(((lambda instance: getattr(instance, '_apply')), (func, axis, subset), kwargs))
        return self

    def _applymap(self, func, subset=None, **kwargs):
        func = partial(func, **kwargs)
        if (subset is None):
            subset = pd.IndexSlice[:]
        subset = non_reducing_slice(subset)
        result = self.data.loc[subset].applymap(func)
        self._update_ctx(result)
        return self

    def applymap(self, func, subset=None, **kwargs):
        '\n        Apply a function elementwise.\n\n        Updates the HTML representation with the result.\n\n        Parameters\n        ----------\n        func : function\n            ``func`` should take a scalar and return a scalar.\n        subset : IndexSlice\n            A valid indexer to limit ``data`` to *before* applying the\n            function. Consider using a pandas.IndexSlice.\n        **kwargs : dict\n            Pass along to ``func``.\n\n        Returns\n        -------\n        self : Styler\n\n        See Also\n        --------\n        Styler.where: Updates the HTML representation with a style which is\n            selected in accordance with the return value of a function.\n        '
        self._todo.append(((lambda instance: getattr(instance, '_applymap')), (func, subset), kwargs))
        return self

    def where(self, cond, value, other=None, subset=None, **kwargs):
        '\n        Apply a function elementwise.\n\n        Updates the HTML representation with a style which is\n        selected in accordance with the return value of a function.\n\n        Parameters\n        ----------\n        cond : callable\n            ``cond`` should take a scalar and return a boolean.\n        value : str\n            Applied when ``cond`` returns true.\n        other : str\n            Applied when ``cond`` returns false.\n        subset : IndexSlice\n            A valid indexer to limit ``data`` to *before* applying the\n            function. Consider using a pandas.IndexSlice.\n        **kwargs : dict\n            Pass along to ``cond``.\n\n        Returns\n        -------\n        self : Styler\n\n        See Also\n        --------\n        Styler.applymap: Updates the HTML representation with the result.\n        '
        if (other is None):
            other = ''
        return self.applymap((lambda val: (value if cond(val) else other)), subset=subset, **kwargs)

    def set_precision(self, precision):
        '\n        Set the precision used to render.\n\n        Parameters\n        ----------\n        precision : int\n\n        Returns\n        -------\n        self : Styler\n        '
        self.precision = precision
        return self

    def set_table_attributes(self, attributes):
        '\n        Set the table attributes.\n\n        These are the items that show up in the opening ``<table>`` tag\n        in addition to automatic (by default) id.\n\n        Parameters\n        ----------\n        attributes : str\n\n        Returns\n        -------\n        self : Styler\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(np.random.randn(10, 4))\n        >>> df.style.set_table_attributes(\'class="pure-table"\')\n        # ... <table class="pure-table"> ...\n        '
        self.table_attributes = attributes
        return self

    def export(self):
        '\n        Export the styles to applied to the current Styler.\n\n        Can be applied to a second style with ``Styler.use``.\n\n        Returns\n        -------\n        styles : list\n\n        See Also\n        --------\n        Styler.use: Set the styles on the current Styler.\n        '
        return self._todo

    def use(self, styles):
        '\n        Set the styles on the current Styler.\n\n        Possibly uses styles from ``Styler.export``.\n\n        Parameters\n        ----------\n        styles : list\n            List of style functions.\n\n        Returns\n        -------\n        self : Styler\n\n        See Also\n        --------\n        Styler.export : Export the styles to applied to the current Styler.\n        '
        self._todo.extend(styles)
        return self

    def set_uuid(self, uuid):
        '\n        Set the uuid for a Styler.\n\n        Parameters\n        ----------\n        uuid : str\n\n        Returns\n        -------\n        self : Styler\n        '
        self.uuid = uuid
        return self

    def set_caption(self, caption):
        '\n        Set the caption on a Styler.\n\n        Parameters\n        ----------\n        caption : str\n\n        Returns\n        -------\n        self : Styler\n        '
        self.caption = caption
        return self

    def set_table_styles(self, table_styles, axis=0, overwrite=True):
        "\n        Set the table styles on a Styler.\n\n        These are placed in a ``<style>`` tag before the generated HTML table.\n\n        This function can be used to style the entire table, columns, rows or\n        specific HTML selectors.\n\n        Parameters\n        ----------\n        table_styles : list or dict\n            If supplying a list, each individual table_style should be a\n            dictionary with ``selector`` and ``props`` keys. ``selector``\n            should be a CSS selector that the style will be applied to\n            (automatically prefixed by the table's UUID) and ``props``\n            should be a list of tuples with ``(attribute, value)``.\n            If supplying a dict, the dict keys should correspond to\n            column names or index values, depending upon the specified\n            `axis` argument. These will be mapped to row or col CSS\n            selectors. MultiIndex values as dict keys should be\n            in their respective tuple form. The dict values should be\n            a list as specified in the form with CSS selectors and\n            props that will be applied to the specified row or column.\n\n            .. versionchanged:: 1.2.0\n\n        axis : {0 or 'index', 1 or 'columns', None}, default 0\n            Apply to each column (``axis=0`` or ``'index'``), to each row\n            (``axis=1`` or ``'columns'``). Only used if `table_styles` is\n            dict.\n\n            .. versionadded:: 1.2.0\n\n        overwrite : boolean, default True\n            Styles are replaced if `True`, or extended if `False`. CSS\n            rules are preserved so most recent styles set will dominate\n            if selectors intersect.\n\n            .. versionadded:: 1.2.0\n\n        Returns\n        -------\n        self : Styler\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(np.random.randn(10, 4),\n        ...                   columns=['A', 'B', 'C', 'D'])\n        >>> df.style.set_table_styles(\n        ...     [{'selector': 'tr:hover',\n        ...       'props': [('background-color', 'yellow')]}]\n        ... )\n\n        Adding column styling by name\n\n        >>> df.style.set_table_styles({\n        ...     'A': [{'selector': '',\n        ...            'props': [('color', 'red')]}],\n        ...     'B': [{'selector': 'td',\n        ...            'props': [('color', 'blue')]}]\n        ... }, overwrite=False)\n\n        Adding row styling\n\n        >>> df.style.set_table_styles({\n        ...     0: [{'selector': 'td:hover',\n        ...          'props': [('font-size', '25px')]}]\n        ... }, axis=1, overwrite=False)\n        "
        if is_dict_like(table_styles):
            if (axis in [0, 'index']):
                (obj, idf) = (self.data.columns, '.col')
            else:
                (obj, idf) = (self.data.index, '.row')
            table_styles = [{'selector': ((s['selector'] + idf) + str(obj.get_loc(key))), 'props': s['props']} for (key, styles) in table_styles.items() for s in styles]
        if ((not overwrite) and (self.table_styles is not None)):
            self.table_styles.extend(table_styles)
        else:
            self.table_styles = table_styles
        return self

    def set_na_rep(self, na_rep):
        '\n        Set the missing data representation on a Styler.\n\n        .. versionadded:: 1.0.0\n\n        Parameters\n        ----------\n        na_rep : str\n\n        Returns\n        -------\n        self : Styler\n        '
        self.na_rep = na_rep
        return self

    def hide_index(self):
        '\n        Hide any indices from rendering.\n\n        Returns\n        -------\n        self : Styler\n        '
        self.hidden_index = True
        return self

    def hide_columns(self, subset):
        '\n        Hide columns from rendering.\n\n        Parameters\n        ----------\n        subset : IndexSlice\n            An argument to ``DataFrame.loc`` that identifies which columns\n            are hidden.\n\n        Returns\n        -------\n        self : Styler\n        '
        subset = non_reducing_slice(subset)
        hidden_df = self.data.loc[subset]
        self.hidden_columns = self.columns.get_indexer_for(hidden_df.columns)
        return self

    @staticmethod
    def _highlight_null(v, null_color):
        return (f'background-color: {null_color}' if pd.isna(v) else '')

    def highlight_null(self, null_color='red', subset=None):
        "\n        Shade the background ``null_color`` for missing values.\n\n        Parameters\n        ----------\n        null_color : str, default 'red'\n        subset : label or list of labels, default None\n            A valid slice for ``data`` to limit the style application to.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        self : Styler\n        "
        self.applymap(self._highlight_null, null_color=null_color, subset=subset)
        return self

    def background_gradient(self, cmap='PuBu', low=0, high=0, axis=0, subset=None, text_color_threshold=0.408, vmin=None, vmax=None):
        "\n        Color the background in a gradient style.\n\n        The background color is determined according\n        to the data in each column (optionally row). Requires matplotlib.\n\n        Parameters\n        ----------\n        cmap : str or colormap\n            Matplotlib colormap.\n        low : float\n            Compress the range by the low.\n        high : float\n            Compress the range by the high.\n        axis : {0 or 'index', 1 or 'columns', None}, default 0\n            Apply to each column (``axis=0`` or ``'index'``), to each row\n            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once\n            with ``axis=None``.\n        subset : IndexSlice\n            A valid slice for ``data`` to limit the style application to.\n        text_color_threshold : float or int\n            Luminance threshold for determining text color. Facilitates text\n            visibility across varying background colors. From 0 to 1.\n            0 = all text is dark colored, 1 = all text is light colored.\n\n            .. versionadded:: 0.24.0\n\n        vmin : float, optional\n            Minimum data value that corresponds to colormap minimum value.\n            When None (default): the minimum value of the data will be used.\n\n            .. versionadded:: 1.0.0\n\n        vmax : float, optional\n            Maximum data value that corresponds to colormap maximum value.\n            When None (default): the maximum value of the data will be used.\n\n            .. versionadded:: 1.0.0\n\n        Returns\n        -------\n        self : Styler\n\n        Raises\n        ------\n        ValueError\n            If ``text_color_threshold`` is not a value from 0 to 1.\n\n        Notes\n        -----\n        Set ``text_color_threshold`` or tune ``low`` and ``high`` to keep the\n        text legible by not using the entire range of the color map. The range\n        of the data is extended by ``low * (x.max() - x.min())`` and ``high *\n        (x.max() - x.min())`` before normalizing.\n        "
        subset = maybe_numeric_slice(self.data, subset)
        subset = non_reducing_slice(subset)
        self.apply(self._background_gradient, cmap=cmap, subset=subset, axis=axis, low=low, high=high, text_color_threshold=text_color_threshold, vmin=vmin, vmax=vmax)
        return self

    @staticmethod
    def _background_gradient(s, cmap='PuBu', low=0, high=0, text_color_threshold=0.408, vmin=None, vmax=None):
        '\n        Color background in a range according to the data.\n        '
        if ((not isinstance(text_color_threshold, (float, int))) or (not (0 <= text_color_threshold <= 1))):
            msg = '`text_color_threshold` must be a value from 0 to 1.'
            raise ValueError(msg)
        with _mpl(Styler.background_gradient) as (plt, colors):
            smin = (np.nanmin(s.to_numpy()) if (vmin is None) else vmin)
            smax = (np.nanmax(s.to_numpy()) if (vmax is None) else vmax)
            rng = (smax - smin)
            norm = colors.Normalize((smin - (rng * low)), (smax + (rng * high)))
            rgbas = plt.cm.get_cmap(cmap)(norm(s.to_numpy(dtype=float)))

            def relative_luminance(rgba) -> float:
                '\n                Calculate relative luminance of a color.\n\n                The calculation adheres to the W3C standards\n                (https://www.w3.org/WAI/GL/wiki/Relative_luminance)\n\n                Parameters\n                ----------\n                color : rgb or rgba tuple\n\n                Returns\n                -------\n                float\n                    The relative luminance as a value from 0 to 1\n                '
                (r, g, b) = (((x / 12.92) if (x <= 0.03928) else ((x + 0.055) / (1.055 ** 2.4))) for x in rgba[:3])
                return (((0.2126 * r) + (0.7152 * g)) + (0.0722 * b))

            def css(rgba) -> str:
                dark = (relative_luminance(rgba) < text_color_threshold)
                text_color = ('#f1f1f1' if dark else '#000000')
                return f'background-color: {colors.rgb2hex(rgba)};color: {text_color};'
            if (s.ndim == 1):
                return [css(rgba) for rgba in rgbas]
            else:
                return pd.DataFrame([[css(rgba) for rgba in row] for row in rgbas], index=s.index, columns=s.columns)

    def set_properties(self, subset=None, **kwargs):
        '\n        Method to set one or more non-data dependent properties or each cell.\n\n        Parameters\n        ----------\n        subset : IndexSlice\n            A valid slice for ``data`` to limit the style application to.\n        **kwargs : dict\n            A dictionary of property, value pairs to be set for each cell.\n\n        Returns\n        -------\n        self : Styler\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(np.random.randn(10, 4))\n        >>> df.style.set_properties(color="white", align="right")\n        >>> df.style.set_properties(**{\'background-color\': \'yellow\'})\n        '
        values = ';'.join((f'{p}: {v}' for (p, v) in kwargs.items()))
        f = (lambda x: values)
        return self.applymap(f, subset=subset)

    @staticmethod
    def _bar(s, align, colors, width=100, vmin=None, vmax=None):
        '\n        Draw bar chart in dataframe cells.\n        '
        smin = (np.nanmin(s.to_numpy()) if (vmin is None) else vmin)
        smax = (np.nanmax(s.to_numpy()) if (vmax is None) else vmax)
        if (align == 'mid'):
            smin = min(0, smin)
            smax = max(0, smax)
        elif (align == 'zero'):
            smax = max(abs(smin), abs(smax))
            smin = (- smax)
        normed = ((width * (s.to_numpy(dtype=float) - smin)) / ((smax - smin) + 1e-12))
        zero = (((- width) * smin) / ((smax - smin) + 1e-12))

        def css_bar(start: float, end: float, color: str) -> str:
            '\n            Generate CSS code to draw a bar from start to end.\n            '
            css = 'width: 10em; height: 80%;'
            if (end > start):
                css += 'background: linear-gradient(90deg,'
                if (start > 0):
                    css += f' transparent {start:.1f}%, {color} {start:.1f}%, '
                e = min(end, width)
                css += f'{color} {e:.1f}%, transparent {e:.1f}%)'
            return css

        def css(x):
            if pd.isna(x):
                return ''
            color = (colors[1] if (x > zero) else colors[0])
            if (align == 'left'):
                return css_bar(0, x, color)
            else:
                return css_bar(min(x, zero), max(x, zero), color)
        if (s.ndim == 1):
            return [css(x) for x in normed]
        else:
            return pd.DataFrame([[css(x) for x in row] for row in normed], index=s.index, columns=s.columns)

    def bar(self, subset=None, axis=0, color='#d65f5f', width=100, align='left', vmin=None, vmax=None):
        "\n        Draw bar chart in the cell backgrounds.\n\n        Parameters\n        ----------\n        subset : IndexSlice, optional\n            A valid slice for `data` to limit the style application to.\n        axis : {0 or 'index', 1 or 'columns', None}, default 0\n            Apply to each column (``axis=0`` or ``'index'``), to each row\n            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once\n            with ``axis=None``.\n        color : str or 2-tuple/list\n            If a str is passed, the color is the same for both\n            negative and positive numbers. If 2-tuple/list is used, the\n            first element is the color_negative and the second is the\n            color_positive (eg: ['#d65f5f', '#5fba7d']).\n        width : float, default 100\n            A number between 0 or 100. The largest value will cover `width`\n            percent of the cell's width.\n        align : {'left', 'zero',' mid'}, default 'left'\n            How to align the bars with the cells.\n\n            - 'left' : the min value starts at the left of the cell.\n            - 'zero' : a value of zero is located at the center of the cell.\n            - 'mid' : the center of the cell is at (max-min)/2, or\n              if values are all negative (positive) the zero is aligned\n              at the right (left) of the cell.\n        vmin : float, optional\n            Minimum bar value, defining the left hand limit\n            of the bar drawing range, lower values are clipped to `vmin`.\n            When None (default): the minimum value of the data will be used.\n\n            .. versionadded:: 0.24.0\n\n        vmax : float, optional\n            Maximum bar value, defining the right hand limit\n            of the bar drawing range, higher values are clipped to `vmax`.\n            When None (default): the maximum value of the data will be used.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        self : Styler\n        "
        if (align not in ('left', 'zero', 'mid')):
            raise ValueError("`align` must be one of {'left', 'zero',' mid'}")
        if (not is_list_like(color)):
            color = [color, color]
        elif (len(color) == 1):
            color = [color[0], color[0]]
        elif (len(color) > 2):
            raise ValueError("`color` must be string or a list-like of length 2: [`color_neg`, `color_pos`] (eg: color=['#d65f5f', '#5fba7d'])")
        subset = maybe_numeric_slice(self.data, subset)
        subset = non_reducing_slice(subset)
        self.apply(self._bar, subset=subset, axis=axis, align=align, colors=color, width=width, vmin=vmin, vmax=vmax)
        return self

    def highlight_max(self, subset=None, color='yellow', axis=0):
        "\n        Highlight the maximum by shading the background.\n\n        Parameters\n        ----------\n        subset : IndexSlice, default None\n            A valid slice for ``data`` to limit the style application to.\n        color : str, default 'yellow'\n        axis : {0 or 'index', 1 or 'columns', None}, default 0\n            Apply to each column (``axis=0`` or ``'index'``), to each row\n            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once\n            with ``axis=None``.\n\n        Returns\n        -------\n        self : Styler\n        "
        return self._highlight_handler(subset=subset, color=color, axis=axis, max_=True)

    def highlight_min(self, subset=None, color='yellow', axis=0):
        "\n        Highlight the minimum by shading the background.\n\n        Parameters\n        ----------\n        subset : IndexSlice, default None\n            A valid slice for ``data`` to limit the style application to.\n        color : str, default 'yellow'\n        axis : {0 or 'index', 1 or 'columns', None}, default 0\n            Apply to each column (``axis=0`` or ``'index'``), to each row\n            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once\n            with ``axis=None``.\n\n        Returns\n        -------\n        self : Styler\n        "
        return self._highlight_handler(subset=subset, color=color, axis=axis, max_=False)

    def _highlight_handler(self, subset=None, color='yellow', axis=None, max_=True):
        subset = non_reducing_slice(maybe_numeric_slice(self.data, subset))
        self.apply(self._highlight_extrema, color=color, axis=axis, subset=subset, max_=max_)
        return self

    @staticmethod
    def _highlight_extrema(data, color='yellow', max_=True):
        '\n        Highlight the min or max in a Series or DataFrame.\n        '
        attr = f'background-color: {color}'
        if max_:
            extrema = (data == np.nanmax(data.to_numpy()))
        else:
            extrema = (data == np.nanmin(data.to_numpy()))
        if (data.ndim == 1):
            return [(attr if v else '') for v in extrema]
        else:
            return pd.DataFrame(np.where(extrema, attr, ''), index=data.index, columns=data.columns)

    @classmethod
    def from_custom_template(cls, searchpath, name):
        '\n        Factory function for creating a subclass of ``Styler``.\n\n        Uses a custom template and Jinja environment.\n\n        Parameters\n        ----------\n        searchpath : str or list\n            Path or paths of directories containing the templates.\n        name : str\n            Name of your custom template to use for rendering.\n\n        Returns\n        -------\n        MyStyler : subclass of Styler\n            Has the correct ``env`` and ``template`` class attributes set.\n        '
        loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(searchpath), cls.loader])

        class MyStyler(cls):
            env = jinja2.Environment(loader=loader)
            template = env.get_template(name)
        return MyStyler

    def pipe(self, func, *args, **kwargs):
        '\n        Apply ``func(self, *args, **kwargs)``, and return the result.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        func : function\n            Function to apply to the Styler.  Alternatively, a\n            ``(callable, keyword)`` tuple where ``keyword`` is a string\n            indicating the keyword of ``callable`` that expects the Styler.\n        *args : optional\n            Arguments passed to `func`.\n        **kwargs : optional\n            A dictionary of keyword arguments passed into ``func``.\n\n        Returns\n        -------\n        object :\n            The value returned by ``func``.\n\n        See Also\n        --------\n        DataFrame.pipe : Analogous method for DataFrame.\n        Styler.apply : Apply a function row-wise, column-wise, or table-wise to\n            modify the dataframe\'s styling.\n\n        Notes\n        -----\n        Like :meth:`DataFrame.pipe`, this method can simplify the\n        application of several user-defined functions to a styler.  Instead\n        of writing:\n\n        .. code-block:: python\n\n            f(g(df.style.set_precision(3), arg1=a), arg2=b, arg3=c)\n\n        users can write:\n\n        .. code-block:: python\n\n            (df.style.set_precision(3)\n               .pipe(g, arg1=a)\n               .pipe(f, arg2=b, arg3=c))\n\n        In particular, this allows users to define functions that take a\n        styler object, along with other parameters, and return the styler after\n        making styling changes (such as calling :meth:`Styler.apply` or\n        :meth:`Styler.set_properties`).  Using ``.pipe``, these user-defined\n        style "transformations" can be interleaved with calls to the built-in\n        Styler interface.\n\n        Examples\n        --------\n        >>> def format_conversion(styler):\n        ...     return (styler.set_properties(**{\'text-align\': \'right\'})\n        ...                   .format({\'conversion\': \'{:.1%}\'}))\n\n        The user-defined ``format_conversion`` function above can be called\n        within a sequence of other style modifications:\n\n        >>> df = pd.DataFrame({\'trial\': list(range(5)),\n        ...                    \'conversion\': [0.75, 0.85, np.nan, 0.7, 0.72]})\n        >>> (df.style\n        ...    .highlight_min(subset=[\'conversion\'], color=\'yellow\')\n        ...    .pipe(format_conversion)\n        ...    .set_caption("Results with minimum conversion highlighted."))\n        '
        return com.pipe(self, func, *args, **kwargs)

def _is_visible(idx_row, idx_col, lengths):
    '\n    Index -> {(idx_row, idx_col): bool}).\n    '
    return ((idx_col, idx_row) in lengths)

def _get_level_lengths(index, hidden_elements=None):
    '\n    Given an index, find the level length for each element.\n\n    Optional argument is a list of index positions which\n    should not be visible.\n\n    Result is a dictionary of (level, initial_position): span\n    '
    if isinstance(index, pd.MultiIndex):
        levels = index.format(sparsify=lib.no_default, adjoin=False)
    else:
        levels = index.format()
    if (hidden_elements is None):
        hidden_elements = []
    lengths = {}
    if (index.nlevels == 1):
        for (i, value) in enumerate(levels):
            if (i not in hidden_elements):
                lengths[(0, i)] = 1
        return lengths
    for (i, lvl) in enumerate(levels):
        for (j, row) in enumerate(lvl):
            if (not get_option('display.multi_sparse')):
                lengths[(i, j)] = 1
            elif ((row is not lib.no_default) and (j not in hidden_elements)):
                last_label = j
                lengths[(i, last_label)] = 1
            elif (row is not lib.no_default):
                last_label = j
                lengths[(i, last_label)] = 0
            elif (j not in hidden_elements):
                lengths[(i, last_label)] += 1
    non_zero_lengths = {element: length for (element, length) in lengths.items() if (length >= 1)}
    return non_zero_lengths

def _maybe_wrap_formatter(formatter, na_rep):
    if isinstance(formatter, str):
        formatter_func = (lambda x: formatter.format(x))
    elif callable(formatter):
        formatter_func = formatter
    else:
        msg = f'Expected a template string or callable, got {formatter} instead'
        raise TypeError(msg)
    if (na_rep is None):
        return formatter_func
    elif isinstance(na_rep, str):
        return (lambda x: (na_rep if pd.isna(x) else formatter_func(x)))
    else:
        msg = f'Expected a string, got {na_rep} instead'
        raise TypeError(msg)
