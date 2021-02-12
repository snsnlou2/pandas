
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast
import numpy as np
from pandas._typing import FrameOrSeriesUnion, Label
from pandas.util._decorators import Appender, Substitution
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import is_integer_dtype, is_list_like, is_scalar
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
import pandas.core.common as com
from pandas.core.frame import _shared_docs
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import Index, MultiIndex, get_objs_combined_axis
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import cartesian_product
from pandas.core.series import Series
if TYPE_CHECKING:
    from pandas import DataFrame

@Substitution('\ndata : DataFrame')
@Appender(_shared_docs['pivot_table'], indents=1)
def pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False):
    index = _convert_by(index)
    columns = _convert_by(columns)
    if isinstance(aggfunc, list):
        pieces: List[DataFrame] = []
        keys = []
        for func in aggfunc:
            table = pivot_table(data, values=values, index=index, columns=columns, fill_value=fill_value, aggfunc=func, margins=margins, dropna=dropna, margins_name=margins_name, observed=observed)
            pieces.append(table)
            keys.append(getattr(func, '__name__', func))
        return concat(pieces, keys=keys, axis=1)
    keys = (index + columns)
    values_passed = (values is not None)
    if values_passed:
        if is_list_like(values):
            values_multi = True
            values = list(values)
        else:
            values_multi = False
            values = [values]
        for i in values:
            if (i not in data):
                raise KeyError(i)
        to_filter = []
        for x in (keys + values):
            if isinstance(x, Grouper):
                x = x.key
            try:
                if (x in data):
                    to_filter.append(x)
            except TypeError:
                pass
        if (len(to_filter) < len(data.columns)):
            data = data[to_filter]
    else:
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)
    grouped = data.groupby(keys, observed=observed)
    agged = grouped.agg(aggfunc)
    if (dropna and isinstance(agged, ABCDataFrame) and len(agged.columns)):
        agged = agged.dropna(how='all')
        for v in values:
            if ((v in data) and is_integer_dtype(data[v]) and (v in agged) and (not is_integer_dtype(agged[v]))):
                agged[v] = maybe_downcast_to_dtype(agged[v], data[v].dtype)
    table = agged
    if ((table.index.nlevels > 1) and index):
        index_names = agged.index.names[:len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if ((name is None) or (name in index_names)):
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        table = agged.unstack(to_unstack)
    if (not dropna):
        if isinstance(table.index, MultiIndex):
            m = MultiIndex.from_arrays(cartesian_product(table.index.levels), names=table.index.names)
            table = table.reindex(m, axis=0)
        if isinstance(table.columns, MultiIndex):
            m = MultiIndex.from_arrays(cartesian_product(table.columns.levels), names=table.columns.names)
            table = table.reindex(m, axis=1)
    if isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)
    if (fill_value is not None):
        _table = table.fillna(fill_value, downcast='infer')
        assert (_table is not None)
        table = _table
    if margins:
        if dropna:
            data = data[data.notna().all(axis=1)]
        table = _add_margins(table, data, values, rows=index, cols=columns, aggfunc=aggfunc, observed=dropna, margins_name=margins_name, fill_value=fill_value)
    if (values_passed and (not values_multi) and (not table.empty) and (table.columns.nlevels > 1)):
        table = table[values[0]]
    if ((len(index) == 0) and (len(columns) > 0)):
        table = table.T
    if (isinstance(table, ABCDataFrame) and dropna):
        table = table.dropna(how='all', axis=1)
    return table

def _add_margins(table, data, values, rows, cols, aggfunc, observed=None, margins_name='All', fill_value=None):
    if (not isinstance(margins_name, str)):
        raise ValueError('margins_name argument must be a string')
    msg = f'Conflicting name "{margins_name}" in margins'
    for level in table.index.names:
        if (margins_name in table.index.get_level_values(level)):
            raise ValueError(msg)
    grand_margin = _compute_grand_margin(data, values, aggfunc, margins_name)
    if (table.ndim == 2):
        for level in table.columns.names[1:]:
            if (margins_name in table.columns.get_level_values(level)):
                raise ValueError(msg)
    key: Union[(str, Tuple[(str, ...)])]
    if (len(rows) > 1):
        key = ((margins_name,) + (('',) * (len(rows) - 1)))
    else:
        key = margins_name
    if ((not values) and isinstance(table, ABCSeries)):
        return table.append(Series({key: grand_margin[margins_name]}))
    elif values:
        marginal_result_set = _generate_marginal_results(table, data, values, rows, cols, aggfunc, observed, margins_name)
        if (not isinstance(marginal_result_set, tuple)):
            return marginal_result_set
        (result, margin_keys, row_margin) = marginal_result_set
    else:
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(table, data, rows, cols, aggfunc, observed, margins_name)
        if (not isinstance(marginal_result_set, tuple)):
            return marginal_result_set
        (result, margin_keys, row_margin) = marginal_result_set
    row_margin = row_margin.reindex(result.columns, fill_value=fill_value)
    for k in margin_keys:
        if isinstance(k, str):
            row_margin[k] = grand_margin[k]
        else:
            row_margin[k] = grand_margin[k[0]]
    from pandas import DataFrame
    margin_dummy = DataFrame(row_margin, columns=[key]).T
    row_names = result.index.names
    for dtype in set(result.dtypes):
        cols = result.select_dtypes([dtype]).columns
        margin_dummy[cols] = margin_dummy[cols].apply(maybe_downcast_to_dtype, args=(dtype,))
    result = result.append(margin_dummy)
    result.index.names = row_names
    return result

def _compute_grand_margin(data, values, aggfunc, margins_name='All'):
    if values:
        grand_margin = {}
        for (k, v) in data[values].items():
            try:
                if isinstance(aggfunc, str):
                    grand_margin[k] = getattr(v, aggfunc)()
                elif isinstance(aggfunc, dict):
                    if isinstance(aggfunc[k], str):
                        grand_margin[k] = getattr(v, aggfunc[k])()
                    else:
                        grand_margin[k] = aggfunc[k](v)
                else:
                    grand_margin[k] = aggfunc(v)
            except TypeError:
                pass
        return grand_margin
    else:
        return {margins_name: aggfunc(data.index)}

def _generate_marginal_results(table, data, values, rows, cols, aggfunc, observed, margins_name='All'):
    if (len(cols) > 0):
        table_pieces = []
        margin_keys = []

        def _all_key(key):
            return ((key, margins_name) + (('',) * (len(cols) - 1)))
        if (len(rows) > 0):
            margin = data[(rows + values)].groupby(rows, observed=observed).agg(aggfunc)
            cat_axis = 1
            for (key, piece) in table.groupby(level=0, axis=cat_axis, observed=observed):
                all_key = _all_key(key)
                piece = piece.copy()
                piece[all_key] = margin[key]
                table_pieces.append(piece)
                margin_keys.append(all_key)
        else:
            from pandas import DataFrame
            cat_axis = 0
            for (key, piece) in table.groupby(level=0, axis=cat_axis, observed=observed):
                if (len(cols) > 1):
                    all_key = _all_key(key)
                else:
                    all_key = margins_name
                table_pieces.append(piece)
                transformed_piece = DataFrame(piece.apply(aggfunc)).T
                transformed_piece.index = Index([all_key], name=piece.index.name)
                table_pieces.append(transformed_piece)
                margin_keys.append(all_key)
        result = concat(table_pieces, axis=cat_axis)
        if (len(rows) == 0):
            return result
    else:
        result = table
        margin_keys = table.columns
    if (len(cols) > 0):
        row_margin = data[(cols + values)].groupby(cols, observed=observed).agg(aggfunc)
        row_margin = row_margin.stack()
        new_order = ([len(cols)] + list(range(len(cols))))
        row_margin.index = row_margin.index.reorder_levels(new_order)
    else:
        row_margin = Series(np.nan, index=result.columns)
    return (result, margin_keys, row_margin)

def _generate_marginal_results_without_values(table, data, rows, cols, aggfunc, observed, margins_name='All'):
    if (len(cols) > 0):
        margin_keys: Union[(List, Index)] = []

        def _all_key():
            if (len(cols) == 1):
                return margins_name
            return ((margins_name,) + (('',) * (len(cols) - 1)))
        if (len(rows) > 0):
            margin = data[rows].groupby(rows, observed=observed).apply(aggfunc)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
        else:
            margin = data.groupby(level=0, axis=0, observed=observed).apply(aggfunc)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
            return result
    else:
        result = table
        margin_keys = table.columns
    if len(cols):
        row_margin = data[cols].groupby(cols, observed=observed).apply(aggfunc)
    else:
        row_margin = Series(np.nan, index=result.columns)
    return (result, margin_keys, row_margin)

def _convert_by(by):
    if (by is None):
        by = []
    elif (is_scalar(by) or isinstance(by, (np.ndarray, Index, ABCSeries, Grouper)) or hasattr(by, '__call__')):
        by = [by]
    else:
        by = list(by)
    return by

@Substitution('\ndata : DataFrame')
@Appender(_shared_docs['pivot'], indents=1)
def pivot(data, index=None, columns=None, values=None):
    if (columns is None):
        raise TypeError("pivot() missing 1 required argument: 'columns'")
    columns = com.convert_to_list_like(columns)
    if (values is None):
        if (index is not None):
            cols = com.convert_to_list_like(index)
        else:
            cols = []
        append = (index is None)
        indexed = data.set_index((cols + columns), append=append)
    else:
        if (index is None):
            index = [Series(data.index, name=data.index.name)]
        else:
            index = com.convert_to_list_like(index)
            index = [data[idx] for idx in index]
        data_columns = [data[col] for col in columns]
        index.extend(data_columns)
        index = MultiIndex.from_arrays(index)
        if (is_list_like(values) and (not isinstance(values, tuple))):
            values = cast(Sequence[Label], values)
            indexed = data._constructor(data[values]._values, index=index, columns=values)
        else:
            indexed = data._constructor_sliced(data[values]._values, index=index)
    return indexed.unstack(columns)

def crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False):
    '\n    Compute a simple cross tabulation of two (or more) factors. By default\n    computes a frequency table of the factors unless an array of values and an\n    aggregation function are passed.\n\n    Parameters\n    ----------\n    index : array-like, Series, or list of arrays/Series\n        Values to group by in the rows.\n    columns : array-like, Series, or list of arrays/Series\n        Values to group by in the columns.\n    values : array-like, optional\n        Array of values to aggregate according to the factors.\n        Requires `aggfunc` be specified.\n    rownames : sequence, default None\n        If passed, must match number of row arrays passed.\n    colnames : sequence, default None\n        If passed, must match number of column arrays passed.\n    aggfunc : function, optional\n        If specified, requires `values` be specified as well.\n    margins : bool, default False\n        Add row/column margins (subtotals).\n    margins_name : str, default \'All\'\n        Name of the row/column that will contain the totals\n        when margins is True.\n    dropna : bool, default True\n        Do not include columns whose entries are all NaN.\n    normalize : bool, {\'all\', \'index\', \'columns\'}, or {0,1}, default False\n        Normalize by dividing all values by the sum of values.\n\n        - If passed \'all\' or `True`, will normalize over all values.\n        - If passed \'index\' will normalize over each row.\n        - If passed \'columns\' will normalize over each column.\n        - If margins is `True`, will also normalize margin values.\n\n    Returns\n    -------\n    DataFrame\n        Cross tabulation of the data.\n\n    See Also\n    --------\n    DataFrame.pivot : Reshape data based on column values.\n    pivot_table : Create a pivot table as a DataFrame.\n\n    Notes\n    -----\n    Any Series passed will have their name attributes used unless row or column\n    names for the cross-tabulation are specified.\n\n    Any input passed containing Categorical data will have **all** of its\n    categories included in the cross-tabulation, even if the actual data does\n    not contain any instances of a particular category.\n\n    In the event that there aren\'t overlapping indexes an empty DataFrame will\n    be returned.\n\n    Examples\n    --------\n    >>> a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",\n    ...               "bar", "bar", "foo", "foo", "foo"], dtype=object)\n    >>> b = np.array(["one", "one", "one", "two", "one", "one",\n    ...               "one", "two", "two", "two", "one"], dtype=object)\n    >>> c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",\n    ...               "shiny", "dull", "shiny", "shiny", "shiny"],\n    ...              dtype=object)\n    >>> pd.crosstab(a, [b, c], rownames=[\'a\'], colnames=[\'b\', \'c\'])\n    b   one        two\n    c   dull shiny dull shiny\n    a\n    bar    1     2    1     0\n    foo    2     2    1     2\n\n    Here \'c\' and \'f\' are not represented in the data and will not be\n    shown in the output because dropna is True by default. Set\n    dropna=False to preserve categories with no data.\n\n    >>> foo = pd.Categorical([\'a\', \'b\'], categories=[\'a\', \'b\', \'c\'])\n    >>> bar = pd.Categorical([\'d\', \'e\'], categories=[\'d\', \'e\', \'f\'])\n    >>> pd.crosstab(foo, bar)\n    col_0  d  e\n    row_0\n    a      1  0\n    b      0  1\n    >>> pd.crosstab(foo, bar, dropna=False)\n    col_0  d  e  f\n    row_0\n    a      1  0  0\n    b      0  1  0\n    c      0  0  0\n    '
    if ((values is None) and (aggfunc is not None)):
        raise ValueError('aggfunc cannot be used without values.')
    if ((values is not None) and (aggfunc is None)):
        raise ValueError('values cannot be used without an aggfunc.')
    index = com.maybe_make_list(index)
    columns = com.maybe_make_list(columns)
    common_idx = None
    pass_objs = [x for x in (index + columns) if isinstance(x, (ABCSeries, ABCDataFrame))]
    if pass_objs:
        common_idx = get_objs_combined_axis(pass_objs, intersect=True, sort=False)
    rownames = _get_names(index, rownames, prefix='row')
    colnames = _get_names(columns, colnames, prefix='col')
    (rownames_mapper, unique_rownames, colnames_mapper, unique_colnames) = _build_names_mapper(rownames, colnames)
    from pandas import DataFrame
    data = {**dict(zip(unique_rownames, index)), **dict(zip(unique_colnames, columns))}
    df = DataFrame(data, index=common_idx)
    original_df_cols = df.columns
    if (values is None):
        df['__dummy__'] = 0
        kwargs = {'aggfunc': len, 'fill_value': 0}
    else:
        df['__dummy__'] = values
        kwargs = {'aggfunc': aggfunc}
    table = df.pivot_table(['__dummy__'], index=unique_rownames, columns=unique_colnames, margins=margins, margins_name=margins_name, dropna=dropna, **kwargs)
    if (not table.empty):
        cols_diff = df.columns.difference(original_df_cols)[0]
        table = table[cols_diff]
    if (normalize is not False):
        table = _normalize(table, normalize=normalize, margins=margins, margins_name=margins_name)
    table = table.rename_axis(index=rownames_mapper, axis=0)
    table = table.rename_axis(columns=colnames_mapper, axis=1)
    return table

def _normalize(table, normalize, margins, margins_name='All'):
    if (not isinstance(normalize, (bool, str))):
        axis_subs = {0: 'index', 1: 'columns'}
        try:
            normalize = axis_subs[normalize]
        except KeyError as err:
            raise ValueError('Not a valid normalize argument') from err
    if (margins is False):
        normalizers: Dict[(Union[(bool, str)], Callable)] = {'all': (lambda x: (x / x.sum(axis=1).sum(axis=0))), 'columns': (lambda x: (x / x.sum())), 'index': (lambda x: x.div(x.sum(axis=1), axis=0))}
        normalizers[True] = normalizers['all']
        try:
            f = normalizers[normalize]
        except KeyError as err:
            raise ValueError('Not a valid normalize argument') from err
        table = f(table)
        table = table.fillna(0)
    elif (margins is True):
        table_index = table.index
        table_columns = table.columns
        last_ind_or_col = table.iloc[(- 1), :].name
        if ((margins_name not in last_ind_or_col) & (margins_name != last_ind_or_col)):
            raise ValueError(f'{margins_name} not in pivoted DataFrame')
        column_margin = table.iloc[:(- 1), (- 1)]
        index_margin = table.iloc[(- 1), :(- 1)]
        table = table.iloc[:(- 1), :(- 1)]
        table = _normalize(table, normalize=normalize, margins=False)
        if (normalize == 'columns'):
            column_margin = (column_margin / column_margin.sum())
            table = concat([table, column_margin], axis=1)
            table = table.fillna(0)
            table.columns = table_columns
        elif (normalize == 'index'):
            index_margin = (index_margin / index_margin.sum())
            table = table.append(index_margin)
            table = table.fillna(0)
            table.index = table_index
        elif ((normalize == 'all') or (normalize is True)):
            column_margin = (column_margin / column_margin.sum())
            index_margin = (index_margin / index_margin.sum())
            index_margin.loc[margins_name] = 1
            table = concat([table, column_margin], axis=1)
            table = table.append(index_margin)
            table = table.fillna(0)
            table.index = table_index
            table.columns = table_columns
        else:
            raise ValueError('Not a valid normalize argument')
    else:
        raise ValueError('Not a valid margins argument')
    return table

def _get_names(arrs, names, prefix='row'):
    if (names is None):
        names = []
        for (i, arr) in enumerate(arrs):
            if (isinstance(arr, ABCSeries) and (arr.name is not None)):
                names.append(arr.name)
            else:
                names.append(f'{prefix}_{i}')
    else:
        if (len(names) != len(arrs)):
            raise AssertionError('arrays and names must have the same length')
        if (not isinstance(names, list)):
            names = list(names)
    return names

def _build_names_mapper(rownames, colnames):
    "\n    Given the names of a DataFrame's rows and columns, returns a set of unique row\n    and column names and mappers that convert to original names.\n\n    A row or column name is replaced if it is duplicate among the rows of the inputs,\n    among the columns of the inputs or between the rows and the columns.\n\n    Parameters\n    ----------\n    rownames: list[str]\n    colnames: list[str]\n\n    Returns\n    -------\n    Tuple(Dict[str, str], List[str], Dict[str, str], List[str])\n\n    rownames_mapper: dict[str, str]\n        a dictionary with new row names as keys and original rownames as values\n    unique_rownames: list[str]\n        a list of rownames with duplicate names replaced by dummy names\n    colnames_mapper: dict[str, str]\n        a dictionary with new column names as keys and original column names as values\n    unique_colnames: list[str]\n        a list of column names with duplicate names replaced by dummy names\n\n    "

    def get_duplicates(names):
        seen: Set = set()
        return {name for name in names if (name not in seen)}
    shared_names = set(rownames).intersection(set(colnames))
    dup_names = ((get_duplicates(rownames) | get_duplicates(colnames)) | shared_names)
    rownames_mapper = {f'row_{i}': name for (i, name) in enumerate(rownames) if (name in dup_names)}
    unique_rownames = [(f'row_{i}' if (name in dup_names) else name) for (i, name) in enumerate(rownames)]
    colnames_mapper = {f'col_{i}': name for (i, name) in enumerate(colnames) if (name in dup_names)}
    unique_colnames = [(f'col_{i}' if (name in dup_names) else name) for (i, name) in enumerate(colnames)]
    return (rownames_mapper, unique_rownames, colnames_mapper, unique_colnames)
