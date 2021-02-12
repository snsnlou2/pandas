
import re
from typing import TYPE_CHECKING, List, cast
import warnings
import numpy as np
from pandas.util._decorators import Appender, deprecate_kwarg
from pandas.core.dtypes.common import is_extension_array_dtype, is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import notna
from pandas.core.arrays import Categorical
import pandas.core.common as com
from pandas.core.indexes.api import Index, MultiIndex
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import tile_compat
from pandas.core.shared_docs import _shared_docs
from pandas.core.tools.numeric import to_numeric
if TYPE_CHECKING:
    from pandas import DataFrame, Series

@Appender((_shared_docs['melt'] % {'caller': 'pd.melt(df, ', 'other': 'DataFrame.melt'}))
def melt(frame, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True):
    if isinstance(frame.columns, MultiIndex):
        cols = [x for c in frame.columns for x in c]
    else:
        cols = list(frame.columns)
    if (value_name in frame.columns):
        warnings.warn("This dataframe has a column name that matches the 'value_name' column name of the resulting Dataframe. In the future this will raise an error, please set the 'value_name' parameter of DataFrame.melt to a unique name.", FutureWarning, stacklevel=3)
    if (id_vars is not None):
        if (not is_list_like(id_vars)):
            id_vars = [id_vars]
        elif (isinstance(frame.columns, MultiIndex) and (not isinstance(id_vars, list))):
            raise ValueError('id_vars must be a list of tuples when columns are a MultiIndex')
        else:
            id_vars = list(id_vars)
            missing = Index(com.flatten(id_vars)).difference(cols)
            if (not missing.empty):
                raise KeyError(f"The following 'id_vars' are not present in the DataFrame: {list(missing)}")
    else:
        id_vars = []
    if (value_vars is not None):
        if (not is_list_like(value_vars)):
            value_vars = [value_vars]
        elif (isinstance(frame.columns, MultiIndex) and (not isinstance(value_vars, list))):
            raise ValueError('value_vars must be a list of tuples when columns are a MultiIndex')
        else:
            value_vars = list(value_vars)
            missing = Index(com.flatten(value_vars)).difference(cols)
            if (not missing.empty):
                raise KeyError(f"The following 'value_vars' are not present in the DataFrame: {list(missing)}")
        if (col_level is not None):
            idx = frame.columns.get_level_values(col_level).get_indexer((id_vars + value_vars))
        else:
            idx = frame.columns.get_indexer((id_vars + value_vars))
        frame = frame.iloc[:, idx]
    else:
        frame = frame.copy()
    if (col_level is not None):
        frame.columns = frame.columns.get_level_values(col_level)
    if (var_name is None):
        if isinstance(frame.columns, MultiIndex):
            if (len(frame.columns.names) == len(set(frame.columns.names))):
                var_name = frame.columns.names
            else:
                var_name = [f'variable_{i}' for i in range(len(frame.columns.names))]
        else:
            var_name = [(frame.columns.name if (frame.columns.name is not None) else 'variable')]
    if isinstance(var_name, str):
        var_name = [var_name]
    (N, K) = frame.shape
    K -= len(id_vars)
    mdata = {}
    for col in id_vars:
        id_data = frame.pop(col)
        if is_extension_array_dtype(id_data):
            id_data = cast('Series', concat(([id_data] * K), ignore_index=True))
        else:
            id_data = np.tile(id_data._values, K)
        mdata[col] = id_data
    mcolumns = ((id_vars + var_name) + [value_name])
    mdata[value_name] = frame._values.ravel('F')
    for (i, col) in enumerate(var_name):
        mdata[col] = np.asanyarray(frame.columns._get_level_values(i)).repeat(N)
    result = frame._constructor(mdata, columns=mcolumns)
    if (not ignore_index):
        result.index = tile_compat(frame.index, K)
    return result

@deprecate_kwarg(old_arg_name='label', new_arg_name=None)
def lreshape(data, groups, dropna=True, label=None):
    '\n    Reshape wide-format data to long. Generalized inverse of DataFrame.pivot.\n\n    Accepts a dictionary, ``groups``, in which each key is a new column name\n    and each value is a list of old column names that will be "melted" under\n    the new column name as part of the reshape.\n\n    Parameters\n    ----------\n    data : DataFrame\n        The wide-format DataFrame.\n    groups : dict\n        {new_name : list_of_columns}.\n    dropna : bool, default True\n        Do not include columns whose entries are all NaN.\n    label : None\n        Not used.\n\n        .. deprecated:: 1.0.0\n\n    Returns\n    -------\n    DataFrame\n        Reshaped DataFrame.\n\n    See Also\n    --------\n    melt : Unpivot a DataFrame from wide to long format, optionally leaving\n        identifiers set.\n    pivot : Create a spreadsheet-style pivot table as a DataFrame.\n    DataFrame.pivot : Pivot without aggregation that can handle\n        non-numeric data.\n    DataFrame.pivot_table : Generalization of pivot that can handle\n        duplicate values for one index/column pair.\n    DataFrame.unstack : Pivot based on the index values instead of a\n        column.\n    wide_to_long : Wide panel to long format. Less flexible but more\n        user-friendly than melt.\n\n    Examples\n    --------\n    >>> data = pd.DataFrame({\'hr1\': [514, 573], \'hr2\': [545, 526],\n    ...                      \'team\': [\'Red Sox\', \'Yankees\'],\n    ...                      \'year1\': [2007, 2007], \'year2\': [2008, 2008]})\n    >>> data\n       hr1  hr2     team  year1  year2\n    0  514  545  Red Sox   2007   2008\n    1  573  526  Yankees   2007   2008\n\n    >>> pd.lreshape(data, {\'year\': [\'year1\', \'year2\'], \'hr\': [\'hr1\', \'hr2\']})\n          team  year   hr\n    0  Red Sox  2007  514\n    1  Yankees  2007  573\n    2  Red Sox  2008  545\n    3  Yankees  2008  526\n    '
    if isinstance(groups, dict):
        keys = list(groups.keys())
        values = list(groups.values())
    else:
        (keys, values) = zip(*groups)
    all_cols = list(set.union(*[set(x) for x in values]))
    id_cols = list(data.columns.difference(all_cols))
    K = len(values[0])
    for seq in values:
        if (len(seq) != K):
            raise ValueError('All column lists must be same length')
    mdata = {}
    pivot_cols = []
    for (target, names) in zip(keys, values):
        to_concat = [data[col]._values for col in names]
        mdata[target] = concat_compat(to_concat)
        pivot_cols.append(target)
    for col in id_cols:
        mdata[col] = np.tile(data[col]._values, K)
    if dropna:
        mask = np.ones(len(mdata[pivot_cols[0]]), dtype=bool)
        for c in pivot_cols:
            mask &= notna(mdata[c])
        if (not mask.all()):
            mdata = {k: v[mask] for (k, v) in mdata.items()}
    return data._constructor(mdata, columns=(id_cols + pivot_cols))

def wide_to_long(df, stubnames, i, j, sep='', suffix='\\d+'):
    '\n    Wide panel to long format. Less flexible but more user-friendly than melt.\n\n    With stubnames [\'A\', \'B\'], this function expects to find one or more\n    group of columns with format\n    A-suffix1, A-suffix2,..., B-suffix1, B-suffix2,...\n    You specify what you want to call this suffix in the resulting long format\n    with `j` (for example `j=\'year\'`)\n\n    Each row of these wide variables are assumed to be uniquely identified by\n    `i` (can be a single column name or a list of column names)\n\n    All remaining variables in the data frame are left intact.\n\n    Parameters\n    ----------\n    df : DataFrame\n        The wide-format DataFrame.\n    stubnames : str or list-like\n        The stub name(s). The wide format variables are assumed to\n        start with the stub names.\n    i : str or list-like\n        Column(s) to use as id variable(s).\n    j : str\n        The name of the sub-observation variable. What you wish to name your\n        suffix in the long format.\n    sep : str, default ""\n        A character indicating the separation of the variable names\n        in the wide format, to be stripped from the names in the long format.\n        For example, if your column names are A-suffix1, A-suffix2, you\n        can strip the hyphen by specifying `sep=\'-\'`.\n    suffix : str, default \'\\\\d+\'\n        A regular expression capturing the wanted suffixes. \'\\\\d+\' captures\n        numeric suffixes. Suffixes with no numbers could be specified with the\n        negated character class \'\\\\D+\'. You can also further disambiguate\n        suffixes, for example, if your wide variables are of the form A-one,\n        B-two,.., and you have an unrelated column A-rating, you can ignore the\n        last one by specifying `suffix=\'(!?one|two)\'`. When all suffixes are\n        numeric, they are cast to int64/float64.\n\n    Returns\n    -------\n    DataFrame\n        A DataFrame that contains each stub name as a variable, with new index\n        (i, j).\n\n    See Also\n    --------\n    melt : Unpivot a DataFrame from wide to long format, optionally leaving\n        identifiers set.\n    pivot : Create a spreadsheet-style pivot table as a DataFrame.\n    DataFrame.pivot : Pivot without aggregation that can handle\n        non-numeric data.\n    DataFrame.pivot_table : Generalization of pivot that can handle\n        duplicate values for one index/column pair.\n    DataFrame.unstack : Pivot based on the index values instead of a\n        column.\n\n    Notes\n    -----\n    All extra variables are left untouched. This simply uses\n    `pandas.melt` under the hood, but is hard-coded to "do the right thing"\n    in a typical case.\n\n    Examples\n    --------\n    >>> np.random.seed(123)\n    >>> df = pd.DataFrame({"A1970" : {0 : "a", 1 : "b", 2 : "c"},\n    ...                    "A1980" : {0 : "d", 1 : "e", 2 : "f"},\n    ...                    "B1970" : {0 : 2.5, 1 : 1.2, 2 : .7},\n    ...                    "B1980" : {0 : 3.2, 1 : 1.3, 2 : .1},\n    ...                    "X"     : dict(zip(range(3), np.random.randn(3)))\n    ...                   })\n    >>> df["id"] = df.index\n    >>> df\n      A1970 A1980  B1970  B1980         X  id\n    0     a     d    2.5    3.2 -1.085631   0\n    1     b     e    1.2    1.3  0.997345   1\n    2     c     f    0.7    0.1  0.282978   2\n    >>> pd.wide_to_long(df, ["A", "B"], i="id", j="year")\n    ... # doctest: +NORMALIZE_WHITESPACE\n                    X  A    B\n    id year\n    0  1970 -1.085631  a  2.5\n    1  1970  0.997345  b  1.2\n    2  1970  0.282978  c  0.7\n    0  1980 -1.085631  d  3.2\n    1  1980  0.997345  e  1.3\n    2  1980  0.282978  f  0.1\n\n    With multiple id columns\n\n    >>> df = pd.DataFrame({\n    ...     \'famid\': [1, 1, 1, 2, 2, 2, 3, 3, 3],\n    ...     \'birth\': [1, 2, 3, 1, 2, 3, 1, 2, 3],\n    ...     \'ht1\': [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],\n    ...     \'ht2\': [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9]\n    ... })\n    >>> df\n       famid  birth  ht1  ht2\n    0      1      1  2.8  3.4\n    1      1      2  2.9  3.8\n    2      1      3  2.2  2.9\n    3      2      1  2.0  3.2\n    4      2      2  1.8  2.8\n    5      2      3  1.9  2.4\n    6      3      1  2.2  3.3\n    7      3      2  2.3  3.4\n    8      3      3  2.1  2.9\n    >>> l = pd.wide_to_long(df, stubnames=\'ht\', i=[\'famid\', \'birth\'], j=\'age\')\n    >>> l\n    ... # doctest: +NORMALIZE_WHITESPACE\n                      ht\n    famid birth age\n    1     1     1    2.8\n                2    3.4\n          2     1    2.9\n                2    3.8\n          3     1    2.2\n                2    2.9\n    2     1     1    2.0\n                2    3.2\n          2     1    1.8\n                2    2.8\n          3     1    1.9\n                2    2.4\n    3     1     1    2.2\n                2    3.3\n          2     1    2.3\n                2    3.4\n          3     1    2.1\n                2    2.9\n\n    Going from long back to wide just takes some creative use of `unstack`\n\n    >>> w = l.unstack()\n    >>> w.columns = w.columns.map(\'{0[0]}{0[1]}\'.format)\n    >>> w.reset_index()\n       famid  birth  ht1  ht2\n    0      1      1  2.8  3.4\n    1      1      2  2.9  3.8\n    2      1      3  2.2  2.9\n    3      2      1  2.0  3.2\n    4      2      2  1.8  2.8\n    5      2      3  1.9  2.4\n    6      3      1  2.2  3.3\n    7      3      2  2.3  3.4\n    8      3      3  2.1  2.9\n\n    Less wieldy column names are also handled\n\n    >>> np.random.seed(0)\n    >>> df = pd.DataFrame({\'A(weekly)-2010\': np.random.rand(3),\n    ...                    \'A(weekly)-2011\': np.random.rand(3),\n    ...                    \'B(weekly)-2010\': np.random.rand(3),\n    ...                    \'B(weekly)-2011\': np.random.rand(3),\n    ...                    \'X\' : np.random.randint(3, size=3)})\n    >>> df[\'id\'] = df.index\n    >>> df # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS\n       A(weekly)-2010  A(weekly)-2011  B(weekly)-2010  B(weekly)-2011  X  id\n    0        0.548814        0.544883        0.437587        0.383442  0   0\n    1        0.715189        0.423655        0.891773        0.791725  1   1\n    2        0.602763        0.645894        0.963663        0.528895  1   2\n\n    >>> pd.wide_to_long(df, [\'A(weekly)\', \'B(weekly)\'], i=\'id\',\n    ...                 j=\'year\', sep=\'-\')\n    ... # doctest: +NORMALIZE_WHITESPACE\n             X  A(weekly)  B(weekly)\n    id year\n    0  2010  0   0.548814   0.437587\n    1  2010  1   0.715189   0.891773\n    2  2010  1   0.602763   0.963663\n    0  2011  0   0.544883   0.383442\n    1  2011  1   0.423655   0.791725\n    2  2011  1   0.645894   0.528895\n\n    If we have many columns, we could also use a regex to find our\n    stubnames and pass that list on to wide_to_long\n\n    >>> stubnames = sorted(\n    ...     set([match[0] for match in df.columns.str.findall(\n    ...         r\'[A-B]\\(.*\\)\').values if match != []])\n    ... )\n    >>> list(stubnames)\n    [\'A(weekly)\', \'B(weekly)\']\n\n    All of the above examples have integers as suffixes. It is possible to\n    have non-integers as suffixes.\n\n    >>> df = pd.DataFrame({\n    ...     \'famid\': [1, 1, 1, 2, 2, 2, 3, 3, 3],\n    ...     \'birth\': [1, 2, 3, 1, 2, 3, 1, 2, 3],\n    ...     \'ht_one\': [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],\n    ...     \'ht_two\': [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9]\n    ... })\n    >>> df\n       famid  birth  ht_one  ht_two\n    0      1      1     2.8     3.4\n    1      1      2     2.9     3.8\n    2      1      3     2.2     2.9\n    3      2      1     2.0     3.2\n    4      2      2     1.8     2.8\n    5      2      3     1.9     2.4\n    6      3      1     2.2     3.3\n    7      3      2     2.3     3.4\n    8      3      3     2.1     2.9\n\n    >>> l = pd.wide_to_long(df, stubnames=\'ht\', i=[\'famid\', \'birth\'], j=\'age\',\n    ...                     sep=\'_\', suffix=r\'\\w+\')\n    >>> l\n    ... # doctest: +NORMALIZE_WHITESPACE\n                      ht\n    famid birth age\n    1     1     one  2.8\n                two  3.4\n          2     one  2.9\n                two  3.8\n          3     one  2.2\n                two  2.9\n    2     1     one  2.0\n                two  3.2\n          2     one  1.8\n                two  2.8\n          3     one  1.9\n                two  2.4\n    3     1     one  2.2\n                two  3.3\n          2     one  2.3\n                two  3.4\n          3     one  2.1\n                two  2.9\n    '

    def get_var_names(df, stub: str, sep: str, suffix: str) -> List[str]:
        regex = f'^{re.escape(stub)}{re.escape(sep)}{suffix}$'
        pattern = re.compile(regex)
        return [col for col in df.columns if pattern.match(col)]

    def melt_stub(df, stub: str, i, j, value_vars, sep: str):
        newdf = melt(df, id_vars=i, value_vars=value_vars, value_name=stub.rstrip(sep), var_name=j)
        newdf[j] = Categorical(newdf[j])
        newdf[j] = newdf[j].str.replace(re.escape((stub + sep)), '', regex=True)
        newdf[j] = to_numeric(newdf[j], errors='ignore')
        return newdf.set_index((i + [j]))
    if (not is_list_like(stubnames)):
        stubnames = [stubnames]
    else:
        stubnames = list(stubnames)
    if any(((col in stubnames) for col in df.columns)):
        raise ValueError("stubname can't be identical to a column name")
    if (not is_list_like(i)):
        i = [i]
    else:
        i = list(i)
    if df[i].duplicated().any():
        raise ValueError('the id variables need to uniquely identify each row')
    value_vars = [get_var_names(df, stub, sep, suffix) for stub in stubnames]
    value_vars_flattened = [e for sublist in value_vars for e in sublist]
    id_vars = list(set(df.columns.tolist()).difference(value_vars_flattened))
    _melted = [melt_stub(df, s, i, j, v, sep) for (s, v) in zip(stubnames, value_vars)]
    melted = _melted[0].join(_melted[1:], how='outer')
    if (len(i) == 1):
        new = df[id_vars].set_index(i).join(melted)
        return new
    new = df[id_vars].merge(melted.reset_index(), on=i).set_index((i + [j]))
    return new
