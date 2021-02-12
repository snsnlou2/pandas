
'\nDataFrame\n---------\nAn efficient 2D container for potentially mixed-type time series or other\nlabeled data series.\n\nSimilar to its R counterpart, data.frame, except providing automatic data\nalignment and a host of useful data manipulation methods having to do with the\nlabeling information\n'
from __future__ import annotations
import collections
from collections import abc
import datetime
from io import StringIO
import itertools
import mmap
from textwrap import dedent
from typing import IO, TYPE_CHECKING, Any, AnyStr, Dict, FrozenSet, Hashable, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Type, Union, cast, overload
import warnings
import numpy as np
import numpy.ma as ma
from pandas._config import get_option
from pandas._libs import algos as libalgos, lib, properties
from pandas._libs.lib import no_default
from pandas._typing import AggFuncType, ArrayLike, Axes, Axis, ColspaceArgType, CompressionOptions, Dtype, FilePathOrBuffer, FloatFormatType, FormattersType, FrameOrSeriesUnion, IndexKeyFunc, IndexLabel, Label, Level, PythonFuncType, Renamer, StorageOptions, Suffixes, ValueKeyFunc
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.util._decorators import Appender, Substitution, deprecate_kwarg, doc, rewrite_axis_style_signature
from pandas.util._validators import validate_axis_style_args, validate_bool_kwarg, validate_percentile
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar, construct_2d_arraylike_from_scalar, find_common_type, infer_dtype_from_scalar, invalidate_string_dtypes, maybe_box_datetimelike, maybe_convert_platform, maybe_downcast_to_dtype, maybe_infer_to_datetimelike, validate_numeric_casting
from pandas.core.dtypes.common import ensure_int64, ensure_platform_int, infer_dtype_from_object, is_bool_dtype, is_dataclass, is_datetime64_any_dtype, is_dict_like, is_dtype_equal, is_extension_array_dtype, is_float, is_float_dtype, is_hashable, is_integer, is_integer_dtype, is_iterator, is_list_like, is_object_dtype, is_scalar, is_sequence, pandas_dtype
from pandas.core.dtypes.missing import isna, notna
from pandas.core import algorithms, common as com, generic, nanops, ops
from pandas.core.accessor import CachedAccessor
from pandas.core.aggregation import aggregate, reconstruct_func, relabel_result, transform
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import extract_array, sanitize_masked_array
from pandas.core.generic import NDFrame, _shared_docs
from pandas.core.indexes import base as ibase
from pandas.core.indexes.api import DatetimeIndex, Index, PeriodIndex, ensure_index, ensure_index_from_sequences
from pandas.core.indexes.multi import MultiIndex, maybe_droplevels
from pandas.core.indexing import check_bool_indexer, convert_to_index_sliceable
from pandas.core.internals import BlockManager
from pandas.core.internals.construction import arrays_to_mgr, dataclasses_to_dicts, init_dict, init_ndarray, masked_rec_array_to_mgr, nested_data_to_arrays, reorder_arrays, sanitize_index, to_arrays, treat_as_nested
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.sorting import get_group_index, lexsort_indexer, nargsort
from pandas.io.common import get_handle
from pandas.io.formats import console, format as fmt
from pandas.io.formats.info import BaseInfo, DataFrameInfo
import pandas.plotting
if TYPE_CHECKING:
    from typing import Literal
    from pandas._typing import TimedeltaConvertibleTypes, TimestampConvertibleTypes
    from pandas.core.groupby.generic import DataFrameGroupBy
    from pandas.core.resample import Resampler
    from pandas.io.formats.style import Styler
_shared_doc_kwargs = {'axes': 'index, columns', 'klass': 'DataFrame', 'axes_single_arg': "{0 or 'index', 1 or 'columns'}", 'axis': "axis : {0 or 'index', 1 or 'columns'}, default 0\n        If 0 or 'index': apply function to each column.\n        If 1 or 'columns': apply function to each row.", 'inplace': '\n    inplace : boolean, default False\n        If True, performs operation inplace and returns None.', 'optional_by': "\n        by : str or list of str\n            Name or list of names to sort by.\n\n            - if `axis` is 0 or `'index'` then `by` may contain index\n              levels and/or column labels.\n            - if `axis` is 1 or `'columns'` then `by` may contain column\n              levels and/or index labels.", 'optional_labels': "labels : array-like, optional\n            New labels / index to conform the axis specified by 'axis' to.", 'optional_axis': "axis : int or str, optional\n            Axis to target. Can be either the axis name ('index', 'columns')\n            or number (0, 1).", 'replace_iloc': '\n    This differs from updating with ``.loc`` or ``.iloc``, which require\n    you to specify a location to update with some value.'}
_numeric_only_doc = 'numeric_only : boolean, default None\n    Include only float, int, boolean data. If None, will attempt to use\n    everything, then use only numeric data\n'
_merge_doc = '\nMerge DataFrame or named Series objects with a database-style join.\n\nThe join is done on columns or indexes. If joining columns on\ncolumns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes\non indexes or indexes on a column or columns, the index will be passed on.\nWhen performing a cross merge, no column specifications to merge on are\nallowed.\n\nParameters\n----------%s\nright : DataFrame or named Series\n    Object to merge with.\nhow : {\'left\', \'right\', \'outer\', \'inner\', \'cross\'}, default \'inner\'\n    Type of merge to be performed.\n\n    * left: use only keys from left frame, similar to a SQL left outer join;\n      preserve key order.\n    * right: use only keys from right frame, similar to a SQL right outer join;\n      preserve key order.\n    * outer: use union of keys from both frames, similar to a SQL full outer\n      join; sort keys lexicographically.\n    * inner: use intersection of keys from both frames, similar to a SQL inner\n      join; preserve the order of the left keys.\n    * cross: creates the cartesian product from both frames, preserves the order\n      of the left keys.\n\n      .. versionadded:: 1.2.0\n\non : label or list\n    Column or index level names to join on. These must be found in both\n    DataFrames. If `on` is None and not merging on indexes then this defaults\n    to the intersection of the columns in both DataFrames.\nleft_on : label or list, or array-like\n    Column or index level names to join on in the left DataFrame. Can also\n    be an array or list of arrays of the length of the left DataFrame.\n    These arrays are treated as if they are columns.\nright_on : label or list, or array-like\n    Column or index level names to join on in the right DataFrame. Can also\n    be an array or list of arrays of the length of the right DataFrame.\n    These arrays are treated as if they are columns.\nleft_index : bool, default False\n    Use the index from the left DataFrame as the join key(s). If it is a\n    MultiIndex, the number of keys in the other DataFrame (either the index\n    or a number of columns) must match the number of levels.\nright_index : bool, default False\n    Use the index from the right DataFrame as the join key. Same caveats as\n    left_index.\nsort : bool, default False\n    Sort the join keys lexicographically in the result DataFrame. If False,\n    the order of the join keys depends on the join type (how keyword).\nsuffixes : list-like, default is ("_x", "_y")\n    A length-2 sequence where each element is optionally a string\n    indicating the suffix to add to overlapping column names in\n    `left` and `right` respectively. Pass a value of `None` instead\n    of a string to indicate that the column name from `left` or\n    `right` should be left as-is, with no suffix. At least one of the\n    values must not be None.\ncopy : bool, default True\n    If False, avoid copy if possible.\nindicator : bool or str, default False\n    If True, adds a column to the output DataFrame called "_merge" with\n    information on the source of each row. The column can be given a different\n    name by providing a string argument. The column will have a Categorical\n    type with the value of "left_only" for observations whose merge key only\n    appears in the left DataFrame, "right_only" for observations\n    whose merge key only appears in the right DataFrame, and "both"\n    if the observation\'s merge key is found in both DataFrames.\n\nvalidate : str, optional\n    If specified, checks if merge is of specified type.\n\n    * "one_to_one" or "1:1": check if merge keys are unique in both\n      left and right datasets.\n    * "one_to_many" or "1:m": check if merge keys are unique in left\n      dataset.\n    * "many_to_one" or "m:1": check if merge keys are unique in right\n      dataset.\n    * "many_to_many" or "m:m": allowed, but does not result in checks.\n\nReturns\n-------\nDataFrame\n    A DataFrame of the two merged objects.\n\nSee Also\n--------\nmerge_ordered : Merge with optional filling/interpolation.\nmerge_asof : Merge on nearest keys.\nDataFrame.join : Similar method using indices.\n\nNotes\n-----\nSupport for specifying index levels as the `on`, `left_on`, and\n`right_on` parameters was added in version 0.23.0\nSupport for merging named Series objects was added in version 0.24.0\n\nExamples\n--------\n>>> df1 = pd.DataFrame({\'lkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],\n...                     \'value\': [1, 2, 3, 5]})\n>>> df2 = pd.DataFrame({\'rkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],\n...                     \'value\': [5, 6, 7, 8]})\n>>> df1\n    lkey value\n0   foo      1\n1   bar      2\n2   baz      3\n3   foo      5\n>>> df2\n    rkey value\n0   foo      5\n1   bar      6\n2   baz      7\n3   foo      8\n\nMerge df1 and df2 on the lkey and rkey columns. The value columns have\nthe default suffixes, _x and _y, appended.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\')\n  lkey  value_x rkey  value_y\n0  foo        1  foo        5\n1  foo        1  foo        8\n2  foo        5  foo        5\n3  foo        5  foo        8\n4  bar        2  bar        6\n5  baz        3  baz        7\n\nMerge DataFrames df1 and df2 with specified left and right suffixes\nappended to any overlapping columns.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\',\n...           suffixes=(\'_left\', \'_right\'))\n  lkey  value_left rkey  value_right\n0  foo           1  foo            5\n1  foo           1  foo            8\n2  foo           5  foo            5\n3  foo           5  foo            8\n4  bar           2  bar            6\n5  baz           3  baz            7\n\nMerge DataFrames df1 and df2, but raise an exception if the DataFrames have\nany overlapping columns.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\', suffixes=(False, False))\nTraceback (most recent call last):\n...\nValueError: columns overlap but no suffix specified:\n    Index([\'value\'], dtype=\'object\')\n\n>>> df1 = pd.DataFrame({\'a\': [\'foo\', \'bar\'], \'b\': [1, 2]})\n>>> df2 = pd.DataFrame({\'a\': [\'foo\', \'baz\'], \'c\': [3, 4]})\n>>> df1\n      a  b\n0   foo  1\n1   bar  2\n>>> df2\n      a  c\n0   foo  3\n1   baz  4\n\n>>> df1.merge(df2, how=\'inner\', on=\'a\')\n      a  b  c\n0   foo  1  3\n\n>>> df1.merge(df2, how=\'left\', on=\'a\')\n      a  b  c\n0   foo  1  3.0\n1   bar  2  NaN\n\n>>> df1 = pd.DataFrame({\'left\': [\'foo\', \'bar\']})\n>>> df2 = pd.DataFrame({\'right\': [7, 8]})\n>>> df1\n    left\n0   foo\n1   bar\n>>> df2\n    right\n0   7\n1   8\n\n>>> df1.merge(df2, how=\'cross\')\n   left  right\n0   foo      7\n1   foo      8\n2   bar      7\n3   bar      8\n'

class DataFrame(NDFrame, OpsMixin):
    '\n    Two-dimensional, size-mutable, potentially heterogeneous tabular data.\n\n    Data structure also contains labeled axes (rows and columns).\n    Arithmetic operations align on both row and column labels. Can be\n    thought of as a dict-like container for Series objects. The primary\n    pandas data structure.\n\n    Parameters\n    ----------\n    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame\n        Dict can contain Series, arrays, constants, dataclass or list-like objects. If\n        data is a dict, column order follows insertion-order.\n\n        .. versionchanged:: 0.25.0\n           If data is a list of dicts, column order follows insertion-order.\n\n    index : Index or array-like\n        Index to use for resulting frame. Will default to RangeIndex if\n        no indexing information part of input data and no index provided.\n    columns : Index or array-like\n        Column labels to use for resulting frame. Will default to\n        RangeIndex (0, 1, 2, ..., n) if no column labels are provided.\n    dtype : dtype, default None\n        Data type to force. Only a single dtype is allowed. If None, infer.\n    copy : bool, default False\n        Copy data from inputs. Only affects DataFrame / 2d ndarray input.\n\n    See Also\n    --------\n    DataFrame.from_records : Constructor from tuples, also record arrays.\n    DataFrame.from_dict : From dicts of Series, arrays, or dicts.\n    read_csv : Read a comma-separated values (csv) file into DataFrame.\n    read_table : Read general delimited file into DataFrame.\n    read_clipboard : Read text from clipboard into DataFrame.\n\n    Examples\n    --------\n    Constructing DataFrame from a dictionary.\n\n    >>> d = {\'col1\': [1, 2], \'col2\': [3, 4]}\n    >>> df = pd.DataFrame(data=d)\n    >>> df\n       col1  col2\n    0     1     3\n    1     2     4\n\n    Notice that the inferred dtype is int64.\n\n    >>> df.dtypes\n    col1    int64\n    col2    int64\n    dtype: object\n\n    To enforce a single dtype:\n\n    >>> df = pd.DataFrame(data=d, dtype=np.int8)\n    >>> df.dtypes\n    col1    int8\n    col2    int8\n    dtype: object\n\n    Constructing DataFrame from numpy ndarray:\n\n    >>> df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),\n    ...                    columns=[\'a\', \'b\', \'c\'])\n    >>> df2\n       a  b  c\n    0  1  2  3\n    1  4  5  6\n    2  7  8  9\n\n    Constructing DataFrame from dataclass:\n\n    >>> from dataclasses import make_dataclass\n    >>> Point = make_dataclass("Point", [("x", int), ("y", int)])\n    >>> pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])\n        x  y\n    0  0  0\n    1  0  3\n    2  2  3\n    '
    _internal_names_set = ({'columns', 'index'} | NDFrame._internal_names_set)
    _typ = 'dataframe'
    _HANDLED_TYPES = (Series, Index, ExtensionArray, np.ndarray)

    @property
    def _constructor(self):
        return DataFrame
    _constructor_sliced = Series
    _hidden_attrs = (NDFrame._hidden_attrs | frozenset([]))
    _accessors = {'sparse'}

    @property
    def _constructor_expanddim(self):

        def constructor(*args, **kwargs):
            raise NotImplementedError('Not supported for DataFrames!')
        return constructor

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        if (data is None):
            data = {}
        if (dtype is not None):
            dtype = self._validate_dtype(dtype)
        if isinstance(data, DataFrame):
            data = data._mgr
        if isinstance(data, BlockManager):
            if ((index is None) and (columns is None) and (dtype is None) and (copy is False)):
                NDFrame.__init__(self, data)
                return
            mgr = self._init_mgr(data, axes={'index': index, 'columns': columns}, dtype=dtype, copy=copy)
        elif isinstance(data, dict):
            mgr = init_dict(data, index, columns, dtype=dtype)
        elif isinstance(data, ma.MaskedArray):
            import numpy.ma.mrecords as mrecords
            if isinstance(data, mrecords.MaskedRecords):
                mgr = masked_rec_array_to_mgr(data, index, columns, dtype, copy)
            else:
                data = sanitize_masked_array(data)
                mgr = init_ndarray(data, index, columns, dtype=dtype, copy=copy)
        elif isinstance(data, (np.ndarray, Series, Index)):
            if data.dtype.names:
                data_columns = list(data.dtype.names)
                data = {k: data[k] for k in data_columns}
                if (columns is None):
                    columns = data_columns
                mgr = init_dict(data, index, columns, dtype=dtype)
            elif (getattr(data, 'name', None) is not None):
                mgr = init_dict({data.name: data}, index, columns, dtype=dtype)
            else:
                mgr = init_ndarray(data, index, columns, dtype=dtype, copy=copy)
        elif is_list_like(data):
            if (not isinstance(data, (abc.Sequence, ExtensionArray))):
                data = list(data)
            if (len(data) > 0):
                if is_dataclass(data[0]):
                    data = dataclasses_to_dicts(data)
                if treat_as_nested(data):
                    (arrays, columns, index) = nested_data_to_arrays(data, columns, index, dtype)
                    mgr = arrays_to_mgr(arrays, columns, index, columns, dtype=dtype)
                else:
                    mgr = init_ndarray(data, index, columns, dtype=dtype, copy=copy)
            else:
                mgr = init_dict({}, index, columns, dtype=dtype)
        else:
            if ((index is None) or (columns is None)):
                raise ValueError('DataFrame constructor not properly called!')
            if (not dtype):
                (dtype, _) = infer_dtype_from_scalar(data, pandas_dtype=True)
            if is_extension_array_dtype(dtype):
                values = [construct_1d_arraylike_from_scalar(data, len(index), dtype) for _ in range(len(columns))]
                mgr = arrays_to_mgr(values, columns, index, columns, dtype=None)
            else:
                values = construct_2d_arraylike_from_scalar(data, len(index), len(columns), dtype, copy)
                mgr = init_ndarray(values, index, columns, dtype=values.dtype, copy=False)
        NDFrame.__init__(self, mgr)

    @property
    def axes(self):
        "\n        Return a list representing the axes of the DataFrame.\n\n        It has the row axis labels and column axis labels as the only members.\n        They are returned in that order.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})\n        >>> df.axes\n        [RangeIndex(start=0, stop=2, step=1), Index(['col1', 'col2'],\n        dtype='object')]\n        "
        return [self.index, self.columns]

    @property
    def shape(self):
        "\n        Return a tuple representing the dimensionality of the DataFrame.\n\n        See Also\n        --------\n        ndarray.shape : Tuple of array dimensions.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})\n        >>> df.shape\n        (2, 2)\n\n        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4],\n        ...                    'col3': [5, 6]})\n        >>> df.shape\n        (2, 3)\n        "
        return (len(self.index), len(self.columns))

    @property
    def _is_homogeneous_type(self):
        '\n        Whether all the columns in a DataFrame have the same type.\n\n        Returns\n        -------\n        bool\n\n        See Also\n        --------\n        Index._is_homogeneous_type : Whether the object has a single\n            dtype.\n        MultiIndex._is_homogeneous_type : Whether all the levels of a\n            MultiIndex have the same dtype.\n\n        Examples\n        --------\n        >>> DataFrame({"A": [1, 2], "B": [3, 4]})._is_homogeneous_type\n        True\n        >>> DataFrame({"A": [1, 2], "B": [3.0, 4.0]})._is_homogeneous_type\n        False\n\n        Items with the same type but different sizes are considered\n        different types.\n\n        >>> DataFrame({\n        ...    "A": np.array([1, 2], dtype=np.int32),\n        ...    "B": np.array([1, 2], dtype=np.int64)})._is_homogeneous_type\n        False\n        '
        if self._mgr.any_extension_types:
            return (len({block.dtype for block in self._mgr.blocks}) == 1)
        else:
            return (not self._is_mixed_type)

    @property
    def _can_fast_transpose(self):
        '\n        Can we transpose this DataFrame without creating any new array objects.\n        '
        if self._mgr.any_extension_types:
            return False
        return (len(self._mgr.blocks) == 1)

    def _repr_fits_vertical_(self):
        '\n        Check length against max_rows.\n        '
        max_rows = get_option('display.max_rows')
        return (len(self) <= max_rows)

    def _repr_fits_horizontal_(self, ignore_width=False):
        '\n        Check if full repr fits in horizontal boundaries imposed by the display\n        options width and max_columns.\n\n        In case of non-interactive session, no boundaries apply.\n\n        `ignore_width` is here so ipynb+HTML output can behave the way\n        users expect. display.max_columns remains in effect.\n        GH3541, GH3573\n        '
        (width, height) = console.get_console_size()
        max_columns = get_option('display.max_columns')
        nb_columns = len(self.columns)
        if ((max_columns and (nb_columns > max_columns)) or ((not ignore_width) and width and (nb_columns > (width // 2)))):
            return False
        if (ignore_width or (not console.in_interactive_session())):
            return True
        if ((get_option('display.width') is not None) or console.in_ipython_frontend()):
            max_rows = 1
        else:
            max_rows = get_option('display.max_rows')
        buf = StringIO()
        d = self
        if (not (max_rows is None)):
            d = d.iloc[:min(max_rows, len(d))]
        else:
            return True
        d.to_string(buf=buf)
        value = buf.getvalue()
        repr_width = max((len(line) for line in value.split('\n')))
        return (repr_width < width)

    def _info_repr(self):
        '\n        True if the repr should show the info view.\n        '
        info_repr_option = (get_option('display.large_repr') == 'info')
        return (info_repr_option and (not (self._repr_fits_horizontal_() and self._repr_fits_vertical_())))

    def __repr__(self):
        '\n        Return a string representation for a particular DataFrame.\n        '
        buf = StringIO('')
        if self._info_repr():
            self.info(buf=buf)
            return buf.getvalue()
        max_rows = get_option('display.max_rows')
        min_rows = get_option('display.min_rows')
        max_cols = get_option('display.max_columns')
        max_colwidth = get_option('display.max_colwidth')
        show_dimensions = get_option('display.show_dimensions')
        if get_option('display.expand_frame_repr'):
            (width, _) = console.get_console_size()
        else:
            width = None
        self.to_string(buf=buf, max_rows=max_rows, min_rows=min_rows, max_cols=max_cols, line_width=width, max_colwidth=max_colwidth, show_dimensions=show_dimensions)
        return buf.getvalue()

    def _repr_html_(self):
        '\n        Return a html representation for a particular DataFrame.\n\n        Mainly for IPython notebook.\n        '
        if self._info_repr():
            buf = StringIO('')
            self.info(buf=buf)
            val = buf.getvalue().replace('<', '&lt;', 1)
            val = val.replace('>', '&gt;', 1)
            return (('<pre>' + val) + '</pre>')
        if get_option('display.notebook_repr_html'):
            max_rows = get_option('display.max_rows')
            min_rows = get_option('display.min_rows')
            max_cols = get_option('display.max_columns')
            show_dimensions = get_option('display.show_dimensions')
            formatter = fmt.DataFrameFormatter(self, columns=None, col_space=None, na_rep='NaN', formatters=None, float_format=None, sparsify=None, justify=None, index_names=True, header=True, index=True, bold_rows=True, escape=True, max_rows=max_rows, min_rows=min_rows, max_cols=max_cols, show_dimensions=show_dimensions, decimal='.')
            return fmt.DataFrameRenderer(formatter).to_html(notebook=True)
        else:
            return None

    @Substitution(header_type='bool or sequence', header='Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names', col_space_type='int, list or dict of int', col_space='The minimum width of each column')
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    def to_string(self, buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, min_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None, max_colwidth=None, encoding=None):
        '\n        Render a DataFrame to a console-friendly tabular output.\n        %(shared_params)s\n        line_width : int, optional\n            Width to wrap a line in characters.\n        max_colwidth : int, optional\n            Max width to truncate each column in characters. By default, no limit.\n\n            .. versionadded:: 1.0.0\n        encoding : str, default "utf-8"\n            Set character encoding.\n\n            .. versionadded:: 1.0\n        %(returns)s\n        See Also\n        --------\n        to_html : Convert DataFrame to HTML.\n\n        Examples\n        --------\n        >>> d = {\'col1\': [1, 2, 3], \'col2\': [4, 5, 6]}\n        >>> df = pd.DataFrame(d)\n        >>> print(df.to_string())\n           col1  col2\n        0     1     4\n        1     2     5\n        2     3     6\n        '
        from pandas import option_context
        with option_context('display.max_colwidth', max_colwidth):
            formatter = fmt.DataFrameFormatter(self, columns=columns, col_space=col_space, na_rep=na_rep, formatters=formatters, float_format=float_format, sparsify=sparsify, justify=justify, index_names=index_names, header=header, index=index, min_rows=min_rows, max_rows=max_rows, max_cols=max_cols, show_dimensions=show_dimensions, decimal=decimal)
            return fmt.DataFrameRenderer(formatter).to_string(buf=buf, encoding=encoding, line_width=line_width)

    @property
    def style(self):
        '\n        Returns a Styler object.\n\n        Contains methods for building a styled HTML representation of the DataFrame.\n\n        See Also\n        --------\n        io.formats.style.Styler : Helps style a DataFrame or Series according to the\n            data with HTML and CSS.\n        '
        from pandas.io.formats.style import Styler
        return Styler(self)
    _shared_docs['items'] = "\n        Iterate over (column name, Series) pairs.\n\n        Iterates over the DataFrame columns, returning a tuple with\n        the column name and the content as a Series.\n\n        Yields\n        ------\n        label : object\n            The column names for the DataFrame being iterated over.\n        content : Series\n            The column entries belonging to each label, as a Series.\n\n        See Also\n        --------\n        DataFrame.iterrows : Iterate over DataFrame rows as\n            (index, Series) pairs.\n        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples\n            of the values.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],\n        ...                   'population': [1864, 22000, 80000]},\n        ...                   index=['panda', 'polar', 'koala'])\n        >>> df\n                species   population\n        panda   bear      1864\n        polar   bear      22000\n        koala   marsupial 80000\n        >>> for label, content in df.items():\n        ...     print(f'label: {label}')\n        ...     print(f'content: {content}', sep='\\n')\n        ...\n        label: species\n        content:\n        panda         bear\n        polar         bear\n        koala    marsupial\n        Name: species, dtype: object\n        label: population\n        content:\n        panda     1864\n        polar    22000\n        koala    80000\n        Name: population, dtype: int64\n        "

    @Appender(_shared_docs['items'])
    def items(self):
        if (self.columns.is_unique and hasattr(self, '_item_cache')):
            for k in self.columns:
                (yield (k, self._get_item_cache(k)))
        else:
            for (i, k) in enumerate(self.columns):
                (yield (k, self._ixs(i, axis=1)))

    @Appender(_shared_docs['items'])
    def iteritems(self):
        (yield from self.items())

    def iterrows(self):
        "\n        Iterate over DataFrame rows as (index, Series) pairs.\n\n        Yields\n        ------\n        index : label or tuple of label\n            The index of the row. A tuple for a `MultiIndex`.\n        data : Series\n            The data of the row as a Series.\n\n        See Also\n        --------\n        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples of the values.\n        DataFrame.items : Iterate over (column name, Series) pairs.\n\n        Notes\n        -----\n        1. Because ``iterrows`` returns a Series for each row,\n           it does **not** preserve dtypes across the rows (dtypes are\n           preserved across columns for DataFrames). For example,\n\n           >>> df = pd.DataFrame([[1, 1.5]], columns=['int', 'float'])\n           >>> row = next(df.iterrows())[1]\n           >>> row\n           int      1.0\n           float    1.5\n           Name: 0, dtype: float64\n           >>> print(row['int'].dtype)\n           float64\n           >>> print(df['int'].dtype)\n           int64\n\n           To preserve dtypes while iterating over the rows, it is better\n           to use :meth:`itertuples` which returns namedtuples of the values\n           and which is generally faster than ``iterrows``.\n\n        2. You should **never modify** something you are iterating over.\n           This is not guaranteed to work in all cases. Depending on the\n           data types, the iterator returns a copy and not a view, and writing\n           to it will have no effect.\n        "
        columns = self.columns
        klass = self._constructor_sliced
        for (k, v) in zip(self.index, self.values):
            s = klass(v, index=columns, name=k)
            (yield (k, s))

    def itertuples(self, index=True, name='Pandas'):
        '\n        Iterate over DataFrame rows as namedtuples.\n\n        Parameters\n        ----------\n        index : bool, default True\n            If True, return the index as the first element of the tuple.\n        name : str or None, default "Pandas"\n            The name of the returned namedtuples or None to return regular\n            tuples.\n\n        Returns\n        -------\n        iterator\n            An object to iterate over namedtuples for each row in the\n            DataFrame with the first field possibly being the index and\n            following fields being the column values.\n\n        See Also\n        --------\n        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series)\n            pairs.\n        DataFrame.items : Iterate over (column name, Series) pairs.\n\n        Notes\n        -----\n        The column names will be renamed to positional names if they are\n        invalid Python identifiers, repeated, or start with an underscore.\n        On python versions < 3.7 regular tuples are returned for DataFrames\n        with a large number of columns (>254).\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'num_legs\': [4, 2], \'num_wings\': [0, 2]},\n        ...                   index=[\'dog\', \'hawk\'])\n        >>> df\n              num_legs  num_wings\n        dog          4          0\n        hawk         2          2\n        >>> for row in df.itertuples():\n        ...     print(row)\n        ...\n        Pandas(Index=\'dog\', num_legs=4, num_wings=0)\n        Pandas(Index=\'hawk\', num_legs=2, num_wings=2)\n\n        By setting the `index` parameter to False we can remove the index\n        as the first element of the tuple:\n\n        >>> for row in df.itertuples(index=False):\n        ...     print(row)\n        ...\n        Pandas(num_legs=4, num_wings=0)\n        Pandas(num_legs=2, num_wings=2)\n\n        With the `name` parameter set we set a custom name for the yielded\n        namedtuples:\n\n        >>> for row in df.itertuples(name=\'Animal\'):\n        ...     print(row)\n        ...\n        Animal(Index=\'dog\', num_legs=4, num_wings=0)\n        Animal(Index=\'hawk\', num_legs=2, num_wings=2)\n        '
        arrays = []
        fields = list(self.columns)
        if index:
            arrays.append(self.index)
            fields.insert(0, 'Index')
        arrays.extend((self.iloc[:, k] for k in range(len(self.columns))))
        if (name is not None):
            itertuple = collections.namedtuple(name, fields, rename=True)
            return map(itertuple._make, zip(*arrays))
        return zip(*arrays)

    def __len__(self):
        '\n        Returns length of info axis, but here we use the index.\n        '
        return len(self.index)

    def dot(self, other):
        '\n        Compute the matrix multiplication between the DataFrame and other.\n\n        This method computes the matrix product between the DataFrame and the\n        values of an other Series, DataFrame or a numpy array.\n\n        It can also be called using ``self @ other`` in Python >= 3.5.\n\n        Parameters\n        ----------\n        other : Series, DataFrame or array-like\n            The other object to compute the matrix product with.\n\n        Returns\n        -------\n        Series or DataFrame\n            If other is a Series, return the matrix product between self and\n            other as a Series. If other is a DataFrame or a numpy.array, return\n            the matrix product of self and other in a DataFrame of a np.array.\n\n        See Also\n        --------\n        Series.dot: Similar method for Series.\n\n        Notes\n        -----\n        The dimensions of DataFrame and other must be compatible in order to\n        compute the matrix multiplication. In addition, the column names of\n        DataFrame and the index of other must contain the same values, as they\n        will be aligned prior to the multiplication.\n\n        The dot method for Series computes the inner product, instead of the\n        matrix product here.\n\n        Examples\n        --------\n        Here we multiply a DataFrame with a Series.\n\n        >>> df = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])\n        >>> s = pd.Series([1, 1, 2, 1])\n        >>> df.dot(s)\n        0    -4\n        1     5\n        dtype: int64\n\n        Here we multiply a DataFrame with another DataFrame.\n\n        >>> other = pd.DataFrame([[0, 1], [1, 2], [-1, -1], [2, 0]])\n        >>> df.dot(other)\n            0   1\n        0   1   4\n        1   2   2\n\n        Note that the dot method give the same result as @\n\n        >>> df @ other\n            0   1\n        0   1   4\n        1   2   2\n\n        The dot method works also if other is an np.array.\n\n        >>> arr = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])\n        >>> df.dot(arr)\n            0   1\n        0   1   4\n        1   2   2\n\n        Note how shuffling of the objects does not change the result.\n\n        >>> s2 = s.reindex([1, 0, 2, 3])\n        >>> df.dot(s2)\n        0    -4\n        1     5\n        dtype: int64\n        '
        if isinstance(other, (Series, DataFrame)):
            common = self.columns.union(other.index)
            if ((len(common) > len(self.columns)) or (len(common) > len(other.index))):
                raise ValueError('matrices are not aligned')
            left = self.reindex(columns=common, copy=False)
            right = other.reindex(index=common, copy=False)
            lvals = left.values
            rvals = right._values
        else:
            left = self
            lvals = self.values
            rvals = np.asarray(other)
            if (lvals.shape[1] != rvals.shape[0]):
                raise ValueError(f'Dot product shape mismatch, {lvals.shape} vs {rvals.shape}')
        if isinstance(other, DataFrame):
            return self._constructor(np.dot(lvals, rvals), index=left.index, columns=other.columns)
        elif isinstance(other, Series):
            return self._constructor_sliced(np.dot(lvals, rvals), index=left.index)
        elif isinstance(rvals, (np.ndarray, Index)):
            result = np.dot(lvals, rvals)
            if (result.ndim == 2):
                return self._constructor(result, index=left.index)
            else:
                return self._constructor_sliced(result, index=left.index)
        else:
            raise TypeError(f'unsupported type: {type(other)}')

    def __matmul__(self, other):
        '\n        Matrix multiplication using binary `@` operator in Python>=3.5.\n        '
        return self.dot(other)

    def __rmatmul__(self, other):
        '\n        Matrix multiplication using binary `@` operator in Python>=3.5.\n        '
        try:
            return self.T.dot(np.transpose(other)).T
        except ValueError as err:
            if ('shape mismatch' not in str(err)):
                raise
            msg = f'shapes {np.shape(other)} and {self.shape} not aligned'
            raise ValueError(msg) from err

    @classmethod
    def from_dict(cls, data, orient='columns', dtype=None, columns=None):
        '\n        Construct DataFrame from dict of array-like or dicts.\n\n        Creates DataFrame object from dictionary by columns or by index\n        allowing dtype specification.\n\n        Parameters\n        ----------\n        data : dict\n            Of the form {field : array-like} or {field : dict}.\n        orient : {\'columns\', \'index\'}, default \'columns\'\n            The "orientation" of the data. If the keys of the passed dict\n            should be the columns of the resulting DataFrame, pass \'columns\'\n            (default). Otherwise if the keys should be rows, pass \'index\'.\n        dtype : dtype, default None\n            Data type to force, otherwise infer.\n        columns : list, default None\n            Column labels to use when ``orient=\'index\'``. Raises a ValueError\n            if used with ``orient=\'columns\'``.\n\n        Returns\n        -------\n        DataFrame\n\n        See Also\n        --------\n        DataFrame.from_records : DataFrame from structured ndarray, sequence\n            of tuples or dicts, or DataFrame.\n        DataFrame : DataFrame object creation using constructor.\n\n        Examples\n        --------\n        By default the keys of the dict become the DataFrame columns:\n\n        >>> data = {\'col_1\': [3, 2, 1, 0], \'col_2\': [\'a\', \'b\', \'c\', \'d\']}\n        >>> pd.DataFrame.from_dict(data)\n           col_1 col_2\n        0      3     a\n        1      2     b\n        2      1     c\n        3      0     d\n\n        Specify ``orient=\'index\'`` to create the DataFrame using dictionary\n        keys as rows:\n\n        >>> data = {\'row_1\': [3, 2, 1, 0], \'row_2\': [\'a\', \'b\', \'c\', \'d\']}\n        >>> pd.DataFrame.from_dict(data, orient=\'index\')\n               0  1  2  3\n        row_1  3  2  1  0\n        row_2  a  b  c  d\n\n        When using the \'index\' orientation, the column names can be\n        specified manually:\n\n        >>> pd.DataFrame.from_dict(data, orient=\'index\',\n        ...                        columns=[\'A\', \'B\', \'C\', \'D\'])\n               A  B  C  D\n        row_1  3  2  1  0\n        row_2  a  b  c  d\n        '
        index = None
        orient = orient.lower()
        if (orient == 'index'):
            if (len(data) > 0):
                if isinstance(list(data.values())[0], (Series, dict)):
                    data = _from_nested_dict(data)
                else:
                    (data, index) = (list(data.values()), list(data.keys()))
        elif (orient == 'columns'):
            if (columns is not None):
                raise ValueError("cannot use columns parameter with orient='columns'")
        else:
            raise ValueError('only recognize index or columns for orient')
        return cls(data, index=index, columns=columns, dtype=dtype)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        '\n        Convert the DataFrame to a NumPy array.\n\n        .. versionadded:: 0.24.0\n\n        By default, the dtype of the returned array will be the common NumPy\n        dtype of all types in the DataFrame. For example, if the dtypes are\n        ``float16`` and ``float32``, the results dtype will be ``float32``.\n        This may require copying data and coercing values, which may be\n        expensive.\n\n        Parameters\n        ----------\n        dtype : str or numpy.dtype, optional\n            The dtype to pass to :meth:`numpy.asarray`.\n        copy : bool, default False\n            Whether to ensure that the returned value is not a view on\n            another array. Note that ``copy=False`` does not *ensure* that\n            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that\n            a copy is made, even if not strictly necessary.\n        na_value : Any, optional\n            The value to use for missing values. The default value depends\n            on `dtype` and the dtypes of the DataFrame columns.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        numpy.ndarray\n\n        See Also\n        --------\n        Series.to_numpy : Similar method for Series.\n\n        Examples\n        --------\n        >>> pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()\n        array([[1, 3],\n               [2, 4]])\n\n        With heterogeneous data, the lowest common type will have to\n        be used.\n\n        >>> df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})\n        >>> df.to_numpy()\n        array([[1. , 3. ],\n               [2. , 4.5]])\n\n        For a mix of numeric and non-numeric types, the output array will\n        have object dtype.\n\n        >>> df[\'C\'] = pd.date_range(\'2000\', periods=2)\n        >>> df.to_numpy()\n        array([[1, 3.0, Timestamp(\'2000-01-01 00:00:00\')],\n               [2, 4.5, Timestamp(\'2000-01-02 00:00:00\')]], dtype=object)\n        '
        self._consolidate_inplace()
        result = self._mgr.as_array(transpose=self._AXIS_REVERSED, dtype=dtype, copy=copy, na_value=na_value)
        if (result.dtype is not dtype):
            result = np.array(result, dtype=dtype, copy=False)
        return result

    def to_dict(self, orient='dict', into=dict):
        "\n        Convert the DataFrame to a dictionary.\n\n        The type of the key-value pairs can be customized with the parameters\n        (see below).\n\n        Parameters\n        ----------\n        orient : str {'dict', 'list', 'series', 'split', 'records', 'index'}\n            Determines the type of the values of the dictionary.\n\n            - 'dict' (default) : dict like {column -> {index -> value}}\n            - 'list' : dict like {column -> [values]}\n            - 'series' : dict like {column -> Series(values)}\n            - 'split' : dict like\n              {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}\n            - 'records' : list like\n              [{column -> value}, ... , {column -> value}]\n            - 'index' : dict like {index -> {column -> value}}\n\n            Abbreviations are allowed. `s` indicates `series` and `sp`\n            indicates `split`.\n\n        into : class, default dict\n            The collections.abc.Mapping subclass used for all Mappings\n            in the return value.  Can be the actual class or an empty\n            instance of the mapping type you want.  If you want a\n            collections.defaultdict, you must pass it initialized.\n\n        Returns\n        -------\n        dict, list or collections.abc.Mapping\n            Return a collections.abc.Mapping object representing the DataFrame.\n            The resulting transformation depends on the `orient` parameter.\n\n        See Also\n        --------\n        DataFrame.from_dict: Create a DataFrame from a dictionary.\n        DataFrame.to_json: Convert a DataFrame to JSON format.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'col1': [1, 2],\n        ...                    'col2': [0.5, 0.75]},\n        ...                   index=['row1', 'row2'])\n        >>> df\n              col1  col2\n        row1     1  0.50\n        row2     2  0.75\n        >>> df.to_dict()\n        {'col1': {'row1': 1, 'row2': 2}, 'col2': {'row1': 0.5, 'row2': 0.75}}\n\n        You can specify the return orientation.\n\n        >>> df.to_dict('series')\n        {'col1': row1    1\n                 row2    2\n        Name: col1, dtype: int64,\n        'col2': row1    0.50\n                row2    0.75\n        Name: col2, dtype: float64}\n\n        >>> df.to_dict('split')\n        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],\n         'data': [[1, 0.5], [2, 0.75]]}\n\n        >>> df.to_dict('records')\n        [{'col1': 1, 'col2': 0.5}, {'col1': 2, 'col2': 0.75}]\n\n        >>> df.to_dict('index')\n        {'row1': {'col1': 1, 'col2': 0.5}, 'row2': {'col1': 2, 'col2': 0.75}}\n\n        You can also specify the mapping type.\n\n        >>> from collections import OrderedDict, defaultdict\n        >>> df.to_dict(into=OrderedDict)\n        OrderedDict([('col1', OrderedDict([('row1', 1), ('row2', 2)])),\n                     ('col2', OrderedDict([('row1', 0.5), ('row2', 0.75)]))])\n\n        If you want a `defaultdict`, you need to initialize it:\n\n        >>> dd = defaultdict(list)\n        >>> df.to_dict('records', into=dd)\n        [defaultdict(<class 'list'>, {'col1': 1, 'col2': 0.5}),\n         defaultdict(<class 'list'>, {'col1': 2, 'col2': 0.75})]\n        "
        if (not self.columns.is_unique):
            warnings.warn('DataFrame columns are not unique, some columns will be omitted.', UserWarning, stacklevel=2)
        into_c = com.standardize_mapping(into)
        orient = orient.lower()
        if (orient.startswith(('d', 'l', 's', 'r', 'i')) and (orient not in {'dict', 'list', 'series', 'split', 'records', 'index'})):
            warnings.warn("Using short name for 'orient' is deprecated. Only the options: ('dict', list, 'series', 'split', 'records', 'index') will be used in a future version. Use one of the above to silence this warning.", FutureWarning)
            if orient.startswith('d'):
                orient = 'dict'
            elif orient.startswith('l'):
                orient = 'list'
            elif orient.startswith('sp'):
                orient = 'split'
            elif orient.startswith('s'):
                orient = 'series'
            elif orient.startswith('r'):
                orient = 'records'
            elif orient.startswith('i'):
                orient = 'index'
        if (orient == 'dict'):
            return into_c(((k, v.to_dict(into)) for (k, v) in self.items()))
        elif (orient == 'list'):
            return into_c(((k, v.tolist()) for (k, v) in self.items()))
        elif (orient == 'split'):
            return into_c((('index', self.index.tolist()), ('columns', self.columns.tolist()), ('data', [list(map(maybe_box_datetimelike, t)) for t in self.itertuples(index=False, name=None)])))
        elif (orient == 'series'):
            return into_c(((k, maybe_box_datetimelike(v)) for (k, v) in self.items()))
        elif (orient == 'records'):
            columns = self.columns.tolist()
            rows = (dict(zip(columns, row)) for row in self.itertuples(index=False, name=None))
            return [into_c(((k, maybe_box_datetimelike(v)) for (k, v) in row.items())) for row in rows]
        elif (orient == 'index'):
            if (not self.index.is_unique):
                raise ValueError("DataFrame index must be unique for orient='index'.")
            return into_c(((t[0], dict(zip(self.columns, t[1:]))) for t in self.itertuples(name=None)))
        else:
            raise ValueError(f"orient '{orient}' not understood")

    def to_gbq(self, destination_table, project_id=None, chunksize=None, reauth=False, if_exists='fail', auth_local_webserver=False, table_schema=None, location=None, progress_bar=True, credentials=None):
        "\n        Write a DataFrame to a Google BigQuery table.\n\n        This function requires the `pandas-gbq package\n        <https://pandas-gbq.readthedocs.io>`__.\n\n        See the `How to authenticate with Google BigQuery\n        <https://pandas-gbq.readthedocs.io/en/latest/howto/authentication.html>`__\n        guide for authentication instructions.\n\n        Parameters\n        ----------\n        destination_table : str\n            Name of table to be written, in the form ``dataset.tablename``.\n        project_id : str, optional\n            Google BigQuery Account project ID. Optional when available from\n            the environment.\n        chunksize : int, optional\n            Number of rows to be inserted in each chunk from the dataframe.\n            Set to ``None`` to load the whole dataframe at once.\n        reauth : bool, default False\n            Force Google BigQuery to re-authenticate the user. This is useful\n            if multiple accounts are used.\n        if_exists : str, default 'fail'\n            Behavior when the destination table exists. Value can be one of:\n\n            ``'fail'``\n                If table exists raise pandas_gbq.gbq.TableCreationError.\n            ``'replace'``\n                If table exists, drop it, recreate it, and insert data.\n            ``'append'``\n                If table exists, insert data. Create if does not exist.\n        auth_local_webserver : bool, default False\n            Use the `local webserver flow`_ instead of the `console flow`_\n            when getting user credentials.\n\n            .. _local webserver flow:\n                https://google-auth-oauthlib.readthedocs.io/en/latest/reference/google_auth_oauthlib.flow.html#google_auth_oauthlib.flow.InstalledAppFlow.run_local_server\n            .. _console flow:\n                https://google-auth-oauthlib.readthedocs.io/en/latest/reference/google_auth_oauthlib.flow.html#google_auth_oauthlib.flow.InstalledAppFlow.run_console\n\n            *New in version 0.2.0 of pandas-gbq*.\n        table_schema : list of dicts, optional\n            List of BigQuery table fields to which according DataFrame\n            columns conform to, e.g. ``[{'name': 'col1', 'type':\n            'STRING'},...]``. If schema is not provided, it will be\n            generated according to dtypes of DataFrame columns. See\n            BigQuery API documentation on available names of a field.\n\n            *New in version 0.3.1 of pandas-gbq*.\n        location : str, optional\n            Location where the load job should run. See the `BigQuery locations\n            documentation\n            <https://cloud.google.com/bigquery/docs/dataset-locations>`__ for a\n            list of available locations. The location must match that of the\n            target dataset.\n\n            *New in version 0.5.0 of pandas-gbq*.\n        progress_bar : bool, default True\n            Use the library `tqdm` to show the progress bar for the upload,\n            chunk by chunk.\n\n            *New in version 0.5.0 of pandas-gbq*.\n        credentials : google.auth.credentials.Credentials, optional\n            Credentials for accessing Google APIs. Use this parameter to\n            override default credentials, such as to use Compute Engine\n            :class:`google.auth.compute_engine.Credentials` or Service\n            Account :class:`google.oauth2.service_account.Credentials`\n            directly.\n\n            *New in version 0.8.0 of pandas-gbq*.\n\n            .. versionadded:: 0.24.0\n\n        See Also\n        --------\n        pandas_gbq.to_gbq : This function in the pandas-gbq library.\n        read_gbq : Read a DataFrame from Google BigQuery.\n        "
        from pandas.io import gbq
        gbq.to_gbq(self, destination_table, project_id=project_id, chunksize=chunksize, reauth=reauth, if_exists=if_exists, auth_local_webserver=auth_local_webserver, table_schema=table_schema, location=location, progress_bar=progress_bar, credentials=credentials)

    @classmethod
    def from_records(cls, data, index=None, exclude=None, columns=None, coerce_float=False, nrows=None):
        "\n        Convert structured or record ndarray to DataFrame.\n\n        Creates a DataFrame object from a structured ndarray, sequence of\n        tuples or dicts, or DataFrame.\n\n        Parameters\n        ----------\n        data : structured ndarray, sequence of tuples or dicts, or DataFrame\n            Structured input data.\n        index : str, list of fields, array-like\n            Field of array to use as the index, alternately a specific set of\n            input labels to use.\n        exclude : sequence, default None\n            Columns or fields to exclude.\n        columns : sequence, default None\n            Column names to use. If the passed data do not have names\n            associated with them, this argument provides names for the\n            columns. Otherwise this argument indicates the order of the columns\n            in the result (any names not found in the data will become all-NA\n            columns).\n        coerce_float : bool, default False\n            Attempt to convert values of non-string, non-numeric objects (like\n            decimal.Decimal) to floating point, useful for SQL result sets.\n        nrows : int, default None\n            Number of rows to read if data is an iterator.\n\n        Returns\n        -------\n        DataFrame\n\n        See Also\n        --------\n        DataFrame.from_dict : DataFrame from dict of array-like or dicts.\n        DataFrame : DataFrame object creation using constructor.\n\n        Examples\n        --------\n        Data can be provided as a structured ndarray:\n\n        >>> data = np.array([(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')],\n        ...                 dtype=[('col_1', 'i4'), ('col_2', 'U1')])\n        >>> pd.DataFrame.from_records(data)\n           col_1 col_2\n        0      3     a\n        1      2     b\n        2      1     c\n        3      0     d\n\n        Data can be provided as a list of dicts:\n\n        >>> data = [{'col_1': 3, 'col_2': 'a'},\n        ...         {'col_1': 2, 'col_2': 'b'},\n        ...         {'col_1': 1, 'col_2': 'c'},\n        ...         {'col_1': 0, 'col_2': 'd'}]\n        >>> pd.DataFrame.from_records(data)\n           col_1 col_2\n        0      3     a\n        1      2     b\n        2      1     c\n        3      0     d\n\n        Data can be provided as a list of tuples with corresponding columns:\n\n        >>> data = [(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')]\n        >>> pd.DataFrame.from_records(data, columns=['col_1', 'col_2'])\n           col_1 col_2\n        0      3     a\n        1      2     b\n        2      1     c\n        3      0     d\n        "
        if (columns is not None):
            columns = ensure_index(columns)
        if is_iterator(data):
            if (nrows == 0):
                return cls()
            try:
                first_row = next(data)
            except StopIteration:
                return cls(index=index, columns=columns)
            dtype = None
            if (hasattr(first_row, 'dtype') and first_row.dtype.names):
                dtype = first_row.dtype
            values = [first_row]
            if (nrows is None):
                values += data
            else:
                values.extend(itertools.islice(data, (nrows - 1)))
            if (dtype is not None):
                data = np.array(values, dtype=dtype)
            else:
                data = values
        if isinstance(data, dict):
            if (columns is None):
                columns = arr_columns = ensure_index(sorted(data))
                arrays = [data[k] for k in columns]
            else:
                arrays = []
                arr_columns_list = []
                for (k, v) in data.items():
                    if (k in columns):
                        arr_columns_list.append(k)
                        arrays.append(v)
                (arrays, arr_columns) = reorder_arrays(arrays, arr_columns_list, columns)
        elif isinstance(data, (np.ndarray, DataFrame)):
            (arrays, columns) = to_arrays(data, columns)
            if (columns is not None):
                columns = ensure_index(columns)
            arr_columns = columns
        else:
            (arrays, arr_columns) = to_arrays(data, columns)
            if coerce_float:
                for (i, arr) in enumerate(arrays):
                    if (arr.dtype == object):
                        arrays[i] = lib.maybe_convert_objects(arr, try_float=True)
            arr_columns = ensure_index(arr_columns)
            if (columns is not None):
                columns = ensure_index(columns)
            else:
                columns = arr_columns
        if (exclude is None):
            exclude = set()
        else:
            exclude = set(exclude)
        result_index = None
        if (index is not None):
            if (isinstance(index, str) or (not hasattr(index, '__iter__'))):
                i = columns.get_loc(index)
                exclude.add(index)
                if (len(arrays) > 0):
                    result_index = Index(arrays[i], name=index)
                else:
                    result_index = Index([], name=index)
            else:
                try:
                    index_data = [arrays[arr_columns.get_loc(field)] for field in index]
                except (KeyError, TypeError):
                    result_index = index
                else:
                    result_index = ensure_index_from_sequences(index_data, names=index)
                    exclude.update(index)
        if any(exclude):
            arr_exclude = [x for x in exclude if (x in arr_columns)]
            to_remove = [arr_columns.get_loc(col) for col in arr_exclude]
            arrays = [v for (i, v) in enumerate(arrays) if (i not in to_remove)]
            arr_columns = arr_columns.drop(arr_exclude)
            columns = columns.drop(exclude)
        mgr = arrays_to_mgr(arrays, arr_columns, result_index, columns)
        return cls(mgr)

    def to_records(self, index=True, column_dtypes=None, index_dtypes=None):
        '\n        Convert DataFrame to a NumPy record array.\n\n        Index will be included as the first field of the record array if\n        requested.\n\n        Parameters\n        ----------\n        index : bool, default True\n            Include index in resulting record array, stored in \'index\'\n            field or using the index label, if set.\n        column_dtypes : str, type, dict, default None\n            .. versionadded:: 0.24.0\n\n            If a string or type, the data type to store all columns. If\n            a dictionary, a mapping of column names and indices (zero-indexed)\n            to specific data types.\n        index_dtypes : str, type, dict, default None\n            .. versionadded:: 0.24.0\n\n            If a string or type, the data type to store all index levels. If\n            a dictionary, a mapping of index level names and indices\n            (zero-indexed) to specific data types.\n\n            This mapping is applied only if `index=True`.\n\n        Returns\n        -------\n        numpy.recarray\n            NumPy ndarray with the DataFrame labels as fields and each row\n            of the DataFrame as entries.\n\n        See Also\n        --------\n        DataFrame.from_records: Convert structured or record ndarray\n            to DataFrame.\n        numpy.recarray: An ndarray that allows field access using\n            attributes, analogous to typed columns in a\n            spreadsheet.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'A\': [1, 2], \'B\': [0.5, 0.75]},\n        ...                   index=[\'a\', \'b\'])\n        >>> df\n           A     B\n        a  1  0.50\n        b  2  0.75\n        >>> df.to_records()\n        rec.array([(\'a\', 1, 0.5 ), (\'b\', 2, 0.75)],\n                  dtype=[(\'index\', \'O\'), (\'A\', \'<i8\'), (\'B\', \'<f8\')])\n\n        If the DataFrame index has no label then the recarray field name\n        is set to \'index\'. If the index has a label then this is used as the\n        field name:\n\n        >>> df.index = df.index.rename("I")\n        >>> df.to_records()\n        rec.array([(\'a\', 1, 0.5 ), (\'b\', 2, 0.75)],\n                  dtype=[(\'I\', \'O\'), (\'A\', \'<i8\'), (\'B\', \'<f8\')])\n\n        The index can be excluded from the record array:\n\n        >>> df.to_records(index=False)\n        rec.array([(1, 0.5 ), (2, 0.75)],\n                  dtype=[(\'A\', \'<i8\'), (\'B\', \'<f8\')])\n\n        Data types can be specified for the columns:\n\n        >>> df.to_records(column_dtypes={"A": "int32"})\n        rec.array([(\'a\', 1, 0.5 ), (\'b\', 2, 0.75)],\n                  dtype=[(\'I\', \'O\'), (\'A\', \'<i4\'), (\'B\', \'<f8\')])\n\n        As well as for the index:\n\n        >>> df.to_records(index_dtypes="<S2")\n        rec.array([(b\'a\', 1, 0.5 ), (b\'b\', 2, 0.75)],\n                  dtype=[(\'I\', \'S2\'), (\'A\', \'<i8\'), (\'B\', \'<f8\')])\n\n        >>> index_dtypes = f"<S{df.index.str.len().max()}"\n        >>> df.to_records(index_dtypes=index_dtypes)\n        rec.array([(b\'a\', 1, 0.5 ), (b\'b\', 2, 0.75)],\n                  dtype=[(\'I\', \'S1\'), (\'A\', \'<i8\'), (\'B\', \'<f8\')])\n        '
        if index:
            if isinstance(self.index, MultiIndex):
                ix_vals = list(map(np.array, zip(*self.index._values)))
            else:
                ix_vals = [self.index.values]
            arrays = (ix_vals + [np.asarray(self.iloc[:, i]) for i in range(len(self.columns))])
            count = 0
            index_names = list(self.index.names)
            if isinstance(self.index, MultiIndex):
                for (i, n) in enumerate(index_names):
                    if (n is None):
                        index_names[i] = f'level_{count}'
                        count += 1
            elif (index_names[0] is None):
                index_names = ['index']
            names = [str(name) for name in itertools.chain(index_names, self.columns)]
        else:
            arrays = [np.asarray(self.iloc[:, i]) for i in range(len(self.columns))]
            names = [str(c) for c in self.columns]
            index_names = []
        index_len = len(index_names)
        formats = []
        for (i, v) in enumerate(arrays):
            index = i
            if (index < index_len):
                dtype_mapping = index_dtypes
                name = index_names[index]
            else:
                index -= index_len
                dtype_mapping = column_dtypes
                name = self.columns[index]
            if is_dict_like(dtype_mapping):
                if (name in dtype_mapping):
                    dtype_mapping = dtype_mapping[name]
                elif (index in dtype_mapping):
                    dtype_mapping = dtype_mapping[index]
                else:
                    dtype_mapping = None
            if (dtype_mapping is None):
                formats.append(v.dtype)
            elif isinstance(dtype_mapping, (type, np.dtype, str)):
                formats.append(dtype_mapping)
            else:
                element = ('row' if (i < index_len) else 'column')
                msg = f'Invalid dtype {dtype_mapping} specified for {element} {name}'
                raise ValueError(msg)
        return np.rec.fromarrays(arrays, dtype={'names': names, 'formats': formats})

    @classmethod
    def _from_arrays(cls, arrays, columns, index, dtype=None, verify_integrity=True):
        '\n        Create DataFrame from a list of arrays corresponding to the columns.\n\n        Parameters\n        ----------\n        arrays : list-like of arrays\n            Each array in the list corresponds to one column, in order.\n        columns : list-like, Index\n            The column names for the resulting DataFrame.\n        index : list-like, Index\n            The rows labels for the resulting DataFrame.\n        dtype : dtype, optional\n            Optional dtype to enforce for all arrays.\n        verify_integrity : bool, default True\n            Validate and homogenize all input. If set to False, it is assumed\n            that all elements of `arrays` are actual arrays how they will be\n            stored in a block (numpy ndarray or ExtensionArray), have the same\n            length as and are aligned with the index, and that `columns` and\n            `index` are ensured to be an Index object.\n\n        Returns\n        -------\n        DataFrame\n        '
        if (dtype is not None):
            dtype = pandas_dtype(dtype)
        mgr = arrays_to_mgr(arrays, columns, index, columns, dtype=dtype, verify_integrity=verify_integrity)
        return cls(mgr)

    @doc(storage_options=generic._shared_docs['storage_options'])
    @deprecate_kwarg(old_arg_name='fname', new_arg_name='path')
    def to_stata(self, path, convert_dates=None, write_index=True, byteorder=None, time_stamp=None, data_label=None, variable_labels=None, version=114, convert_strl=None, compression='infer', storage_options=None):
        '\n        Export DataFrame object to Stata dta format.\n\n        Writes the DataFrame to a Stata dataset file.\n        "dta" files contain a Stata dataset.\n\n        Parameters\n        ----------\n        path : str, buffer or path object\n            String, path object (pathlib.Path or py._path.local.LocalPath) or\n            object implementing a binary write() function. If using a buffer\n            then the buffer will not be automatically closed after the file\n            data has been written.\n\n            .. versionchanged:: 1.0.0\n\n            Previously this was "fname"\n\n        convert_dates : dict\n            Dictionary mapping columns containing datetime types to stata\n            internal format to use when writing the dates. Options are \'tc\',\n            \'td\', \'tm\', \'tw\', \'th\', \'tq\', \'ty\'. Column can be either an integer\n            or a name. Datetime columns that do not have a conversion type\n            specified will be converted to \'tc\'. Raises NotImplementedError if\n            a datetime column has timezone information.\n        write_index : bool\n            Write the index to Stata dataset.\n        byteorder : str\n            Can be ">", "<", "little", or "big". default is `sys.byteorder`.\n        time_stamp : datetime\n            A datetime to use as file creation date.  Default is the current\n            time.\n        data_label : str, optional\n            A label for the data set.  Must be 80 characters or smaller.\n        variable_labels : dict\n            Dictionary containing columns as keys and variable labels as\n            values. Each label must be 80 characters or smaller.\n        version : {{114, 117, 118, 119, None}}, default 114\n            Version to use in the output dta file. Set to None to let pandas\n            decide between 118 or 119 formats depending on the number of\n            columns in the frame. Version 114 can be read by Stata 10 and\n            later. Version 117 can be read by Stata 13 or later. Version 118\n            is supported in Stata 14 and later. Version 119 is supported in\n            Stata 15 and later. Version 114 limits string variables to 244\n            characters or fewer while versions 117 and later allow strings\n            with lengths up to 2,000,000 characters. Versions 118 and 119\n            support Unicode characters, and version 119 supports more than\n            32,767 variables.\n\n            Version 119 should usually only be used when the number of\n            variables exceeds the capacity of dta format 118. Exporting\n            smaller datasets in format 119 may have unintended consequences,\n            and, as of November 2020, Stata SE cannot read version 119 files.\n\n            .. versionchanged:: 1.0.0\n\n                Added support for formats 118 and 119.\n\n        convert_strl : list, optional\n            List of column names to convert to string columns to Stata StrL\n            format. Only available if version is 117.  Storing strings in the\n            StrL format can produce smaller dta files if strings have more than\n            8 characters and values are repeated.\n        compression : str or dict, default \'infer\'\n            For on-the-fly compression of the output dta. If string, specifies\n            compression mode. If dict, value at key \'method\' specifies\n            compression mode. Compression mode must be one of {{\'infer\', \'gzip\',\n            \'bz2\', \'zip\', \'xz\', None}}. If compression mode is \'infer\' and\n            `fname` is path-like, then detect compression from the following\n            extensions: \'.gz\', \'.bz2\', \'.zip\', or \'.xz\' (otherwise no\n            compression). If dict and compression mode is one of {{\'zip\',\n            \'gzip\', \'bz2\'}}, or inferred as one of the above, other entries\n            passed as additional compression options.\n\n            .. versionadded:: 1.1.0\n\n        {storage_options}\n\n            .. versionadded:: 1.2.0\n\n        Raises\n        ------\n        NotImplementedError\n            * If datetimes contain timezone information\n            * Column dtype is not representable in Stata\n        ValueError\n            * Columns listed in convert_dates are neither datetime64[ns]\n              or datetime.datetime\n            * Column listed in convert_dates is not in DataFrame\n            * Categorical label contains more than 32,000 characters\n\n        See Also\n        --------\n        read_stata : Import Stata data files.\n        io.stata.StataWriter : Low-level writer for Stata data files.\n        io.stata.StataWriter117 : Low-level writer for version 117 files.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({{\'animal\': [\'falcon\', \'parrot\', \'falcon\',\n        ...                               \'parrot\'],\n        ...                    \'speed\': [350, 18, 361, 15]}})\n        >>> df.to_stata(\'animals.dta\')  # doctest: +SKIP\n        '
        if (version not in (114, 117, 118, 119, None)):
            raise ValueError('Only formats 114, 117, 118 and 119 are supported.')
        if (version == 114):
            if (convert_strl is not None):
                raise ValueError('strl is not supported in format 114')
            from pandas.io.stata import StataWriter as statawriter
        elif (version == 117):
            from pandas.io.stata import StataWriter117 as statawriter
        else:
            from pandas.io.stata import StataWriterUTF8 as statawriter
        kwargs: Dict[(str, Any)] = {}
        if ((version is None) or (version >= 117)):
            kwargs['convert_strl'] = convert_strl
        if ((version is None) or (version >= 118)):
            kwargs['version'] = version
        writer = statawriter(path, self, convert_dates=convert_dates, byteorder=byteorder, time_stamp=time_stamp, data_label=data_label, write_index=write_index, variable_labels=variable_labels, compression=compression, storage_options=storage_options, **kwargs)
        writer.write_file()

    @deprecate_kwarg(old_arg_name='fname', new_arg_name='path')
    def to_feather(self, path, **kwargs):
        '\n        Write a DataFrame to the binary Feather format.\n\n        Parameters\n        ----------\n        path : str or file-like object\n            If a string, it will be used as Root Directory path.\n        **kwargs :\n            Additional keywords passed to :func:`pyarrow.feather.write_feather`.\n            Starting with pyarrow 0.17, this includes the `compression`,\n            `compression_level`, `chunksize` and `version` keywords.\n\n            .. versionadded:: 1.1.0\n        '
        from pandas.io.feather_format import to_feather
        to_feather(self, path, **kwargs)

    @doc(Series.to_markdown, klass=_shared_doc_kwargs['klass'], storage_options=_shared_docs['storage_options'], examples='Examples\n        --------\n        >>> df = pd.DataFrame(\n        ...     data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}\n        ... )\n        >>> print(df.to_markdown())\n        |    | animal_1   | animal_2   |\n        |---:|:-----------|:-----------|\n        |  0 | elk        | dog        |\n        |  1 | pig        | quetzal    |\n\n        Output markdown with a tabulate option.\n\n        >>> print(df.to_markdown(tablefmt="grid"))\n        +----+------------+------------+\n        |    | animal_1   | animal_2   |\n        +====+============+============+\n        |  0 | elk        | dog        |\n        +----+------------+------------+\n        |  1 | pig        | quetzal    |\n        +----+------------+------------+\n        ')
    def to_markdown(self, buf=None, mode='wt', index=True, storage_options=None, **kwargs):
        if ('showindex' in kwargs):
            warnings.warn("'showindex' is deprecated. Only 'index' will be used in a future version. Use 'index' to silence this warning.", FutureWarning, stacklevel=2)
        kwargs.setdefault('headers', 'keys')
        kwargs.setdefault('tablefmt', 'pipe')
        kwargs.setdefault('showindex', index)
        tabulate = import_optional_dependency('tabulate')
        result = tabulate.tabulate(self, **kwargs)
        if (buf is None):
            return result
        with get_handle(buf, mode, storage_options=storage_options) as handles:
            assert (not isinstance(handles.handle, (str, mmap.mmap)))
            handles.handle.writelines(result)
        return None

    @doc(storage_options=generic._shared_docs['storage_options'])
    @deprecate_kwarg(old_arg_name='fname', new_arg_name='path')
    def to_parquet(self, path=None, engine='auto', compression='snappy', index=None, partition_cols=None, storage_options=None, **kwargs):
        '\n        Write a DataFrame to the binary parquet format.\n\n        This function writes the dataframe as a `parquet file\n        <https://parquet.apache.org/>`_. You can choose different parquet\n        backends, and have the option of compression. See\n        :ref:`the user guide <io.parquet>` for more details.\n\n        Parameters\n        ----------\n        path : str or file-like object, default None\n            If a string, it will be used as Root Directory path\n            when writing a partitioned dataset. By file-like object,\n            we refer to objects with a write() method, such as a file handle\n            (e.g. via builtin open function) or io.BytesIO. The engine\n            fastparquet does not accept file-like objects. If path is None,\n            a bytes object is returned.\n\n            .. versionchanged:: 1.2.0\n\n            Previously this was "fname"\n\n        engine : {{\'auto\', \'pyarrow\', \'fastparquet\'}}, default \'auto\'\n            Parquet library to use. If \'auto\', then the option\n            ``io.parquet.engine`` is used. The default ``io.parquet.engine``\n            behavior is to try \'pyarrow\', falling back to \'fastparquet\' if\n            \'pyarrow\' is unavailable.\n        compression : {{\'snappy\', \'gzip\', \'brotli\', None}}, default \'snappy\'\n            Name of the compression to use. Use ``None`` for no compression.\n        index : bool, default None\n            If ``True``, include the dataframe\'s index(es) in the file output.\n            If ``False``, they will not be written to the file.\n            If ``None``, similar to ``True`` the dataframe\'s index(es)\n            will be saved. However, instead of being saved as values,\n            the RangeIndex will be stored as a range in the metadata so it\n            doesn\'t require much space and is faster. Other indexes will\n            be included as columns in the file output.\n\n            .. versionadded:: 0.24.0\n\n        partition_cols : list, optional, default None\n            Column names by which to partition the dataset.\n            Columns are partitioned in the order they are given.\n            Must be None if path is not a string.\n\n            .. versionadded:: 0.24.0\n\n        {storage_options}\n\n            .. versionadded:: 1.2.0\n\n        **kwargs\n            Additional arguments passed to the parquet library. See\n            :ref:`pandas io <io.parquet>` for more details.\n\n        Returns\n        -------\n        bytes if no path argument is provided else None\n\n        See Also\n        --------\n        read_parquet : Read a parquet file.\n        DataFrame.to_csv : Write a csv file.\n        DataFrame.to_sql : Write to a sql table.\n        DataFrame.to_hdf : Write to hdf.\n\n        Notes\n        -----\n        This function requires either the `fastparquet\n        <https://pypi.org/project/fastparquet>`_ or `pyarrow\n        <https://arrow.apache.org/docs/python/>`_ library.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(data={{\'col1\': [1, 2], \'col2\': [3, 4]}})\n        >>> df.to_parquet(\'df.parquet.gzip\',\n        ...               compression=\'gzip\')  # doctest: +SKIP\n        >>> pd.read_parquet(\'df.parquet.gzip\')  # doctest: +SKIP\n           col1  col2\n        0     1     3\n        1     2     4\n\n        If you want to get a buffer to the parquet content you can use a io.BytesIO\n        object, as long as you don\'t use partition_cols, which creates multiple files.\n\n        >>> import io\n        >>> f = io.BytesIO()\n        >>> df.to_parquet(f)\n        >>> f.seek(0)\n        0\n        >>> content = f.read()\n        '
        from pandas.io.parquet import to_parquet
        return to_parquet(self, path, engine, compression=compression, index=index, partition_cols=partition_cols, storage_options=storage_options, **kwargs)

    @Substitution(header_type='bool', header='Whether to print column labels, default True', col_space_type='str or int, list or dict of int or str', col_space='The minimum width of each column in CSS length units.  An int is assumed to be px units.\n\n            .. versionadded:: 0.25.0\n                Ability to use str')
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    def to_html(self, buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', bold_rows=True, classes=None, escape=True, notebook=False, border=None, table_id=None, render_links=False, encoding=None):
        '\n        Render a DataFrame as an HTML table.\n        %(shared_params)s\n        bold_rows : bool, default True\n            Make the row labels bold in the output.\n        classes : str or list or tuple, default None\n            CSS class(es) to apply to the resulting html table.\n        escape : bool, default True\n            Convert the characters <, >, and & to HTML-safe sequences.\n        notebook : {True, False}, default False\n            Whether the generated HTML is for IPython Notebook.\n        border : int\n            A ``border=border`` attribute is included in the opening\n            `<table>` tag. Default ``pd.options.display.html.border``.\n        encoding : str, default "utf-8"\n            Set character encoding.\n\n            .. versionadded:: 1.0\n\n        table_id : str, optional\n            A css id is included in the opening `<table>` tag if specified.\n        render_links : bool, default False\n            Convert URLs to HTML links.\n\n            .. versionadded:: 0.24.0\n        %(returns)s\n        See Also\n        --------\n        to_string : Convert DataFrame to a string.\n        '
        if ((justify is not None) and (justify not in fmt._VALID_JUSTIFY_PARAMETERS)):
            raise ValueError('Invalid value for justify parameter')
        formatter = fmt.DataFrameFormatter(self, columns=columns, col_space=col_space, na_rep=na_rep, header=header, index=index, formatters=formatters, float_format=float_format, bold_rows=bold_rows, sparsify=sparsify, justify=justify, index_names=index_names, escape=escape, decimal=decimal, max_rows=max_rows, max_cols=max_cols, show_dimensions=show_dimensions)
        return fmt.DataFrameRenderer(formatter).to_html(buf=buf, classes=classes, notebook=notebook, border=border, encoding=encoding, table_id=table_id, render_links=render_links)

    @Substitution(klass='DataFrame', type_sub=' and columns', max_cols_sub=dedent('            max_cols : int, optional\n                When to switch from the verbose to the truncated output. If the\n                DataFrame has more than `max_cols` columns, the truncated output\n                is used. By default, the setting in\n                ``pandas.options.display.max_info_columns`` is used.'), show_counts_sub=dedent('            show_counts : bool, optional\n                Whether to show the non-null counts. By default, this is shown\n                only if the DataFrame is smaller than\n                ``pandas.options.display.max_info_rows`` and\n                ``pandas.options.display.max_info_columns``. A value of True always\n                shows the counts, and False never shows the counts.\n            null_counts : bool, optional\n                .. deprecated:: 1.2.0\n                    Use show_counts instead.'), examples_sub=dedent('            >>> int_values = [1, 2, 3, 4, 5]\n            >>> text_values = [\'alpha\', \'beta\', \'gamma\', \'delta\', \'epsilon\']\n            >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]\n            >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,\n            ...                   "float_col": float_values})\n            >>> df\n                int_col text_col  float_col\n            0        1    alpha       0.00\n            1        2     beta       0.25\n            2        3    gamma       0.50\n            3        4    delta       0.75\n            4        5  epsilon       1.00\n\n            Prints information of all columns:\n\n            >>> df.info(verbose=True)\n            <class \'pandas.core.frame.DataFrame\'>\n            RangeIndex: 5 entries, 0 to 4\n            Data columns (total 3 columns):\n             #   Column     Non-Null Count  Dtype\n            ---  ------     --------------  -----\n             0   int_col    5 non-null      int64\n             1   text_col   5 non-null      object\n             2   float_col  5 non-null      float64\n            dtypes: float64(1), int64(1), object(1)\n            memory usage: 248.0+ bytes\n\n            Prints a summary of columns count and its dtypes but not per column\n            information:\n\n            >>> df.info(verbose=False)\n            <class \'pandas.core.frame.DataFrame\'>\n            RangeIndex: 5 entries, 0 to 4\n            Columns: 3 entries, int_col to float_col\n            dtypes: float64(1), int64(1), object(1)\n            memory usage: 248.0+ bytes\n\n            Pipe output of DataFrame.info to buffer instead of sys.stdout, get\n            buffer content and writes to a text file:\n\n            >>> import io\n            >>> buffer = io.StringIO()\n            >>> df.info(buf=buffer)\n            >>> s = buffer.getvalue()\n            >>> with open("df_info.txt", "w",\n            ...           encoding="utf-8") as f:  # doctest: +SKIP\n            ...     f.write(s)\n            260\n\n            The `memory_usage` parameter allows deep introspection mode, specially\n            useful for big DataFrames and fine-tune memory optimization:\n\n            >>> random_strings_array = np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)\n            >>> df = pd.DataFrame({\n            ...     \'column_1\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6),\n            ...     \'column_2\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6),\n            ...     \'column_3\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)\n            ... })\n            >>> df.info()\n            <class \'pandas.core.frame.DataFrame\'>\n            RangeIndex: 1000000 entries, 0 to 999999\n            Data columns (total 3 columns):\n             #   Column    Non-Null Count    Dtype\n            ---  ------    --------------    -----\n             0   column_1  1000000 non-null  object\n             1   column_2  1000000 non-null  object\n             2   column_3  1000000 non-null  object\n            dtypes: object(3)\n            memory usage: 22.9+ MB\n\n            >>> df.info(memory_usage=\'deep\')\n            <class \'pandas.core.frame.DataFrame\'>\n            RangeIndex: 1000000 entries, 0 to 999999\n            Data columns (total 3 columns):\n             #   Column    Non-Null Count    Dtype\n            ---  ------    --------------    -----\n             0   column_1  1000000 non-null  object\n             1   column_2  1000000 non-null  object\n             2   column_3  1000000 non-null  object\n            dtypes: object(3)\n            memory usage: 165.9 MB'), see_also_sub=dedent('            DataFrame.describe: Generate descriptive statistics of DataFrame\n                columns.\n            DataFrame.memory_usage: Memory usage of DataFrame columns.'), version_added_sub='')
    @doc(BaseInfo.render)
    def info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None):
        if (null_counts is not None):
            if (show_counts is not None):
                raise ValueError('null_counts used with show_counts. Use show_counts.')
            warnings.warn('null_counts is deprecated. Use show_counts instead', FutureWarning, stacklevel=2)
            show_counts = null_counts
        info = DataFrameInfo(data=self, memory_usage=memory_usage)
        info.render(buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts)

    def memory_usage(self, index=True, deep=False):
        "\n        Return the memory usage of each column in bytes.\n\n        The memory usage can optionally include the contribution of\n        the index and elements of `object` dtype.\n\n        This value is displayed in `DataFrame.info` by default. This can be\n        suppressed by setting ``pandas.options.display.memory_usage`` to False.\n\n        Parameters\n        ----------\n        index : bool, default True\n            Specifies whether to include the memory usage of the DataFrame's\n            index in returned Series. If ``index=True``, the memory usage of\n            the index is the first item in the output.\n        deep : bool, default False\n            If True, introspect the data deeply by interrogating\n            `object` dtypes for system-level memory consumption, and include\n            it in the returned values.\n\n        Returns\n        -------\n        Series\n            A Series whose index is the original column names and whose values\n            is the memory usage of each column in bytes.\n\n        See Also\n        --------\n        numpy.ndarray.nbytes : Total bytes consumed by the elements of an\n            ndarray.\n        Series.memory_usage : Bytes consumed by a Series.\n        Categorical : Memory-efficient array for string values with\n            many repeated values.\n        DataFrame.info : Concise summary of a DataFrame.\n\n        Examples\n        --------\n        >>> dtypes = ['int64', 'float64', 'complex128', 'object', 'bool']\n        >>> data = dict([(t, np.ones(shape=5000, dtype=int).astype(t))\n        ...              for t in dtypes])\n        >>> df = pd.DataFrame(data)\n        >>> df.head()\n           int64  float64            complex128  object  bool\n        0      1      1.0              1.0+0.0j       1  True\n        1      1      1.0              1.0+0.0j       1  True\n        2      1      1.0              1.0+0.0j       1  True\n        3      1      1.0              1.0+0.0j       1  True\n        4      1      1.0              1.0+0.0j       1  True\n\n        >>> df.memory_usage()\n        Index           128\n        int64         40000\n        float64       40000\n        complex128    80000\n        object        40000\n        bool           5000\n        dtype: int64\n\n        >>> df.memory_usage(index=False)\n        int64         40000\n        float64       40000\n        complex128    80000\n        object        40000\n        bool           5000\n        dtype: int64\n\n        The memory footprint of `object` dtype columns is ignored by default:\n\n        >>> df.memory_usage(deep=True)\n        Index            128\n        int64          40000\n        float64        40000\n        complex128     80000\n        object        180000\n        bool            5000\n        dtype: int64\n\n        Use a Categorical for efficient storage of an object-dtype column with\n        many repeated values.\n\n        >>> df['object'].astype('category').memory_usage(deep=True)\n        5244\n        "
        result = self._constructor_sliced([c.memory_usage(index=False, deep=deep) for (col, c) in self.items()], index=self.columns)
        if index:
            result = self._constructor_sliced(self.index.memory_usage(deep=deep), index=['Index']).append(result)
        return result

    def transpose(self, *args, copy=False):
        "\n        Transpose index and columns.\n\n        Reflect the DataFrame over its main diagonal by writing rows as columns\n        and vice-versa. The property :attr:`.T` is an accessor to the method\n        :meth:`transpose`.\n\n        Parameters\n        ----------\n        *args : tuple, optional\n            Accepted for compatibility with NumPy.\n        copy : bool, default False\n            Whether to copy the data after transposing, even for DataFrames\n            with a single dtype.\n\n            Note that a copy is always required for mixed dtype DataFrames,\n            or for DataFrames with any extension types.\n\n        Returns\n        -------\n        DataFrame\n            The transposed DataFrame.\n\n        See Also\n        --------\n        numpy.transpose : Permute the dimensions of a given array.\n\n        Notes\n        -----\n        Transposing a DataFrame with mixed dtypes will result in a homogeneous\n        DataFrame with the `object` dtype. In such a case, a copy of the data\n        is always made.\n\n        Examples\n        --------\n        **Square DataFrame with homogeneous dtype**\n\n        >>> d1 = {'col1': [1, 2], 'col2': [3, 4]}\n        >>> df1 = pd.DataFrame(data=d1)\n        >>> df1\n           col1  col2\n        0     1     3\n        1     2     4\n\n        >>> df1_transposed = df1.T # or df1.transpose()\n        >>> df1_transposed\n              0  1\n        col1  1  2\n        col2  3  4\n\n        When the dtype is homogeneous in the original DataFrame, we get a\n        transposed DataFrame with the same dtype:\n\n        >>> df1.dtypes\n        col1    int64\n        col2    int64\n        dtype: object\n        >>> df1_transposed.dtypes\n        0    int64\n        1    int64\n        dtype: object\n\n        **Non-square DataFrame with mixed dtypes**\n\n        >>> d2 = {'name': ['Alice', 'Bob'],\n        ...       'score': [9.5, 8],\n        ...       'employed': [False, True],\n        ...       'kids': [0, 0]}\n        >>> df2 = pd.DataFrame(data=d2)\n        >>> df2\n            name  score  employed  kids\n        0  Alice    9.5     False     0\n        1    Bob    8.0      True     0\n\n        >>> df2_transposed = df2.T # or df2.transpose()\n        >>> df2_transposed\n                      0     1\n        name      Alice   Bob\n        score       9.5   8.0\n        employed  False  True\n        kids          0     0\n\n        When the DataFrame has mixed dtypes, we get a transposed DataFrame with\n        the `object` dtype:\n\n        >>> df2.dtypes\n        name         object\n        score       float64\n        employed       bool\n        kids          int64\n        dtype: object\n        >>> df2_transposed.dtypes\n        0    object\n        1    object\n        dtype: object\n        "
        nv.validate_transpose(args, {})
        dtypes = list(self.dtypes)
        if (self._is_homogeneous_type and dtypes and is_extension_array_dtype(dtypes[0])):
            dtype = dtypes[0]
            arr_type = dtype.construct_array_type()
            values = self.values
            new_values = [arr_type._from_sequence(row, dtype=dtype) for row in values]
            result = self._constructor(dict(zip(self.index, new_values)), index=self.columns)
        else:
            new_values = self.values.T
            if copy:
                new_values = new_values.copy()
            result = self._constructor(new_values, index=self.columns, columns=self.index)
        return result.__finalize__(self, method='transpose')

    @property
    def T(self):
        return self.transpose()

    def _ixs(self, i, axis=0):
        '\n        Parameters\n        ----------\n        i : int\n        axis : int\n\n        Notes\n        -----\n        If slice passed, the resulting data will be a view.\n        '
        if (axis == 0):
            new_values = self._mgr.fast_xs(i)
            copy = (isinstance(new_values, np.ndarray) and (new_values.base is None))
            result = self._constructor_sliced(new_values, index=self.columns, name=self.index[i], dtype=new_values.dtype)
            result._set_is_copy(self, copy=copy)
            return result
        else:
            label = self.columns[i]
            values = self._mgr.iget(i)
            result = self._box_col_values(values, i)
            result._set_as_cached(label, self)
            return result

    def _get_column_array(self, i):
        "\n        Get the values of the i'th column (ndarray or ExtensionArray, as stored\n        in the Block)\n        "
        return self._mgr.iget_values(i)

    def _iter_column_arrays(self):
        '\n        Iterate over the arrays of all columns in order.\n        This returns the values as stored in the Block (ndarray or ExtensionArray).\n        '
        for i in range(len(self.columns)):
            (yield self._get_column_array(i))

    def __getitem__(self, key):
        key = lib.item_from_zerodim(key)
        key = com.apply_if_callable(key, self)
        if is_hashable(key):
            if (self.columns.is_unique and (key in self.columns)):
                if isinstance(self.columns, MultiIndex):
                    return self._getitem_multilevel(key)
                return self._get_item_cache(key)
        indexer = convert_to_index_sliceable(self, key)
        if (indexer is not None):
            if isinstance(indexer, np.ndarray):
                indexer = lib.maybe_indices_to_slice(indexer.astype(np.intp, copy=False), len(self))
            return self._slice(indexer, axis=0)
        if isinstance(key, DataFrame):
            return self.where(key)
        if com.is_bool_indexer(key):
            return self._getitem_bool_array(key)
        is_single_key = (isinstance(key, tuple) or (not is_list_like(key)))
        if is_single_key:
            if (self.columns.nlevels > 1):
                return self._getitem_multilevel(key)
            indexer = self.columns.get_loc(key)
            if is_integer(indexer):
                indexer = [indexer]
        else:
            if is_iterator(key):
                key = list(key)
            indexer = self.loc._get_listlike_indexer(key, axis=1, raise_missing=True)[1]
        if (getattr(indexer, 'dtype', None) == bool):
            indexer = np.where(indexer)[0]
        data = self._take_with_is_copy(indexer, axis=1)
        if is_single_key:
            if ((data.shape[1] == 1) and (not isinstance(self.columns, MultiIndex))):
                data = data._get_item_cache(key)
        return data

    def _getitem_bool_array(self, key):
        if (isinstance(key, Series) and (not key.index.equals(self.index))):
            warnings.warn('Boolean Series key will be reindexed to match DataFrame index.', UserWarning, stacklevel=3)
        elif (len(key) != len(self.index)):
            raise ValueError(f'Item wrong length {len(key)} instead of {len(self.index)}.')
        key = check_bool_indexer(self.index, key)
        indexer = key.nonzero()[0]
        return self._take_with_is_copy(indexer, axis=0)

    def _getitem_multilevel(self, key):
        loc = self.columns.get_loc(key)
        if isinstance(loc, (slice, np.ndarray)):
            new_columns = self.columns[loc]
            result_columns = maybe_droplevels(new_columns, key)
            if self._is_mixed_type:
                result = self.reindex(columns=new_columns)
                result.columns = result_columns
            else:
                new_values = self.values[:, loc]
                result = self._constructor(new_values, index=self.index, columns=result_columns)
                result = result.__finalize__(self)
            if (len(result.columns) == 1):
                top = result.columns[0]
                if isinstance(top, tuple):
                    top = top[0]
                if (top == ''):
                    result = result['']
                    if isinstance(result, Series):
                        result = self._constructor_sliced(result, index=self.index, name=key)
            result._set_is_copy(self)
            return result
        else:
            return self._ixs(loc, axis=1)

    def _get_value(self, index, col, takeable=False):
        '\n        Quickly retrieve single value at passed column and index.\n\n        Parameters\n        ----------\n        index : row label\n        col : column label\n        takeable : interpret the index/col as indexers, default False\n\n        Returns\n        -------\n        scalar\n        '
        if takeable:
            series = self._ixs(col, axis=1)
            return series._values[index]
        series = self._get_item_cache(col)
        engine = self.index._engine
        try:
            loc = engine.get_loc(index)
            return series._values[loc]
        except KeyError:
            if (self.index.nlevels > 1):
                raise
        col = self.columns.get_loc(col)
        index = self.index.get_loc(index)
        return self._get_value(index, col, takeable=True)

    def __setitem__(self, key, value):
        key = com.apply_if_callable(key, self)
        indexer = convert_to_index_sliceable(self, key)
        if (indexer is not None):
            return self._setitem_slice(indexer, value)
        if (isinstance(key, DataFrame) or (getattr(key, 'ndim', None) == 2)):
            self._setitem_frame(key, value)
        elif isinstance(key, (Series, np.ndarray, list, Index)):
            self._setitem_array(key, value)
        elif isinstance(value, DataFrame):
            self._set_item_frame_value(key, value)
        else:
            self._set_item(key, value)

    def _setitem_slice(self, key, value):
        self._check_setitem_copy()
        self.iloc[key] = value

    def _setitem_array(self, key, value):
        if com.is_bool_indexer(key):
            if (len(key) != len(self.index)):
                raise ValueError(f'Item wrong length {len(key)} instead of {len(self.index)}!')
            key = check_bool_indexer(self.index, key)
            indexer = key.nonzero()[0]
            self._check_setitem_copy()
            self.iloc[indexer] = value
        elif isinstance(value, DataFrame):
            if (len(value.columns) != len(key)):
                raise ValueError('Columns must be same length as key')
            for (k1, k2) in zip(key, value.columns):
                self[k1] = value[k2]
        else:
            self.loc._ensure_listlike_indexer(key, axis=1, value=value)
            indexer = self.loc._get_listlike_indexer(key, axis=1, raise_missing=False)[1]
            self._check_setitem_copy()
            self.iloc[:, indexer] = value

    def _setitem_frame(self, key, value):
        if isinstance(key, np.ndarray):
            if (key.shape != self.shape):
                raise ValueError('Array conditional must be same shape as self')
            key = self._constructor(key, **self._construct_axes_dict())
        if (key.size and (not is_bool_dtype(key.values))):
            raise TypeError('Must pass DataFrame or 2-d ndarray with boolean values only')
        self._check_inplace_setting(value)
        self._check_setitem_copy()
        self._where((- key), value, inplace=True)

    def _set_item_frame_value(self, key, value):
        self._ensure_valid_index(value)
        if (isinstance(self.columns, MultiIndex) and (key in self.columns)):
            loc = self.columns.get_loc(key)
            if isinstance(loc, (slice, Series, np.ndarray, Index)):
                cols = maybe_droplevels(self.columns[loc], key)
                if (len(cols) and (not cols.equals(value.columns))):
                    value = value.reindex(cols, axis=1)
        value = _reindex_for_setitem(value, self.index)
        value = value.T
        self._set_item_mgr(key, value)

    def _iset_item_mgr(self, loc, value):
        self._mgr.iset(loc, value)
        self._clear_item_cache()

    def _set_item_mgr(self, key, value):
        value = _maybe_atleast_2d(value)
        try:
            loc = self._info_axis.get_loc(key)
        except KeyError:
            self._mgr.insert(len(self._info_axis), key, value)
        else:
            self._iset_item_mgr(loc, value)
        if len(self):
            self._check_setitem_copy()

    def _iset_item(self, loc, value):
        value = self._sanitize_column(value)
        value = _maybe_atleast_2d(value)
        self._iset_item_mgr(loc, value)
        if len(self):
            self._check_setitem_copy()

    def _set_item(self, key, value):
        '\n        Add series to DataFrame in specified column.\n\n        If series is a numpy-array (not a Series/TimeSeries), it must be the\n        same length as the DataFrames index or an error will be thrown.\n\n        Series/TimeSeries will be conformed to the DataFrames index to\n        ensure homogeneity.\n        '
        value = self._sanitize_column(value)
        if ((key in self.columns) and (value.ndim == 1) and (not is_extension_array_dtype(value))):
            if ((not self.columns.is_unique) or isinstance(self.columns, MultiIndex)):
                existing_piece = self[key]
                if isinstance(existing_piece, DataFrame):
                    value = np.tile(value, (len(existing_piece.columns), 1))
        self._set_item_mgr(key, value)

    def _set_value(self, index, col, value, takeable=False):
        '\n        Put single value at passed column and index.\n\n        Parameters\n        ----------\n        index : row label\n        col : column label\n        value : scalar\n        takeable : interpret the index/col as indexers, default False\n        '
        try:
            if (takeable is True):
                series = self._ixs(col, axis=1)
                series._set_value(index, value, takeable=True)
                return
            series = self._get_item_cache(col)
            engine = self.index._engine
            loc = engine.get_loc(index)
            validate_numeric_casting(series.dtype, value)
            series._values[loc] = value
        except (KeyError, TypeError):
            if takeable:
                self.iloc[(index, col)] = value
            else:
                self.loc[(index, col)] = value
            self._item_cache.pop(col, None)

    def _ensure_valid_index(self, value):
        "\n        Ensure that if we don't have an index, that we can create one from the\n        passed value.\n        "
        if ((not len(self.index)) and is_list_like(value) and len(value)):
            try:
                value = Series(value)
            except (ValueError, NotImplementedError, TypeError) as err:
                raise ValueError('Cannot set a frame with no defined index and a value that cannot be converted to a Series') from err
            index_copy = value.index.copy()
            if (self.index.name is not None):
                index_copy.name = self.index.name
            self._mgr = self._mgr.reindex_axis(index_copy, axis=1, fill_value=np.nan)

    def _box_col_values(self, values, loc):
        '\n        Provide boxed values for a column.\n        '
        name = self.columns[loc]
        klass = self._constructor_sliced
        return klass(values, index=self.index, name=name, fastpath=True)

    def query(self, expr, inplace=False, **kwargs):
        '\n        Query the columns of a DataFrame with a boolean expression.\n\n        Parameters\n        ----------\n        expr : str\n            The query string to evaluate.\n\n            You can refer to variables\n            in the environment by prefixing them with an \'@\' character like\n            ``@a + b``.\n\n            You can refer to column names that are not valid Python variable names\n            by surrounding them in backticks. Thus, column names containing spaces\n            or punctuations (besides underscores) or starting with digits must be\n            surrounded by backticks. (For example, a column named "Area (cm^2) would\n            be referenced as `Area (cm^2)`). Column names which are Python keywords\n            (like "list", "for", "import", etc) cannot be used.\n\n            For example, if one of your columns is called ``a a`` and you want\n            to sum it with ``b``, your query should be ```a a` + b``.\n\n            .. versionadded:: 0.25.0\n                Backtick quoting introduced.\n\n            .. versionadded:: 1.0.0\n                Expanding functionality of backtick quoting for more than only spaces.\n\n        inplace : bool\n            Whether the query should modify the data in place or return\n            a modified copy.\n        **kwargs\n            See the documentation for :func:`eval` for complete details\n            on the keyword arguments accepted by :meth:`DataFrame.query`.\n\n        Returns\n        -------\n        DataFrame or None\n            DataFrame resulting from the provided query expression or\n            None if ``inplace=True``.\n\n        See Also\n        --------\n        eval : Evaluate a string describing operations on\n            DataFrame columns.\n        DataFrame.eval : Evaluate a string describing operations on\n            DataFrame columns.\n\n        Notes\n        -----\n        The result of the evaluation of this expression is first passed to\n        :attr:`DataFrame.loc` and if that fails because of a\n        multidimensional key (e.g., a DataFrame) then the result will be passed\n        to :meth:`DataFrame.__getitem__`.\n\n        This method uses the top-level :func:`eval` function to\n        evaluate the passed query.\n\n        The :meth:`~pandas.DataFrame.query` method uses a slightly\n        modified Python syntax by default. For example, the ``&`` and ``|``\n        (bitwise) operators have the precedence of their boolean cousins,\n        :keyword:`and` and :keyword:`or`. This *is* syntactically valid Python,\n        however the semantics are different.\n\n        You can change the semantics of the expression by passing the keyword\n        argument ``parser=\'python\'``. This enforces the same semantics as\n        evaluation in Python space. Likewise, you can pass ``engine=\'python\'``\n        to evaluate an expression using Python itself as a backend. This is not\n        recommended as it is inefficient compared to using ``numexpr`` as the\n        engine.\n\n        The :attr:`DataFrame.index` and\n        :attr:`DataFrame.columns` attributes of the\n        :class:`~pandas.DataFrame` instance are placed in the query namespace\n        by default, which allows you to treat both the index and columns of the\n        frame as a column in the frame.\n        The identifier ``index`` is used for the frame index; you can also\n        use the name of the index to identify it in a query. Please note that\n        Python keywords may not be used as identifiers.\n\n        For further details and examples see the ``query`` documentation in\n        :ref:`indexing <indexing.query>`.\n\n        *Backtick quoted variables*\n\n        Backtick quoted variables are parsed as literal Python code and\n        are converted internally to a Python valid identifier.\n        This can lead to the following problems.\n\n        During parsing a number of disallowed characters inside the backtick\n        quoted string are replaced by strings that are allowed as a Python identifier.\n        These characters include all operators in Python, the space character, the\n        question mark, the exclamation mark, the dollar sign, and the euro sign.\n        For other characters that fall outside the ASCII range (U+0001..U+007F)\n        and those that are not further specified in PEP 3131,\n        the query parser will raise an error.\n        This excludes whitespace different than the space character,\n        but also the hashtag (as it is used for comments) and the backtick\n        itself (backtick can also not be escaped).\n\n        In a special case, quotes that make a pair around a backtick can\n        confuse the parser.\n        For example, ```it\'s` > `that\'s``` will raise an error,\n        as it forms a quoted string (``\'s > `that\'``) with a backtick inside.\n\n        See also the Python documentation about lexical analysis\n        (https://docs.python.org/3/reference/lexical_analysis.html)\n        in combination with the source code in :mod:`pandas.core.computation.parsing`.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'A\': range(1, 6),\n        ...                    \'B\': range(10, 0, -2),\n        ...                    \'C C\': range(10, 5, -1)})\n        >>> df\n           A   B  C C\n        0  1  10   10\n        1  2   8    9\n        2  3   6    8\n        3  4   4    7\n        4  5   2    6\n        >>> df.query(\'A > B\')\n           A  B  C C\n        4  5  2    6\n\n        The previous expression is equivalent to\n\n        >>> df[df.A > df.B]\n           A  B  C C\n        4  5  2    6\n\n        For columns with spaces in their name, you can use backtick quoting.\n\n        >>> df.query(\'B == `C C`\')\n           A   B  C C\n        0  1  10   10\n\n        The previous expression is equivalent to\n\n        >>> df[df.B == df[\'C C\']]\n           A   B  C C\n        0  1  10   10\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (not isinstance(expr, str)):
            msg = f'expr must be a string to be evaluated, {type(expr)} given'
            raise ValueError(msg)
        kwargs['level'] = (kwargs.pop('level', 0) + 1)
        kwargs['target'] = None
        res = self.eval(expr, **kwargs)
        try:
            result = self.loc[res]
        except ValueError:
            result = self[res]
        if inplace:
            self._update_inplace(result)
        else:
            return result

    def eval(self, expr, inplace=False, **kwargs):
        "\n        Evaluate a string describing operations on DataFrame columns.\n\n        Operates on columns only, not specific rows or elements.  This allows\n        `eval` to run arbitrary code, which can make you vulnerable to code\n        injection if you pass user input to this function.\n\n        Parameters\n        ----------\n        expr : str\n            The expression string to evaluate.\n        inplace : bool, default False\n            If the expression contains an assignment, whether to perform the\n            operation inplace and mutate the existing DataFrame. Otherwise,\n            a new DataFrame is returned.\n        **kwargs\n            See the documentation for :func:`eval` for complete details\n            on the keyword arguments accepted by\n            :meth:`~pandas.DataFrame.query`.\n\n        Returns\n        -------\n        ndarray, scalar, pandas object, or None\n            The result of the evaluation or None if ``inplace=True``.\n\n        See Also\n        --------\n        DataFrame.query : Evaluates a boolean expression to query the columns\n            of a frame.\n        DataFrame.assign : Can evaluate an expression or function to create new\n            values for a column.\n        eval : Evaluate a Python expression as a string using various\n            backends.\n\n        Notes\n        -----\n        For more details see the API documentation for :func:`~eval`.\n        For detailed examples see :ref:`enhancing performance with eval\n        <enhancingperf.eval>`.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A': range(1, 6), 'B': range(10, 0, -2)})\n        >>> df\n           A   B\n        0  1  10\n        1  2   8\n        2  3   6\n        3  4   4\n        4  5   2\n        >>> df.eval('A + B')\n        0    11\n        1    10\n        2     9\n        3     8\n        4     7\n        dtype: int64\n\n        Assignment is allowed though by default the original DataFrame is not\n        modified.\n\n        >>> df.eval('C = A + B')\n           A   B   C\n        0  1  10  11\n        1  2   8  10\n        2  3   6   9\n        3  4   4   8\n        4  5   2   7\n        >>> df\n           A   B\n        0  1  10\n        1  2   8\n        2  3   6\n        3  4   4\n        4  5   2\n\n        Use ``inplace=True`` to modify the original DataFrame.\n\n        >>> df.eval('C = A + B', inplace=True)\n        >>> df\n           A   B   C\n        0  1  10  11\n        1  2   8  10\n        2  3   6   9\n        3  4   4   8\n        4  5   2   7\n\n        Multiple columns can be assigned to using multi-line expressions:\n\n        >>> df.eval(\n        ...     '''\n        ... C = A + B\n        ... D = A - B\n        ... '''\n        ... )\n           A   B   C  D\n        0  1  10  11 -9\n        1  2   8  10 -6\n        2  3   6   9 -3\n        3  4   4   8  0\n        4  5   2   7  3\n        "
        from pandas.core.computation.eval import eval as _eval
        inplace = validate_bool_kwarg(inplace, 'inplace')
        resolvers = kwargs.pop('resolvers', None)
        kwargs['level'] = (kwargs.pop('level', 0) + 1)
        if (resolvers is None):
            index_resolvers = self._get_index_resolvers()
            column_resolvers = self._get_cleaned_column_resolvers()
            resolvers = (column_resolvers, index_resolvers)
        if ('target' not in kwargs):
            kwargs['target'] = self
        kwargs['resolvers'] = (kwargs.get('resolvers', ()) + tuple(resolvers))
        return _eval(expr, inplace=inplace, **kwargs)

    def select_dtypes(self, include=None, exclude=None):
        "\n        Return a subset of the DataFrame's columns based on the column dtypes.\n\n        Parameters\n        ----------\n        include, exclude : scalar or list-like\n            A selection of dtypes or strings to be included/excluded. At least\n            one of these parameters must be supplied.\n\n        Returns\n        -------\n        DataFrame\n            The subset of the frame including the dtypes in ``include`` and\n            excluding the dtypes in ``exclude``.\n\n        Raises\n        ------\n        ValueError\n            * If both of ``include`` and ``exclude`` are empty\n            * If ``include`` and ``exclude`` have overlapping elements\n            * If any kind of string dtype is passed in.\n\n        See Also\n        --------\n        DataFrame.dtypes: Return Series with the data type of each column.\n\n        Notes\n        -----\n        * To select all *numeric* types, use ``np.number`` or ``'number'``\n        * To select strings you must use the ``object`` dtype, but note that\n          this will return *all* object dtype columns\n        * See the `numpy dtype hierarchy\n          <https://numpy.org/doc/stable/reference/arrays.scalars.html>`__\n        * To select datetimes, use ``np.datetime64``, ``'datetime'`` or\n          ``'datetime64'``\n        * To select timedeltas, use ``np.timedelta64``, ``'timedelta'`` or\n          ``'timedelta64'``\n        * To select Pandas categorical dtypes, use ``'category'``\n        * To select Pandas datetimetz dtypes, use ``'datetimetz'`` (new in\n          0.20.0) or ``'datetime64[ns, tz]'``\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'a': [1, 2] * 3,\n        ...                    'b': [True, False] * 3,\n        ...                    'c': [1.0, 2.0] * 3})\n        >>> df\n                a      b  c\n        0       1   True  1.0\n        1       2  False  2.0\n        2       1   True  1.0\n        3       2  False  2.0\n        4       1   True  1.0\n        5       2  False  2.0\n\n        >>> df.select_dtypes(include='bool')\n           b\n        0  True\n        1  False\n        2  True\n        3  False\n        4  True\n        5  False\n\n        >>> df.select_dtypes(include=['float64'])\n           c\n        0  1.0\n        1  2.0\n        2  1.0\n        3  2.0\n        4  1.0\n        5  2.0\n\n        >>> df.select_dtypes(exclude=['int64'])\n               b    c\n        0   True  1.0\n        1  False  2.0\n        2   True  1.0\n        3  False  2.0\n        4   True  1.0\n        5  False  2.0\n        "
        if (not is_list_like(include)):
            include = ((include,) if (include is not None) else ())
        if (not is_list_like(exclude)):
            exclude = ((exclude,) if (exclude is not None) else ())
        selection = (frozenset(include), frozenset(exclude))
        if (not any(selection)):
            raise ValueError('at least one of include or exclude must be nonempty')
        include = frozenset((infer_dtype_from_object(x) for x in include))
        exclude = frozenset((infer_dtype_from_object(x) for x in exclude))
        for dtypes in (include, exclude):
            invalidate_string_dtypes(dtypes)
        if (not include.isdisjoint(exclude)):
            raise ValueError(f'include and exclude overlap on {(include & exclude)}')
        keep_these = np.full(self.shape[1], True)

        def extract_unique_dtypes_from_dtypes_set(dtypes_set: FrozenSet[Dtype], unique_dtypes: np.ndarray) -> List[Dtype]:
            extracted_dtypes = [unique_dtype for unique_dtype in unique_dtypes if (issubclass(unique_dtype.type, tuple(dtypes_set)) or ((np.number in dtypes_set) and getattr(unique_dtype, '_is_numeric', False)))]
            return extracted_dtypes
        unique_dtypes = self.dtypes.unique()
        if include:
            included_dtypes = extract_unique_dtypes_from_dtypes_set(include, unique_dtypes)
            keep_these &= self.dtypes.isin(included_dtypes)
        if exclude:
            excluded_dtypes = extract_unique_dtypes_from_dtypes_set(exclude, unique_dtypes)
            keep_these &= (~ self.dtypes.isin(excluded_dtypes))
        return self.iloc[:, keep_these.values]

    def insert(self, loc, column, value, allow_duplicates=False):
        '\n        Insert column into DataFrame at specified location.\n\n        Raises a ValueError if `column` is already contained in the DataFrame,\n        unless `allow_duplicates` is set to True.\n\n        Parameters\n        ----------\n        loc : int\n            Insertion index. Must verify 0 <= loc <= len(columns).\n        column : str, number, or hashable object\n            Label of the inserted column.\n        value : int, Series, or array-like\n        allow_duplicates : bool, optional\n\n        See Also\n        --------\n        Index.insert : Insert new item by index.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'col1\': [1, 2], \'col2\': [3, 4]})\n        >>> df\n           col1  col2\n        0     1     3\n        1     2     4\n        >>> df.insert(1, "newcol", [99, 99])\n        >>> df\n           col1  newcol  col2\n        0     1      99     3\n        1     2      99     4\n        >>> df.insert(0, "col1", [100, 100], allow_duplicates=True)\n        >>> df\n           col1  col1  newcol  col2\n        0   100     1      99     3\n        1   100     2      99     4\n        '
        if (allow_duplicates and (not self.flags.allows_duplicate_labels)):
            raise ValueError("Cannot specify 'allow_duplicates=True' when 'self.flags.allows_duplicate_labels' is False.")
        value = self._sanitize_column(value)
        value = _maybe_atleast_2d(value)
        self._mgr.insert(loc, column, value, allow_duplicates=allow_duplicates)

    def assign(self, **kwargs):
        "\n        Assign new columns to a DataFrame.\n\n        Returns a new object with all original columns in addition to new ones.\n        Existing columns that are re-assigned will be overwritten.\n\n        Parameters\n        ----------\n        **kwargs : dict of {str: callable or Series}\n            The column names are keywords. If the values are\n            callable, they are computed on the DataFrame and\n            assigned to the new columns. The callable must not\n            change input DataFrame (though pandas doesn't check it).\n            If the values are not callable, (e.g. a Series, scalar, or array),\n            they are simply assigned.\n\n        Returns\n        -------\n        DataFrame\n            A new DataFrame with the new columns in addition to\n            all the existing columns.\n\n        Notes\n        -----\n        Assigning multiple columns within the same ``assign`` is possible.\n        Later items in '\\*\\*kwargs' may refer to newly created or modified\n        columns in 'df'; items are computed and assigned into 'df' in order.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'temp_c': [17.0, 25.0]},\n        ...                   index=['Portland', 'Berkeley'])\n        >>> df\n                  temp_c\n        Portland    17.0\n        Berkeley    25.0\n\n        Where the value is a callable, evaluated on `df`:\n\n        >>> df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)\n                  temp_c  temp_f\n        Portland    17.0    62.6\n        Berkeley    25.0    77.0\n\n        Alternatively, the same behavior can be achieved by directly\n        referencing an existing Series or sequence:\n\n        >>> df.assign(temp_f=df['temp_c'] * 9 / 5 + 32)\n                  temp_c  temp_f\n        Portland    17.0    62.6\n        Berkeley    25.0    77.0\n\n        You can create multiple columns within the same assign where one\n        of the columns depends on another one defined within the same assign:\n\n        >>> df.assign(temp_f=lambda x: x['temp_c'] * 9 / 5 + 32,\n        ...           temp_k=lambda x: (x['temp_f'] +  459.67) * 5 / 9)\n                  temp_c  temp_f  temp_k\n        Portland    17.0    62.6  290.15\n        Berkeley    25.0    77.0  298.15\n        "
        data = self.copy()
        for (k, v) in kwargs.items():
            data[k] = com.apply_if_callable(v, data)
        return data

    def _sanitize_column(self, value):
        '\n        Ensures new columns (which go into the BlockManager as new blocks) are\n        always copied and converted into an array.\n\n        Parameters\n        ----------\n        value : scalar, Series, or array-like\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        self._ensure_valid_index(value)
        if isinstance(value, Series):
            value = _reindex_for_setitem(value, self.index)
        elif isinstance(value, ExtensionArray):
            value = value.copy()
            value = sanitize_index(value, self.index)
        elif (isinstance(value, Index) or is_sequence(value)):
            value = sanitize_index(value, self.index)
            if (not isinstance(value, (np.ndarray, Index))):
                if (isinstance(value, list) and (len(value) > 0)):
                    value = maybe_convert_platform(value)
                else:
                    value = com.asarray_tuplesafe(value)
            elif (value.ndim == 2):
                value = value.copy().T
            elif isinstance(value, Index):
                value = value.copy(deep=True)
            else:
                value = value.copy()
            if is_object_dtype(value.dtype):
                value = maybe_infer_to_datetimelike(value)
        else:
            value = construct_1d_arraylike_from_scalar(value, len(self), dtype=None)
        return value

    @property
    def _series(self):
        return {item: Series(self._mgr.iget(idx), index=self.index, name=item, fastpath=True) for (idx, item) in enumerate(self.columns)}

    def lookup(self, row_labels, col_labels):
        '\n        Label-based "fancy indexing" function for DataFrame.\n        Given equal-length arrays of row and column labels, return an\n        array of the values corresponding to each (row, col) pair.\n\n        .. deprecated:: 1.2.0\n            DataFrame.lookup is deprecated,\n            use DataFrame.melt and DataFrame.loc instead.\n            For an example see :meth:`~pandas.DataFrame.lookup`\n            in the user guide.\n\n        Parameters\n        ----------\n        row_labels : sequence\n            The row labels to use for lookup.\n        col_labels : sequence\n            The column labels to use for lookup.\n\n        Returns\n        -------\n        numpy.ndarray\n            The found values.\n        '
        msg = "The 'lookup' method is deprecated and will beremoved in a future version.You can use DataFrame.melt and DataFrame.locas a substitute."
        warnings.warn(msg, FutureWarning, stacklevel=2)
        n = len(row_labels)
        if (n != len(col_labels)):
            raise ValueError('Row labels must have same size as column labels')
        if (not (self.index.is_unique and self.columns.is_unique)):
            raise ValueError('DataFrame.lookup requires unique index and columns')
        thresh = 1000
        if ((not self._is_mixed_type) or (n > thresh)):
            values = self.values
            ridx = self.index.get_indexer(row_labels)
            cidx = self.columns.get_indexer(col_labels)
            if (ridx == (- 1)).any():
                raise KeyError('One or more row labels was not found')
            if (cidx == (- 1)).any():
                raise KeyError('One or more column labels was not found')
            flat_index = ((ridx * len(self.columns)) + cidx)
            result = values.flat[flat_index]
        else:
            result = np.empty(n, dtype='O')
            for (i, (r, c)) in enumerate(zip(row_labels, col_labels)):
                result[i] = self._get_value(r, c)
        if is_object_dtype(result):
            result = lib.maybe_convert_objects(result)
        return result

    def _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy):
        frame = self
        columns = axes['columns']
        if (columns is not None):
            frame = frame._reindex_columns(columns, method, copy, level, fill_value, limit, tolerance)
        index = axes['index']
        if (index is not None):
            frame = frame._reindex_index(index, method, copy, level, fill_value, limit, tolerance)
        return frame

    def _reindex_index(self, new_index, method, copy, level, fill_value=np.nan, limit=None, tolerance=None):
        (new_index, indexer) = self.index.reindex(new_index, method=method, level=level, limit=limit, tolerance=tolerance)
        return self._reindex_with_indexers({0: [new_index, indexer]}, copy=copy, fill_value=fill_value, allow_dups=False)

    def _reindex_columns(self, new_columns, method, copy, level, fill_value=None, limit=None, tolerance=None):
        (new_columns, indexer) = self.columns.reindex(new_columns, method=method, level=level, limit=limit, tolerance=tolerance)
        return self._reindex_with_indexers({1: [new_columns, indexer]}, copy=copy, fill_value=fill_value, allow_dups=False)

    def _reindex_multi(self, axes, copy, fill_value):
        '\n        We are guaranteed non-Nones in the axes.\n        '
        (new_index, row_indexer) = self.index.reindex(axes['index'])
        (new_columns, col_indexer) = self.columns.reindex(axes['columns'])
        if ((row_indexer is not None) and (col_indexer is not None)):
            indexer = (row_indexer, col_indexer)
            new_values = algorithms.take_2d_multi(self.values, indexer, fill_value=fill_value)
            return self._constructor(new_values, index=new_index, columns=new_columns)
        else:
            return self._reindex_with_indexers({0: [new_index, row_indexer], 1: [new_columns, col_indexer]}, copy=copy, fill_value=fill_value)

    @doc(NDFrame.align, **_shared_doc_kwargs)
    def align(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None, limit=None, fill_axis=0, broadcast_axis=None):
        return super().align(other, join=join, axis=axis, level=level, copy=copy, fill_value=fill_value, method=method, limit=limit, fill_axis=fill_axis, broadcast_axis=broadcast_axis)

    @Appender('\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})\n\n        Change the row labels.\n\n        >>> df.set_axis([\'a\', \'b\', \'c\'], axis=\'index\')\n           A  B\n        a  1  4\n        b  2  5\n        c  3  6\n\n        Change the column labels.\n\n        >>> df.set_axis([\'I\', \'II\'], axis=\'columns\')\n           I  II\n        0  1   4\n        1  2   5\n        2  3   6\n\n        Now, update the labels inplace.\n\n        >>> df.set_axis([\'i\', \'ii\'], axis=\'columns\', inplace=True)\n        >>> df\n           i  ii\n        0  1   4\n        1  2   5\n        2  3   6\n        ')
    @Substitution(**_shared_doc_kwargs, extended_summary_sub=' column or', axis_description_sub=', and 1 identifies the columns', see_also_sub=' or columns')
    @Appender(NDFrame.set_axis.__doc__)
    def set_axis(self, labels, axis=0, inplace=False):
        return super().set_axis(labels, axis=axis, inplace=inplace)

    @Substitution(**_shared_doc_kwargs)
    @Appender(NDFrame.reindex.__doc__)
    @rewrite_axis_style_signature('labels', [('method', None), ('copy', True), ('level', None), ('fill_value', np.nan), ('limit', None), ('tolerance', None)])
    def reindex(self, *args, **kwargs):
        axes = validate_axis_style_args(self, args, kwargs, 'labels', 'reindex')
        kwargs.update(axes)
        kwargs.pop('axis', None)
        kwargs.pop('labels', None)
        return super().reindex(**kwargs)

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        "\n        Drop specified labels from rows or columns.\n\n        Remove rows or columns by specifying label names and corresponding\n        axis, or by specifying directly index or column names. When using a\n        multi-index, labels on different levels can be removed by specifying\n        the level.\n\n        Parameters\n        ----------\n        labels : single label or list-like\n            Index or column labels to drop.\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            Whether to drop labels from the index (0 or 'index') or\n            columns (1 or 'columns').\n        index : single label or list-like\n            Alternative to specifying axis (``labels, axis=0``\n            is equivalent to ``index=labels``).\n        columns : single label or list-like\n            Alternative to specifying axis (``labels, axis=1``\n            is equivalent to ``columns=labels``).\n        level : int or level name, optional\n            For MultiIndex, level from which the labels will be removed.\n        inplace : bool, default False\n            If False, return a copy. Otherwise, do operation\n            inplace and return None.\n        errors : {'ignore', 'raise'}, default 'raise'\n            If 'ignore', suppress error and only existing labels are\n            dropped.\n\n        Returns\n        -------\n        DataFrame or None\n            DataFrame without the removed index or column labels or\n            None if ``inplace=True``.\n\n        Raises\n        ------\n        KeyError\n            If any of the labels is not found in the selected axis.\n\n        See Also\n        --------\n        DataFrame.loc : Label-location based indexer for selection by label.\n        DataFrame.dropna : Return DataFrame with labels on given axis omitted\n            where (all or any) data are missing.\n        DataFrame.drop_duplicates : Return DataFrame with duplicate rows\n            removed, optionally only considering certain columns.\n        Series.drop : Return Series with specified index labels removed.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(np.arange(12).reshape(3, 4),\n        ...                   columns=['A', 'B', 'C', 'D'])\n        >>> df\n           A  B   C   D\n        0  0  1   2   3\n        1  4  5   6   7\n        2  8  9  10  11\n\n        Drop columns\n\n        >>> df.drop(['B', 'C'], axis=1)\n           A   D\n        0  0   3\n        1  4   7\n        2  8  11\n\n        >>> df.drop(columns=['B', 'C'])\n           A   D\n        0  0   3\n        1  4   7\n        2  8  11\n\n        Drop a row by index\n\n        >>> df.drop([0, 1])\n           A  B   C   D\n        2  8  9  10  11\n\n        Drop columns and/or rows of MultiIndex DataFrame\n\n        >>> midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'],\n        ...                              ['speed', 'weight', 'length']],\n        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],\n        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])\n        >>> df = pd.DataFrame(index=midx, columns=['big', 'small'],\n        ...                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],\n        ...                         [250, 150], [1.5, 0.8], [320, 250],\n        ...                         [1, 0.8], [0.3, 0.2]])\n        >>> df\n                        big     small\n        lama    speed   45.0    30.0\n                weight  200.0   100.0\n                length  1.5     1.0\n        cow     speed   30.0    20.0\n                weight  250.0   150.0\n                length  1.5     0.8\n        falcon  speed   320.0   250.0\n                weight  1.0     0.8\n                length  0.3     0.2\n\n        >>> df.drop(index='cow', columns='small')\n                        big\n        lama    speed   45.0\n                weight  200.0\n                length  1.5\n        falcon  speed   320.0\n                weight  1.0\n                length  0.3\n\n        >>> df.drop(index='length', level=1)\n                        big     small\n        lama    speed   45.0    30.0\n                weight  200.0   100.0\n        cow     speed   30.0    20.0\n                weight  250.0   150.0\n        falcon  speed   320.0   250.0\n                weight  1.0     0.8\n        "
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    @rewrite_axis_style_signature('mapper', [('copy', True), ('inplace', False), ('level', None), ('errors', 'ignore')])
    def rename(self, mapper=None, *, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore'):
        '\n        Alter axes labels.\n\n        Function / dict values must be unique (1-to-1). Labels not contained in\n        a dict / Series will be left as-is. Extra labels listed don\'t throw an\n        error.\n\n        See the :ref:`user guide <basics.rename>` for more.\n\n        Parameters\n        ----------\n        mapper : dict-like or function\n            Dict-like or function transformations to apply to\n            that axis\' values. Use either ``mapper`` and ``axis`` to\n            specify the axis to target with ``mapper``, or ``index`` and\n            ``columns``.\n        index : dict-like or function\n            Alternative to specifying axis (``mapper, axis=0``\n            is equivalent to ``index=mapper``).\n        columns : dict-like or function\n            Alternative to specifying axis (``mapper, axis=1``\n            is equivalent to ``columns=mapper``).\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            Axis to target with ``mapper``. Can be either the axis name\n            (\'index\', \'columns\') or number (0, 1). The default is \'index\'.\n        copy : bool, default True\n            Also copy underlying data.\n        inplace : bool, default False\n            Whether to return a new DataFrame. If True then value of copy is\n            ignored.\n        level : int or level name, default None\n            In case of a MultiIndex, only rename labels in the specified\n            level.\n        errors : {\'ignore\', \'raise\'}, default \'ignore\'\n            If \'raise\', raise a `KeyError` when a dict-like `mapper`, `index`,\n            or `columns` contains labels that are not present in the Index\n            being transformed.\n            If \'ignore\', existing keys will be renamed and extra keys will be\n            ignored.\n\n        Returns\n        -------\n        DataFrame or None\n            DataFrame with the renamed axis labels or None if ``inplace=True``.\n\n        Raises\n        ------\n        KeyError\n            If any of the labels is not found in the selected axis and\n            "errors=\'raise\'".\n\n        See Also\n        --------\n        DataFrame.rename_axis : Set the name of the axis.\n\n        Examples\n        --------\n        ``DataFrame.rename`` supports two calling conventions\n\n        * ``(index=index_mapper, columns=columns_mapper, ...)``\n        * ``(mapper, axis={\'index\', \'columns\'}, ...)``\n\n        We *highly* recommend using keyword arguments to clarify your\n        intent.\n\n        Rename columns using a mapping:\n\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})\n        >>> df.rename(columns={"A": "a", "B": "c"})\n           a  c\n        0  1  4\n        1  2  5\n        2  3  6\n\n        Rename index using a mapping:\n\n        >>> df.rename(index={0: "x", 1: "y", 2: "z"})\n           A  B\n        x  1  4\n        y  2  5\n        z  3  6\n\n        Cast index labels to a different type:\n\n        >>> df.index\n        RangeIndex(start=0, stop=3, step=1)\n        >>> df.rename(index=str).index\n        Index([\'0\', \'1\', \'2\'], dtype=\'object\')\n\n        >>> df.rename(columns={"A": "a", "B": "b", "C": "c"}, errors="raise")\n        Traceback (most recent call last):\n        KeyError: [\'C\'] not found in axis\n\n        Using axis-style parameters:\n\n        >>> df.rename(str.lower, axis=\'columns\')\n           a  b\n        0  1  4\n        1  2  5\n        2  3  6\n\n        >>> df.rename({1: 2, 2: 4}, axis=\'index\')\n           A  B\n        0  1  4\n        2  2  5\n        4  3  6\n        '
        return super().rename(mapper=mapper, index=index, columns=columns, axis=axis, copy=copy, inplace=inplace, level=level, errors=errors)

    @doc(NDFrame.fillna, **_shared_doc_kwargs)
    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        return super().fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)

    def pop(self, item):
        "\n        Return item and drop from frame. Raise KeyError if not found.\n\n        Parameters\n        ----------\n        item : label\n            Label of column to be popped.\n\n        Returns\n        -------\n        Series\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([('falcon', 'bird', 389.0),\n        ...                    ('parrot', 'bird', 24.0),\n        ...                    ('lion', 'mammal', 80.5),\n        ...                    ('monkey', 'mammal', np.nan)],\n        ...                   columns=('name', 'class', 'max_speed'))\n        >>> df\n             name   class  max_speed\n        0  falcon    bird      389.0\n        1  parrot    bird       24.0\n        2    lion  mammal       80.5\n        3  monkey  mammal        NaN\n\n        >>> df.pop('class')\n        0      bird\n        1      bird\n        2    mammal\n        3    mammal\n        Name: class, dtype: object\n\n        >>> df\n             name  max_speed\n        0  falcon      389.0\n        1  parrot       24.0\n        2    lion       80.5\n        3  monkey        NaN\n        "
        return super().pop(item=item)

    @doc(NDFrame.replace, **_shared_doc_kwargs)
    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
        return super().replace(to_replace=to_replace, value=value, inplace=inplace, limit=limit, regex=regex, method=method)

    def _replace_columnwise(self, mapping, inplace, regex):
        '\n        Dispatch to Series.replace column-wise.\n\n\n        Parameters\n        ----------\n        mapping : dict\n            of the form {col: (target, value)}\n        inplace : bool\n        regex : bool or same types as `to_replace` in DataFrame.replace\n\n        Returns\n        -------\n        DataFrame or None\n        '
        res = (self if inplace else self.copy())
        ax = self.columns
        for i in range(len(ax)):
            if (ax[i] in mapping):
                ser = self.iloc[:, i]
                (target, value) = mapping[ax[i]]
                newobj = ser.replace(target, value, regex=regex)
                res.iloc[:, i] = newobj
        if inplace:
            return
        return res.__finalize__(self)

    @doc(NDFrame.shift, klass=_shared_doc_kwargs['klass'])
    def shift(self, periods=1, freq=None, axis=0, fill_value=lib.no_default):
        axis = self._get_axis_number(axis)
        ncols = len(self.columns)
        if ((axis == 1) and (periods != 0) and (fill_value is lib.no_default) and (ncols > 0)):
            label = self.columns[0]
            if (periods > 0):
                result = self.iloc[:, :(- periods)]
                for col in range(min(ncols, abs(periods))):
                    filler = self.iloc[:, 0].shift(len(self))
                    result.insert(0, label, filler, allow_duplicates=True)
            else:
                result = self.iloc[:, (- periods):]
                for col in range(min(ncols, abs(periods))):
                    filler = self.iloc[:, (- 1)].shift(len(self))
                    result.insert(len(result.columns), label, filler, allow_duplicates=True)
            result.columns = self.columns.copy()
            return result
        return super().shift(periods=periods, freq=freq, axis=axis, fill_value=fill_value)

    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        '\n        Set the DataFrame index using existing columns.\n\n        Set the DataFrame index (row labels) using one or more existing\n        columns or arrays (of the correct length). The index can replace the\n        existing index or expand on it.\n\n        Parameters\n        ----------\n        keys : label or array-like or list of labels/arrays\n            This parameter can be either a single column key, a single array of\n            the same length as the calling DataFrame, or a list containing an\n            arbitrary combination of column keys and arrays. Here, "array"\n            encompasses :class:`Series`, :class:`Index`, ``np.ndarray``, and\n            instances of :class:`~collections.abc.Iterator`.\n        drop : bool, default True\n            Delete columns to be used as the new index.\n        append : bool, default False\n            Whether to append columns to existing index.\n        inplace : bool, default False\n            If True, modifies the DataFrame in place (do not create a new object).\n        verify_integrity : bool, default False\n            Check the new index for duplicates. Otherwise defer the check until\n            necessary. Setting to False will improve the performance of this\n            method.\n\n        Returns\n        -------\n        DataFrame or None\n            Changed row labels or None if ``inplace=True``.\n\n        See Also\n        --------\n        DataFrame.reset_index : Opposite of set_index.\n        DataFrame.reindex : Change to new indices or expand indices.\n        DataFrame.reindex_like : Change to same indices as other DataFrame.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'month\': [1, 4, 7, 10],\n        ...                    \'year\': [2012, 2014, 2013, 2014],\n        ...                    \'sale\': [55, 40, 84, 31]})\n        >>> df\n           month  year  sale\n        0      1  2012    55\n        1      4  2014    40\n        2      7  2013    84\n        3     10  2014    31\n\n        Set the index to become the \'month\' column:\n\n        >>> df.set_index(\'month\')\n               year  sale\n        month\n        1      2012    55\n        4      2014    40\n        7      2013    84\n        10     2014    31\n\n        Create a MultiIndex using columns \'year\' and \'month\':\n\n        >>> df.set_index([\'year\', \'month\'])\n                    sale\n        year  month\n        2012  1     55\n        2014  4     40\n        2013  7     84\n        2014  10    31\n\n        Create a MultiIndex using an Index and a column:\n\n        >>> df.set_index([pd.Index([1, 2, 3, 4]), \'year\'])\n                 month  sale\n           year\n        1  2012  1      55\n        2  2014  4      40\n        3  2013  7      84\n        4  2014  10     31\n\n        Create a MultiIndex using two Series:\n\n        >>> s = pd.Series([1, 2, 3, 4])\n        >>> df.set_index([s, s**2])\n              month  year  sale\n        1 1       1  2012    55\n        2 4       4  2014    40\n        3 9       7  2013    84\n        4 16     10  2014    31\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._check_inplace_and_allows_duplicate_labels(inplace)
        if (not isinstance(keys, list)):
            keys = [keys]
        err_msg = 'The parameter "keys" may be a column key, one-dimensional array, or a list containing only valid column keys and one-dimensional arrays.'
        missing: List[Label] = []
        for col in keys:
            if isinstance(col, (Index, Series, np.ndarray, list, abc.Iterator)):
                if (getattr(col, 'ndim', 1) != 1):
                    raise ValueError(err_msg)
            else:
                try:
                    found = (col in self.columns)
                except TypeError as err:
                    raise TypeError(f'{err_msg}. Received column of type {type(col)}') from err
                else:
                    if (not found):
                        missing.append(col)
        if missing:
            raise KeyError(f'None of {missing} are in the columns')
        if inplace:
            frame = self
        else:
            frame = self.copy()
        arrays = []
        names: List[Label] = []
        if append:
            names = list(self.index.names)
            if isinstance(self.index, MultiIndex):
                for i in range(self.index.nlevels):
                    arrays.append(self.index._get_level_values(i))
            else:
                arrays.append(self.index)
        to_remove: List[Label] = []
        for col in keys:
            if isinstance(col, MultiIndex):
                for n in range(col.nlevels):
                    arrays.append(col._get_level_values(n))
                names.extend(col.names)
            elif isinstance(col, (Index, Series)):
                arrays.append(col)
                names.append(col.name)
            elif isinstance(col, (list, np.ndarray)):
                arrays.append(col)
                names.append(None)
            elif isinstance(col, abc.Iterator):
                arrays.append(list(col))
                names.append(None)
            else:
                arrays.append(frame[col]._values)
                names.append(col)
                if drop:
                    to_remove.append(col)
            if (len(arrays[(- 1)]) != len(self)):
                raise ValueError(f'Length mismatch: Expected {len(self)} rows, received array of length {len(arrays[(- 1)])}')
        index = ensure_index_from_sequences(arrays, names)
        if (verify_integrity and (not index.is_unique)):
            duplicates = index[index.duplicated()].unique()
            raise ValueError(f'Index has duplicate keys: {duplicates}')
        for c in set(to_remove):
            del frame[c]
        index._cleanup()
        frame.index = index
        if (not inplace):
            return frame

    @overload
    def reset_index(self, level=..., drop=..., inplace=..., col_level=..., col_fill=...):
        ...

    @overload
    def reset_index(self, level=..., drop=..., inplace=..., col_level=..., col_fill=...):
        ...

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        "\n        Reset the index, or a level of it.\n\n        Reset the index of the DataFrame, and use the default one instead.\n        If the DataFrame has a MultiIndex, this method can remove one or more\n        levels.\n\n        Parameters\n        ----------\n        level : int, str, tuple, or list, default None\n            Only remove the given levels from the index. Removes all levels by\n            default.\n        drop : bool, default False\n            Do not try to insert index into dataframe columns. This resets\n            the index to the default integer index.\n        inplace : bool, default False\n            Modify the DataFrame in place (do not create a new object).\n        col_level : int or str, default 0\n            If the columns have multiple levels, determines which level the\n            labels are inserted into. By default it is inserted into the first\n            level.\n        col_fill : object, default ''\n            If the columns have multiple levels, determines how the other\n            levels are named. If None then the index name is repeated.\n\n        Returns\n        -------\n        DataFrame or None\n            DataFrame with the new index or None if ``inplace=True``.\n\n        See Also\n        --------\n        DataFrame.set_index : Opposite of reset_index.\n        DataFrame.reindex : Change to new indices or expand indices.\n        DataFrame.reindex_like : Change to same indices as other DataFrame.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([('bird', 389.0),\n        ...                    ('bird', 24.0),\n        ...                    ('mammal', 80.5),\n        ...                    ('mammal', np.nan)],\n        ...                   index=['falcon', 'parrot', 'lion', 'monkey'],\n        ...                   columns=('class', 'max_speed'))\n        >>> df\n                 class  max_speed\n        falcon    bird      389.0\n        parrot    bird       24.0\n        lion    mammal       80.5\n        monkey  mammal        NaN\n\n        When we reset the index, the old index is added as a column, and a\n        new sequential index is used:\n\n        >>> df.reset_index()\n            index   class  max_speed\n        0  falcon    bird      389.0\n        1  parrot    bird       24.0\n        2    lion  mammal       80.5\n        3  monkey  mammal        NaN\n\n        We can use the `drop` parameter to avoid the old index being added as\n        a column:\n\n        >>> df.reset_index(drop=True)\n            class  max_speed\n        0    bird      389.0\n        1    bird       24.0\n        2  mammal       80.5\n        3  mammal        NaN\n\n        You can also use `reset_index` with `MultiIndex`.\n\n        >>> index = pd.MultiIndex.from_tuples([('bird', 'falcon'),\n        ...                                    ('bird', 'parrot'),\n        ...                                    ('mammal', 'lion'),\n        ...                                    ('mammal', 'monkey')],\n        ...                                   names=['class', 'name'])\n        >>> columns = pd.MultiIndex.from_tuples([('speed', 'max'),\n        ...                                      ('species', 'type')])\n        >>> df = pd.DataFrame([(389.0, 'fly'),\n        ...                    ( 24.0, 'fly'),\n        ...                    ( 80.5, 'run'),\n        ...                    (np.nan, 'jump')],\n        ...                   index=index,\n        ...                   columns=columns)\n        >>> df\n                       speed species\n                         max    type\n        class  name\n        bird   falcon  389.0     fly\n               parrot   24.0     fly\n        mammal lion     80.5     run\n               monkey    NaN    jump\n\n        If the index has multiple levels, we can reset a subset of them:\n\n        >>> df.reset_index(level='class')\n                 class  speed species\n                          max    type\n        name\n        falcon    bird  389.0     fly\n        parrot    bird   24.0     fly\n        lion    mammal   80.5     run\n        monkey  mammal    NaN    jump\n\n        If we are not dropping the index, by default, it is placed in the top\n        level. We can place it in another level:\n\n        >>> df.reset_index(level='class', col_level=1)\n                        speed species\n                 class    max    type\n        name\n        falcon    bird  389.0     fly\n        parrot    bird   24.0     fly\n        lion    mammal   80.5     run\n        monkey  mammal    NaN    jump\n\n        When the index is inserted under another level, we can specify under\n        which one with the parameter `col_fill`:\n\n        >>> df.reset_index(level='class', col_level=1, col_fill='species')\n                      species  speed species\n                        class    max    type\n        name\n        falcon           bird  389.0     fly\n        parrot           bird   24.0     fly\n        lion           mammal   80.5     run\n        monkey         mammal    NaN    jump\n\n        If we specify a nonexistent level for `col_fill`, it is created:\n\n        >>> df.reset_index(level='class', col_level=1, col_fill='genus')\n                        genus  speed species\n                        class    max    type\n        name\n        falcon           bird  389.0     fly\n        parrot           bird   24.0     fly\n        lion           mammal   80.5     run\n        monkey         mammal    NaN    jump\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._check_inplace_and_allows_duplicate_labels(inplace)
        if inplace:
            new_obj = self
        else:
            new_obj = self.copy()
        new_index = ibase.default_index(len(new_obj))
        if (level is not None):
            if (not isinstance(level, (tuple, list))):
                level = [level]
            level = [self.index._get_level_number(lev) for lev in level]
            if (len(level) < self.index.nlevels):
                new_index = self.index.droplevel(level)
        if (not drop):
            to_insert: Iterable[Tuple[(Any, Optional[Any])]]
            if isinstance(self.index, MultiIndex):
                names = [(n if (n is not None) else f'level_{i}') for (i, n) in enumerate(self.index.names)]
                to_insert = zip(self.index.levels, self.index.codes)
            else:
                default = ('index' if ('index' not in self) else 'level_0')
                names = ([default] if (self.index.name is None) else [self.index.name])
                to_insert = ((self.index, None),)
            multi_col = isinstance(self.columns, MultiIndex)
            for (i, (lev, lab)) in reversed(list(enumerate(to_insert))):
                if (not ((level is None) or (i in level))):
                    continue
                name = names[i]
                if multi_col:
                    col_name = (list(name) if isinstance(name, tuple) else [name])
                    if (col_fill is None):
                        if (len(col_name) not in (1, self.columns.nlevels)):
                            raise ValueError(f'col_fill=None is incompatible with incomplete column name {name}')
                        col_fill = col_name[0]
                    lev_num = self.columns._get_level_number(col_level)
                    name_lst = (([col_fill] * lev_num) + col_name)
                    missing = (self.columns.nlevels - len(name_lst))
                    name_lst += ([col_fill] * missing)
                    name = tuple(name_lst)
                level_values = lev._values
                if (level_values.dtype == np.object_):
                    level_values = lib.maybe_convert_objects(level_values)
                if (lab is not None):
                    level_values = algorithms.take(level_values, lab, allow_fill=True, fill_value=lev._na_value)
                new_obj.insert(0, name, level_values)
        new_obj.index = new_index
        if (not inplace):
            return new_obj
        return None

    @doc(NDFrame.isna, klass=_shared_doc_kwargs['klass'])
    def isna(self):
        result = self._constructor(self._mgr.isna(func=isna))
        return result.__finalize__(self, method='isna')

    @doc(NDFrame.isna, klass=_shared_doc_kwargs['klass'])
    def isnull(self):
        return self.isna()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs['klass'])
    def notna(self):
        return (~ self.isna())

    @doc(NDFrame.notna, klass=_shared_doc_kwargs['klass'])
    def notnull(self):
        return (~ self.isna())

    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        '\n        Remove missing values.\n\n        See the :ref:`User Guide <missing_data>` for more on which values are\n        considered missing, and how to work with missing data.\n\n        Parameters\n        ----------\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            Determine if rows or columns which contain missing values are\n            removed.\n\n            * 0, or \'index\' : Drop rows which contain missing values.\n            * 1, or \'columns\' : Drop columns which contain missing value.\n\n            .. versionchanged:: 1.0.0\n\n               Pass tuple or list to drop on multiple axes.\n               Only a single axis is allowed.\n\n        how : {\'any\', \'all\'}, default \'any\'\n            Determine if row or column is removed from DataFrame, when we have\n            at least one NA or all NA.\n\n            * \'any\' : If any NA values are present, drop that row or column.\n            * \'all\' : If all values are NA, drop that row or column.\n\n        thresh : int, optional\n            Require that many non-NA values.\n        subset : array-like, optional\n            Labels along other axis to consider, e.g. if you are dropping rows\n            these would be a list of columns to include.\n        inplace : bool, default False\n            If True, do operation inplace and return None.\n\n        Returns\n        -------\n        DataFrame or None\n            DataFrame with NA entries dropped from it or None if ``inplace=True``.\n\n        See Also\n        --------\n        DataFrame.isna: Indicate missing values.\n        DataFrame.notna : Indicate existing (non-missing) values.\n        DataFrame.fillna : Replace missing values.\n        Series.dropna : Drop missing values.\n        Index.dropna : Drop missing indices.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"name": [\'Alfred\', \'Batman\', \'Catwoman\'],\n        ...                    "toy": [np.nan, \'Batmobile\', \'Bullwhip\'],\n        ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),\n        ...                             pd.NaT]})\n        >>> df\n               name        toy       born\n        0    Alfred        NaN        NaT\n        1    Batman  Batmobile 1940-04-25\n        2  Catwoman   Bullwhip        NaT\n\n        Drop the rows where at least one element is missing.\n\n        >>> df.dropna()\n             name        toy       born\n        1  Batman  Batmobile 1940-04-25\n\n        Drop the columns where at least one element is missing.\n\n        >>> df.dropna(axis=\'columns\')\n               name\n        0    Alfred\n        1    Batman\n        2  Catwoman\n\n        Drop the rows where all elements are missing.\n\n        >>> df.dropna(how=\'all\')\n               name        toy       born\n        0    Alfred        NaN        NaT\n        1    Batman  Batmobile 1940-04-25\n        2  Catwoman   Bullwhip        NaT\n\n        Keep only the rows with at least 2 non-NA values.\n\n        >>> df.dropna(thresh=2)\n               name        toy       born\n        1    Batman  Batmobile 1940-04-25\n        2  Catwoman   Bullwhip        NaT\n\n        Define in which columns to look for missing values.\n\n        >>> df.dropna(subset=[\'name\', \'toy\'])\n               name        toy       born\n        1    Batman  Batmobile 1940-04-25\n        2  Catwoman   Bullwhip        NaT\n\n        Keep the DataFrame with valid entries in the same variable.\n\n        >>> df.dropna(inplace=True)\n        >>> df\n             name        toy       born\n        1  Batman  Batmobile 1940-04-25\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if isinstance(axis, (tuple, list)):
            raise TypeError('supplying multiple axes to axis is no longer supported.')
        axis = self._get_axis_number(axis)
        agg_axis = (1 - axis)
        agg_obj = self
        if (subset is not None):
            ax = self._get_axis(agg_axis)
            indices = ax.get_indexer_for(subset)
            check = (indices == (- 1))
            if check.any():
                raise KeyError(list(np.compress(check, subset)))
            agg_obj = self.take(indices, axis=agg_axis)
        count = agg_obj.count(axis=agg_axis)
        if (thresh is not None):
            mask = (count >= thresh)
        elif (how == 'any'):
            mask = (count == len(agg_obj._get_axis(agg_axis)))
        elif (how == 'all'):
            mask = (count > 0)
        elif (how is not None):
            raise ValueError(f'invalid how option: {how}')
        else:
            raise TypeError('must specify how or thresh')
        result = self.loc(axis=axis)[mask]
        if inplace:
            self._update_inplace(result)
        else:
            return result

    def drop_duplicates(self, subset=None, keep='first', inplace=False, ignore_index=False):
        "\n        Return DataFrame with duplicate rows removed.\n\n        Considering certain columns is optional. Indexes, including time indexes\n        are ignored.\n\n        Parameters\n        ----------\n        subset : column label or sequence of labels, optional\n            Only consider certain columns for identifying duplicates, by\n            default use all of the columns.\n        keep : {'first', 'last', False}, default 'first'\n            Determines which duplicates (if any) to keep.\n            - ``first`` : Drop duplicates except for the first occurrence.\n            - ``last`` : Drop duplicates except for the last occurrence.\n            - False : Drop all duplicates.\n        inplace : bool, default False\n            Whether to drop duplicates in place or to return a copy.\n        ignore_index : bool, default False\n            If True, the resulting axis will be labeled 0, 1, …, n - 1.\n\n            .. versionadded:: 1.0.0\n\n        Returns\n        -------\n        DataFrame or None\n            DataFrame with duplicates removed or None if ``inplace=True``.\n\n        See Also\n        --------\n        DataFrame.value_counts: Count unique combinations of columns.\n\n        Examples\n        --------\n        Consider dataset containing ramen rating.\n\n        >>> df = pd.DataFrame({\n        ...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],\n        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],\n        ...     'rating': [4, 4, 3.5, 15, 5]\n        ... })\n        >>> df\n            brand style  rating\n        0  Yum Yum   cup     4.0\n        1  Yum Yum   cup     4.0\n        2  Indomie   cup     3.5\n        3  Indomie  pack    15.0\n        4  Indomie  pack     5.0\n\n        By default, it removes duplicate rows based on all columns.\n\n        >>> df.drop_duplicates()\n            brand style  rating\n        0  Yum Yum   cup     4.0\n        2  Indomie   cup     3.5\n        3  Indomie  pack    15.0\n        4  Indomie  pack     5.0\n\n        To remove duplicates on specific column(s), use ``subset``.\n\n        >>> df.drop_duplicates(subset=['brand'])\n            brand style  rating\n        0  Yum Yum   cup     4.0\n        2  Indomie   cup     3.5\n\n        To remove duplicates and keep last occurrences, use ``keep``.\n\n        >>> df.drop_duplicates(subset=['brand', 'style'], keep='last')\n            brand style  rating\n        1  Yum Yum   cup     4.0\n        2  Indomie   cup     3.5\n        4  Indomie  pack     5.0\n        "
        if self.empty:
            return self.copy()
        inplace = validate_bool_kwarg(inplace, 'inplace')
        ignore_index = validate_bool_kwarg(ignore_index, 'ignore_index')
        duplicated = self.duplicated(subset, keep=keep)
        result = self[(- duplicated)]
        if ignore_index:
            result.index = ibase.default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, subset=None, keep='first'):
        "\n        Return boolean Series denoting duplicate rows.\n\n        Considering certain columns is optional.\n\n        Parameters\n        ----------\n        subset : column label or sequence of labels, optional\n            Only consider certain columns for identifying duplicates, by\n            default use all of the columns.\n        keep : {'first', 'last', False}, default 'first'\n            Determines which duplicates (if any) to mark.\n\n            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.\n            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.\n            - False : Mark all duplicates as ``True``.\n\n        Returns\n        -------\n        Series\n            Boolean series for each duplicated rows.\n\n        See Also\n        --------\n        Index.duplicated : Equivalent method on index.\n        Series.duplicated : Equivalent method on Series.\n        Series.drop_duplicates : Remove duplicate values from Series.\n        DataFrame.drop_duplicates : Remove duplicate values from DataFrame.\n\n        Examples\n        --------\n        Consider dataset containing ramen rating.\n\n        >>> df = pd.DataFrame({\n        ...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],\n        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],\n        ...     'rating': [4, 4, 3.5, 15, 5]\n        ... })\n        >>> df\n            brand style  rating\n        0  Yum Yum   cup     4.0\n        1  Yum Yum   cup     4.0\n        2  Indomie   cup     3.5\n        3  Indomie  pack    15.0\n        4  Indomie  pack     5.0\n\n        By default, for each set of duplicated values, the first occurrence\n        is set on False and all others on True.\n\n        >>> df.duplicated()\n        0    False\n        1     True\n        2    False\n        3    False\n        4    False\n        dtype: bool\n\n        By using 'last', the last occurrence of each set of duplicated values\n        is set on False and all others on True.\n\n        >>> df.duplicated(keep='last')\n        0     True\n        1    False\n        2    False\n        3    False\n        4    False\n        dtype: bool\n\n        By setting ``keep`` on False, all duplicates are True.\n\n        >>> df.duplicated(keep=False)\n        0     True\n        1     True\n        2    False\n        3    False\n        4    False\n        dtype: bool\n\n        To find duplicates on specific column(s), use ``subset``.\n\n        >>> df.duplicated(subset=['brand'])\n        0    False\n        1     True\n        2    False\n        3     True\n        4     True\n        dtype: bool\n        "
        from pandas._libs.hashtable import SIZE_HINT_LIMIT, duplicated_int64
        if self.empty:
            return self._constructor_sliced(dtype=bool)

        def f(vals):
            (labels, shape) = algorithms.factorize(vals, size_hint=min(len(self), SIZE_HINT_LIMIT))
            return (labels.astype('i8', copy=False), len(shape))
        if (subset is None):
            subset = self.columns
        elif ((not np.iterable(subset)) or isinstance(subset, str) or (isinstance(subset, tuple) and (subset in self.columns))):
            subset = (subset,)
        subset = cast(Iterable, subset)
        diff = Index(subset).difference(self.columns)
        if (not diff.empty):
            raise KeyError(diff)
        vals = (col.values for (name, col) in self.items() if (name in subset))
        (labels, shape) = map(list, zip(*map(f, vals)))
        ids = get_group_index(labels, shape, sort=False, xnull=False)
        result = self._constructor_sliced(duplicated_int64(ids, keep), index=self.index)
        return result.__finalize__(self, method='duplicated')

    @Substitution(**_shared_doc_kwargs)
    @Appender(NDFrame.sort_values.__doc__)
    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None):
        inplace = validate_bool_kwarg(inplace, 'inplace')
        axis = self._get_axis_number(axis)
        if (not isinstance(by, list)):
            by = [by]
        if (is_sequence(ascending) and (len(by) != len(ascending))):
            raise ValueError(f'Length of ascending ({len(ascending)}) != length of by ({len(by)})')
        if (len(by) > 1):
            keys = [self._get_label_or_level_values(x, axis=axis) for x in by]
            if (key is not None):
                keys = [Series(k, name=name) for (k, name) in zip(keys, by)]
            indexer = lexsort_indexer(keys, orders=ascending, na_position=na_position, key=key)
            indexer = ensure_platform_int(indexer)
        else:
            by = by[0]
            k = self._get_label_or_level_values(by, axis=axis)
            if (key is not None):
                k = Series(k, name=by)
            if isinstance(ascending, (tuple, list)):
                ascending = ascending[0]
            indexer = nargsort(k, kind=kind, ascending=ascending, na_position=na_position, key=key)
        new_data = self._mgr.take(indexer, axis=self._get_block_manager_axis(axis), verify=False)
        if ignore_index:
            new_data.axes[1] = ibase.default_index(len(indexer))
        result = self._constructor(new_data)
        if inplace:
            return self._update_inplace(result)
        else:
            return result.__finalize__(self, method='sort_values')

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None):
        '\n        Sort object by labels (along an axis).\n\n        Returns a new DataFrame sorted by label if `inplace` argument is\n        ``False``, otherwise updates the original DataFrame and returns None.\n\n        Parameters\n        ----------\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            The axis along which to sort.  The value 0 identifies the rows,\n            and 1 identifies the columns.\n        level : int or level name or list of ints or list of level names\n            If not None, sort on values in specified index level(s).\n        ascending : bool or list of bools, default True\n            Sort ascending vs. descending. When the index is a MultiIndex the\n            sort direction can be controlled for each level individually.\n        inplace : bool, default False\n            If True, perform operation in-place.\n        kind : {\'quicksort\', \'mergesort\', \'heapsort\', \'stable\'}, default \'quicksort\'\n            Choice of sorting algorithm. See also :func:`numpy.sort` for more\n            information. `mergesort` and `stable` are the only stable algorithms. For\n            DataFrames, this option is only applied when sorting on a single\n            column or label.\n        na_position : {\'first\', \'last\'}, default \'last\'\n            Puts NaNs at the beginning if `first`; `last` puts NaNs at the end.\n            Not implemented for MultiIndex.\n        sort_remaining : bool, default True\n            If True and sorting by level and index is multilevel, sort by other\n            levels too (in order) after sorting by specified level.\n        ignore_index : bool, default False\n            If True, the resulting axis will be labeled 0, 1, …, n - 1.\n\n            .. versionadded:: 1.0.0\n\n        key : callable, optional\n            If not None, apply the key function to the index values\n            before sorting. This is similar to the `key` argument in the\n            builtin :meth:`sorted` function, with the notable difference that\n            this `key` function should be *vectorized*. It should expect an\n            ``Index`` and return an ``Index`` of the same shape. For MultiIndex\n            inputs, the key is applied *per level*.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        DataFrame or None\n            The original DataFrame sorted by the labels or None if ``inplace=True``.\n\n        See Also\n        --------\n        Series.sort_index : Sort Series by the index.\n        DataFrame.sort_values : Sort DataFrame by the value.\n        Series.sort_values : Sort Series by the value.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([1, 2, 3, 4, 5], index=[100, 29, 234, 1, 150],\n        ...                   columns=[\'A\'])\n        >>> df.sort_index()\n             A\n        1    4\n        29   2\n        100  1\n        150  5\n        234  3\n\n        By default, it sorts in ascending order, to sort in descending order,\n        use ``ascending=False``\n\n        >>> df.sort_index(ascending=False)\n             A\n        234  3\n        150  5\n        100  1\n        29   2\n        1    4\n\n        A key function can be specified which is applied to the index before\n        sorting. For a ``MultiIndex`` this is applied to each level separately.\n\n        >>> df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=[\'A\', \'b\', \'C\', \'d\'])\n        >>> df.sort_index(key=lambda x: x.str.lower())\n           a\n        A  1\n        b  2\n        C  3\n        d  4\n        '
        return super().sort_index(axis, level, ascending, inplace, kind, na_position, sort_remaining, ignore_index, key)

    def value_counts(self, subset=None, normalize=False, sort=True, ascending=False):
        "\n        Return a Series containing counts of unique rows in the DataFrame.\n\n        .. versionadded:: 1.1.0\n\n        Parameters\n        ----------\n        subset : list-like, optional\n            Columns to use when counting unique combinations.\n        normalize : bool, default False\n            Return proportions rather than frequencies.\n        sort : bool, default True\n            Sort by frequencies.\n        ascending : bool, default False\n            Sort in ascending order.\n\n        Returns\n        -------\n        Series\n\n        See Also\n        --------\n        Series.value_counts: Equivalent method on Series.\n\n        Notes\n        -----\n        The returned Series will have a MultiIndex with one level per input\n        column. By default, rows that contain any NA values are omitted from\n        the result. By default, the resulting Series will be in descending\n        order so that the first element is the most frequently-occurring row.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'num_legs': [2, 4, 4, 6],\n        ...                    'num_wings': [2, 0, 0, 0]},\n        ...                   index=['falcon', 'dog', 'cat', 'ant'])\n        >>> df\n                num_legs  num_wings\n        falcon         2          2\n        dog            4          0\n        cat            4          0\n        ant            6          0\n\n        >>> df.value_counts()\n        num_legs  num_wings\n        4         0            2\n        2         2            1\n        6         0            1\n        dtype: int64\n\n        >>> df.value_counts(sort=False)\n        num_legs  num_wings\n        2         2            1\n        4         0            2\n        6         0            1\n        dtype: int64\n\n        >>> df.value_counts(ascending=True)\n        num_legs  num_wings\n        2         2            1\n        6         0            1\n        4         0            2\n        dtype: int64\n\n        >>> df.value_counts(normalize=True)\n        num_legs  num_wings\n        4         0            0.50\n        2         2            0.25\n        6         0            0.25\n        dtype: float64\n        "
        if (subset is None):
            subset = self.columns.tolist()
        counts = self.groupby(subset).grouper.size()
        if sort:
            counts = counts.sort_values(ascending=ascending)
        if normalize:
            counts /= counts.sum()
        if (len(subset) == 1):
            counts.index = MultiIndex.from_arrays([counts.index], names=[counts.index.name])
        return counts

    def nlargest(self, n, columns, keep='first'):
        '\n        Return the first `n` rows ordered by `columns` in descending order.\n\n        Return the first `n` rows with the largest values in `columns`, in\n        descending order. The columns that are not specified are returned as\n        well, but not used for ordering.\n\n        This method is equivalent to\n        ``df.sort_values(columns, ascending=False).head(n)``, but more\n        performant.\n\n        Parameters\n        ----------\n        n : int\n            Number of rows to return.\n        columns : label or list of labels\n            Column label(s) to order by.\n        keep : {\'first\', \'last\', \'all\'}, default \'first\'\n            Where there are duplicate values:\n\n            - `first` : prioritize the first occurrence(s)\n            - `last` : prioritize the last occurrence(s)\n            - ``all`` : do not drop any duplicates, even it means\n                        selecting more than `n` items.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        DataFrame\n            The first `n` rows ordered by the given columns in descending\n            order.\n\n        See Also\n        --------\n        DataFrame.nsmallest : Return the first `n` rows ordered by `columns` in\n            ascending order.\n        DataFrame.sort_values : Sort DataFrame by the values.\n        DataFrame.head : Return the first `n` rows without re-ordering.\n\n        Notes\n        -----\n        This function cannot be used with all column types. For example, when\n        specifying columns with `object` or `category` dtypes, ``TypeError`` is\n        raised.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'population\': [59000000, 65000000, 434000,\n        ...                                   434000, 434000, 337000, 11300,\n        ...                                   11300, 11300],\n        ...                    \'GDP\': [1937894, 2583560 , 12011, 4520, 12128,\n        ...                            17036, 182, 38, 311],\n        ...                    \'alpha-2\': ["IT", "FR", "MT", "MV", "BN",\n        ...                                "IS", "NR", "TV", "AI"]},\n        ...                   index=["Italy", "France", "Malta",\n        ...                          "Maldives", "Brunei", "Iceland",\n        ...                          "Nauru", "Tuvalu", "Anguilla"])\n        >>> df\n                  population      GDP alpha-2\n        Italy       59000000  1937894      IT\n        France      65000000  2583560      FR\n        Malta         434000    12011      MT\n        Maldives      434000     4520      MV\n        Brunei        434000    12128      BN\n        Iceland       337000    17036      IS\n        Nauru          11300      182      NR\n        Tuvalu         11300       38      TV\n        Anguilla       11300      311      AI\n\n        In the following example, we will use ``nlargest`` to select the three\n        rows having the largest values in column "population".\n\n        >>> df.nlargest(3, \'population\')\n                population      GDP alpha-2\n        France    65000000  2583560      FR\n        Italy     59000000  1937894      IT\n        Malta       434000    12011      MT\n\n        When using ``keep=\'last\'``, ties are resolved in reverse order:\n\n        >>> df.nlargest(3, \'population\', keep=\'last\')\n                population      GDP alpha-2\n        France    65000000  2583560      FR\n        Italy     59000000  1937894      IT\n        Brunei      434000    12128      BN\n\n        When using ``keep=\'all\'``, all duplicate items are maintained:\n\n        >>> df.nlargest(3, \'population\', keep=\'all\')\n                  population      GDP alpha-2\n        France      65000000  2583560      FR\n        Italy       59000000  1937894      IT\n        Malta         434000    12011      MT\n        Maldives      434000     4520      MV\n        Brunei        434000    12128      BN\n\n        To order by the largest values in column "population" and then "GDP",\n        we can specify multiple columns like in the next example.\n\n        >>> df.nlargest(3, [\'population\', \'GDP\'])\n                population      GDP alpha-2\n        France    65000000  2583560      FR\n        Italy     59000000  1937894      IT\n        Brunei      434000    12128      BN\n        '
        return algorithms.SelectNFrame(self, n=n, keep=keep, columns=columns).nlargest()

    def nsmallest(self, n, columns, keep='first'):
        '\n        Return the first `n` rows ordered by `columns` in ascending order.\n\n        Return the first `n` rows with the smallest values in `columns`, in\n        ascending order. The columns that are not specified are returned as\n        well, but not used for ordering.\n\n        This method is equivalent to\n        ``df.sort_values(columns, ascending=True).head(n)``, but more\n        performant.\n\n        Parameters\n        ----------\n        n : int\n            Number of items to retrieve.\n        columns : list or str\n            Column name or names to order by.\n        keep : {\'first\', \'last\', \'all\'}, default \'first\'\n            Where there are duplicate values:\n\n            - ``first`` : take the first occurrence.\n            - ``last`` : take the last occurrence.\n            - ``all`` : do not drop any duplicates, even it means\n              selecting more than `n` items.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        DataFrame\n\n        See Also\n        --------\n        DataFrame.nlargest : Return the first `n` rows ordered by `columns` in\n            descending order.\n        DataFrame.sort_values : Sort DataFrame by the values.\n        DataFrame.head : Return the first `n` rows without re-ordering.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'population\': [59000000, 65000000, 434000,\n        ...                                   434000, 434000, 337000, 337000,\n        ...                                   11300, 11300],\n        ...                    \'GDP\': [1937894, 2583560 , 12011, 4520, 12128,\n        ...                            17036, 182, 38, 311],\n        ...                    \'alpha-2\': ["IT", "FR", "MT", "MV", "BN",\n        ...                                "IS", "NR", "TV", "AI"]},\n        ...                   index=["Italy", "France", "Malta",\n        ...                          "Maldives", "Brunei", "Iceland",\n        ...                          "Nauru", "Tuvalu", "Anguilla"])\n        >>> df\n                  population      GDP alpha-2\n        Italy       59000000  1937894      IT\n        France      65000000  2583560      FR\n        Malta         434000    12011      MT\n        Maldives      434000     4520      MV\n        Brunei        434000    12128      BN\n        Iceland       337000    17036      IS\n        Nauru         337000      182      NR\n        Tuvalu         11300       38      TV\n        Anguilla       11300      311      AI\n\n        In the following example, we will use ``nsmallest`` to select the\n        three rows having the smallest values in column "population".\n\n        >>> df.nsmallest(3, \'population\')\n                  population    GDP alpha-2\n        Tuvalu         11300     38      TV\n        Anguilla       11300    311      AI\n        Iceland       337000  17036      IS\n\n        When using ``keep=\'last\'``, ties are resolved in reverse order:\n\n        >>> df.nsmallest(3, \'population\', keep=\'last\')\n                  population  GDP alpha-2\n        Anguilla       11300  311      AI\n        Tuvalu         11300   38      TV\n        Nauru         337000  182      NR\n\n        When using ``keep=\'all\'``, all duplicate items are maintained:\n\n        >>> df.nsmallest(3, \'population\', keep=\'all\')\n                  population    GDP alpha-2\n        Tuvalu         11300     38      TV\n        Anguilla       11300    311      AI\n        Iceland       337000  17036      IS\n        Nauru         337000    182      NR\n\n        To order by the smallest values in column "population" and then "GDP", we can\n        specify multiple columns like in the next example.\n\n        >>> df.nsmallest(3, [\'population\', \'GDP\'])\n                  population  GDP alpha-2\n        Tuvalu         11300   38      TV\n        Anguilla       11300  311      AI\n        Nauru         337000  182      NR\n        '
        return algorithms.SelectNFrame(self, n=n, keep=keep, columns=columns).nsmallest()

    def swaplevel(self, i=(- 2), j=(- 1), axis=0):
        "\n        Swap levels i and j in a MultiIndex on a particular axis.\n\n        Parameters\n        ----------\n        i, j : int or str\n            Levels of the indices to be swapped. Can pass level name as string.\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            The axis to swap levels on. 0 or 'index' for row-wise, 1 or\n            'columns' for column-wise.\n\n        Returns\n        -------\n        DataFrame\n        "
        result = self.copy()
        axis = self._get_axis_number(axis)
        if (not isinstance(result._get_axis(axis), MultiIndex)):
            raise TypeError('Can only swap levels on a hierarchical axis.')
        if (axis == 0):
            assert isinstance(result.index, MultiIndex)
            result.index = result.index.swaplevel(i, j)
        else:
            assert isinstance(result.columns, MultiIndex)
            result.columns = result.columns.swaplevel(i, j)
        return result

    def reorder_levels(self, order, axis=0):
        "\n        Rearrange index levels using input order. May not drop or duplicate levels.\n\n        Parameters\n        ----------\n        order : list of int or list of str\n            List representing new level order. Reference level by number\n            (position) or by key (label).\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            Where to reorder levels.\n\n        Returns\n        -------\n        DataFrame\n        "
        axis = self._get_axis_number(axis)
        if (not isinstance(self._get_axis(axis), MultiIndex)):
            raise TypeError('Can only reorder levels on a hierarchical axis.')
        result = self.copy()
        if (axis == 0):
            assert isinstance(result.index, MultiIndex)
            result.index = result.index.reorder_levels(order)
        else:
            assert isinstance(result.columns, MultiIndex)
            result.columns = result.columns.reorder_levels(order)
        return result

    def _cmp_method(self, other, op):
        axis = 1
        (self, other) = ops.align_method_FRAME(self, other, axis, flex=False, level=None)
        new_data = self._dispatch_frame_op(other, op, axis=axis)
        return self._construct_result(new_data)

    def _arith_method(self, other, op):
        if ops.should_reindex_frame_op(self, other, op, 1, 1, None, None):
            return ops.frame_arith_method_with_reindex(self, other, op)
        axis = 1
        (self, other) = ops.align_method_FRAME(self, other, axis, flex=True, level=None)
        new_data = self._dispatch_frame_op(other, op, axis=axis)
        return self._construct_result(new_data)
    _logical_method = _arith_method

    def _dispatch_frame_op(self, right, func, axis=None):
        '\n        Evaluate the frame operation func(left, right) by evaluating\n        column-by-column, dispatching to the Series implementation.\n\n        Parameters\n        ----------\n        right : scalar, Series, or DataFrame\n        func : arithmetic or comparison operator\n        axis : {None, 0, 1}\n\n        Returns\n        -------\n        DataFrame\n        '
        array_op = ops.get_array_op(func)
        right = lib.item_from_zerodim(right)
        if (not is_list_like(right)):
            bm = self._mgr.apply(array_op, right=right)
            return type(self)(bm)
        elif isinstance(right, DataFrame):
            assert self.index.equals(right.index)
            assert self.columns.equals(right.columns)
            bm = self._mgr.operate_blockwise(right._mgr, array_op)
            return type(self)(bm)
        elif (isinstance(right, Series) and (axis == 1)):
            assert right.index.equals(self.columns)
            right = right._values
            assert (not isinstance(right, np.ndarray))
            arrays = [array_op(_left, _right) for (_left, _right) in zip(self._iter_column_arrays(), right)]
        elif isinstance(right, Series):
            assert right.index.equals(self.index)
            right = right._values
            arrays = [array_op(left, right) for left in self._iter_column_arrays()]
        else:
            raise NotImplementedError(right)
        return type(self)._from_arrays(arrays, self.columns, self.index, verify_integrity=False)

    def _combine_frame(self, other, func, fill_value=None):
        if (fill_value is None):
            _arith_op = func
        else:

            def _arith_op(left, right):
                (left, right) = ops.fill_binop(left, right, fill_value)
                return func(left, right)
        new_data = self._dispatch_frame_op(other, _arith_op)
        return new_data

    def _construct_result(self, result):
        '\n        Wrap the result of an arithmetic, comparison, or logical operation.\n\n        Parameters\n        ----------\n        result : DataFrame\n\n        Returns\n        -------\n        DataFrame\n        '
        out = self._constructor(result, copy=False)
        out.columns = self.columns
        out.index = self.index
        return out

    def __divmod__(self, other):
        div = (self // other)
        mod = (self - (div * other))
        return (div, mod)

    def __rdivmod__(self, other):
        div = (other // self)
        mod = (other - (div * self))
        return (div, mod)

    @doc(_shared_docs['compare'], '\nReturns\n-------\nDataFrame\n    DataFrame that shows the differences stacked side by side.\n\n    The resulting index will be a MultiIndex with \'self\' and \'other\'\n    stacked alternately at the inner level.\n\nRaises\n------\nValueError\n    When the two DataFrames don\'t have identical labels or shape.\n\nSee Also\n--------\nSeries.compare : Compare with another Series and show differences.\nDataFrame.equals : Test whether two objects contain the same elements.\n\nNotes\n-----\nMatching NaNs will not appear as a difference.\n\nCan only compare identically-labeled\n(i.e. same shape, identical row and column labels) DataFrames\n\nExamples\n--------\n>>> df = pd.DataFrame(\n...     {{\n...         "col1": ["a", "a", "b", "b", "a"],\n...         "col2": [1.0, 2.0, 3.0, np.nan, 5.0],\n...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0]\n...     }},\n...     columns=["col1", "col2", "col3"],\n... )\n>>> df\n  col1  col2  col3\n0    a   1.0   1.0\n1    a   2.0   2.0\n2    b   3.0   3.0\n3    b   NaN   4.0\n4    a   5.0   5.0\n\n>>> df2 = df.copy()\n>>> df2.loc[0, \'col1\'] = \'c\'\n>>> df2.loc[2, \'col3\'] = 4.0\n>>> df2\n  col1  col2  col3\n0    c   1.0   1.0\n1    a   2.0   2.0\n2    b   3.0   4.0\n3    b   NaN   4.0\n4    a   5.0   5.0\n\nAlign the differences on columns\n\n>>> df.compare(df2)\n  col1       col3\n  self other self other\n0    a     c  NaN   NaN\n2  NaN   NaN  3.0   4.0\n\nStack the differences on rows\n\n>>> df.compare(df2, align_axis=0)\n        col1  col3\n0 self     a   NaN\n  other    c   NaN\n2 self   NaN   3.0\n  other  NaN   4.0\n\nKeep the equal values\n\n>>> df.compare(df2, keep_equal=True)\n  col1       col3\n  self other self other\n0    a     c  1.0   1.0\n2    b     b  3.0   4.0\n\nKeep all original rows and columns\n\n>>> df.compare(df2, keep_shape=True)\n  col1       col2       col3\n  self other self other self other\n0    a     c  NaN   NaN  NaN   NaN\n1  NaN   NaN  NaN   NaN  NaN   NaN\n2  NaN   NaN  NaN   NaN  3.0   4.0\n3  NaN   NaN  NaN   NaN  NaN   NaN\n4  NaN   NaN  NaN   NaN  NaN   NaN\n\nKeep all original rows and columns and also all original values\n\n>>> df.compare(df2, keep_shape=True, keep_equal=True)\n  col1       col2       col3\n  self other self other self other\n0    a     c  1.0   1.0  1.0   1.0\n1    a     a  2.0   2.0  2.0   2.0\n2    b     b  3.0   3.0  3.0   4.0\n3    b     b  NaN   NaN  4.0   4.0\n4    a     a  5.0   5.0  5.0   5.0\n', klass=_shared_doc_kwargs['klass'])
    def compare(self, other, align_axis=1, keep_shape=False, keep_equal=False):
        return super().compare(other=other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal)

    def combine(self, other, func, fill_value=None, overwrite=True):
        "\n        Perform column-wise combine with another DataFrame.\n\n        Combines a DataFrame with `other` DataFrame using `func`\n        to element-wise combine columns. The row and column indexes of the\n        resulting DataFrame will be the union of the two.\n\n        Parameters\n        ----------\n        other : DataFrame\n            The DataFrame to merge column-wise.\n        func : function\n            Function that takes two series as inputs and return a Series or a\n            scalar. Used to merge the two dataframes column by columns.\n        fill_value : scalar value, default None\n            The value to fill NaNs with prior to passing any column to the\n            merge func.\n        overwrite : bool, default True\n            If True, columns in `self` that do not exist in `other` will be\n            overwritten with NaNs.\n\n        Returns\n        -------\n        DataFrame\n            Combination of the provided DataFrames.\n\n        See Also\n        --------\n        DataFrame.combine_first : Combine two DataFrame objects and default to\n            non-null values in frame calling the method.\n\n        Examples\n        --------\n        Combine using a simple function that chooses the smaller column.\n\n        >>> df1 = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})\n        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})\n        >>> take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2\n        >>> df1.combine(df2, take_smaller)\n           A  B\n        0  0  3\n        1  0  3\n\n        Example using a true element-wise combine function.\n\n        >>> df1 = pd.DataFrame({'A': [5, 0], 'B': [2, 4]})\n        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})\n        >>> df1.combine(df2, np.minimum)\n           A  B\n        0  1  2\n        1  0  3\n\n        Using `fill_value` fills Nones prior to passing the column to the\n        merge function.\n\n        >>> df1 = pd.DataFrame({'A': [0, 0], 'B': [None, 4]})\n        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})\n        >>> df1.combine(df2, take_smaller, fill_value=-5)\n           A    B\n        0  0 -5.0\n        1  0  4.0\n\n        However, if the same element in both dataframes is None, that None\n        is preserved\n\n        >>> df1 = pd.DataFrame({'A': [0, 0], 'B': [None, 4]})\n        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [None, 3]})\n        >>> df1.combine(df2, take_smaller, fill_value=-5)\n            A    B\n        0  0 -5.0\n        1  0  3.0\n\n        Example that demonstrates the use of `overwrite` and behavior when\n        the axis differ between the dataframes.\n\n        >>> df1 = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})\n        >>> df2 = pd.DataFrame({'B': [3, 3], 'C': [-10, 1], }, index=[1, 2])\n        >>> df1.combine(df2, take_smaller)\n             A    B     C\n        0  NaN  NaN   NaN\n        1  NaN  3.0 -10.0\n        2  NaN  3.0   1.0\n\n        >>> df1.combine(df2, take_smaller, overwrite=False)\n             A    B     C\n        0  0.0  NaN   NaN\n        1  0.0  3.0 -10.0\n        2  NaN  3.0   1.0\n\n        Demonstrating the preference of the passed in dataframe.\n\n        >>> df2 = pd.DataFrame({'B': [3, 3], 'C': [1, 1], }, index=[1, 2])\n        >>> df2.combine(df1, take_smaller)\n           A    B   C\n        0  0.0  NaN NaN\n        1  0.0  3.0 NaN\n        2  NaN  3.0 NaN\n\n        >>> df2.combine(df1, take_smaller, overwrite=False)\n             A    B   C\n        0  0.0  NaN NaN\n        1  0.0  3.0 1.0\n        2  NaN  3.0 1.0\n        "
        other_idxlen = len(other.index)
        (this, other) = self.align(other, copy=False)
        new_index = this.index
        if (other.empty and (len(new_index) == len(self.index))):
            return self.copy()
        if (self.empty and (len(other) == other_idxlen)):
            return other.copy()
        new_columns = this.columns.union(other.columns)
        do_fill = (fill_value is not None)
        result = {}
        for col in new_columns:
            series = this[col]
            otherSeries = other[col]
            this_dtype = series.dtype
            other_dtype = otherSeries.dtype
            this_mask = isna(series)
            other_mask = isna(otherSeries)
            if ((not overwrite) and other_mask.all()):
                result[col] = this[col].copy()
                continue
            if do_fill:
                series = series.copy()
                otherSeries = otherSeries.copy()
                series[this_mask] = fill_value
                otherSeries[other_mask] = fill_value
            if (col not in self.columns):
                new_dtype = other_dtype
                try:
                    series = series.astype(new_dtype, copy=False)
                except ValueError:
                    pass
            else:
                new_dtype = find_common_type([this_dtype, other_dtype])
                if (not is_dtype_equal(this_dtype, new_dtype)):
                    series = series.astype(new_dtype)
                if (not is_dtype_equal(other_dtype, new_dtype)):
                    otherSeries = otherSeries.astype(new_dtype)
            arr = func(series, otherSeries)
            arr = maybe_downcast_to_dtype(arr, new_dtype)
            result[col] = arr
        return self._constructor(result, index=new_index, columns=new_columns)

    def combine_first(self, other):
        "\n        Update null elements with value in the same location in `other`.\n\n        Combine two DataFrame objects by filling null values in one DataFrame\n        with non-null values from other DataFrame. The row and column indexes\n        of the resulting DataFrame will be the union of the two.\n\n        Parameters\n        ----------\n        other : DataFrame\n            Provided DataFrame to use to fill null values.\n\n        Returns\n        -------\n        DataFrame\n\n        See Also\n        --------\n        DataFrame.combine : Perform series-wise operation on two DataFrames\n            using a given function.\n\n        Examples\n        --------\n        >>> df1 = pd.DataFrame({'A': [None, 0], 'B': [None, 4]})\n        >>> df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})\n        >>> df1.combine_first(df2)\n             A    B\n        0  1.0  3.0\n        1  0.0  4.0\n\n        Null values still persist if the location of that null value\n        does not exist in `other`\n\n        >>> df1 = pd.DataFrame({'A': [None, 0], 'B': [4, None]})\n        >>> df2 = pd.DataFrame({'B': [3, 3], 'C': [1, 1]}, index=[1, 2])\n        >>> df1.combine_first(df2)\n             A    B    C\n        0  NaN  4.0  NaN\n        1  0.0  3.0  1.0\n        2  NaN  3.0  1.0\n        "
        import pandas.core.computation.expressions as expressions

        def combiner(x, y):
            mask = extract_array(isna(x))
            x_values = extract_array(x, extract_numpy=True)
            y_values = extract_array(y, extract_numpy=True)
            if (y.name not in self.columns):
                return y_values
            return expressions.where(mask, y_values, x_values)
        return self.combine(other, combiner, overwrite=False)

    def update(self, other, join='left', overwrite=True, filter_func=None, errors='ignore'):
        "\n        Modify in place using non-NA values from another DataFrame.\n\n        Aligns on indices. There is no return value.\n\n        Parameters\n        ----------\n        other : DataFrame, or object coercible into a DataFrame\n            Should have at least one matching index/column label\n            with the original DataFrame. If a Series is passed,\n            its name attribute must be set, and that will be\n            used as the column name to align with the original DataFrame.\n        join : {'left'}, default 'left'\n            Only left join is implemented, keeping the index and columns of the\n            original object.\n        overwrite : bool, default True\n            How to handle non-NA values for overlapping keys:\n\n            * True: overwrite original DataFrame's values\n              with values from `other`.\n            * False: only update values that are NA in\n              the original DataFrame.\n\n        filter_func : callable(1d-array) -> bool 1d-array, optional\n            Can choose to replace values other than NA. Return True for values\n            that should be updated.\n        errors : {'raise', 'ignore'}, default 'ignore'\n            If 'raise', will raise a ValueError if the DataFrame and `other`\n            both contain non-NA data in the same place.\n\n            .. versionchanged:: 0.24.0\n               Changed from `raise_conflict=False|True`\n               to `errors='ignore'|'raise'`.\n\n        Returns\n        -------\n        None : method directly changes calling object\n\n        Raises\n        ------\n        ValueError\n            * When `errors='raise'` and there's overlapping non-NA data.\n            * When `errors` is not either `'ignore'` or `'raise'`\n        NotImplementedError\n            * If `join != 'left'`\n\n        See Also\n        --------\n        dict.update : Similar method for dictionaries.\n        DataFrame.merge : For column(s)-on-column(s) operations.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A': [1, 2, 3],\n        ...                    'B': [400, 500, 600]})\n        >>> new_df = pd.DataFrame({'B': [4, 5, 6],\n        ...                        'C': [7, 8, 9]})\n        >>> df.update(new_df)\n        >>> df\n           A  B\n        0  1  4\n        1  2  5\n        2  3  6\n\n        The DataFrame's length does not increase as a result of the update,\n        only values at matching index/column labels are updated.\n\n        >>> df = pd.DataFrame({'A': ['a', 'b', 'c'],\n        ...                    'B': ['x', 'y', 'z']})\n        >>> new_df = pd.DataFrame({'B': ['d', 'e', 'f', 'g', 'h', 'i']})\n        >>> df.update(new_df)\n        >>> df\n           A  B\n        0  a  d\n        1  b  e\n        2  c  f\n\n        For Series, its name attribute must be set.\n\n        >>> df = pd.DataFrame({'A': ['a', 'b', 'c'],\n        ...                    'B': ['x', 'y', 'z']})\n        >>> new_column = pd.Series(['d', 'e'], name='B', index=[0, 2])\n        >>> df.update(new_column)\n        >>> df\n           A  B\n        0  a  d\n        1  b  y\n        2  c  e\n        >>> df = pd.DataFrame({'A': ['a', 'b', 'c'],\n        ...                    'B': ['x', 'y', 'z']})\n        >>> new_df = pd.DataFrame({'B': ['d', 'e']}, index=[1, 2])\n        >>> df.update(new_df)\n        >>> df\n           A  B\n        0  a  x\n        1  b  d\n        2  c  e\n\n        If `other` contains NaNs the corresponding values are not updated\n        in the original dataframe.\n\n        >>> df = pd.DataFrame({'A': [1, 2, 3],\n        ...                    'B': [400, 500, 600]})\n        >>> new_df = pd.DataFrame({'B': [4, np.nan, 6]})\n        >>> df.update(new_df)\n        >>> df\n           A      B\n        0  1    4.0\n        1  2  500.0\n        2  3    6.0\n        "
        import pandas.core.computation.expressions as expressions
        if (join != 'left'):
            raise NotImplementedError('Only left join is supported')
        if (errors not in ['ignore', 'raise']):
            raise ValueError("The parameter errors must be either 'ignore' or 'raise'")
        if (not isinstance(other, DataFrame)):
            other = DataFrame(other)
        other = other.reindex_like(self)
        for col in self.columns:
            this = self[col]._values
            that = other[col]._values
            if (filter_func is not None):
                with np.errstate(all='ignore'):
                    mask = ((~ filter_func(this)) | isna(that))
            else:
                if (errors == 'raise'):
                    mask_this = notna(that)
                    mask_that = notna(this)
                    if any((mask_this & mask_that)):
                        raise ValueError('Data overlaps.')
                if overwrite:
                    mask = isna(that)
                else:
                    mask = notna(this)
            if mask.all():
                continue
            self[col] = expressions.where(mask, this, that)

    @Appender('\nExamples\n--------\n>>> df = pd.DataFrame({\'Animal\': [\'Falcon\', \'Falcon\',\n...                               \'Parrot\', \'Parrot\'],\n...                    \'Max Speed\': [380., 370., 24., 26.]})\n>>> df\n   Animal  Max Speed\n0  Falcon      380.0\n1  Falcon      370.0\n2  Parrot       24.0\n3  Parrot       26.0\n>>> df.groupby([\'Animal\']).mean()\n        Max Speed\nAnimal\nFalcon      375.0\nParrot       25.0\n\n**Hierarchical Indexes**\n\nWe can groupby different levels of a hierarchical index\nusing the `level` parameter:\n\n>>> arrays = [[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],\n...           [\'Captive\', \'Wild\', \'Captive\', \'Wild\']]\n>>> index = pd.MultiIndex.from_arrays(arrays, names=(\'Animal\', \'Type\'))\n>>> df = pd.DataFrame({\'Max Speed\': [390., 350., 30., 20.]},\n...                   index=index)\n>>> df\n                Max Speed\nAnimal Type\nFalcon Captive      390.0\n       Wild         350.0\nParrot Captive       30.0\n       Wild          20.0\n>>> df.groupby(level=0).mean()\n        Max Speed\nAnimal\nFalcon      370.0\nParrot       25.0\n>>> df.groupby(level="Type").mean()\n         Max Speed\nType\nCaptive      210.0\nWild         185.0\n\nWe can also choose to include NA in group keys or not by setting\n`dropna` parameter, the default setting is `True`:\n\n>>> l = [[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]]\n>>> df = pd.DataFrame(l, columns=["a", "b", "c"])\n\n>>> df.groupby(by=["b"]).sum()\n    a   c\nb\n1.0 2   3\n2.0 2   5\n\n>>> df.groupby(by=["b"], dropna=False).sum()\n    a   c\nb\n1.0 2   3\n2.0 2   5\nNaN 1   4\n\n>>> l = [["a", 12, 12], [None, 12.3, 33.], ["b", 12.3, 123], ["a", 1, 1]]\n>>> df = pd.DataFrame(l, columns=["a", "b", "c"])\n\n>>> df.groupby(by="a").sum()\n    b     c\na\na   13.0   13.0\nb   12.3  123.0\n\n>>> df.groupby(by="a", dropna=False).sum()\n    b     c\na\na   13.0   13.0\nb   12.3  123.0\nNaN 12.3   33.0\n')
    @Appender((_shared_docs['groupby'] % _shared_doc_kwargs))
    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=no_default, observed=False, dropna=True):
        from pandas.core.groupby.generic import DataFrameGroupBy
        if (squeeze is not no_default):
            warnings.warn('The `squeeze` parameter is deprecated and will be removed in a future version.', FutureWarning, stacklevel=2)
        else:
            squeeze = False
        if ((level is None) and (by is None)):
            raise TypeError("You have to supply one of 'by' and 'level'")
        axis = self._get_axis_number(axis)
        return DataFrameGroupBy(obj=self, keys=by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, squeeze=squeeze, observed=observed, dropna=dropna)
    _shared_docs['pivot'] = '\n        Return reshaped DataFrame organized by given index / column values.\n\n        Reshape data (produce a "pivot" table) based on column values. Uses\n        unique values from specified `index` / `columns` to form axes of the\n        resulting DataFrame. This function does not support data\n        aggregation, multiple values will result in a MultiIndex in the\n        columns. See the :ref:`User Guide <reshaping>` for more on reshaping.\n\n        Parameters\n        ----------%s\n        index : str or object or a list of str, optional\n            Column to use to make new frame\'s index. If None, uses\n            existing index.\n\n            .. versionchanged:: 1.1.0\n               Also accept list of index names.\n\n        columns : str or object or a list of str\n            Column to use to make new frame\'s columns.\n\n            .. versionchanged:: 1.1.0\n               Also accept list of columns names.\n\n        values : str, object or a list of the previous, optional\n            Column(s) to use for populating new frame\'s values. If not\n            specified, all remaining columns will be used and the result will\n            have hierarchically indexed columns.\n\n        Returns\n        -------\n        DataFrame\n            Returns reshaped DataFrame.\n\n        Raises\n        ------\n        ValueError:\n            When there are any `index`, `columns` combinations with multiple\n            values. `DataFrame.pivot_table` when you need to aggregate.\n\n        See Also\n        --------\n        DataFrame.pivot_table : Generalization of pivot that can handle\n            duplicate values for one index/column pair.\n        DataFrame.unstack : Pivot based on the index values instead of a\n            column.\n        wide_to_long : Wide panel to long format. Less flexible but more\n            user-friendly than melt.\n\n        Notes\n        -----\n        For finer-tuned control, see hierarchical indexing documentation along\n        with the related stack/unstack methods.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'foo\': [\'one\', \'one\', \'one\', \'two\', \'two\',\n        ...                            \'two\'],\n        ...                    \'bar\': [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\'],\n        ...                    \'baz\': [1, 2, 3, 4, 5, 6],\n        ...                    \'zoo\': [\'x\', \'y\', \'z\', \'q\', \'w\', \'t\']})\n        >>> df\n            foo   bar  baz  zoo\n        0   one   A    1    x\n        1   one   B    2    y\n        2   one   C    3    z\n        3   two   A    4    q\n        4   two   B    5    w\n        5   two   C    6    t\n\n        >>> df.pivot(index=\'foo\', columns=\'bar\', values=\'baz\')\n        bar  A   B   C\n        foo\n        one  1   2   3\n        two  4   5   6\n\n        >>> df.pivot(index=\'foo\', columns=\'bar\')[\'baz\']\n        bar  A   B   C\n        foo\n        one  1   2   3\n        two  4   5   6\n\n        >>> df.pivot(index=\'foo\', columns=\'bar\', values=[\'baz\', \'zoo\'])\n              baz       zoo\n        bar   A  B  C   A  B  C\n        foo\n        one   1  2  3   x  y  z\n        two   4  5  6   q  w  t\n\n        You could also assign a list of column names or a list of index names.\n\n        >>> df = pd.DataFrame({\n        ...        "lev1": [1, 1, 1, 2, 2, 2],\n        ...        "lev2": [1, 1, 2, 1, 1, 2],\n        ...        "lev3": [1, 2, 1, 2, 1, 2],\n        ...        "lev4": [1, 2, 3, 4, 5, 6],\n        ...        "values": [0, 1, 2, 3, 4, 5]})\n        >>> df\n            lev1 lev2 lev3 lev4 values\n        0   1    1    1    1    0\n        1   1    1    2    2    1\n        2   1    2    1    3    2\n        3   2    1    2    4    3\n        4   2    1    1    5    4\n        5   2    2    2    6    5\n\n        >>> df.pivot(index="lev1", columns=["lev2", "lev3"],values="values")\n        lev2    1         2\n        lev3    1    2    1    2\n        lev1\n        1     0.0  1.0  2.0  NaN\n        2     4.0  3.0  NaN  5.0\n\n        >>> df.pivot(index=["lev1", "lev2"], columns=["lev3"],values="values")\n              lev3    1    2\n        lev1  lev2\n           1     1  0.0  1.0\n                 2  2.0  NaN\n           2     1  4.0  3.0\n                 2  NaN  5.0\n\n        A ValueError is raised if there are any duplicates.\n\n        >>> df = pd.DataFrame({"foo": [\'one\', \'one\', \'two\', \'two\'],\n        ...                    "bar": [\'A\', \'A\', \'B\', \'C\'],\n        ...                    "baz": [1, 2, 3, 4]})\n        >>> df\n           foo bar  baz\n        0  one   A    1\n        1  one   A    2\n        2  two   B    3\n        3  two   C    4\n\n        Notice that the first two rows are the same for our `index`\n        and `columns` arguments.\n\n        >>> df.pivot(index=\'foo\', columns=\'bar\', values=\'baz\')\n        Traceback (most recent call last):\n           ...\n        ValueError: Index contains duplicate entries, cannot reshape\n        '

    @Substitution('')
    @Appender(_shared_docs['pivot'])
    def pivot(self, index=None, columns=None, values=None):
        from pandas.core.reshape.pivot import pivot
        return pivot(self, index=index, columns=columns, values=values)
    _shared_docs['pivot_table'] = '\n        Create a spreadsheet-style pivot table as a DataFrame.\n\n        The levels in the pivot table will be stored in MultiIndex objects\n        (hierarchical indexes) on the index and columns of the result DataFrame.\n\n        Parameters\n        ----------%s\n        values : column to aggregate, optional\n        index : column, Grouper, array, or list of the previous\n            If an array is passed, it must be the same length as the data. The\n            list can contain any of the other types (except list).\n            Keys to group by on the pivot table index.  If an array is passed,\n            it is being used as the same manner as column values.\n        columns : column, Grouper, array, or list of the previous\n            If an array is passed, it must be the same length as the data. The\n            list can contain any of the other types (except list).\n            Keys to group by on the pivot table column.  If an array is passed,\n            it is being used as the same manner as column values.\n        aggfunc : function, list of functions, dict, default numpy.mean\n            If list of functions passed, the resulting pivot table will have\n            hierarchical columns whose top level are the function names\n            (inferred from the function objects themselves)\n            If dict is passed, the key is column to aggregate and value\n            is function or list of functions.\n        fill_value : scalar, default None\n            Value to replace missing values with (in the resulting pivot table,\n            after aggregation).\n        margins : bool, default False\n            Add all row / columns (e.g. for subtotal / grand totals).\n        dropna : bool, default True\n            Do not include columns whose entries are all NaN.\n        margins_name : str, default \'All\'\n            Name of the row / column that will contain the totals\n            when margins is True.\n        observed : bool, default False\n            This only applies if any of the groupers are Categoricals.\n            If True: only show observed values for categorical groupers.\n            If False: show all values for categorical groupers.\n\n            .. versionchanged:: 0.25.0\n\n        Returns\n        -------\n        DataFrame\n            An Excel style pivot table.\n\n        See Also\n        --------\n        DataFrame.pivot : Pivot without aggregation that can handle\n            non-numeric data.\n        DataFrame.melt: Unpivot a DataFrame from wide to long format,\n            optionally leaving identifiers set.\n        wide_to_long : Wide panel to long format. Less flexible but more\n            user-friendly than melt.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",\n        ...                          "bar", "bar", "bar", "bar"],\n        ...                    "B": ["one", "one", "one", "two", "two",\n        ...                          "one", "one", "two", "two"],\n        ...                    "C": ["small", "large", "large", "small",\n        ...                          "small", "large", "small", "small",\n        ...                          "large"],\n        ...                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],\n        ...                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})\n        >>> df\n             A    B      C  D  E\n        0  foo  one  small  1  2\n        1  foo  one  large  2  4\n        2  foo  one  large  2  5\n        3  foo  two  small  3  5\n        4  foo  two  small  3  6\n        5  bar  one  large  4  6\n        6  bar  one  small  5  8\n        7  bar  two  small  6  9\n        8  bar  two  large  7  9\n\n        This first example aggregates values by taking the sum.\n\n        >>> table = pd.pivot_table(df, values=\'D\', index=[\'A\', \'B\'],\n        ...                     columns=[\'C\'], aggfunc=np.sum)\n        >>> table\n        C        large  small\n        A   B\n        bar one    4.0    5.0\n            two    7.0    6.0\n        foo one    4.0    1.0\n            two    NaN    6.0\n\n        We can also fill missing values using the `fill_value` parameter.\n\n        >>> table = pd.pivot_table(df, values=\'D\', index=[\'A\', \'B\'],\n        ...                     columns=[\'C\'], aggfunc=np.sum, fill_value=0)\n        >>> table\n        C        large  small\n        A   B\n        bar one      4      5\n            two      7      6\n        foo one      4      1\n            two      0      6\n\n        The next example aggregates by taking the mean across multiple columns.\n\n        >>> table = pd.pivot_table(df, values=[\'D\', \'E\'], index=[\'A\', \'C\'],\n        ...                     aggfunc={\'D\': np.mean,\n        ...                              \'E\': np.mean})\n        >>> table\n                        D         E\n        A   C\n        bar large  5.500000  7.500000\n            small  5.500000  8.500000\n        foo large  2.000000  4.500000\n            small  2.333333  4.333333\n\n        We can also calculate multiple types of aggregations for any given\n        value column.\n\n        >>> table = pd.pivot_table(df, values=[\'D\', \'E\'], index=[\'A\', \'C\'],\n        ...                     aggfunc={\'D\': np.mean,\n        ...                              \'E\': [min, max, np.mean]})\n        >>> table\n                        D    E\n                    mean  max      mean  min\n        A   C\n        bar large  5.500000  9.0  7.500000  6.0\n            small  5.500000  9.0  8.500000  8.0\n        foo large  2.000000  5.0  4.500000  4.0\n            small  2.333333  6.0  4.333333  2.0\n        '

    @Substitution('')
    @Appender(_shared_docs['pivot_table'])
    def pivot_table(self, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False):
        from pandas.core.reshape.pivot import pivot_table
        return pivot_table(self, values=values, index=index, columns=columns, aggfunc=aggfunc, fill_value=fill_value, margins=margins, dropna=dropna, margins_name=margins_name, observed=observed)

    def stack(self, level=(- 1), dropna=True):
        "\n        Stack the prescribed level(s) from columns to index.\n\n        Return a reshaped DataFrame or Series having a multi-level\n        index with one or more new inner-most levels compared to the current\n        DataFrame. The new inner-most levels are created by pivoting the\n        columns of the current dataframe:\n\n          - if the columns have a single level, the output is a Series;\n          - if the columns have multiple levels, the new index\n            level(s) is (are) taken from the prescribed level(s) and\n            the output is a DataFrame.\n\n        Parameters\n        ----------\n        level : int, str, list, default -1\n            Level(s) to stack from the column axis onto the index\n            axis, defined as one index or label, or a list of indices\n            or labels.\n        dropna : bool, default True\n            Whether to drop rows in the resulting Frame/Series with\n            missing values. Stacking a column level onto the index\n            axis can create combinations of index and column values\n            that are missing from the original dataframe. See Examples\n            section.\n\n        Returns\n        -------\n        DataFrame or Series\n            Stacked dataframe or series.\n\n        See Also\n        --------\n        DataFrame.unstack : Unstack prescribed level(s) from index axis\n             onto column axis.\n        DataFrame.pivot : Reshape dataframe from long format to wide\n             format.\n        DataFrame.pivot_table : Create a spreadsheet-style pivot table\n             as a DataFrame.\n\n        Notes\n        -----\n        The function is named by analogy with a collection of books\n        being reorganized from being side by side on a horizontal\n        position (the columns of the dataframe) to being stacked\n        vertically on top of each other (in the index of the\n        dataframe).\n\n        Examples\n        --------\n        **Single level columns**\n\n        >>> df_single_level_cols = pd.DataFrame([[0, 1], [2, 3]],\n        ...                                     index=['cat', 'dog'],\n        ...                                     columns=['weight', 'height'])\n\n        Stacking a dataframe with a single level column axis returns a Series:\n\n        >>> df_single_level_cols\n             weight height\n        cat       0      1\n        dog       2      3\n        >>> df_single_level_cols.stack()\n        cat  weight    0\n             height    1\n        dog  weight    2\n             height    3\n        dtype: int64\n\n        **Multi level columns: simple case**\n\n        >>> multicol1 = pd.MultiIndex.from_tuples([('weight', 'kg'),\n        ...                                        ('weight', 'pounds')])\n        >>> df_multi_level_cols1 = pd.DataFrame([[1, 2], [2, 4]],\n        ...                                     index=['cat', 'dog'],\n        ...                                     columns=multicol1)\n\n        Stacking a dataframe with a multi-level column axis:\n\n        >>> df_multi_level_cols1\n             weight\n                 kg    pounds\n        cat       1        2\n        dog       2        4\n        >>> df_multi_level_cols1.stack()\n                    weight\n        cat kg           1\n            pounds       2\n        dog kg           2\n            pounds       4\n\n        **Missing values**\n\n        >>> multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),\n        ...                                        ('height', 'm')])\n        >>> df_multi_level_cols2 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],\n        ...                                     index=['cat', 'dog'],\n        ...                                     columns=multicol2)\n\n        It is common to have missing values when stacking a dataframe\n        with multi-level columns, as the stacked dataframe typically\n        has more values than the original dataframe. Missing values\n        are filled with NaNs:\n\n        >>> df_multi_level_cols2\n            weight height\n                kg      m\n        cat    1.0    2.0\n        dog    3.0    4.0\n        >>> df_multi_level_cols2.stack()\n                height  weight\n        cat kg     NaN     1.0\n            m      2.0     NaN\n        dog kg     NaN     3.0\n            m      4.0     NaN\n\n        **Prescribing the level(s) to be stacked**\n\n        The first parameter controls which level or levels are stacked:\n\n        >>> df_multi_level_cols2.stack(0)\n                     kg    m\n        cat height  NaN  2.0\n            weight  1.0  NaN\n        dog height  NaN  4.0\n            weight  3.0  NaN\n        >>> df_multi_level_cols2.stack([0, 1])\n        cat  height  m     2.0\n             weight  kg    1.0\n        dog  height  m     4.0\n             weight  kg    3.0\n        dtype: float64\n\n        **Dropping missing values**\n\n        >>> df_multi_level_cols3 = pd.DataFrame([[None, 1.0], [2.0, 3.0]],\n        ...                                     index=['cat', 'dog'],\n        ...                                     columns=multicol2)\n\n        Note that rows where all values are missing are dropped by\n        default but this behaviour can be controlled via the dropna\n        keyword parameter:\n\n        >>> df_multi_level_cols3\n            weight height\n                kg      m\n        cat    NaN    1.0\n        dog    2.0    3.0\n        >>> df_multi_level_cols3.stack(dropna=False)\n                height  weight\n        cat kg     NaN     NaN\n            m      1.0     NaN\n        dog kg     NaN     2.0\n            m      3.0     NaN\n        >>> df_multi_level_cols3.stack(dropna=True)\n                height  weight\n        cat m      1.0     NaN\n        dog kg     NaN     2.0\n            m      3.0     NaN\n        "
        from pandas.core.reshape.reshape import stack, stack_multiple
        if isinstance(level, (tuple, list)):
            result = stack_multiple(self, level, dropna=dropna)
        else:
            result = stack(self, level, dropna=dropna)
        return result.__finalize__(self, method='stack')

    def explode(self, column, ignore_index=False):
        "\n        Transform each element of a list-like to a row, replicating index values.\n\n        .. versionadded:: 0.25.0\n\n        Parameters\n        ----------\n        column : str or tuple\n            Column to explode.\n        ignore_index : bool, default False\n            If True, the resulting index will be labeled 0, 1, …, n - 1.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        DataFrame\n            Exploded lists to rows of the subset columns;\n            index will be duplicated for these rows.\n\n        Raises\n        ------\n        ValueError :\n            if columns of the frame are not unique.\n\n        See Also\n        --------\n        DataFrame.unstack : Pivot a level of the (necessarily hierarchical)\n            index labels.\n        DataFrame.melt : Unpivot a DataFrame from wide format to long format.\n        Series.explode : Explode a DataFrame from list-like columns to long format.\n\n        Notes\n        -----\n        This routine will explode list-likes including lists, tuples, sets,\n        Series, and np.ndarray. The result dtype of the subset rows will\n        be object. Scalars will be returned unchanged, and empty list-likes will\n        result in a np.nan for that row. In addition, the ordering of rows in the\n        output will be non-deterministic when exploding sets.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A': [[1, 2, 3], 'foo', [], [3, 4]], 'B': 1})\n        >>> df\n                   A  B\n        0  [1, 2, 3]  1\n        1        foo  1\n        2         []  1\n        3     [3, 4]  1\n\n        >>> df.explode('A')\n             A  B\n        0    1  1\n        0    2  1\n        0    3  1\n        1  foo  1\n        2  NaN  1\n        3    3  1\n        3    4  1\n        "
        if (not (is_scalar(column) or isinstance(column, tuple))):
            raise ValueError('column must be a scalar')
        if (not self.columns.is_unique):
            raise ValueError('columns must be unique')
        df = self.reset_index(drop=True)
        result = df[column].explode()
        result = df.drop([column], axis=1).join(result)
        if ignore_index:
            result.index = ibase.default_index(len(result))
        else:
            result.index = self.index.take(result.index)
        result = result.reindex(columns=self.columns, copy=False)
        return result

    def unstack(self, level=(- 1), fill_value=None):
        "\n        Pivot a level of the (necessarily hierarchical) index labels.\n\n        Returns a DataFrame having a new level of column labels whose inner-most level\n        consists of the pivoted index labels.\n\n        If the index is not a MultiIndex, the output will be a Series\n        (the analogue of stack when the columns are not a MultiIndex).\n\n        Parameters\n        ----------\n        level : int, str, or list of these, default -1 (last level)\n            Level(s) of index to unstack, can pass level name.\n        fill_value : int, str or dict\n            Replace NaN with this value if the unstack produces missing values.\n\n        Returns\n        -------\n        Series or DataFrame\n\n        See Also\n        --------\n        DataFrame.pivot : Pivot a table based on column values.\n        DataFrame.stack : Pivot a level of the column labels (inverse operation\n            from `unstack`).\n\n        Examples\n        --------\n        >>> index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),\n        ...                                    ('two', 'a'), ('two', 'b')])\n        >>> s = pd.Series(np.arange(1.0, 5.0), index=index)\n        >>> s\n        one  a   1.0\n             b   2.0\n        two  a   3.0\n             b   4.0\n        dtype: float64\n\n        >>> s.unstack(level=-1)\n             a   b\n        one  1.0  2.0\n        two  3.0  4.0\n\n        >>> s.unstack(level=0)\n           one  two\n        a  1.0   3.0\n        b  2.0   4.0\n\n        >>> df = s.unstack(level=0)\n        >>> df.unstack()\n        one  a  1.0\n             b  2.0\n        two  a  3.0\n             b  4.0\n        dtype: float64\n        "
        from pandas.core.reshape.reshape import unstack
        result = unstack(self, level, fill_value)
        return result.__finalize__(self, method='unstack')

    @Appender((_shared_docs['melt'] % {'caller': 'df.melt(', 'other': 'melt'}))
    def melt(self, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True):
        return melt(self, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name, col_level=col_level, ignore_index=ignore_index)

    @doc(Series.diff, klass='Dataframe', extra_params="axis : {0 or 'index', 1 or 'columns'}, default 0\n    Take difference over rows (0) or columns (1).\n", other_klass='Series', examples=dedent("\n        Difference with previous row\n\n        >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],\n        ...                    'b': [1, 1, 2, 3, 5, 8],\n        ...                    'c': [1, 4, 9, 16, 25, 36]})\n        >>> df\n           a  b   c\n        0  1  1   1\n        1  2  1   4\n        2  3  2   9\n        3  4  3  16\n        4  5  5  25\n        5  6  8  36\n\n        >>> df.diff()\n             a    b     c\n        0  NaN  NaN   NaN\n        1  1.0  0.0   3.0\n        2  1.0  1.0   5.0\n        3  1.0  1.0   7.0\n        4  1.0  2.0   9.0\n        5  1.0  3.0  11.0\n\n        Difference with previous column\n\n        >>> df.diff(axis=1)\n            a  b   c\n        0 NaN  0   0\n        1 NaN -1   3\n        2 NaN -1   7\n        3 NaN -1  13\n        4 NaN  0  20\n        5 NaN  2  28\n\n        Difference with 3rd previous row\n\n        >>> df.diff(periods=3)\n             a    b     c\n        0  NaN  NaN   NaN\n        1  NaN  NaN   NaN\n        2  NaN  NaN   NaN\n        3  3.0  2.0  15.0\n        4  3.0  4.0  21.0\n        5  3.0  6.0  27.0\n\n        Difference with following row\n\n        >>> df.diff(periods=-1)\n             a    b     c\n        0 -1.0  0.0  -3.0\n        1 -1.0 -1.0  -5.0\n        2 -1.0 -1.0  -7.0\n        3 -1.0 -2.0  -9.0\n        4 -1.0 -3.0 -11.0\n        5  NaN  NaN   NaN\n\n        Overflow in input dtype\n\n        >>> df = pd.DataFrame({'a': [1, 0]}, dtype=np.uint8)\n        >>> df.diff()\n               a\n        0    NaN\n        1  255.0"))
    def diff(self, periods=1, axis=0):
        if (not isinstance(periods, int)):
            if (not (is_float(periods) and periods.is_integer())):
                raise ValueError('periods must be an integer')
            periods = int(periods)
        bm_axis = self._get_block_manager_axis(axis)
        if ((bm_axis == 0) and (periods != 0)):
            return (self - self.shift(periods, axis=axis))
        new_data = self._mgr.diff(n=periods, axis=bm_axis)
        return self._constructor(new_data).__finalize__(self, 'diff')

    def _gotitem(self, key, ndim, subset=None):
        '\n        Sub-classes to define. Return a sliced object.\n\n        Parameters\n        ----------\n        key : string / list of selections\n        ndim : 1,2\n            requested ndim of result\n        subset : object, default None\n            subset to act on\n        '
        if (subset is None):
            subset = self
        elif (subset.ndim == 1):
            return subset
        return subset[key]
    _agg_summary_and_see_also_doc = dedent('\n    The aggregation operations are always performed over an axis, either the\n    index (default) or the column axis. This behavior is different from\n    `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,\n    `var`), where the default is to compute the aggregation of the flattened\n    array, e.g., ``numpy.mean(arr_2d)`` as opposed to\n    ``numpy.mean(arr_2d, axis=0)``.\n\n    `agg` is an alias for `aggregate`. Use the alias.\n\n    See Also\n    --------\n    DataFrame.apply : Perform any type of operations.\n    DataFrame.transform : Perform transformation type operations.\n    core.groupby.GroupBy : Perform operations over groups.\n    core.resample.Resampler : Perform operations over resampled bins.\n    core.window.Rolling : Perform operations over rolling window.\n    core.window.Expanding : Perform operations over expanding window.\n    core.window.ExponentialMovingWindow : Perform operation over exponential weighted\n        window.\n    ')
    _agg_examples_doc = dedent('\n    Examples\n    --------\n    >>> df = pd.DataFrame([[1, 2, 3],\n    ...                    [4, 5, 6],\n    ...                    [7, 8, 9],\n    ...                    [np.nan, np.nan, np.nan]],\n    ...                   columns=[\'A\', \'B\', \'C\'])\n\n    Aggregate these functions over the rows.\n\n    >>> df.agg([\'sum\', \'min\'])\n            A     B     C\n    sum  12.0  15.0  18.0\n    min   1.0   2.0   3.0\n\n    Different aggregations per column.\n\n    >>> df.agg({\'A\' : [\'sum\', \'min\'], \'B\' : [\'min\', \'max\']})\n            A    B\n    sum  12.0  NaN\n    min   1.0  2.0\n    max   NaN  8.0\n\n    Aggregate different functions over the columns and rename the index of the resulting\n    DataFrame.\n\n    >>> df.agg(x=(\'A\', max), y=(\'B\', \'min\'), z=(\'C\', np.mean))\n         A    B    C\n    x  7.0  NaN  NaN\n    y  NaN  2.0  NaN\n    z  NaN  NaN  6.0\n\n    Aggregate over the columns.\n\n    >>> df.agg("mean", axis="columns")\n    0    2.0\n    1    5.0\n    2    8.0\n    3    NaN\n    dtype: float64\n    ')

    @doc(_shared_docs['aggregate'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'], see_also=_agg_summary_and_see_also_doc, examples=_agg_examples_doc)
    def aggregate(self, func=None, axis=0, *args, **kwargs):
        axis = self._get_axis_number(axis)
        (relabeling, func, columns, order) = reconstruct_func(func, **kwargs)
        result = None
        try:
            (result, how) = self._aggregate(func, axis, *args, **kwargs)
        except TypeError as err:
            exc = TypeError(f'DataFrame constructor called with incompatible data and dtype: {err}')
            raise exc from err
        if (result is None):
            return self.apply(func, axis=axis, args=args, **kwargs)
        if relabeling:
            assert (columns is not None)
            assert (order is not None)
            result_in_dict = relabel_result(result, func, columns, order)
            result = DataFrame(result_in_dict, index=columns)
        return result

    def _aggregate(self, arg, axis=0, *args, **kwargs):
        if (axis == 1):
            (result, how) = aggregate(self.T, arg, *args, **kwargs)
            result = (result.T if (result is not None) else result)
            return (result, how)
        return aggregate(self, arg, *args, **kwargs)
    agg = aggregate

    @doc(_shared_docs['transform'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'])
    def transform(self, func, axis=0, *args, **kwargs):
        result = transform(self, func, axis, *args, **kwargs)
        assert isinstance(result, DataFrame)
        return result

    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds):
        "\n        Apply a function along an axis of the DataFrame.\n\n        Objects passed to the function are Series objects whose index is\n        either the DataFrame's index (``axis=0``) or the DataFrame's columns\n        (``axis=1``). By default (``result_type=None``), the final return type\n        is inferred from the return type of the applied function. Otherwise,\n        it depends on the `result_type` argument.\n\n        Parameters\n        ----------\n        func : function\n            Function to apply to each column or row.\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            Axis along which the function is applied:\n\n            * 0 or 'index': apply function to each column.\n            * 1 or 'columns': apply function to each row.\n\n        raw : bool, default False\n            Determines if row or column is passed as a Series or ndarray object:\n\n            * ``False`` : passes each row or column as a Series to the\n              function.\n            * ``True`` : the passed function will receive ndarray objects\n              instead.\n              If you are just applying a NumPy reduction function this will\n              achieve much better performance.\n\n        result_type : {'expand', 'reduce', 'broadcast', None}, default None\n            These only act when ``axis=1`` (columns):\n\n            * 'expand' : list-like results will be turned into columns.\n            * 'reduce' : returns a Series if possible rather than expanding\n              list-like results. This is the opposite of 'expand'.\n            * 'broadcast' : results will be broadcast to the original shape\n              of the DataFrame, the original index and columns will be\n              retained.\n\n            The default behaviour (None) depends on the return value of the\n            applied function: list-like results will be returned as a Series\n            of those. However if the apply function returns a Series these\n            are expanded to columns.\n        args : tuple\n            Positional arguments to pass to `func` in addition to the\n            array/series.\n        **kwds\n            Additional keyword arguments to pass as keywords arguments to\n            `func`.\n\n        Returns\n        -------\n        Series or DataFrame\n            Result of applying ``func`` along the given axis of the\n            DataFrame.\n\n        See Also\n        --------\n        DataFrame.applymap: For elementwise operations.\n        DataFrame.aggregate: Only perform aggregating type operations.\n        DataFrame.transform: Only perform transforming type operations.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])\n        >>> df\n           A  B\n        0  4  9\n        1  4  9\n        2  4  9\n\n        Using a numpy universal function (in this case the same as\n        ``np.sqrt(df)``):\n\n        >>> df.apply(np.sqrt)\n             A    B\n        0  2.0  3.0\n        1  2.0  3.0\n        2  2.0  3.0\n\n        Using a reducing function on either axis\n\n        >>> df.apply(np.sum, axis=0)\n        A    12\n        B    27\n        dtype: int64\n\n        >>> df.apply(np.sum, axis=1)\n        0    13\n        1    13\n        2    13\n        dtype: int64\n\n        Returning a list-like will result in a Series\n\n        >>> df.apply(lambda x: [1, 2], axis=1)\n        0    [1, 2]\n        1    [1, 2]\n        2    [1, 2]\n        dtype: object\n\n        Passing ``result_type='expand'`` will expand list-like results\n        to columns of a Dataframe\n\n        >>> df.apply(lambda x: [1, 2], axis=1, result_type='expand')\n           0  1\n        0  1  2\n        1  1  2\n        2  1  2\n\n        Returning a Series inside the function is similar to passing\n        ``result_type='expand'``. The resulting column names\n        will be the Series index.\n\n        >>> df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)\n           foo  bar\n        0    1    2\n        1    1    2\n        2    1    2\n\n        Passing ``result_type='broadcast'`` will ensure the same shape\n        result, whether list-like or scalar is returned by the function,\n        and broadcast it along the axis. The resulting column names will\n        be the originals.\n\n        >>> df.apply(lambda x: [1, 2], axis=1, result_type='broadcast')\n           A  B\n        0  1  2\n        1  1  2\n        2  1  2\n        "
        from pandas.core.apply import frame_apply
        op = frame_apply(self, func=func, axis=axis, raw=raw, result_type=result_type, args=args, kwds=kwds)
        return op.get_result()

    def applymap(self, func, na_action=None):
        "\n        Apply a function to a Dataframe elementwise.\n\n        This method applies a function that accepts and returns a scalar\n        to every element of a DataFrame.\n\n        Parameters\n        ----------\n        func : callable\n            Python function, returns a single value from a single value.\n        na_action : {None, 'ignore'}, default None\n            If ‘ignore’, propagate NaN values, without passing them to func.\n\n            .. versionadded:: 1.2\n\n        Returns\n        -------\n        DataFrame\n            Transformed DataFrame.\n\n        See Also\n        --------\n        DataFrame.apply : Apply a function along input axis of DataFrame.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])\n        >>> df\n               0      1\n        0  1.000  2.120\n        1  3.356  4.567\n\n        >>> df.applymap(lambda x: len(str(x)))\n           0  1\n        0  3  4\n        1  5  5\n\n        Like Series.map, NA values can be ignored:\n\n        >>> df_copy = df.copy()\n        >>> df_copy.iloc[0, 0] = pd.NA\n        >>> df_copy.applymap(lambda x: len(str(x)), na_action='ignore')\n              0  1\n        0  <NA>  4\n        1     5  5\n\n        Note that a vectorized version of `func` often exists, which will\n        be much faster. You could square each number elementwise.\n\n        >>> df.applymap(lambda x: x**2)\n                   0          1\n        0   1.000000   4.494400\n        1  11.262736  20.857489\n\n        But it's better to avoid applymap in that case.\n\n        >>> df ** 2\n                   0          1\n        0   1.000000   4.494400\n        1  11.262736  20.857489\n        "
        if (na_action not in {'ignore', None}):
            raise ValueError(f"na_action must be 'ignore' or None. Got {repr(na_action)}")
        ignore_na = (na_action == 'ignore')

        def infer(x):
            if x.empty:
                return lib.map_infer(x, func, ignore_na=ignore_na)
            return lib.map_infer(x.astype(object)._values, func, ignore_na=ignore_na)
        return self.apply(infer).__finalize__(self, 'applymap')

    def append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        "\n        Append rows of `other` to the end of caller, returning a new object.\n\n        Columns in `other` that are not in the caller are added as new columns.\n\n        Parameters\n        ----------\n        other : DataFrame or Series/dict-like object, or list of these\n            The data to append.\n        ignore_index : bool, default False\n            If True, the resulting axis will be labeled 0, 1, …, n - 1.\n        verify_integrity : bool, default False\n            If True, raise ValueError on creating index with duplicates.\n        sort : bool, default False\n            Sort columns if the columns of `self` and `other` are not aligned.\n\n            .. versionchanged:: 1.0.0\n\n                Changed to not sort by default.\n\n        Returns\n        -------\n        DataFrame\n\n        See Also\n        --------\n        concat : General function to concatenate DataFrame or Series objects.\n\n        Notes\n        -----\n        If a list of dict/series is passed and the keys are all contained in\n        the DataFrame's index, the order of the columns in the resulting\n        DataFrame will be unchanged.\n\n        Iteratively appending rows to a DataFrame can be more computationally\n        intensive than a single concatenate. A better solution is to append\n        those rows to a list and then concatenate the list with the original\n        DataFrame all at once.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))\n        >>> df\n           A  B\n        0  1  2\n        1  3  4\n        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))\n        >>> df.append(df2)\n           A  B\n        0  1  2\n        1  3  4\n        0  5  6\n        1  7  8\n\n        With `ignore_index` set to True:\n\n        >>> df.append(df2, ignore_index=True)\n           A  B\n        0  1  2\n        1  3  4\n        2  5  6\n        3  7  8\n\n        The following, while not recommended methods for generating DataFrames,\n        show two ways to generate a DataFrame from multiple data sources.\n\n        Less efficient:\n\n        >>> df = pd.DataFrame(columns=['A'])\n        >>> for i in range(5):\n        ...     df = df.append({'A': i}, ignore_index=True)\n        >>> df\n           A\n        0  0\n        1  1\n        2  2\n        3  3\n        4  4\n\n        More efficient:\n\n        >>> pd.concat([pd.DataFrame([i], columns=['A']) for i in range(5)],\n        ...           ignore_index=True)\n           A\n        0  0\n        1  1\n        2  2\n        3  3\n        4  4\n        "
        if isinstance(other, (Series, dict)):
            if isinstance(other, dict):
                if (not ignore_index):
                    raise TypeError('Can only append a dict if ignore_index=True')
                other = Series(other)
            if ((other.name is None) and (not ignore_index)):
                raise TypeError('Can only append a Series if ignore_index=True or if the Series has a name')
            index = Index([other.name], name=self.index.name)
            idx_diff = other.index.difference(self.columns)
            try:
                combined_columns = self.columns.append(idx_diff)
            except TypeError:
                combined_columns = self.columns.astype(object).append(idx_diff)
            other = other.reindex(combined_columns, copy=False).to_frame().T.infer_objects().rename_axis(index.names, copy=False)
            if (not self.columns.equals(combined_columns)):
                self = self.reindex(columns=combined_columns)
        elif isinstance(other, list):
            if (not other):
                pass
            elif (not isinstance(other[0], DataFrame)):
                other = DataFrame(other)
                if (self.columns.get_indexer(other.columns) >= 0).all():
                    other = other.reindex(columns=self.columns)
        from pandas.core.reshape.concat import concat
        if isinstance(other, (list, tuple)):
            to_concat = [self, *other]
        else:
            to_concat = [self, other]
        return concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort).__finalize__(self, method='append')

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        "\n        Join columns of another DataFrame.\n\n        Join columns with `other` DataFrame either on index or on a key\n        column. Efficiently join multiple DataFrame objects by index at once by\n        passing a list.\n\n        Parameters\n        ----------\n        other : DataFrame, Series, or list of DataFrame\n            Index should be similar to one of the columns in this one. If a\n            Series is passed, its name attribute must be set, and that will be\n            used as the column name in the resulting joined DataFrame.\n        on : str, list of str, or array-like, optional\n            Column or index level name(s) in the caller to join on the index\n            in `other`, otherwise joins index-on-index. If multiple\n            values given, the `other` DataFrame must have a MultiIndex. Can\n            pass an array as the join key if it is not already contained in\n            the calling DataFrame. Like an Excel VLOOKUP operation.\n        how : {'left', 'right', 'outer', 'inner'}, default 'left'\n            How to handle the operation of the two objects.\n\n            * left: use calling frame's index (or column if on is specified)\n            * right: use `other`'s index.\n            * outer: form union of calling frame's index (or column if on is\n              specified) with `other`'s index, and sort it.\n              lexicographically.\n            * inner: form intersection of calling frame's index (or column if\n              on is specified) with `other`'s index, preserving the order\n              of the calling's one.\n        lsuffix : str, default ''\n            Suffix to use from left frame's overlapping columns.\n        rsuffix : str, default ''\n            Suffix to use from right frame's overlapping columns.\n        sort : bool, default False\n            Order result DataFrame lexicographically by the join key. If False,\n            the order of the join key depends on the join type (how keyword).\n\n        Returns\n        -------\n        DataFrame\n            A dataframe containing columns from both the caller and `other`.\n\n        See Also\n        --------\n        DataFrame.merge : For column(s)-on-column(s) operations.\n\n        Notes\n        -----\n        Parameters `on`, `lsuffix`, and `rsuffix` are not supported when\n        passing a list of `DataFrame` objects.\n\n        Support for specifying index levels as the `on` parameter was added\n        in version 0.23.0.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],\n        ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})\n\n        >>> df\n          key   A\n        0  K0  A0\n        1  K1  A1\n        2  K2  A2\n        3  K3  A3\n        4  K4  A4\n        5  K5  A5\n\n        >>> other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],\n        ...                       'B': ['B0', 'B1', 'B2']})\n\n        >>> other\n          key   B\n        0  K0  B0\n        1  K1  B1\n        2  K2  B2\n\n        Join DataFrames using their indexes.\n\n        >>> df.join(other, lsuffix='_caller', rsuffix='_other')\n          key_caller   A key_other    B\n        0         K0  A0        K0   B0\n        1         K1  A1        K1   B1\n        2         K2  A2        K2   B2\n        3         K3  A3       NaN  NaN\n        4         K4  A4       NaN  NaN\n        5         K5  A5       NaN  NaN\n\n        If we want to join using the key columns, we need to set key to be\n        the index in both `df` and `other`. The joined DataFrame will have\n        key as its index.\n\n        >>> df.set_index('key').join(other.set_index('key'))\n              A    B\n        key\n        K0   A0   B0\n        K1   A1   B1\n        K2   A2   B2\n        K3   A3  NaN\n        K4   A4  NaN\n        K5   A5  NaN\n\n        Another option to join using the key columns is to use the `on`\n        parameter. DataFrame.join always uses `other`'s index but we can use\n        any column in `df`. This method preserves the original DataFrame's\n        index in the result.\n\n        >>> df.join(other.set_index('key'), on='key')\n          key   A    B\n        0  K0  A0   B0\n        1  K1  A1   B1\n        2  K2  A2   B2\n        3  K3  A3  NaN\n        4  K4  A4  NaN\n        5  K5  A5  NaN\n        "
        return self._join_compat(other, on=on, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort)

    def _join_compat(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        from pandas.core.reshape.concat import concat
        from pandas.core.reshape.merge import merge
        if isinstance(other, Series):
            if (other.name is None):
                raise ValueError('Other Series must have a name')
            other = DataFrame({other.name: other})
        if isinstance(other, DataFrame):
            if (how == 'cross'):
                return merge(self, other, how=how, on=on, suffixes=(lsuffix, rsuffix), sort=sort)
            return merge(self, other, left_on=on, how=how, left_index=(on is None), right_index=True, suffixes=(lsuffix, rsuffix), sort=sort)
        else:
            if (on is not None):
                raise ValueError('Joining multiple DataFrames only supported for joining on index')
            frames = ([self] + list(other))
            can_concat = all((df.index.is_unique for df in frames))
            if can_concat:
                if (how == 'left'):
                    res = concat(frames, axis=1, join='outer', verify_integrity=True, sort=sort)
                    return res.reindex(self.index, copy=False)
                else:
                    return concat(frames, axis=1, join=how, verify_integrity=True, sort=sort)
            joined = frames[0]
            for frame in frames[1:]:
                joined = merge(joined, frame, how=how, left_index=True, right_index=True)
            return joined

    @Substitution('')
    @Appender(_merge_doc, indents=2)
    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        from pandas.core.reshape.merge import merge
        return merge(self, right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)

    def round(self, decimals=0, *args, **kwargs):
        "\n        Round a DataFrame to a variable number of decimal places.\n\n        Parameters\n        ----------\n        decimals : int, dict, Series\n            Number of decimal places to round each column to. If an int is\n            given, round each column to the same number of places.\n            Otherwise dict and Series round to variable numbers of places.\n            Column names should be in the keys if `decimals` is a\n            dict-like, or in the index if `decimals` is a Series. Any\n            columns not included in `decimals` will be left as is. Elements\n            of `decimals` which are not columns of the input will be\n            ignored.\n        *args\n            Additional keywords have no effect but might be accepted for\n            compatibility with numpy.\n        **kwargs\n            Additional keywords have no effect but might be accepted for\n            compatibility with numpy.\n\n        Returns\n        -------\n        DataFrame\n            A DataFrame with the affected columns rounded to the specified\n            number of decimal places.\n\n        See Also\n        --------\n        numpy.around : Round a numpy array to the given number of decimals.\n        Series.round : Round a Series to the given number of decimals.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([(.21, .32), (.01, .67), (.66, .03), (.21, .18)],\n        ...                   columns=['dogs', 'cats'])\n        >>> df\n            dogs  cats\n        0  0.21  0.32\n        1  0.01  0.67\n        2  0.66  0.03\n        3  0.21  0.18\n\n        By providing an integer each column is rounded to the same number\n        of decimal places\n\n        >>> df.round(1)\n            dogs  cats\n        0   0.2   0.3\n        1   0.0   0.7\n        2   0.7   0.0\n        3   0.2   0.2\n\n        With a dict, the number of places for specific columns can be\n        specified with the column names as key and the number of decimal\n        places as value\n\n        >>> df.round({'dogs': 1, 'cats': 0})\n            dogs  cats\n        0   0.2   0.0\n        1   0.0   1.0\n        2   0.7   0.0\n        3   0.2   0.0\n\n        Using a Series, the number of places for specific columns can be\n        specified with the column names as index and the number of\n        decimal places as value\n\n        >>> decimals = pd.Series([0, 1], index=['cats', 'dogs'])\n        >>> df.round(decimals)\n            dogs  cats\n        0   0.2   0.0\n        1   0.0   1.0\n        2   0.7   0.0\n        3   0.2   0.0\n        "
        from pandas.core.reshape.concat import concat

        def _dict_round(df, decimals):
            for (col, vals) in df.items():
                try:
                    (yield _series_round(vals, decimals[col]))
                except KeyError:
                    (yield vals)

        def _series_round(s, decimals):
            if (is_integer_dtype(s) or is_float_dtype(s)):
                return s.round(decimals)
            return s
        nv.validate_round(args, kwargs)
        if isinstance(decimals, (dict, Series)):
            if isinstance(decimals, Series):
                if (not decimals.index.is_unique):
                    raise ValueError('Index of decimals must be unique')
            new_cols = list(_dict_round(self, decimals))
        elif is_integer(decimals):
            new_cols = [_series_round(v, decimals) for (_, v) in self.items()]
        else:
            raise TypeError('decimals must be an integer, a dict-like or a Series')
        if (len(new_cols) > 0):
            return self._constructor(concat(new_cols, axis=1), index=self.index, columns=self.columns)
        else:
            return self

    def corr(self, method='pearson', min_periods=1):
        "\n        Compute pairwise correlation of columns, excluding NA/null values.\n\n        Parameters\n        ----------\n        method : {'pearson', 'kendall', 'spearman'} or callable\n            Method of correlation:\n\n            * pearson : standard correlation coefficient\n            * kendall : Kendall Tau correlation coefficient\n            * spearman : Spearman rank correlation\n            * callable: callable with input two 1d ndarrays\n                and returning a float. Note that the returned matrix from corr\n                will have 1 along the diagonals and will be symmetric\n                regardless of the callable's behavior.\n\n                .. versionadded:: 0.24.0\n\n        min_periods : int, optional\n            Minimum number of observations required per pair of columns\n            to have a valid result. Currently only available for Pearson\n            and Spearman correlation.\n\n        Returns\n        -------\n        DataFrame\n            Correlation matrix.\n\n        See Also\n        --------\n        DataFrame.corrwith : Compute pairwise correlation with another\n            DataFrame or Series.\n        Series.corr : Compute the correlation between two Series.\n\n        Examples\n        --------\n        >>> def histogram_intersection(a, b):\n        ...     v = np.minimum(a, b).sum().round(decimals=1)\n        ...     return v\n        >>> df = pd.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],\n        ...                   columns=['dogs', 'cats'])\n        >>> df.corr(method=histogram_intersection)\n              dogs  cats\n        dogs   1.0   0.3\n        cats   0.3   1.0\n        "
        numeric_df = self._get_numeric_data()
        cols = numeric_df.columns
        idx = cols.copy()
        mat = numeric_df.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if (method == 'pearson'):
            correl = libalgos.nancorr(mat, minp=min_periods)
        elif (method == 'spearman'):
            correl = libalgos.nancorr_spearman(mat, minp=min_periods)
        elif ((method == 'kendall') or callable(method)):
            if (min_periods is None):
                min_periods = 1
            mat = mat.T
            corrf = nanops.get_corr_func(method)
            K = len(cols)
            correl = np.empty((K, K), dtype=float)
            mask = np.isfinite(mat)
            for (i, ac) in enumerate(mat):
                for (j, bc) in enumerate(mat):
                    if (i > j):
                        continue
                    valid = (mask[i] & mask[j])
                    if (valid.sum() < min_periods):
                        c = np.nan
                    elif (i == j):
                        c = 1.0
                    elif (not valid.all()):
                        c = corrf(ac[valid], bc[valid])
                    else:
                        c = corrf(ac, bc)
                    correl[(i, j)] = c
                    correl[(j, i)] = c
        else:
            raise ValueError(f"method must be either 'pearson', 'spearman', 'kendall', or a callable, '{method}' was supplied")
        return self._constructor(correl, index=idx, columns=cols)

    def cov(self, min_periods=None, ddof=1):
        "\n        Compute pairwise covariance of columns, excluding NA/null values.\n\n        Compute the pairwise covariance among the series of a DataFrame.\n        The returned data frame is the `covariance matrix\n        <https://en.wikipedia.org/wiki/Covariance_matrix>`__ of the columns\n        of the DataFrame.\n\n        Both NA and null values are automatically excluded from the\n        calculation. (See the note below about bias from missing values.)\n        A threshold can be set for the minimum number of\n        observations for each value created. Comparisons with observations\n        below this threshold will be returned as ``NaN``.\n\n        This method is generally used for the analysis of time series data to\n        understand the relationship between different measures\n        across time.\n\n        Parameters\n        ----------\n        min_periods : int, optional\n            Minimum number of observations required per pair of columns\n            to have a valid result.\n\n        ddof : int, default 1\n            Delta degrees of freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        DataFrame\n            The covariance matrix of the series of the DataFrame.\n\n        See Also\n        --------\n        Series.cov : Compute covariance with another Series.\n        core.window.ExponentialMovingWindow.cov: Exponential weighted sample covariance.\n        core.window.Expanding.cov : Expanding sample covariance.\n        core.window.Rolling.cov : Rolling sample covariance.\n\n        Notes\n        -----\n        Returns the covariance matrix of the DataFrame's time series.\n        The covariance is normalized by N-ddof.\n\n        For DataFrames that have Series that are missing data (assuming that\n        data is `missing at random\n        <https://en.wikipedia.org/wiki/Missing_data#Missing_at_random>`__)\n        the returned covariance matrix will be an unbiased estimate\n        of the variance and covariance between the member Series.\n\n        However, for many applications this estimate may not be acceptable\n        because the estimate covariance matrix is not guaranteed to be positive\n        semi-definite. This could lead to estimate correlations having\n        absolute values which are greater than one, and/or a non-invertible\n        covariance matrix. See `Estimation of covariance matrices\n        <https://en.wikipedia.org/w/index.php?title=Estimation_of_covariance_\n        matrices>`__ for more details.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([(1, 2), (0, 3), (2, 0), (1, 1)],\n        ...                   columns=['dogs', 'cats'])\n        >>> df.cov()\n                  dogs      cats\n        dogs  0.666667 -1.000000\n        cats -1.000000  1.666667\n\n        >>> np.random.seed(42)\n        >>> df = pd.DataFrame(np.random.randn(1000, 5),\n        ...                   columns=['a', 'b', 'c', 'd', 'e'])\n        >>> df.cov()\n                  a         b         c         d         e\n        a  0.998438 -0.020161  0.059277 -0.008943  0.014144\n        b -0.020161  1.059352 -0.008543 -0.024738  0.009826\n        c  0.059277 -0.008543  1.010670 -0.001486 -0.000271\n        d -0.008943 -0.024738 -0.001486  0.921297 -0.013692\n        e  0.014144  0.009826 -0.000271 -0.013692  0.977795\n\n        **Minimum number of periods**\n\n        This method also supports an optional ``min_periods`` keyword\n        that specifies the required minimum number of non-NA observations for\n        each column pair in order to have a valid result:\n\n        >>> np.random.seed(42)\n        >>> df = pd.DataFrame(np.random.randn(20, 3),\n        ...                   columns=['a', 'b', 'c'])\n        >>> df.loc[df.index[:5], 'a'] = np.nan\n        >>> df.loc[df.index[5:10], 'b'] = np.nan\n        >>> df.cov(min_periods=12)\n                  a         b         c\n        a  0.316741       NaN -0.150812\n        b       NaN  1.248003  0.191417\n        c -0.150812  0.191417  0.895202\n        "
        numeric_df = self._get_numeric_data()
        cols = numeric_df.columns
        idx = cols.copy()
        mat = numeric_df.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if notna(mat).all():
            if ((min_periods is not None) and (min_periods > len(mat))):
                base_cov = np.empty((mat.shape[1], mat.shape[1]))
                base_cov.fill(np.nan)
            else:
                base_cov = np.cov(mat.T, ddof=ddof)
            base_cov = base_cov.reshape((len(cols), len(cols)))
        else:
            base_cov = libalgos.nancorr(mat, cov=True, minp=min_periods)
        return self._constructor(base_cov, index=idx, columns=cols)

    def corrwith(self, other, axis=0, drop=False, method='pearson'):
        "\n        Compute pairwise correlation.\n\n        Pairwise correlation is computed between rows or columns of\n        DataFrame with rows or columns of Series or DataFrame. DataFrames\n        are first aligned along both axes before computing the\n        correlations.\n\n        Parameters\n        ----------\n        other : DataFrame, Series\n            Object with which to compute correlations.\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            The axis to use. 0 or 'index' to compute column-wise, 1 or 'columns' for\n            row-wise.\n        drop : bool, default False\n            Drop missing indices from result.\n        method : {'pearson', 'kendall', 'spearman'} or callable\n            Method of correlation:\n\n            * pearson : standard correlation coefficient\n            * kendall : Kendall Tau correlation coefficient\n            * spearman : Spearman rank correlation\n            * callable: callable with input two 1d ndarrays\n                and returning a float.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        Series\n            Pairwise correlations.\n\n        See Also\n        --------\n        DataFrame.corr : Compute pairwise correlation of columns.\n        "
        axis = self._get_axis_number(axis)
        this = self._get_numeric_data()
        if isinstance(other, Series):
            return this.apply((lambda x: other.corr(x, method=method)), axis=axis)
        other = other._get_numeric_data()
        (left, right) = this.align(other, join='inner', copy=False)
        if (axis == 1):
            left = left.T
            right = right.T
        if (method == 'pearson'):
            left = (left + (right * 0))
            right = (right + (left * 0))
            ldem = (left - left.mean())
            rdem = (right - right.mean())
            num = (ldem * rdem).sum()
            dom = (((left.count() - 1) * left.std()) * right.std())
            correl = (num / dom)
        elif ((method in ['kendall', 'spearman']) or callable(method)):

            def c(x):
                return nanops.nancorr(x[0], x[1], method=method)
            correl = self._constructor_sliced(map(c, zip(left.values.T, right.values.T)), index=left.columns)
        else:
            raise ValueError(f"Invalid method {method} was passed, valid methods are: 'pearson', 'kendall', 'spearman', or callable")
        if (not drop):
            raxis = (1 if (axis == 0) else 0)
            result_index = this._get_axis(raxis).union(other._get_axis(raxis))
            idx_diff = result_index.difference(correl.index)
            if (len(idx_diff) > 0):
                correl = correl.append(Series(([np.nan] * len(idx_diff)), index=idx_diff))
        return correl

    def count(self, axis=0, level=None, numeric_only=False):
        '\n        Count non-NA cells for each column or row.\n\n        The values `None`, `NaN`, `NaT`, and optionally `numpy.inf` (depending\n        on `pandas.options.mode.use_inf_as_na`) are considered NA.\n\n        Parameters\n        ----------\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            If 0 or \'index\' counts are generated for each column.\n            If 1 or \'columns\' counts are generated for each row.\n        level : int or str, optional\n            If the axis is a `MultiIndex` (hierarchical), count along a\n            particular `level`, collapsing into a `DataFrame`.\n            A `str` specifies the level name.\n        numeric_only : bool, default False\n            Include only `float`, `int` or `boolean` data.\n\n        Returns\n        -------\n        Series or DataFrame\n            For each column/row the number of non-NA/null entries.\n            If `level` is specified returns a `DataFrame`.\n\n        See Also\n        --------\n        Series.count: Number of non-NA elements in a Series.\n        DataFrame.value_counts: Count unique combinations of columns.\n        DataFrame.shape: Number of DataFrame rows and columns (including NA\n            elements).\n        DataFrame.isna: Boolean same-sized DataFrame showing places of NA\n            elements.\n\n        Examples\n        --------\n        Constructing DataFrame from a dictionary:\n\n        >>> df = pd.DataFrame({"Person":\n        ...                    ["John", "Myla", "Lewis", "John", "Myla"],\n        ...                    "Age": [24., np.nan, 21., 33, 26],\n        ...                    "Single": [False, True, True, True, False]})\n        >>> df\n           Person   Age  Single\n        0    John  24.0   False\n        1    Myla   NaN    True\n        2   Lewis  21.0    True\n        3    John  33.0    True\n        4    Myla  26.0   False\n\n        Notice the uncounted NA values:\n\n        >>> df.count()\n        Person    5\n        Age       4\n        Single    5\n        dtype: int64\n\n        Counts for each **row**:\n\n        >>> df.count(axis=\'columns\')\n        0    3\n        1    2\n        2    3\n        3    3\n        4    3\n        dtype: int64\n\n        Counts for one level of a `MultiIndex`:\n\n        >>> df.set_index(["Person", "Single"]).count(level="Person")\n                Age\n        Person\n        John      2\n        Lewis     1\n        Myla      1\n        '
        axis = self._get_axis_number(axis)
        if (level is not None):
            return self._count_level(level, axis=axis, numeric_only=numeric_only)
        if numeric_only:
            frame = self._get_numeric_data()
        else:
            frame = self
        if (len(frame._get_axis(axis)) == 0):
            result = self._constructor_sliced(0, index=frame._get_agg_axis(axis))
        elif (frame._is_mixed_type or frame._mgr.any_extension_types):
            result = notna(frame).sum(axis=axis)
        else:
            series_counts = notna(frame).sum(axis=axis)
            counts = series_counts.values
            result = self._constructor_sliced(counts, index=frame._get_agg_axis(axis))
        return result.astype('int64')

    def _count_level(self, level, axis=0, numeric_only=False):
        if numeric_only:
            frame = self._get_numeric_data()
        else:
            frame = self
        count_axis = frame._get_axis(axis)
        agg_axis = frame._get_agg_axis(axis)
        if (not isinstance(count_axis, MultiIndex)):
            raise TypeError(f'Can only count levels on hierarchical {self._get_axis_name(axis)}.')
        if frame._is_mixed_type:
            values_mask = notna(frame).values
        else:
            values_mask = notna(frame.values)
        index_mask = notna(count_axis.get_level_values(level=level))
        if (axis == 1):
            mask = (index_mask & values_mask)
        else:
            mask = (index_mask.reshape((- 1), 1) & values_mask)
        if isinstance(level, str):
            level = count_axis._get_level_number(level)
        level_name = count_axis._names[level]
        level_index = count_axis.levels[level]._shallow_copy(name=level_name)
        level_codes = ensure_int64(count_axis.codes[level])
        counts = lib.count_level_2d(mask, level_codes, len(level_index), axis=axis)
        if (axis == 1):
            result = self._constructor(counts, index=agg_axis, columns=level_index)
        else:
            result = self._constructor(counts, index=level_index, columns=agg_axis)
        return result

    def _reduce(self, op, name, *, axis=0, skipna=True, numeric_only=None, filter_type=None, **kwds):
        assert ((filter_type is None) or (filter_type == 'bool')), filter_type
        out_dtype = ('bool' if (filter_type == 'bool') else None)
        own_dtypes = [arr.dtype for arr in self._iter_column_arrays()]
        dtype_is_dt = np.array([is_datetime64_any_dtype(dtype) for dtype in own_dtypes], dtype=bool)
        if ((numeric_only is None) and (name in ['mean', 'median']) and dtype_is_dt.any()):
            warnings.warn('DataFrame.mean and DataFrame.median with numeric_only=None will include datetime64 and datetime64tz columns in a future version.', FutureWarning, stacklevel=5)
            cols = self.columns[(~ dtype_is_dt)]
            self = self[cols]
        axis = self._get_axis_number(axis)
        labels = self._get_agg_axis(axis)
        assert (axis in [0, 1])

        def func(values: np.ndarray):
            return op(values, axis=axis, skipna=skipna, **kwds)

        def blk_func(values):
            if isinstance(values, ExtensionArray):
                return values._reduce(name, skipna=skipna, **kwds)
            else:
                return op(values, axis=1, skipna=skipna, **kwds)

        def _get_data() -> DataFrame:
            if (filter_type is None):
                data = self._get_numeric_data()
            else:
                assert (filter_type == 'bool')
                data = self._get_bool_data()
            return data
        if ((numeric_only is not None) or (axis == 0)):
            df = self
            if (numeric_only is True):
                df = _get_data()
            if (axis == 1):
                df = df.T
                axis = 0
            ignore_failures = (numeric_only is None)
            (res, indexer) = df._mgr.reduce(blk_func, ignore_failures=ignore_failures)
            out = df._constructor(res).iloc[0]
            if (out_dtype is not None):
                out = out.astype(out_dtype)
            if ((axis == 0) and (len(self) == 0) and (name in ['sum', 'prod'])):
                out = out.astype(np.float64)
            return out
        assert (numeric_only is None)
        data = self
        values = data.values
        try:
            result = func(values)
        except TypeError:
            data = _get_data()
            labels = data._get_agg_axis(axis)
            values = data.values
            with np.errstate(all='ignore'):
                result = func(values)
        if ((filter_type == 'bool') and notna(result).all()):
            result = result.astype(np.bool_)
        elif ((filter_type is None) and is_object_dtype(result.dtype)):
            try:
                result = result.astype(np.float64)
            except (ValueError, TypeError):
                pass
        result = self._constructor_sliced(result, index=labels)
        return result

    def nunique(self, axis=0, dropna=True):
        "\n        Count distinct observations over requested axis.\n\n        Return Series with number of distinct observations. Can ignore NaN\n        values.\n\n        Parameters\n        ----------\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for\n            column-wise.\n        dropna : bool, default True\n            Don't include NaN in the counts.\n\n        Returns\n        -------\n        Series\n\n        See Also\n        --------\n        Series.nunique: Method nunique for Series.\n        DataFrame.count: Count non-NA cells for each column or row.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})\n        >>> df.nunique()\n        A    3\n        B    1\n        dtype: int64\n\n        >>> df.nunique(axis=1)\n        0    1\n        1    2\n        2    2\n        dtype: int64\n        "
        return self.apply(Series.nunique, axis=axis, dropna=dropna)

    def idxmin(self, axis=0, skipna=True):
        '\n        Return index of first occurrence of minimum over requested axis.\n\n        NA/null values are excluded.\n\n        Parameters\n        ----------\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            The axis to use. 0 or \'index\' for row-wise, 1 or \'columns\' for column-wise.\n        skipna : bool, default True\n            Exclude NA/null values. If an entire row/column is NA, the result\n            will be NA.\n\n        Returns\n        -------\n        Series\n            Indexes of minima along the specified axis.\n\n        Raises\n        ------\n        ValueError\n            * If the row/column is empty\n\n        See Also\n        --------\n        Series.idxmin : Return index of the minimum element.\n\n        Notes\n        -----\n        This method is the DataFrame version of ``ndarray.argmin``.\n\n        Examples\n        --------\n        Consider a dataset containing food consumption in Argentina.\n\n        >>> df = pd.DataFrame({\'consumption\': [10.51, 103.11, 55.48],\n        ...                    \'co2_emissions\': [37.2, 19.66, 1712]},\n        ...                    index=[\'Pork\', \'Wheat Products\', \'Beef\'])\n\n        >>> df\n                        consumption  co2_emissions\n        Pork                  10.51         37.20\n        Wheat Products       103.11         19.66\n        Beef                  55.48       1712.00\n\n        By default, it returns the index for the minimum value in each column.\n\n        >>> df.idxmin()\n        consumption                Pork\n        co2_emissions    Wheat Products\n        dtype: object\n\n        To return the index for the minimum value in each row, use ``axis="columns"``.\n\n        >>> df.idxmin(axis="columns")\n        Pork                consumption\n        Wheat Products    co2_emissions\n        Beef                consumption\n        dtype: object\n        '
        axis = self._get_axis_number(axis)
        res = self._reduce(nanops.nanargmin, 'argmin', axis=axis, skipna=skipna, numeric_only=False)
        indices = res._values
        assert isinstance(indices, np.ndarray)
        index = self._get_axis(axis)
        result = [(index[i] if (i >= 0) else np.nan) for i in indices]
        return self._constructor_sliced(result, index=self._get_agg_axis(axis))

    def idxmax(self, axis=0, skipna=True):
        '\n        Return index of first occurrence of maximum over requested axis.\n\n        NA/null values are excluded.\n\n        Parameters\n        ----------\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            The axis to use. 0 or \'index\' for row-wise, 1 or \'columns\' for column-wise.\n        skipna : bool, default True\n            Exclude NA/null values. If an entire row/column is NA, the result\n            will be NA.\n\n        Returns\n        -------\n        Series\n            Indexes of maxima along the specified axis.\n\n        Raises\n        ------\n        ValueError\n            * If the row/column is empty\n\n        See Also\n        --------\n        Series.idxmax : Return index of the maximum element.\n\n        Notes\n        -----\n        This method is the DataFrame version of ``ndarray.argmax``.\n\n        Examples\n        --------\n        Consider a dataset containing food consumption in Argentina.\n\n        >>> df = pd.DataFrame({\'consumption\': [10.51, 103.11, 55.48],\n        ...                    \'co2_emissions\': [37.2, 19.66, 1712]},\n        ...                    index=[\'Pork\', \'Wheat Products\', \'Beef\'])\n\n        >>> df\n                        consumption  co2_emissions\n        Pork                  10.51         37.20\n        Wheat Products       103.11         19.66\n        Beef                  55.48       1712.00\n\n        By default, it returns the index for the maximum value in each column.\n\n        >>> df.idxmax()\n        consumption     Wheat Products\n        co2_emissions             Beef\n        dtype: object\n\n        To return the index for the maximum value in each row, use ``axis="columns"``.\n\n        >>> df.idxmax(axis="columns")\n        Pork              co2_emissions\n        Wheat Products     consumption\n        Beef              co2_emissions\n        dtype: object\n        '
        axis = self._get_axis_number(axis)
        res = self._reduce(nanops.nanargmax, 'argmax', axis=axis, skipna=skipna, numeric_only=False)
        indices = res._values
        assert isinstance(indices, np.ndarray)
        index = self._get_axis(axis)
        result = [(index[i] if (i >= 0) else np.nan) for i in indices]
        return self._constructor_sliced(result, index=self._get_agg_axis(axis))

    def _get_agg_axis(self, axis_num):
        "\n        Let's be explicit about this.\n        "
        if (axis_num == 0):
            return self.columns
        elif (axis_num == 1):
            return self.index
        else:
            raise ValueError(f'Axis must be 0 or 1 (got {repr(axis_num)})')

    def mode(self, axis=0, numeric_only=False, dropna=True):
        "\n        Get the mode(s) of each element along the selected axis.\n\n        The mode of a set of values is the value that appears most often.\n        It can be multiple values.\n\n        Parameters\n        ----------\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            The axis to iterate over while searching for the mode:\n\n            * 0 or 'index' : get mode of each column\n            * 1 or 'columns' : get mode of each row.\n\n        numeric_only : bool, default False\n            If True, only apply to numeric columns.\n        dropna : bool, default True\n            Don't consider counts of NaN/NaT.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        DataFrame\n            The modes of each column or row.\n\n        See Also\n        --------\n        Series.mode : Return the highest frequency value in a Series.\n        Series.value_counts : Return the counts of values in a Series.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame([('bird', 2, 2),\n        ...                    ('mammal', 4, np.nan),\n        ...                    ('arthropod', 8, 0),\n        ...                    ('bird', 2, np.nan)],\n        ...                   index=('falcon', 'horse', 'spider', 'ostrich'),\n        ...                   columns=('species', 'legs', 'wings'))\n        >>> df\n                   species  legs  wings\n        falcon        bird     2    2.0\n        horse       mammal     4    NaN\n        spider   arthropod     8    0.0\n        ostrich       bird     2    NaN\n\n        By default, missing values are not considered, and the mode of wings\n        are both 0 and 2. Because the resulting DataFrame has two rows,\n        the second row of ``species`` and ``legs`` contains ``NaN``.\n\n        >>> df.mode()\n          species  legs  wings\n        0    bird   2.0    0.0\n        1     NaN   NaN    2.0\n\n        Setting ``dropna=False`` ``NaN`` values are considered and they can be\n        the mode (like for wings).\n\n        >>> df.mode(dropna=False)\n          species  legs  wings\n        0    bird     2    NaN\n\n        Setting ``numeric_only=True``, only the mode of numeric columns is\n        computed, and columns of other types are ignored.\n\n        >>> df.mode(numeric_only=True)\n           legs  wings\n        0   2.0    0.0\n        1   NaN    2.0\n\n        To compute the mode over columns and not rows, use the axis parameter:\n\n        >>> df.mode(axis='columns', numeric_only=True)\n                   0    1\n        falcon   2.0  NaN\n        horse    4.0  NaN\n        spider   0.0  8.0\n        ostrich  2.0  NaN\n        "
        data = (self if (not numeric_only) else self._get_numeric_data())

        def f(s):
            return s.mode(dropna=dropna)
        data = data.apply(f, axis=axis)
        if data.empty:
            data.index = ibase.default_index(0)
        return data

    def quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear'):
        "\n        Return values at the given quantile over requested axis.\n\n        Parameters\n        ----------\n        q : float or array-like, default 0.5 (50% quantile)\n            Value between 0 <= q <= 1, the quantile(s) to compute.\n        axis : {0, 1, 'index', 'columns'}, default 0\n            Equals 0 or 'index' for row-wise, 1 or 'columns' for column-wise.\n        numeric_only : bool, default True\n            If False, the quantile of datetime and timedelta data will be\n            computed as well.\n        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}\n            This optional parameter specifies the interpolation method to use,\n            when the desired quantile lies between two data points `i` and `j`:\n\n            * linear: `i + (j - i) * fraction`, where `fraction` is the\n              fractional part of the index surrounded by `i` and `j`.\n            * lower: `i`.\n            * higher: `j`.\n            * nearest: `i` or `j` whichever is nearest.\n            * midpoint: (`i` + `j`) / 2.\n\n        Returns\n        -------\n        Series or DataFrame\n\n            If ``q`` is an array, a DataFrame will be returned where the\n              index is ``q``, the columns are the columns of self, and the\n              values are the quantiles.\n            If ``q`` is a float, a Series will be returned where the\n              index is the columns of self and the values are the quantiles.\n\n        See Also\n        --------\n        core.window.Rolling.quantile: Rolling quantile.\n        numpy.percentile: Numpy function to compute the percentile.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),\n        ...                   columns=['a', 'b'])\n        >>> df.quantile(.1)\n        a    1.3\n        b    3.7\n        Name: 0.1, dtype: float64\n        >>> df.quantile([.1, .5])\n               a     b\n        0.1  1.3   3.7\n        0.5  2.5  55.0\n\n        Specifying `numeric_only=False` will also compute the quantile of\n        datetime and timedelta data.\n\n        >>> df = pd.DataFrame({'A': [1, 2],\n        ...                    'B': [pd.Timestamp('2010'),\n        ...                          pd.Timestamp('2011')],\n        ...                    'C': [pd.Timedelta('1 days'),\n        ...                          pd.Timedelta('2 days')]})\n        >>> df.quantile(0.5, numeric_only=False)\n        A                    1.5\n        B    2010-07-02 12:00:00\n        C        1 days 12:00:00\n        Name: 0.5, dtype: object\n        "
        validate_percentile(q)
        data = (self._get_numeric_data() if numeric_only else self)
        axis = self._get_axis_number(axis)
        is_transposed = (axis == 1)
        if is_transposed:
            data = data.T
        if (len(data.columns) == 0):
            cols = Index([], name=self.columns.name)
            if is_list_like(q):
                return self._constructor([], index=q, columns=cols)
            return self._constructor_sliced([], index=cols, name=q, dtype=np.float64)
        result = data._mgr.quantile(qs=q, axis=1, interpolation=interpolation, transposed=is_transposed)
        if (result.ndim == 2):
            result = self._constructor(result)
        else:
            result = self._constructor_sliced(result, name=q)
        if is_transposed:
            result = result.T
        return result

    @doc(NDFrame.asfreq, **_shared_doc_kwargs)
    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        return super().asfreq(freq=freq, method=method, how=how, normalize=normalize, fill_value=fill_value)

    @doc(NDFrame.resample, **_shared_doc_kwargs)
    def resample(self, rule, axis=0, closed=None, label=None, convention='start', kind=None, loffset=None, base=None, on=None, level=None, origin='start_day', offset=None):
        return super().resample(rule=rule, axis=axis, closed=closed, label=label, convention=convention, kind=kind, loffset=loffset, base=base, on=on, level=level, origin=origin, offset=offset)

    def to_timestamp(self, freq=None, how='start', axis=0, copy=True):
        "\n        Cast to DatetimeIndex of timestamps, at *beginning* of period.\n\n        Parameters\n        ----------\n        freq : str, default frequency of PeriodIndex\n            Desired frequency.\n        how : {'s', 'e', 'start', 'end'}\n            Convention for converting period to timestamp; start of period\n            vs. end.\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            The axis to convert (the index by default).\n        copy : bool, default True\n            If False then underlying input data is not copied.\n\n        Returns\n        -------\n        DataFrame with DatetimeIndex\n        "
        new_obj = self.copy(deep=copy)
        axis_name = self._get_axis_name(axis)
        old_ax = getattr(self, axis_name)
        if (not isinstance(old_ax, PeriodIndex)):
            raise TypeError(f'unsupported Type {type(old_ax).__name__}')
        new_ax = old_ax.to_timestamp(freq=freq, how=how)
        setattr(new_obj, axis_name, new_ax)
        return new_obj

    def to_period(self, freq=None, axis=0, copy=True):
        "\n        Convert DataFrame from DatetimeIndex to PeriodIndex.\n\n        Convert DataFrame from DatetimeIndex to PeriodIndex with desired\n        frequency (inferred from index if not passed).\n\n        Parameters\n        ----------\n        freq : str, default\n            Frequency of the PeriodIndex.\n        axis : {0 or 'index', 1 or 'columns'}, default 0\n            The axis to convert (the index by default).\n        copy : bool, default True\n            If False then underlying input data is not copied.\n\n        Returns\n        -------\n        DataFrame with PeriodIndex\n        "
        new_obj = self.copy(deep=copy)
        axis_name = self._get_axis_name(axis)
        old_ax = getattr(self, axis_name)
        if (not isinstance(old_ax, DatetimeIndex)):
            raise TypeError(f'unsupported Type {type(old_ax).__name__}')
        new_ax = old_ax.to_period(freq=freq)
        setattr(new_obj, axis_name, new_ax)
        return new_obj

    def isin(self, values):
        "\n        Whether each element in the DataFrame is contained in values.\n\n        Parameters\n        ----------\n        values : iterable, Series, DataFrame or dict\n            The result will only be true at a location if all the\n            labels match. If `values` is a Series, that's the index. If\n            `values` is a dict, the keys must be the column names,\n            which must match. If `values` is a DataFrame,\n            then both the index and column labels must match.\n\n        Returns\n        -------\n        DataFrame\n            DataFrame of booleans showing whether each element in the DataFrame\n            is contained in values.\n\n        See Also\n        --------\n        DataFrame.eq: Equality test for DataFrame.\n        Series.isin: Equivalent method on Series.\n        Series.str.contains: Test if pattern or regex is contained within a\n            string of a Series or Index.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},\n        ...                   index=['falcon', 'dog'])\n        >>> df\n                num_legs  num_wings\n        falcon         2          2\n        dog            4          0\n\n        When ``values`` is a list check whether every value in the DataFrame\n        is present in the list (which animals have 0 or 2 legs or wings)\n\n        >>> df.isin([0, 2])\n                num_legs  num_wings\n        falcon      True       True\n        dog        False       True\n\n        When ``values`` is a dict, we can pass values to check for each\n        column separately:\n\n        >>> df.isin({'num_wings': [0, 3]})\n                num_legs  num_wings\n        falcon     False      False\n        dog        False       True\n\n        When ``values`` is a Series or DataFrame the index and column must\n        match. Note that 'falcon' does not match based on the number of legs\n        in df2.\n\n        >>> other = pd.DataFrame({'num_legs': [8, 2], 'num_wings': [0, 2]},\n        ...                      index=['spider', 'falcon'])\n        >>> df.isin(other)\n                num_legs  num_wings\n        falcon      True       True\n        dog        False      False\n        "
        if isinstance(values, dict):
            from pandas.core.reshape.concat import concat
            values = collections.defaultdict(list, values)
            return concat((self.iloc[:, [i]].isin(values[col]) for (i, col) in enumerate(self.columns)), axis=1)
        elif isinstance(values, Series):
            if (not values.index.is_unique):
                raise ValueError('cannot compute isin with a duplicate axis.')
            return self.eq(values.reindex_like(self), axis='index')
        elif isinstance(values, DataFrame):
            if (not (values.columns.is_unique and values.index.is_unique)):
                raise ValueError('cannot compute isin with a duplicate axis.')
            return self.eq(values.reindex_like(self))
        else:
            if (not is_list_like(values)):
                raise TypeError(f"only list-like or dict-like objects are allowed to be passed to DataFrame.isin(), you passed a '{type(values).__name__}'")
            return self._constructor(algorithms.isin(self.values.ravel(), values).reshape(self.shape), self.index, self.columns)
    _AXIS_ORDERS = ['index', 'columns']
    _AXIS_TO_AXIS_NUMBER = {**NDFrame._AXIS_TO_AXIS_NUMBER, 1: 1, 'columns': 1}
    _AXIS_REVERSED = True
    _AXIS_LEN = len(_AXIS_ORDERS)
    _info_axis_number = 1
    _info_axis_name = 'columns'
    index = properties.AxisProperty(axis=1, doc='The index (row labels) of the DataFrame.')
    columns = properties.AxisProperty(axis=0, doc='The column labels of the DataFrame.')

    @property
    def _AXIS_NUMBERS(self):
        '.. deprecated:: 1.1.0'
        super()._AXIS_NUMBERS
        return {'index': 0, 'columns': 1}

    @property
    def _AXIS_NAMES(self):
        '.. deprecated:: 1.1.0'
        super()._AXIS_NAMES
        return {0: 'index', 1: 'columns'}
    plot = CachedAccessor('plot', pandas.plotting.PlotAccessor)
    hist = pandas.plotting.hist_frame
    boxplot = pandas.plotting.boxplot_frame
    sparse = CachedAccessor('sparse', SparseFrameAccessor)
DataFrame._add_numeric_operations()
ops.add_flex_arithmetic_methods(DataFrame)

def _from_nested_dict(data):
    new_data: collections.defaultdict = collections.defaultdict(dict)
    for (index, s) in data.items():
        for (col, v) in s.items():
            new_data[col][index] = v
    return new_data

def _reindex_for_setitem(value, index):
    if (value.index.equals(index) or (not len(index))):
        return value._values.copy()
    try:
        reindexed_value = value.reindex(index)._values
    except ValueError as err:
        if (not value.index.is_unique):
            raise err
        raise TypeError('incompatible index of inserted column with frame index') from err
    return reindexed_value

def _maybe_atleast_2d(value):
    if is_extension_array_dtype(value):
        return value
    return np.atleast_2d(np.asarray(value))
