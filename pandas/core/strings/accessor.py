
import codecs
from functools import wraps
import re
from typing import Dict, List, Optional
import warnings
import numpy as np
import pandas._libs.lib as lib
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_object, is_bool_dtype, is_categorical_dtype, is_integer, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCMultiIndex, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.base import NoNewAttributesMixin
_shared_docs = {}
_cpython_optimized_encoders = ('utf-8', 'utf8', 'latin-1', 'latin1', 'iso-8859-1', 'mbcs', 'ascii')
_cpython_optimized_decoders = (_cpython_optimized_encoders + ('utf-16', 'utf-32'))

def forbid_nonstring_types(forbidden, name=None):
    "\n    Decorator to forbid specific types for a method of StringMethods.\n\n    For calling `.str.{method}` on a Series or Index, it is necessary to first\n    initialize the :class:`StringMethods` object, and then call the method.\n    However, different methods allow different input types, and so this can not\n    be checked during :meth:`StringMethods.__init__`, but must be done on a\n    per-method basis. This decorator exists to facilitate this process, and\n    make it explicit which (inferred) types are disallowed by the method.\n\n    :meth:`StringMethods.__init__` allows the *union* of types its different\n    methods allow (after skipping NaNs; see :meth:`StringMethods._validate`),\n    namely: ['string', 'empty', 'bytes', 'mixed', 'mixed-integer'].\n\n    The default string types ['string', 'empty'] are allowed for all methods.\n    For the additional types ['bytes', 'mixed', 'mixed-integer'], each method\n    then needs to forbid the types it is not intended for.\n\n    Parameters\n    ----------\n    forbidden : list-of-str or None\n        List of forbidden non-string types, may be one or more of\n        `['bytes', 'mixed', 'mixed-integer']`.\n    name : str, default None\n        Name of the method to use in the error message. By default, this is\n        None, in which case the name from the method being wrapped will be\n        copied. However, for working with further wrappers (like _pat_wrapper\n        and _noarg_wrapper), it is necessary to specify the name.\n\n    Returns\n    -------\n    func : wrapper\n        The method to which the decorator is applied, with an added check that\n        enforces the inferred type to not be in the list of forbidden types.\n\n    Raises\n    ------\n    TypeError\n        If the inferred type of the underlying data is in `forbidden`.\n    "
    forbidden = ([] if (forbidden is None) else forbidden)
    allowed_types = ({'string', 'empty', 'bytes', 'mixed', 'mixed-integer'} - set(forbidden))

    def _forbid_nonstring_types(func):
        func_name = (func.__name__ if (name is None) else name)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if (self._inferred_dtype not in allowed_types):
                msg = f"Cannot use .str.{func_name} with values of inferred dtype '{self._inferred_dtype}'."
                raise TypeError(msg)
            return func(self, *args, **kwargs)
        wrapper.__name__ = func_name
        return wrapper
    return _forbid_nonstring_types

def _map_and_wrap(name, docstring):

    @forbid_nonstring_types(['bytes'], name=name)
    def wrapper(self):
        result = getattr(self._array, f'_str_{name}')()
        return self._wrap_result(result)
    wrapper.__doc__ = docstring
    return wrapper

class StringMethods(NoNewAttributesMixin):
    '\n    Vectorized string functions for Series and Index.\n\n    NAs stay NA unless handled otherwise by a particular method.\n    Patterned after Python\'s string methods, with some inspiration from\n    R\'s stringr package.\n\n    Examples\n    --------\n    >>> s = pd.Series(["A_Str_Series"])\n    >>> s\n    0    A_Str_Series\n    dtype: object\n\n    >>> s.str.split("_")\n    0    [A, Str, Series]\n    dtype: object\n\n    >>> s.str.replace("_", "")\n    0    AStrSeries\n    dtype: object\n    '

    def __init__(self, data):
        from pandas.core.arrays.string_ import StringDtype
        self._inferred_dtype = self._validate(data)
        self._is_categorical = is_categorical_dtype(data.dtype)
        self._is_string = isinstance(data.dtype, StringDtype)
        array = data.array
        self._array = array
        self._index = self._name = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name
        self._parent = (data._values.categories if self._is_categorical else data)
        self._orig = data
        self._freeze()

    @staticmethod
    def _validate(data):
        '\n        Auxiliary function for StringMethods, infers and checks dtype of data.\n\n        This is a "first line of defence" at the creation of the StringMethods-\n        object, and just checks that the dtype is in the\n        *union* of the allowed types over all string methods below; this\n        restriction is then refined on a per-method basis using the decorator\n        @forbid_nonstring_types (more info in the corresponding docstring).\n\n        This really should exclude all series/index with any non-string values,\n        but that isn\'t practical for performance reasons until we have a str\n        dtype (GH 9343 / 13877)\n\n        Parameters\n        ----------\n        data : The content of the Series\n\n        Returns\n        -------\n        dtype : inferred dtype of data\n        '
        from pandas import StringDtype
        if isinstance(data, ABCMultiIndex):
            raise AttributeError('Can only use .str accessor with Index, not MultiIndex')
        allowed_types = ['string', 'empty', 'bytes', 'mixed', 'mixed-integer']
        values = getattr(data, 'values', data)
        values = getattr(values, 'categories', values)
        if isinstance(values.dtype, StringDtype):
            return 'string'
        try:
            inferred_dtype = lib.infer_dtype(values, skipna=True)
        except ValueError:
            inferred_dtype = None
        if (inferred_dtype not in allowed_types):
            raise AttributeError('Can only use .str accessor with string values!')
        return inferred_dtype

    def __getitem__(self, key):
        result = self._array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self):
        warnings.warn('Columnar iteration over characters will be deprecated in future releases.', FutureWarning, stacklevel=2)
        i = 0
        g = self.get(i)
        while g.notna().any():
            (yield g)
            i += 1
            g = self.get(i)

    def _wrap_result(self, result, name=None, expand=None, fill_value=np.nan, returns_string=True):
        from pandas import Index, MultiIndex
        if ((not hasattr(result, 'ndim')) or (not hasattr(result, 'dtype'))):
            if isinstance(result, ABCDataFrame):
                result = result.__finalize__(self._orig, name='str')
            return result
        assert (result.ndim < 3)
        if (expand is None):
            expand = (result.ndim != 1)
        elif ((expand is True) and (not isinstance(self._orig, ABCIndex))):

            def cons_row(x):
                if is_list_like(x):
                    return x
                else:
                    return [x]
            result = [cons_row(x) for x in result]
            if result:
                max_len = max((len(x) for x in result))
                result = [((x * max_len) if ((len(x) == 0) or (x[0] is np.nan)) else x) for x in result]
        if (not isinstance(expand, bool)):
            raise ValueError('expand must be True or False')
        if (expand is False):
            if (name is None):
                name = getattr(result, 'name', None)
            if (name is None):
                name = self._orig.name
        if isinstance(self._orig, ABCIndex):
            if is_bool_dtype(result):
                return result
            if expand:
                result = list(result)
                out = MultiIndex.from_tuples(result, names=name)
                if (out.nlevels == 1):
                    out = out.get_level_values(0)
                return out
            else:
                return Index(result, name=name)
        else:
            index = self._orig.index
            dtype: Optional[str]
            if (self._is_string and returns_string):
                dtype = 'string'
            else:
                dtype = None
            if expand:
                cons = self._orig._constructor_expanddim
                result = cons(result, columns=name, index=index, dtype=dtype)
            else:
                cons = self._orig._constructor
                result = cons(result, name=name, index=index)
            result = result.__finalize__(self._orig, method='str')
            if ((name is not None) and (result.ndim == 1)):
                result.name = name
            return result

    def _get_series_list(self, others):
        '\n        Auxiliary function for :meth:`str.cat`. Turn potentially mixed input\n        into a list of Series (elements without an index must match the length\n        of the calling Series/Index).\n\n        Parameters\n        ----------\n        others : Series, DataFrame, np.ndarray, list-like or list-like of\n            Objects that are either Series, Index or np.ndarray (1-dim).\n\n        Returns\n        -------\n        list of Series\n            Others transformed into list of Series.\n        '
        from pandas import DataFrame, Series
        idx = (self._orig if isinstance(self._orig, ABCIndex) else self._orig.index)
        if isinstance(others, ABCSeries):
            return [others]
        elif isinstance(others, ABCIndex):
            return [Series(others._values, index=idx)]
        elif isinstance(others, ABCDataFrame):
            return [others[x] for x in others]
        elif (isinstance(others, np.ndarray) and (others.ndim == 2)):
            others = DataFrame(others, index=idx)
            return [others[x] for x in others]
        elif is_list_like(others, allow_sets=False):
            others = list(others)
            if all(((isinstance(x, (ABCSeries, ABCIndex)) or (isinstance(x, np.ndarray) and (x.ndim == 1))) for x in others)):
                los: List[Series] = []
                while others:
                    los = (los + self._get_series_list(others.pop(0)))
                return los
            elif all(((not is_list_like(x)) for x in others)):
                return [Series(others, index=idx)]
        raise TypeError('others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])')

    @forbid_nonstring_types(['bytes', 'mixed', 'mixed-integer'])
    def cat(self, others=None, sep=None, na_rep=None, join='left'):
        "\n        Concatenate strings in the Series/Index with given separator.\n\n        If `others` is specified, this function concatenates the Series/Index\n        and elements of `others` element-wise.\n        If `others` is not passed, then all values in the Series/Index are\n        concatenated into a single string with a given `sep`.\n\n        Parameters\n        ----------\n        others : Series, Index, DataFrame, np.ndarray or list-like\n            Series, Index, DataFrame, np.ndarray (one- or two-dimensional) and\n            other list-likes of strings must have the same length as the\n            calling Series/Index, with the exception of indexed objects (i.e.\n            Series/Index/DataFrame) if `join` is not None.\n\n            If others is a list-like that contains a combination of Series,\n            Index or np.ndarray (1-dim), then all elements will be unpacked and\n            must satisfy the above criteria individually.\n\n            If others is None, the method returns the concatenation of all\n            strings in the calling Series/Index.\n        sep : str, default ''\n            The separator between the different elements/columns. By default\n            the empty string `''` is used.\n        na_rep : str or None, default None\n            Representation that is inserted for all missing values:\n\n            - If `na_rep` is None, and `others` is None, missing values in the\n              Series/Index are omitted from the result.\n            - If `na_rep` is None, and `others` is not None, a row containing a\n              missing value in any of the columns (before concatenation) will\n              have a missing value in the result.\n        join : {'left', 'right', 'outer', 'inner'}, default 'left'\n            Determines the join-style between the calling Series/Index and any\n            Series/Index/DataFrame in `others` (objects without an index need\n            to match the length of the calling Series/Index). To disable\n            alignment, use `.values` on any Series/Index/DataFrame in `others`.\n\n            .. versionadded:: 0.23.0\n            .. versionchanged:: 1.0.0\n                Changed default of `join` from None to `'left'`.\n\n        Returns\n        -------\n        str, Series or Index\n            If `others` is None, `str` is returned, otherwise a `Series/Index`\n            (same type as caller) of objects is returned.\n\n        See Also\n        --------\n        split : Split each string in the Series/Index.\n        join : Join lists contained as elements in the Series/Index.\n\n        Examples\n        --------\n        When not passing `others`, all values are concatenated into a single\n        string:\n\n        >>> s = pd.Series(['a', 'b', np.nan, 'd'])\n        >>> s.str.cat(sep=' ')\n        'a b d'\n\n        By default, NA values in the Series are ignored. Using `na_rep`, they\n        can be given a representation:\n\n        >>> s.str.cat(sep=' ', na_rep='?')\n        'a b ? d'\n\n        If `others` is specified, corresponding values are concatenated with\n        the separator. Result will be a Series of strings.\n\n        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',')\n        0    a,A\n        1    b,B\n        2    NaN\n        3    d,D\n        dtype: object\n\n        Missing values will remain missing in the result, but can again be\n        represented using `na_rep`\n\n        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',', na_rep='-')\n        0    a,A\n        1    b,B\n        2    -,C\n        3    d,D\n        dtype: object\n\n        If `sep` is not specified, the values are concatenated without\n        separation.\n\n        >>> s.str.cat(['A', 'B', 'C', 'D'], na_rep='-')\n        0    aA\n        1    bB\n        2    -C\n        3    dD\n        dtype: object\n\n        Series with different indexes can be aligned before concatenation. The\n        `join`-keyword works as in other methods.\n\n        >>> t = pd.Series(['d', 'a', 'e', 'c'], index=[3, 0, 4, 2])\n        >>> s.str.cat(t, join='left', na_rep='-')\n        0    aa\n        1    b-\n        2    -c\n        3    dd\n        dtype: object\n        >>>\n        >>> s.str.cat(t, join='outer', na_rep='-')\n        0    aa\n        1    b-\n        2    -c\n        3    dd\n        4    -e\n        dtype: object\n        >>>\n        >>> s.str.cat(t, join='inner', na_rep='-')\n        0    aa\n        2    -c\n        3    dd\n        dtype: object\n        >>>\n        >>> s.str.cat(t, join='right', na_rep='-')\n        3    dd\n        0    aa\n        4    -e\n        2    -c\n        dtype: object\n\n        For more examples, see :ref:`here <text.concatenate>`.\n        "
        from pandas import Index, Series, concat
        if isinstance(others, str):
            raise ValueError('Did you mean to supply a `sep` keyword?')
        if (sep is None):
            sep = ''
        if isinstance(self._orig, ABCIndex):
            data = Series(self._orig, index=self._orig)
        else:
            data = self._orig
        if (others is None):
            data = ensure_object(data)
            na_mask = isna(data)
            if ((na_rep is None) and na_mask.any()):
                data = data[(~ na_mask)]
            elif ((na_rep is not None) and na_mask.any()):
                data = np.where(na_mask, na_rep, data)
            return sep.join(data)
        try:
            others = self._get_series_list(others)
        except ValueError as err:
            raise ValueError('If `others` contains arrays or lists (or other list-likes without an index), these must all be of the same length as the calling Series/Index.') from err
        if any(((not data.index.equals(x.index)) for x in others)):
            others = concat(others, axis=1, join=(join if (join == 'inner') else 'outer'), keys=range(len(others)), sort=False, copy=False)
            (data, others) = data.align(others, join=join)
            others = [others[x] for x in others]
        all_cols = [ensure_object(x) for x in ([data] + others)]
        na_masks = np.array([isna(x) for x in all_cols])
        union_mask = np.logical_or.reduce(na_masks, axis=0)
        if ((na_rep is None) and union_mask.any()):
            result = np.empty(len(data), dtype=object)
            np.putmask(result, union_mask, np.nan)
            not_masked = (~ union_mask)
            result[not_masked] = cat_safe([x[not_masked] for x in all_cols], sep)
        elif ((na_rep is not None) and union_mask.any()):
            all_cols = [np.where(nm, na_rep, col) for (nm, col) in zip(na_masks, all_cols)]
            result = cat_safe(all_cols, sep)
        else:
            result = cat_safe(all_cols, sep)
        if isinstance(self._orig, ABCIndex):
            result = Index(result, dtype=object, name=self._orig.name)
        else:
            if is_categorical_dtype(self._orig.dtype):
                dtype = None
            else:
                dtype = self._orig.dtype
            result = Series(result, dtype=dtype, index=data.index, name=self._orig.name)
            result = result.__finalize__(self._orig, method='str_cat')
        return result
    _shared_docs['str_split'] = '\n    Split strings around given separator/delimiter.\n\n    Splits the string in the Series/Index from the %(side)s,\n    at the specified delimiter string. Equivalent to :meth:`str.%(method)s`.\n\n    Parameters\n    ----------\n    pat : str, optional\n        String or regular expression to split on.\n        If not specified, split on whitespace.\n    n : int, default -1 (all)\n        Limit number of splits in output.\n        ``None``, 0 and -1 will be interpreted as return all splits.\n    expand : bool, default False\n        Expand the split strings into separate columns.\n\n        * If ``True``, return DataFrame/MultiIndex expanding dimensionality.\n        * If ``False``, return Series/Index, containing lists of strings.\n\n    Returns\n    -------\n    Series, Index, DataFrame or MultiIndex\n        Type matches caller unless ``expand=True`` (see Notes).\n\n    See Also\n    --------\n    Series.str.split : Split strings around given separator/delimiter.\n    Series.str.rsplit : Splits string around given separator/delimiter,\n        starting from the right.\n    Series.str.join : Join lists contained as elements in the Series/Index\n        with passed delimiter.\n    str.split : Standard library version for split.\n    str.rsplit : Standard library version for rsplit.\n\n    Notes\n    -----\n    The handling of the `n` keyword depends on the number of found splits:\n\n    - If found splits > `n`,  make first `n` splits only\n    - If found splits <= `n`, make all splits\n    - If for a certain row the number of found splits < `n`,\n      append `None` for padding up to `n` if ``expand=True``\n\n    If using ``expand=True``, Series and Index callers return DataFrame and\n    MultiIndex objects, respectively.\n\n    Examples\n    --------\n    >>> s = pd.Series(\n    ...     [\n    ...         "this is a regular sentence",\n    ...         "https://docs.python.org/3/tutorial/index.html",\n    ...         np.nan\n    ...     ]\n    ... )\n    >>> s\n    0                       this is a regular sentence\n    1    https://docs.python.org/3/tutorial/index.html\n    2                                              NaN\n    dtype: object\n\n    In the default setting, the string is split by whitespace.\n\n    >>> s.str.split()\n    0                   [this, is, a, regular, sentence]\n    1    [https://docs.python.org/3/tutorial/index.html]\n    2                                                NaN\n    dtype: object\n\n    Without the `n` parameter, the outputs of `rsplit` and `split`\n    are identical.\n\n    >>> s.str.rsplit()\n    0                   [this, is, a, regular, sentence]\n    1    [https://docs.python.org/3/tutorial/index.html]\n    2                                                NaN\n    dtype: object\n\n    The `n` parameter can be used to limit the number of splits on the\n    delimiter. The outputs of `split` and `rsplit` are different.\n\n    >>> s.str.split(n=2)\n    0                     [this, is, a regular sentence]\n    1    [https://docs.python.org/3/tutorial/index.html]\n    2                                                NaN\n    dtype: object\n\n    >>> s.str.rsplit(n=2)\n    0                     [this is a, regular, sentence]\n    1    [https://docs.python.org/3/tutorial/index.html]\n    2                                                NaN\n    dtype: object\n\n    The `pat` parameter can be used to split by other characters.\n\n    >>> s.str.split(pat="/")\n    0                         [this is a regular sentence]\n    1    [https:, , docs.python.org, 3, tutorial, index...\n    2                                                  NaN\n    dtype: object\n\n    When using ``expand=True``, the split elements will expand out into\n    separate columns. If NaN is present, it is propagated throughout\n    the columns during the split.\n\n    >>> s.str.split(expand=True)\n                                                   0     1     2        3         4\n    0                                           this    is     a  regular  sentence\n    1  https://docs.python.org/3/tutorial/index.html  None  None     None      None\n    2                                            NaN   NaN   NaN      NaN       NaN\n\n    For slightly more complex use cases like splitting the html document name\n    from a url, a combination of parameter settings can be used.\n\n    >>> s.str.rsplit("/", n=1, expand=True)\n                                        0           1\n    0          this is a regular sentence        None\n    1  https://docs.python.org/3/tutorial  index.html\n    2                                 NaN         NaN\n\n    Remember to escape special characters when explicitly using regular\n    expressions.\n\n    >>> s = pd.Series(["1+1=2"])\n    >>> s\n    0    1+1=2\n    dtype: object\n    >>> s.str.split(r"\\+|=", expand=True)\n         0    1    2\n    0    1    1    2\n    '

    @Appender((_shared_docs['str_split'] % {'side': 'beginning', 'method': 'split'}))
    @forbid_nonstring_types(['bytes'])
    def split(self, pat=None, n=(- 1), expand=False):
        result = self._array._str_split(pat, n, expand)
        return self._wrap_result(result, returns_string=expand, expand=expand)

    @Appender((_shared_docs['str_split'] % {'side': 'end', 'method': 'rsplit'}))
    @forbid_nonstring_types(['bytes'])
    def rsplit(self, pat=None, n=(- 1), expand=False):
        result = self._array._str_rsplit(pat, n=n)
        return self._wrap_result(result, expand=expand, returns_string=expand)
    _shared_docs['str_partition'] = "\n    Split the string at the %(side)s occurrence of `sep`.\n\n    This method splits the string at the %(side)s occurrence of `sep`,\n    and returns 3 elements containing the part before the separator,\n    the separator itself, and the part after the separator.\n    If the separator is not found, return %(return)s.\n\n    Parameters\n    ----------\n    sep : str, default whitespace\n        String to split on.\n    expand : bool, default True\n        If True, return DataFrame/MultiIndex expanding dimensionality.\n        If False, return Series/Index.\n\n    Returns\n    -------\n    DataFrame/MultiIndex or Series/Index of objects\n\n    See Also\n    --------\n    %(also)s\n    Series.str.split : Split strings around given separators.\n    str.partition : Standard library version.\n\n    Examples\n    --------\n\n    >>> s = pd.Series(['Linda van der Berg', 'George Pitt-Rivers'])\n    >>> s\n    0    Linda van der Berg\n    1    George Pitt-Rivers\n    dtype: object\n\n    >>> s.str.partition()\n            0  1             2\n    0   Linda     van der Berg\n    1  George      Pitt-Rivers\n\n    To partition by the last space instead of the first one:\n\n    >>> s.str.rpartition()\n                   0  1            2\n    0  Linda van der            Berg\n    1         George     Pitt-Rivers\n\n    To partition by something different than a space:\n\n    >>> s.str.partition('-')\n                        0  1       2\n    0  Linda van der Berg\n    1         George Pitt  -  Rivers\n\n    To return a Series containing tuples instead of a DataFrame:\n\n    >>> s.str.partition('-', expand=False)\n    0    (Linda van der Berg, , )\n    1    (George Pitt, -, Rivers)\n    dtype: object\n\n    Also available on indices:\n\n    >>> idx = pd.Index(['X 123', 'Y 999'])\n    >>> idx\n    Index(['X 123', 'Y 999'], dtype='object')\n\n    Which will create a MultiIndex:\n\n    >>> idx.str.partition()\n    MultiIndex([('X', ' ', '123'),\n                ('Y', ' ', '999')],\n               )\n\n    Or an index with tuples with ``expand=False``:\n\n    >>> idx.str.partition(expand=False)\n    Index([('X', ' ', '123'), ('Y', ' ', '999')], dtype='object')\n    "

    @Appender((_shared_docs['str_partition'] % {'side': 'first', 'return': '3 elements containing the string itself, followed by two empty strings', 'also': 'rpartition : Split the string at the last occurrence of `sep`.'}))
    @forbid_nonstring_types(['bytes'])
    def partition(self, sep=' ', expand=True):
        result = self._array._str_partition(sep, expand)
        return self._wrap_result(result, expand=expand, returns_string=expand)

    @Appender((_shared_docs['str_partition'] % {'side': 'last', 'return': '3 elements containing two empty strings, followed by the string itself', 'also': 'partition : Split the string at the first occurrence of `sep`.'}))
    @forbid_nonstring_types(['bytes'])
    def rpartition(self, sep=' ', expand=True):
        result = self._array._str_rpartition(sep, expand)
        return self._wrap_result(result, expand=expand, returns_string=expand)

    def get(self, i):
        '\n        Extract element from each component at specified position.\n\n        Extract element from lists, tuples, or strings in each element in the\n        Series/Index.\n\n        Parameters\n        ----------\n        i : int\n            Position of element to extract.\n\n        Returns\n        -------\n        Series or Index\n\n        Examples\n        --------\n        >>> s = pd.Series(["String",\n        ...               (1, 2, 3),\n        ...               ["a", "b", "c"],\n        ...               123,\n        ...               -456,\n        ...               {1: "Hello", "2": "World"}])\n        >>> s\n        0                        String\n        1                     (1, 2, 3)\n        2                     [a, b, c]\n        3                           123\n        4                          -456\n        5    {1: \'Hello\', \'2\': \'World\'}\n        dtype: object\n\n        >>> s.str.get(1)\n        0        t\n        1        2\n        2        b\n        3      NaN\n        4      NaN\n        5    Hello\n        dtype: object\n\n        >>> s.str.get(-1)\n        0      g\n        1      3\n        2      c\n        3    NaN\n        4    NaN\n        5    None\n        dtype: object\n        '
        result = self._array._str_get(i)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def join(self, sep):
        "\n        Join lists contained as elements in the Series/Index with passed delimiter.\n\n        If the elements of a Series are lists themselves, join the content of these\n        lists using the delimiter passed to the function.\n        This function is an equivalent to :meth:`str.join`.\n\n        Parameters\n        ----------\n        sep : str\n            Delimiter to use between list entries.\n\n        Returns\n        -------\n        Series/Index: object\n            The list entries concatenated by intervening occurrences of the\n            delimiter.\n\n        Raises\n        ------\n        AttributeError\n            If the supplied Series contains neither strings nor lists.\n\n        See Also\n        --------\n        str.join : Standard library version of this method.\n        Series.str.split : Split strings around given separator/delimiter.\n\n        Notes\n        -----\n        If any of the list items is not a string object, the result of the join\n        will be `NaN`.\n\n        Examples\n        --------\n        Example with a list that contains non-string elements.\n\n        >>> s = pd.Series([['lion', 'elephant', 'zebra'],\n        ...                [1.1, 2.2, 3.3],\n        ...                ['cat', np.nan, 'dog'],\n        ...                ['cow', 4.5, 'goat'],\n        ...                ['duck', ['swan', 'fish'], 'guppy']])\n        >>> s\n        0        [lion, elephant, zebra]\n        1                [1.1, 2.2, 3.3]\n        2                [cat, nan, dog]\n        3               [cow, 4.5, goat]\n        4    [duck, [swan, fish], guppy]\n        dtype: object\n\n        Join all lists using a '-'. The lists containing object(s) of types other\n        than str will produce a NaN.\n\n        >>> s.str.join('-')\n        0    lion-elephant-zebra\n        1                    NaN\n        2                    NaN\n        3                    NaN\n        4                    NaN\n        dtype: object\n        "
        result = self._array._str_join(sep)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def contains(self, pat, case=True, flags=0, na=None, regex=True):
        "\n        Test if pattern or regex is contained within a string of a Series or Index.\n\n        Return boolean Series or Index based on whether a given pattern or regex is\n        contained within a string of a Series or Index.\n\n        Parameters\n        ----------\n        pat : str\n            Character sequence or regular expression.\n        case : bool, default True\n            If True, case sensitive.\n        flags : int, default 0 (no flags)\n            Flags to pass through to the re module, e.g. re.IGNORECASE.\n        na : scalar, optional\n            Fill value for missing values. The default depends on dtype of the\n            array. For object-dtype, ``numpy.nan`` is used. For ``StringDtype``,\n            ``pandas.NA`` is used.\n        regex : bool, default True\n            If True, assumes the pat is a regular expression.\n\n            If False, treats the pat as a literal string.\n\n        Returns\n        -------\n        Series or Index of boolean values\n            A Series or Index of boolean values indicating whether the\n            given pattern is contained within the string of each element\n            of the Series or Index.\n\n        See Also\n        --------\n        match : Analogous, but stricter, relying on re.match instead of re.search.\n        Series.str.startswith : Test if the start of each string element matches a\n            pattern.\n        Series.str.endswith : Same as startswith, but tests the end of string.\n\n        Examples\n        --------\n        Returning a Series of booleans using only a literal pattern.\n\n        >>> s1 = pd.Series(['Mouse', 'dog', 'house and parrot', '23', np.NaN])\n        >>> s1.str.contains('og', regex=False)\n        0    False\n        1     True\n        2    False\n        3    False\n        4      NaN\n        dtype: object\n\n        Returning an Index of booleans using only a literal pattern.\n\n        >>> ind = pd.Index(['Mouse', 'dog', 'house and parrot', '23.0', np.NaN])\n        >>> ind.str.contains('23', regex=False)\n        Index([False, False, False, True, nan], dtype='object')\n\n        Specifying case sensitivity using `case`.\n\n        >>> s1.str.contains('oG', case=True, regex=True)\n        0    False\n        1    False\n        2    False\n        3    False\n        4      NaN\n        dtype: object\n\n        Specifying `na` to be `False` instead of `NaN` replaces NaN values\n        with `False`. If Series or Index does not contain NaN values\n        the resultant dtype will be `bool`, otherwise, an `object` dtype.\n\n        >>> s1.str.contains('og', na=False, regex=True)\n        0    False\n        1     True\n        2    False\n        3    False\n        4    False\n        dtype: bool\n\n        Returning 'house' or 'dog' when either expression occurs in a string.\n\n        >>> s1.str.contains('house|dog', regex=True)\n        0    False\n        1     True\n        2     True\n        3    False\n        4      NaN\n        dtype: object\n\n        Ignoring case sensitivity using `flags` with regex.\n\n        >>> import re\n        >>> s1.str.contains('PARROT', flags=re.IGNORECASE, regex=True)\n        0    False\n        1    False\n        2     True\n        3    False\n        4      NaN\n        dtype: object\n\n        Returning any digit using regular expression.\n\n        >>> s1.str.contains('\\\\d', regex=True)\n        0    False\n        1    False\n        2    False\n        3     True\n        4      NaN\n        dtype: object\n\n        Ensure `pat` is a not a literal pattern when `regex` is set to True.\n        Note in the following example one might expect only `s2[1]` and `s2[3]` to\n        return `True`. However, '.0' as a regex matches any character\n        followed by a 0.\n\n        >>> s2 = pd.Series(['40', '40.0', '41', '41.0', '35'])\n        >>> s2.str.contains('.0', regex=True)\n        0     True\n        1     True\n        2    False\n        3     True\n        4    False\n        dtype: bool\n        "
        result = self._array._str_contains(pat, case, flags, na, regex)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def match(self, pat, case=True, flags=0, na=None):
        '\n        Determine if each string starts with a match of a regular expression.\n\n        Parameters\n        ----------\n        pat : str\n            Character sequence or regular expression.\n        case : bool, default True\n            If True, case sensitive.\n        flags : int, default 0 (no flags)\n            Regex module flags, e.g. re.IGNORECASE.\n        na : scalar, optional\n            Fill value for missing values. The default depends on dtype of the\n            array. For object-dtype, ``numpy.nan`` is used. For ``StringDtype``,\n            ``pandas.NA`` is used.\n\n        Returns\n        -------\n        Series/array of boolean values\n\n        See Also\n        --------\n        fullmatch : Stricter matching that requires the entire string to match.\n        contains : Analogous, but less strict, relying on re.search instead of\n            re.match.\n        extract : Extract matched groups.\n        '
        result = self._array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def fullmatch(self, pat, case=True, flags=0, na=None):
        '\n        Determine if each string entirely matches a regular expression.\n\n        .. versionadded:: 1.1.0\n\n        Parameters\n        ----------\n        pat : str\n            Character sequence or regular expression.\n        case : bool, default True\n            If True, case sensitive.\n        flags : int, default 0 (no flags)\n            Regex module flags, e.g. re.IGNORECASE.\n        na : scalar, optional.\n            Fill value for missing values. The default depends on dtype of the\n            array. For object-dtype, ``numpy.nan`` is used. For ``StringDtype``,\n            ``pandas.NA`` is used.\n\n        Returns\n        -------\n        Series/array of boolean values\n\n        See Also\n        --------\n        match : Similar, but also returns `True` when only a *prefix* of the string\n            matches the regular expression.\n        extract : Extract matched groups.\n        '
        result = self._array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def replace(self, pat, repl, n=(- 1), case=None, flags=0, regex=None):
        '\n        Replace each occurrence of pattern/regex in the Series/Index.\n\n        Equivalent to :meth:`str.replace` or :func:`re.sub`, depending on\n        the regex value.\n\n        Parameters\n        ----------\n        pat : str or compiled regex\n            String can be a character sequence or regular expression.\n        repl : str or callable\n            Replacement string or a callable. The callable is passed the regex\n            match object and must return a replacement string to be used.\n            See :func:`re.sub`.\n        n : int, default -1 (all)\n            Number of replacements to make from start.\n        case : bool, default None\n            Determines if replace is case sensitive:\n\n            - If True, case sensitive (the default if `pat` is a string)\n            - Set to False for case insensitive\n            - Cannot be set if `pat` is a compiled regex.\n\n        flags : int, default 0 (no flags)\n            Regex module flags, e.g. re.IGNORECASE. Cannot be set if `pat` is a compiled\n            regex.\n        regex : bool, default True\n            Determines if assumes the passed-in pattern is a regular expression:\n\n            - If True, assumes the passed-in pattern is a regular expression.\n            - If False, treats the pattern as a literal string\n            - Cannot be set to False if `pat` is a compiled regex or `repl` is\n              a callable.\n\n            .. versionadded:: 0.23.0\n\n        Returns\n        -------\n        Series or Index of object\n            A copy of the object with all matching occurrences of `pat` replaced by\n            `repl`.\n\n        Raises\n        ------\n        ValueError\n            * if `regex` is False and `repl` is a callable or `pat` is a compiled\n              regex\n            * if `pat` is a compiled regex and `case` or `flags` is set\n\n        Notes\n        -----\n        When `pat` is a compiled regex, all flags should be included in the\n        compiled regex. Use of `case`, `flags`, or `regex=False` with a compiled\n        regex will raise an error.\n\n        Examples\n        --------\n        When `pat` is a string and `regex` is True (the default), the given `pat`\n        is compiled as a regex. When `repl` is a string, it replaces matching\n        regex patterns as with :meth:`re.sub`. NaN value(s) in the Series are\n        left as is:\n\n        >>> pd.Series([\'foo\', \'fuz\', np.nan]).str.replace(\'f.\', \'ba\', regex=True)\n        0    bao\n        1    baz\n        2    NaN\n        dtype: object\n\n        When `pat` is a string and `regex` is False, every `pat` is replaced with\n        `repl` as with :meth:`str.replace`:\n\n        >>> pd.Series([\'f.o\', \'fuz\', np.nan]).str.replace(\'f.\', \'ba\', regex=False)\n        0    bao\n        1    fuz\n        2    NaN\n        dtype: object\n\n        When `repl` is a callable, it is called on every `pat` using\n        :func:`re.sub`. The callable should expect one positional argument\n        (a regex object) and return a string.\n\n        To get the idea:\n\n        >>> pd.Series([\'foo\', \'fuz\', np.nan]).str.replace(\'f\', repr)\n        0    <re.Match object; span=(0, 1), match=\'f\'>oo\n        1    <re.Match object; span=(0, 1), match=\'f\'>uz\n        2                                            NaN\n        dtype: object\n\n        Reverse every lowercase alphabetic word:\n\n        >>> repl = lambda m: m.group(0)[::-1]\n        >>> pd.Series([\'foo 123\', \'bar baz\', np.nan]).str.replace(r\'[a-z]+\', repl)\n        0    oof 123\n        1    rab zab\n        2        NaN\n        dtype: object\n\n        Using regex groups (extract second group and swap case):\n\n        >>> pat = r"(?P<one>\\w+) (?P<two>\\w+) (?P<three>\\w+)"\n        >>> repl = lambda m: m.group(\'two\').swapcase()\n        >>> pd.Series([\'One Two Three\', \'Foo Bar Baz\']).str.replace(pat, repl)\n        0    tWO\n        1    bAR\n        dtype: object\n\n        Using a compiled regex with flags\n\n        >>> import re\n        >>> regex_pat = re.compile(r\'FUZ\', flags=re.IGNORECASE)\n        >>> pd.Series([\'foo\', \'fuz\', np.nan]).str.replace(regex_pat, \'bar\')\n        0    foo\n        1    bar\n        2    NaN\n        dtype: object\n        '
        if (regex is None):
            if (isinstance(pat, str) and any(((c in pat) for c in '.+*|^$?[](){}\\'))):
                msg = 'The default value of regex will change from True to False in a future version.'
                if (len(pat) == 1):
                    msg += ' In addition, single character regular expressions will*not* be treated as literal strings when regex=True.'
                warnings.warn(msg, FutureWarning, stacklevel=3)
            regex = True
        result = self._array._str_replace(pat, repl, n=n, case=case, flags=flags, regex=regex)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def repeat(self, repeats):
        "\n        Duplicate each string in the Series or Index.\n\n        Parameters\n        ----------\n        repeats : int or sequence of int\n            Same value for all (int) or different value per (sequence).\n\n        Returns\n        -------\n        Series or Index of object\n            Series or Index of repeated string objects specified by\n            input parameter repeats.\n\n        Examples\n        --------\n        >>> s = pd.Series(['a', 'b', 'c'])\n        >>> s\n        0    a\n        1    b\n        2    c\n        dtype: object\n\n        Single int repeats string in Series\n\n        >>> s.str.repeat(repeats=2)\n        0    aa\n        1    bb\n        2    cc\n        dtype: object\n\n        Sequence of int repeats corresponding string in Series\n\n        >>> s.str.repeat(repeats=[1, 2, 3])\n        0      a\n        1     bb\n        2    ccc\n        dtype: object\n        "
        result = self._array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def pad(self, width, side='left', fillchar=' '):
        '\n        Pad strings in the Series/Index up to width.\n\n        Parameters\n        ----------\n        width : int\n            Minimum width of resulting string; additional characters will be filled\n            with character defined in `fillchar`.\n        side : {\'left\', \'right\', \'both\'}, default \'left\'\n            Side from which to fill resulting string.\n        fillchar : str, default \' \'\n            Additional character for filling, default is whitespace.\n\n        Returns\n        -------\n        Series or Index of object\n            Returns Series or Index with minimum number of char in object.\n\n        See Also\n        --------\n        Series.str.rjust : Fills the left side of strings with an arbitrary\n            character. Equivalent to ``Series.str.pad(side=\'left\')``.\n        Series.str.ljust : Fills the right side of strings with an arbitrary\n            character. Equivalent to ``Series.str.pad(side=\'right\')``.\n        Series.str.center : Fills both sides of strings with an arbitrary\n            character. Equivalent to ``Series.str.pad(side=\'both\')``.\n        Series.str.zfill : Pad strings in the Series/Index by prepending \'0\'\n            character. Equivalent to ``Series.str.pad(side=\'left\', fillchar=\'0\')``.\n\n        Examples\n        --------\n        >>> s = pd.Series(["caribou", "tiger"])\n        >>> s\n        0    caribou\n        1      tiger\n        dtype: object\n\n        >>> s.str.pad(width=10)\n        0       caribou\n        1         tiger\n        dtype: object\n\n        >>> s.str.pad(width=10, side=\'right\', fillchar=\'-\')\n        0    caribou---\n        1    tiger-----\n        dtype: object\n\n        >>> s.str.pad(width=10, side=\'both\', fillchar=\'-\')\n        0    -caribou--\n        1    --tiger---\n        dtype: object\n        '
        if (not isinstance(fillchar, str)):
            msg = f'fillchar must be a character, not {type(fillchar).__name__}'
            raise TypeError(msg)
        if (len(fillchar) != 1):
            raise TypeError('fillchar must be a character, not str')
        if (not is_integer(width)):
            msg = f'width must be of integer type, not {type(width).__name__}'
            raise TypeError(msg)
        result = self._array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)
    _shared_docs['str_pad'] = '\n    Pad %(side)s side of strings in the Series/Index.\n\n    Equivalent to :meth:`str.%(method)s`.\n\n    Parameters\n    ----------\n    width : int\n        Minimum width of resulting string; additional characters will be filled\n        with ``fillchar``.\n    fillchar : str\n        Additional character for filling, default is whitespace.\n\n    Returns\n    -------\n    filled : Series/Index of objects.\n    '

    @Appender((_shared_docs['str_pad'] % {'side': 'left and right', 'method': 'center'}))
    @forbid_nonstring_types(['bytes'])
    def center(self, width, fillchar=' '):
        return self.pad(width, side='both', fillchar=fillchar)

    @Appender((_shared_docs['str_pad'] % {'side': 'right', 'method': 'ljust'}))
    @forbid_nonstring_types(['bytes'])
    def ljust(self, width, fillchar=' '):
        return self.pad(width, side='right', fillchar=fillchar)

    @Appender((_shared_docs['str_pad'] % {'side': 'left', 'method': 'rjust'}))
    @forbid_nonstring_types(['bytes'])
    def rjust(self, width, fillchar=' '):
        return self.pad(width, side='left', fillchar=fillchar)

    @forbid_nonstring_types(['bytes'])
    def zfill(self, width):
        "\n        Pad strings in the Series/Index by prepending '0' characters.\n\n        Strings in the Series/Index are padded with '0' characters on the\n        left of the string to reach a total string length  `width`. Strings\n        in the Series/Index with length greater or equal to `width` are\n        unchanged.\n\n        Parameters\n        ----------\n        width : int\n            Minimum length of resulting string; strings with length less\n            than `width` be prepended with '0' characters.\n\n        Returns\n        -------\n        Series/Index of objects.\n\n        See Also\n        --------\n        Series.str.rjust : Fills the left side of strings with an arbitrary\n            character.\n        Series.str.ljust : Fills the right side of strings with an arbitrary\n            character.\n        Series.str.pad : Fills the specified sides of strings with an arbitrary\n            character.\n        Series.str.center : Fills both sides of strings with an arbitrary\n            character.\n\n        Notes\n        -----\n        Differs from :meth:`str.zfill` which has special handling\n        for '+'/'-' in the string.\n\n        Examples\n        --------\n        >>> s = pd.Series(['-1', '1', '1000', 10, np.nan])\n        >>> s\n        0      -1\n        1       1\n        2    1000\n        3      10\n        4     NaN\n        dtype: object\n\n        Note that ``10`` and ``NaN`` are not strings, therefore they are\n        converted to ``NaN``. The minus sign in ``'-1'`` is treated as a\n        regular character and the zero is added to the left of it\n        (:meth:`str.zfill` would have moved it to the left). ``1000``\n        remains unchanged as it is longer than `width`.\n\n        >>> s.str.zfill(3)\n        0     0-1\n        1     001\n        2    1000\n        3     NaN\n        4     NaN\n        dtype: object\n        "
        result = self.pad(width, side='left', fillchar='0')
        return self._wrap_result(result)

    def slice(self, start=None, stop=None, step=None):
        '\n        Slice substrings from each element in the Series or Index.\n\n        Parameters\n        ----------\n        start : int, optional\n            Start position for slice operation.\n        stop : int, optional\n            Stop position for slice operation.\n        step : int, optional\n            Step size for slice operation.\n\n        Returns\n        -------\n        Series or Index of object\n            Series or Index from sliced substring from original string object.\n\n        See Also\n        --------\n        Series.str.slice_replace : Replace a slice with a string.\n        Series.str.get : Return element at position.\n            Equivalent to `Series.str.slice(start=i, stop=i+1)` with `i`\n            being the position.\n\n        Examples\n        --------\n        >>> s = pd.Series(["koala", "fox", "chameleon"])\n        >>> s\n        0        koala\n        1          fox\n        2    chameleon\n        dtype: object\n\n        >>> s.str.slice(start=1)\n        0        oala\n        1          ox\n        2    hameleon\n        dtype: object\n\n        >>> s.str.slice(start=-1)\n        0           a\n        1           x\n        2           n\n        dtype: object\n\n        >>> s.str.slice(stop=2)\n        0    ko\n        1    fo\n        2    ch\n        dtype: object\n\n        >>> s.str.slice(step=2)\n        0      kaa\n        1       fx\n        2    caeen\n        dtype: object\n\n        >>> s.str.slice(start=0, stop=5, step=3)\n        0    kl\n        1     f\n        2    cm\n        dtype: object\n\n        Equivalent behaviour to:\n\n        >>> s.str[0:5:3]\n        0    kl\n        1     f\n        2    cm\n        dtype: object\n        '
        result = self._array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def slice_replace(self, start=None, stop=None, repl=None):
        "\n        Replace a positional slice of a string with another value.\n\n        Parameters\n        ----------\n        start : int, optional\n            Left index position to use for the slice. If not specified (None),\n            the slice is unbounded on the left, i.e. slice from the start\n            of the string.\n        stop : int, optional\n            Right index position to use for the slice. If not specified (None),\n            the slice is unbounded on the right, i.e. slice until the\n            end of the string.\n        repl : str, optional\n            String for replacement. If not specified (None), the sliced region\n            is replaced with an empty string.\n\n        Returns\n        -------\n        Series or Index\n            Same type as the original object.\n\n        See Also\n        --------\n        Series.str.slice : Just slicing without replacement.\n\n        Examples\n        --------\n        >>> s = pd.Series(['a', 'ab', 'abc', 'abdc', 'abcde'])\n        >>> s\n        0        a\n        1       ab\n        2      abc\n        3     abdc\n        4    abcde\n        dtype: object\n\n        Specify just `start`, meaning replace `start` until the end of the\n        string with `repl`.\n\n        >>> s.str.slice_replace(1, repl='X')\n        0    aX\n        1    aX\n        2    aX\n        3    aX\n        4    aX\n        dtype: object\n\n        Specify just `stop`, meaning the start of the string to `stop` is replaced\n        with `repl`, and the rest of the string is included.\n\n        >>> s.str.slice_replace(stop=2, repl='X')\n        0       X\n        1       X\n        2      Xc\n        3     Xdc\n        4    Xcde\n        dtype: object\n\n        Specify `start` and `stop`, meaning the slice from `start` to `stop` is\n        replaced with `repl`. Everything before or after `start` and `stop` is\n        included as is.\n\n        >>> s.str.slice_replace(start=1, stop=3, repl='X')\n        0      aX\n        1      aX\n        2      aX\n        3     aXc\n        4    aXde\n        dtype: object\n        "
        result = self._array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def decode(self, encoding, errors='strict'):
        '\n        Decode character string in the Series/Index using indicated encoding.\n\n        Equivalent to :meth:`str.decode` in python2 and :meth:`bytes.decode` in\n        python3.\n\n        Parameters\n        ----------\n        encoding : str\n        errors : str, optional\n\n        Returns\n        -------\n        Series or Index\n        '
        if (encoding in _cpython_optimized_decoders):
            f = (lambda x: x.decode(encoding, errors))
        else:
            decoder = codecs.getdecoder(encoding)
            f = (lambda x: decoder(x, errors)[0])
        arr = self._array
        result = arr._str_map(f)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def encode(self, encoding, errors='strict'):
        '\n        Encode character string in the Series/Index using indicated encoding.\n\n        Equivalent to :meth:`str.encode`.\n\n        Parameters\n        ----------\n        encoding : str\n        errors : str, optional\n\n        Returns\n        -------\n        encoded : Series/Index of objects\n        '
        result = self._array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)
    _shared_docs['str_strip'] = "\n    Remove %(position)s characters.\n\n    Strip whitespaces (including newlines) or a set of specified characters\n    from each string in the Series/Index from %(side)s.\n    Equivalent to :meth:`str.%(method)s`.\n\n    Parameters\n    ----------\n    to_strip : str or None, default None\n        Specifying the set of characters to be removed.\n        All combinations of this set of characters will be stripped.\n        If None then whitespaces are removed.\n\n    Returns\n    -------\n    Series or Index of object\n\n    See Also\n    --------\n    Series.str.strip : Remove leading and trailing characters in Series/Index.\n    Series.str.lstrip : Remove leading characters in Series/Index.\n    Series.str.rstrip : Remove trailing characters in Series/Index.\n\n    Examples\n    --------\n    >>> s = pd.Series(['1. Ant.  ', '2. Bee!\\n', '3. Cat?\\t', np.nan])\n    >>> s\n    0    1. Ant.\n    1    2. Bee!\\n\n    2    3. Cat?\\t\n    3          NaN\n    dtype: object\n\n    >>> s.str.strip()\n    0    1. Ant.\n    1    2. Bee!\n    2    3. Cat?\n    3        NaN\n    dtype: object\n\n    >>> s.str.lstrip('123.')\n    0    Ant.\n    1    Bee!\\n\n    2    Cat?\\t\n    3       NaN\n    dtype: object\n\n    >>> s.str.rstrip('.!? \\n\\t')\n    0    1. Ant\n    1    2. Bee\n    2    3. Cat\n    3       NaN\n    dtype: object\n\n    >>> s.str.strip('123.!? \\n\\t')\n    0    Ant\n    1    Bee\n    2    Cat\n    3    NaN\n    dtype: object\n    "

    @Appender((_shared_docs['str_strip'] % {'side': 'left and right sides', 'method': 'strip', 'position': 'leading and trailing'}))
    @forbid_nonstring_types(['bytes'])
    def strip(self, to_strip=None):
        result = self._array._str_strip(to_strip)
        return self._wrap_result(result)

    @Appender((_shared_docs['str_strip'] % {'side': 'left side', 'method': 'lstrip', 'position': 'leading'}))
    @forbid_nonstring_types(['bytes'])
    def lstrip(self, to_strip=None):
        result = self._array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @Appender((_shared_docs['str_strip'] % {'side': 'right side', 'method': 'rstrip', 'position': 'trailing'}))
    @forbid_nonstring_types(['bytes'])
    def rstrip(self, to_strip=None):
        result = self._array._str_rstrip(to_strip)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def wrap(self, width, **kwargs):
        "\n        Wrap strings in Series/Index at specified line width.\n\n        This method has the same keyword parameters and defaults as\n        :class:`textwrap.TextWrapper`.\n\n        Parameters\n        ----------\n        width : int\n            Maximum line width.\n        expand_tabs : bool, optional\n            If True, tab characters will be expanded to spaces (default: True).\n        replace_whitespace : bool, optional\n            If True, each whitespace character (as defined by string.whitespace)\n            remaining after tab expansion will be replaced by a single space\n            (default: True).\n        drop_whitespace : bool, optional\n            If True, whitespace that, after wrapping, happens to end up at the\n            beginning or end of a line is dropped (default: True).\n        break_long_words : bool, optional\n            If True, then words longer than width will be broken in order to ensure\n            that no lines are longer than width. If it is false, long words will\n            not be broken, and some lines may be longer than width (default: True).\n        break_on_hyphens : bool, optional\n            If True, wrapping will occur preferably on whitespace and right after\n            hyphens in compound words, as it is customary in English. If false,\n            only whitespaces will be considered as potentially good places for line\n            breaks, but you need to set break_long_words to false if you want truly\n            insecable words (default: True).\n\n        Returns\n        -------\n        Series or Index\n\n        Notes\n        -----\n        Internally, this method uses a :class:`textwrap.TextWrapper` instance with\n        default settings. To achieve behavior matching R's stringr library str_wrap\n        function, use the arguments:\n\n        - expand_tabs = False\n        - replace_whitespace = True\n        - drop_whitespace = True\n        - break_long_words = False\n        - break_on_hyphens = False\n\n        Examples\n        --------\n        >>> s = pd.Series(['line to be wrapped', 'another line to be wrapped'])\n        >>> s.str.wrap(12)\n        0             line to be\\nwrapped\n        1    another line\\nto be\\nwrapped\n        dtype: object\n        "
        result = self._array._str_wrap(width, **kwargs)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def get_dummies(self, sep='|'):
        '\n        Return DataFrame of dummy/indicator variables for Series.\n\n        Each string in Series is split by sep and returned as a DataFrame\n        of dummy/indicator variables.\n\n        Parameters\n        ----------\n        sep : str, default "|"\n            String to split on.\n\n        Returns\n        -------\n        DataFrame\n            Dummy variables corresponding to values of the Series.\n\n        See Also\n        --------\n        get_dummies : Convert categorical variable into dummy/indicator\n            variables.\n\n        Examples\n        --------\n        >>> pd.Series([\'a|b\', \'a\', \'a|c\']).str.get_dummies()\n        a  b  c\n        0  1  1  0\n        1  1  0  0\n        2  1  0  1\n\n        >>> pd.Series([\'a|b\', np.nan, \'a|c\']).str.get_dummies()\n        a  b  c\n        0  1  1  0\n        1  0  0  0\n        2  1  0  1\n        '
        (result, name) = self._array._str_get_dummies(sep)
        return self._wrap_result(result, name=name, expand=True, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def translate(self, table):
        '\n        Map all characters in the string through the given mapping table.\n\n        Equivalent to standard :meth:`str.translate`.\n\n        Parameters\n        ----------\n        table : dict\n            Table is a mapping of Unicode ordinals to Unicode ordinals, strings, or\n            None. Unmapped characters are left untouched.\n            Characters mapped to None are deleted. :meth:`str.maketrans` is a\n            helper function for making translation tables.\n\n        Returns\n        -------\n        Series or Index\n        '
        result = self._array._str_translate(table)
        return self._wrap_result(result)

    @forbid_nonstring_types(['bytes'])
    def count(self, pat, flags=0):
        "\n        Count occurrences of pattern in each string of the Series/Index.\n\n        This function is used to count the number of times a particular regex\n        pattern is repeated in each of the string elements of the\n        :class:`~pandas.Series`.\n\n        Parameters\n        ----------\n        pat : str\n            Valid regular expression.\n        flags : int, default 0, meaning no flags\n            Flags for the `re` module. For a complete list, `see here\n            <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.\n        **kwargs\n            For compatibility with other string methods. Not used.\n\n        Returns\n        -------\n        Series or Index\n            Same type as the calling object containing the integer counts.\n\n        See Also\n        --------\n        re : Standard library module for regular expressions.\n        str.count : Standard library version, without regular expression support.\n\n        Notes\n        -----\n        Some characters need to be escaped when passing in `pat`.\n        eg. ``'$'`` has a special meaning in regex and must be escaped when\n        finding this literal character.\n\n        Examples\n        --------\n        >>> s = pd.Series(['A', 'B', 'Aaba', 'Baca', np.nan, 'CABA', 'cat'])\n        >>> s.str.count('a')\n        0    0.0\n        1    0.0\n        2    2.0\n        3    2.0\n        4    NaN\n        5    0.0\n        6    1.0\n        dtype: float64\n\n        Escape ``'$'`` to find the literal dollar sign.\n\n        >>> s = pd.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])\n        >>> s.str.count('\\\\$')\n        0    1\n        1    0\n        2    1\n        3    2\n        4    2\n        5    0\n        dtype: int64\n\n        This is also available on Index\n\n        >>> pd.Index(['A', 'A', 'Aaba', 'cat']).str.count('a')\n        Int64Index([0, 0, 2, 1], dtype='int64')\n        "
        result = self._array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def startswith(self, pat, na=None):
        "\n        Test if the start of each string element matches a pattern.\n\n        Equivalent to :meth:`str.startswith`.\n\n        Parameters\n        ----------\n        pat : str\n            Character sequence. Regular expressions are not accepted.\n        na : object, default NaN\n            Object shown if element tested is not a string. The default depends\n            on dtype of the array. For object-dtype, ``numpy.nan`` is used.\n            For ``StringDtype``, ``pandas.NA`` is used.\n\n        Returns\n        -------\n        Series or Index of bool\n            A Series of booleans indicating whether the given pattern matches\n            the start of each string element.\n\n        See Also\n        --------\n        str.startswith : Python standard library string method.\n        Series.str.endswith : Same as startswith, but tests the end of string.\n        Series.str.contains : Tests if string element contains a pattern.\n\n        Examples\n        --------\n        >>> s = pd.Series(['bat', 'Bear', 'cat', np.nan])\n        >>> s\n        0     bat\n        1    Bear\n        2     cat\n        3     NaN\n        dtype: object\n\n        >>> s.str.startswith('b')\n        0     True\n        1    False\n        2    False\n        3      NaN\n        dtype: object\n\n        Specifying `na` to be `False` instead of `NaN`.\n\n        >>> s.str.startswith('b', na=False)\n        0     True\n        1    False\n        2    False\n        3    False\n        dtype: bool\n        "
        result = self._array._str_startswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def endswith(self, pat, na=None):
        "\n        Test if the end of each string element matches a pattern.\n\n        Equivalent to :meth:`str.endswith`.\n\n        Parameters\n        ----------\n        pat : str\n            Character sequence. Regular expressions are not accepted.\n        na : object, default NaN\n            Object shown if element tested is not a string. The default depends\n            on dtype of the array. For object-dtype, ``numpy.nan`` is used.\n            For ``StringDtype``, ``pandas.NA`` is used.\n\n        Returns\n        -------\n        Series or Index of bool\n            A Series of booleans indicating whether the given pattern matches\n            the end of each string element.\n\n        See Also\n        --------\n        str.endswith : Python standard library string method.\n        Series.str.startswith : Same as endswith, but tests the start of string.\n        Series.str.contains : Tests if string element contains a pattern.\n\n        Examples\n        --------\n        >>> s = pd.Series(['bat', 'bear', 'caT', np.nan])\n        >>> s\n        0     bat\n        1    bear\n        2     caT\n        3     NaN\n        dtype: object\n\n        >>> s.str.endswith('t')\n        0     True\n        1    False\n        2    False\n        3      NaN\n        dtype: object\n\n        Specifying `na` to be `False` instead of `NaN`.\n\n        >>> s.str.endswith('t', na=False)\n        0     True\n        1    False\n        2    False\n        3    False\n        dtype: bool\n        "
        result = self._array._str_endswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def findall(self, pat, flags=0):
        "\n        Find all occurrences of pattern or regular expression in the Series/Index.\n\n        Equivalent to applying :func:`re.findall` to all the elements in the\n        Series/Index.\n\n        Parameters\n        ----------\n        pat : str\n            Pattern or regular expression.\n        flags : int, default 0\n            Flags from ``re`` module, e.g. `re.IGNORECASE` (default is 0, which\n            means no flags).\n\n        Returns\n        -------\n        Series/Index of lists of strings\n            All non-overlapping matches of pattern or regular expression in each\n            string of this Series/Index.\n\n        See Also\n        --------\n        count : Count occurrences of pattern or regular expression in each string\n            of the Series/Index.\n        extractall : For each string in the Series, extract groups from all matches\n            of regular expression and return a DataFrame with one row for each\n            match and one column for each group.\n        re.findall : The equivalent ``re`` function to all non-overlapping matches\n            of pattern or regular expression in string, as a list of strings.\n\n        Examples\n        --------\n        >>> s = pd.Series(['Lion', 'Monkey', 'Rabbit'])\n\n        The search for the pattern 'Monkey' returns one match:\n\n        >>> s.str.findall('Monkey')\n        0          []\n        1    [Monkey]\n        2          []\n        dtype: object\n\n        On the other hand, the search for the pattern 'MONKEY' doesn't return any\n        match:\n\n        >>> s.str.findall('MONKEY')\n        0    []\n        1    []\n        2    []\n        dtype: object\n\n        Flags can be added to the pattern or regular expression. For instance,\n        to find the pattern 'MONKEY' ignoring the case:\n\n        >>> import re\n        >>> s.str.findall('MONKEY', flags=re.IGNORECASE)\n        0          []\n        1    [Monkey]\n        2          []\n        dtype: object\n\n        When the pattern matches more than one string in the Series, all matches\n        are returned:\n\n        >>> s.str.findall('on')\n        0    [on]\n        1    [on]\n        2      []\n        dtype: object\n\n        Regular expressions are supported too. For instance, the search for all the\n        strings ending with the word 'on' is shown next:\n\n        >>> s.str.findall('on$')\n        0    [on]\n        1      []\n        2      []\n        dtype: object\n\n        If the pattern is found more than once in the same string, then a list of\n        multiple strings is returned:\n\n        >>> s.str.findall('b')\n        0        []\n        1        []\n        2    [b, b]\n        dtype: object\n        "
        result = self._array._str_findall(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def extract(self, pat, flags=0, expand=True):
        "\n        Extract capture groups in the regex `pat` as columns in a DataFrame.\n\n        For each subject string in the Series, extract groups from the\n        first match of regular expression `pat`.\n\n        Parameters\n        ----------\n        pat : str\n            Regular expression pattern with capturing groups.\n        flags : int, default 0 (no flags)\n            Flags from the ``re`` module, e.g. ``re.IGNORECASE``, that\n            modify regular expression matching for things like case,\n            spaces, etc. For more details, see :mod:`re`.\n        expand : bool, default True\n            If True, return DataFrame with one column per capture group.\n            If False, return a Series/Index if there is one capture group\n            or DataFrame if there are multiple capture groups.\n\n        Returns\n        -------\n        DataFrame or Series or Index\n            A DataFrame with one row for each subject string, and one\n            column for each group. Any capture group names in regular\n            expression pat will be used for column names; otherwise\n            capture group numbers will be used. The dtype of each result\n            column is always object, even when no match is found. If\n            ``expand=False`` and pat has only one capture group, then\n            return a Series (if subject is a Series) or Index (if subject\n            is an Index).\n\n        See Also\n        --------\n        extractall : Returns all matches (not just the first match).\n\n        Examples\n        --------\n        A pattern with two groups will return a DataFrame with two columns.\n        Non-matches will be NaN.\n\n        >>> s = pd.Series(['a1', 'b2', 'c3'])\n        >>> s.str.extract(r'([ab])(\\d)')\n            0    1\n        0    a    1\n        1    b    2\n        2  NaN  NaN\n\n        A pattern may contain optional groups.\n\n        >>> s.str.extract(r'([ab])?(\\d)')\n            0  1\n        0    a  1\n        1    b  2\n        2  NaN  3\n\n        Named groups will become column names in the result.\n\n        >>> s.str.extract(r'(?P<letter>[ab])(?P<digit>\\d)')\n        letter digit\n        0      a     1\n        1      b     2\n        2    NaN   NaN\n\n        A pattern with one group will return a DataFrame with one column\n        if expand=True.\n\n        >>> s.str.extract(r'[ab](\\d)', expand=True)\n            0\n        0    1\n        1    2\n        2  NaN\n\n        A pattern with one group will return a Series if expand=False.\n\n        >>> s.str.extract(r'[ab](\\d)', expand=False)\n        0      1\n        1      2\n        2    NaN\n        dtype: object\n        "
        return str_extract(self, pat, flags, expand=expand)

    @forbid_nonstring_types(['bytes'])
    def extractall(self, pat, flags=0):
        '\n        Extract capture groups in the regex `pat` as columns in DataFrame.\n\n        For each subject string in the Series, extract groups from all\n        matches of regular expression pat. When each subject string in the\n        Series has exactly one match, extractall(pat).xs(0, level=\'match\')\n        is the same as extract(pat).\n\n        Parameters\n        ----------\n        pat : str\n            Regular expression pattern with capturing groups.\n        flags : int, default 0 (no flags)\n            A ``re`` module flag, for example ``re.IGNORECASE``. These allow\n            to modify regular expression matching for things like case, spaces,\n            etc. Multiple flags can be combined with the bitwise OR operator,\n            for example ``re.IGNORECASE | re.MULTILINE``.\n\n        Returns\n        -------\n        DataFrame\n            A ``DataFrame`` with one row for each match, and one column for each\n            group. Its rows have a ``MultiIndex`` with first levels that come from\n            the subject ``Series``. The last level is named \'match\' and indexes the\n            matches in each item of the ``Series``. Any capture group names in\n            regular expression pat will be used for column names; otherwise capture\n            group numbers will be used.\n\n        See Also\n        --------\n        extract : Returns first match only (not all matches).\n\n        Examples\n        --------\n        A pattern with one group will return a DataFrame with one column.\n        Indices with no matches will not appear in the result.\n\n        >>> s = pd.Series(["a1a2", "b1", "c1"], index=["A", "B", "C"])\n        >>> s.str.extractall(r"[ab](\\d)")\n                0\n        match\n        A 0      1\n        1      2\n        B 0      1\n\n        Capture group names are used for column names of the result.\n\n        >>> s.str.extractall(r"[ab](?P<digit>\\d)")\n                digit\n        match\n        A 0         1\n        1         2\n        B 0         1\n\n        A pattern with two groups will return a DataFrame with two columns.\n\n        >>> s.str.extractall(r"(?P<letter>[ab])(?P<digit>\\d)")\n                letter digit\n        match\n        A 0          a     1\n        1          a     2\n        B 0          b     1\n\n        Optional groups that do not match are NaN in the result.\n\n        >>> s.str.extractall(r"(?P<letter>[ab])?(?P<digit>\\d)")\n                letter digit\n        match\n        A 0          a     1\n        1          a     2\n        B 0          b     1\n        C 0        NaN     1\n        '
        return str_extractall(self._orig, pat, flags)
    _shared_docs['find'] = '\n    Return %(side)s indexes in each strings in the Series/Index.\n\n    Each of returned indexes corresponds to the position where the\n    substring is fully contained between [start:end]. Return -1 on\n    failure. Equivalent to standard :meth:`str.%(method)s`.\n\n    Parameters\n    ----------\n    sub : str\n        Substring being searched.\n    start : int\n        Left edge index.\n    end : int\n        Right edge index.\n\n    Returns\n    -------\n    Series or Index of int.\n\n    See Also\n    --------\n    %(also)s\n    '

    @Appender((_shared_docs['find'] % {'side': 'lowest', 'method': 'find', 'also': 'rfind : Return highest indexes in each strings.'}))
    @forbid_nonstring_types(['bytes'])
    def find(self, sub, start=0, end=None):
        if (not isinstance(sub, str)):
            msg = f'expected a string object, not {type(sub).__name__}'
            raise TypeError(msg)
        result = self._array._str_find(sub, start, end)
        return self._wrap_result(result, returns_string=False)

    @Appender((_shared_docs['find'] % {'side': 'highest', 'method': 'rfind', 'also': 'find : Return lowest indexes in each strings.'}))
    @forbid_nonstring_types(['bytes'])
    def rfind(self, sub, start=0, end=None):
        if (not isinstance(sub, str)):
            msg = f'expected a string object, not {type(sub).__name__}'
            raise TypeError(msg)
        result = self._array._str_rfind(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(['bytes'])
    def normalize(self, form):
        "\n        Return the Unicode normal form for the strings in the Series/Index.\n\n        For more information on the forms, see the\n        :func:`unicodedata.normalize`.\n\n        Parameters\n        ----------\n        form : {'NFC', 'NFKC', 'NFD', 'NFKD'}\n            Unicode form.\n\n        Returns\n        -------\n        normalized : Series/Index of objects\n        "
        result = self._array._str_normalize(form)
        return self._wrap_result(result)
    _shared_docs['index'] = '\n    Return %(side)s indexes in each string in Series/Index.\n\n    Each of the returned indexes corresponds to the position where the\n    substring is fully contained between [start:end]. This is the same\n    as ``str.%(similar)s`` except instead of returning -1, it raises a\n    ValueError when the substring is not found. Equivalent to standard\n    ``str.%(method)s``.\n\n    Parameters\n    ----------\n    sub : str\n        Substring being searched.\n    start : int\n        Left edge index.\n    end : int\n        Right edge index.\n\n    Returns\n    -------\n    Series or Index of object\n\n    See Also\n    --------\n    %(also)s\n    '

    @Appender((_shared_docs['index'] % {'side': 'lowest', 'similar': 'find', 'method': 'index', 'also': 'rindex : Return highest indexes in each strings.'}))
    @forbid_nonstring_types(['bytes'])
    def index(self, sub, start=0, end=None):
        if (not isinstance(sub, str)):
            msg = f'expected a string object, not {type(sub).__name__}'
            raise TypeError(msg)
        result = self._array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @Appender((_shared_docs['index'] % {'side': 'highest', 'similar': 'rfind', 'method': 'rindex', 'also': 'index : Return lowest indexes in each strings.'}))
    @forbid_nonstring_types(['bytes'])
    def rindex(self, sub, start=0, end=None):
        if (not isinstance(sub, str)):
            msg = f'expected a string object, not {type(sub).__name__}'
            raise TypeError(msg)
        result = self._array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def len(self):
        "\n        Compute the length of each element in the Series/Index.\n\n        The element may be a sequence (such as a string, tuple or list) or a collection\n        (such as a dictionary).\n\n        Returns\n        -------\n        Series or Index of int\n            A Series or Index of integer values indicating the length of each\n            element in the Series or Index.\n\n        See Also\n        --------\n        str.len : Python built-in function returning the length of an object.\n        Series.size : Returns the length of the Series.\n\n        Examples\n        --------\n        Returns the length (number of characters) in a string. Returns the\n        number of entries for dictionaries, lists or tuples.\n\n        >>> s = pd.Series(['dog',\n        ...                 '',\n        ...                 5,\n        ...                 {'foo' : 'bar'},\n        ...                 [2, 3, 5, 7],\n        ...                 ('one', 'two', 'three')])\n        >>> s\n        0                  dog\n        1\n        2                    5\n        3       {'foo': 'bar'}\n        4         [2, 3, 5, 7]\n        5    (one, two, three)\n        dtype: object\n        >>> s.str.len()\n        0    3.0\n        1    0.0\n        2    NaN\n        3    1.0\n        4    4.0\n        5    3.0\n        dtype: float64\n        "
        result = self._array._str_len()
        return self._wrap_result(result, returns_string=False)
    _shared_docs['casemethods'] = "\n    Convert strings in the Series/Index to %(type)s.\n    %(version)s\n    Equivalent to :meth:`str.%(method)s`.\n\n    Returns\n    -------\n    Series or Index of object\n\n    See Also\n    --------\n    Series.str.lower : Converts all characters to lowercase.\n    Series.str.upper : Converts all characters to uppercase.\n    Series.str.title : Converts first character of each word to uppercase and\n        remaining to lowercase.\n    Series.str.capitalize : Converts first character to uppercase and\n        remaining to lowercase.\n    Series.str.swapcase : Converts uppercase to lowercase and lowercase to\n        uppercase.\n    Series.str.casefold: Removes all case distinctions in the string.\n\n    Examples\n    --------\n    >>> s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])\n    >>> s\n    0                 lower\n    1              CAPITALS\n    2    this is a sentence\n    3              SwApCaSe\n    dtype: object\n\n    >>> s.str.lower()\n    0                 lower\n    1              capitals\n    2    this is a sentence\n    3              swapcase\n    dtype: object\n\n    >>> s.str.upper()\n    0                 LOWER\n    1              CAPITALS\n    2    THIS IS A SENTENCE\n    3              SWAPCASE\n    dtype: object\n\n    >>> s.str.title()\n    0                 Lower\n    1              Capitals\n    2    This Is A Sentence\n    3              Swapcase\n    dtype: object\n\n    >>> s.str.capitalize()\n    0                 Lower\n    1              Capitals\n    2    This is a sentence\n    3              Swapcase\n    dtype: object\n\n    >>> s.str.swapcase()\n    0                 LOWER\n    1              capitals\n    2    THIS IS A SENTENCE\n    3              sWaPcAsE\n    dtype: object\n    "
    _doc_args = {}
    _doc_args['lower'] = {'type': 'lowercase', 'method': 'lower', 'version': ''}
    _doc_args['upper'] = {'type': 'uppercase', 'method': 'upper', 'version': ''}
    _doc_args['title'] = {'type': 'titlecase', 'method': 'title', 'version': ''}
    _doc_args['capitalize'] = {'type': 'be capitalized', 'method': 'capitalize', 'version': ''}
    _doc_args['swapcase'] = {'type': 'be swapcased', 'method': 'swapcase', 'version': ''}
    _doc_args['casefold'] = {'type': 'be casefolded', 'method': 'casefold', 'version': '\n    .. versionadded:: 0.25.0\n'}

    @Appender((_shared_docs['casemethods'] % _doc_args['lower']))
    @forbid_nonstring_types(['bytes'])
    def lower(self):
        result = self._array._str_lower()
        return self._wrap_result(result)

    @Appender((_shared_docs['casemethods'] % _doc_args['upper']))
    @forbid_nonstring_types(['bytes'])
    def upper(self):
        result = self._array._str_upper()
        return self._wrap_result(result)

    @Appender((_shared_docs['casemethods'] % _doc_args['title']))
    @forbid_nonstring_types(['bytes'])
    def title(self):
        result = self._array._str_title()
        return self._wrap_result(result)

    @Appender((_shared_docs['casemethods'] % _doc_args['capitalize']))
    @forbid_nonstring_types(['bytes'])
    def capitalize(self):
        result = self._array._str_capitalize()
        return self._wrap_result(result)

    @Appender((_shared_docs['casemethods'] % _doc_args['swapcase']))
    @forbid_nonstring_types(['bytes'])
    def swapcase(self):
        result = self._array._str_swapcase()
        return self._wrap_result(result)

    @Appender((_shared_docs['casemethods'] % _doc_args['casefold']))
    @forbid_nonstring_types(['bytes'])
    def casefold(self):
        result = self._array._str_casefold()
        return self._wrap_result(result)
    _shared_docs['ismethods'] = "\n    Check whether all characters in each string are %(type)s.\n\n    This is equivalent to running the Python string method\n    :meth:`str.%(method)s` for each element of the Series/Index. If a string\n    has zero characters, ``False`` is returned for that check.\n\n    Returns\n    -------\n    Series or Index of bool\n        Series or Index of boolean values with the same length as the original\n        Series/Index.\n\n    See Also\n    --------\n    Series.str.isalpha : Check whether all characters are alphabetic.\n    Series.str.isnumeric : Check whether all characters are numeric.\n    Series.str.isalnum : Check whether all characters are alphanumeric.\n    Series.str.isdigit : Check whether all characters are digits.\n    Series.str.isdecimal : Check whether all characters are decimal.\n    Series.str.isspace : Check whether all characters are whitespace.\n    Series.str.islower : Check whether all characters are lowercase.\n    Series.str.isupper : Check whether all characters are uppercase.\n    Series.str.istitle : Check whether all characters are titlecase.\n\n    Examples\n    --------\n    **Checks for Alphabetic and Numeric Characters**\n\n    >>> s1 = pd.Series(['one', 'one1', '1', ''])\n\n    >>> s1.str.isalpha()\n    0     True\n    1    False\n    2    False\n    3    False\n    dtype: bool\n\n    >>> s1.str.isnumeric()\n    0    False\n    1    False\n    2     True\n    3    False\n    dtype: bool\n\n    >>> s1.str.isalnum()\n    0     True\n    1     True\n    2     True\n    3    False\n    dtype: bool\n\n    Note that checks against characters mixed with any additional punctuation\n    or whitespace will evaluate to false for an alphanumeric check.\n\n    >>> s2 = pd.Series(['A B', '1.5', '3,000'])\n    >>> s2.str.isalnum()\n    0    False\n    1    False\n    2    False\n    dtype: bool\n\n    **More Detailed Checks for Numeric Characters**\n\n    There are several different but overlapping sets of numeric characters that\n    can be checked for.\n\n    >>> s3 = pd.Series(['23', '³', '⅕', ''])\n\n    The ``s3.str.isdecimal`` method checks for characters used to form numbers\n    in base 10.\n\n    >>> s3.str.isdecimal()\n    0     True\n    1    False\n    2    False\n    3    False\n    dtype: bool\n\n    The ``s.str.isdigit`` method is the same as ``s3.str.isdecimal`` but also\n    includes special digits, like superscripted and subscripted digits in\n    unicode.\n\n    >>> s3.str.isdigit()\n    0     True\n    1     True\n    2    False\n    3    False\n    dtype: bool\n\n    The ``s.str.isnumeric`` method is the same as ``s3.str.isdigit`` but also\n    includes other characters that can represent quantities such as unicode\n    fractions.\n\n    >>> s3.str.isnumeric()\n    0     True\n    1     True\n    2     True\n    3    False\n    dtype: bool\n\n    **Checks for Whitespace**\n\n    >>> s4 = pd.Series([' ', '\\t\\r\\n ', ''])\n    >>> s4.str.isspace()\n    0     True\n    1     True\n    2    False\n    dtype: bool\n\n    **Checks for Character Case**\n\n    >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])\n\n    >>> s5.str.islower()\n    0     True\n    1    False\n    2    False\n    3    False\n    dtype: bool\n\n    >>> s5.str.isupper()\n    0    False\n    1    False\n    2     True\n    3    False\n    dtype: bool\n\n    The ``s5.str.istitle`` method checks for whether all words are in title\n    case (whether only the first letter of each word is capitalized). Words are\n    assumed to be as any sequence of non-numeric characters separated by\n    whitespace characters.\n\n    >>> s5.str.istitle()\n    0    False\n    1     True\n    2    False\n    3    False\n    dtype: bool\n    "
    _doc_args['isalnum'] = {'type': 'alphanumeric', 'method': 'isalnum'}
    _doc_args['isalpha'] = {'type': 'alphabetic', 'method': 'isalpha'}
    _doc_args['isdigit'] = {'type': 'digits', 'method': 'isdigit'}
    _doc_args['isspace'] = {'type': 'whitespace', 'method': 'isspace'}
    _doc_args['islower'] = {'type': 'lowercase', 'method': 'islower'}
    _doc_args['isupper'] = {'type': 'uppercase', 'method': 'isupper'}
    _doc_args['istitle'] = {'type': 'titlecase', 'method': 'istitle'}
    _doc_args['isnumeric'] = {'type': 'numeric', 'method': 'isnumeric'}
    _doc_args['isdecimal'] = {'type': 'decimal', 'method': 'isdecimal'}
    isalnum = _map_and_wrap('isalnum', docstring=(_shared_docs['ismethods'] % _doc_args['isalnum']))
    isalpha = _map_and_wrap('isalpha', docstring=(_shared_docs['ismethods'] % _doc_args['isalpha']))
    isdigit = _map_and_wrap('isdigit', docstring=(_shared_docs['ismethods'] % _doc_args['isdigit']))
    isspace = _map_and_wrap('isspace', docstring=(_shared_docs['ismethods'] % _doc_args['isalnum']))
    islower = _map_and_wrap('islower', docstring=(_shared_docs['ismethods'] % _doc_args['islower']))
    isupper = _map_and_wrap('isupper', docstring=(_shared_docs['ismethods'] % _doc_args['isupper']))
    istitle = _map_and_wrap('istitle', docstring=(_shared_docs['ismethods'] % _doc_args['istitle']))
    isnumeric = _map_and_wrap('isnumeric', docstring=(_shared_docs['ismethods'] % _doc_args['isnumeric']))
    isdecimal = _map_and_wrap('isdecimal', docstring=(_shared_docs['ismethods'] % _doc_args['isdecimal']))

def cat_safe(list_of_columns, sep):
    '\n    Auxiliary function for :meth:`str.cat`.\n\n    Same signature as cat_core, but handles TypeErrors in concatenation, which\n    happen if the arrays in list_of columns have the wrong dtypes or content.\n\n    Parameters\n    ----------\n    list_of_columns : list of numpy arrays\n        List of arrays to be concatenated with sep;\n        these arrays may not contain NaNs!\n    sep : string\n        The separator string for concatenating the columns.\n\n    Returns\n    -------\n    nd.array\n        The concatenation of list_of_columns with sep.\n    '
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if (dtype not in ['string', 'empty']):
                raise TypeError(f'Concatenation requires list-likes containing only strings (or missing values). Offending values found in column {dtype}') from None
    return result

def cat_core(list_of_columns, sep):
    '\n    Auxiliary function for :meth:`str.cat`\n\n    Parameters\n    ----------\n    list_of_columns : list of numpy arrays\n        List of arrays to be concatenated with sep;\n        these arrays may not contain NaNs!\n    sep : string\n        The separator string for concatenating the columns.\n\n    Returns\n    -------\n    nd.array\n        The concatenation of list_of_columns with sep.\n    '
    if (sep == ''):
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = ([sep] * ((2 * len(list_of_columns)) - 1))
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)

def _groups_or_na_fun(regex):
    'Used in both extract_noexpand and extract_frame'
    if (regex.groups == 0):
        raise ValueError('pattern contains no capture groups')
    empty_row = ([np.nan] * regex.groups)

    def f(x):
        if (not isinstance(x, str)):
            return empty_row
        m = regex.search(x)
        if m:
            return [(np.nan if (item is None) else item) for item in m.groups()]
        else:
            return empty_row
    return f

def _result_dtype(arr):
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, StringDtype):
        return arr.dtype.name
    else:
        return object

def _get_single_group_name(rx):
    try:
        return list(rx.groupindex.keys()).pop()
    except IndexError:
        return None

def _str_extract_noexpand(arr, pat, flags=0):
    '\n    Find groups in each string in the Series using passed regular\n    expression. This function is called from\n    str_extract(expand=False), and can return Series, DataFrame, or\n    Index.\n\n    '
    from pandas import DataFrame, array
    regex = re.compile(pat, flags=flags)
    groups_or_na = _groups_or_na_fun(regex)
    result_dtype = _result_dtype(arr)
    if (regex.groups == 1):
        result = np.array([groups_or_na(val)[0] for val in arr], dtype=object)
        name = _get_single_group_name(regex)
        result = array(result, dtype=result_dtype)
    else:
        if isinstance(arr, ABCIndex):
            raise ValueError('only one regex group is supported with Index')
        name = None
        names = dict(zip(regex.groupindex.values(), regex.groupindex.keys()))
        columns = [names.get((1 + i), i) for i in range(regex.groups)]
        if (arr.size == 0):
            result = DataFrame(columns=columns, dtype=object)
        else:
            dtype = _result_dtype(arr)
            result = DataFrame([groups_or_na(val) for val in arr], columns=columns, index=arr.index, dtype=dtype)
    return (result, name)

def _str_extract_frame(arr, pat, flags=0):
    '\n    For each subject string in the Series, extract groups from the\n    first match of regular expression pat. This function is called from\n    str_extract(expand=True), and always returns a DataFrame.\n\n    '
    from pandas import DataFrame
    regex = re.compile(pat, flags=flags)
    groups_or_na = _groups_or_na_fun(regex)
    names = dict(zip(regex.groupindex.values(), regex.groupindex.keys()))
    columns = [names.get((1 + i), i) for i in range(regex.groups)]
    if (len(arr) == 0):
        return DataFrame(columns=columns, dtype=object)
    try:
        result_index = arr.index
    except AttributeError:
        result_index = None
    dtype = _result_dtype(arr)
    return DataFrame([groups_or_na(val) for val in arr], columns=columns, index=result_index, dtype=dtype)

def str_extract(arr, pat, flags=0, expand=True):
    if (not isinstance(expand, bool)):
        raise ValueError('expand must be True or False')
    if expand:
        result = _str_extract_frame(arr._orig, pat, flags=flags)
        return result.__finalize__(arr._orig, method='str_extract')
    else:
        (result, name) = _str_extract_noexpand(arr._orig, pat, flags=flags)
        return arr._wrap_result(result, name=name, expand=expand)

def str_extractall(arr, pat, flags=0):
    regex = re.compile(pat, flags=flags)
    if (regex.groups == 0):
        raise ValueError('pattern contains no capture groups')
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True)
    names = dict(zip(regex.groupindex.values(), regex.groupindex.keys()))
    columns = [names.get((1 + i), i) for i in range(regex.groups)]
    match_list = []
    index_list = []
    is_mi = (arr.index.nlevels > 1)
    for (subject_key, subject) in arr.items():
        if isinstance(subject, str):
            if (not is_mi):
                subject_key = (subject_key,)
            for (match_i, match_tuple) in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = [(np.NaN if (group == '') else group) for group in match_tuple]
                match_list.append(na_tuple)
                result_key = tuple((subject_key + (match_i,)))
                index_list.append(result_key)
    from pandas import MultiIndex
    index = MultiIndex.from_tuples(index_list, names=(arr.index.names + ['match']))
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(match_list, index=index, columns=columns, dtype=dtype)
    return result
