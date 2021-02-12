
'\naggregation.py contains utility functions to handle multiple named and lambda\nkwarg aggregations in groupby and DataFrame/Series aggregation\n'
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, AggObjType, Axis, FrameOrSeries, FrameOrSeriesUnion, Label
from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import is_dict_like, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCNDFrame, ABCSeries
from pandas.core.base import DataError, SpecificationError
import pandas.core.common as com
from pandas.core.indexes.api import Index
if TYPE_CHECKING:
    from pandas.core.series import Series

def reconstruct_func(func, **kwargs):
    '\n    This is the internal function to reconstruct func given if there is relabeling\n    or not and also normalize the keyword to get new order of columns.\n\n    If named aggregation is applied, `func` will be None, and kwargs contains the\n    column and aggregation function information to be parsed;\n    If named aggregation is not applied, `func` is either string (e.g. \'min\') or\n    Callable, or list of them (e.g. [\'min\', np.max]), or the dictionary of column name\n    and str/Callable/list of them (e.g. {\'A\': \'min\'}, or {\'A\': [np.min, lambda x: x]})\n\n    If relabeling is True, will return relabeling, reconstructed func, column\n    names, and the reconstructed order of columns.\n    If relabeling is False, the columns and order will be None.\n\n    Parameters\n    ----------\n    func: agg function (e.g. \'min\' or Callable) or list of agg functions\n        (e.g. [\'min\', np.max]) or dictionary (e.g. {\'A\': [\'min\', np.max]}).\n    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and\n        normalize_keyword_aggregation function for relabelling\n\n    Returns\n    -------\n    relabelling: bool, if there is relabelling or not\n    func: normalized and mangled func\n    columns: list of column names\n    order: list of columns indices\n\n    Examples\n    --------\n    >>> reconstruct_func(None, **{"foo": ("col", "min")})\n    (True, defaultdict(<class \'list\'>, {\'col\': [\'min\']}), (\'foo\',), array([0]))\n\n    >>> reconstruct_func("min")\n    (False, \'min\', None, None)\n    '
    relabeling = ((func is None) and is_multi_agg_with_relabel(**kwargs))
    columns: Optional[List[str]] = None
    order: Optional[List[int]] = None
    if (not relabeling):
        if (isinstance(func, list) and (len(func) > len(set(func)))):
            raise SpecificationError('Function names must be unique if there is no new column names assigned')
        elif (func is None):
            raise TypeError("Must provide 'func' or tuples of '(column, aggfunc).")
    if relabeling:
        (func, columns, order) = normalize_keyword_aggregation(kwargs)
    return (relabeling, func, columns, order)

def is_multi_agg_with_relabel(**kwargs):
    '\n    Check whether kwargs passed to .agg look like multi-agg with relabeling.\n\n    Parameters\n    ----------\n    **kwargs : dict\n\n    Returns\n    -------\n    bool\n\n    Examples\n    --------\n    >>> is_multi_agg_with_relabel(a="max")\n    False\n    >>> is_multi_agg_with_relabel(a_max=("a", "max"), a_min=("a", "min"))\n    True\n    >>> is_multi_agg_with_relabel()\n    False\n    '
    return (all(((isinstance(v, tuple) and (len(v) == 2)) for v in kwargs.values())) and (len(kwargs) > 0))

def normalize_keyword_aggregation(kwargs):
    '\n    Normalize user-provided "named aggregation" kwargs.\n    Transforms from the new ``Mapping[str, NamedAgg]`` style kwargs\n    to the old Dict[str, List[scalar]]].\n\n    Parameters\n    ----------\n    kwargs : dict\n\n    Returns\n    -------\n    aggspec : dict\n        The transformed kwargs.\n    columns : List[str]\n        The user-provided keys.\n    col_idx_order : List[int]\n        List of columns indices.\n\n    Examples\n    --------\n    >>> normalize_keyword_aggregation({"output": ("input", "sum")})\n    (defaultdict(<class \'list\'>, {\'input\': [\'sum\']}), (\'output\',), array([0]))\n    '
    aggspec: DefaultDict = defaultdict(list)
    order = []
    (columns, pairs) = list(zip(*kwargs.items()))
    for (name, (column, aggfunc)) in zip(columns, pairs):
        aggspec[column].append(aggfunc)
        order.append((column, (com.get_callable_name(aggfunc) or aggfunc)))
    uniquified_order = _make_unique_kwarg_list(order)
    aggspec_order = [(column, (com.get_callable_name(aggfunc) or aggfunc)) for (column, aggfuncs) in aggspec.items() for aggfunc in aggfuncs]
    uniquified_aggspec = _make_unique_kwarg_list(aggspec_order)
    col_idx_order = Index(uniquified_aggspec).get_indexer(uniquified_order)
    return (aggspec, columns, col_idx_order)

def _make_unique_kwarg_list(seq):
    "\n    Uniquify aggfunc name of the pairs in the order list\n\n    Examples:\n    --------\n    >>> kwarg_list = [('a', '<lambda>'), ('a', '<lambda>'), ('b', '<lambda>')]\n    >>> _make_unique_kwarg_list(kwarg_list)\n    [('a', '<lambda>_0'), ('a', '<lambda>_1'), ('b', '<lambda>')]\n    "
    return [((pair[0], '_'.join([pair[1], str(seq[:i].count(pair))])) if (seq.count(pair) > 1) else pair) for (i, pair) in enumerate(seq)]

def _managle_lambda_list(aggfuncs):
    '\n    Possibly mangle a list of aggfuncs.\n\n    Parameters\n    ----------\n    aggfuncs : Sequence\n\n    Returns\n    -------\n    mangled: list-like\n        A new AggSpec sequence, where lambdas have been converted\n        to have unique names.\n\n    Notes\n    -----\n    If just one aggfunc is passed, the name will not be mangled.\n    '
    if (len(aggfuncs) <= 1):
        return aggfuncs
    i = 0
    mangled_aggfuncs = []
    for aggfunc in aggfuncs:
        if (com.get_callable_name(aggfunc) == '<lambda>'):
            aggfunc = partial(aggfunc)
            aggfunc.__name__ = f'<lambda_{i}>'
            i += 1
        mangled_aggfuncs.append(aggfunc)
    return mangled_aggfuncs

def maybe_mangle_lambdas(agg_spec):
    "\n    Make new lambdas with unique names.\n\n    Parameters\n    ----------\n    agg_spec : Any\n        An argument to GroupBy.agg.\n        Non-dict-like `agg_spec` are pass through as is.\n        For dict-like `agg_spec` a new spec is returned\n        with name-mangled lambdas.\n\n    Returns\n    -------\n    mangled : Any\n        Same type as the input.\n\n    Examples\n    --------\n    >>> maybe_mangle_lambdas('sum')\n    'sum'\n    >>> maybe_mangle_lambdas([lambda: 1, lambda: 2])  # doctest: +SKIP\n    [<function __main__.<lambda_0>,\n     <function pandas...._make_lambda.<locals>.f(*args, **kwargs)>]\n    "
    is_dict = is_dict_like(agg_spec)
    if (not (is_dict or is_list_like(agg_spec))):
        return agg_spec
    mangled_aggspec = type(agg_spec)()
    if is_dict:
        for (key, aggfuncs) in agg_spec.items():
            if (is_list_like(aggfuncs) and (not is_dict_like(aggfuncs))):
                mangled_aggfuncs = _managle_lambda_list(aggfuncs)
            else:
                mangled_aggfuncs = aggfuncs
            mangled_aggspec[key] = mangled_aggfuncs
    else:
        mangled_aggspec = _managle_lambda_list(agg_spec)
    return mangled_aggspec

def relabel_result(result, func, columns, order):
    '\n    Internal function to reorder result if relabelling is True for\n    dataframe.agg, and return the reordered result in dict.\n\n    Parameters:\n    ----------\n    result: Result from aggregation\n    func: Dict of (column name, funcs)\n    columns: New columns name for relabelling\n    order: New order for relabelling\n\n    Examples:\n    ---------\n    >>> result = DataFrame({"A": [np.nan, 2, np.nan],\n    ...       "C": [6, np.nan, np.nan], "B": [np.nan, 4, 2.5]})  # doctest: +SKIP\n    >>> funcs = {"A": ["max"], "C": ["max"], "B": ["mean", "min"]}\n    >>> columns = ("foo", "aab", "bar", "dat")\n    >>> order = [0, 1, 2, 3]\n    >>> _relabel_result(result, func, columns, order)  # doctest: +SKIP\n    dict(A=Series([2.0, NaN, NaN, NaN], index=["foo", "aab", "bar", "dat"]),\n         C=Series([NaN, 6.0, NaN, NaN], index=["foo", "aab", "bar", "dat"]),\n         B=Series([NaN, NaN, 2.5, 4.0], index=["foo", "aab", "bar", "dat"]))\n    '
    reordered_indexes = [pair[0] for pair in sorted(zip(columns, order), key=(lambda t: t[1]))]
    reordered_result_in_dict: Dict[(Label, 'Series')] = {}
    idx = 0
    reorder_mask = ((not isinstance(result, ABCSeries)) and (len(result.columns) > 1))
    for (col, fun) in func.items():
        s = result[col].dropna()
        if reorder_mask:
            fun = [(com.get_callable_name(f) if (not isinstance(f, str)) else f) for f in fun]
            col_idx_order = Index(s.index).get_indexer(fun)
            s = s[col_idx_order]
        s.index = reordered_indexes[idx:(idx + len(fun))]
        reordered_result_in_dict[col] = s.reindex(columns, copy=False)
        idx = (idx + len(fun))
    return reordered_result_in_dict

def validate_func_kwargs(kwargs):
    '\n    Validates types of user-provided "named aggregation" kwargs.\n    `TypeError` is raised if aggfunc is not `str` or callable.\n\n    Parameters\n    ----------\n    kwargs : dict\n\n    Returns\n    -------\n    columns : List[str]\n        List of user-provied keys.\n    func : List[Union[str, callable[...,Any]]]\n        List of user-provided aggfuncs\n\n    Examples\n    --------\n    >>> validate_func_kwargs({\'one\': \'min\', \'two\': \'max\'})\n    ([\'one\', \'two\'], [\'min\', \'max\'])\n    '
    no_arg_message = "Must provide 'func' or named aggregation **kwargs."
    tuple_given_message = 'func is expected but received {} in **kwargs.'
    columns = list(kwargs)
    func = []
    for col_func in kwargs.values():
        if (not (isinstance(col_func, str) or callable(col_func))):
            raise TypeError(tuple_given_message.format(type(col_func).__name__))
        func.append(col_func)
    if (not columns):
        raise TypeError(no_arg_message)
    return (columns, func)

def transform(obj, func, axis, *args, **kwargs):
    "\n    Transform a DataFrame or Series\n\n    Parameters\n    ----------\n    obj : DataFrame or Series\n        Object to compute the transform on.\n    func : string, function, list, or dictionary\n        Function(s) to compute the transform with.\n    axis : {0 or 'index', 1 or 'columns'}\n        Axis along which the function is applied:\n\n        * 0 or 'index': apply function to each column.\n        * 1 or 'columns': apply function to each row.\n\n    Returns\n    -------\n    DataFrame or Series\n        Result of applying ``func`` along the given axis of the\n        Series or DataFrame.\n\n    Raises\n    ------\n    ValueError\n        If the transform function fails or does not transform.\n    "
    is_series = (obj.ndim == 1)
    if (obj._get_axis_number(axis) == 1):
        assert (not is_series)
        return transform(obj.T, func, 0, *args, **kwargs).T
    if (is_list_like(func) and (not is_dict_like(func))):
        func = cast(List[AggFuncTypeBase], func)
        if is_series:
            func = {(com.get_callable_name(v) or v): v for v in func}
        else:
            func = {col: func for col in obj}
    if is_dict_like(func):
        func = cast(AggFuncTypeDict, func)
        return transform_dict_like(obj, func, *args, **kwargs)
    func = cast(AggFuncTypeBase, func)
    try:
        result = transform_str_or_callable(obj, func, *args, **kwargs)
    except Exception:
        raise ValueError('Transform function failed')
    if (isinstance(result, (ABCSeries, ABCDataFrame)) and result.empty):
        raise ValueError('Transform function failed')
    if ((not isinstance(result, (ABCSeries, ABCDataFrame))) or (not result.index.equals(obj.index))):
        raise ValueError('Function did not transform')
    return result

def transform_dict_like(obj, func, *args, **kwargs):
    '\n    Compute transform in the case of a dict-like func\n    '
    from pandas.core.reshape.concat import concat
    if (len(func) == 0):
        raise ValueError('No transform functions were provided')
    if (obj.ndim != 1):
        cols = sorted((set(func.keys()) - set(obj.columns)))
        if (len(cols) > 0):
            raise SpecificationError(f'Column(s) {cols} do not exist')
    if any((is_dict_like(v) for (_, v) in func.items())):
        raise SpecificationError('nested renamer is not supported')
    results: Dict[(Label, FrameOrSeriesUnion)] = {}
    for (name, how) in func.items():
        colg = obj._gotitem(name, ndim=1)
        try:
            results[name] = transform(colg, how, 0, *args, **kwargs)
        except Exception as err:
            if ((str(err) == 'Function did not transform') or (str(err) == 'No transform functions were provided')):
                raise err
    if (len(results) == 0):
        raise ValueError('Transform function failed')
    return concat(results, axis=1)

def transform_str_or_callable(obj, func, *args, **kwargs):
    '\n    Compute transform in the case of a string or callable func\n    '
    if isinstance(func, str):
        return obj._try_aggregate_string_function(func, *args, **kwargs)
    if ((not args) and (not kwargs)):
        f = obj._get_cython_func(func)
        if f:
            return getattr(obj, f)()
    try:
        return obj.apply(func, args=args, **kwargs)
    except Exception:
        return func(obj, *args, **kwargs)

def aggregate(obj, arg, *args, **kwargs):
    '\n    Provide an implementation for the aggregators.\n\n    Parameters\n    ----------\n    obj : Pandas object to compute aggregation on.\n    arg : string, dict, function.\n    *args : args to pass on to the function.\n    **kwargs : kwargs to pass on to the function.\n\n    Returns\n    -------\n    tuple of result, how.\n\n    Notes\n    -----\n    how can be a string describe the required post-processing, or\n    None if not required.\n    '
    _axis = kwargs.pop('_axis', None)
    if (_axis is None):
        _axis = getattr(obj, 'axis', 0)
    if isinstance(arg, str):
        return (obj._try_aggregate_string_function(arg, *args, **kwargs), None)
    elif is_dict_like(arg):
        arg = cast(AggFuncTypeDict, arg)
        return (agg_dict_like(obj, arg, _axis), True)
    elif is_list_like(arg):
        arg = cast(List[AggFuncTypeBase], arg)
        return (agg_list_like(obj, arg, _axis=_axis), None)
    else:
        result = None
    if callable(arg):
        f = obj._get_cython_func(arg)
        if (f and (not args) and (not kwargs)):
            return (getattr(obj, f)(), None)
    return (result, True)

def agg_list_like(obj, arg, _axis):
    '\n    Compute aggregation in the case of a list-like argument.\n\n    Parameters\n    ----------\n    obj : Pandas object to compute aggregation on.\n    arg : list\n        Aggregations to compute.\n    _axis : int, 0 or 1\n        Axis to compute aggregation on.\n\n    Returns\n    -------\n    Result of aggregation.\n    '
    from pandas.core.reshape.concat import concat
    if (_axis != 0):
        raise NotImplementedError('axis other than 0 is not supported')
    if (obj._selected_obj.ndim == 1):
        selected_obj = obj._selected_obj
    else:
        selected_obj = obj._obj_with_exclusions
    results = []
    keys = []
    if (selected_obj.ndim == 1):
        for a in arg:
            colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
            try:
                new_res = colg.aggregate(a)
            except TypeError:
                pass
            else:
                results.append(new_res)
                name = (com.get_callable_name(a) or a)
                keys.append(name)
    else:
        for (index, col) in enumerate(selected_obj):
            colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
            try:
                new_res = colg.aggregate(arg)
            except (TypeError, DataError):
                pass
            except ValueError as err:
                if ('Must produce aggregated value' in str(err)):
                    pass
                elif ('no results' in str(err)):
                    pass
                else:
                    raise
            else:
                results.append(new_res)
                keys.append(col)
    if (not len(results)):
        raise ValueError('no results')
    try:
        return concat(results, keys=keys, axis=1, sort=False)
    except TypeError as err:
        from pandas import Series
        result = Series(results, index=keys, name=obj.name)
        if is_nested_object(result):
            raise ValueError('cannot combine transform and aggregation operations') from err
        return result

def agg_dict_like(obj, arg, _axis):
    '\n    Compute aggregation in the case of a dict-like argument.\n\n    Parameters\n    ----------\n    obj : Pandas object to compute aggregation on.\n    arg : dict\n        label-aggregation pairs to compute.\n    _axis : int, 0 or 1\n        Axis to compute aggregation on.\n\n    Returns\n    -------\n    Result of aggregation.\n    '
    is_aggregator = (lambda x: isinstance(x, (list, tuple, dict)))
    if (_axis != 0):
        raise ValueError('Can only pass dict with axis=0')
    selected_obj = obj._selected_obj
    if any((is_aggregator(x) for x in arg.values())):
        new_arg: AggFuncTypeDict = {}
        for (k, v) in arg.items():
            if (not isinstance(v, (tuple, list, dict))):
                new_arg[k] = [v]
            else:
                new_arg[k] = v
            if isinstance(v, dict):
                raise SpecificationError('nested renamer is not supported')
            elif isinstance(selected_obj, ABCSeries):
                raise SpecificationError('nested renamer is not supported')
            elif (isinstance(selected_obj, ABCDataFrame) and (k not in selected_obj.columns)):
                raise KeyError(f"Column '{k}' does not exist!")
        arg = new_arg
    else:
        keys = list(arg.keys())
        if (isinstance(selected_obj, ABCDataFrame) and (len(selected_obj.columns.intersection(keys)) != len(keys))):
            cols = sorted((set(keys) - set(selected_obj.columns.intersection(keys))))
            raise SpecificationError(f'Column(s) {cols} do not exist')
    from pandas.core.reshape.concat import concat
    if (selected_obj.ndim == 1):
        colg = obj._gotitem(obj._selection, ndim=1)
        results = {key: colg.agg(how) for (key, how) in arg.items()}
    else:
        results = {key: obj._gotitem(key, ndim=1).agg(how) for (key, how) in arg.items()}
    keys = list(arg.keys())
    is_ndframe = [isinstance(r, ABCNDFrame) for r in results.values()]
    if all(is_ndframe):
        keys_to_use = [k for k in keys if (not results[k].empty)]
        keys_to_use = (keys_to_use if (keys_to_use != []) else keys)
        axis = (0 if isinstance(obj, ABCSeries) else 1)
        result = concat({k: results[k] for k in keys_to_use}, axis=axis)
    elif any(is_ndframe):
        raise ValueError('cannot perform both aggregation and transformation operations simultaneously')
    else:
        from pandas import Series
        if (obj.ndim == 1):
            obj = cast('Series', obj)
            name = obj.name
        else:
            name = None
        result = Series(results, name=name)
    return result
