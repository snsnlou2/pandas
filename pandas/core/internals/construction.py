
'\nFunctions for preparing various inputs passed to the DataFrame or Series\nconstructors before passing them to a BlockManager.\n'
from collections import abc
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import numpy.ma as ma
from pandas._libs import lib
from pandas._typing import Axis, DtypeObj, Label, Scalar
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar, construct_1d_ndarray_preserving_na, dict_compat, maybe_cast_to_datetime, maybe_convert_platform, maybe_infer_to_datetimelike, maybe_upcast
from pandas.core.dtypes.common import is_categorical_dtype, is_datetime64tz_dtype, is_dtype_equal, is_extension_array_dtype, is_integer_dtype, is_list_like, is_named_tuple, is_object_dtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCDatetimeIndex, ABCIndex, ABCSeries, ABCTimedeltaIndex
from pandas.core import algorithms, common as com
from pandas.core.arrays import Categorical
from pandas.core.construction import extract_array, sanitize_array
from pandas.core.indexes import base as ibase
from pandas.core.indexes.api import Index, ensure_index, get_objs_combined_axis, union_indexes
from pandas.core.internals.managers import create_block_manager_from_arrays, create_block_manager_from_blocks
if TYPE_CHECKING:
    from numpy.ma.mrecords import MaskedRecords
    from pandas import Series

def arrays_to_mgr(arrays, arr_names, index, columns, dtype=None, verify_integrity=True):
    '\n    Segregate Series based on type and coerce into matrices.\n\n    Needs to handle a lot of exceptional cases.\n    '
    arr_names = ensure_index(arr_names)
    if verify_integrity:
        if (index is None):
            index = extract_index(arrays)
        else:
            index = ensure_index(index)
        arrays = _homogenize(arrays, index, dtype)
        columns = ensure_index(columns)
    else:
        columns = ensure_index(columns)
        index = ensure_index(index)
    axes = [columns, index]
    return create_block_manager_from_arrays(arrays, arr_names, axes)

def masked_rec_array_to_mgr(data, index, columns, dtype, copy):
    '\n    Extract from a masked rec array and create the manager.\n    '
    fdata = ma.getdata(data)
    if (index is None):
        index = _get_names_from_index(fdata)
        if (index is None):
            index = ibase.default_index(len(data))
    index = ensure_index(index)
    if (columns is not None):
        columns = ensure_index(columns)
    (arrays, arr_columns) = to_arrays(fdata, columns)
    new_arrays = []
    for col in arr_columns:
        arr = data[col]
        fv = arr.fill_value
        mask = ma.getmaskarray(arr)
        if mask.any():
            (arr, fv) = maybe_upcast(arr, fill_value=fv, copy=True)
            arr[mask] = fv
        new_arrays.append(arr)
    (arrays, arr_columns) = reorder_arrays(new_arrays, arr_columns, columns)
    if (columns is None):
        columns = arr_columns
    mgr = arrays_to_mgr(arrays, arr_columns, index, columns, dtype)
    if copy:
        mgr = mgr.copy()
    return mgr

def init_ndarray(values, index, columns, dtype, copy):
    if isinstance(values, ABCSeries):
        if (columns is None):
            if (values.name is not None):
                columns = [values.name]
        if (index is None):
            index = values.index
        else:
            values = values.reindex(index)
        if ((not len(values)) and (columns is not None) and len(columns)):
            values = np.empty((0, 1), dtype=object)
    if (is_categorical_dtype(getattr(values, 'dtype', None)) or is_categorical_dtype(dtype)):
        if (not hasattr(values, 'dtype')):
            values = _prep_ndarray(values, copy=copy)
            values = values.ravel()
        elif copy:
            values = values.copy()
        (index, columns) = _get_axes(len(values), 1, index, columns)
        return arrays_to_mgr([values], columns, index, columns, dtype=dtype)
    elif (is_extension_array_dtype(values) or is_extension_array_dtype(dtype)):
        if (isinstance(values, np.ndarray) and (values.ndim > 1)):
            values = [values[:, n] for n in range(values.shape[1])]
        else:
            values = [values]
        if (columns is None):
            columns = Index(range(len(values)))
        return arrays_to_mgr(values, columns, index, columns, dtype=dtype)
    values = _prep_ndarray(values, copy=copy)
    if ((dtype is not None) and (not is_dtype_equal(values.dtype, dtype))):
        try:
            values = construct_1d_ndarray_preserving_na(values.ravel(), dtype=dtype, copy=False).reshape(values.shape)
        except Exception as orig:
            raise ValueError(f"failed to cast to '{dtype}' (Exception was: {orig})") from orig
    (index, columns) = _get_axes(values.shape[0], values.shape[1], index=index, columns=columns)
    values = values.T
    if ((dtype is None) and is_object_dtype(values.dtype)):
        if ((values.ndim == 2) and (values.shape[0] != 1)):
            dvals_list = [maybe_infer_to_datetimelike(row) for row in values]
            for n in range(len(dvals_list)):
                if isinstance(dvals_list[n], np.ndarray):
                    dvals_list[n] = dvals_list[n].reshape(1, (- 1))
            from pandas.core.internals.blocks import make_block
            block_values = [make_block(dvals_list[n], placement=[n], ndim=2) for n in range(len(dvals_list))]
        else:
            datelike_vals = maybe_infer_to_datetimelike(values)
            block_values = [datelike_vals]
    else:
        block_values = [values]
    return create_block_manager_from_blocks(block_values, [columns, index])

def init_dict(data, index, columns, dtype=None):
    '\n    Segregate Series based on type and coerce into matrices.\n    Needs to handle a lot of exceptional cases.\n    '
    arrays: Union[(Sequence[Any], 'Series')]
    if (columns is not None):
        from pandas.core.series import Series
        arrays = Series(data, index=columns, dtype=object)
        data_names = arrays.index
        missing = arrays.isna()
        if (index is None):
            index = extract_index(arrays[(~ missing)])
        else:
            index = ensure_index(index)
        if (missing.any() and (not is_integer_dtype(dtype))):
            if ((dtype is None) or ((not is_extension_array_dtype(dtype)) and np.issubdtype(dtype, np.flexible))):
                nan_dtype = np.dtype(object)
            else:
                nan_dtype = dtype
            val = construct_1d_arraylike_from_scalar(np.nan, len(index), nan_dtype)
            arrays.loc[missing] = ([val] * missing.sum())
    else:
        keys = list(data.keys())
        columns = data_names = Index(keys)
        arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]
        arrays = [(arr if (not isinstance(arr, ABCIndex)) else arr._data) for arr in arrays]
        arrays = [(arr if (not is_datetime64tz_dtype(arr)) else arr.copy()) for arr in arrays]
    return arrays_to_mgr(arrays, data_names, index, columns, dtype=dtype)

def nested_data_to_arrays(data, columns, index, dtype):
    '\n    Convert a single sequence of arrays to multiple arrays.\n    '
    if (is_named_tuple(data[0]) and (columns is None)):
        columns = data[0]._fields
    (arrays, columns) = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)
    if (index is None):
        if isinstance(data[0], ABCSeries):
            index = _get_names_from_index(data)
        elif isinstance(data[0], Categorical):
            index = ibase.default_index(len(data[0]))
        else:
            index = ibase.default_index(len(data))
    return (arrays, columns, index)

def treat_as_nested(data):
    '\n    Check if we should use nested_data_to_arrays.\n    '
    return ((len(data) > 0) and is_list_like(data[0]) and (getattr(data[0], 'ndim', 1) == 1))

def _prep_ndarray(values, copy=True):
    if (not isinstance(values, (np.ndarray, ABCSeries, Index))):
        if (len(values) == 0):
            return np.empty((0, 0), dtype=object)
        elif isinstance(values, range):
            arr = np.arange(values.start, values.stop, values.step, dtype='int64')
            return arr[(..., np.newaxis)]

        def convert(v):
            return maybe_convert_platform(v)
        try:
            if (is_list_like(values[0]) or hasattr(values[0], 'len')):
                values = np.array([convert(v) for v in values])
            elif (isinstance(values[0], np.ndarray) and (values[0].ndim == 0)):
                values = np.array([convert(v) for v in values])
            else:
                values = convert(values)
        except (ValueError, TypeError):
            values = convert(values)
    else:
        values = np.asarray(values)
        if copy:
            values = values.copy()
    if (values.ndim == 1):
        values = values.reshape((values.shape[0], 1))
    elif (values.ndim != 2):
        raise ValueError(f'Must pass 2-d input. shape={values.shape}')
    return values

def _homogenize(data, index, dtype):
    oindex = None
    homogenized = []
    for val in data:
        if isinstance(val, ABCSeries):
            if (dtype is not None):
                val = val.astype(dtype)
            if (val.index is not index):
                val = val.reindex(index, copy=False)
        else:
            if isinstance(val, dict):
                if (oindex is None):
                    oindex = index.astype('O')
                if isinstance(index, (ABCDatetimeIndex, ABCTimedeltaIndex)):
                    val = dict_compat(val)
                else:
                    val = dict(val)
                val = lib.fast_multiget(val, oindex._values, default=np.nan)
            val = sanitize_array(val, index, dtype=dtype, copy=False, raise_cast_failure=False)
        homogenized.append(val)
    return homogenized

def extract_index(data):
    '\n    Try to infer an Index from the passed data, raise ValueError on failure.\n    '
    index = None
    if (len(data) == 0):
        index = Index([])
    elif (len(data) > 0):
        raw_lengths = []
        indexes: List[Union[(List[Label], Index)]] = []
        have_raw_arrays = False
        have_series = False
        have_dicts = False
        for val in data:
            if isinstance(val, ABCSeries):
                have_series = True
                indexes.append(val.index)
            elif isinstance(val, dict):
                have_dicts = True
                indexes.append(list(val.keys()))
            elif (is_list_like(val) and (getattr(val, 'ndim', 1) == 1)):
                have_raw_arrays = True
                raw_lengths.append(len(val))
        if ((not indexes) and (not raw_lengths)):
            raise ValueError('If using all scalar values, you must pass an index')
        if have_series:
            index = union_indexes(indexes)
        elif have_dicts:
            index = union_indexes(indexes, sort=False)
        if have_raw_arrays:
            lengths = list(set(raw_lengths))
            if (len(lengths) > 1):
                raise ValueError('All arrays must be of the same length')
            if have_dicts:
                raise ValueError('Mixing dicts with non-Series may lead to ambiguous ordering.')
            if have_series:
                assert (index is not None)
                if (lengths[0] != len(index)):
                    msg = f'array length {lengths[0]} does not match index length {len(index)}'
                    raise ValueError(msg)
            else:
                index = ibase.default_index(lengths[0])
    return ensure_index(index)

def reorder_arrays(arrays, arr_columns, columns):
    if ((columns is not None) and len(columns) and (arr_columns is not None) and len(arr_columns)):
        indexer = ensure_index(arr_columns).get_indexer(columns)
        arr_columns = ensure_index([arr_columns[i] for i in indexer])
        arrays = [arrays[i] for i in indexer]
    return (arrays, arr_columns)

def _get_names_from_index(data):
    has_some_name = any(((getattr(s, 'name', None) is not None) for s in data))
    if (not has_some_name):
        return ibase.default_index(len(data))
    index: List[Label] = list(range(len(data)))
    count = 0
    for (i, s) in enumerate(data):
        n = getattr(s, 'name', None)
        if (n is not None):
            index[i] = n
        else:
            index[i] = f'Unnamed {count}'
            count += 1
    return index

def _get_axes(N, K, index, columns):
    if (index is None):
        index = ibase.default_index(N)
    else:
        index = ensure_index(index)
    if (columns is None):
        columns = ibase.default_index(K)
    else:
        columns = ensure_index(columns)
    return (index, columns)

def dataclasses_to_dicts(data):
    '\n    Converts a list of dataclass instances to a list of dictionaries.\n\n    Parameters\n    ----------\n    data : List[Type[dataclass]]\n\n    Returns\n    --------\n    list_dict : List[dict]\n\n    Examples\n    --------\n    >>> @dataclass\n    >>> class Point:\n    ...     x: int\n    ...     y: int\n\n    >>> dataclasses_to_dicts([Point(1,2), Point(2,3)])\n    [{"x":1,"y":2},{"x":2,"y":3}]\n\n    '
    from dataclasses import asdict
    return list(map(asdict, data))

def to_arrays(data, columns, dtype=None):
    '\n    Return list of arrays, columns.\n    '
    if isinstance(data, ABCDataFrame):
        if (columns is not None):
            arrays = [data._ixs(i, axis=1).values for (i, col) in enumerate(data.columns) if (col in columns)]
        else:
            columns = data.columns
            arrays = [data._ixs(i, axis=1).values for i in range(len(columns))]
        return (arrays, columns)
    if (not len(data)):
        if isinstance(data, np.ndarray):
            columns = data.dtype.names
            if (columns is not None):
                return (([[]] * len(columns)), columns)
        return ([], [])
    elif isinstance(data[0], Categorical):
        if (columns is None):
            columns = ibase.default_index(len(data))
        return (data, columns)
    elif (isinstance(data, np.ndarray) and (data.dtype.names is not None)):
        columns = list(data.dtype.names)
        arrays = [data[k] for k in columns]
        return (arrays, columns)
    if isinstance(data[0], (list, tuple)):
        (content, columns) = _list_to_arrays(data, columns)
    elif isinstance(data[0], abc.Mapping):
        (content, columns) = _list_of_dict_to_arrays(data, columns)
    elif isinstance(data[0], ABCSeries):
        (content, columns) = _list_of_series_to_arrays(data, columns)
    else:
        data = [tuple(x) for x in data]
        (content, columns) = _list_to_arrays(data, columns)
    (content, columns) = _finalize_columns_and_data(content, columns, dtype)
    return (content, columns)

def _list_to_arrays(data, columns):
    if isinstance(data[0], tuple):
        content = lib.to_object_array_tuples(data)
    else:
        content = lib.to_object_array(data)
    return (content, columns)

def _list_of_series_to_arrays(data, columns):
    if (columns is None):
        pass_data = [x for x in data if isinstance(x, (ABCSeries, ABCDataFrame))]
        columns = get_objs_combined_axis(pass_data, sort=False)
    indexer_cache: Dict[(int, Scalar)] = {}
    aligned_values = []
    for s in data:
        index = getattr(s, 'index', None)
        if (index is None):
            index = ibase.default_index(len(s))
        if (id(index) in indexer_cache):
            indexer = indexer_cache[id(index)]
        else:
            indexer = indexer_cache[id(index)] = index.get_indexer(columns)
        values = extract_array(s, extract_numpy=True)
        aligned_values.append(algorithms.take_1d(values, indexer))
    content = np.vstack(aligned_values)
    return (content, columns)

def _list_of_dict_to_arrays(data, columns):
    '\n    Convert list of dicts to numpy arrays\n\n    if `columns` is not passed, column names are inferred from the records\n    - for OrderedDict and dicts, the column names match\n      the key insertion-order from the first record to the last.\n    - For other kinds of dict-likes, the keys are lexically sorted.\n\n    Parameters\n    ----------\n    data : iterable\n        collection of records (OrderedDict, dict)\n    columns: iterables or None\n\n    Returns\n    -------\n    tuple\n        arrays, columns\n    '
    if (columns is None):
        gen = (list(x.keys()) for x in data)
        sort = (not any((isinstance(d, dict) for d in data)))
        columns = lib.fast_unique_multiple_list_gen(gen, sort=sort)
    data = [(((type(d) is dict) and d) or dict(d)) for d in data]
    content = lib.dicts_to_array(data, list(columns))
    return (content, columns)

def _finalize_columns_and_data(content, columns, dtype):
    '\n    Ensure we have valid columns, cast object dtypes if possible.\n    '
    content = list(content.T)
    try:
        columns = _validate_or_indexify_columns(content, columns)
    except AssertionError as err:
        raise ValueError(err) from err
    if (len(content) and (content[0].dtype == np.object_)):
        content = _convert_object_array(content, dtype=dtype)
    return (content, columns)

def _validate_or_indexify_columns(content, columns):
    '\n    If columns is None, make numbers as column names; Otherwise, validate that\n    columns have valid length.\n\n    Parameters\n    ----------\n    content: list of data\n    columns: Iterable or None\n\n    Returns\n    -------\n    columns: If columns is Iterable, return as is; If columns is None, assign\n    positional column index value as columns.\n\n    Raises\n    ------\n    1. AssertionError when content is not composed of list of lists, and if\n        length of columns is not equal to length of content.\n    2. ValueError when content is list of lists, but length of each sub-list\n        is not equal\n    3. ValueError when content is list of lists, but length of sub-list is\n        not equal to length of content\n    '
    if (columns is None):
        columns = ibase.default_index(len(content))
    else:
        is_mi_list = (isinstance(columns, list) and all((isinstance(col, list) for col in columns)))
        if ((not is_mi_list) and (len(columns) != len(content))):
            raise AssertionError(f'{len(columns)} columns passed, passed data had {len(content)} columns')
        elif is_mi_list:
            if (len({len(col) for col in columns}) > 1):
                raise ValueError('Length of columns passed for MultiIndex columns is different')
            elif (columns and (len(columns[0]) != len(content))):
                raise ValueError(f'{len(columns[0])} columns passed, passed data had {len(content)} columns')
    return columns

def _convert_object_array(content, dtype=None):
    '\n    Internal function to convert object array.\n\n    Parameters\n    ----------\n    content: list of processed data records\n    dtype: np.dtype, default is None\n\n    Returns\n    -------\n    arrays: casted content if not object dtype, otherwise return as is in list.\n    '

    def convert(arr):
        if (dtype != np.dtype('O')):
            arr = lib.maybe_convert_objects(arr)
            arr = maybe_cast_to_datetime(arr, dtype)
        return arr
    arrays = [convert(arr) for arr in content]
    return arrays

def sanitize_index(data, index):
    '\n    Sanitize an index type to return an ndarray of the underlying, pass\n    through a non-Index.\n    '
    if (len(data) != len(index)):
        raise ValueError(f'Length of values ({len(data)}) does not match length of index ({len(index)})')
    if isinstance(data, np.ndarray):
        if (data.dtype.kind in ['M', 'm']):
            data = sanitize_array(data, index, copy=False)
    return data
