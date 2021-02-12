
from collections import abc, defaultdict
import copy
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Union
import numpy as np
from pandas._libs.writers import convert_json_to_lines
from pandas._typing import Scalar
from pandas.util._decorators import deprecate
import pandas as pd
from pandas import DataFrame

def convert_to_line_delimits(s):
    '\n    Helper function that converts JSON lists to line delimited JSON.\n    '
    if ((not (s[0] == '[')) and (s[(- 1)] == ']')):
        return s
    s = s[1:(- 1)]
    return convert_json_to_lines(s)

def nested_to_record(ds, prefix='', sep='.', level=0, max_level=None):
    '\n    A simplified json_normalize\n\n    Converts a nested dict into a flat dict ("record"), unlike json_normalize,\n    it does not attempt to extract a subset of the data.\n\n    Parameters\n    ----------\n    ds : dict or list of dicts\n    prefix: the prefix, optional, default: ""\n    sep : str, default \'.\'\n        Nested records will generate names separated by sep,\n        e.g., for sep=\'.\', { \'foo\' : { \'bar\' : 0 } } -> foo.bar\n    level: int, optional, default: 0\n        The number of levels in the json string.\n\n    max_level: int, optional, default: None\n        The max depth to normalize.\n\n        .. versionadded:: 0.25.0\n\n    Returns\n    -------\n    d - dict or list of dicts, matching `ds`\n\n    Examples\n    --------\n    IN[52]: nested_to_record(dict(flat1=1,dict1=dict(c=1,d=2),\n                                  nested=dict(e=dict(c=1,d=2),d=2)))\n    Out[52]:\n    {\'dict1.c\': 1,\n     \'dict1.d\': 2,\n     \'flat1\': 1,\n     \'nested.d\': 2,\n     \'nested.e.c\': 1,\n     \'nested.e.d\': 2}\n    '
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for (k, v) in d.items():
            if (not isinstance(k, str)):
                k = str(k)
            if (level == 0):
                newkey = k
            else:
                newkey = ((prefix + sep) + k)
            if ((not isinstance(v, dict)) or ((max_level is not None) and (level >= max_level))):
                if (level != 0):
                    v = new_d.pop(k)
                    new_d[newkey] = v
                continue
            else:
                v = new_d.pop(k)
                new_d.update(nested_to_record(v, newkey, sep, (level + 1), max_level))
        new_ds.append(new_d)
    if singleton:
        return new_ds[0]
    return new_ds

def _json_normalize(data, record_path=None, meta=None, meta_prefix=None, record_prefix=None, errors='raise', sep='.', max_level=None):
    '\n    Normalize semi-structured JSON data into a flat table.\n\n    Parameters\n    ----------\n    data : dict or list of dicts\n        Unserialized JSON objects.\n    record_path : str or list of str, default None\n        Path in each object to list of records. If not passed, data will be\n        assumed to be an array of records.\n    meta : list of paths (str or list of str), default None\n        Fields to use as metadata for each record in resulting table.\n    meta_prefix : str, default None\n        If True, prefix records with dotted (?) path, e.g. foo.bar.field if\n        meta is [\'foo\', \'bar\'].\n    record_prefix : str, default None\n        If True, prefix records with dotted (?) path, e.g. foo.bar.field if\n        path to records is [\'foo\', \'bar\'].\n    errors : {\'raise\', \'ignore\'}, default \'raise\'\n        Configures error handling.\n\n        * \'ignore\' : will ignore KeyError if keys listed in meta are not\n          always present.\n        * \'raise\' : will raise KeyError if keys listed in meta are not\n          always present.\n    sep : str, default \'.\'\n        Nested records will generate names separated by sep.\n        e.g., for sep=\'.\', {\'foo\': {\'bar\': 0}} -> foo.bar.\n    max_level : int, default None\n        Max number of levels(depth of dict) to normalize.\n        if None, normalizes all levels.\n\n        .. versionadded:: 0.25.0\n\n    Returns\n    -------\n    frame : DataFrame\n    Normalize semi-structured JSON data into a flat table.\n\n    Examples\n    --------\n    >>> data = [{\'id\': 1, \'name\': {\'first\': \'Coleen\', \'last\': \'Volk\'}},\n    ...         {\'name\': {\'given\': \'Mose\', \'family\': \'Regner\'}},\n    ...         {\'id\': 2, \'name\': \'Faye Raker\'}]\n    >>> pd.json_normalize(data)\n        id name.first name.last name.given name.family        name\n    0  1.0     Coleen      Volk        NaN         NaN         NaN\n    1  NaN        NaN       NaN       Mose      Regner         NaN\n    2  2.0        NaN       NaN        NaN         NaN  Faye Raker\n\n    >>> data = [{\'id\': 1,\n    ...          \'name\': "Cole Volk",\n    ...          \'fitness\': {\'height\': 130, \'weight\': 60}},\n    ...         {\'name\': "Mose Reg",\n    ...          \'fitness\': {\'height\': 130, \'weight\': 60}},\n    ...         {\'id\': 2, \'name\': \'Faye Raker\',\n    ...          \'fitness\': {\'height\': 130, \'weight\': 60}}]\n    >>> pd.json_normalize(data, max_level=0)\n        id        name                        fitness\n    0  1.0   Cole Volk  {\'height\': 130, \'weight\': 60}\n    1  NaN    Mose Reg  {\'height\': 130, \'weight\': 60}\n    2  2.0  Faye Raker  {\'height\': 130, \'weight\': 60}\n\n    Normalizes nested data up to level 1.\n\n    >>> data = [{\'id\': 1,\n    ...          \'name\': "Cole Volk",\n    ...          \'fitness\': {\'height\': 130, \'weight\': 60}},\n    ...         {\'name\': "Mose Reg",\n    ...          \'fitness\': {\'height\': 130, \'weight\': 60}},\n    ...         {\'id\': 2, \'name\': \'Faye Raker\',\n    ...          \'fitness\': {\'height\': 130, \'weight\': 60}}]\n    >>> pd.json_normalize(data, max_level=1)\n        id        name  fitness.height  fitness.weight\n    0  1.0   Cole Volk             130              60\n    1  NaN    Mose Reg             130              60\n    2  2.0  Faye Raker             130              60\n\n    >>> data = [{\'state\': \'Florida\',\n    ...          \'shortname\': \'FL\',\n    ...          \'info\': {\'governor\': \'Rick Scott\'},\n    ...          \'counties\': [{\'name\': \'Dade\', \'population\': 12345},\n    ...                       {\'name\': \'Broward\', \'population\': 40000},\n    ...                       {\'name\': \'Palm Beach\', \'population\': 60000}]},\n    ...         {\'state\': \'Ohio\',\n    ...          \'shortname\': \'OH\',\n    ...          \'info\': {\'governor\': \'John Kasich\'},\n    ...          \'counties\': [{\'name\': \'Summit\', \'population\': 1234},\n    ...                       {\'name\': \'Cuyahoga\', \'population\': 1337}]}]\n    >>> result = pd.json_normalize(data, \'counties\', [\'state\', \'shortname\',\n    ...                                            [\'info\', \'governor\']])\n    >>> result\n             name  population    state shortname info.governor\n    0        Dade       12345   Florida    FL    Rick Scott\n    1     Broward       40000   Florida    FL    Rick Scott\n    2  Palm Beach       60000   Florida    FL    Rick Scott\n    3      Summit        1234   Ohio       OH    John Kasich\n    4    Cuyahoga        1337   Ohio       OH    John Kasich\n\n    >>> data = {\'A\': [1, 2]}\n    >>> pd.json_normalize(data, \'A\', record_prefix=\'Prefix.\')\n        Prefix.0\n    0          1\n    1          2\n\n    Returns normalized data with columns prefixed with the given string.\n    '

    def _pull_field(js: Dict[(str, Any)], spec: Union[(List, str)]) -> Union[(Scalar, Iterable)]:
        'Internal function to pull field'
        result = js
        if isinstance(spec, list):
            for field in spec:
                result = result[field]
        else:
            result = result[spec]
        return result

    def _pull_records(js: Dict[(str, Any)], spec: Union[(List, str)]) -> List:
        '\n        Internal function to pull field for records, and similar to\n        _pull_field, but require to return list. And will raise error\n        if has non iterable value.\n        '
        result = _pull_field(js, spec)
        if (not isinstance(result, list)):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(f'{js} has non list value {result} for path {spec}. Must be list or null.')
        return result
    if (isinstance(data, list) and (not data)):
        return DataFrame()
    elif isinstance(data, dict):
        data = [data]
    elif (isinstance(data, abc.Iterable) and (not isinstance(data, str))):
        data = list(data)
    else:
        raise NotImplementedError
    if (record_path is None):
        if any(([isinstance(x, dict) for x in y.values()] for y in data)):
            data = nested_to_record(data, sep=sep, max_level=max_level)
        return DataFrame(data)
    elif (not isinstance(record_path, list)):
        record_path = [record_path]
    if (meta is None):
        meta = []
    elif (not isinstance(meta, list)):
        meta = [meta]
    _meta = [(m if isinstance(m, list) else [m]) for m in meta]
    records: List = []
    lengths = []
    meta_vals: DefaultDict = defaultdict(list)
    meta_keys = [sep.join(val) for val in _meta]

    def _recursive_extract(data, path, seen_meta, level=0):
        if isinstance(data, dict):
            data = [data]
        if (len(path) > 1):
            for obj in data:
                for (val, key) in zip(_meta, meta_keys):
                    if ((level + 1) == len(val)):
                        seen_meta[key] = _pull_field(obj, val[(- 1)])
                _recursive_extract(obj[path[0]], path[1:], seen_meta, level=(level + 1))
        else:
            for obj in data:
                recs = _pull_records(obj, path[0])
                recs = [(nested_to_record(r, sep=sep, max_level=max_level) if isinstance(r, dict) else r) for r in recs]
                lengths.append(len(recs))
                for (val, key) in zip(_meta, meta_keys):
                    if ((level + 1) > len(val)):
                        meta_val = seen_meta[key]
                    else:
                        try:
                            meta_val = _pull_field(obj, val[level:])
                        except KeyError as e:
                            if (errors == 'ignore'):
                                meta_val = np.nan
                            else:
                                raise KeyError(f"Try running with errors='ignore' as key {e} is not always present") from e
                    meta_vals[key].append(meta_val)
                records.extend(recs)
    _recursive_extract(data, record_path, {}, level=0)
    result = DataFrame(records)
    if (record_prefix is not None):
        result = result.rename(columns=(lambda x: f'{record_prefix}{x}'))
    for (k, v) in meta_vals.items():
        if (meta_prefix is not None):
            k = (meta_prefix + k)
        if (k in result):
            raise ValueError(f'Conflicting metadata name {k}, need distinguishing prefix ')
        result[k] = np.array(v, dtype=object).repeat(lengths)
    return result
json_normalize = deprecate('pandas.io.json.json_normalize', _json_normalize, '1.0.0', 'pandas.json_normalize')
