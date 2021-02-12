
from typing import List
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import is_integer, is_list_like
_writers = {}

def register_writer(klass):
    '\n    Add engine to the excel writer registry.io.excel.\n\n    You must use this method to integrate with ``to_excel``.\n\n    Parameters\n    ----------\n    klass : ExcelWriter\n    '
    if (not callable(klass)):
        raise ValueError('Can only register callables as engines')
    engine_name = klass.engine
    _writers[engine_name] = klass

def get_default_writer(ext):
    '\n    Return the default writer for the given extension.\n\n    Parameters\n    ----------\n    ext : str\n        The excel file extension for which to get the default engine.\n\n    Returns\n    -------\n    str\n        The default engine for the extension.\n    '
    _default_writers = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xls': 'xlwt', 'ods': 'odf'}
    xlsxwriter = import_optional_dependency('xlsxwriter', raise_on_missing=False, on_version='warn')
    if xlsxwriter:
        _default_writers['xlsx'] = 'xlsxwriter'
    return _default_writers[ext]

def get_writer(engine_name):
    try:
        return _writers[engine_name]
    except KeyError as err:
        raise ValueError(f"No Excel writer '{engine_name}'") from err

def _excel2num(x):
    "\n    Convert Excel column name like 'AB' to 0-based column index.\n\n    Parameters\n    ----------\n    x : str\n        The Excel column name to convert to a 0-based column index.\n\n    Returns\n    -------\n    num : int\n        The column index corresponding to the name.\n\n    Raises\n    ------\n    ValueError\n        Part of the Excel column name was invalid.\n    "
    index = 0
    for c in x.upper().strip():
        cp = ord(c)
        if ((cp < ord('A')) or (cp > ord('Z'))):
            raise ValueError(f'Invalid column name: {x}')
        index = ((((index * 26) + cp) - ord('A')) + 1)
    return (index - 1)

def _range2cols(areas):
    "\n    Convert comma separated list of column names and ranges to indices.\n\n    Parameters\n    ----------\n    areas : str\n        A string containing a sequence of column ranges (or areas).\n\n    Returns\n    -------\n    cols : list\n        A list of 0-based column indices.\n\n    Examples\n    --------\n    >>> _range2cols('A:E')\n    [0, 1, 2, 3, 4]\n    >>> _range2cols('A,C,Z:AB')\n    [0, 2, 25, 26, 27]\n    "
    cols: List[int] = []
    for rng in areas.split(','):
        if (':' in rng):
            rngs = rng.split(':')
            cols.extend(range(_excel2num(rngs[0]), (_excel2num(rngs[1]) + 1)))
        else:
            cols.append(_excel2num(rng))
    return cols

def maybe_convert_usecols(usecols):
    '\n    Convert `usecols` into a compatible format for parsing in `parsers.py`.\n\n    Parameters\n    ----------\n    usecols : object\n        The use-columns object to potentially convert.\n\n    Returns\n    -------\n    converted : object\n        The compatible format of `usecols`.\n    '
    if (usecols is None):
        return usecols
    if is_integer(usecols):
        raise ValueError('Passing an integer for `usecols` is no longer supported.  Please pass in a list of int from 0 to `usecols` inclusive instead.')
    if isinstance(usecols, str):
        return _range2cols(usecols)
    return usecols

def validate_freeze_panes(freeze_panes):
    if (freeze_panes is not None):
        if ((len(freeze_panes) == 2) and all((isinstance(item, int) for item in freeze_panes))):
            return True
        raise ValueError('freeze_panes must be of form (row, column) where row and column are integers')
    return False

def fill_mi_header(row, control_row):
    '\n    Forward fill blank entries in row but only inside the same parent index.\n\n    Used for creating headers in Multiindex.\n\n    Parameters\n    ----------\n    row : list\n        List of items in a single row.\n    control_row : list of bool\n        Helps to determine if particular column is in same parent index as the\n        previous value. Used to stop propagation of empty cells between\n        different indexes.\n\n    Returns\n    -------\n    Returns changed row and control_row\n    '
    last = row[0]
    for i in range(1, len(row)):
        if (not control_row[i]):
            last = row[i]
        if ((row[i] == '') or (row[i] is None)):
            row[i] = last
        else:
            control_row[i] = False
            last = row[i]
    return (row, control_row)

def pop_header_name(row, index_col):
    '\n    Pop the header name for MultiIndex parsing.\n\n    Parameters\n    ----------\n    row : list\n        The data row to parse for the header name.\n    index_col : int, list\n        The index columns for our data. Assumed to be non-null.\n\n    Returns\n    -------\n    header_name : str\n        The extracted header name.\n    trimmed_row : list\n        The original data row with the header name removed.\n    '
    i = (index_col if (not is_list_like(index_col)) else max(index_col))
    header_name = row[i]
    header_name = (None if (header_name == '') else header_name)
    return (header_name, ((row[:i] + ['']) + row[(i + 1):]))
