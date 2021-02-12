
'\nExpose public exceptions & warnings\n'
from pandas._config.config import OptionError
from pandas._libs.tslibs import OutOfBoundsDatetime, OutOfBoundsTimedelta

class NullFrequencyError(ValueError):
    '\n    Error raised when a null `freq` attribute is used in an operation\n    that needs a non-null frequency, particularly `DatetimeIndex.shift`,\n    `TimedeltaIndex.shift`, `PeriodIndex.shift`.\n    '
    pass

class PerformanceWarning(Warning):
    '\n    Warning raised when there is a possible performance impact.\n    '

class UnsupportedFunctionCall(ValueError):
    '\n    Exception raised when attempting to call a numpy function\n    on a pandas object, but that function is not supported by\n    the object e.g. ``np.cumsum(groupby_object)``.\n    '

class UnsortedIndexError(KeyError):
    '\n    Error raised when attempting to get a slice of a MultiIndex,\n    and the index has not been lexsorted. Subclass of `KeyError`.\n    '

class ParserError(ValueError):
    '\n    Exception that is raised by an error encountered in parsing file contents.\n\n    This is a generic error raised for errors encountered when functions like\n    `read_csv` or `read_html` are parsing contents of a file.\n\n    See Also\n    --------\n    read_csv : Read CSV (comma-separated) file into a DataFrame.\n    read_html : Read HTML table into a DataFrame.\n    '

class DtypeWarning(Warning):
    "\n    Warning raised when reading different dtypes in a column from a file.\n\n    Raised for a dtype incompatibility. This can happen whenever `read_csv`\n    or `read_table` encounter non-uniform dtypes in a column(s) of a given\n    CSV file.\n\n    See Also\n    --------\n    read_csv : Read CSV (comma-separated) file into a DataFrame.\n    read_table : Read general delimited file into a DataFrame.\n\n    Notes\n    -----\n    This warning is issued when dealing with larger files because the dtype\n    checking happens per chunk read.\n\n    Despite the warning, the CSV file is read with mixed types in a single\n    column which will be an object type. See the examples below to better\n    understand this issue.\n\n    Examples\n    --------\n    This example creates and reads a large CSV file with a column that contains\n    `int` and `str`.\n\n    >>> df = pd.DataFrame({'a': (['1'] * 100000 + ['X'] * 100000 +\n    ...                          ['1'] * 100000),\n    ...                    'b': ['b'] * 300000})\n    >>> df.to_csv('test.csv', index=False)\n    >>> df2 = pd.read_csv('test.csv')\n    ... # DtypeWarning: Columns (0) have mixed types\n\n    Important to notice that ``df2`` will contain both `str` and `int` for the\n    same input, '1'.\n\n    >>> df2.iloc[262140, 0]\n    '1'\n    >>> type(df2.iloc[262140, 0])\n    <class 'str'>\n    >>> df2.iloc[262150, 0]\n    1\n    >>> type(df2.iloc[262150, 0])\n    <class 'int'>\n\n    One way to solve this issue is using the `dtype` parameter in the\n    `read_csv` and `read_table` functions to explicit the conversion:\n\n    >>> df2 = pd.read_csv('test.csv', sep=',', dtype={'a': str})\n\n    No warning was issued.\n\n    >>> import os\n    >>> os.remove('test.csv')\n    "

class EmptyDataError(ValueError):
    '\n    Exception that is thrown in `pd.read_csv` (by both the C and\n    Python engines) when empty data or header is encountered.\n    '

class ParserWarning(Warning):
    "\n    Warning raised when reading a file that doesn't use the default 'c' parser.\n\n    Raised by `pd.read_csv` and `pd.read_table` when it is necessary to change\n    parsers, generally from the default 'c' parser to 'python'.\n\n    It happens due to a lack of support or functionality for parsing a\n    particular attribute of a CSV file with the requested engine.\n\n    Currently, 'c' unsupported options include the following parameters:\n\n    1. `sep` other than a single character (e.g. regex separators)\n    2. `skipfooter` higher than 0\n    3. `sep=None` with `delim_whitespace=False`\n\n    The warning can be avoided by adding `engine='python'` as a parameter in\n    `pd.read_csv` and `pd.read_table` methods.\n\n    See Also\n    --------\n    pd.read_csv : Read CSV (comma-separated) file into DataFrame.\n    pd.read_table : Read general delimited file into DataFrame.\n\n    Examples\n    --------\n    Using a `sep` in `pd.read_csv` other than a single character:\n\n    >>> import io\n    >>> csv = '''a;b;c\n    ...           1;1,8\n    ...           1;2,1'''\n    >>> df = pd.read_csv(io.StringIO(csv), sep='[;,]')  # doctest: +SKIP\n    ... # ParserWarning: Falling back to the 'python' engine...\n\n    Adding `engine='python'` to `pd.read_csv` removes the Warning:\n\n    >>> df = pd.read_csv(io.StringIO(csv), sep='[;,]', engine='python')\n    "

class MergeError(ValueError):
    '\n    Error raised when problems arise during merging due to problems\n    with input data. Subclass of `ValueError`.\n    '

class AccessorRegistrationWarning(Warning):
    '\n    Warning for attribute conflicts in accessor registration.\n    '

class AbstractMethodError(NotImplementedError):
    '\n    Raise this error instead of NotImplementedError for abstract methods\n    while keeping compatibility with Python 2 and Python 3.\n    '

    def __init__(self, class_instance, methodtype='method'):
        types = {'method', 'classmethod', 'staticmethod', 'property'}
        if (methodtype not in types):
            raise ValueError(f'methodtype must be one of {methodtype}, got {types} instead.')
        self.methodtype = methodtype
        self.class_instance = class_instance

    def __str__(self):
        if (self.methodtype == 'classmethod'):
            name = self.class_instance.__name__
        else:
            name = type(self.class_instance).__name__
        return f'This {self.methodtype} must be defined in the concrete class {name}'

class NumbaUtilError(Exception):
    '\n    Error raised for unsupported Numba engine routines.\n    '

class DuplicateLabelError(ValueError):
    "\n    Error raised when an operation would introduce duplicate labels.\n\n    .. versionadded:: 1.2.0\n\n    Examples\n    --------\n    >>> s = pd.Series([0, 1, 2], index=['a', 'b', 'c']).set_flags(\n    ...     allows_duplicate_labels=False\n    ... )\n    >>> s.reindex(['a', 'a', 'b'])\n    Traceback (most recent call last):\n       ...\n    DuplicateLabelError: Index has duplicates.\n          positions\n    label\n    a        [0, 1]\n    "

class InvalidIndexError(Exception):
    '\n    Exception raised when attempting to use an invalid index key.\n\n    .. versionadded:: 1.1.0\n    '
