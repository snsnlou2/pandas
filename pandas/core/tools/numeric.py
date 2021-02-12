
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import maybe_downcast_numeric
from pandas.core.dtypes.common import ensure_object, is_datetime_or_timedelta_dtype, is_decimal, is_integer_dtype, is_number, is_numeric_dtype, is_scalar, needs_i8_conversion
from pandas.core.dtypes.generic import ABCIndex, ABCSeries
import pandas as pd
from pandas.core.arrays.numeric import NumericArray

def to_numeric(arg, errors='raise', downcast=None):
    '\n    Convert argument to a numeric type.\n\n    The default return dtype is `float64` or `int64`\n    depending on the data supplied. Use the `downcast` parameter\n    to obtain other dtypes.\n\n    Please note that precision loss may occur if really large numbers\n    are passed in. Due to the internal limitations of `ndarray`, if\n    numbers smaller than `-9223372036854775808` (np.iinfo(np.int64).min)\n    or larger than `18446744073709551615` (np.iinfo(np.uint64).max) are\n    passed in, it is very likely they will be converted to float so that\n    they can stored in an `ndarray`. These warnings apply similarly to\n    `Series` since it internally leverages `ndarray`.\n\n    Parameters\n    ----------\n    arg : scalar, list, tuple, 1-d array, or Series\n        Argument to be converted.\n    errors : {\'ignore\', \'raise\', \'coerce\'}, default \'raise\'\n        - If \'raise\', then invalid parsing will raise an exception.\n        - If \'coerce\', then invalid parsing will be set as NaN.\n        - If \'ignore\', then invalid parsing will return the input.\n    downcast : {\'integer\', \'signed\', \'unsigned\', \'float\'}, default None\n        If not None, and if the data has been successfully cast to a\n        numerical dtype (or if the data was numeric to begin with),\n        downcast that resulting data to the smallest numerical dtype\n        possible according to the following rules:\n\n        - \'integer\' or \'signed\': smallest signed int dtype (min.: np.int8)\n        - \'unsigned\': smallest unsigned int dtype (min.: np.uint8)\n        - \'float\': smallest float dtype (min.: np.float32)\n\n        As this behaviour is separate from the core conversion to\n        numeric values, any errors raised during the downcasting\n        will be surfaced regardless of the value of the \'errors\' input.\n\n        In addition, downcasting will only occur if the size\n        of the resulting data\'s dtype is strictly larger than\n        the dtype it is to be cast to, so if none of the dtypes\n        checked satisfy that specification, no downcasting will be\n        performed on the data.\n\n    Returns\n    -------\n    ret\n        Numeric if parsing succeeded.\n        Return type depends on input.  Series if Series, otherwise ndarray.\n\n    See Also\n    --------\n    DataFrame.astype : Cast argument to a specified dtype.\n    to_datetime : Convert argument to datetime.\n    to_timedelta : Convert argument to timedelta.\n    numpy.ndarray.astype : Cast a numpy array to a specified type.\n    DataFrame.convert_dtypes : Convert dtypes.\n\n    Examples\n    --------\n    Take separate series and convert to numeric, coercing when told to\n\n    >>> s = pd.Series([\'1.0\', \'2\', -3])\n    >>> pd.to_numeric(s)\n    0    1.0\n    1    2.0\n    2   -3.0\n    dtype: float64\n    >>> pd.to_numeric(s, downcast=\'float\')\n    0    1.0\n    1    2.0\n    2   -3.0\n    dtype: float32\n    >>> pd.to_numeric(s, downcast=\'signed\')\n    0    1\n    1    2\n    2   -3\n    dtype: int8\n    >>> s = pd.Series([\'apple\', \'1.0\', \'2\', -3])\n    >>> pd.to_numeric(s, errors=\'ignore\')\n    0    apple\n    1      1.0\n    2        2\n    3       -3\n    dtype: object\n    >>> pd.to_numeric(s, errors=\'coerce\')\n    0    NaN\n    1    1.0\n    2    2.0\n    3   -3.0\n    dtype: float64\n\n    Downcasting of nullable integer and floating dtypes is supported:\n\n    >>> s = pd.Series([1, 2, 3], dtype="Int64")\n    >>> pd.to_numeric(s, downcast="integer")\n    0    1\n    1    2\n    2    3\n    dtype: Int8\n    >>> s = pd.Series([1.0, 2.1, 3.0], dtype="Float64")\n    >>> pd.to_numeric(s, downcast="float")\n    0    1.0\n    1    2.1\n    2    3.0\n    dtype: Float32\n    '
    if (downcast not in (None, 'integer', 'signed', 'unsigned', 'float')):
        raise ValueError('invalid downcasting method provided')
    if (errors not in ('ignore', 'raise', 'coerce')):
        raise ValueError('invalid error value specified')
    is_series = False
    is_index = False
    is_scalars = False
    if isinstance(arg, ABCSeries):
        is_series = True
        values = arg.values
    elif isinstance(arg, ABCIndex):
        is_index = True
        if needs_i8_conversion(arg.dtype):
            values = arg.asi8
        else:
            values = arg.values
    elif isinstance(arg, (list, tuple)):
        values = np.array(arg, dtype='O')
    elif is_scalar(arg):
        if is_decimal(arg):
            return float(arg)
        if is_number(arg):
            return arg
        is_scalars = True
        values = np.array([arg], dtype='O')
    elif (getattr(arg, 'ndim', 1) > 1):
        raise TypeError('arg must be a list, tuple, 1-d array, or Series')
    else:
        values = arg
    if isinstance(values, NumericArray):
        mask = values._mask
        values = values._data[(~ mask)]
    else:
        mask = None
    values_dtype = getattr(values, 'dtype', None)
    if is_numeric_dtype(values_dtype):
        pass
    elif is_datetime_or_timedelta_dtype(values_dtype):
        values = values.view(np.int64)
    else:
        values = ensure_object(values)
        coerce_numeric = (errors not in ('ignore', 'raise'))
        try:
            values = lib.maybe_convert_numeric(values, set(), coerce_numeric=coerce_numeric)
        except (ValueError, TypeError):
            if (errors == 'raise'):
                raise
    if ((downcast is not None) and is_numeric_dtype(values.dtype)):
        typecodes = None
        if (downcast in ('integer', 'signed')):
            typecodes = np.typecodes['Integer']
        elif ((downcast == 'unsigned') and ((not len(values)) or (np.min(values) >= 0))):
            typecodes = np.typecodes['UnsignedInteger']
        elif (downcast == 'float'):
            typecodes = np.typecodes['Float']
            float_32_char = np.dtype(np.float32).char
            float_32_ind = typecodes.index(float_32_char)
            typecodes = typecodes[float_32_ind:]
        if (typecodes is not None):
            for dtype in typecodes:
                dtype = np.dtype(dtype)
                if (dtype.itemsize <= values.dtype.itemsize):
                    values = maybe_downcast_numeric(values, dtype)
                    if (values.dtype == dtype):
                        break
    if (mask is not None):
        data = np.zeros(mask.shape, dtype=values.dtype)
        data[(~ mask)] = values
        from pandas.core.arrays import FloatingArray, IntegerArray
        klass = (IntegerArray if is_integer_dtype(data.dtype) else FloatingArray)
        values = klass(data, mask)
    if is_series:
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif is_index:
        return pd.Index(values, name=arg.name)
    elif is_scalars:
        return values[0]
    else:
        return values
