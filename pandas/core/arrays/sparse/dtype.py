
'Sparse Dtype'
import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type
import warnings
import numpy as np
from pandas._typing import Dtype, DtypeObj
from pandas.errors import PerformanceWarning
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.cast import astype_nansafe
from pandas.core.dtypes.common import is_bool_dtype, is_extension_array_dtype, is_object_dtype, is_scalar, is_string_dtype, pandas_dtype
from pandas.core.dtypes.missing import isna, na_value_for_dtype
if TYPE_CHECKING:
    from pandas.core.arrays.sparse.array import SparseArray

@register_extension_dtype
class SparseDtype(ExtensionDtype):
    '\n    Dtype for data stored in :class:`SparseArray`.\n\n    This dtype implements the pandas ExtensionDtype interface.\n\n    .. versionadded:: 0.24.0\n\n    Parameters\n    ----------\n    dtype : str, ExtensionDtype, numpy.dtype, type, default numpy.float64\n        The dtype of the underlying array storing the non-fill value values.\n    fill_value : scalar, optional\n        The scalar value not stored in the SparseArray. By default, this\n        depends on `dtype`.\n\n        =========== ==========\n        dtype       na_value\n        =========== ==========\n        float       ``np.nan``\n        int         ``0``\n        bool        ``False``\n        datetime64  ``pd.NaT``\n        timedelta64 ``pd.NaT``\n        =========== ==========\n\n        The default value may be overridden by specifying a `fill_value`.\n\n    Attributes\n    ----------\n    None\n\n    Methods\n    -------\n    None\n    '
    _metadata = ('_dtype', '_fill_value', '_is_na_fill_value')

    def __init__(self, dtype=np.float64, fill_value=None):
        if isinstance(dtype, type(self)):
            if (fill_value is None):
                fill_value = dtype.fill_value
            dtype = dtype.subtype
        dtype = pandas_dtype(dtype)
        if is_string_dtype(dtype):
            dtype = np.dtype('object')
        if (fill_value is None):
            fill_value = na_value_for_dtype(dtype)
        if (not is_scalar(fill_value)):
            raise ValueError(f'fill_value must be a scalar. Got {fill_value} instead')
        self._dtype = dtype
        self._fill_value = fill_value

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
        if isinstance(other, type(self)):
            subtype = (self.subtype == other.subtype)
            if self._is_na_fill_value:
                fill_value = ((other._is_na_fill_value and isinstance(self.fill_value, type(other.fill_value))) or isinstance(other.fill_value, type(self.fill_value)))
            else:
                fill_value = (self.fill_value == other.fill_value)
            return (subtype and fill_value)
        return False

    @property
    def fill_value(self):
        "\n        The fill value of the array.\n\n        Converting the SparseArray to a dense ndarray will fill the\n        array with this value.\n\n        .. warning::\n\n           It's possible to end up with a SparseArray that has ``fill_value``\n           values in ``sp_values``. This can occur, for example, when setting\n           ``SparseArray.fill_value`` directly.\n        "
        return self._fill_value

    @property
    def _is_na_fill_value(self):
        return isna(self.fill_value)

    @property
    def _is_numeric(self):
        return (not is_object_dtype(self.subtype))

    @property
    def _is_boolean(self):
        return is_bool_dtype(self.subtype)

    @property
    def kind(self):
        "\n        The sparse kind. Either 'integer', or 'block'.\n        "
        return self.subtype.kind

    @property
    def type(self):
        return self.subtype.type

    @property
    def subtype(self):
        return self._dtype

    @property
    def name(self):
        return f'Sparse[{self.subtype.name}, {repr(self.fill_value)}]'

    def __repr__(self):
        return self.name

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        from pandas.core.arrays.sparse.array import SparseArray
        return SparseArray

    @classmethod
    def construct_from_string(cls, string):
        "\n        Construct a SparseDtype from a string form.\n\n        Parameters\n        ----------\n        string : str\n            Can take the following forms.\n\n            string           dtype\n            ================ ============================\n            'int'            SparseDtype[np.int64, 0]\n            'Sparse'         SparseDtype[np.float64, nan]\n            'Sparse[int]'    SparseDtype[np.int64, 0]\n            'Sparse[int, 0]' SparseDtype[np.int64, 0]\n            ================ ============================\n\n            It is not possible to specify non-default fill values\n            with a string. An argument like ``'Sparse[int, 1]'``\n            will raise a ``TypeError`` because the default fill value\n            for integers is 0.\n\n        Returns\n        -------\n        SparseDtype\n        "
        if (not isinstance(string, str)):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        msg = f"Cannot construct a 'SparseDtype' from '{string}'"
        if string.startswith('Sparse'):
            try:
                (sub_type, has_fill_value) = cls._parse_subtype(string)
            except ValueError as err:
                raise TypeError(msg) from err
            else:
                result = SparseDtype(sub_type)
                msg = f'''Cannot construct a 'SparseDtype' from '{string}'.

It looks like the fill_value in the string is not the default for the dtype. Non-default fill_values are not supported. Use the 'SparseDtype()' constructor instead.'''
                if (has_fill_value and (str(result) != string)):
                    raise TypeError(msg)
                return result
        else:
            raise TypeError(msg)

    @staticmethod
    def _parse_subtype(dtype):
        '\n        Parse a string to get the subtype\n\n        Parameters\n        ----------\n        dtype : str\n            A string like\n\n            * Sparse[subtype]\n            * Sparse[subtype, fill_value]\n\n        Returns\n        -------\n        subtype : str\n\n        Raises\n        ------\n        ValueError\n            When the subtype cannot be extracted.\n        '
        xpr = re.compile('Sparse\\[(?P<subtype>[^,]*)(, )?(?P<fill_value>.*?)?\\]$')
        m = xpr.match(dtype)
        has_fill_value = False
        if m:
            subtype = m.groupdict()['subtype']
            has_fill_value = bool(m.groupdict()['fill_value'])
        elif (dtype == 'Sparse'):
            subtype = 'float64'
        else:
            raise ValueError(f'Cannot parse {dtype}')
        return (subtype, has_fill_value)

    @classmethod
    def is_dtype(cls, dtype):
        dtype = getattr(dtype, 'dtype', dtype)
        if (isinstance(dtype, str) and dtype.startswith('Sparse')):
            (sub_type, _) = cls._parse_subtype(dtype)
            dtype = np.dtype(sub_type)
        elif isinstance(dtype, cls):
            return True
        return (isinstance(dtype, np.dtype) or (dtype == 'Sparse'))

    def update_dtype(self, dtype):
        '\n        Convert the SparseDtype to a new dtype.\n\n        This takes care of converting the ``fill_value``.\n\n        Parameters\n        ----------\n        dtype : Union[str, numpy.dtype, SparseDtype]\n            The new dtype to use.\n\n            * For a SparseDtype, it is simply returned\n            * For a NumPy dtype (or str), the current fill value\n              is converted to the new dtype, and a SparseDtype\n              with `dtype` and the new fill value is returned.\n\n        Returns\n        -------\n        SparseDtype\n            A new SparseDtype with the correct `dtype` and fill value\n            for that `dtype`.\n\n        Raises\n        ------\n        ValueError\n            When the current fill value cannot be converted to the\n            new `dtype` (e.g. trying to convert ``np.nan`` to an\n            integer dtype).\n\n\n        Examples\n        --------\n        >>> SparseDtype(int, 0).update_dtype(float)\n        Sparse[float64, 0.0]\n\n        >>> SparseDtype(int, 1).update_dtype(SparseDtype(float, np.nan))\n        Sparse[float64, nan]\n        '
        cls = type(self)
        dtype = pandas_dtype(dtype)
        if (not isinstance(dtype, cls)):
            if is_extension_array_dtype(dtype):
                raise TypeError('sparse arrays of extension dtypes not supported')
            fill_value = astype_nansafe(np.array(self.fill_value), dtype).item()
            dtype = cls(dtype, fill_value=fill_value)
        return dtype

    @property
    def _subtype_with_str(self):
        "\n        Whether the SparseDtype's subtype should be considered ``str``.\n\n        Typically, pandas will store string data in an object-dtype array.\n        When converting values to a dtype, e.g. in ``.astype``, we need to\n        be more specific, we need the actual underlying type.\n\n        Returns\n        -------\n        >>> SparseDtype(int, 1)._subtype_with_str\n        dtype('int64')\n\n        >>> SparseDtype(object, 1)._subtype_with_str\n        dtype('O')\n\n        >>> dtype = SparseDtype(str, '')\n        >>> dtype.subtype\n        dtype('O')\n\n        >>> dtype._subtype_with_str\n        <class 'str'>\n        "
        if isinstance(self.fill_value, str):
            return type(self.fill_value)
        return self.subtype

    def _get_common_dtype(self, dtypes):
        if any(((isinstance(x, ExtensionDtype) and (not isinstance(x, SparseDtype))) for x in dtypes)):
            return None
        fill_values = [x.fill_value for x in dtypes if isinstance(x, SparseDtype)]
        fill_value = fill_values[0]
        if (not ((len(set(fill_values)) == 1) or isna(fill_values).all())):
            warnings.warn(f"Concatenating sparse arrays with multiple fill values: '{fill_values}'. Picking the first and converting the rest.", PerformanceWarning, stacklevel=6)
        np_dtypes = [(x.subtype if isinstance(x, SparseDtype) else x) for x in dtypes]
        return SparseDtype(np.find_common_type(np_dtypes, []), fill_value=fill_value)
