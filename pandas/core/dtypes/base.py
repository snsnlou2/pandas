
'\nExtend pandas with custom array types.\n'
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union
import numpy as np
from pandas._typing import DtypeObj
from pandas.errors import AbstractMethodError
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
if TYPE_CHECKING:
    from pandas.core.arrays import ExtensionArray

class ExtensionDtype():
    "\n    A custom data type, to be paired with an ExtensionArray.\n\n    See Also\n    --------\n    extensions.register_extension_dtype: Register an ExtensionType\n        with pandas as class decorator.\n    extensions.ExtensionArray: Abstract base class for custom 1-D array types.\n\n    Notes\n    -----\n    The interface includes the following abstract methods that must\n    be implemented by subclasses:\n\n    * type\n    * name\n\n    The following attributes and methods influence the behavior of the dtype in\n    pandas operations\n\n    * _is_numeric\n    * _is_boolean\n    * _get_common_dtype\n\n    Optionally one can override construct_array_type for construction\n    with the name of this dtype via the Registry. See\n    :meth:`extensions.register_extension_dtype`.\n\n    * construct_array_type\n\n    The `na_value` class attribute can be used to set the default NA value\n    for this type. :attr:`numpy.nan` is used by default.\n\n    ExtensionDtypes are required to be hashable. The base class provides\n    a default implementation, which relies on the ``_metadata`` class\n    attribute. ``_metadata`` should be a tuple containing the strings\n    that define your data type. For example, with ``PeriodDtype`` that's\n    the ``freq`` attribute.\n\n    **If you have a parametrized dtype you should set the ``_metadata``\n    class property**.\n\n    Ideally, the attributes in ``_metadata`` will match the\n    parameters to your ``ExtensionDtype.__init__`` (if any). If any of\n    the attributes in ``_metadata`` don't implement the standard\n    ``__eq__`` or ``__hash__``, the default implementations here will not\n    work.\n\n    .. versionchanged:: 0.24.0\n\n       Added ``_metadata``, ``__hash__``, and changed the default definition\n       of ``__eq__``.\n\n    For interaction with Apache Arrow (pyarrow), a ``__from_arrow__`` method\n    can be implemented: this method receives a pyarrow Array or ChunkedArray\n    as only argument and is expected to return the appropriate pandas\n    ExtensionArray for this dtype and the passed values::\n\n        class ExtensionDtype:\n\n            def __from_arrow__(\n                self, array: Union[pyarrow.Array, pyarrow.ChunkedArray]\n            ) -> ExtensionArray:\n                ...\n\n    This class does not inherit from 'abc.ABCMeta' for performance reasons.\n    Methods and properties required by the interface raise\n    ``pandas.errors.AbstractMethodError`` and no ``register`` method is\n    provided for registering virtual subclasses.\n    "
    _metadata = ()

    def __str__(self):
        return self.name

    def __eq__(self, other):
        "\n        Check whether 'other' is equal to self.\n\n        By default, 'other' is considered equal if either\n\n        * it's a string matching 'self.name'.\n        * it's an instance of this type and all of the attributes\n          in ``self._metadata`` are equal between `self` and `other`.\n\n        Parameters\n        ----------\n        other : Any\n\n        Returns\n        -------\n        bool\n        "
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
        if isinstance(other, type(self)):
            return all(((getattr(self, attr) == getattr(other, attr)) for attr in self._metadata))
        return False

    def __hash__(self):
        return hash(tuple((getattr(self, attr) for attr in self._metadata)))

    def __ne__(self, other):
        return (not self.__eq__(other))

    @property
    def na_value(self):
        '\n        Default NA value to use for this type.\n\n        This is used in e.g. ExtensionArray.take. This should be the\n        user-facing "boxed" version of the NA value, not the physical NA value\n        for storage.  e.g. for JSONArray, this is an empty dictionary.\n        '
        return np.nan

    @property
    def type(self):
        "\n        The scalar type for the array, e.g. ``int``\n\n        It's expected ``ExtensionArray[item]`` returns an instance\n        of ``ExtensionDtype.type`` for scalar ``item``, assuming\n        that value is valid (not NA). NA values do not need to be\n        instances of `type`.\n        "
        raise AbstractMethodError(self)

    @property
    def kind(self):
        "\n        A character code (one of 'biufcmMOSUV'), default 'O'\n\n        This should match the NumPy dtype used when the array is\n        converted to an ndarray, which is probably 'O' for object if\n        the extension type cannot be represented as a built-in NumPy\n        type.\n\n        See Also\n        --------\n        numpy.dtype.kind\n        "
        return 'O'

    @property
    def name(self):
        '\n        A string identifying the data type.\n\n        Will be used for display in, e.g. ``Series.dtype``\n        '
        raise AbstractMethodError(self)

    @property
    def names(self):
        '\n        Ordered list of field names, or None if there are no fields.\n\n        This is for compatibility with NumPy arrays, and may be removed in the\n        future.\n        '
        return None

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        raise NotImplementedError

    @classmethod
    def construct_from_string(cls, string):
        '\n        Construct this type from a string.\n\n        This is useful mainly for data types that accept parameters.\n        For example, a period dtype accepts a frequency parameter that\n        can be set as ``period[H]`` (where H means hourly frequency).\n\n        By default, in the abstract class, just the name of the type is\n        expected. But subclasses can overwrite this method to accept\n        parameters.\n\n        Parameters\n        ----------\n        string : str\n            The name of the type, for example ``category``.\n\n        Returns\n        -------\n        ExtensionDtype\n            Instance of the dtype.\n\n        Raises\n        ------\n        TypeError\n            If a class cannot be constructed from this \'string\'.\n\n        Examples\n        --------\n        For extension dtypes with arguments the following may be an\n        adequate implementation.\n\n        >>> @classmethod\n        ... def construct_from_string(cls, string):\n        ...     pattern = re.compile(r"^my_type\\[(?P<arg_name>.+)\\]$")\n        ...     match = pattern.match(string)\n        ...     if match:\n        ...         return cls(**match.groupdict())\n        ...     else:\n        ...         raise TypeError(\n        ...             f"Cannot construct a \'{cls.__name__}\' from \'{string}\'"\n        ...         )\n        '
        if (not isinstance(string, str)):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        assert isinstance(cls.name, str), (cls, type(cls.name))
        if (string != cls.name):
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()

    @classmethod
    def is_dtype(cls, dtype):
        "\n        Check if we match 'dtype'.\n\n        Parameters\n        ----------\n        dtype : object\n            The object to check.\n\n        Returns\n        -------\n        bool\n\n        Notes\n        -----\n        The default implementation is True if\n\n        1. ``cls.construct_from_string(dtype)`` is an instance\n           of ``cls``.\n        2. ``dtype`` is an object and is an instance of ``cls``\n        3. ``dtype`` has a ``dtype`` attribute, and any of the above\n           conditions is true for ``dtype.dtype``.\n        "
        dtype = getattr(dtype, 'dtype', dtype)
        if isinstance(dtype, (ABCSeries, ABCIndex, ABCDataFrame, np.dtype)):
            return False
        elif (dtype is None):
            return False
        elif isinstance(dtype, cls):
            return True
        if isinstance(dtype, str):
            try:
                return (cls.construct_from_string(dtype) is not None)
            except TypeError:
                return False
        return False

    @property
    def _is_numeric(self):
        "\n        Whether columns with this dtype should be considered numeric.\n\n        By default ExtensionDtypes are assumed to be non-numeric.\n        They'll be excluded from operations that exclude non-numeric\n        columns, like (groupby) reductions, plotting, etc.\n        "
        return False

    @property
    def _is_boolean(self):
        '\n        Whether this dtype should be considered boolean.\n\n        By default, ExtensionDtypes are assumed to be non-numeric.\n        Setting this to True will affect the behavior of several places,\n        e.g.\n\n        * is_bool\n        * boolean indexing\n\n        Returns\n        -------\n        bool\n        '
        return False

    def _get_common_dtype(self, dtypes):
        '\n        Return the common dtype, if one exists.\n\n        Used in `find_common_type` implementation. This is for example used\n        to determine the resulting dtype in a concat operation.\n\n        If no common dtype exists, return None (which gives the other dtypes\n        the chance to determine a common dtype). If all dtypes in the list\n        return None, then the common dtype will be "object" dtype (this means\n        it is never needed to return "object" dtype from this method itself).\n\n        Parameters\n        ----------\n        dtypes : list of dtypes\n            The dtypes for which to determine a common dtype. This is a list\n            of np.dtype or ExtensionDtype instances.\n\n        Returns\n        -------\n        Common dtype (np.dtype or ExtensionDtype) or None\n        '
        if (len(set(dtypes)) == 1):
            return self
        else:
            return None

def register_extension_dtype(cls):
    '\n    Register an ExtensionType with pandas as class decorator.\n\n    .. versionadded:: 0.24.0\n\n    This enables operations like ``.astype(name)`` for the name\n    of the ExtensionDtype.\n\n    Returns\n    -------\n    callable\n        A class decorator.\n\n    Examples\n    --------\n    >>> from pandas.api.extensions import register_extension_dtype\n    >>> from pandas.api.extensions import ExtensionDtype\n    >>> @register_extension_dtype\n    ... class MyExtensionDtype(ExtensionDtype):\n    ...     name = "myextension"\n    '
    registry.register(cls)
    return cls

class Registry():
    '\n    Registry for dtype inference.\n\n    The registry allows one to map a string repr of a extension\n    dtype to an extension dtype. The string alias can be used in several\n    places, including\n\n    * Series and Index constructors\n    * :meth:`pandas.array`\n    * :meth:`pandas.Series.astype`\n\n    Multiple extension types can be registered.\n    These are tried in order.\n    '

    def __init__(self):
        self.dtypes: List[Type[ExtensionDtype]] = []

    def register(self, dtype):
        '\n        Parameters\n        ----------\n        dtype : ExtensionDtype class\n        '
        if (not issubclass(dtype, ExtensionDtype)):
            raise ValueError('can only register pandas extension dtypes')
        self.dtypes.append(dtype)

    def find(self, dtype):
        '\n        Parameters\n        ----------\n        dtype : Type[ExtensionDtype] or str\n\n        Returns\n        -------\n        return the first matching dtype, otherwise return None\n        '
        if (not isinstance(dtype, str)):
            dtype_type = dtype
            if (not isinstance(dtype, type)):
                dtype_type = type(dtype)
            if issubclass(dtype_type, ExtensionDtype):
                return dtype
            return None
        for dtype_type in self.dtypes:
            try:
                return dtype_type.construct_from_string(dtype)
            except TypeError:
                pass
        return None
registry = Registry()
