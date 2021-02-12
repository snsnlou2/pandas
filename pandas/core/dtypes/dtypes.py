
'\nDefine extension dtypes.\n'
import re
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Tuple, Type, Union, cast
import numpy as np
import pytz
from pandas._libs.interval import Interval
from pandas._libs.tslibs import NaT, Period, Timestamp, dtypes, timezones, to_offset
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._typing import Dtype, DtypeObj, Ordered
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.generic import ABCCategoricalIndex, ABCIndex
from pandas.core.dtypes.inference import is_bool, is_list_like
if TYPE_CHECKING:
    import pyarrow
    from pandas import Categorical
    from pandas.core.arrays import DatetimeArray, IntervalArray, PeriodArray
str_type = str

class PandasExtensionDtype(ExtensionDtype):
    '\n    A np.dtype duck-typed class, suitable for holding a custom dtype.\n\n    THIS IS NOT A REAL NUMPY DTYPE\n    '
    subdtype = None
    num = 100
    shape = ()
    itemsize = 8
    base = None
    isbuiltin = 0
    isnative = 0
    _cache = {}

    def __str__(self):
        '\n        Return a string representation for a particular Object\n        '
        return self.name

    def __repr__(self):
        '\n        Return a string representation for a particular object.\n        '
        return str(self)

    def __hash__(self):
        raise NotImplementedError('sub-classes should implement an __hash__ method')

    def __getstate__(self):
        return {k: getattr(self, k, None) for k in self._metadata}

    @classmethod
    def reset_cache(cls):
        ' clear the cache '
        cls._cache = {}

class CategoricalDtypeType(type):
    '\n    the type of CategoricalDtype, this metaclass determines subclass ability\n    '
    pass

@register_extension_dtype
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    "\n    Type for categorical data with the categories and orderedness.\n\n    Parameters\n    ----------\n    categories : sequence, optional\n        Must be unique, and must not contain any nulls.\n        The categories are stored in an Index,\n        and if an index is provided the dtype of that index will be used.\n    ordered : bool or None, default False\n        Whether or not this categorical is treated as a ordered categorical.\n        None can be used to maintain the ordered value of existing categoricals when\n        used in operations that combine categoricals, e.g. astype, and will resolve to\n        False if there is no existing ordered to maintain.\n\n    Attributes\n    ----------\n    categories\n    ordered\n\n    Methods\n    -------\n    None\n\n    See Also\n    --------\n    Categorical : Represent a categorical variable in classic R / S-plus fashion.\n\n    Notes\n    -----\n    This class is useful for specifying the type of a ``Categorical``\n    independent of the values. See :ref:`categorical.categoricaldtype`\n    for more.\n\n    Examples\n    --------\n    >>> t = pd.CategoricalDtype(categories=['b', 'a'], ordered=True)\n    >>> pd.Series(['a', 'b', 'a', 'c'], dtype=t)\n    0      a\n    1      b\n    2      a\n    3    NaN\n    dtype: category\n    Categories (2, object): ['b' < 'a']\n\n    An empty CategoricalDtype with a specific dtype can be created\n    by providing an empty index. As follows,\n\n    >>> pd.CategoricalDtype(pd.DatetimeIndex([])).categories.dtype\n    dtype('<M8[ns]')\n    "
    name = 'category'
    type = CategoricalDtypeType
    kind = 'O'
    str = '|O08'
    base = np.dtype('O')
    _metadata = ('categories', 'ordered')
    _cache = {}

    def __init__(self, categories=None, ordered=False):
        self._finalize(categories, ordered, fastpath=False)

    @classmethod
    def _from_fastpath(cls, categories=None, ordered=None):
        self = cls.__new__(cls)
        self._finalize(categories, ordered, fastpath=True)
        return self

    @classmethod
    def _from_categorical_dtype(cls, dtype, categories=None, ordered=None):
        if (categories is ordered is None):
            return dtype
        if (categories is None):
            categories = dtype.categories
        if (ordered is None):
            ordered = dtype.ordered
        return cls(categories, ordered)

    @classmethod
    def _from_values_or_dtype(cls, values=None, categories=None, ordered=None, dtype=None):
        '\n        Construct dtype from the input parameters used in :class:`Categorical`.\n\n        This constructor method specifically does not do the factorization\n        step, if that is needed to find the categories. This constructor may\n        therefore return ``CategoricalDtype(categories=None, ordered=None)``,\n        which may not be useful. Additional steps may therefore have to be\n        taken to create the final dtype.\n\n        The return dtype is specified from the inputs in this prioritized\n        order:\n        1. if dtype is a CategoricalDtype, return dtype\n        2. if dtype is the string \'category\', create a CategoricalDtype from\n           the supplied categories and ordered parameters, and return that.\n        3. if values is a categorical, use value.dtype, but override it with\n           categories and ordered if either/both of those are not None.\n        4. if dtype is None and values is not a categorical, construct the\n           dtype from categories and ordered, even if either of those is None.\n\n        Parameters\n        ----------\n        values : list-like, optional\n            The list-like must be 1-dimensional.\n        categories : list-like, optional\n            Categories for the CategoricalDtype.\n        ordered : bool, optional\n            Designating if the categories are ordered.\n        dtype : CategoricalDtype or the string "category", optional\n            If ``CategoricalDtype``, cannot be used together with\n            `categories` or `ordered`.\n\n        Returns\n        -------\n        CategoricalDtype\n\n        Examples\n        --------\n        >>> pd.CategoricalDtype._from_values_or_dtype()\n        CategoricalDtype(categories=None, ordered=None)\n        >>> pd.CategoricalDtype._from_values_or_dtype(\n        ...     categories=[\'a\', \'b\'], ordered=True\n        ... )\n        CategoricalDtype(categories=[\'a\', \'b\'], ordered=True)\n        >>> dtype1 = pd.CategoricalDtype([\'a\', \'b\'], ordered=True)\n        >>> dtype2 = pd.CategoricalDtype([\'x\', \'y\'], ordered=False)\n        >>> c = pd.Categorical([0, 1], dtype=dtype1, fastpath=True)\n        >>> pd.CategoricalDtype._from_values_or_dtype(\n        ...     c, [\'x\', \'y\'], ordered=True, dtype=dtype2\n        ... )\n        Traceback (most recent call last):\n            ...\n        ValueError: Cannot specify `categories` or `ordered` together with\n        `dtype`.\n\n        The supplied dtype takes precedence over values\' dtype:\n\n        >>> pd.CategoricalDtype._from_values_or_dtype(c, dtype=dtype2)\n        CategoricalDtype(categories=[\'x\', \'y\'], ordered=False)\n        '
        if (dtype is not None):
            if isinstance(dtype, str):
                if (dtype == 'category'):
                    dtype = CategoricalDtype(categories, ordered)
                else:
                    raise ValueError(f'Unknown dtype {repr(dtype)}')
            elif ((categories is not None) or (ordered is not None)):
                raise ValueError('Cannot specify `categories` or `ordered` together with `dtype`.')
            elif (not isinstance(dtype, CategoricalDtype)):
                raise ValueError(f'Cannot not construct CategoricalDtype from {dtype}')
        elif cls.is_dtype(values):
            dtype = values.dtype._from_categorical_dtype(values.dtype, categories, ordered)
        else:
            dtype = CategoricalDtype(categories, ordered)
        return cast(CategoricalDtype, dtype)

    @classmethod
    def construct_from_string(cls, string):
        '\n        Construct a CategoricalDtype from a string.\n\n        Parameters\n        ----------\n        string : str\n            Must be the string "category" in order to be successfully constructed.\n\n        Returns\n        -------\n        CategoricalDtype\n            Instance of the dtype.\n\n        Raises\n        ------\n        TypeError\n            If a CategoricalDtype cannot be constructed from the input.\n        '
        if (not isinstance(string, str)):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if (string != cls.name):
            raise TypeError(f"Cannot construct a 'CategoricalDtype' from '{string}'")
        return cls(ordered=None)

    def _finalize(self, categories, ordered, fastpath=False):
        if (ordered is not None):
            self.validate_ordered(ordered)
        if (categories is not None):
            categories = self.validate_categories(categories, fastpath=fastpath)
        self._categories = categories
        self._ordered = ordered

    def __setstate__(self, state):
        self._categories = state.pop('categories', None)
        self._ordered = state.pop('ordered', False)

    def __hash__(self):
        if (self.categories is None):
            if self.ordered:
                return (- 1)
            else:
                return (- 2)
        return int(self._hash_categories(self.categories, self.ordered))

    def __eq__(self, other):
        "\n        Rules for CDT equality:\n        1) Any CDT is equal to the string 'category'\n        2) Any CDT is equal to itself\n        3) Any CDT is equal to a CDT with categories=None regardless of ordered\n        4) A CDT with ordered=True is only equal to another CDT with\n           ordered=True and identical categories in the same order\n        5) A CDT with ordered={False, None} is only equal to another CDT with\n           ordered={False, None} and identical categories, but same order is\n           not required. There is no distinction between False/None.\n        6) Any other comparison returns False\n        "
        if isinstance(other, str):
            return (other == self.name)
        elif (other is self):
            return True
        elif (not (hasattr(other, 'ordered') and hasattr(other, 'categories'))):
            return False
        elif ((self.categories is None) or (other.categories is None)):
            return (self.categories is other.categories)
        elif (self.ordered or other.ordered):
            return ((self.ordered == other.ordered) and self.categories.equals(other.categories))
        else:
            left = self.categories
            right = other.categories
            if (not (left.dtype == right.dtype)):
                return False
            if (len(left) != len(right)):
                return False
            if self.categories.equals(other.categories):
                return True
            if (left.dtype != object):
                indexer = left.get_indexer(right)
                return (indexer != (- 1)).all()
            return (hash(self) == hash(other))

    def __repr__(self):
        if (self.categories is None):
            data = 'None'
        else:
            data = self.categories._format_data(name=type(self).__name__)
            if (data is None):
                data = str(self.categories._range)
            data = data.rstrip(', ')
        return f'CategoricalDtype(categories={data}, ordered={self.ordered})'

    @staticmethod
    def _hash_categories(categories, ordered=True):
        from pandas.core.util.hashing import combine_hash_arrays, hash_array, hash_tuples
        if (len(categories) and isinstance(categories[0], tuple)):
            categories = list(categories)
            cat_array = hash_tuples(categories)
        else:
            if (categories.dtype == 'O'):
                if (len({type(x) for x in categories}) != 1):
                    hashed = hash((tuple(categories), ordered))
                    return hashed
            if DatetimeTZDtype.is_dtype(categories.dtype):
                categories = categories.astype('datetime64[ns]')
            cat_array = hash_array(np.asarray(categories), categorize=False)
        if ordered:
            cat_array = np.vstack([cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)])
        else:
            cat_array = [cat_array]
        hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
        return np.bitwise_xor.reduce(hashed)

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        from pandas import Categorical
        return Categorical

    @staticmethod
    def validate_ordered(ordered):
        "\n        Validates that we have a valid ordered parameter. If\n        it is not a boolean, a TypeError will be raised.\n\n        Parameters\n        ----------\n        ordered : object\n            The parameter to be verified.\n\n        Raises\n        ------\n        TypeError\n            If 'ordered' is not a boolean.\n        "
        if (not is_bool(ordered)):
            raise TypeError("'ordered' must either be 'True' or 'False'")

    @staticmethod
    def validate_categories(categories, fastpath=False):
        '\n        Validates that we have good categories\n\n        Parameters\n        ----------\n        categories : array-like\n        fastpath : bool\n            Whether to skip nan and uniqueness checks\n\n        Returns\n        -------\n        categories : Index\n        '
        from pandas.core.indexes.base import Index
        if ((not fastpath) and (not is_list_like(categories))):
            raise TypeError(f"Parameter 'categories' must be list-like, was {repr(categories)}")
        elif (not isinstance(categories, ABCIndex)):
            categories = Index(categories, tupleize_cols=False)
        if (not fastpath):
            if categories.hasnans:
                raise ValueError('Categorical categories cannot be null')
            if (not categories.is_unique):
                raise ValueError('Categorical categories must be unique')
        if isinstance(categories, ABCCategoricalIndex):
            categories = categories.categories
        return categories

    def update_dtype(self, dtype):
        '\n        Returns a CategoricalDtype with categories and ordered taken from dtype\n        if specified, otherwise falling back to self if unspecified\n\n        Parameters\n        ----------\n        dtype : CategoricalDtype\n\n        Returns\n        -------\n        new_dtype : CategoricalDtype\n        '
        if (isinstance(dtype, str) and (dtype == 'category')):
            return self
        elif (not self.is_dtype(dtype)):
            raise ValueError(f'a CategoricalDtype must be passed to perform an update, got {repr(dtype)}')
        else:
            dtype = cast(CategoricalDtype, dtype)
        new_categories = (dtype.categories if (dtype.categories is not None) else self.categories)
        new_ordered = (dtype.ordered if (dtype.ordered is not None) else self.ordered)
        return CategoricalDtype(new_categories, new_ordered)

    @property
    def categories(self):
        '\n        An ``Index`` containing the unique categories allowed.\n        '
        return self._categories

    @property
    def ordered(self):
        '\n        Whether the categories have an ordered relationship.\n        '
        return self._ordered

    @property
    def _is_boolean(self):
        from pandas.core.dtypes.common import is_bool_dtype
        return is_bool_dtype(self.categories)

    def _get_common_dtype(self, dtypes):
        from pandas.core.arrays.sparse import SparseDtype
        if all((isinstance(x, CategoricalDtype) for x in dtypes)):
            first = dtypes[0]
            if all(((first == other) for other in dtypes[1:])):
                return first
        non_init_cats = [(isinstance(x, CategoricalDtype) and (x.categories is None)) for x in dtypes]
        if all(non_init_cats):
            return self
        elif any(non_init_cats):
            return None
        dtypes = [(x.subtype if isinstance(x, SparseDtype) else x) for x in dtypes]
        non_cat_dtypes = [(x.categories.dtype if isinstance(x, CategoricalDtype) else x) for x in dtypes]
        from pandas.core.dtypes.cast import find_common_type
        return find_common_type(non_cat_dtypes)

@register_extension_dtype
class DatetimeTZDtype(PandasExtensionDtype):
    '\n    An ExtensionDtype for timezone-aware datetime data.\n\n    **This is not an actual numpy dtype**, but a duck type.\n\n    Parameters\n    ----------\n    unit : str, default "ns"\n        The precision of the datetime data. Currently limited\n        to ``"ns"``.\n    tz : str, int, or datetime.tzinfo\n        The timezone.\n\n    Attributes\n    ----------\n    unit\n    tz\n\n    Methods\n    -------\n    None\n\n    Raises\n    ------\n    pytz.UnknownTimeZoneError\n        When the requested timezone cannot be found.\n\n    Examples\n    --------\n    >>> pd.DatetimeTZDtype(tz=\'UTC\')\n    datetime64[ns, UTC]\n\n    >>> pd.DatetimeTZDtype(tz=\'dateutil/US/Central\')\n    datetime64[ns, tzfile(\'/usr/share/zoneinfo/US/Central\')]\n    '
    type = Timestamp
    kind = 'M'
    str = '|M8[ns]'
    num = 101
    base = np.dtype('M8[ns]')
    na_value = NaT
    _metadata = ('unit', 'tz')
    _match = re.compile('(datetime64|M8)\\[(?P<unit>.+), (?P<tz>.+)\\]')
    _cache = {}

    def __init__(self, unit='ns', tz=None):
        if isinstance(unit, DatetimeTZDtype):
            (unit, tz) = (unit.unit, unit.tz)
        if (unit != 'ns'):
            if (isinstance(unit, str) and (tz is None)):
                result = type(self).construct_from_string(unit)
                unit = result.unit
                tz = result.tz
                msg = f"Passing a dtype alias like 'datetime64[ns, {tz}]' to DatetimeTZDtype is no longer supported. Use 'DatetimeTZDtype.construct_from_string()' instead."
                raise ValueError(msg)
            else:
                raise ValueError('DatetimeTZDtype only supports ns units')
        if tz:
            tz = timezones.maybe_get_tz(tz)
            tz = timezones.tz_standardize(tz)
        elif (tz is not None):
            raise pytz.UnknownTimeZoneError(tz)
        if (tz is None):
            raise TypeError("A 'tz' is required.")
        self._unit = unit
        self._tz = tz

    @property
    def unit(self):
        '\n        The precision of the datetime data.\n        '
        return self._unit

    @property
    def tz(self):
        '\n        The timezone.\n        '
        return self._tz

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        from pandas.core.arrays import DatetimeArray
        return DatetimeArray

    @classmethod
    def construct_from_string(cls, string):
        "\n        Construct a DatetimeTZDtype from a string.\n\n        Parameters\n        ----------\n        string : str\n            The string alias for this DatetimeTZDtype.\n            Should be formatted like ``datetime64[ns, <tz>]``,\n            where ``<tz>`` is the timezone name.\n\n        Examples\n        --------\n        >>> DatetimeTZDtype.construct_from_string('datetime64[ns, UTC]')\n        datetime64[ns, UTC]\n        "
        if (not isinstance(string, str)):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        match = cls._match.match(string)
        if match:
            d = match.groupdict()
            try:
                return cls(unit=d['unit'], tz=d['tz'])
            except (KeyError, TypeError, ValueError) as err:
                raise TypeError(msg) from err
        raise TypeError(msg)

    def __str__(self):
        return f'datetime64[{self.unit}, {self.tz}]'

    @property
    def name(self):
        'A string representation of the dtype.'
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, str):
            if other.startswith('M8['):
                other = ('datetime64[' + other[3:])
            return (other == self.name)
        return (isinstance(other, DatetimeTZDtype) and (self.unit == other.unit) and (str(self.tz) == str(other.tz)))

    def __setstate__(self, state):
        self._tz = state['tz']
        self._unit = state['unit']

@register_extension_dtype
class PeriodDtype(dtypes.PeriodDtypeBase, PandasExtensionDtype):
    "\n    An ExtensionDtype for Period data.\n\n    **This is not an actual numpy dtype**, but a duck type.\n\n    Parameters\n    ----------\n    freq : str or DateOffset\n        The frequency of this PeriodDtype.\n\n    Attributes\n    ----------\n    freq\n\n    Methods\n    -------\n    None\n\n    Examples\n    --------\n    >>> pd.PeriodDtype(freq='D')\n    period[D]\n\n    >>> pd.PeriodDtype(freq=pd.offsets.MonthEnd())\n    period[M]\n    "
    type = Period
    kind = 'O'
    str = '|O08'
    base = np.dtype('O')
    num = 102
    _metadata = ('freq',)
    _match = re.compile('(P|p)eriod\\[(?P<freq>.+)\\]')
    _cache = {}

    def __new__(cls, freq=None):
        '\n        Parameters\n        ----------\n        freq : frequency\n        '
        if isinstance(freq, PeriodDtype):
            return freq
        elif (freq is None):
            u = dtypes.PeriodDtypeBase.__new__(cls, (- 10000))
            u._freq = None
            return u
        if (not isinstance(freq, BaseOffset)):
            freq = cls._parse_dtype_strict(freq)
        try:
            return cls._cache[freq.freqstr]
        except KeyError:
            dtype_code = freq._period_dtype_code
            u = dtypes.PeriodDtypeBase.__new__(cls, dtype_code)
            u._freq = freq
            cls._cache[freq.freqstr] = u
            return u

    def __reduce__(self):
        return (type(self), (self.freq,))

    @property
    def freq(self):
        '\n        The frequency object of this PeriodDtype.\n        '
        return self._freq

    @classmethod
    def _parse_dtype_strict(cls, freq):
        if isinstance(freq, str):
            if (freq.startswith('period[') or freq.startswith('Period[')):
                m = cls._match.search(freq)
                if (m is not None):
                    freq = m.group('freq')
            freq = to_offset(freq)
            if (freq is not None):
                return freq
        raise ValueError('could not construct PeriodDtype')

    @classmethod
    def construct_from_string(cls, string):
        '\n        Strict construction from a string, raise a TypeError if not\n        possible\n        '
        if ((isinstance(string, str) and (string.startswith('period[') or string.startswith('Period['))) or isinstance(string, BaseOffset)):
            try:
                return cls(freq=string)
            except ValueError:
                pass
        if isinstance(string, str):
            msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        else:
            msg = f"'construct_from_string' expects a string, got {type(string)}"
        raise TypeError(msg)

    def __str__(self):
        return self.name

    @property
    def name(self):
        return f'period[{self.freq.freqstr}]'

    @property
    def na_value(self):
        return NaT

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, str):
            return ((other == self.name) or (other == self.name.title()))
        return (isinstance(other, PeriodDtype) and (self.freq == other.freq))

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __setstate__(self, state):
        self._freq = state['freq']

    @classmethod
    def is_dtype(cls, dtype):
        '\n        Return a boolean if we if the passed type is an actual dtype that we\n        can match (via string or type)\n        '
        if isinstance(dtype, str):
            if (dtype.startswith('period[') or dtype.startswith('Period[')):
                try:
                    if (cls._parse_dtype_strict(dtype) is not None):
                        return True
                    else:
                        return False
                except ValueError:
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        from pandas.core.arrays import PeriodArray
        return PeriodArray

    def __from_arrow__(self, array):
        '\n        Construct PeriodArray from pyarrow Array/ChunkedArray.\n        '
        import pyarrow
        from pandas.core.arrays import PeriodArray
        from pandas.core.arrays._arrow_utils import pyarrow_array_to_numpy_and_mask
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        results = []
        for arr in chunks:
            (data, mask) = pyarrow_array_to_numpy_and_mask(arr, dtype='int64')
            parr = PeriodArray(data.copy(), freq=self.freq, copy=False)
            parr[(~ mask)] = NaT
            results.append(parr)
        return PeriodArray._concat_same_type(results)

@register_extension_dtype
class IntervalDtype(PandasExtensionDtype):
    "\n    An ExtensionDtype for Interval data.\n\n    **This is not an actual numpy dtype**, but a duck type.\n\n    Parameters\n    ----------\n    subtype : str, np.dtype\n        The dtype of the Interval bounds.\n\n    Attributes\n    ----------\n    subtype\n\n    Methods\n    -------\n    None\n\n    Examples\n    --------\n    >>> pd.IntervalDtype(subtype='int64')\n    interval[int64]\n    "
    name = 'interval'
    kind = 'O'
    str = '|O08'
    base = np.dtype('O')
    num = 103
    _metadata = ('subtype',)
    _match = re.compile('(I|i)nterval\\[(?P<subtype>.+)\\]')
    _cache = {}

    def __new__(cls, subtype=None):
        from pandas.core.dtypes.common import is_string_dtype, pandas_dtype
        if isinstance(subtype, IntervalDtype):
            return subtype
        elif (subtype is None):
            u = object.__new__(cls)
            u._subtype = None
            return u
        elif (isinstance(subtype, str) and (subtype.lower() == 'interval')):
            subtype = None
        else:
            if isinstance(subtype, str):
                m = cls._match.search(subtype)
                if (m is not None):
                    subtype = m.group('subtype')
            try:
                subtype = pandas_dtype(subtype)
            except TypeError as err:
                raise TypeError('could not construct IntervalDtype') from err
        if (CategoricalDtype.is_dtype(subtype) or is_string_dtype(subtype)):
            msg = 'category, object, and string subtypes are not supported for IntervalDtype'
            raise TypeError(msg)
        try:
            return cls._cache[str(subtype)]
        except KeyError:
            u = object.__new__(cls)
            u._subtype = subtype
            cls._cache[str(subtype)] = u
            return u

    @property
    def subtype(self):
        '\n        The dtype of the Interval bounds.\n        '
        return self._subtype

    @classmethod
    def construct_array_type(cls):
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        from pandas.core.arrays import IntervalArray
        return IntervalArray

    @classmethod
    def construct_from_string(cls, string):
        '\n        attempt to construct this type from a string, raise a TypeError\n        if its not possible\n        '
        if (not isinstance(string, str)):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if ((string.lower() == 'interval') or (cls._match.search(string) is not None)):
            return cls(string)
        msg = f'''Cannot construct a 'IntervalDtype' from '{string}'.

Incorrectly formatted string passed to constructor. Valid formats include Interval or Interval[dtype] where dtype is numeric, datetime, or timedelta'''
        raise TypeError(msg)

    @property
    def type(self):
        return Interval

    def __str__(self):
        if (self.subtype is None):
            return 'interval'
        return f'interval[{self.subtype}]'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, str):
            return (other.lower() in (self.name.lower(), str(self).lower()))
        elif (not isinstance(other, IntervalDtype)):
            return False
        elif ((self.subtype is None) or (other.subtype is None)):
            return True
        else:
            from pandas.core.dtypes.common import is_dtype_equal
            return is_dtype_equal(self.subtype, other.subtype)

    def __setstate__(self, state):
        self._subtype = state['subtype']

    @classmethod
    def is_dtype(cls, dtype):
        '\n        Return a boolean if we if the passed type is an actual dtype that we\n        can match (via string or type)\n        '
        if isinstance(dtype, str):
            if dtype.lower().startswith('interval'):
                try:
                    if (cls.construct_from_string(dtype) is not None):
                        return True
                    else:
                        return False
                except (ValueError, TypeError):
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    def __from_arrow__(self, array):
        '\n        Construct IntervalArray from pyarrow Array/ChunkedArray.\n        '
        import pyarrow
        from pandas.core.arrays import IntervalArray
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        results = []
        for arr in chunks:
            left = np.asarray(arr.storage.field('left'), dtype=self.subtype)
            right = np.asarray(arr.storage.field('right'), dtype=self.subtype)
            iarr = IntervalArray.from_arrays(left, right, closed=array.type.closed)
            results.append(iarr)
        return IntervalArray._concat_same_type(results)

    def _get_common_dtype(self, dtypes):
        if (not all((isinstance(x, IntervalDtype) for x in dtypes))):
            return None
        from pandas.core.dtypes.cast import find_common_type
        common = find_common_type([cast('IntervalDtype', x).subtype for x in dtypes])
        if (common == object):
            return np.dtype(object)
        return IntervalDtype(common)
