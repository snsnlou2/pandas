
from copy import copy as copy_func
from datetime import datetime
from itertools import zip_longest
import operator
from typing import TYPE_CHECKING, Any, Callable, FrozenSet, Hashable, List, NewType, Optional, Sequence, Set, Tuple, TypeVar, Union, cast
import warnings
import numpy as np
from pandas._libs import algos as libalgos, index as libindex, lib
import pandas._libs.join as libjoin
from pandas._libs.lib import is_datetime_array, no_default
from pandas._libs.tslibs import IncompatibleFrequency, OutOfBoundsDatetime, Timestamp
from pandas._libs.tslibs.timezones import tz_compare
from pandas._typing import AnyArrayLike, ArrayLike, Dtype, DtypeObj, Label, Shape, final
from pandas.compat.numpy import function as nv
from pandas.errors import DuplicateLabelError, InvalidIndexError
from pandas.util._decorators import Appender, cache_readonly, doc
from pandas.core.dtypes.cast import find_common_type, maybe_cast_to_integer_array, maybe_promote, validate_numeric_casting
from pandas.core.dtypes.common import ensure_int64, ensure_object, ensure_platform_int, is_bool_dtype, is_categorical_dtype, is_datetime64_any_dtype, is_dtype_equal, is_extension_array_dtype, is_float, is_float_dtype, is_hashable, is_integer, is_integer_dtype, is_interval_dtype, is_iterator, is_list_like, is_object_dtype, is_period_dtype, is_scalar, is_signed_integer_dtype, is_timedelta64_dtype, is_unsigned_integer_dtype, needs_i8_conversion, pandas_dtype, validate_all_hashable
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, IntervalDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCDatetimeIndex, ABCMultiIndex, ABCPandasArray, ABCPeriodIndex, ABCSeries, ABCTimedeltaIndex
from pandas.core.dtypes.missing import array_equivalent, isna
from pandas.core import missing, ops
from pandas.core.accessor import CachedAccessor
import pandas.core.algorithms as algos
from pandas.core.arrays import Categorical, ExtensionArray
from pandas.core.arrays.datetimes import tz_to_dtype, validate_tz_from_dtype
from pandas.core.base import IndexOpsMixin, PandasObject
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexers import deprecate_ndim_indexing
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops import get_op_result_name
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import ensure_key_mapped, nargsort
from pandas.core.strings import StringMethods
from pandas.io.formats.printing import PrettyDict, default_pprint, format_object_attrs, format_object_summary, pprint_thing
if TYPE_CHECKING:
    from pandas import MultiIndex, RangeIndex, Series
__all__ = ['Index']
_unsortable_types = frozenset(('mixed', 'mixed-integer'))
_index_doc_kwargs = {'klass': 'Index', 'inplace': '', 'target_klass': 'Index', 'raises_section': '', 'unique': 'Index', 'duplicated': 'np.ndarray'}
_index_shared_docs = {}
str_t = str
_o_dtype = np.dtype(object)
_Identity = NewType('_Identity', object)

def _new_Index(cls, d):
    "\n    This is called upon unpickling, rather than the default which doesn't\n    have arguments and breaks __new__.\n    "
    if issubclass(cls, ABCPeriodIndex):
        from pandas.core.indexes.period import _new_PeriodIndex
        return _new_PeriodIndex(cls, **d)
    if issubclass(cls, ABCMultiIndex):
        if (('labels' in d) and ('codes' not in d)):
            d['codes'] = d.pop('labels')
    return cls.__new__(cls, **d)
_IndexT = TypeVar('_IndexT', bound='Index')

class Index(IndexOpsMixin, PandasObject):
    "\n    Immutable sequence used for indexing and alignment. The basic object\n    storing axis labels for all pandas objects.\n\n    Parameters\n    ----------\n    data : array-like (1-dimensional)\n    dtype : NumPy dtype (default: object)\n        If dtype is None, we find the dtype that best fits the data.\n        If an actual dtype is provided, we coerce to that dtype if it's safe.\n        Otherwise, an error will be raised.\n    copy : bool\n        Make a copy of input ndarray.\n    name : object\n        Name to be stored in the index.\n    tupleize_cols : bool (default: True)\n        When True, attempt to create a MultiIndex if possible.\n\n    See Also\n    --------\n    RangeIndex : Index implementing a monotonic integer range.\n    CategoricalIndex : Index of :class:`Categorical` s.\n    MultiIndex : A multi-level, or hierarchical Index.\n    IntervalIndex : An Index of :class:`Interval` s.\n    DatetimeIndex : Index of datetime64 data.\n    TimedeltaIndex : Index of timedelta64 data.\n    PeriodIndex : Index of Period data.\n    Int64Index : A special case of :class:`Index` with purely integer labels.\n    UInt64Index : A special case of :class:`Index` with purely unsigned integer labels.\n    Float64Index : A special case of :class:`Index` with purely float labels.\n\n    Notes\n    -----\n    An Index instance can **only** contain hashable objects\n\n    Examples\n    --------\n    >>> pd.Index([1, 2, 3])\n    Int64Index([1, 2, 3], dtype='int64')\n\n    >>> pd.Index(list('abc'))\n    Index(['a', 'b', 'c'], dtype='object')\n    "
    _hidden_attrs = ((PandasObject._hidden_attrs | IndexOpsMixin._hidden_attrs) | frozenset(['contains', 'set_value']))
    _join_precedence = 1

    def _left_indexer_unique(self, left, right):
        return libjoin.left_join_indexer_unique(left, right)

    def _left_indexer(self, left, right):
        return libjoin.left_join_indexer(left, right)

    def _inner_indexer(self, left, right):
        return libjoin.inner_join_indexer(left, right)

    def _outer_indexer(self, left, right):
        return libjoin.outer_join_indexer(left, right)
    _typ = 'index'
    _id = None
    _name = None
    _no_setting_name = False
    _comparables = ['name']
    _attributes = ['name']
    _is_numeric_dtype = False
    _can_hold_na = True
    _can_hold_strings = True
    _defer_to_indexing = False
    _engine_type = libindex.ObjectEngine
    _supports_partial_string_indexing = False
    _accessors = {'str'}
    str = CachedAccessor('str', StringMethods)

    def __new__(cls, data=None, dtype=None, copy=False, name=None, tupleize_cols=True, **kwargs):
        if kwargs:
            warnings.warn("Passing keywords other than 'data', 'dtype', 'copy', 'name', 'tupleize_cols' is deprecated and will raise TypeError in a future version.  Use the specific Index subclass directly instead", FutureWarning, stacklevel=2)
        from pandas.core.indexes.range import RangeIndex
        name = maybe_extract_name(name, data, cls)
        if (dtype is not None):
            dtype = pandas_dtype(dtype)
        if ('tz' in kwargs):
            tz = kwargs.pop('tz')
            validate_tz_from_dtype(dtype, tz)
            dtype = tz_to_dtype(tz)
        if isinstance(data, ABCPandasArray):
            data = data.to_numpy()
        data_dtype = getattr(data, 'dtype', None)
        if isinstance(data, RangeIndex):
            return RangeIndex(start=data, copy=copy, dtype=dtype, name=name)
        elif isinstance(data, range):
            return RangeIndex.from_range(data, dtype=dtype, name=name)
        elif (is_categorical_dtype(data_dtype) or is_categorical_dtype(dtype)):
            from pandas.core.indexes.category import CategoricalIndex
            return _maybe_asobject(dtype, CategoricalIndex, data, copy, name, **kwargs)
        elif (is_interval_dtype(data_dtype) or is_interval_dtype(dtype)):
            from pandas.core.indexes.interval import IntervalIndex
            return _maybe_asobject(dtype, IntervalIndex, data, copy, name, **kwargs)
        elif (is_datetime64_any_dtype(data_dtype) or is_datetime64_any_dtype(dtype)):
            from pandas import DatetimeIndex
            return _maybe_asobject(dtype, DatetimeIndex, data, copy, name, **kwargs)
        elif (is_timedelta64_dtype(data_dtype) or is_timedelta64_dtype(dtype)):
            from pandas import TimedeltaIndex
            return _maybe_asobject(dtype, TimedeltaIndex, data, copy, name, **kwargs)
        elif (is_period_dtype(data_dtype) or is_period_dtype(dtype)):
            from pandas import PeriodIndex
            return _maybe_asobject(dtype, PeriodIndex, data, copy, name, **kwargs)
        elif (is_extension_array_dtype(data_dtype) or is_extension_array_dtype(dtype)):
            if (not ((dtype is None) or is_object_dtype(dtype))):
                ea_cls = dtype.construct_array_type()
                data = ea_cls._from_sequence(data, dtype=dtype, copy=False)
            else:
                data = np.asarray(data, dtype=object)
            data = data.astype(object)
            return Index(data, dtype=object, copy=copy, name=name, **kwargs)
        elif isinstance(data, (np.ndarray, Index, ABCSeries)):
            if (dtype is not None):
                data = _maybe_cast_with_dtype(data, dtype, copy)
                dtype = data.dtype
            if (data.dtype.kind in ['i', 'u', 'f']):
                arr = data
            else:
                arr = com.asarray_tuplesafe(data, dtype=object)
                if (dtype is None):
                    new_data = _maybe_cast_data_without_dtype(arr)
                    new_dtype = new_data.dtype
                    return cls(new_data, dtype=new_dtype, copy=copy, name=name, **kwargs)
            klass = cls._dtype_to_subclass(arr.dtype)
            arr = klass._ensure_array(arr, dtype, copy)
            if kwargs:
                raise TypeError(f'Unexpected keyword arguments {repr(set(kwargs))}')
            return klass._simple_new(arr, name)
        elif is_scalar(data):
            raise cls._scalar_data_error(data)
        elif hasattr(data, '__array__'):
            return Index(np.asarray(data), dtype=dtype, copy=copy, name=name, **kwargs)
        else:
            if (tupleize_cols and is_list_like(data)):
                if is_iterator(data):
                    data = list(data)
                if (data and all((isinstance(e, tuple) for e in data))):
                    from pandas.core.indexes.multi import MultiIndex
                    return MultiIndex.from_tuples(data, names=(name or kwargs.get('names')))
            subarr = com.asarray_tuplesafe(data, dtype=object)
            return Index(subarr, dtype=dtype, copy=copy, name=name, **kwargs)

    @classmethod
    def _ensure_array(cls, data, dtype, copy):
        '\n        Ensure we have a valid array to pass to _simple_new.\n        '
        if (data.ndim > 1):
            raise ValueError('Index data must be 1-dimensional')
        if copy:
            data = data.copy()
        return data

    @classmethod
    def _dtype_to_subclass(cls, dtype):
        if (isinstance(dtype, DatetimeTZDtype) or (dtype == np.dtype('M8[ns]'))):
            from pandas import DatetimeIndex
            return DatetimeIndex
        elif (dtype == 'm8[ns]'):
            from pandas import TimedeltaIndex
            return TimedeltaIndex
        elif isinstance(dtype, CategoricalDtype):
            from pandas import CategoricalIndex
            return CategoricalIndex
        elif isinstance(dtype, IntervalDtype):
            from pandas import IntervalIndex
            return IntervalIndex
        elif isinstance(dtype, PeriodDtype):
            from pandas import PeriodIndex
            return PeriodIndex
        elif is_float_dtype(dtype):
            from pandas import Float64Index
            return Float64Index
        elif is_unsigned_integer_dtype(dtype):
            from pandas import UInt64Index
            return UInt64Index
        elif is_signed_integer_dtype(dtype):
            from pandas import Int64Index
            return Int64Index
        elif (dtype == object):
            return Index
        raise NotImplementedError(dtype)
    "\n    NOTE for new Index creation:\n\n    - _simple_new: It returns new Index with the same type as the caller.\n      All metadata (such as name) must be provided by caller's responsibility.\n      Using _shallow_copy is recommended because it fills these metadata\n      otherwise specified.\n\n    - _shallow_copy: It returns new Index with the same type (using\n      _simple_new), but fills caller's metadata otherwise specified. Passed\n      kwargs will overwrite corresponding metadata.\n\n    See each method's docstring.\n    "

    @property
    def asi8(self):
        '\n        Integer representation of the values.\n\n        Returns\n        -------\n        ndarray\n            An ndarray with int64 dtype.\n        '
        warnings.warn('Index.asi8 is deprecated and will be removed in a future version', FutureWarning, stacklevel=2)
        return None

    @classmethod
    def _simple_new(cls, values, name=None):
        '\n        We require that we have a dtype compat for the values. If we are passed\n        a non-dtype compat, then coerce using the constructor.\n\n        Must be careful not to recurse.\n        '
        assert isinstance(values, np.ndarray), type(values)
        result = object.__new__(cls)
        result._data = values
        result._index_data = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        return result

    @cache_readonly
    def _constructor(self):
        return type(self)

    @final
    def _maybe_check_unique(self):
        "\n        Check that an Index has no duplicates.\n\n        This is typically only called via\n        `NDFrame.flags.allows_duplicate_labels.setter` when it's set to\n        True (duplicates aren't allowed).\n\n        Raises\n        ------\n        DuplicateLabelError\n            When the index is not unique.\n        "
        if (not self.is_unique):
            msg = 'Index has duplicates.'
            duplicates = self._format_duplicate_message()
            msg += f'''
{duplicates}'''
            raise DuplicateLabelError(msg)

    @final
    def _format_duplicate_message(self):
        "\n        Construct the DataFrame for a DuplicateLabelError.\n\n        This returns a DataFrame indicating the labels and positions\n        of duplicates in an index. This should only be called when it's\n        already known that duplicates are present.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['a', 'b', 'a'])\n        >>> idx._format_duplicate_message()\n            positions\n        label\n        a        [0, 2]\n        "
        from pandas import Series
        duplicates = self[self.duplicated(keep='first')].unique()
        assert len(duplicates)
        out = Series(np.arange(len(self))).groupby(self).agg(list)[duplicates]
        if (self.nlevels == 1):
            out = out.rename_axis('label')
        return out.to_frame(name='positions')

    @final
    def _get_attributes_dict(self):
        '\n        Return an attributes dict for my class.\n        '
        return {k: getattr(self, k, None) for k in self._attributes}

    def _shallow_copy(self, values=None, name=no_default):
        "\n        Create a new Index with the same class as the caller, don't copy the\n        data, use the same object attributes with passed in attributes taking\n        precedence.\n\n        *this is an internal non-public method*\n\n        Parameters\n        ----------\n        values : the values to create the new Index, optional\n        name : Label, defaults to self.name\n        "
        name = (self.name if (name is no_default) else name)
        if (values is not None):
            return self._simple_new(values, name=name)
        result = self._simple_new(self._values, name=name)
        result._cache = self._cache
        return result

    @final
    def is_(self, other):
        '\n        More flexible, faster check like ``is`` but that works through views.\n\n        Note: this is *not* the same as ``Index.identical()``, which checks\n        that metadata is also the same.\n\n        Parameters\n        ----------\n        other : object\n            Other object to compare against.\n\n        Returns\n        -------\n        bool\n            True if both have same underlying data, False otherwise.\n\n        See Also\n        --------\n        Index.identical : Works like ``Index.is_`` but also checks metadata.\n        '
        if (self is other):
            return True
        elif (not hasattr(other, '_id')):
            return False
        elif ((self._id is None) or (other._id is None)):
            return False
        else:
            return (self._id is other._id)

    @final
    def _reset_identity(self):
        '\n        Initializes or resets ``_id`` attribute with new object.\n        '
        self._id = _Identity(object())

    @final
    def _cleanup(self):
        self._engine.clear_mapping()

    @cache_readonly
    def _engine(self):
        target_values = self._get_engine_target()
        return self._engine_type((lambda : target_values), len(self))

    @cache_readonly
    def _dir_additions_for_owner(self):
        "\n        Add the string-like labels to the owner dataframe/series dir output.\n\n        If this is a MultiIndex, it's first level values are used.\n        "
        return {c for c in self.unique(level=0)[:100] if (isinstance(c, str) and c.isidentifier())}

    def __len__(self):
        '\n        Return the length of the Index.\n        '
        return len(self._data)

    def __array__(self, dtype=None):
        '\n        The array interface, return my values.\n        '
        return np.asarray(self._data, dtype=dtype)

    def __array_wrap__(self, result, context=None):
        '\n        Gets called after a ufunc and other functions.\n        '
        result = lib.item_from_zerodim(result)
        if (is_bool_dtype(result) or lib.is_scalar(result) or (np.ndim(result) > 1)):
            return result
        attrs = self._get_attributes_dict()
        return Index(result, **attrs)

    @cache_readonly
    def dtype(self):
        '\n        Return the dtype object of the underlying data.\n        '
        return self._data.dtype

    @final
    def ravel(self, order='C'):
        '\n        Return an ndarray of the flattened values of the underlying data.\n\n        Returns\n        -------\n        numpy.ndarray\n            Flattened array.\n\n        See Also\n        --------\n        numpy.ndarray.ravel : Return a flattened array.\n        '
        warnings.warn('Index.ravel returning ndarray is deprecated; in a future version this will return a view on self.', FutureWarning, stacklevel=2)
        values = self._get_engine_target()
        return values.ravel(order=order)

    def view(self, cls=None):
        if ((cls is not None) and (not hasattr(cls, '_typ'))):
            result = self._data.view(cls)
        else:
            result = self._shallow_copy()
        if isinstance(result, Index):
            result._id = self._id
        return result

    def astype(self, dtype, copy=True):
        "\n        Create an Index with values cast to dtypes.\n\n        The class of a new Index is determined by dtype. When conversion is\n        impossible, a TypeError exception is raised.\n\n        Parameters\n        ----------\n        dtype : numpy dtype or pandas type\n            Note that any signed integer `dtype` is treated as ``'int64'``,\n            and any unsigned integer `dtype` is treated as ``'uint64'``,\n            regardless of the size.\n        copy : bool, default True\n            By default, astype always returns a newly allocated object.\n            If copy is set to False and internal requirements on dtype are\n            satisfied, the original data is used to create a new Index\n            or the original Index is returned.\n\n        Returns\n        -------\n        Index\n            Index with values cast to specified dtype.\n        "
        if (dtype is not None):
            dtype = pandas_dtype(dtype)
        if is_dtype_equal(self.dtype, dtype):
            return (self.copy() if copy else self)
        elif is_categorical_dtype(dtype):
            from pandas.core.indexes.category import CategoricalIndex
            return CategoricalIndex(self._values, name=self.name, dtype=dtype, copy=copy)
        elif is_extension_array_dtype(dtype):
            return Index(np.asarray(self), name=self.name, dtype=dtype, copy=copy)
        try:
            casted = self._values.astype(dtype, copy=copy)
        except (TypeError, ValueError) as err:
            raise TypeError(f'Cannot cast {type(self).__name__} to dtype {dtype}') from err
        return Index(casted, name=self.name, dtype=dtype)
    _index_shared_docs['take'] = "\n        Return a new %(klass)s of the values selected by the indices.\n\n        For internal compatibility with numpy arrays.\n\n        Parameters\n        ----------\n        indices : list\n            Indices to be taken.\n        axis : int, optional\n            The axis over which to select values, always 0.\n        allow_fill : bool, default True\n        fill_value : bool, default None\n            If allow_fill=True and fill_value is not None, indices specified by\n            -1 is regarded as NA. If Index doesn't hold NA, raise ValueError.\n\n        Returns\n        -------\n        numpy.ndarray\n            Elements of given indices.\n\n        See Also\n        --------\n        numpy.ndarray.take: Return an array formed from the\n            elements of a at the given indices.\n        "

    @Appender((_index_shared_docs['take'] % _index_doc_kwargs))
    def take(self, indices, axis=0, allow_fill=True, fill_value=None, **kwargs):
        if kwargs:
            nv.validate_take((), kwargs)
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)
        taken = algos.take(self._values, indices, allow_fill=allow_fill, fill_value=self._na_value)
        return self._shallow_copy(taken)

    def _maybe_disallow_fill(self, allow_fill, fill_value, indices):
        '\n        We only use pandas-style take when allow_fill is True _and_\n        fill_value is not None.\n        '
        if (allow_fill and (fill_value is not None)):
            if self._can_hold_na:
                if (indices < (- 1)).any():
                    raise ValueError('When allow_fill=True and fill_value is not None, all indices must be >= -1')
            else:
                cls_name = type(self).__name__
                raise ValueError(f'Unable to fill values because {cls_name} cannot contain NA')
        else:
            allow_fill = False
        return allow_fill
    _index_shared_docs['repeat'] = "\n        Repeat elements of a %(klass)s.\n\n        Returns a new %(klass)s where each element of the current %(klass)s\n        is repeated consecutively a given number of times.\n\n        Parameters\n        ----------\n        repeats : int or array of ints\n            The number of repetitions for each element. This should be a\n            non-negative integer. Repeating 0 times will return an empty\n            %(klass)s.\n        axis : None\n            Must be ``None``. Has no effect but is accepted for compatibility\n            with numpy.\n\n        Returns\n        -------\n        repeated_index : %(klass)s\n            Newly created %(klass)s with repeated elements.\n\n        See Also\n        --------\n        Series.repeat : Equivalent function for Series.\n        numpy.repeat : Similar method for :class:`numpy.ndarray`.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['a', 'b', 'c'])\n        >>> idx\n        Index(['a', 'b', 'c'], dtype='object')\n        >>> idx.repeat(2)\n        Index(['a', 'a', 'b', 'b', 'c', 'c'], dtype='object')\n        >>> idx.repeat([1, 2, 3])\n        Index(['a', 'b', 'b', 'c', 'c', 'c'], dtype='object')\n        "

    @Appender((_index_shared_docs['repeat'] % _index_doc_kwargs))
    def repeat(self, repeats, axis=None):
        repeats = ensure_platform_int(repeats)
        nv.validate_repeat((), {'axis': axis})
        return self._shallow_copy(self._values.repeat(repeats))

    def copy(self, name=None, deep=False, dtype=None, names=None):
        '\n        Make a copy of this object.\n\n        Name and dtype sets those attributes on the new object.\n\n        Parameters\n        ----------\n        name : Label, optional\n            Set name for new object.\n        deep : bool, default False\n        dtype : numpy dtype or pandas type, optional\n            Set dtype for new object.\n\n            .. deprecated:: 1.2.0\n                use ``astype`` method instead.\n        names : list-like, optional\n            Kept for compatibility with MultiIndex. Should not be used.\n\n        Returns\n        -------\n        Index\n            Index refer to new object which is a copy of this object.\n\n        Notes\n        -----\n        In most cases, there should be no functional difference from using\n        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.\n        '
        name = self._validate_names(name=name, names=names, deep=deep)[0]
        if deep:
            new_index = self._shallow_copy(self._data.copy(), name=name)
        else:
            new_index = self._shallow_copy(name=name)
        if dtype:
            warnings.warn('parameter dtype is deprecated and will be removed in a future version. Use the astype method instead.', FutureWarning, stacklevel=2)
            new_index = new_index.astype(dtype)
        return new_index

    @final
    def __copy__(self, **kwargs):
        return self.copy(**kwargs)

    @final
    def __deepcopy__(self, memo=None):
        '\n        Parameters\n        ----------\n        memo, default None\n            Standard signature. Unused\n        '
        return self.copy(deep=True)

    def __repr__(self):
        '\n        Return a string representation for this object.\n        '
        klass_name = type(self).__name__
        data = self._format_data()
        attrs = self._format_attrs()
        space = self._format_space()
        attrs_str = [f'{k}={v}' for (k, v) in attrs]
        prepr = f',{space}'.join(attrs_str)
        if (data is None):
            data = ''
        return f'{klass_name}({data}{prepr})'

    def _format_space(self):
        return ' '

    @property
    def _formatter_func(self):
        '\n        Return the formatter function.\n        '
        return default_pprint

    def _format_data(self, name=None):
        '\n        Return the formatted data as a unicode string.\n        '
        is_justify = True
        if (self.inferred_type == 'string'):
            is_justify = False
        elif (self.inferred_type == 'categorical'):
            if is_object_dtype(self.categories):
                is_justify = False
        return format_object_summary(self, self._formatter_func, is_justify=is_justify, name=name)

    def _format_attrs(self):
        '\n        Return a list of tuples of the (attr,formatted_value).\n        '
        return format_object_attrs(self)

    def _mpl_repr(self):
        return self.values

    def format(self, name=False, formatter=None, na_rep='NaN'):
        '\n        Render a string representation of the Index.\n        '
        header = []
        if name:
            header.append((pprint_thing(self.name, escape_chars=('\t', '\r', '\n')) if (self.name is not None) else ''))
        if (formatter is not None):
            return (header + list(self.map(formatter)))
        return self._format_with_header(header, na_rep=na_rep)

    def _format_with_header(self, header, na_rep='NaN'):
        from pandas.io.formats.format import format_array
        values = self._values
        if is_object_dtype(values.dtype):
            values = lib.maybe_convert_objects(values, safe=1)
            result = [pprint_thing(x, escape_chars=('\t', '\r', '\n')) for x in values]
            mask = isna(values)
            if mask.any():
                result_arr = np.array(result)
                result_arr[mask] = na_rep
                result = result_arr.tolist()
        else:
            result = trim_front(format_array(values, None, justify='left'))
        return (header + result)

    def to_native_types(self, slicer=None, **kwargs):
        '\n        Format specified values of `self` and return them.\n\n        .. deprecated:: 1.2.0\n\n        Parameters\n        ----------\n        slicer : int, array-like\n            An indexer into `self` that specifies which values\n            are used in the formatting process.\n        kwargs : dict\n            Options for specifying how the values should be formatted.\n            These options include the following:\n\n            1) na_rep : str\n                The value that serves as a placeholder for NULL values\n            2) quoting : bool or None\n                Whether or not there are quoted values in `self`\n            3) date_format : str\n                The format used to represent date-like values.\n\n        Returns\n        -------\n        numpy.ndarray\n            Formatted values.\n        '
        warnings.warn("The 'to_native_types' method is deprecated and will be removed in a future version. Use 'astype(str)' instead.", FutureWarning, stacklevel=2)
        values = self
        if (slicer is not None):
            values = values[slicer]
        return values._format_native_types(**kwargs)

    def _format_native_types(self, na_rep='', quoting=None, **kwargs):
        '\n        Actually format specific types of the index.\n        '
        mask = isna(self)
        if ((not self.is_object()) and (not quoting)):
            values = np.asarray(self).astype(str)
        else:
            values = np.array(self, dtype=object, copy=True)
        values[mask] = na_rep
        return values

    def _summary(self, name=None):
        '\n        Return a summarized representation.\n\n        Parameters\n        ----------\n        name : str\n            name to use in the summary representation\n\n        Returns\n        -------\n        String with a summarized representation of the index\n        '
        if (len(self) > 0):
            head = self[0]
            if (hasattr(head, 'format') and (not isinstance(head, str))):
                head = head.format()
            tail = self[(- 1)]
            if (hasattr(tail, 'format') and (not isinstance(tail, str))):
                tail = tail.format()
            index_summary = f', {head} to {tail}'
        else:
            index_summary = ''
        if (name is None):
            name = type(self).__name__
        return f'{name}: {len(self)} entries{index_summary}'

    def to_flat_index(self):
        '\n        Identity method.\n\n        .. versionadded:: 0.24.0\n\n        This is implemented for compatibility with subclass implementations\n        when chaining.\n\n        Returns\n        -------\n        pd.Index\n            Caller.\n\n        See Also\n        --------\n        MultiIndex.to_flat_index : Subclass implementation.\n        '
        return self

    def to_series(self, index=None, name=None):
        "\n        Create a Series with both index and values equal to the index keys.\n\n        Useful with map for returning an indexer based on an index.\n\n        Parameters\n        ----------\n        index : Index, optional\n            Index of resulting Series. If None, defaults to original index.\n        name : str, optional\n            Name of resulting Series. If None, defaults to name of original\n            index.\n\n        Returns\n        -------\n        Series\n            The dtype will be based on the type of the Index values.\n\n        See Also\n        --------\n        Index.to_frame : Convert an Index to a DataFrame.\n        Series.to_frame : Convert Series to DataFrame.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')\n\n        By default, the original Index and original name is reused.\n\n        >>> idx.to_series()\n        animal\n        Ant      Ant\n        Bear    Bear\n        Cow      Cow\n        Name: animal, dtype: object\n\n        To enforce a new Index, specify new labels to ``index``:\n\n        >>> idx.to_series(index=[0, 1, 2])\n        0     Ant\n        1    Bear\n        2     Cow\n        Name: animal, dtype: object\n\n        To override the name of the resulting column, specify `name`:\n\n        >>> idx.to_series(name='zoo')\n        animal\n        Ant      Ant\n        Bear    Bear\n        Cow      Cow\n        Name: zoo, dtype: object\n        "
        from pandas import Series
        if (index is None):
            index = self._shallow_copy()
        if (name is None):
            name = self.name
        return Series(self._values.copy(), index=index, name=name)

    def to_frame(self, index=True, name=None):
        "\n        Create a DataFrame with a column containing the Index.\n\n        .. versionadded:: 0.24.0\n\n        Parameters\n        ----------\n        index : bool, default True\n            Set the index of the returned DataFrame as the original Index.\n\n        name : object, default None\n            The passed name should substitute for the index name (if it has\n            one).\n\n        Returns\n        -------\n        DataFrame\n            DataFrame containing the original Index data.\n\n        See Also\n        --------\n        Index.to_series : Convert an Index to a Series.\n        Series.to_frame : Convert Series to DataFrame.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')\n        >>> idx.to_frame()\n               animal\n        animal\n        Ant       Ant\n        Bear     Bear\n        Cow       Cow\n\n        By default, the original Index is reused. To enforce a new Index:\n\n        >>> idx.to_frame(index=False)\n            animal\n        0   Ant\n        1  Bear\n        2   Cow\n\n        To override the name of the resulting column, specify `name`:\n\n        >>> idx.to_frame(index=False, name='zoo')\n            zoo\n        0   Ant\n        1  Bear\n        2   Cow\n        "
        from pandas import DataFrame
        if (name is None):
            name = (self.name or 0)
        result = DataFrame({name: self._values.copy()})
        if index:
            result.index = self
        return result

    @property
    def name(self):
        '\n        Return Index or MultiIndex name.\n        '
        return self._name

    @name.setter
    def name(self, value):
        if self._no_setting_name:
            raise RuntimeError("Cannot set name on a level of a MultiIndex. Use 'MultiIndex.set_names' instead.")
        maybe_extract_name(value, None, type(self))
        self._name = value

    @final
    def _validate_names(self, name=None, names=None, deep=False):
        "\n        Handles the quirks of having a singular 'name' parameter for general\n        Index and plural 'names' parameter for MultiIndex.\n        "
        from copy import deepcopy
        if ((names is not None) and (name is not None)):
            raise TypeError('Can only provide one of `names` and `name`')
        elif ((names is None) and (name is None)):
            new_names = (deepcopy(self.names) if deep else self.names)
        elif (names is not None):
            if (not is_list_like(names)):
                raise TypeError('Must pass list-like as `names`.')
            new_names = names
        elif (not is_list_like(name)):
            new_names = [name]
        else:
            new_names = name
        if (len(new_names) != len(self.names)):
            raise ValueError(f'Length of new names must be {len(self.names)}, got {len(new_names)}')
        validate_all_hashable(*new_names, error_name=f'{type(self).__name__}.name')
        return new_names

    def _get_names(self):
        return FrozenList((self.name,))

    def _set_names(self, values, level=None):
        '\n        Set new names on index. Each name has to be a hashable type.\n\n        Parameters\n        ----------\n        values : str or sequence\n            name(s) to set\n        level : int, level name, or sequence of int/level names (default None)\n            If the index is a MultiIndex (hierarchical), level(s) to set (None\n            for all levels).  Otherwise level must be None\n\n        Raises\n        ------\n        TypeError if each name is not hashable.\n        '
        if (not is_list_like(values)):
            raise ValueError('Names must be a list-like')
        if (len(values) != 1):
            raise ValueError(f'Length of new names must be 1, got {len(values)}')
        validate_all_hashable(*values, error_name=f'{type(self).__name__}.name')
        self._name = values[0]
    names = property(fset=_set_names, fget=_get_names)

    @final
    def set_names(self, names, level=None, inplace=False):
        "\n        Set Index or MultiIndex name.\n\n        Able to set new names partially and by level.\n\n        Parameters\n        ----------\n        names : label or list of label\n            Name(s) to set.\n        level : int, label or list of int or label, optional\n            If the index is a MultiIndex, level(s) to set (None for all\n            levels). Otherwise level must be None.\n        inplace : bool, default False\n            Modifies the object directly, instead of creating a new Index or\n            MultiIndex.\n\n        Returns\n        -------\n        Index or None\n            The same type as the caller or None if ``inplace=True``.\n\n        See Also\n        --------\n        Index.rename : Able to set new names without level.\n\n        Examples\n        --------\n        >>> idx = pd.Index([1, 2, 3, 4])\n        >>> idx\n        Int64Index([1, 2, 3, 4], dtype='int64')\n        >>> idx.set_names('quarter')\n        Int64Index([1, 2, 3, 4], dtype='int64', name='quarter')\n\n        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],\n        ...                                   [2018, 2019]])\n        >>> idx\n        MultiIndex([('python', 2018),\n                    ('python', 2019),\n                    ( 'cobra', 2018),\n                    ( 'cobra', 2019)],\n                   )\n        >>> idx.set_names(['kind', 'year'], inplace=True)\n        >>> idx\n        MultiIndex([('python', 2018),\n                    ('python', 2019),\n                    ( 'cobra', 2018),\n                    ( 'cobra', 2019)],\n                   names=['kind', 'year'])\n        >>> idx.set_names('species', level=0)\n        MultiIndex([('python', 2018),\n                    ('python', 2019),\n                    ( 'cobra', 2018),\n                    ( 'cobra', 2019)],\n                   names=['species', 'year'])\n        "
        if ((level is not None) and (not isinstance(self, ABCMultiIndex))):
            raise ValueError('Level must be None for non-MultiIndex')
        if ((level is not None) and (not is_list_like(level)) and is_list_like(names)):
            raise TypeError('Names must be a string when a single level is provided.')
        if ((not is_list_like(names)) and (level is None) and (self.nlevels > 1)):
            raise TypeError('Must pass list-like as `names`.')
        if (not is_list_like(names)):
            names = [names]
        if ((level is not None) and (not is_list_like(level))):
            level = [level]
        if inplace:
            idx = self
        else:
            idx = self._shallow_copy()
        idx._set_names(names, level=level)
        if (not inplace):
            return idx

    def rename(self, name, inplace=False):
        "\n        Alter Index or MultiIndex name.\n\n        Able to set new names without level. Defaults to returning new index.\n        Length of names must match number of levels in MultiIndex.\n\n        Parameters\n        ----------\n        name : label or list of labels\n            Name(s) to set.\n        inplace : bool, default False\n            Modifies the object directly, instead of creating a new Index or\n            MultiIndex.\n\n        Returns\n        -------\n        Index or None\n            The same type as the caller or None if ``inplace=True``.\n\n        See Also\n        --------\n        Index.set_names : Able to set new names partially and by level.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['A', 'C', 'A', 'B'], name='score')\n        >>> idx.rename('grade')\n        Index(['A', 'C', 'A', 'B'], dtype='object', name='grade')\n\n        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],\n        ...                                   [2018, 2019]],\n        ...                                   names=['kind', 'year'])\n        >>> idx\n        MultiIndex([('python', 2018),\n                    ('python', 2019),\n                    ( 'cobra', 2018),\n                    ( 'cobra', 2019)],\n                   names=['kind', 'year'])\n        >>> idx.rename(['species', 'year'])\n        MultiIndex([('python', 2018),\n                    ('python', 2019),\n                    ( 'cobra', 2018),\n                    ( 'cobra', 2019)],\n                   names=['species', 'year'])\n        >>> idx.rename('species')\n        Traceback (most recent call last):\n        TypeError: Must pass list-like as `names`.\n        "
        return self.set_names([name], inplace=inplace)

    @property
    def nlevels(self):
        '\n        Number of levels.\n        '
        return 1

    def _sort_levels_monotonic(self):
        '\n        Compat with MultiIndex.\n        '
        return self

    @final
    def _validate_index_level(self, level):
        '\n        Validate index level.\n\n        For single-level Index getting level number is a no-op, but some\n        verification must be done like in MultiIndex.\n\n        '
        if isinstance(level, int):
            if ((level < 0) and (level != (- 1))):
                raise IndexError(f'Too many levels: Index has only 1 level, {level} is not a valid level number')
            elif (level > 0):
                raise IndexError(f'Too many levels: Index has only 1 level, not {(level + 1)}')
        elif (level != self.name):
            raise KeyError(f'Requested level ({level}) does not match index name ({self.name})')

    def _get_level_number(self, level):
        self._validate_index_level(level)
        return 0

    def sortlevel(self, level=None, ascending=True, sort_remaining=None):
        '\n        For internal compatibility with the Index API.\n\n        Sort the Index. This is for compat with MultiIndex\n\n        Parameters\n        ----------\n        ascending : bool, default True\n            False to sort in descending order\n\n        level, sort_remaining are compat parameters\n\n        Returns\n        -------\n        Index\n        '
        if (not isinstance(ascending, (list, bool))):
            raise TypeError('ascending must be a single bool value ora list of bool values of length 1')
        if isinstance(ascending, list):
            if (len(ascending) != 1):
                raise TypeError('ascending must be a list of bool values of length 1')
            ascending = ascending[0]
        if (not isinstance(ascending, bool)):
            raise TypeError('ascending must be a bool value')
        return self.sort_values(return_indexer=True, ascending=ascending)

    def _get_level_values(self, level):
        "\n        Return an Index of values for requested level.\n\n        This is primarily useful to get an individual level of values from a\n        MultiIndex, but is provided on Index as well for compatibility.\n\n        Parameters\n        ----------\n        level : int or str\n            It is either the integer position or the name of the level.\n\n        Returns\n        -------\n        Index\n            Calling object, as there is only one level in the Index.\n\n        See Also\n        --------\n        MultiIndex.get_level_values : Get values for a level of a MultiIndex.\n\n        Notes\n        -----\n        For Index, level should be 0, since there are no multiple levels.\n\n        Examples\n        --------\n        >>> idx = pd.Index(list('abc'))\n        >>> idx\n        Index(['a', 'b', 'c'], dtype='object')\n\n        Get level values by supplying `level` as integer:\n\n        >>> idx.get_level_values(0)\n        Index(['a', 'b', 'c'], dtype='object')\n        "
        self._validate_index_level(level)
        return self
    get_level_values = _get_level_values

    @final
    def droplevel(self, level=0):
        "\n        Return index with requested level(s) removed.\n\n        If resulting index has only 1 level left, the result will be\n        of Index type, not MultiIndex.\n\n        Parameters\n        ----------\n        level : int, str, or list-like, default 0\n            If a string is given, must be the name of a level\n            If list-like, elements must be names or indexes of levels.\n\n        Returns\n        -------\n        Index or MultiIndex\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays(\n        ... [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z'])\n        >>> mi\n        MultiIndex([(1, 3, 5),\n                    (2, 4, 6)],\n                   names=['x', 'y', 'z'])\n\n        >>> mi.droplevel()\n        MultiIndex([(3, 5),\n                    (4, 6)],\n                   names=['y', 'z'])\n\n        >>> mi.droplevel(2)\n        MultiIndex([(1, 3),\n                    (2, 4)],\n                   names=['x', 'y'])\n\n        >>> mi.droplevel('z')\n        MultiIndex([(1, 3),\n                    (2, 4)],\n                   names=['x', 'y'])\n\n        >>> mi.droplevel(['x', 'y'])\n        Int64Index([5, 6], dtype='int64', name='z')\n        "
        if (not isinstance(level, (tuple, list))):
            level = [level]
        levnums = sorted((self._get_level_number(lev) for lev in level))[::(- 1)]
        return self._drop_level_numbers(levnums)

    def _drop_level_numbers(self, levnums):
        '\n        Drop MultiIndex levels by level _number_, not name.\n        '
        if (not levnums):
            return self
        if (len(levnums) >= self.nlevels):
            raise ValueError(f'Cannot remove {len(levnums)} levels from an index with {self.nlevels} levels: at least one level must be left.')
        self = cast('MultiIndex', self)
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)
        for i in levnums:
            new_levels.pop(i)
            new_codes.pop(i)
            new_names.pop(i)
        if (len(new_levels) == 1):
            mask = (new_codes[0] == (- 1))
            result = new_levels[0].take(new_codes[0])
            if mask.any():
                result = result.putmask(mask, np.nan)
            result._name = new_names[0]
            return result
        else:
            from pandas.core.indexes.multi import MultiIndex
            return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    def _get_grouper_for_level(self, mapper, level=None):
        '\n        Get index grouper corresponding to an index level\n\n        Parameters\n        ----------\n        mapper: Group mapping function or None\n            Function mapping index values to groups\n        level : int or None\n            Index level\n\n        Returns\n        -------\n        grouper : Index\n            Index of values to group on.\n        labels : ndarray of int or None\n            Array of locations in level_index.\n        uniques : Index or None\n            Index of unique values for level.\n        '
        assert ((level is None) or (level == 0))
        if (mapper is None):
            grouper = self
        else:
            grouper = self.map(mapper)
        return (grouper, None, None)

    @final
    @property
    def is_monotonic(self):
        '\n        Alias for is_monotonic_increasing.\n        '
        return self.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self):
        '\n        Return if the index is monotonic increasing (only equal or\n        increasing) values.\n\n        Examples\n        --------\n        >>> Index([1, 2, 3]).is_monotonic_increasing\n        True\n        >>> Index([1, 2, 2]).is_monotonic_increasing\n        True\n        >>> Index([1, 3, 2]).is_monotonic_increasing\n        False\n        '
        return self._engine.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        '\n        Return if the index is monotonic decreasing (only equal or\n        decreasing) values.\n\n        Examples\n        --------\n        >>> Index([3, 2, 1]).is_monotonic_decreasing\n        True\n        >>> Index([3, 2, 2]).is_monotonic_decreasing\n        True\n        >>> Index([3, 1, 2]).is_monotonic_decreasing\n        False\n        '
        return self._engine.is_monotonic_decreasing

    @property
    def _is_strictly_monotonic_increasing(self):
        '\n        Return if the index is strictly monotonic increasing\n        (only increasing) values.\n\n        Examples\n        --------\n        >>> Index([1, 2, 3])._is_strictly_monotonic_increasing\n        True\n        >>> Index([1, 2, 2])._is_strictly_monotonic_increasing\n        False\n        >>> Index([1, 3, 2])._is_strictly_monotonic_increasing\n        False\n        '
        return (self.is_unique and self.is_monotonic_increasing)

    @property
    def _is_strictly_monotonic_decreasing(self):
        '\n        Return if the index is strictly monotonic decreasing\n        (only decreasing) values.\n\n        Examples\n        --------\n        >>> Index([3, 2, 1])._is_strictly_monotonic_decreasing\n        True\n        >>> Index([3, 2, 2])._is_strictly_monotonic_decreasing\n        False\n        >>> Index([3, 1, 2])._is_strictly_monotonic_decreasing\n        False\n        '
        return (self.is_unique and self.is_monotonic_decreasing)

    @cache_readonly
    def is_unique(self):
        '\n        Return if the index has unique values.\n        '
        return self._engine.is_unique

    @property
    def has_duplicates(self):
        '\n        Check if the Index has duplicate values.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index has duplicate values.\n\n        Examples\n        --------\n        >>> idx = pd.Index([1, 5, 7, 7])\n        >>> idx.has_duplicates\n        True\n\n        >>> idx = pd.Index([1, 5, 7])\n        >>> idx.has_duplicates\n        False\n\n        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",\n        ...                 "Watermelon"]).astype("category")\n        >>> idx.has_duplicates\n        True\n\n        >>> idx = pd.Index(["Orange", "Apple",\n        ...                 "Watermelon"]).astype("category")\n        >>> idx.has_duplicates\n        False\n        '
        return (not self.is_unique)

    @final
    def is_boolean(self):
        '\n        Check if the Index only consists of booleans.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index only consists of booleans.\n\n        See Also\n        --------\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n        is_mixed : Check if the Index holds data with mixed data types.\n\n        Examples\n        --------\n        >>> idx = pd.Index([True, False, True])\n        >>> idx.is_boolean()\n        True\n\n        >>> idx = pd.Index(["True", "False", "True"])\n        >>> idx.is_boolean()\n        False\n\n        >>> idx = pd.Index([True, False, "True"])\n        >>> idx.is_boolean()\n        False\n        '
        return (self.inferred_type in ['boolean'])

    @final
    def is_integer(self):
        '\n        Check if the Index only consists of integers.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index only consists of integers.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n        is_mixed : Check if the Index holds data with mixed data types.\n\n        Examples\n        --------\n        >>> idx = pd.Index([1, 2, 3, 4])\n        >>> idx.is_integer()\n        True\n\n        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])\n        >>> idx.is_integer()\n        False\n\n        >>> idx = pd.Index(["Apple", "Mango", "Watermelon"])\n        >>> idx.is_integer()\n        False\n        '
        return (self.inferred_type in ['integer'])

    @final
    def is_floating(self):
        '\n        Check if the Index is a floating type.\n\n        The Index may consist of only floats, NaNs, or a mix of floats,\n        integers, or NaNs.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index only consists of only consists of floats, NaNs, or\n            a mix of floats, integers, or NaNs.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n        is_mixed : Check if the Index holds data with mixed data types.\n\n        Examples\n        --------\n        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])\n        >>> idx.is_floating()\n        True\n\n        >>> idx = pd.Index([1.0, 2.0, np.nan, 4.0])\n        >>> idx.is_floating()\n        True\n\n        >>> idx = pd.Index([1, 2, 3, 4, np.nan])\n        >>> idx.is_floating()\n        True\n\n        >>> idx = pd.Index([1, 2, 3, 4])\n        >>> idx.is_floating()\n        False\n        '
        return (self.inferred_type in ['floating', 'mixed-integer-float', 'integer-na'])

    @final
    def is_numeric(self):
        '\n        Check if the Index only consists of numeric data.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index only consists of numeric data.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n        is_mixed : Check if the Index holds data with mixed data types.\n\n        Examples\n        --------\n        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])\n        >>> idx.is_numeric()\n        True\n\n        >>> idx = pd.Index([1, 2, 3, 4.0])\n        >>> idx.is_numeric()\n        True\n\n        >>> idx = pd.Index([1, 2, 3, 4])\n        >>> idx.is_numeric()\n        True\n\n        >>> idx = pd.Index([1, 2, 3, 4.0, np.nan])\n        >>> idx.is_numeric()\n        True\n\n        >>> idx = pd.Index([1, 2, 3, 4.0, np.nan, "Apple"])\n        >>> idx.is_numeric()\n        False\n        '
        return (self.inferred_type in ['integer', 'floating'])

    @final
    def is_object(self):
        '\n        Check if the Index is of the object dtype.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index is of the object dtype.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n        is_mixed : Check if the Index holds data with mixed data types.\n\n        Examples\n        --------\n        >>> idx = pd.Index(["Apple", "Mango", "Watermelon"])\n        >>> idx.is_object()\n        True\n\n        >>> idx = pd.Index(["Apple", "Mango", 2.0])\n        >>> idx.is_object()\n        True\n\n        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",\n        ...                 "Watermelon"]).astype("category")\n        >>> idx.is_object()\n        False\n\n        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])\n        >>> idx.is_object()\n        False\n        '
        return is_object_dtype(self.dtype)

    @final
    def is_categorical(self):
        '\n        Check if the Index holds categorical data.\n\n        Returns\n        -------\n        bool\n            True if the Index is categorical.\n\n        See Also\n        --------\n        CategoricalIndex : Index for categorical data.\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_interval : Check if the Index holds Interval objects.\n        is_mixed : Check if the Index holds data with mixed data types.\n\n        Examples\n        --------\n        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",\n        ...                 "Watermelon"]).astype("category")\n        >>> idx.is_categorical()\n        True\n\n        >>> idx = pd.Index([1, 3, 5, 7])\n        >>> idx.is_categorical()\n        False\n\n        >>> s = pd.Series(["Peter", "Victor", "Elisabeth", "Mar"])\n        >>> s\n        0        Peter\n        1       Victor\n        2    Elisabeth\n        3          Mar\n        dtype: object\n        >>> s.index.is_categorical()\n        False\n        '
        return (self.inferred_type in ['categorical'])

    @final
    def is_interval(self):
        '\n        Check if the Index holds Interval objects.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index holds Interval objects.\n\n        See Also\n        --------\n        IntervalIndex : Index for Interval objects.\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_mixed : Check if the Index holds data with mixed data types.\n\n        Examples\n        --------\n        >>> idx = pd.Index([pd.Interval(left=0, right=5),\n        ...                 pd.Interval(left=5, right=10)])\n        >>> idx.is_interval()\n        True\n\n        >>> idx = pd.Index([1, 3, 5, 7])\n        >>> idx.is_interval()\n        False\n        '
        return (self.inferred_type in ['interval'])

    @final
    def is_mixed(self):
        "\n        Check if the Index holds data with mixed data types.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index holds data with mixed data types.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['a', np.nan, 'b'])\n        >>> idx.is_mixed()\n        True\n\n        >>> idx = pd.Index([1.0, 2.0, 3.0, 5.0])\n        >>> idx.is_mixed()\n        False\n        "
        warnings.warn('Index.is_mixed is deprecated and will be removed in a future version. Check index.inferred_type directly instead.', FutureWarning, stacklevel=2)
        return (self.inferred_type in ['mixed'])

    @final
    def holds_integer(self):
        '\n        Whether the type is an integer type.\n        '
        return (self.inferred_type in ['integer', 'mixed-integer'])

    @cache_readonly
    def inferred_type(self):
        '\n        Return a string of the type inferred from the values.\n        '
        return lib.infer_dtype(self._values, skipna=False)

    @cache_readonly
    def _is_all_dates(self):
        '\n        Whether or not the index values only consist of dates.\n        '
        return is_datetime_array(ensure_object(self._values))

    @cache_readonly
    def is_all_dates(self):
        '\n        Whether or not the index values only consist of dates.\n        '
        warnings.warn('Index.is_all_dates is deprecated, will be removed in a future version.  check index.inferred_type instead', FutureWarning, stacklevel=2)
        return self._is_all_dates

    def __reduce__(self):
        d = {'data': self._data}
        d.update(self._get_attributes_dict())
        return (_new_Index, (type(self), d), None)
    _na_value = np.nan
    'The expected NA value to use with this index.'

    @cache_readonly
    def _isnan(self):
        '\n        Return if each value is NaN.\n        '
        if self._can_hold_na:
            return isna(self)
        else:
            values = np.empty(len(self), dtype=np.bool_)
            values.fill(False)
            return values

    @cache_readonly
    @final
    def _nan_idxs(self):
        if self._can_hold_na:
            return self._isnan.nonzero()[0]
        else:
            return np.array([], dtype=np.intp)

    @cache_readonly
    def hasnans(self):
        '\n        Return if I have any nans; enables various perf speedups.\n        '
        if self._can_hold_na:
            return bool(self._isnan.any())
        else:
            return False

    @final
    def isna(self):
        "\n        Detect missing values.\n\n        Return a boolean same-sized object indicating if the values are NA.\n        NA values, such as ``None``, :attr:`numpy.NaN` or :attr:`pd.NaT`, get\n        mapped to ``True`` values.\n        Everything else get mapped to ``False`` values. Characters such as\n        empty strings `''` or :attr:`numpy.inf` are not considered NA values\n        (unless you set ``pandas.options.mode.use_inf_as_na = True``).\n\n        Returns\n        -------\n        numpy.ndarray\n            A boolean array of whether my values are NA.\n\n        See Also\n        --------\n        Index.notna : Boolean inverse of isna.\n        Index.dropna : Omit entries with missing values.\n        isna : Top-level isna.\n        Series.isna : Detect missing values in Series object.\n\n        Examples\n        --------\n        Show which entries in a pandas.Index are NA. The result is an\n        array.\n\n        >>> idx = pd.Index([5.2, 6.0, np.NaN])\n        >>> idx\n        Float64Index([5.2, 6.0, nan], dtype='float64')\n        >>> idx.isna()\n        array([False, False,  True])\n\n        Empty strings are not considered NA values. None is considered an NA\n        value.\n\n        >>> idx = pd.Index(['black', '', 'red', None])\n        >>> idx\n        Index(['black', '', 'red', None], dtype='object')\n        >>> idx.isna()\n        array([False, False, False,  True])\n\n        For datetimes, `NaT` (Not a Time) is considered as an NA value.\n\n        >>> idx = pd.DatetimeIndex([pd.Timestamp('1940-04-25'),\n        ...                         pd.Timestamp(''), None, pd.NaT])\n        >>> idx\n        DatetimeIndex(['1940-04-25', 'NaT', 'NaT', 'NaT'],\n                      dtype='datetime64[ns]', freq=None)\n        >>> idx.isna()\n        array([False,  True,  True,  True])\n        "
        return self._isnan
    isnull = isna

    @final
    def notna(self):
        "\n        Detect existing (non-missing) values.\n\n        Return a boolean same-sized object indicating if the values are not NA.\n        Non-missing values get mapped to ``True``. Characters such as empty\n        strings ``''`` or :attr:`numpy.inf` are not considered NA values\n        (unless you set ``pandas.options.mode.use_inf_as_na = True``).\n        NA values, such as None or :attr:`numpy.NaN`, get mapped to ``False``\n        values.\n\n        Returns\n        -------\n        numpy.ndarray\n            Boolean array to indicate which entries are not NA.\n\n        See Also\n        --------\n        Index.notnull : Alias of notna.\n        Index.isna: Inverse of notna.\n        notna : Top-level notna.\n\n        Examples\n        --------\n        Show which entries in an Index are not NA. The result is an\n        array.\n\n        >>> idx = pd.Index([5.2, 6.0, np.NaN])\n        >>> idx\n        Float64Index([5.2, 6.0, nan], dtype='float64')\n        >>> idx.notna()\n        array([ True,  True, False])\n\n        Empty strings are not considered NA values. None is considered a NA\n        value.\n\n        >>> idx = pd.Index(['black', '', 'red', None])\n        >>> idx\n        Index(['black', '', 'red', None], dtype='object')\n        >>> idx.notna()\n        array([ True,  True,  True, False])\n        "
        return (~ self.isna())
    notnull = notna

    def fillna(self, value=None, downcast=None):
        "\n        Fill NA/NaN values with the specified value.\n\n        Parameters\n        ----------\n        value : scalar\n            Scalar value to use to fill holes (e.g. 0).\n            This value cannot be a list-likes.\n        downcast : dict, default is None\n            A dict of item->dtype of what to downcast if possible,\n            or the string 'infer' which will try to downcast to an appropriate\n            equal type (e.g. float64 to int64 if possible).\n\n        Returns\n        -------\n        Index\n\n        See Also\n        --------\n        DataFrame.fillna : Fill NaN values of a DataFrame.\n        Series.fillna : Fill NaN Values of a Series.\n        "
        value = self._require_scalar(value)
        if self.hasnans:
            result = self.putmask(self._isnan, value)
            if (downcast is None):
                return Index(result, name=self.name)
        return self._shallow_copy()

    def dropna(self, how='any'):
        "\n        Return Index without NA/NaN values.\n\n        Parameters\n        ----------\n        how : {'any', 'all'}, default 'any'\n            If the Index is a MultiIndex, drop the value when any or all levels\n            are NaN.\n\n        Returns\n        -------\n        Index\n        "
        if (how not in ('any', 'all')):
            raise ValueError(f'invalid how option: {how}')
        if self.hasnans:
            return self._shallow_copy(self._values[(~ self._isnan)])
        return self._shallow_copy()

    def unique(self, level=None):
        '\n        Return unique values in the index.\n\n        Unique values are returned in order of appearance, this does NOT sort.\n\n        Parameters\n        ----------\n        level : int or str, optional, default None\n            Only return values from specified level (for MultiIndex).\n\n        Returns\n        -------\n        Index without duplicates\n\n        See Also\n        --------\n        unique : Numpy array of unique values in that column.\n        Series.unique : Return unique values of Series object.\n        '
        if (level is not None):
            self._validate_index_level(level)
        if self.is_unique:
            return self._shallow_copy()
        result = super().unique()
        return self._shallow_copy(result)

    @final
    def drop_duplicates(self, keep='first'):
        "\n        Return Index with duplicate values removed.\n\n        Parameters\n        ----------\n        keep : {'first', 'last', ``False``}, default 'first'\n            - 'first' : Drop duplicates except for the first occurrence.\n            - 'last' : Drop duplicates except for the last occurrence.\n            - ``False`` : Drop all duplicates.\n\n        Returns\n        -------\n        deduplicated : Index\n\n        See Also\n        --------\n        Series.drop_duplicates : Equivalent method on Series.\n        DataFrame.drop_duplicates : Equivalent method on DataFrame.\n        Index.duplicated : Related method on Index, indicating duplicate\n            Index values.\n\n        Examples\n        --------\n        Generate an pandas.Index with duplicate values.\n\n        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])\n\n        The `keep` parameter controls  which duplicate values are removed.\n        The value 'first' keeps the first occurrence for each\n        set of duplicated entries. The default value of keep is 'first'.\n\n        >>> idx.drop_duplicates(keep='first')\n        Index(['lama', 'cow', 'beetle', 'hippo'], dtype='object')\n\n        The value 'last' keeps the last occurrence for each set of duplicated\n        entries.\n\n        >>> idx.drop_duplicates(keep='last')\n        Index(['cow', 'beetle', 'lama', 'hippo'], dtype='object')\n\n        The value ``False`` discards all sets of duplicated entries.\n\n        >>> idx.drop_duplicates(keep=False)\n        Index(['cow', 'beetle', 'hippo'], dtype='object')\n        "
        if self.is_unique:
            return self._shallow_copy()
        return super().drop_duplicates(keep=keep)

    def duplicated(self, keep='first'):
        "\n        Indicate duplicate index values.\n\n        Duplicated values are indicated as ``True`` values in the resulting\n        array. Either all duplicates, all except the first, or all except the\n        last occurrence of duplicates can be indicated.\n\n        Parameters\n        ----------\n        keep : {'first', 'last', False}, default 'first'\n            The value or values in a set of duplicates to mark as missing.\n\n            - 'first' : Mark duplicates as ``True`` except for the first\n              occurrence.\n            - 'last' : Mark duplicates as ``True`` except for the last\n              occurrence.\n            - ``False`` : Mark all duplicates as ``True``.\n\n        Returns\n        -------\n        numpy.ndarray\n\n        See Also\n        --------\n        Series.duplicated : Equivalent method on pandas.Series.\n        DataFrame.duplicated : Equivalent method on pandas.DataFrame.\n        Index.drop_duplicates : Remove duplicate values from Index.\n\n        Examples\n        --------\n        By default, for each set of duplicated values, the first occurrence is\n        set to False and all others to True:\n\n        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])\n        >>> idx.duplicated()\n        array([False, False,  True, False,  True])\n\n        which is equivalent to\n\n        >>> idx.duplicated(keep='first')\n        array([False, False,  True, False,  True])\n\n        By using 'last', the last occurrence of each set of duplicated values\n        is set on False and all others on True:\n\n        >>> idx.duplicated(keep='last')\n        array([ True, False,  True, False, False])\n\n        By setting keep on ``False``, all duplicates are True:\n\n        >>> idx.duplicated(keep=False)\n        array([ True, False,  True, False,  True])\n        "
        if self.is_unique:
            return np.zeros(len(self), dtype=bool)
        return super().duplicated(keep=keep)

    def _get_unique_index(self, dropna=False):
        '\n        Returns an index containing unique values.\n\n        Parameters\n        ----------\n        dropna : bool, default False\n            If True, NaN values are dropped.\n\n        Returns\n        -------\n        uniques : index\n        '
        if (self.is_unique and (not dropna)):
            return self
        if (not self.is_unique):
            values = self.unique()
            if (not isinstance(self, ABCMultiIndex)):
                values = values._data
        else:
            values = self._values
        if (dropna and (not isinstance(self, ABCMultiIndex))):
            if self.hasnans:
                values = values[(~ isna(values))]
        return self._shallow_copy(values)

    def __iadd__(self, other):
        return (self + other)

    @final
    def __and__(self, other):
        warnings.warn('Index.__and__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__and__.  Use index.intersection(other) instead', FutureWarning, stacklevel=2)
        return self.intersection(other)

    @final
    def __or__(self, other):
        warnings.warn('Index.__or__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__or__.  Use index.union(other) instead', FutureWarning, stacklevel=2)
        return self.union(other)

    @final
    def __xor__(self, other):
        warnings.warn('Index.__xor__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__xor__.  Use index.symmetric_difference(other) instead', FutureWarning, stacklevel=2)
        return self.symmetric_difference(other)

    @final
    def __nonzero__(self):
        raise ValueError(f'The truth value of a {type(self).__name__} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().')
    __bool__ = __nonzero__

    def _get_reconciled_name_object(self, other):
        '\n        If the result of a set operation will be self,\n        return self, unless the name changes, in which\n        case make a shallow copy of self.\n        '
        name = get_op_result_name(self, other)
        if (self.name != name):
            return self.rename(name)
        return self

    @final
    def _validate_sort_keyword(self, sort):
        if (sort not in [None, False]):
            raise ValueError(f"The 'sort' keyword only takes the values of None or False; {sort} was passed.")

    @final
    def union(self, other, sort=None):
        '\n        Form the union of two Index objects.\n\n        If the Index objects are incompatible, both Index objects will be\n        cast to dtype(\'object\') first.\n\n            .. versionchanged:: 0.25.0\n\n        Parameters\n        ----------\n        other : Index or array-like\n        sort : bool or None, default None\n            Whether to sort the resulting Index.\n\n            * None : Sort the result, except when\n\n              1. `self` and `other` are equal.\n              2. `self` or `other` has length 0.\n              3. Some values in `self` or `other` cannot be compared.\n                 A RuntimeWarning is issued in this case.\n\n            * False : do not sort the result.\n\n            .. versionadded:: 0.24.0\n\n            .. versionchanged:: 0.24.1\n\n               Changed the default value from ``True`` to ``None``\n               (without change in behaviour).\n\n        Returns\n        -------\n        union : Index\n\n        Examples\n        --------\n        Union matching dtypes\n\n        >>> idx1 = pd.Index([1, 2, 3, 4])\n        >>> idx2 = pd.Index([3, 4, 5, 6])\n        >>> idx1.union(idx2)\n        Int64Index([1, 2, 3, 4, 5, 6], dtype=\'int64\')\n\n        Union mismatched dtypes\n\n        >>> idx1 = pd.Index([\'a\', \'b\', \'c\', \'d\'])\n        >>> idx2 = pd.Index([1, 2, 3, 4])\n        >>> idx1.union(idx2)\n        Index([\'a\', \'b\', \'c\', \'d\', 1, 2, 3, 4], dtype=\'object\')\n\n        MultiIndex case\n\n        >>> idx1 = pd.MultiIndex.from_arrays(\n        ...     [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]\n        ... )\n        >>> idx1\n        MultiIndex([(1,  \'Red\'),\n            (1, \'Blue\'),\n            (2,  \'Red\'),\n            (2, \'Blue\')],\n           )\n        >>> idx2 = pd.MultiIndex.from_arrays(\n        ...     [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]\n        ... )\n        >>> idx2\n        MultiIndex([(3,   \'Red\'),\n            (3, \'Green\'),\n            (2,   \'Red\'),\n            (2, \'Green\')],\n           )\n        >>> idx1.union(idx2)\n        MultiIndex([(1,  \'Blue\'),\n            (1,   \'Red\'),\n            (2,  \'Blue\'),\n            (2, \'Green\'),\n            (2,   \'Red\'),\n            (3, \'Green\'),\n            (3,   \'Red\')],\n           )\n        >>> idx1.union(idx2, sort=False)\n        MultiIndex([(1,   \'Red\'),\n            (1,  \'Blue\'),\n            (2,   \'Red\'),\n            (2,  \'Blue\'),\n            (3,   \'Red\'),\n            (3, \'Green\'),\n            (2, \'Green\')],\n           )\n        '
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        (other, result_name) = self._convert_can_do_setop(other)
        if (not is_dtype_equal(self.dtype, other.dtype)):
            if (isinstance(self, ABCMultiIndex) and (not is_object_dtype(unpack_nested_dtype(other)))):
                raise NotImplementedError('Can only union MultiIndex with MultiIndex or Index of tuples, try mi.to_flat_index().union(other) instead.')
            dtype = find_common_type([self.dtype, other.dtype])
            if (self._is_numeric_dtype and other._is_numeric_dtype):
                if (not (is_integer_dtype(self.dtype) and is_integer_dtype(other.dtype))):
                    dtype = 'float64'
                else:
                    dtype = object
            left = self.astype(dtype, copy=False)
            right = other.astype(dtype, copy=False)
            return left.union(right, sort=sort)
        elif ((not len(other)) or self.equals(other)):
            return self._get_reconciled_name_object(other)
        elif (not len(self)):
            return other._get_reconciled_name_object(self)
        result = self._union(other, sort=sort)
        return self._wrap_setop_result(other, result)

    def _union(self, other, sort):
        '\n        Specific union logic should go here. In subclasses, union behavior\n        should be overwritten here rather than in `self.union`.\n\n        Parameters\n        ----------\n        other : Index or array-like\n        sort : False or None, default False\n            Whether to sort the resulting index.\n\n            * False : do not sort the result.\n            * None : sort the result, except when `self` and `other` are equal\n              or when the values cannot be compared.\n\n        Returns\n        -------\n        Index\n        '
        lvals = self._values
        rvals = other._values
        if ((sort is None) and self.is_monotonic and other.is_monotonic):
            try:
                result = self._outer_indexer(lvals, rvals)[0]
            except TypeError:
                result = list(lvals)
                value_set = set(lvals)
                result.extend([x for x in rvals if (x not in value_set)])
                result = Index(result)._values
        else:
            if self.is_unique:
                indexer = self.get_indexer(other)
                missing = (indexer == (- 1)).nonzero()[0]
            else:
                missing = algos.unique1d(self.get_indexer_non_unique(other)[1])
            if (len(missing) > 0):
                other_diff = algos.take_nd(rvals, missing, allow_fill=False)
                result = concat_compat((lvals, other_diff))
            else:
                result = lvals
            if (sort is None):
                try:
                    result = algos.safe_sort(result)
                except TypeError as err:
                    warnings.warn(f'{err}, sort order is undefined for incomparable objects', RuntimeWarning, stacklevel=3)
        return result

    @final
    def _wrap_setop_result(self, other, result):
        if (isinstance(self, (ABCDatetimeIndex, ABCTimedeltaIndex)) and isinstance(result, np.ndarray)):
            result = type(self._data)._simple_new(result, dtype=self.dtype)
        elif (is_categorical_dtype(self.dtype) and isinstance(result, np.ndarray)):
            result = Categorical(result, dtype=self.dtype)
        name = get_op_result_name(self, other)
        if isinstance(result, Index):
            if (result.name != name):
                return result.rename(name)
            return result
        else:
            return self._shallow_copy(result, name=name)

    @final
    def intersection(self, other, sort=False):
        "\n        Form the intersection of two Index objects.\n\n        This returns a new Index with elements common to the index and `other`.\n\n        Parameters\n        ----------\n        other : Index or array-like\n        sort : False or None, default False\n            Whether to sort the resulting index.\n\n            * False : do not sort the result.\n            * None : sort the result, except when `self` and `other` are equal\n              or when the values cannot be compared.\n\n            .. versionadded:: 0.24.0\n\n            .. versionchanged:: 0.24.1\n\n               Changed the default from ``True`` to ``False``, to match\n               the behaviour of 0.23.4 and earlier.\n\n        Returns\n        -------\n        intersection : Index\n\n        Examples\n        --------\n        >>> idx1 = pd.Index([1, 2, 3, 4])\n        >>> idx2 = pd.Index([3, 4, 5, 6])\n        >>> idx1.intersection(idx2)\n        Int64Index([3, 4], dtype='int64')\n        "
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        (other, result_name) = self._convert_can_do_setop(other)
        if self.equals(other):
            if self.has_duplicates:
                return self.unique()._get_reconciled_name_object(other)
            return self._get_reconciled_name_object(other)
        elif (not self._should_compare(other)):
            if isinstance(self, ABCMultiIndex):
                return self[:0].rename(result_name)
            return Index([], name=result_name)
        elif (not is_dtype_equal(self.dtype, other.dtype)):
            dtype = find_common_type([self.dtype, other.dtype])
            this = self.astype(dtype, copy=False)
            other = other.astype(dtype, copy=False)
            return this.intersection(other, sort=sort)
        result = self._intersection(other, sort=sort)
        return self._wrap_setop_result(other, result)

    def _intersection(self, other, sort=False):
        '\n        intersection specialized to the case with matching dtypes.\n        '
        lvals = self._values
        rvals = other._values
        if (self.is_monotonic and other.is_monotonic):
            try:
                result = self._inner_indexer(lvals, rvals)[0]
            except TypeError:
                pass
            else:
                return algos.unique1d(result)
        try:
            indexer = other.get_indexer(lvals)
        except (InvalidIndexError, IncompatibleFrequency):
            (indexer, _) = other.get_indexer_non_unique(lvals)
        mask = (indexer != (- 1))
        indexer = indexer.take(mask.nonzero()[0])
        result = other.take(indexer).unique()._values
        if (sort is None):
            result = algos.safe_sort(result)
        assert Index(result).is_unique
        return result

    def difference(self, other, sort=None):
        "\n        Return a new Index with elements of index not in `other`.\n\n        This is the set difference of two Index objects.\n\n        Parameters\n        ----------\n        other : Index or array-like\n        sort : False or None, default None\n            Whether to sort the resulting index. By default, the\n            values are attempted to be sorted, but any TypeError from\n            incomparable elements is caught by pandas.\n\n            * None : Attempt to sort the result, but catch any TypeErrors\n              from comparing incomparable elements.\n            * False : Do not sort the result.\n\n            .. versionadded:: 0.24.0\n\n            .. versionchanged:: 0.24.1\n\n               Changed the default value from ``True`` to ``None``\n               (without change in behaviour).\n\n        Returns\n        -------\n        difference : Index\n\n        Examples\n        --------\n        >>> idx1 = pd.Index([2, 1, 3, 4])\n        >>> idx2 = pd.Index([3, 4, 5, 6])\n        >>> idx1.difference(idx2)\n        Int64Index([1, 2], dtype='int64')\n        >>> idx1.difference(idx2, sort=False)\n        Int64Index([2, 1], dtype='int64')\n        "
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        (other, result_name) = self._convert_can_do_setop(other)
        if self.equals(other):
            return self[:0].rename(result_name)
        if (len(other) == 0):
            return self.rename(result_name)
        result = self._difference(other, sort=sort)
        return self._wrap_setop_result(other, result)

    def _difference(self, other, sort):
        this = self._get_unique_index()
        indexer = this.get_indexer(other)
        indexer = indexer.take((indexer != (- 1)).nonzero()[0])
        label_diff = np.setdiff1d(np.arange(this.size), indexer, assume_unique=True)
        the_diff = this._values.take(label_diff)
        if (sort is None):
            try:
                the_diff = algos.safe_sort(the_diff)
            except TypeError:
                pass
        return the_diff

    def symmetric_difference(self, other, result_name=None, sort=None):
        "\n        Compute the symmetric difference of two Index objects.\n\n        Parameters\n        ----------\n        other : Index or array-like\n        result_name : str\n        sort : False or None, default None\n            Whether to sort the resulting index. By default, the\n            values are attempted to be sorted, but any TypeError from\n            incomparable elements is caught by pandas.\n\n            * None : Attempt to sort the result, but catch any TypeErrors\n              from comparing incomparable elements.\n            * False : Do not sort the result.\n\n            .. versionadded:: 0.24.0\n\n            .. versionchanged:: 0.24.1\n\n               Changed the default value from ``True`` to ``None``\n               (without change in behaviour).\n\n        Returns\n        -------\n        symmetric_difference : Index\n\n        Notes\n        -----\n        ``symmetric_difference`` contains elements that appear in either\n        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by\n        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates\n        dropped.\n\n        Examples\n        --------\n        >>> idx1 = pd.Index([1, 2, 3, 4])\n        >>> idx2 = pd.Index([2, 3, 4, 5])\n        >>> idx1.symmetric_difference(idx2)\n        Int64Index([1, 5], dtype='int64')\n\n        You can also use the ``^`` operator:\n\n        >>> idx1 ^ idx2\n        Int64Index([1, 5], dtype='int64')\n        "
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        (other, result_name_update) = self._convert_can_do_setop(other)
        if (result_name is None):
            result_name = result_name_update
        if (not self._should_compare(other)):
            return self.union(other).rename(result_name)
        elif (not is_dtype_equal(self.dtype, other.dtype)):
            dtype = find_common_type([self.dtype, other.dtype])
            this = self.astype(dtype, copy=False)
            that = other.astype(dtype, copy=False)
            return this.symmetric_difference(that, sort=sort).rename(result_name)
        this = self._get_unique_index()
        other = other._get_unique_index()
        indexer = this.get_indexer_for(other)
        common_indexer = indexer.take((indexer != (- 1)).nonzero()[0])
        left_indexer = np.setdiff1d(np.arange(this.size), common_indexer, assume_unique=True)
        left_diff = this._values.take(left_indexer)
        right_indexer = (indexer == (- 1)).nonzero()[0]
        right_diff = other._values.take(right_indexer)
        the_diff = concat_compat([left_diff, right_diff])
        if (sort is None):
            try:
                the_diff = algos.safe_sort(the_diff)
            except TypeError:
                pass
        return Index(the_diff, name=result_name)

    def _assert_can_do_setop(self, other):
        if (not is_list_like(other)):
            raise TypeError('Input must be Index or array-like')
        return True

    def _convert_can_do_setop(self, other):
        if (not isinstance(other, Index)):
            other = Index(other, name=self.name)
            result_name = self.name
        else:
            result_name = get_op_result_name(self, other)
        return (other, result_name)

    def get_loc(self, key, method=None, tolerance=None):
        "\n        Get integer location, slice or boolean mask for requested label.\n\n        Parameters\n        ----------\n        key : label\n        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional\n            * default: exact matches only.\n            * pad / ffill: find the PREVIOUS index value if no exact match.\n            * backfill / bfill: use NEXT index value if no exact match\n            * nearest: use the NEAREST index value if no exact match. Tied\n              distances are broken by preferring the larger index value.\n        tolerance : int or float, optional\n            Maximum distance from index value for inexact matches. The value of\n            the index at the matching location must satisfy the equation\n            ``abs(index[loc] - key) <= tolerance``.\n\n        Returns\n        -------\n        loc : int if unique index, slice if monotonic index, else mask\n\n        Examples\n        --------\n        >>> unique_index = pd.Index(list('abc'))\n        >>> unique_index.get_loc('b')\n        1\n\n        >>> monotonic_index = pd.Index(list('abbc'))\n        >>> monotonic_index.get_loc('b')\n        slice(1, 3, None)\n\n        >>> non_monotonic_index = pd.Index(list('abcb'))\n        >>> non_monotonic_index.get_loc('b')\n        array([False,  True, False,  True])\n        "
        if (method is None):
            if (tolerance is not None):
                raise ValueError('tolerance argument only valid if using pad, backfill or nearest lookups')
            casted_key = self._maybe_cast_indexer(key)
            try:
                return self._engine.get_loc(casted_key)
            except KeyError as err:
                raise KeyError(key) from err
        if (tolerance is not None):
            tolerance = self._convert_tolerance(tolerance, np.asarray(key))
        indexer = self.get_indexer([key], method=method, tolerance=tolerance)
        if ((indexer.ndim > 1) or (indexer.size > 1)):
            raise TypeError('get_loc requires scalar valued input')
        loc = indexer.item()
        if (loc == (- 1)):
            raise KeyError(key)
        return loc
    _index_shared_docs['get_indexer'] = "\n        Compute indexer and mask for new index given the current index. The\n        indexer should be then used as an input to ndarray.take to align the\n        current data to the new index.\n\n        Parameters\n        ----------\n        target : %(target_klass)s\n        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional\n            * default: exact matches only.\n            * pad / ffill: find the PREVIOUS index value if no exact match.\n            * backfill / bfill: use NEXT index value if no exact match\n            * nearest: use the NEAREST index value if no exact match. Tied\n              distances are broken by preferring the larger index value.\n        limit : int, optional\n            Maximum number of consecutive labels in ``target`` to match for\n            inexact matches.\n        tolerance : optional\n            Maximum distance between original and new labels for inexact\n            matches. The values of the index at the matching locations must\n            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.\n\n            Tolerance may be a scalar value, which applies the same tolerance\n            to all values, or list-like, which applies variable tolerance per\n            element. List-like includes list, tuple, array, Series, and must be\n            the same size as the index and its dtype must exactly match the\n            index's type.\n\n        Returns\n        -------\n        indexer : ndarray of int\n            Integers from 0 to n - 1 indicating that the index at these\n            positions matches the corresponding target values. Missing values\n            in the target are marked by -1.\n        %(raises_section)s\n        Examples\n        --------\n        >>> index = pd.Index(['c', 'a', 'b'])\n        >>> index.get_indexer(['a', 'b', 'x'])\n        array([ 1,  2, -1])\n\n        Notice that the return value is an array of locations in ``index``\n        and ``x`` is marked by -1, as it is not in ``index``.\n        "

    @Appender((_index_shared_docs['get_indexer'] % _index_doc_kwargs))
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        method = missing.clean_reindex_fill_method(method)
        target = ensure_index(target)
        self._check_indexing_method(method)
        if (not self._index_as_unique):
            raise InvalidIndexError(self._requires_unique_msg)
        if (target.is_boolean() and self.is_numeric()):
            return ensure_platform_int(np.repeat((- 1), target.size))
        (pself, ptarget) = self._maybe_promote(target)
        if ((pself is not self) or (ptarget is not target)):
            return pself.get_indexer(ptarget, method=method, limit=limit, tolerance=tolerance)
        return self._get_indexer(target, method, limit, tolerance)

    def _get_indexer(self, target, method=None, limit=None, tolerance=None):
        if (tolerance is not None):
            tolerance = self._convert_tolerance(tolerance, target)
        if (not is_dtype_equal(self.dtype, target.dtype)):
            this = self.astype(object)
            target = target.astype(object)
            return this.get_indexer(target, method=method, limit=limit, tolerance=tolerance)
        if (method in ['pad', 'backfill']):
            indexer = self._get_fill_indexer(target, method, limit, tolerance)
        elif (method == 'nearest'):
            indexer = self._get_nearest_indexer(target, limit, tolerance)
        else:
            if (tolerance is not None):
                raise ValueError('tolerance argument only valid if doing pad, backfill or nearest reindexing')
            if (limit is not None):
                raise ValueError('limit argument only valid if doing pad, backfill or nearest reindexing')
            indexer = self._engine.get_indexer(target._get_engine_target())
        return ensure_platform_int(indexer)

    def _check_indexing_method(self, method):
        '\n        Raise if we have a get_indexer `method` that is not supported or valid.\n        '
        if (not (is_interval_dtype(self.dtype) or is_categorical_dtype(self.dtype))):
            return
        if (method is None):
            return
        if (method in ['bfill', 'backfill', 'pad', 'ffill', 'nearest']):
            raise NotImplementedError(f'method {method} not yet implemented for {type(self).__name__}')
        raise ValueError('Invalid fill method')

    def _convert_tolerance(self, tolerance, target):
        tolerance = np.asarray(tolerance)
        if ((target.size != tolerance.size) and (tolerance.size > 1)):
            raise ValueError('list-like tolerance size must match target index size')
        return tolerance

    @final
    def _get_fill_indexer(self, target, method, limit=None, tolerance=None):
        target_values = target._get_engine_target()
        if (self.is_monotonic_increasing and target.is_monotonic_increasing):
            engine_method = (self._engine.get_pad_indexer if (method == 'pad') else self._engine.get_backfill_indexer)
            indexer = engine_method(target_values, limit)
        else:
            indexer = self._get_fill_indexer_searchsorted(target, method, limit)
        if ((tolerance is not None) and len(self)):
            indexer = self._filter_indexer_tolerance(target_values, indexer, tolerance)
        return indexer

    @final
    def _get_fill_indexer_searchsorted(self, target, method, limit=None):
        '\n        Fallback pad/backfill get_indexer that works for monotonic decreasing\n        indexes and non-monotonic targets.\n        '
        if (limit is not None):
            raise ValueError(f'limit argument for {repr(method)} method only well-defined if index and target are monotonic')
        side = ('left' if (method == 'pad') else 'right')
        indexer = self.get_indexer(target)
        nonexact = (indexer == (- 1))
        indexer[nonexact] = self._searchsorted_monotonic(target[nonexact], side)
        if (side == 'left'):
            indexer[nonexact] -= 1
        else:
            indexer[(indexer == len(self))] = (- 1)
        return indexer

    @final
    def _get_nearest_indexer(self, target, limit, tolerance):
        '\n        Get the indexer for the nearest index labels; requires an index with\n        values that can be subtracted from each other (e.g., not strings or\n        tuples).\n        '
        if (not len(self)):
            return self._get_fill_indexer(target, 'pad')
        left_indexer = self.get_indexer(target, 'pad', limit=limit)
        right_indexer = self.get_indexer(target, 'backfill', limit=limit)
        target_values = target._values
        left_distances = np.abs((self._values[left_indexer] - target_values))
        right_distances = np.abs((self._values[right_indexer] - target_values))
        op = (operator.lt if self.is_monotonic_increasing else operator.le)
        indexer = np.where((op(left_distances, right_distances) | (right_indexer == (- 1))), left_indexer, right_indexer)
        if (tolerance is not None):
            indexer = self._filter_indexer_tolerance(target_values, indexer, tolerance)
        return indexer

    @final
    def _filter_indexer_tolerance(self, target, indexer, tolerance):
        distance = abs((self._values[indexer] - target))
        return np.where((distance <= tolerance), indexer, (- 1))

    def _get_partial_string_timestamp_match_key(self, key):
        '\n        Translate any partial string timestamp matches in key, returning the\n        new key.\n\n        Only relevant for MultiIndex.\n        '
        return key

    @final
    def _validate_positional_slice(self, key):
        '\n        For positional indexing, a slice must have either int or None\n        for each of start, stop, and step.\n        '
        self._validate_indexer('positional', key.start, 'iloc')
        self._validate_indexer('positional', key.stop, 'iloc')
        self._validate_indexer('positional', key.step, 'iloc')

    def _convert_slice_indexer(self, key, kind):
        "\n        Convert a slice indexer.\n\n        By definition, these are labels unless 'iloc' is passed in.\n        Floats are not allowed as the start, step, or stop of the slice.\n\n        Parameters\n        ----------\n        key : label of the slice bound\n        kind : {'loc', 'getitem'}\n        "
        assert (kind in ['loc', 'getitem']), kind
        (start, stop, step) = (key.start, key.stop, key.step)

        def is_int(v):
            return ((v is None) or is_integer(v))
        is_index_slice = (is_int(start) and is_int(stop) and is_int(step))
        is_positional = (is_index_slice and (not (self.is_integer() or self.is_categorical())))
        if (kind == 'getitem'):
            '\n            called from the getitem slicers, validate that we are in fact\n            integers\n            '
            if (self.is_integer() or is_index_slice):
                self._validate_indexer('slice', key.start, 'getitem')
                self._validate_indexer('slice', key.stop, 'getitem')
                self._validate_indexer('slice', key.step, 'getitem')
                return key
        if is_positional:
            try:
                if (start is not None):
                    self.get_loc(start)
                if (stop is not None):
                    self.get_loc(stop)
                is_positional = False
            except KeyError:
                pass
        if com.is_null_slice(key):
            indexer = key
        elif is_positional:
            if (kind == 'loc'):
                warnings.warn('Slicing a positional slice with .loc is not supported, and will raise TypeError in a future version.  Use .loc with labels or .iloc with positions instead.', FutureWarning, stacklevel=6)
            indexer = key
        else:
            indexer = self.slice_indexer(start, stop, step, kind=kind)
        return indexer

    def _convert_listlike_indexer(self, keyarr):
        '\n        Parameters\n        ----------\n        keyarr : list-like\n            Indexer to convert.\n\n        Returns\n        -------\n        indexer : numpy.ndarray or None\n            Return an ndarray or None if cannot convert.\n        keyarr : numpy.ndarray\n            Return tuple-safe keys.\n        '
        if isinstance(keyarr, Index):
            pass
        else:
            keyarr = self._convert_arr_indexer(keyarr)
        indexer = self._convert_list_indexer(keyarr)
        return (indexer, keyarr)

    def _convert_arr_indexer(self, keyarr):
        '\n        Convert an array-like indexer to the appropriate dtype.\n\n        Parameters\n        ----------\n        keyarr : array-like\n            Indexer to convert.\n\n        Returns\n        -------\n        converted_keyarr : array-like\n        '
        return com.asarray_tuplesafe(keyarr)

    def _convert_list_indexer(self, keyarr):
        '\n        Convert a list-like indexer to the appropriate dtype.\n\n        Parameters\n        ----------\n        keyarr : Index (or sub-class)\n            Indexer to convert.\n        kind : iloc, loc, optional\n\n        Returns\n        -------\n        positional indexer or None\n        '
        return None

    @final
    def _invalid_indexer(self, form, key):
        '\n        Consistent invalid indexer message.\n        '
        return TypeError(f'cannot do {form} indexing on {type(self).__name__} with these indexers [{key}] of type {type(key).__name__}')

    @final
    def _can_reindex(self, indexer):
        '\n        Check if we are allowing reindexing with this particular indexer.\n\n        Parameters\n        ----------\n        indexer : an integer indexer\n\n        Raises\n        ------\n        ValueError if its a duplicate axis\n        '
        if ((not self._index_as_unique) and len(indexer)):
            raise ValueError('cannot reindex from a duplicate axis')

    def reindex(self, target, method=None, level=None, limit=None, tolerance=None):
        "\n        Create index with target's values.\n\n        Parameters\n        ----------\n        target : an iterable\n\n        Returns\n        -------\n        new_index : pd.Index\n            Resulting index.\n        indexer : np.ndarray or None\n            Indices of output values in original index.\n        "
        preserve_names = (not hasattr(target, 'name'))
        target = ensure_has_len(target)
        if ((not isinstance(target, Index)) and (len(target) == 0)):
            target = self[:0]
        else:
            target = ensure_index(target)
        if (level is not None):
            if (method is not None):
                raise TypeError('Fill method not supported if level passed')
            (_, indexer, _) = self._join_level(target, level, how='right', return_indexers=True)
        elif self.equals(target):
            indexer = None
        elif self._index_as_unique:
            indexer = self.get_indexer(target, method=method, limit=limit, tolerance=tolerance)
        else:
            if ((method is not None) or (limit is not None)):
                raise ValueError('cannot reindex a non-unique index with a method or limit')
            (indexer, _) = self.get_indexer_non_unique(target)
        if (preserve_names and (target.nlevels == 1) and (target.name != self.name)):
            target = target.copy()
            target.name = self.name
        return (target, indexer)

    def _reindex_non_unique(self, target):
        "\n        Create a new index with target's values (move/add/delete values as\n        necessary) use with non-unique Index and a possibly non-unique target.\n\n        Parameters\n        ----------\n        target : an iterable\n\n        Returns\n        -------\n        new_index : pd.Index\n            Resulting index.\n        indexer : np.ndarray or None\n            Indices of output values in original index.\n\n        "
        target = ensure_index(target)
        if (len(target) == 0):
            return (self[:0], np.array([], dtype=np.intp), None)
        (indexer, missing) = self.get_indexer_non_unique(target)
        check = (indexer != (- 1))
        new_labels = self.take(indexer[check])
        new_indexer = None
        if len(missing):
            length = np.arange(len(indexer))
            missing = ensure_platform_int(missing)
            missing_labels = target.take(missing)
            missing_indexer = ensure_int64(length[(~ check)])
            cur_labels = self.take(indexer[check]).values
            cur_indexer = ensure_int64(length[check])
            new_labels = np.empty((len(indexer),), dtype=object)
            new_labels[cur_indexer] = cur_labels
            new_labels[missing_indexer] = missing_labels
            if target.is_unique:
                new_indexer = np.arange(len(indexer))
                new_indexer[cur_indexer] = np.arange(len(cur_labels))
                new_indexer[missing_indexer] = (- 1)
            else:
                indexer[(~ check)] = (- 1)
                new_indexer = np.arange(len(self.take(indexer)))
                new_indexer[(~ check)] = (- 1)
        if isinstance(self, ABCMultiIndex):
            new_index = type(self).from_tuples(new_labels, names=self.names)
        else:
            new_index = Index(new_labels, name=self.name)
        return (new_index, indexer, new_indexer)

    def join(self, other, how='left', level=None, return_indexers=False, sort=False):
        "\n        Compute join_index and indexers to conform data\n        structures to the new index.\n\n        Parameters\n        ----------\n        other : Index\n        how : {'left', 'right', 'inner', 'outer'}\n        level : int or level name, default None\n        return_indexers : bool, default False\n        sort : bool, default False\n            Sort the join keys lexicographically in the result Index. If False,\n            the order of the join keys depends on the join type (how keyword).\n\n        Returns\n        -------\n        join_index, (left_indexer, right_indexer)\n        "
        other = ensure_index(other)
        self_is_mi = isinstance(self, ABCMultiIndex)
        other_is_mi = isinstance(other, ABCMultiIndex)
        if ((level is None) and (self_is_mi or other_is_mi)):
            if (self.names == other.names):
                pass
            else:
                return self._join_multi(other, how=how, return_indexers=return_indexers)
        if ((level is not None) and (self_is_mi or other_is_mi)):
            return self._join_level(other, level, how=how, return_indexers=return_indexers)
        if ((len(other) == 0) and (how in ('left', 'outer'))):
            join_index = self._shallow_copy()
            if return_indexers:
                rindexer = np.repeat((- 1), len(join_index))
                return (join_index, None, rindexer)
            else:
                return join_index
        if ((len(self) == 0) and (how in ('right', 'outer'))):
            join_index = other._shallow_copy()
            if return_indexers:
                lindexer = np.repeat((- 1), len(join_index))
                return (join_index, lindexer, None)
            else:
                return join_index
        if (self._join_precedence < other._join_precedence):
            how = {'right': 'left', 'left': 'right'}.get(how, how)
            result = other.join(self, how=how, level=level, return_indexers=return_indexers)
            if return_indexers:
                (x, y, z) = result
                result = (x, z, y)
            return result
        if (not is_dtype_equal(self.dtype, other.dtype)):
            this = self.astype('O')
            other = other.astype('O')
            return this.join(other, how=how, return_indexers=return_indexers)
        _validate_join_method(how)
        if ((not self.is_unique) and (not other.is_unique)):
            return self._join_non_unique(other, how=how, return_indexers=return_indexers)
        elif ((not self.is_unique) or (not other.is_unique)):
            if (self.is_monotonic and other.is_monotonic):
                return self._join_monotonic(other, how=how, return_indexers=return_indexers)
            else:
                return self._join_non_unique(other, how=how, return_indexers=return_indexers)
        elif (self.is_monotonic and other.is_monotonic):
            try:
                return self._join_monotonic(other, how=how, return_indexers=return_indexers)
            except TypeError:
                pass
        if (how == 'left'):
            join_index = self
        elif (how == 'right'):
            join_index = other
        elif (how == 'inner'):
            join_index = self.intersection(other, sort=False)
        elif (how == 'outer'):
            join_index = self.union(other)
        if sort:
            join_index = join_index.sort_values()
        if return_indexers:
            if (join_index is self):
                lindexer = None
            else:
                lindexer = self.get_indexer(join_index)
            if (join_index is other):
                rindexer = None
            else:
                rindexer = other.get_indexer(join_index)
            return (join_index, lindexer, rindexer)
        else:
            return join_index

    @final
    def _join_multi(self, other, how, return_indexers=True):
        from pandas.core.indexes.multi import MultiIndex
        from pandas.core.reshape.merge import restore_dropped_levels_multijoin
        self_names_list = list(com.not_none(*self.names))
        other_names_list = list(com.not_none(*other.names))
        self_names_order = self_names_list.index
        other_names_order = other_names_list.index
        self_names = set(self_names_list)
        other_names = set(other_names_list)
        overlap = (self_names & other_names)
        if (not overlap):
            raise ValueError('cannot join with no overlapping index names')
        if (isinstance(self, MultiIndex) and isinstance(other, MultiIndex)):
            ldrop_names = sorted((self_names - overlap), key=self_names_order)
            rdrop_names = sorted((other_names - overlap), key=other_names_order)
            if (not len((ldrop_names + rdrop_names))):
                self_jnlevels = self
                other_jnlevels = other.reorder_levels(self.names)
            else:
                self_jnlevels = self.droplevel(ldrop_names)
                other_jnlevels = other.droplevel(rdrop_names)
            (join_idx, lidx, ridx) = self_jnlevels.join(other_jnlevels, how, return_indexers=True)
            dropped_names = (ldrop_names + rdrop_names)
            (levels, codes, names) = restore_dropped_levels_multijoin(self, other, dropped_names, join_idx, lidx, ridx)
            multi_join_idx = MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=False)
            multi_join_idx = multi_join_idx.remove_unused_levels()
            if return_indexers:
                return (multi_join_idx, lidx, ridx)
            else:
                return multi_join_idx
        jl = list(overlap)[0]
        flip_order = False
        if isinstance(self, MultiIndex):
            (self, other) = (other, self)
            flip_order = True
            how = {'right': 'left', 'left': 'right'}.get(how, how)
        level = other.names.index(jl)
        result = self._join_level(other, level, how=how, return_indexers=return_indexers)
        if (flip_order and isinstance(result, tuple)):
            return (result[0], result[2], result[1])
        return result

    @final
    def _join_non_unique(self, other, how='left', return_indexers=False):
        from pandas.core.reshape.merge import get_join_indexers
        assert (self.dtype == other.dtype)
        lvalues = self._get_engine_target()
        rvalues = other._get_engine_target()
        (left_idx, right_idx) = get_join_indexers([lvalues], [rvalues], how=how, sort=True)
        left_idx = ensure_platform_int(left_idx)
        right_idx = ensure_platform_int(right_idx)
        join_index = np.asarray(lvalues.take(left_idx))
        mask = (left_idx == (- 1))
        np.putmask(join_index, mask, rvalues.take(right_idx))
        join_index = self._wrap_joined_index(join_index, other)
        if return_indexers:
            return (join_index, left_idx, right_idx)
        else:
            return join_index

    @final
    def _join_level(self, other, level, how='left', return_indexers=False, keep_order=True):
        '\n        The join method *only* affects the level of the resulting\n        MultiIndex. Otherwise it just exactly aligns the Index data to the\n        labels of the level in the MultiIndex.\n\n        If ```keep_order == True```, the order of the data indexed by the\n        MultiIndex will not be changed; otherwise, it will tie out\n        with `other`.\n        '
        from pandas.core.indexes.multi import MultiIndex

        def _get_leaf_sorter(labels):
            '\n            Returns sorter for the inner most level while preserving the\n            order of higher levels.\n            '
            if (labels[0].size == 0):
                return np.empty(0, dtype='int64')
            if (len(labels) == 1):
                lab = ensure_int64(labels[0])
                (sorter, _) = libalgos.groupsort_indexer(lab, (1 + lab.max()))
                return sorter
            tic = (labels[0][:(- 1)] != labels[0][1:])
            for lab in labels[1:(- 1)]:
                tic |= (lab[:(- 1)] != lab[1:])
            starts = np.hstack(([True], tic, [True])).nonzero()[0]
            lab = ensure_int64(labels[(- 1)])
            return lib.get_level_sorter(lab, ensure_int64(starts))
        if (isinstance(self, MultiIndex) and isinstance(other, MultiIndex)):
            raise TypeError('Join on level between two MultiIndex objects is ambiguous')
        (left, right) = (self, other)
        flip_order = (not isinstance(self, MultiIndex))
        if flip_order:
            (left, right) = (right, left)
            how = {'right': 'left', 'left': 'right'}.get(how, how)
        assert isinstance(left, MultiIndex)
        level = left._get_level_number(level)
        old_level = left.levels[level]
        if (not right.is_unique):
            raise NotImplementedError('Index._join_level on non-unique index is not implemented')
        (new_level, left_lev_indexer, right_lev_indexer) = old_level.join(right, how=how, return_indexers=True)
        if (left_lev_indexer is None):
            if (keep_order or (len(left) == 0)):
                left_indexer = None
                join_index = left
            else:
                left_indexer = _get_leaf_sorter(left.codes[:(level + 1)])
                join_index = left[left_indexer]
        else:
            left_lev_indexer = ensure_int64(left_lev_indexer)
            rev_indexer = lib.get_reverse_indexer(left_lev_indexer, len(old_level))
            old_codes = left.codes[level]
            new_lev_codes = algos.take_nd(rev_indexer, old_codes[(old_codes != (- 1))], allow_fill=False)
            new_codes = list(left.codes)
            new_codes[level] = new_lev_codes
            new_levels = list(left.levels)
            new_levels[level] = new_level
            if keep_order:
                left_indexer = np.arange(len(left), dtype=np.intp)
                mask = (new_lev_codes != (- 1))
                if (not mask.all()):
                    new_codes = [lab[mask] for lab in new_codes]
                    left_indexer = left_indexer[mask]
            elif (level == 0):
                ngroups = (1 + new_lev_codes.max())
                (left_indexer, counts) = libalgos.groupsort_indexer(new_lev_codes, ngroups)
                left_indexer = left_indexer[counts[0]:]
                new_codes = [lab[left_indexer] for lab in new_codes]
            else:
                mask = (new_lev_codes != (- 1))
                mask_all = mask.all()
                if (not mask_all):
                    new_codes = [lab[mask] for lab in new_codes]
                left_indexer = _get_leaf_sorter(new_codes[:(level + 1)])
                new_codes = [lab[left_indexer] for lab in new_codes]
                if (not mask_all):
                    left_indexer = mask.nonzero()[0][left_indexer]
            join_index = MultiIndex(levels=new_levels, codes=new_codes, names=left.names, verify_integrity=False)
        if (right_lev_indexer is not None):
            right_indexer = algos.take_nd(right_lev_indexer, join_index.codes[level], allow_fill=False)
        else:
            right_indexer = join_index.codes[level]
        if flip_order:
            (left_indexer, right_indexer) = (right_indexer, left_indexer)
        if return_indexers:
            left_indexer = (None if (left_indexer is None) else ensure_platform_int(left_indexer))
            right_indexer = (None if (right_indexer is None) else ensure_platform_int(right_indexer))
            return (join_index, left_indexer, right_indexer)
        else:
            return join_index

    @final
    def _join_monotonic(self, other, how='left', return_indexers=False):
        assert (other.dtype == self.dtype)
        if self.equals(other):
            ret_index = (other if (how == 'right') else self)
            if return_indexers:
                return (ret_index, None, None)
            else:
                return ret_index
        sv = self._get_engine_target()
        ov = other._get_engine_target()
        if (self.is_unique and other.is_unique):
            if (how == 'left'):
                join_index = self
                lidx = None
                ridx = self._left_indexer_unique(sv, ov)
            elif (how == 'right'):
                join_index = other
                lidx = self._left_indexer_unique(ov, sv)
                ridx = None
            elif (how == 'inner'):
                (join_index, lidx, ridx) = self._inner_indexer(sv, ov)
                join_index = self._wrap_joined_index(join_index, other)
            elif (how == 'outer'):
                (join_index, lidx, ridx) = self._outer_indexer(sv, ov)
                join_index = self._wrap_joined_index(join_index, other)
        else:
            if (how == 'left'):
                (join_index, lidx, ridx) = self._left_indexer(sv, ov)
            elif (how == 'right'):
                (join_index, ridx, lidx) = self._left_indexer(ov, sv)
            elif (how == 'inner'):
                (join_index, lidx, ridx) = self._inner_indexer(sv, ov)
            elif (how == 'outer'):
                (join_index, lidx, ridx) = self._outer_indexer(sv, ov)
            join_index = self._wrap_joined_index(join_index, other)
        if return_indexers:
            lidx = (None if (lidx is None) else ensure_platform_int(lidx))
            ridx = (None if (ridx is None) else ensure_platform_int(ridx))
            return (join_index, lidx, ridx)
        else:
            return join_index

    def _wrap_joined_index(self, joined, other):
        assert (other.dtype == self.dtype)
        if isinstance(self, ABCMultiIndex):
            name = (self.names if (self.names == other.names) else None)
        else:
            name = get_op_result_name(self, other)
        return self._constructor(joined, name=name)

    @property
    def values(self):
        '\n        Return an array representing the data in the Index.\n\n        .. warning::\n\n           We recommend using :attr:`Index.array` or\n           :meth:`Index.to_numpy`, depending on whether you need\n           a reference to the underlying data or a NumPy array.\n\n        Returns\n        -------\n        array: numpy.ndarray or ExtensionArray\n\n        See Also\n        --------\n        Index.array : Reference to the underlying data.\n        Index.to_numpy : A NumPy array representing the underlying data.\n        '
        return self._data

    @cache_readonly
    @doc(IndexOpsMixin.array)
    def array(self):
        array = self._data
        if isinstance(array, np.ndarray):
            from pandas.core.arrays.numpy_ import PandasArray
            array = PandasArray(array)
        return array

    @property
    def _values(self):
        "\n        The best array representation.\n\n        This is an ndarray or ExtensionArray.\n\n        ``_values`` are consistent between ``Series`` and ``Index``.\n\n        It may differ from the public '.values' method.\n\n        index             | values          | _values       |\n        ----------------- | --------------- | ------------- |\n        Index             | ndarray         | ndarray       |\n        CategoricalIndex  | Categorical     | Categorical   |\n        DatetimeIndex     | ndarray[M8ns]   | DatetimeArray |\n        DatetimeIndex[tz] | ndarray[M8ns]   | DatetimeArray |\n        PeriodIndex       | ndarray[object] | PeriodArray   |\n        IntervalIndex     | IntervalArray   | IntervalArray |\n\n        See Also\n        --------\n        values : Values\n        "
        return self._data

    def _get_engine_target(self):
        '\n        Get the ndarray that we can pass to the IndexEngine constructor.\n        '
        return self._values

    @doc(IndexOpsMixin.memory_usage)
    def memory_usage(self, deep=False):
        result = super().memory_usage(deep=deep)
        result += self._engine.sizeof(deep=deep)
        return result

    def where(self, cond, other=None):
        "\n        Replace values where the condition is False.\n\n        The replacement is taken from other.\n\n        Parameters\n        ----------\n        cond : bool array-like with the same length as self\n            Condition to select the values on.\n        other : scalar, or array-like, default None\n            Replacement if the condition is False.\n\n        Returns\n        -------\n        pandas.Index\n            A copy of self with values replaced from other\n            where the condition is False.\n\n        See Also\n        --------\n        Series.where : Same method for Series.\n        DataFrame.where : Same method for DataFrame.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['car', 'bike', 'train', 'tractor'])\n        >>> idx\n        Index(['car', 'bike', 'train', 'tractor'], dtype='object')\n        >>> idx.where(idx.isin(['car', 'train']), 'other')\n        Index(['car', 'other', 'train', 'other'], dtype='object')\n        "
        if (other is None):
            other = self._na_value
        values = self.values
        try:
            self._validate_fill_value(other)
        except (ValueError, TypeError):
            return self.astype(object).where(cond, other)
        values = np.where(cond, values, other)
        return Index(values, name=self.name)

    @final
    @classmethod
    def _scalar_data_error(cls, data):
        return TypeError(f'{cls.__name__}(...) must be called with a collection of some kind, {repr(data)} was passed')

    @final
    @classmethod
    def _string_data_error(cls, data):
        raise TypeError('String dtype not supported, you may need to explicitly cast to a numeric type')

    def _validate_fill_value(self, value):
        '\n        Check if the value can be inserted into our array without casting,\n        and convert it to an appropriate native type if necessary.\n\n        Raises\n        ------\n        TypeError\n            If the value cannot be inserted into an array of this dtype.\n        '
        return value

    @final
    def _require_scalar(self, value):
        '\n        Check that this is a scalar value that we can use for setitem-like\n        operations without changing dtype.\n        '
        if (not is_scalar(value)):
            raise TypeError(f"'value' must be a scalar, passed: {type(value).__name__}")
        return value

    @property
    def _has_complex_internals(self):
        '\n        Indicates if an index is not directly backed by a numpy array\n        '
        return False

    def _is_memory_usage_qualified(self):
        '\n        Return a boolean if we need a qualified .info display.\n        '
        return self.is_object()

    def is_type_compatible(self, kind):
        '\n        Whether the index type is compatible with the provided type.\n        '
        return (kind == self.inferred_type)

    def __contains__(self, key):
        "\n        Return a boolean indicating whether the provided key is in the index.\n\n        Parameters\n        ----------\n        key : label\n            The key to check if it is present in the index.\n\n        Returns\n        -------\n        bool\n            Whether the key search is in the index.\n\n        Raises\n        ------\n        TypeError\n            If the key is not hashable.\n\n        See Also\n        --------\n        Index.isin : Returns an ndarray of boolean dtype indicating whether the\n            list-like key is in the index.\n\n        Examples\n        --------\n        >>> idx = pd.Index([1, 2, 3, 4])\n        >>> idx\n        Int64Index([1, 2, 3, 4], dtype='int64')\n\n        >>> 2 in idx\n        True\n        >>> 6 in idx\n        False\n        "
        hash(key)
        try:
            return (key in self._engine)
        except (OverflowError, TypeError, ValueError):
            return False

    @final
    def __hash__(self):
        raise TypeError(f'unhashable type: {repr(type(self).__name__)}')

    @final
    def __setitem__(self, key, value):
        raise TypeError('Index does not support mutable operations')

    def __getitem__(self, key):
        "\n        Override numpy.ndarray's __getitem__ method to work as desired.\n\n        This function adds lists and Series as valid boolean indexers\n        (ndarrays only supports ndarray with dtype=bool).\n\n        If resulting ndim != 1, plain ndarray is returned instead of\n        corresponding `Index` subclass.\n\n        "
        getitem = self._data.__getitem__
        promote = self._shallow_copy
        if is_scalar(key):
            key = com.cast_scalar_indexer(key, warn_float=True)
            return getitem(key)
        if isinstance(key, slice):
            return promote(getitem(key))
        if com.is_bool_indexer(key):
            key = np.asarray(key, dtype=bool)
        result = getitem(key)
        if (not is_scalar(result)):
            if (np.ndim(result) > 1):
                deprecate_ndim_indexing(result)
                return result
            return promote(result)
        else:
            return result

    @final
    def _can_hold_identifiers_and_holds_name(self, name):
        "\n        Faster check for ``name in self`` when we know `name` is a Python\n        identifier (e.g. in NDFrame.__getattr__, which hits this to support\n        . key lookup). For indexes that can't hold identifiers (everything\n        but object & categorical) we just return False.\n\n        https://github.com/pandas-dev/pandas/issues/19764\n        "
        if (self.is_object() or self.is_categorical()):
            return (name in self)
        return False

    def append(self, other):
        '\n        Append a collection of Index options together.\n\n        Parameters\n        ----------\n        other : Index or list/tuple of indices\n\n        Returns\n        -------\n        appended : Index\n        '
        to_concat = [self]
        if isinstance(other, (list, tuple)):
            to_concat += list(other)
        else:
            to_concat.append(other)
        for obj in to_concat:
            if (not isinstance(obj, Index)):
                raise TypeError('all inputs must be Index')
        names = {obj.name for obj in to_concat}
        name = (None if (len(names) > 1) else self.name)
        return self._concat(to_concat, name)

    def _concat(self, to_concat, name):
        '\n        Concatenate multiple Index objects.\n        '
        to_concat_vals = [x._values for x in to_concat]
        result = concat_compat(to_concat_vals)
        return Index(result, name=name)

    def putmask(self, mask, value):
        '\n        Return a new Index of the values set with the mask.\n\n        Returns\n        -------\n        Index\n\n        See Also\n        --------\n        numpy.ndarray.putmask : Changes elements of an array\n            based on conditional and input values.\n        '
        values = self._values.copy()
        try:
            converted = self._validate_fill_value(value)
        except (ValueError, TypeError) as err:
            if is_object_dtype(self):
                raise err
            return self.astype(object).putmask(mask, value)
        np.putmask(values, mask, converted)
        return self._shallow_copy(values)

    def equals(self, other):
        '\n        Determine if two Index object are equal.\n\n        The things that are being compared are:\n\n        * The elements inside the Index object.\n        * The order of the elements inside the Index object.\n\n        Parameters\n        ----------\n        other : Any\n            The other object to compare against.\n\n        Returns\n        -------\n        bool\n            True if "other" is an Index and it has the same elements and order\n            as the calling index; False otherwise.\n\n        Examples\n        --------\n        >>> idx1 = pd.Index([1, 2, 3])\n        >>> idx1\n        Int64Index([1, 2, 3], dtype=\'int64\')\n        >>> idx1.equals(pd.Index([1, 2, 3]))\n        True\n\n        The elements inside are compared\n\n        >>> idx2 = pd.Index(["1", "2", "3"])\n        >>> idx2\n        Index([\'1\', \'2\', \'3\'], dtype=\'object\')\n\n        >>> idx1.equals(idx2)\n        False\n\n        The order is compared\n\n        >>> ascending_idx = pd.Index([1, 2, 3])\n        >>> ascending_idx\n        Int64Index([1, 2, 3], dtype=\'int64\')\n        >>> descending_idx = pd.Index([3, 2, 1])\n        >>> descending_idx\n        Int64Index([3, 2, 1], dtype=\'int64\')\n        >>> ascending_idx.equals(descending_idx)\n        False\n\n        The dtype is *not* compared\n\n        >>> int64_idx = pd.Int64Index([1, 2, 3])\n        >>> int64_idx\n        Int64Index([1, 2, 3], dtype=\'int64\')\n        >>> uint64_idx = pd.UInt64Index([1, 2, 3])\n        >>> uint64_idx\n        UInt64Index([1, 2, 3], dtype=\'uint64\')\n        >>> int64_idx.equals(uint64_idx)\n        True\n        '
        if self.is_(other):
            return True
        if (not isinstance(other, Index)):
            return False
        if (is_object_dtype(self.dtype) and (not is_object_dtype(other.dtype))):
            return other.equals(self)
        if isinstance(other, ABCMultiIndex):
            return other.equals(self)
        if is_extension_array_dtype(other.dtype):
            return other.equals(self)
        return array_equivalent(self._values, other._values)

    @final
    def identical(self, other):
        '\n        Similar to equals, but checks that object attributes and types are also equal.\n\n        Returns\n        -------\n        bool\n            If two Index objects have equal elements and same type True,\n            otherwise False.\n        '
        return (self.equals(other) and all(((getattr(self, c, None) == getattr(other, c, None)) for c in self._comparables)) and (type(self) == type(other)))

    @final
    def asof(self, label):
        "\n        Return the label from the index, or, if not present, the previous one.\n\n        Assuming that the index is sorted, return the passed index label if it\n        is in the index, or return the previous index label if the passed one\n        is not in the index.\n\n        Parameters\n        ----------\n        label : object\n            The label up to which the method returns the latest index label.\n\n        Returns\n        -------\n        object\n            The passed label if it is in the index. The previous label if the\n            passed label is not in the sorted index or `NaN` if there is no\n            such label.\n\n        See Also\n        --------\n        Series.asof : Return the latest value in a Series up to the\n            passed index.\n        merge_asof : Perform an asof merge (similar to left join but it\n            matches on nearest key rather than equal key).\n        Index.get_loc : An `asof` is a thin wrapper around `get_loc`\n            with method='pad'.\n\n        Examples\n        --------\n        `Index.asof` returns the latest index label up to the passed label.\n\n        >>> idx = pd.Index(['2013-12-31', '2014-01-02', '2014-01-03'])\n        >>> idx.asof('2014-01-01')\n        '2013-12-31'\n\n        If the label is in the index, the method returns the passed label.\n\n        >>> idx.asof('2014-01-02')\n        '2014-01-02'\n\n        If all of the labels in the index are later than the passed label,\n        NaN is returned.\n\n        >>> idx.asof('1999-01-02')\n        nan\n\n        If the index is not sorted, an error is raised.\n\n        >>> idx_not_sorted = pd.Index(['2013-12-31', '2015-01-02',\n        ...                            '2014-01-03'])\n        >>> idx_not_sorted.asof('2013-12-31')\n        Traceback (most recent call last):\n        ValueError: index must be monotonic increasing or decreasing\n        "
        try:
            loc = self.get_loc(label, method='pad')
        except KeyError:
            return self._na_value
        else:
            if isinstance(loc, slice):
                loc = loc.indices(len(self))[(- 1)]
            return self[loc]

    def asof_locs(self, where, mask):
        '\n        Return the locations (indices) of labels in the index.\n\n        As in the `asof` function, if the label (a particular entry in\n        `where`) is not in the index, the latest index label up to the\n        passed label is chosen and its index returned.\n\n        If all of the labels in the index are later than a label in `where`,\n        -1 is returned.\n\n        `mask` is used to ignore NA values in the index during calculation.\n\n        Parameters\n        ----------\n        where : Index\n            An Index consisting of an array of timestamps.\n        mask : array-like\n            Array of booleans denoting where values in the original\n            data are not NA.\n\n        Returns\n        -------\n        numpy.ndarray\n            An array of locations (indices) of the labels from the Index\n            which correspond to the return values of the `asof` function\n            for every element in `where`.\n        '
        locs = self._values[mask].searchsorted(where._values, side='right')
        locs = np.where((locs > 0), (locs - 1), 0)
        result = np.arange(len(self))[mask].take(locs)
        first_value = cast(Any, self._values[mask.argmax()])
        result[((locs == 0) & (where._values < first_value))] = (- 1)
        return result

    @final
    def sort_values(self, return_indexer=False, ascending=True, na_position='last', key=None):
        "\n        Return a sorted copy of the index.\n\n        Return a sorted copy of the index, and optionally return the indices\n        that sorted the index itself.\n\n        Parameters\n        ----------\n        return_indexer : bool, default False\n            Should the indices that would sort the index be returned.\n        ascending : bool, default True\n            Should the index values be sorted in an ascending order.\n        na_position : {'first' or 'last'}, default 'last'\n            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at\n            the end.\n\n            .. versionadded:: 1.2.0\n\n        key : callable, optional\n            If not None, apply the key function to the index values\n            before sorting. This is similar to the `key` argument in the\n            builtin :meth:`sorted` function, with the notable difference that\n            this `key` function should be *vectorized*. It should expect an\n            ``Index`` and return an ``Index`` of the same shape.\n\n            .. versionadded:: 1.1.0\n\n        Returns\n        -------\n        sorted_index : pandas.Index\n            Sorted copy of the index.\n        indexer : numpy.ndarray, optional\n            The indices that the index itself was sorted by.\n\n        See Also\n        --------\n        Series.sort_values : Sort values of a Series.\n        DataFrame.sort_values : Sort values in a DataFrame.\n\n        Examples\n        --------\n        >>> idx = pd.Index([10, 100, 1, 1000])\n        >>> idx\n        Int64Index([10, 100, 1, 1000], dtype='int64')\n\n        Sort values in ascending order (default behavior).\n\n        >>> idx.sort_values()\n        Int64Index([1, 10, 100, 1000], dtype='int64')\n\n        Sort values in descending order, and also get the indices `idx` was\n        sorted by.\n\n        >>> idx.sort_values(ascending=False, return_indexer=True)\n        (Int64Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2]))\n        "
        idx = ensure_key_mapped(self, key)
        if (not isinstance(self, ABCMultiIndex)):
            _as = nargsort(items=idx, ascending=ascending, na_position=na_position, key=key)
        else:
            _as = idx.argsort()
            if (not ascending):
                _as = _as[::(- 1)]
        sorted_index = self.take(_as)
        if return_indexer:
            return (sorted_index, _as)
        else:
            return sorted_index

    @final
    def sort(self, *args, **kwargs):
        '\n        Use sort_values instead.\n        '
        raise TypeError('cannot sort an Index object in-place, use sort_values instead')

    def shift(self, periods=1, freq=None):
        "\n        Shift index by desired number of time frequency increments.\n\n        This method is for shifting the values of datetime-like indexes\n        by a specified time increment a given number of times.\n\n        Parameters\n        ----------\n        periods : int, default 1\n            Number of periods (or increments) to shift by,\n            can be positive or negative.\n        freq : pandas.DateOffset, pandas.Timedelta or str, optional\n            Frequency increment to shift by.\n            If None, the index is shifted by its own `freq` attribute.\n            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.\n\n        Returns\n        -------\n        pandas.Index\n            Shifted index.\n\n        See Also\n        --------\n        Series.shift : Shift values of Series.\n\n        Notes\n        -----\n        This method is only implemented for datetime-like index classes,\n        i.e., DatetimeIndex, PeriodIndex and TimedeltaIndex.\n\n        Examples\n        --------\n        Put the first 5 month starts of 2011 into an index.\n\n        >>> month_starts = pd.date_range('1/1/2011', periods=5, freq='MS')\n        >>> month_starts\n        DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01',\n                       '2011-05-01'],\n                      dtype='datetime64[ns]', freq='MS')\n\n        Shift the index by 10 days.\n\n        >>> month_starts.shift(10, freq='D')\n        DatetimeIndex(['2011-01-11', '2011-02-11', '2011-03-11', '2011-04-11',\n                       '2011-05-11'],\n                      dtype='datetime64[ns]', freq=None)\n\n        The default value of `freq` is the `freq` attribute of the index,\n        which is 'MS' (month start) in this example.\n\n        >>> month_starts.shift(10)\n        DatetimeIndex(['2011-11-01', '2011-12-01', '2012-01-01', '2012-02-01',\n                       '2012-03-01'],\n                      dtype='datetime64[ns]', freq='MS')\n        "
        raise NotImplementedError(f'This method is only implemented for DatetimeIndex, PeriodIndex and TimedeltaIndex; Got type {type(self).__name__}')

    def argsort(self, *args, **kwargs):
        "\n        Return the integer indices that would sort the index.\n\n        Parameters\n        ----------\n        *args\n            Passed to `numpy.ndarray.argsort`.\n        **kwargs\n            Passed to `numpy.ndarray.argsort`.\n\n        Returns\n        -------\n        numpy.ndarray\n            Integer indices that would sort the index if used as\n            an indexer.\n\n        See Also\n        --------\n        numpy.argsort : Similar method for NumPy arrays.\n        Index.sort_values : Return sorted copy of Index.\n\n        Examples\n        --------\n        >>> idx = pd.Index(['b', 'a', 'd', 'c'])\n        >>> idx\n        Index(['b', 'a', 'd', 'c'], dtype='object')\n\n        >>> order = idx.argsort()\n        >>> order\n        array([1, 0, 3, 2])\n\n        >>> idx[order]\n        Index(['a', 'b', 'c', 'd'], dtype='object')\n        "
        return self._data.argsort(*args, **kwargs)

    @final
    def get_value(self, series, key):
        "\n        Fast lookup of value from 1-dimensional ndarray.\n\n        Only use this if you know what you're doing.\n\n        Returns\n        -------\n        scalar or Series\n        "
        warnings.warn('get_value is deprecated and will be removed in a future version. Use Series[key] instead', FutureWarning, stacklevel=2)
        self._check_indexing_error(key)
        try:
            loc = self.get_loc(key)
        except KeyError:
            if (not self._should_fallback_to_positional()):
                raise
            elif is_integer(key):
                loc = key
            else:
                raise
        return self._get_values_for_loc(series, loc, key)

    def _check_indexing_error(self, key):
        if (not is_scalar(key)):
            raise InvalidIndexError(key)

    def _should_fallback_to_positional(self):
        '\n        Should an integer key be treated as positional?\n        '
        return ((not self.holds_integer()) and (not self.is_boolean()))

    def _get_values_for_loc(self, series, loc, key):
        '\n        Do a positional lookup on the given Series, returning either a scalar\n        or a Series.\n\n        Assumes that `series.index is self`\n\n        key is included for MultiIndex compat.\n        '
        if is_integer(loc):
            return series._values[loc]
        return series.iloc[loc]

    @final
    def set_value(self, arr, key, value):
        "\n        Fast lookup of value from 1-dimensional ndarray.\n\n        .. deprecated:: 1.0\n\n        Notes\n        -----\n        Only use this if you know what you're doing.\n        "
        warnings.warn("The 'set_value' method is deprecated, and will be removed in a future version.", FutureWarning, stacklevel=2)
        loc = self._engine.get_loc(key)
        validate_numeric_casting(arr.dtype, value)
        arr[loc] = value
    _index_shared_docs['get_indexer_non_unique'] = '\n        Compute indexer and mask for new index given the current index. The\n        indexer should be then used as an input to ndarray.take to align the\n        current data to the new index.\n\n        Parameters\n        ----------\n        target : %(target_klass)s\n\n        Returns\n        -------\n        indexer : ndarray of int\n            Integers from 0 to n - 1 indicating that the index at these\n            positions matches the corresponding target values. Missing values\n            in the target are marked by -1.\n        missing : ndarray of int\n            An indexer into the target of the values not found.\n            These correspond to the -1 in the indexer array.\n        '

    @Appender((_index_shared_docs['get_indexer_non_unique'] % _index_doc_kwargs))
    def get_indexer_non_unique(self, target):
        target = ensure_index(target)
        if (target.is_boolean() and self.is_numeric()):
            return self._get_indexer_non_comparable(target, method=None, unique=False)
        (pself, ptarget) = self._maybe_promote(target)
        if ((pself is not self) or (ptarget is not target)):
            return pself.get_indexer_non_unique(ptarget)
        if (not self._should_compare(target)):
            return self._get_indexer_non_comparable(target, method=None, unique=False)
        if (not is_dtype_equal(self.dtype, target.dtype)):
            dtype = find_common_type([self.dtype, target.dtype])
            if ((dtype.kind in ['i', 'u']) and is_categorical_dtype(target.dtype) and target.hasnans):
                dtype = np.dtype(np.float64)
            this = self.astype(dtype, copy=False)
            that = target.astype(dtype, copy=False)
            return this.get_indexer_non_unique(that)
        tgt_values = target._get_engine_target()
        (indexer, missing) = self._engine.get_indexer_non_unique(tgt_values)
        return (ensure_platform_int(indexer), missing)

    @final
    def get_indexer_for(self, target, **kwargs):
        '\n        Guaranteed return of an indexer even when non-unique.\n\n        This dispatches to get_indexer or get_indexer_non_unique\n        as appropriate.\n\n        Returns\n        -------\n        numpy.ndarray\n            List of indices.\n        '
        if self._index_as_unique:
            return self.get_indexer(target, **kwargs)
        (indexer, _) = self.get_indexer_non_unique(target)
        return indexer

    def _get_indexer_non_comparable(self, target, method, unique=True):
        '\n        Called from get_indexer or get_indexer_non_unique when the target\n        is of a non-comparable dtype.\n\n        For get_indexer lookups with method=None, get_indexer is an _equality_\n        check, so non-comparable dtypes mean we will always have no matches.\n\n        For get_indexer lookups with a method, get_indexer is an _inequality_\n        check, so non-comparable dtypes mean we will always raise TypeError.\n\n        Parameters\n        ----------\n        target : Index\n        method : str or None\n        unique : bool, default True\n            * True if called from get_indexer.\n            * False if called from get_indexer_non_unique.\n\n        Raises\n        ------\n        TypeError\n            If doing an inequality check, i.e. method is not None.\n        '
        if (method is not None):
            other = unpack_nested_dtype(target)
            raise TypeError(f'Cannot compare dtypes {self.dtype} and {other.dtype}')
        no_matches = ((- 1) * np.ones(target.shape, dtype=np.intp))
        if unique:
            return no_matches
        else:
            missing = np.arange(len(target), dtype=np.intp)
            return (no_matches, missing)

    @property
    def _index_as_unique(self):
        '\n        Whether we should treat this as unique for the sake of\n        get_indexer vs get_indexer_non_unique.\n\n        For IntervalIndex compat.\n        '
        return self.is_unique
    _requires_unique_msg = 'Reindexing only valid with uniquely valued Index objects'

    @final
    def _maybe_promote(self, other):
        '\n        When dealing with an object-dtype Index and a non-object Index, see\n        if we can upcast the object-dtype one to improve performance.\n        '
        if ((self.inferred_type == 'date') and isinstance(other, ABCDatetimeIndex)):
            try:
                return (type(other)(self), other)
            except OutOfBoundsDatetime:
                return (self, other)
        elif ((self.inferred_type == 'timedelta') and isinstance(other, ABCTimedeltaIndex)):
            return (type(other)(self), other)
        elif (self.inferred_type == 'boolean'):
            if (not is_object_dtype(self.dtype)):
                return (self.astype('object'), other.astype('object'))
        if ((not is_object_dtype(self.dtype)) and is_object_dtype(other.dtype)):
            (other, self) = other._maybe_promote(self)
        return (self, other)

    def _should_compare(self, other):
        '\n        Check if `self == other` can ever have non-False entries.\n        '
        other = unpack_nested_dtype(other)
        dtype = other.dtype
        return (self._is_comparable_dtype(dtype) or is_object_dtype(dtype))

    def _is_comparable_dtype(self, dtype):
        '\n        Can we compare values of the given dtype to our own?\n        '
        return True

    @final
    def groupby(self, values):
        '\n        Group the index labels by a given array of values.\n\n        Parameters\n        ----------\n        values : array\n            Values used to determine the groups.\n\n        Returns\n        -------\n        dict\n            {group name -> group labels}\n        '
        if isinstance(values, ABCMultiIndex):
            values = values._values
        values = Categorical(values)
        result = values._reverse_indexer()
        result = {k: self.take(v) for (k, v) in result.items()}
        return PrettyDict(result)

    def map(self, mapper, na_action=None):
        "\n        Map values using input correspondence (a dict, Series, or function).\n\n        Parameters\n        ----------\n        mapper : function, dict, or Series\n            Mapping correspondence.\n        na_action : {None, 'ignore'}\n            If 'ignore', propagate NA values, without passing them to the\n            mapping correspondence.\n\n        Returns\n        -------\n        applied : Union[Index, MultiIndex], inferred\n            The output of the mapping function applied to the index.\n            If the function returns a tuple with more than one element\n            a MultiIndex will be returned.\n        "
        from pandas.core.indexes.multi import MultiIndex
        new_values = super()._map_values(mapper, na_action=na_action)
        attributes = self._get_attributes_dict()
        if (new_values.size and isinstance(new_values[0], tuple)):
            if isinstance(self, MultiIndex):
                names = self.names
            elif attributes.get('name'):
                names = ([attributes.get('name')] * len(new_values[0]))
            else:
                names = None
            return MultiIndex.from_tuples(new_values, names=names)
        attributes['copy'] = False
        if (not new_values.size):
            attributes['dtype'] = self.dtype
        return Index(new_values, **attributes)

    @final
    def _transform_index(self, func, level=None):
        '\n        Apply function to all values found in index.\n\n        This includes transforming multiindex entries separately.\n        Only apply function to one level of the MultiIndex if level is specified.\n        '
        if isinstance(self, ABCMultiIndex):
            if (level is not None):
                items = [tuple(((func(y) if (i == level) else y) for (i, y) in enumerate(x))) for x in self]
            else:
                items = [tuple((func(y) for y in x)) for x in self]
            return type(self).from_tuples(items, names=self.names)
        else:
            items = [func(x) for x in self]
            return Index(items, name=self.name, tupleize_cols=False)

    def isin(self, values, level=None):
        "\n        Return a boolean array where the index values are in `values`.\n\n        Compute boolean array of whether each index value is found in the\n        passed set of values. The length of the returned boolean array matches\n        the length of the index.\n\n        Parameters\n        ----------\n        values : set or list-like\n            Sought values.\n        level : str or int, optional\n            Name or position of the index level to use (if the index is a\n            `MultiIndex`).\n\n        Returns\n        -------\n        is_contained : ndarray\n            NumPy array of boolean values.\n\n        See Also\n        --------\n        Series.isin : Same for Series.\n        DataFrame.isin : Same method for DataFrames.\n\n        Notes\n        -----\n        In the case of `MultiIndex` you must either specify `values` as a\n        list-like object containing tuples that are the same length as the\n        number of levels, or specify `level`. Otherwise it will raise a\n        ``ValueError``.\n\n        If `level` is specified:\n\n        - if it is the name of one *and only one* index level, use that level;\n        - otherwise it should be a number indicating level position.\n\n        Examples\n        --------\n        >>> idx = pd.Index([1,2,3])\n        >>> idx\n        Int64Index([1, 2, 3], dtype='int64')\n\n        Check whether each index value in a list of values.\n\n        >>> idx.isin([1, 4])\n        array([ True, False, False])\n\n        >>> midx = pd.MultiIndex.from_arrays([[1,2,3],\n        ...                                  ['red', 'blue', 'green']],\n        ...                                  names=('number', 'color'))\n        >>> midx\n        MultiIndex([(1,   'red'),\n                    (2,  'blue'),\n                    (3, 'green')],\n                   names=['number', 'color'])\n\n        Check whether the strings in the 'color' level of the MultiIndex\n        are in a list of colors.\n\n        >>> midx.isin(['red', 'orange', 'yellow'], level='color')\n        array([ True, False, False])\n\n        To check across the levels of a MultiIndex, pass a list of tuples:\n\n        >>> midx.isin([(1, 'red'), (3, 'red')])\n        array([ True, False, False])\n\n        For a DatetimeIndex, string values in `values` are converted to\n        Timestamps.\n\n        >>> dates = ['2000-03-11', '2000-03-12', '2000-03-13']\n        >>> dti = pd.to_datetime(dates)\n        >>> dti\n        DatetimeIndex(['2000-03-11', '2000-03-12', '2000-03-13'],\n        dtype='datetime64[ns]', freq=None)\n\n        >>> dti.isin(['2000-03-11'])\n        array([ True, False, False])\n        "
        if (level is not None):
            self._validate_index_level(level)
        return algos.isin(self._values, values)

    def _get_string_slice(self, key):
        raise NotImplementedError

    def slice_indexer(self, start=None, end=None, step=None, kind=None):
        "\n        Compute the slice indexer for input labels and step.\n\n        Index needs to be ordered and unique.\n\n        Parameters\n        ----------\n        start : label, default None\n            If None, defaults to the beginning.\n        end : label, default None\n            If None, defaults to the end.\n        step : int, default None\n        kind : str, default None\n\n        Returns\n        -------\n        indexer : slice\n\n        Raises\n        ------\n        KeyError : If key does not exist, or key is not unique and index is\n            not ordered.\n\n        Notes\n        -----\n        This function assumes that the data is sorted, so use at your own peril\n\n        Examples\n        --------\n        This is a method on all index types. For example you can do:\n\n        >>> idx = pd.Index(list('abcd'))\n        >>> idx.slice_indexer(start='b', end='c')\n        slice(1, 3, None)\n\n        >>> idx = pd.MultiIndex.from_arrays([list('abcd'), list('efgh')])\n        >>> idx.slice_indexer(start='b', end=('c', 'g'))\n        slice(1, 3, None)\n        "
        (start_slice, end_slice) = self.slice_locs(start, end, step=step, kind=kind)
        if (not is_scalar(start_slice)):
            raise AssertionError('Start slice bound is non-scalar')
        if (not is_scalar(end_slice)):
            raise AssertionError('End slice bound is non-scalar')
        return slice(start_slice, end_slice, step)

    def _maybe_cast_indexer(self, key):
        '\n        If we have a float key and are not a floating index, then try to cast\n        to an int if equivalent.\n        '
        if (not self.is_floating()):
            return com.cast_scalar_indexer(key)
        return key

    @final
    def _validate_indexer(self, form, key, kind):
        '\n        If we are positional indexer, validate that we have appropriate\n        typed bounds must be an integer.\n        '
        assert (kind in ['getitem', 'iloc'])
        if ((key is not None) and (not is_integer(key))):
            raise self._invalid_indexer(form, key)

    def _maybe_cast_slice_bound(self, label, side, kind):
        "\n        This function should be overloaded in subclasses that allow non-trivial\n        casting on label-slice bounds, e.g. datetime-like indices allowing\n        strings containing formatted datetimes.\n\n        Parameters\n        ----------\n        label : object\n        side : {'left', 'right'}\n        kind : {'loc', 'getitem'} or None\n\n        Returns\n        -------\n        label : object\n\n        Notes\n        -----\n        Value of `side` parameter should be validated in caller.\n        "
        assert (kind in ['loc', 'getitem', None])
        if ((is_float(label) or is_integer(label)) and (label not in self._values)):
            raise self._invalid_indexer('slice', label)
        return label

    def _searchsorted_monotonic(self, label, side='left'):
        if self.is_monotonic_increasing:
            return self.searchsorted(label, side=side)
        elif self.is_monotonic_decreasing:
            pos = self[::(- 1)].searchsorted(label, side=('right' if (side == 'left') else 'left'))
            return (len(self) - pos)
        raise ValueError('index must be monotonic increasing or decreasing')

    def get_slice_bound(self, label, side, kind):
        "\n        Calculate slice bound that corresponds to given label.\n\n        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position\n        of given label.\n\n        Parameters\n        ----------\n        label : object\n        side : {'left', 'right'}\n        kind : {'loc', 'getitem'} or None\n\n        Returns\n        -------\n        int\n            Index of label.\n        "
        assert (kind in ['loc', 'getitem', None])
        if (side not in ('left', 'right')):
            raise ValueError(f"Invalid value for side kwarg, must be either 'left' or 'right': {side}")
        original_label = label
        label = self._maybe_cast_slice_bound(label, side, kind)
        try:
            slc = self.get_loc(label)
        except KeyError as err:
            try:
                return self._searchsorted_monotonic(label, side)
            except ValueError:
                raise err
        if isinstance(slc, np.ndarray):
            if is_bool_dtype(slc):
                slc = lib.maybe_booleans_to_slice(slc.view('u1'))
            else:
                slc = lib.maybe_indices_to_slice(slc.astype(np.intp, copy=False), len(self))
            if isinstance(slc, np.ndarray):
                raise KeyError(f'Cannot get {side} slice bound for non-unique label: {repr(original_label)}')
        if isinstance(slc, slice):
            if (side == 'left'):
                return slc.start
            else:
                return slc.stop
        elif (side == 'right'):
            return (slc + 1)
        else:
            return slc

    def slice_locs(self, start=None, end=None, step=None, kind=None):
        "\n        Compute slice locations for input labels.\n\n        Parameters\n        ----------\n        start : label, default None\n            If None, defaults to the beginning.\n        end : label, default None\n            If None, defaults to the end.\n        step : int, defaults None\n            If None, defaults to 1.\n        kind : {'loc', 'getitem'} or None\n\n        Returns\n        -------\n        start, end : int\n\n        See Also\n        --------\n        Index.get_loc : Get location for a single label.\n\n        Notes\n        -----\n        This method only works if the index is monotonic or unique.\n\n        Examples\n        --------\n        >>> idx = pd.Index(list('abcd'))\n        >>> idx.slice_locs(start='b', end='c')\n        (1, 3)\n        "
        inc = ((step is None) or (step >= 0))
        if (not inc):
            (start, end) = (end, start)
        if (isinstance(start, (str, datetime)) and isinstance(end, (str, datetime))):
            try:
                ts_start = Timestamp(start)
                ts_end = Timestamp(end)
            except (ValueError, TypeError):
                pass
            else:
                if (not tz_compare(ts_start.tzinfo, ts_end.tzinfo)):
                    raise ValueError('Both dates must have the same UTC offset')
        start_slice = None
        if (start is not None):
            start_slice = self.get_slice_bound(start, 'left', kind)
        if (start_slice is None):
            start_slice = 0
        end_slice = None
        if (end is not None):
            end_slice = self.get_slice_bound(end, 'right', kind)
        if (end_slice is None):
            end_slice = len(self)
        if (not inc):
            (end_slice, start_slice) = ((start_slice - 1), (end_slice - 1))
            if (end_slice == (- 1)):
                end_slice -= len(self)
            if (start_slice == (- 1)):
                start_slice -= len(self)
        return (start_slice, end_slice)

    def delete(self, loc):
        "\n        Make new Index with passed location(-s) deleted.\n\n        Parameters\n        ----------\n        loc : int or list of int\n            Location of item(-s) which will be deleted.\n            Use a list of locations to delete more than one value at the same time.\n\n        Returns\n        -------\n        Index\n            New Index with passed location(-s) deleted.\n\n        See Also\n        --------\n        numpy.delete : Delete any rows and column from NumPy array (ndarray).\n\n        Examples\n        --------\n        >>> idx = pd.Index(['a', 'b', 'c'])\n        >>> idx.delete(1)\n        Index(['a', 'c'], dtype='object')\n\n        >>> idx = pd.Index(['a', 'b', 'c'])\n        >>> idx.delete([0, 2])\n        Index(['b'], dtype='object')\n        "
        return self._shallow_copy(np.delete(self._data, loc))

    def insert(self, loc, item):
        '\n        Make new Index inserting new item at location.\n\n        Follows Python list.append semantics for negative values.\n\n        Parameters\n        ----------\n        loc : int\n        item : object\n\n        Returns\n        -------\n        new_index : Index\n        '
        try:
            item = self._validate_fill_value(item)
        except TypeError:
            if is_scalar(item):
                (dtype, item) = maybe_promote(self.dtype, item)
            else:
                dtype = np.dtype(object)
            return self.astype(dtype).insert(loc, item)
        arr = np.asarray(self)
        item = Index([item], dtype=self.dtype)._values
        idx = np.concatenate((arr[:loc], item, arr[loc:]))
        return Index(idx, name=self.name)

    def drop(self, labels, errors='raise'):
        "\n        Make new Index with passed list of labels deleted.\n\n        Parameters\n        ----------\n        labels : array-like\n        errors : {'ignore', 'raise'}, default 'raise'\n            If 'ignore', suppress error and existing labels are dropped.\n\n        Returns\n        -------\n        dropped : Index\n\n        Raises\n        ------\n        KeyError\n            If not all of the labels are found in the selected axis\n        "
        arr_dtype = ('object' if (self.dtype == 'object') else None)
        labels = com.index_labels_to_array(labels, dtype=arr_dtype)
        indexer = self.get_indexer_for(labels)
        mask = (indexer == (- 1))
        if mask.any():
            if (errors != 'ignore'):
                raise KeyError(f'{labels[mask]} not found in axis')
            indexer = indexer[(~ mask)]
        return self.delete(indexer)

    def _cmp_method(self, other, op):
        '\n        Wrapper used to dispatch comparison operations.\n        '
        if self.is_(other):
            if (op in {operator.eq, operator.le, operator.ge}):
                arr = np.ones(len(self), dtype=bool)
                if (self._can_hold_na and (not isinstance(self, ABCMultiIndex))):
                    arr[self.isna()] = False
                return arr
            elif (op in {operator.ne, operator.lt, operator.gt}):
                return np.zeros(len(self), dtype=bool)
        if (isinstance(other, (np.ndarray, Index, ABCSeries, ExtensionArray)) and (len(self) != len(other))):
            raise ValueError('Lengths must match to compare')
        if (not isinstance(other, ABCMultiIndex)):
            other = extract_array(other, extract_numpy=True)
        else:
            other = np.asarray(other)
        if (is_object_dtype(self.dtype) and isinstance(other, ExtensionArray)):
            with np.errstate(all='ignore'):
                result = op(self._values, other)
        elif (is_object_dtype(self.dtype) and (not isinstance(self, ABCMultiIndex))):
            with np.errstate(all='ignore'):
                result = ops.comp_method_OBJECT_ARRAY(op, self._values, other)
        elif is_interval_dtype(self.dtype):
            with np.errstate(all='ignore'):
                result = op(self._values, np.asarray(other))
        else:
            with np.errstate(all='ignore'):
                result = ops.comparison_op(self._values, other, op)
        return result

    def _arith_method(self, other, op):
        '\n        Wrapper used to dispatch arithmetic operations.\n        '
        from pandas import Series
        result = op(Series(self), other)
        if isinstance(result, tuple):
            return (Index(result[0]), Index(result[1]))
        return Index(result)

    def _unary_method(self, op):
        result = op(self._values)
        return Index(result, name=self.name)

    def __abs__(self):
        return self._unary_method(operator.abs)

    def __neg__(self):
        return self._unary_method(operator.neg)

    def __pos__(self):
        return self._unary_method(operator.pos)

    def __inv__(self):
        return self._unary_method((lambda x: (- x)))

    def any(self, *args, **kwargs):
        '\n        Return whether any element is Truthy.\n\n        Parameters\n        ----------\n        *args\n            These parameters will be passed to numpy.any.\n        **kwargs\n            These parameters will be passed to numpy.any.\n\n        Returns\n        -------\n        any : bool or array_like (if axis is specified)\n            A single element array_like may be converted to bool.\n\n        See Also\n        --------\n        Index.all : Return whether all elements are True.\n        Series.all : Return whether all elements are True.\n\n        Notes\n        -----\n        Not a Number (NaN), positive infinity and negative infinity\n        evaluate to True because these are not equal to zero.\n\n        Examples\n        --------\n        >>> index = pd.Index([0, 1, 2])\n        >>> index.any()\n        True\n\n        >>> index = pd.Index([0, 0, 0])\n        >>> index.any()\n        False\n        '
        self._maybe_disable_logical_methods('any')
        return np.any(self.values)

    def all(self):
        '\n        Return whether all elements are Truthy.\n\n        Parameters\n        ----------\n        *args\n            These parameters will be passed to numpy.all.\n        **kwargs\n            These parameters will be passed to numpy.all.\n\n        Returns\n        -------\n        all : bool or array_like (if axis is specified)\n            A single element array_like may be converted to bool.\n\n        See Also\n        --------\n        Index.any : Return whether any element in an Index is True.\n        Series.any : Return whether any element in a Series is True.\n        Series.all : Return whether all elements in a Series are True.\n\n        Notes\n        -----\n        Not a Number (NaN), positive infinity and negative infinity\n        evaluate to True because these are not equal to zero.\n\n        Examples\n        --------\n        **all**\n\n        True, because nonzero integers are considered True.\n\n        >>> pd.Index([1, 2, 3]).all()\n        True\n\n        False, because ``0`` is considered False.\n\n        >>> pd.Index([0, 1, 2]).all()\n        False\n\n        **any**\n\n        True, because ``1`` is considered True.\n\n        >>> pd.Index([0, 0, 1]).any()\n        True\n\n        False, because ``0`` is considered False.\n\n        >>> pd.Index([0, 0, 0]).any()\n        False\n        '
        self._maybe_disable_logical_methods('all')
        return np.all(self.values)

    @final
    def _maybe_disable_logical_methods(self, opname):
        '\n        raise if this Index subclass does not support any or all.\n        '
        if (isinstance(self, ABCMultiIndex) or needs_i8_conversion(self.dtype) or is_interval_dtype(self.dtype) or is_categorical_dtype(self.dtype) or is_float_dtype(self.dtype)):
            make_invalid_op(opname)(self)

    @property
    def shape(self):
        '\n        Return a tuple of the shape of the underlying data.\n        '
        return self._values.shape

def ensure_index_from_sequences(sequences, names=None):
    '\n    Construct an index from sequences of data.\n\n    A single sequence returns an Index. Many sequences returns a\n    MultiIndex.\n\n    Parameters\n    ----------\n    sequences : sequence of sequences\n    names : sequence of str\n\n    Returns\n    -------\n    index : Index or MultiIndex\n\n    Examples\n    --------\n    >>> ensure_index_from_sequences([[1, 2, 3]], names=["name"])\n    Int64Index([1, 2, 3], dtype=\'int64\', name=\'name\')\n\n    >>> ensure_index_from_sequences([["a", "a"], ["a", "b"]], names=["L1", "L2"])\n    MultiIndex([(\'a\', \'a\'),\n                (\'a\', \'b\')],\n               names=[\'L1\', \'L2\'])\n\n    See Also\n    --------\n    ensure_index\n    '
    from pandas.core.indexes.multi import MultiIndex
    if (len(sequences) == 1):
        if (names is not None):
            names = names[0]
        return Index(sequences[0], name=names)
    else:
        return MultiIndex.from_arrays(sequences, names=names)

def ensure_index(index_like, copy=False):
    "\n    Ensure that we have an index from some index-like object.\n\n    Parameters\n    ----------\n    index_like : sequence\n        An Index or other sequence\n    copy : bool, default False\n\n    Returns\n    -------\n    index : Index or MultiIndex\n\n    See Also\n    --------\n    ensure_index_from_sequences\n\n    Examples\n    --------\n    >>> ensure_index(['a', 'b'])\n    Index(['a', 'b'], dtype='object')\n\n    >>> ensure_index([('a', 'a'),  ('b', 'c')])\n    Index([('a', 'a'), ('b', 'c')], dtype='object')\n\n    >>> ensure_index([['a', 'a'], ['b', 'c']])\n    MultiIndex([('a', 'b'),\n            ('a', 'c')],\n           )\n    "
    if isinstance(index_like, Index):
        if copy:
            index_like = index_like.copy()
        return index_like
    if hasattr(index_like, 'name'):
        name = index_like.name
        return Index(index_like, name=name, copy=copy)
    if is_iterator(index_like):
        index_like = list(index_like)
    if isinstance(index_like, list):
        if (type(index_like) != list):
            index_like = list(index_like)
        (converted, all_arrays) = lib.clean_index_list(index_like)
        if ((len(converted) > 0) and all_arrays):
            from pandas.core.indexes.multi import MultiIndex
            return MultiIndex.from_arrays(converted)
        else:
            if (isinstance(converted, np.ndarray) and (converted.dtype == np.int64)):
                alt = np.asarray(index_like)
                if (alt.dtype == np.uint64):
                    converted = alt
            index_like = converted
    elif copy:
        index_like = copy_func(index_like)
    return Index(index_like)

def ensure_has_len(seq):
    '\n    If seq is an iterator, put its values into a list.\n    '
    try:
        len(seq)
    except TypeError:
        return list(seq)
    else:
        return seq

def trim_front(strings):
    '\n    Trims zeros and decimal points.\n\n    Examples\n    --------\n    >>> trim_front([" a", " b"])\n    [\'a\', \'b\']\n\n    >>> trim_front([" a", " "])\n    [\'a\', \'\']\n    '
    if (not strings):
        return strings
    while (all(strings) and all(((x[0] == ' ') for x in strings))):
        strings = [x[1:] for x in strings]
    return strings

def _validate_join_method(method):
    if (method not in ['left', 'right', 'inner', 'outer']):
        raise ValueError(f'do not recognize join method {method}')

def default_index(n):
    from pandas.core.indexes.range import RangeIndex
    return RangeIndex(0, n, name=None)

def maybe_extract_name(name, obj, cls):
    '\n    If no name is passed, then extract it from data, validating hashability.\n    '
    if ((name is None) and isinstance(obj, (Index, ABCSeries))):
        name = obj.name
    if (not is_hashable(name)):
        raise TypeError(f'{cls.__name__}.name must be a hashable type')
    return name

def _maybe_cast_with_dtype(data, dtype, copy):
    '\n    If a dtype is passed, cast to the closest matching dtype that is supported\n    by Index.\n\n    Parameters\n    ----------\n    data : np.ndarray\n    dtype : np.dtype\n    copy : bool\n\n    Returns\n    -------\n    np.ndarray\n    '
    if is_integer_dtype(dtype):
        inferred = lib.infer_dtype(data, skipna=False)
        if (inferred == 'integer'):
            data = maybe_cast_to_integer_array(data, dtype, copy=copy)
        elif (inferred in ['floating', 'mixed-integer-float']):
            if isna(data).any():
                raise ValueError('cannot convert float NaN to integer')
            if (inferred == 'mixed-integer-float'):
                data = maybe_cast_to_integer_array(data, dtype)
            try:
                data = _try_convert_to_int_array(data, copy, dtype)
            except ValueError:
                data = np.array(data, dtype=np.float64, copy=copy)
        elif (inferred != 'string'):
            data = data.astype(dtype)
    elif is_float_dtype(dtype):
        inferred = lib.infer_dtype(data, skipna=False)
        if (inferred != 'string'):
            data = data.astype(dtype)
    else:
        data = np.array(data, dtype=dtype, copy=copy)
    return data

def _maybe_cast_data_without_dtype(subarr):
    '\n    If we have an arraylike input but no passed dtype, try to infer\n    a supported dtype.\n\n    Parameters\n    ----------\n    subarr : np.ndarray, Index, or Series\n\n    Returns\n    -------\n    converted : np.ndarray or ExtensionArray\n    dtype : np.dtype or ExtensionDtype\n    '
    from pandas.core.arrays import DatetimeArray, IntervalArray, PeriodArray, TimedeltaArray
    assert (subarr.dtype == object), subarr.dtype
    inferred = lib.infer_dtype(subarr, skipna=False)
    if (inferred == 'integer'):
        try:
            data = _try_convert_to_int_array(subarr, False, None)
            return data
        except ValueError:
            pass
        return subarr
    elif (inferred in ['floating', 'mixed-integer-float', 'integer-na']):
        data = np.asarray(subarr).astype(np.float64, copy=False)
        return data
    elif (inferred == 'interval'):
        try:
            data = IntervalArray._from_sequence(subarr, copy=False)
            return data
        except ValueError:
            pass
    elif (inferred == 'boolean'):
        pass
    elif (inferred != 'string'):
        if inferred.startswith('datetime'):
            try:
                data = DatetimeArray._from_sequence(subarr, copy=False)
                return data
            except (ValueError, OutOfBoundsDatetime):
                pass
        elif inferred.startswith('timedelta'):
            data = TimedeltaArray._from_sequence(subarr, copy=False)
            return data
        elif (inferred == 'period'):
            try:
                data = PeriodArray._from_sequence(subarr)
                return data
            except IncompatibleFrequency:
                pass
    return subarr

def _try_convert_to_int_array(data, copy, dtype):
    '\n    Attempt to convert an array of data into an integer array.\n\n    Parameters\n    ----------\n    data : The data to convert.\n    copy : bool\n        Whether to copy the data or not.\n    dtype : np.dtype\n\n    Returns\n    -------\n    int_array : data converted to either an ndarray[int64] or ndarray[uint64]\n\n    Raises\n    ------\n    ValueError if the conversion was not successful.\n    '
    if (not is_unsigned_integer_dtype(dtype)):
        try:
            res = data.astype('i8', copy=False)
            if (res == data).all():
                return res
        except (OverflowError, TypeError, ValueError):
            pass
    try:
        res = data.astype('u8', copy=False)
        if (res == data).all():
            return res
    except (OverflowError, TypeError, ValueError):
        pass
    raise ValueError

def _maybe_asobject(dtype, klass, data, copy, name, **kwargs):
    '\n    If an object dtype was specified, create the non-object Index\n    and then convert it to object.\n\n    Parameters\n    ----------\n    dtype : np.dtype, ExtensionDtype, str\n    klass : Index subclass\n    data : list-like\n    copy : bool\n    name : hashable\n    **kwargs\n\n    Returns\n    -------\n    Index\n\n    Notes\n    -----\n    We assume that calling .astype(object) on this klass will make a copy.\n    '
    if is_dtype_equal(_o_dtype, dtype):
        index = klass(data, copy=False, name=name, **kwargs)
        return index.astype(object)
    return klass(data, dtype=dtype, copy=copy, name=name, **kwargs)

def get_unanimous_names(*indexes):
    "\n    Return common name if all indices agree, otherwise None (level-by-level).\n\n    Parameters\n    ----------\n    indexes : list of Index objects\n\n    Returns\n    -------\n    list\n        A list representing the unanimous 'names' found.\n    "
    name_tups = [tuple(i.names) for i in indexes]
    name_sets = [{*ns} for ns in zip_longest(*name_tups)]
    names = tuple(((ns.pop() if (len(ns) == 1) else None) for ns in name_sets))
    return names

def unpack_nested_dtype(other):
    '\n    When checking if our dtype is comparable with another, we need\n    to unpack CategoricalDtype to look at its categories.dtype.\n\n    Parameters\n    ----------\n    other : Index\n\n    Returns\n    -------\n    Index\n    '
    dtype = other.dtype
    if is_categorical_dtype(dtype):
        return dtype.categories
    return other
