
from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import TYPE_CHECKING, Dict, Hashable, List, Optional, Sequence, Type, TypeVar, Union, cast
from warnings import warn
import numpy as np
from pandas._config import get_option
from pandas._libs import NaT, algos as libalgos, hashtable as htable
from pandas._libs.lib import no_default
from pandas._typing import ArrayLike, Dtype, NpDtype, Ordered, Scalar
from pandas.compat.numpy import function as nv
from pandas.util._decorators import cache_readonly, deprecate_kwarg
from pandas.util._validators import validate_bool_kwarg, validate_fillna_kwargs
from pandas.core.dtypes.cast import coerce_indexer_dtype, maybe_cast_to_extension_array, maybe_infer_to_datetimelike, sanitize_to_nanoseconds
from pandas.core.dtypes.common import ensure_int64, ensure_object, is_categorical_dtype, is_datetime64_dtype, is_dict_like, is_dtype_equal, is_extension_array_dtype, is_hashable, is_integer_dtype, is_list_like, is_object_dtype, is_scalar, is_timedelta64_dtype, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import ABCIndex, ABCSeries
from pandas.core.dtypes.missing import is_valid_nat_for_dtype, isna, notna
from pandas.core import ops
from pandas.core.accessor import PandasDelegate, delegate_names
import pandas.core.algorithms as algorithms
from pandas.core.algorithms import factorize, get_data_algo, take_1d, unique1d
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.base import ExtensionArray, NoNewAttributesMixin, PandasObject
import pandas.core.common as com
from pandas.core.construction import array, extract_array, sanitize_array
from pandas.core.indexers import deprecate_ndim_indexing
from pandas.core.missing import interpolate_2d
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
if TYPE_CHECKING:
    from pandas import Index
CategoricalT = TypeVar('CategoricalT', bound='Categorical')

def _cat_compare_op(op):
    opname = f'__{op.__name__}__'
    fill_value = (True if (op is operator.ne) else False)

    @unpack_zerodim_and_defer(opname)
    def func(self, other):
        hashable = is_hashable(other)
        if (is_list_like(other) and (len(other) != len(self)) and (not hashable)):
            raise ValueError('Lengths must match.')
        if (not self.ordered):
            if (opname in ['__lt__', '__gt__', '__le__', '__ge__']):
                raise TypeError('Unordered Categoricals can only compare equality or not')
        if isinstance(other, Categorical):
            msg = "Categoricals can only be compared if 'categories' are the same."
            if (not self._categories_match_up_to_permutation(other)):
                raise TypeError(msg)
            if ((not self.ordered) and (not self.categories.equals(other.categories))):
                other_codes = recode_for_categories(other.codes, other.categories, self.categories, copy=False)
            else:
                other_codes = other._codes
            ret = op(self._codes, other_codes)
            mask = ((self._codes == (- 1)) | (other_codes == (- 1)))
            if mask.any():
                ret[mask] = fill_value
            return ret
        if hashable:
            if (other in self.categories):
                i = self._unbox_scalar(other)
                ret = op(self._codes, i)
                if (opname not in {'__eq__', '__ge__', '__gt__'}):
                    mask = (self._codes == (- 1))
                    ret[mask] = fill_value
                return ret
            else:
                return ops.invalid_comparison(self, other, op)
        else:
            if (opname not in ['__eq__', '__ne__']):
                raise TypeError(f'''Cannot compare a Categorical for op {opname} with type {type(other)}.
If you want to compare values, use 'np.asarray(cat) <op> other'.''')
            if (isinstance(other, ExtensionArray) and needs_i8_conversion(other.dtype)):
                return op(other, self)
            return getattr(np.array(self), opname)(np.array(other))
    func.__name__ = opname
    return func

def contains(cat, key, container):
    '\n    Helper for membership check for ``key`` in ``cat``.\n\n    This is a helper method for :method:`__contains__`\n    and :class:`CategoricalIndex.__contains__`.\n\n    Returns True if ``key`` is in ``cat.categories`` and the\n    location of ``key`` in ``categories`` is in ``container``.\n\n    Parameters\n    ----------\n    cat : :class:`Categorical`or :class:`categoricalIndex`\n    key : a hashable object\n        The key to check membership for.\n    container : Container (e.g. list-like or mapping)\n        The container to check for membership in.\n\n    Returns\n    -------\n    is_in : bool\n        True if ``key`` is in ``self.categories`` and location of\n        ``key`` in ``categories`` is in ``container``, else False.\n\n    Notes\n    -----\n    This method does not check for NaN values. Do that separately\n    before calling this method.\n    '
    hash(key)
    try:
        loc = cat.categories.get_loc(key)
    except (KeyError, TypeError):
        return False
    if is_scalar(loc):
        return (loc in container)
    else:
        return any(((loc_ in container) for loc_ in loc))

class Categorical(NDArrayBackedExtensionArray, PandasObject, ObjectStringArrayMixin):
    "\n    Represent a categorical variable in classic R / S-plus fashion.\n\n    `Categoricals` can only take on only a limited, and usually fixed, number\n    of possible values (`categories`). In contrast to statistical categorical\n    variables, a `Categorical` might have an order, but numerical operations\n    (additions, divisions, ...) are not possible.\n\n    All values of the `Categorical` are either in `categories` or `np.nan`.\n    Assigning values outside of `categories` will raise a `ValueError`. Order\n    is defined by the order of the `categories`, not lexical order of the\n    values.\n\n    Parameters\n    ----------\n    values : list-like\n        The values of the categorical. If categories are given, values not in\n        categories will be replaced with NaN.\n    categories : Index-like (unique), optional\n        The unique categories for this categorical. If not given, the\n        categories are assumed to be the unique values of `values` (sorted, if\n        possible, otherwise in the order in which they appear).\n    ordered : bool, default False\n        Whether or not this categorical is treated as a ordered categorical.\n        If True, the resulting categorical will be ordered.\n        An ordered categorical respects, when sorted, the order of its\n        `categories` attribute (which in turn is the `categories` argument, if\n        provided).\n    dtype : CategoricalDtype\n        An instance of ``CategoricalDtype`` to use for this categorical.\n\n    Attributes\n    ----------\n    categories : Index\n        The categories of this categorical\n    codes : ndarray\n        The codes (integer positions, which point to the categories) of this\n        categorical, read only.\n    ordered : bool\n        Whether or not this Categorical is ordered.\n    dtype : CategoricalDtype\n        The instance of ``CategoricalDtype`` storing the ``categories``\n        and ``ordered``.\n\n    Methods\n    -------\n    from_codes\n    __array__\n\n    Raises\n    ------\n    ValueError\n        If the categories do not validate.\n    TypeError\n        If an explicit ``ordered=True`` is given but no `categories` and the\n        `values` are not sortable.\n\n    See Also\n    --------\n    CategoricalDtype : Type for categorical data.\n    CategoricalIndex : An Index with an underlying ``Categorical``.\n\n    Notes\n    -----\n    See the `user guide\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_\n    for more.\n\n    Examples\n    --------\n    >>> pd.Categorical([1, 2, 3, 1, 2, 3])\n    [1, 2, 3, 1, 2, 3]\n    Categories (3, int64): [1, 2, 3]\n\n    >>> pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])\n    ['a', 'b', 'c', 'a', 'b', 'c']\n    Categories (3, object): ['a', 'b', 'c']\n\n    Missing values are not included as a category.\n\n    >>> c = pd.Categorical([1, 2, 3, 1, 2, 3, np.nan])\n    >>> c\n    [1, 2, 3, 1, 2, 3, NaN]\n    Categories (3, int64): [1, 2, 3]\n\n    However, their presence is indicated in the `codes` attribute\n    by code `-1`.\n\n    >>> c.codes\n    array([ 0,  1,  2,  0,  1,  2, -1], dtype=int8)\n\n    Ordered `Categoricals` can be sorted according to the custom order\n    of the categories and can have a min and max value.\n\n    >>> c = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True,\n    ...                    categories=['c', 'b', 'a'])\n    >>> c\n    ['a', 'b', 'c', 'a', 'b', 'c']\n    Categories (3, object): ['c' < 'b' < 'a']\n    >>> c.min()\n    'c'\n    "
    __array_priority__ = 1000
    _dtype = CategoricalDtype(ordered=False)
    _hidden_attrs = (PandasObject._hidden_attrs | frozenset(['tolist']))
    _typ = 'categorical'
    _can_hold_na = True

    def __init__(self, values, categories=None, ordered=None, dtype=None, fastpath=False, copy=True):
        dtype = CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)
        if fastpath:
            self._codes = coerce_indexer_dtype(values, dtype.categories)
            self._dtype = self._dtype.update_dtype(dtype)
            return
        if (not is_list_like(values)):
            warn('Allowing scalars in the Categorical constructor is deprecated and will raise in a future version.  Use `[value]` instead', FutureWarning, stacklevel=2)
            values = [values]
        null_mask = np.array(False)
        if is_categorical_dtype(values):
            if (dtype.categories is None):
                dtype = CategoricalDtype(values.categories, dtype.ordered)
        elif (not isinstance(values, (ABCIndex, ABCSeries, ExtensionArray))):
            values = maybe_infer_to_datetimelike(values)
            if (not isinstance(values, (np.ndarray, ExtensionArray))):
                values = com.convert_to_list_like(values)
                sanitize_dtype = (np.dtype('O') if (len(values) == 0) else None)
                null_mask = isna(values)
                if null_mask.any():
                    values = [values[idx] for idx in np.where((~ null_mask))[0]]
                values = sanitize_array(values, None, dtype=sanitize_dtype)
            else:
                values = sanitize_to_nanoseconds(values)
        if (dtype.categories is None):
            try:
                (codes, categories) = factorize(values, sort=True)
            except TypeError as err:
                (codes, categories) = factorize(values, sort=False)
                if dtype.ordered:
                    raise TypeError("'values' is not ordered, please explicitly specify the categories order by passing in a categories argument.") from err
            except ValueError as err:
                raise NotImplementedError('> 1 ndim Categorical are not supported at this time') from err
            dtype = CategoricalDtype(categories, dtype.ordered)
        elif is_categorical_dtype(values.dtype):
            old_codes = extract_array(values)._codes
            codes = recode_for_categories(old_codes, values.dtype.categories, dtype.categories, copy=copy)
        else:
            codes = _get_codes_for_values(values, dtype.categories)
        if null_mask.any():
            full_codes = (- np.ones(null_mask.shape, dtype=codes.dtype))
            full_codes[(~ null_mask)] = codes
            codes = full_codes
        self._dtype = self._dtype.update_dtype(dtype)
        self._codes = coerce_indexer_dtype(codes, dtype.categories)

    @property
    def dtype(self):
        '\n        The :class:`~pandas.api.types.CategoricalDtype` for this instance.\n        '
        return self._dtype

    @property
    def _constructor(self):
        return Categorical

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return Categorical(scalars, dtype=dtype, copy=copy)

    def astype(self, dtype, copy=True):
        '\n        Coerce this type to another dtype\n\n        Parameters\n        ----------\n        dtype : numpy dtype or pandas type\n        copy : bool, default True\n            By default, astype always returns a newly allocated object.\n            If copy is set to False and dtype is categorical, the original\n            object is returned.\n        '
        dtype = pandas_dtype(dtype)
        if (self.dtype is dtype):
            result = (self.copy() if copy else self)
        elif is_categorical_dtype(dtype):
            dtype = cast(Union[(str, CategoricalDtype)], dtype)
            dtype = self.dtype.update_dtype(dtype)
            self = (self.copy() if copy else self)
            result = self._set_dtype(dtype)
        elif is_extension_array_dtype(dtype):
            result = array(self, dtype=dtype, copy=copy)
        elif (is_integer_dtype(dtype) and self.isna().any()):
            raise ValueError('Cannot convert float NaN to integer')
        elif ((len(self.codes) == 0) or (len(self.categories) == 0)):
            result = np.array(self, dtype=dtype, copy=copy)
        else:
            try:
                astyped_cats = self.categories.astype(dtype=dtype, copy=copy)
            except (TypeError, ValueError):
                msg = f'Cannot cast {self.categories.dtype} dtype to {dtype}'
                raise ValueError(msg)
            astyped_cats = extract_array(astyped_cats, extract_numpy=True)
            result = take_1d(astyped_cats, libalgos.ensure_platform_int(self._codes))
        return result

    @cache_readonly
    def itemsize(self):
        '\n        return the size of a single category\n        '
        return self.categories.itemsize

    def tolist(self):
        '\n        Return a list of the values.\n\n        These are each a scalar type, which is a Python scalar\n        (for str, int, float) or a pandas scalar\n        (for Timestamp/Timedelta/Interval/Period)\n        '
        return list(self)
    to_list = tolist

    @classmethod
    def _from_inferred_categories(cls, inferred_categories, inferred_codes, dtype, true_values=None):
        '\n        Construct a Categorical from inferred values.\n\n        For inferred categories (`dtype` is None) the categories are sorted.\n        For explicit `dtype`, the `inferred_categories` are cast to the\n        appropriate type.\n\n        Parameters\n        ----------\n        inferred_categories : Index\n        inferred_codes : Index\n        dtype : CategoricalDtype or \'category\'\n        true_values : list, optional\n            If none are provided, the default ones are\n            "True", "TRUE", and "true."\n\n        Returns\n        -------\n        Categorical\n        '
        from pandas import Index, to_datetime, to_numeric, to_timedelta
        cats = Index(inferred_categories)
        known_categories = (isinstance(dtype, CategoricalDtype) and (dtype.categories is not None))
        if known_categories:
            if dtype.categories.is_numeric():
                cats = to_numeric(inferred_categories, errors='coerce')
            elif is_datetime64_dtype(dtype.categories):
                cats = to_datetime(inferred_categories, errors='coerce')
            elif is_timedelta64_dtype(dtype.categories):
                cats = to_timedelta(inferred_categories, errors='coerce')
            elif dtype.categories.is_boolean():
                if (true_values is None):
                    true_values = ['True', 'TRUE', 'true']
                cats = cats.isin(true_values)
        if known_categories:
            categories = dtype.categories
            codes = recode_for_categories(inferred_codes, cats, categories)
        elif (not cats.is_monotonic_increasing):
            unsorted = cats.copy()
            categories = cats.sort_values()
            codes = recode_for_categories(inferred_codes, unsorted, categories)
            dtype = CategoricalDtype(categories, ordered=False)
        else:
            dtype = CategoricalDtype(cats, ordered=False)
            codes = inferred_codes
        return cls(codes, dtype=dtype, fastpath=True)

    @classmethod
    def from_codes(cls, codes, categories=None, ordered=None, dtype=None):
        '\n        Make a Categorical type from codes and categories or dtype.\n\n        This constructor is useful if you already have codes and\n        categories/dtype and so do not need the (computation intensive)\n        factorization step, which is usually done on the constructor.\n\n        If your data does not follow this convention, please use the normal\n        constructor.\n\n        Parameters\n        ----------\n        codes : array-like of int\n            An integer array, where each integer points to a category in\n            categories or dtype.categories, or else is -1 for NaN.\n        categories : index-like, optional\n            The categories for the categorical. Items need to be unique.\n            If the categories are not given here, then they must be provided\n            in `dtype`.\n        ordered : bool, optional\n            Whether or not this categorical is treated as an ordered\n            categorical. If not given here or in `dtype`, the resulting\n            categorical will be unordered.\n        dtype : CategoricalDtype or "category", optional\n            If :class:`CategoricalDtype`, cannot be used together with\n            `categories` or `ordered`.\n\n            .. versionadded:: 0.24.0\n\n               When `dtype` is provided, neither `categories` nor `ordered`\n               should be provided.\n\n        Returns\n        -------\n        Categorical\n\n        Examples\n        --------\n        >>> dtype = pd.CategoricalDtype([\'a\', \'b\'], ordered=True)\n        >>> pd.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)\n        [\'a\', \'b\', \'a\', \'b\']\n        Categories (2, object): [\'a\' < \'b\']\n        '
        dtype = CategoricalDtype._from_values_or_dtype(categories=categories, ordered=ordered, dtype=dtype)
        if (dtype.categories is None):
            msg = "The categories must be provided in 'categories' or 'dtype'. Both were None."
            raise ValueError(msg)
        if (is_extension_array_dtype(codes) and is_integer_dtype(codes)):
            if isna(codes).any():
                raise ValueError('codes cannot contain NA values')
            codes = codes.to_numpy(dtype=np.int64)
        else:
            codes = np.asarray(codes)
        if (len(codes) and (not is_integer_dtype(codes))):
            raise ValueError('codes need to be array-like integers')
        if (len(codes) and ((codes.max() >= len(dtype.categories)) or (codes.min() < (- 1)))):
            raise ValueError('codes need to be between -1 and len(categories)-1')
        return cls(codes, dtype=dtype, fastpath=True)

    @property
    def categories(self):
        '\n        The categories of this categorical.\n\n        Setting assigns new values to each category (effectively a rename of\n        each individual category).\n\n        The assigned value has to be a list-like object. All items must be\n        unique and the number of items in the new categories must be the same\n        as the number of items in the old categories.\n\n        Assigning to `categories` is a inplace operation!\n\n        Raises\n        ------\n        ValueError\n            If the new categories do not validate as categories or if the\n            number of new categories is unequal the number of old categories\n\n        See Also\n        --------\n        rename_categories : Rename categories.\n        reorder_categories : Reorder categories.\n        add_categories : Add new categories.\n        remove_categories : Remove the specified categories.\n        remove_unused_categories : Remove categories which are not used.\n        set_categories : Set the categories to the specified ones.\n        '
        return self.dtype.categories

    @categories.setter
    def categories(self, categories):
        new_dtype = CategoricalDtype(categories, ordered=self.ordered)
        if ((self.dtype.categories is not None) and (len(self.dtype.categories) != len(new_dtype.categories))):
            raise ValueError('new categories need to have the same number of items as the old categories!')
        self._dtype = new_dtype

    @property
    def ordered(self):
        '\n        Whether the categories have an ordered relationship.\n        '
        return self.dtype.ordered

    @property
    def codes(self):
        '\n        The category codes of this categorical.\n\n        Codes are an array of integers which are the positions of the actual\n        values in the categories array.\n\n        There is no setter, use the other categorical methods and the normal item\n        setter to change values in the categorical.\n\n        Returns\n        -------\n        ndarray[int]\n            A non-writable view of the `codes` array.\n        '
        v = self._codes.view()
        v.flags.writeable = False
        return v

    def _set_categories(self, categories, fastpath=False):
        "\n        Sets new categories inplace\n\n        Parameters\n        ----------\n        fastpath : bool, default False\n           Don't perform validation of the categories for uniqueness or nulls\n\n        Examples\n        --------\n        >>> c = pd.Categorical(['a', 'b'])\n        >>> c\n        ['a', 'b']\n        Categories (2, object): ['a', 'b']\n\n        >>> c._set_categories(pd.Index(['a', 'c']))\n        >>> c\n        ['a', 'c']\n        Categories (2, object): ['a', 'c']\n        "
        if fastpath:
            new_dtype = CategoricalDtype._from_fastpath(categories, self.ordered)
        else:
            new_dtype = CategoricalDtype(categories, ordered=self.ordered)
        if ((not fastpath) and (self.dtype.categories is not None) and (len(new_dtype.categories) != len(self.dtype.categories))):
            raise ValueError('new categories need to have the same number of items than the old categories!')
        self._dtype = new_dtype

    def _set_dtype(self, dtype):
        "\n        Internal method for directly updating the CategoricalDtype\n\n        Parameters\n        ----------\n        dtype : CategoricalDtype\n\n        Notes\n        -----\n        We don't do any validation here. It's assumed that the dtype is\n        a (valid) instance of `CategoricalDtype`.\n        "
        codes = recode_for_categories(self.codes, self.categories, dtype.categories)
        return type(self)(codes, dtype=dtype, fastpath=True)

    def set_ordered(self, value, inplace=False):
        '\n        Set the ordered attribute to the boolean value.\n\n        Parameters\n        ----------\n        value : bool\n           Set whether this categorical is ordered (True) or not (False).\n        inplace : bool, default False\n           Whether or not to set the ordered attribute in-place or return\n           a copy of this categorical with ordered set to the value.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        new_dtype = CategoricalDtype(self.categories, ordered=value)
        cat = (self if inplace else self.copy())
        cat._dtype = new_dtype
        if (not inplace):
            return cat

    def as_ordered(self, inplace=False):
        '\n        Set the Categorical to be ordered.\n\n        Parameters\n        ----------\n        inplace : bool, default False\n           Whether or not to set the ordered attribute in-place or return\n           a copy of this categorical with ordered set to True.\n\n        Returns\n        -------\n        Categorical or None\n            Ordered Categorical or None if ``inplace=True``.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        return self.set_ordered(True, inplace=inplace)

    def as_unordered(self, inplace=False):
        '\n        Set the Categorical to be unordered.\n\n        Parameters\n        ----------\n        inplace : bool, default False\n           Whether or not to set the ordered attribute in-place or return\n           a copy of this categorical with ordered set to False.\n\n        Returns\n        -------\n        Categorical or None\n            Unordered Categorical or None if ``inplace=True``.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        return self.set_ordered(False, inplace=inplace)

    def set_categories(self, new_categories, ordered=None, rename=False, inplace=False):
        '\n        Set the categories to the specified new_categories.\n\n        `new_categories` can include new categories (which will result in\n        unused categories) or remove old categories (which results in values\n        set to NaN). If `rename==True`, the categories will simple be renamed\n        (less or more items than in old categories will result in values set to\n        NaN or in unused categories respectively).\n\n        This method can be used to perform more than one action of adding,\n        removing, and reordering simultaneously and is therefore faster than\n        performing the individual steps via the more specialised methods.\n\n        On the other hand this methods does not do checks (e.g., whether the\n        old categories are included in the new categories on a reorder), which\n        can result in surprising changes, for example when using special string\n        dtypes, which does not considers a S1 string equal to a single char\n        python string.\n\n        Parameters\n        ----------\n        new_categories : Index-like\n           The categories in new order.\n        ordered : bool, default False\n           Whether or not the categorical is treated as a ordered categorical.\n           If not given, do not change the ordered information.\n        rename : bool, default False\n           Whether or not the new_categories should be considered as a rename\n           of the old categories or as reordered categories.\n        inplace : bool, default False\n           Whether or not to reorder the categories in-place or return a copy\n           of this categorical with reordered categories.\n\n        Returns\n        -------\n        Categorical with reordered categories or None if inplace.\n\n        Raises\n        ------\n        ValueError\n            If new_categories does not validate as categories\n\n        See Also\n        --------\n        rename_categories : Rename categories.\n        reorder_categories : Reorder categories.\n        add_categories : Add new categories.\n        remove_categories : Remove the specified categories.\n        remove_unused_categories : Remove categories which are not used.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (ordered is None):
            ordered = self.dtype.ordered
        new_dtype = CategoricalDtype(new_categories, ordered=ordered)
        cat = (self if inplace else self.copy())
        if rename:
            if ((cat.dtype.categories is not None) and (len(new_dtype.categories) < len(cat.dtype.categories))):
                cat._codes[(cat._codes >= len(new_dtype.categories))] = (- 1)
        else:
            codes = recode_for_categories(cat.codes, cat.categories, new_dtype.categories)
            cat._codes = codes
        cat._dtype = new_dtype
        if (not inplace):
            return cat

    def rename_categories(self, new_categories, inplace=False):
        "\n        Rename categories.\n\n        Parameters\n        ----------\n        new_categories : list-like, dict-like or callable\n\n            New categories which will replace old categories.\n\n            * list-like: all items must be unique and the number of items in\n              the new categories must match the existing number of categories.\n\n            * dict-like: specifies a mapping from\n              old categories to new. Categories not contained in the mapping\n              are passed through and extra categories in the mapping are\n              ignored.\n\n            * callable : a callable that is called on all items in the old\n              categories and whose return values comprise the new categories.\n\n        inplace : bool, default False\n            Whether or not to rename the categories inplace or return a copy of\n            this categorical with renamed categories.\n\n        Returns\n        -------\n        cat : Categorical or None\n            Categorical with removed categories or None if ``inplace=True``.\n\n        Raises\n        ------\n        ValueError\n            If new categories are list-like and do not have the same number of\n            items than the current categories or do not validate as categories\n\n        See Also\n        --------\n        reorder_categories : Reorder categories.\n        add_categories : Add new categories.\n        remove_categories : Remove the specified categories.\n        remove_unused_categories : Remove categories which are not used.\n        set_categories : Set the categories to the specified ones.\n\n        Examples\n        --------\n        >>> c = pd.Categorical(['a', 'a', 'b'])\n        >>> c.rename_categories([0, 1])\n        [0, 0, 1]\n        Categories (2, int64): [0, 1]\n\n        For dict-like ``new_categories``, extra keys are ignored and\n        categories not in the dictionary are passed through\n\n        >>> c.rename_categories({'a': 'A', 'c': 'C'})\n        ['A', 'A', 'b']\n        Categories (2, object): ['A', 'b']\n\n        You may also provide a callable to create the new categories\n\n        >>> c.rename_categories(lambda x: x.upper())\n        ['A', 'A', 'B']\n        Categories (2, object): ['A', 'B']\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        cat = (self if inplace else self.copy())
        if is_dict_like(new_categories):
            cat.categories = [new_categories.get(item, item) for item in cat.categories]
        elif callable(new_categories):
            cat.categories = [new_categories(item) for item in cat.categories]
        else:
            cat.categories = new_categories
        if (not inplace):
            return cat

    def reorder_categories(self, new_categories, ordered=None, inplace=False):
        '\n        Reorder categories as specified in new_categories.\n\n        `new_categories` need to include all old categories and no new category\n        items.\n\n        Parameters\n        ----------\n        new_categories : Index-like\n           The categories in new order.\n        ordered : bool, optional\n           Whether or not the categorical is treated as a ordered categorical.\n           If not given, do not change the ordered information.\n        inplace : bool, default False\n           Whether or not to reorder the categories inplace or return a copy of\n           this categorical with reordered categories.\n\n        Returns\n        -------\n        cat : Categorical or None\n            Categorical with removed categories or None if ``inplace=True``.\n\n        Raises\n        ------\n        ValueError\n            If the new categories do not contain all old category items or any\n            new ones\n\n        See Also\n        --------\n        rename_categories : Rename categories.\n        add_categories : Add new categories.\n        remove_categories : Remove the specified categories.\n        remove_unused_categories : Remove categories which are not used.\n        set_categories : Set the categories to the specified ones.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (set(self.dtype.categories) != set(new_categories)):
            raise ValueError('items in new_categories are not the same as in old categories')
        return self.set_categories(new_categories, ordered=ordered, inplace=inplace)

    def add_categories(self, new_categories, inplace=False):
        '\n        Add new categories.\n\n        `new_categories` will be included at the last/highest place in the\n        categories and will be unused directly after this call.\n\n        Parameters\n        ----------\n        new_categories : category or list-like of category\n           The new categories to be included.\n        inplace : bool, default False\n           Whether or not to add the categories inplace or return a copy of\n           this categorical with added categories.\n\n        Returns\n        -------\n        cat : Categorical or None\n            Categorical with new categories added or None if ``inplace=True``.\n\n        Raises\n        ------\n        ValueError\n            If the new categories include old categories or do not validate as\n            categories\n\n        See Also\n        --------\n        rename_categories : Rename categories.\n        reorder_categories : Reorder categories.\n        remove_categories : Remove the specified categories.\n        remove_unused_categories : Remove categories which are not used.\n        set_categories : Set the categories to the specified ones.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (not is_list_like(new_categories)):
            new_categories = [new_categories]
        already_included = (set(new_categories) & set(self.dtype.categories))
        if (len(already_included) != 0):
            raise ValueError(f'new categories must not include old categories: {already_included}')
        new_categories = (list(self.dtype.categories) + list(new_categories))
        new_dtype = CategoricalDtype(new_categories, self.ordered)
        cat = (self if inplace else self.copy())
        cat._dtype = new_dtype
        cat._codes = coerce_indexer_dtype(cat._codes, new_dtype.categories)
        if (not inplace):
            return cat

    def remove_categories(self, removals, inplace=False):
        '\n        Remove the specified categories.\n\n        `removals` must be included in the old categories. Values which were in\n        the removed categories will be set to NaN\n\n        Parameters\n        ----------\n        removals : category or list of categories\n           The categories which should be removed.\n        inplace : bool, default False\n           Whether or not to remove the categories inplace or return a copy of\n           this categorical with removed categories.\n\n        Returns\n        -------\n        cat : Categorical or None\n            Categorical with removed categories or None if ``inplace=True``.\n\n        Raises\n        ------\n        ValueError\n            If the removals are not contained in the categories\n\n        See Also\n        --------\n        rename_categories : Rename categories.\n        reorder_categories : Reorder categories.\n        add_categories : Add new categories.\n        remove_unused_categories : Remove categories which are not used.\n        set_categories : Set the categories to the specified ones.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (not is_list_like(removals)):
            removals = [removals]
        removal_set = set(removals)
        not_included = (removal_set - set(self.dtype.categories))
        new_categories = [c for c in self.dtype.categories if (c not in removal_set)]
        if any(isna(removals)):
            not_included = {x for x in not_included if notna(x)}
            new_categories = [x for x in new_categories if notna(x)]
        if (len(not_included) != 0):
            raise ValueError(f'removals must all be in old categories: {not_included}')
        return self.set_categories(new_categories, ordered=self.ordered, rename=False, inplace=inplace)

    def remove_unused_categories(self, inplace=no_default):
        '\n        Remove categories which are not used.\n\n        Parameters\n        ----------\n        inplace : bool, default False\n           Whether or not to drop unused categories inplace or return a copy of\n           this categorical with unused categories dropped.\n\n           .. deprecated:: 1.2.0\n\n        Returns\n        -------\n        cat : Categorical or None\n            Categorical with unused categories dropped or None if ``inplace=True``.\n\n        See Also\n        --------\n        rename_categories : Rename categories.\n        reorder_categories : Reorder categories.\n        add_categories : Add new categories.\n        remove_categories : Remove the specified categories.\n        set_categories : Set the categories to the specified ones.\n        '
        if (inplace is not no_default):
            warn('The `inplace` parameter in pandas.Categorical.remove_unused_categories is deprecated and will be removed in a future version.', FutureWarning, stacklevel=2)
        else:
            inplace = False
        inplace = validate_bool_kwarg(inplace, 'inplace')
        cat = (self if inplace else self.copy())
        (idx, inv) = np.unique(cat._codes, return_inverse=True)
        if ((idx.size != 0) and (idx[0] == (- 1))):
            (idx, inv) = (idx[1:], (inv - 1))
        new_categories = cat.dtype.categories.take(idx)
        new_dtype = CategoricalDtype._from_fastpath(new_categories, ordered=self.ordered)
        cat._dtype = new_dtype
        cat._codes = coerce_indexer_dtype(inv, new_dtype.categories)
        if (not inplace):
            return cat

    def map(self, mapper):
        "\n        Map categories using input correspondence (dict, Series, or function).\n\n        Maps the categories to new categories. If the mapping correspondence is\n        one-to-one the result is a :class:`~pandas.Categorical` which has the\n        same order property as the original, otherwise a :class:`~pandas.Index`\n        is returned. NaN values are unaffected.\n\n        If a `dict` or :class:`~pandas.Series` is used any unmapped category is\n        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`\n        will be returned.\n\n        Parameters\n        ----------\n        mapper : function, dict, or Series\n            Mapping correspondence.\n\n        Returns\n        -------\n        pandas.Categorical or pandas.Index\n            Mapped categorical.\n\n        See Also\n        --------\n        CategoricalIndex.map : Apply a mapping correspondence on a\n            :class:`~pandas.CategoricalIndex`.\n        Index.map : Apply a mapping correspondence on an\n            :class:`~pandas.Index`.\n        Series.map : Apply a mapping correspondence on a\n            :class:`~pandas.Series`.\n        Series.apply : Apply more complex functions on a\n            :class:`~pandas.Series`.\n\n        Examples\n        --------\n        >>> cat = pd.Categorical(['a', 'b', 'c'])\n        >>> cat\n        ['a', 'b', 'c']\n        Categories (3, object): ['a', 'b', 'c']\n        >>> cat.map(lambda x: x.upper())\n        ['A', 'B', 'C']\n        Categories (3, object): ['A', 'B', 'C']\n        >>> cat.map({'a': 'first', 'b': 'second', 'c': 'third'})\n        ['first', 'second', 'third']\n        Categories (3, object): ['first', 'second', 'third']\n\n        If the mapping is one-to-one the ordering of the categories is\n        preserved:\n\n        >>> cat = pd.Categorical(['a', 'b', 'c'], ordered=True)\n        >>> cat\n        ['a', 'b', 'c']\n        Categories (3, object): ['a' < 'b' < 'c']\n        >>> cat.map({'a': 3, 'b': 2, 'c': 1})\n        [3, 2, 1]\n        Categories (3, int64): [3 < 2 < 1]\n\n        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:\n\n        >>> cat.map({'a': 'first', 'b': 'second', 'c': 'first'})\n        Index(['first', 'second', 'first'], dtype='object')\n\n        If a `dict` is used, all unmapped categories are mapped to `NaN` and\n        the result is an :class:`~pandas.Index`:\n\n        >>> cat.map({'a': 'first', 'b': 'second'})\n        Index(['first', 'second', nan], dtype='object')\n        "
        new_categories = self.categories.map(mapper)
        try:
            return self.from_codes(self._codes.copy(), categories=new_categories, ordered=self.ordered)
        except ValueError:
            if np.any((self._codes == (- 1))):
                new_categories = new_categories.insert(len(new_categories), np.nan)
            return np.take(new_categories, self._codes)
    __eq__ = _cat_compare_op(operator.eq)
    __ne__ = _cat_compare_op(operator.ne)
    __lt__ = _cat_compare_op(operator.lt)
    __gt__ = _cat_compare_op(operator.gt)
    __le__ = _cat_compare_op(operator.le)
    __ge__ = _cat_compare_op(operator.ge)

    def _validate_searchsorted_value(self, value):
        if is_scalar(value):
            codes = self._unbox_scalar(value)
        else:
            locs = [self.categories.get_loc(x) for x in value]
            codes = np.array(locs, dtype=self.codes.dtype)
        return codes

    def _validate_fill_value(self, fill_value):
        '\n        Convert a user-facing fill_value to a representation to use with our\n        underlying ndarray, raising TypeError if this is not possible.\n\n        Parameters\n        ----------\n        fill_value : object\n\n        Returns\n        -------\n        fill_value : int\n\n        Raises\n        ------\n        TypeError\n        '
        if is_valid_nat_for_dtype(fill_value, self.categories.dtype):
            fill_value = (- 1)
        elif (fill_value in self.categories):
            fill_value = self._unbox_scalar(fill_value)
        else:
            raise TypeError(f"'fill_value={fill_value}' is not present in this Categorical's categories")
        return fill_value
    _validate_scalar = _validate_fill_value

    def __array__(self, dtype=None):
        '\n        The numpy array interface.\n\n        Returns\n        -------\n        numpy.array\n            A numpy array of either the specified dtype or,\n            if dtype==None (default), the same dtype as\n            categorical.categories.dtype.\n        '
        ret = take_1d(self.categories._values, self._codes)
        if (dtype and (not is_dtype_equal(dtype, self.categories.dtype))):
            return np.asarray(ret, dtype)
        return np.asarray(ret)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        result = ops.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if (result is not NotImplemented):
            return result
        raise TypeError(f'Object with dtype {self.dtype} cannot perform the numpy op {ufunc.__name__}')

    def __setstate__(self, state):
        'Necessary for making this object picklable'
        if (not isinstance(state, dict)):
            raise Exception('invalid pickle state')
        if ('_dtype' not in state):
            state['_dtype'] = CategoricalDtype(state['_categories'], state['_ordered'])
        for (k, v) in state.items():
            setattr(self, k, v)

    @property
    def nbytes(self):
        return (self._codes.nbytes + self.dtype.categories.values.nbytes)

    def memory_usage(self, deep=False):
        '\n        Memory usage of my values\n\n        Parameters\n        ----------\n        deep : bool\n            Introspect the data deeply, interrogate\n            `object` dtypes for system-level memory consumption\n\n        Returns\n        -------\n        bytes used\n\n        Notes\n        -----\n        Memory usage does not include memory consumed by elements that\n        are not components of the array if deep=False\n\n        See Also\n        --------\n        numpy.ndarray.nbytes\n        '
        return (self._codes.nbytes + self.dtype.categories.memory_usage(deep=deep))

    def isna(self):
        '\n        Detect missing values\n\n        Missing values (-1 in .codes) are detected.\n\n        Returns\n        -------\n        a boolean array of whether my values are null\n\n        See Also\n        --------\n        isna : Top-level isna.\n        isnull : Alias of isna.\n        Categorical.notna : Boolean inverse of Categorical.isna.\n\n        '
        return (self._codes == (- 1))
    isnull = isna

    def notna(self):
        '\n        Inverse of isna\n\n        Both missing values (-1 in .codes) and NA as a category are detected as\n        null.\n\n        Returns\n        -------\n        a boolean array of whether my values are not null\n\n        See Also\n        --------\n        notna : Top-level notna.\n        notnull : Alias of notna.\n        Categorical.isna : Boolean inverse of Categorical.notna.\n\n        '
        return (~ self.isna())
    notnull = notna

    def value_counts(self, dropna=True):
        "\n        Return a Series containing counts of each category.\n\n        Every category will have an entry, even those with a count of 0.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include counts of NaN.\n\n        Returns\n        -------\n        counts : Series\n\n        See Also\n        --------\n        Series.value_counts\n        "
        from pandas import CategoricalIndex, Series
        (code, cat) = (self._codes, self.categories)
        (ncat, mask) = (len(cat), (code >= 0))
        (ix, clean) = (np.arange(ncat), mask.all())
        if (dropna or clean):
            obs = (code if clean else code[mask])
            count = np.bincount(obs, minlength=(ncat or 0))
        else:
            count = np.bincount(np.where(mask, code, ncat))
            ix = np.append(ix, (- 1))
        ix = self._from_backing_data(ix)
        return Series(count, index=CategoricalIndex(ix), dtype='int64')

    def _internal_get_values(self):
        '\n        Return the values.\n\n        For internal compatibility with pandas formatting.\n\n        Returns\n        -------\n        np.ndarray or Index\n            A numpy array of the same dtype as categorical.categories.dtype or\n            Index if datetime / periods.\n        '
        if needs_i8_conversion(self.categories.dtype):
            return self.categories.take(self._codes, fill_value=NaT)
        elif (is_integer_dtype(self.categories) and ((- 1) in self._codes)):
            return self.categories.astype('object').take(self._codes, fill_value=np.nan)
        return np.array(self)

    def check_for_ordered(self, op):
        ' assert that we are ordered '
        if (not self.ordered):
            raise TypeError(f'''Categorical is not ordered for operation {op}
you can use .as_ordered() to change the Categorical to an ordered one
''')

    def argsort(self, ascending=True, kind='quicksort', **kwargs):
        "\n        Return the indices that would sort the Categorical.\n\n        .. versionchanged:: 0.25.0\n\n           Changed to sort missing values at the end.\n\n        Parameters\n        ----------\n        ascending : bool, default True\n            Whether the indices should result in an ascending\n            or descending sort.\n        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional\n            Sorting algorithm.\n        **kwargs:\n            passed through to :func:`numpy.argsort`.\n\n        Returns\n        -------\n        numpy.array\n\n        See Also\n        --------\n        numpy.ndarray.argsort\n\n        Notes\n        -----\n        While an ordering is applied to the category values, arg-sorting\n        in this context refers more to organizing and grouping together\n        based on matching category values. Thus, this function can be\n        called on an unordered Categorical instance unlike the functions\n        'Categorical.min' and 'Categorical.max'.\n\n        Examples\n        --------\n        >>> pd.Categorical(['b', 'b', 'a', 'c']).argsort()\n        array([2, 0, 1, 3])\n\n        >>> cat = pd.Categorical(['b', 'b', 'a', 'c'],\n        ...                      categories=['c', 'b', 'a'],\n        ...                      ordered=True)\n        >>> cat.argsort()\n        array([3, 0, 1, 2])\n\n        Missing values are placed at the end\n\n        >>> cat = pd.Categorical([2, None, 1])\n        >>> cat.argsort()\n        array([2, 0, 1])\n        "
        return super().argsort(ascending=ascending, kind=kind, **kwargs)

    def sort_values(self, inplace=False, ascending=True, na_position='last'):
        "\n        Sort the Categorical by category value returning a new\n        Categorical by default.\n\n        While an ordering is applied to the category values, sorting in this\n        context refers more to organizing and grouping together based on\n        matching category values. Thus, this function can be called on an\n        unordered Categorical instance unlike the functions 'Categorical.min'\n        and 'Categorical.max'.\n\n        Parameters\n        ----------\n        inplace : bool, default False\n            Do operation in place.\n        ascending : bool, default True\n            Order ascending. Passing False orders descending. The\n            ordering parameter provides the method by which the\n            category values are organized.\n        na_position : {'first', 'last'} (optional, default='last')\n            'first' puts NaNs at the beginning\n            'last' puts NaNs at the end\n\n        Returns\n        -------\n        Categorical or None\n\n        See Also\n        --------\n        Categorical.sort\n        Series.sort_values\n\n        Examples\n        --------\n        >>> c = pd.Categorical([1, 2, 2, 1, 5])\n        >>> c\n        [1, 2, 2, 1, 5]\n        Categories (3, int64): [1, 2, 5]\n        >>> c.sort_values()\n        [1, 1, 2, 2, 5]\n        Categories (3, int64): [1, 2, 5]\n        >>> c.sort_values(ascending=False)\n        [5, 2, 2, 1, 1]\n        Categories (3, int64): [1, 2, 5]\n\n        Inplace sorting can be done as well:\n\n        >>> c.sort_values(inplace=True)\n        >>> c\n        [1, 1, 2, 2, 5]\n        Categories (3, int64): [1, 2, 5]\n        >>>\n        >>> c = pd.Categorical([1, 2, 2, 1, 5])\n\n        'sort_values' behaviour with NaNs. Note that 'na_position'\n        is independent of the 'ascending' parameter:\n\n        >>> c = pd.Categorical([np.nan, 2, 2, np.nan, 5])\n        >>> c\n        [NaN, 2, 2, NaN, 5]\n        Categories (2, int64): [2, 5]\n        >>> c.sort_values()\n        [2, 2, 5, NaN, NaN]\n        Categories (2, int64): [2, 5]\n        >>> c.sort_values(ascending=False)\n        [5, 2, 2, NaN, NaN]\n        Categories (2, int64): [2, 5]\n        >>> c.sort_values(na_position='first')\n        [NaN, NaN, 2, 2, 5]\n        Categories (2, int64): [2, 5]\n        >>> c.sort_values(ascending=False, na_position='first')\n        [NaN, NaN, 5, 2, 2]\n        Categories (2, int64): [2, 5]\n        "
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (na_position not in ['last', 'first']):
            raise ValueError(f'invalid na_position: {repr(na_position)}')
        sorted_idx = nargsort(self, ascending=ascending, na_position=na_position)
        if inplace:
            self._codes[:] = self._codes[sorted_idx]
        else:
            codes = self._codes[sorted_idx]
            return self._from_backing_data(codes)

    def _values_for_rank(self):
        '\n        For correctly ranking ordered categorical data. See GH#15420\n\n        Ordered categorical data should be ranked on the basis of\n        codes with -1 translated to NaN.\n\n        Returns\n        -------\n        numpy.array\n\n        '
        from pandas import Series
        if self.ordered:
            values = self.codes
            mask = (values == (- 1))
            if mask.any():
                values = values.astype('float64')
                values[mask] = np.nan
        elif self.categories.is_numeric():
            values = np.array(self)
        else:
            values = np.array(self.rename_categories(Series(self.categories).rank().values))
        return values

    def view(self, dtype=None):
        if (dtype is not None):
            raise NotImplementedError(dtype)
        return self._from_backing_data(self._ndarray)

    def to_dense(self):
        "\n        Return my 'dense' representation\n\n        For internal compatibility with numpy arrays.\n\n        Returns\n        -------\n        dense : array\n        "
        warn('Categorical.to_dense is deprecated and will be removed in a future version.  Use np.asarray(cat) instead.', FutureWarning, stacklevel=2)
        return np.asarray(self)

    def fillna(self, value=None, method=None, limit=None):
        "\n        Fill NA/NaN values using the specified method.\n\n        Parameters\n        ----------\n        value : scalar, dict, Series\n            If a scalar value is passed it is used to fill all missing values.\n            Alternatively, a Series or dict can be used to fill in different\n            values for each index. The value should not be a list. The\n            value(s) passed should either be in the categories or should be\n            NaN.\n        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None\n            Method to use for filling holes in reindexed Series\n            pad / ffill: propagate last valid observation forward to next valid\n            backfill / bfill: use NEXT valid observation to fill gap\n        limit : int, default None\n            (Not implemented yet for Categorical!)\n            If method is specified, this is the maximum number of consecutive\n            NaN values to forward/backward fill. In other words, if there is\n            a gap with more than this number of consecutive NaNs, it will only\n            be partially filled. If method is not specified, this is the\n            maximum number of entries along the entire axis where NaNs will be\n            filled.\n\n        Returns\n        -------\n        filled : Categorical with NA/NaN filled\n        "
        (value, method) = validate_fillna_kwargs(value, method, validate_scalar_dict_value=False)
        value = extract_array(value, extract_numpy=True)
        if (value is None):
            value = np.nan
        if (limit is not None):
            raise NotImplementedError('specifying a limit for fillna has not been implemented yet')
        if (method is not None):
            values = np.asarray(self).reshape((- 1), len(self))
            values = interpolate_2d(values, method, 0, None).astype(self.categories.dtype)[0]
            codes = _get_codes_for_values(values, self.categories)
        else:
            codes = self._ndarray.copy()
            mask = self.isna()
            new_codes = self._validate_setitem_value(value)
            np.putmask(codes, mask, new_codes)
        return self._from_backing_data(codes)

    @property
    def _ndarray(self):
        return self._codes

    def _from_backing_data(self, arr):
        return self._constructor(arr, dtype=self.dtype, fastpath=True)

    def _box_func(self, i):
        if (i == (- 1)):
            return np.NaN
        return self.categories[i]

    def _unbox_scalar(self, key):
        code = self.categories.get_loc(key)
        code = self._codes.dtype.type(code)
        return code

    def take_nd(self, indexer, allow_fill=False, fill_value=None):
        warn('Categorical.take_nd is deprecated, use Categorical.take instead', FutureWarning, stacklevel=2)
        return self.take(indexer, allow_fill=allow_fill, fill_value=fill_value)

    def __iter__(self):
        '\n        Returns an Iterator over the values of this Categorical.\n        '
        return iter(self._internal_get_values().tolist())

    def __contains__(self, key):
        '\n        Returns True if `key` is in this Categorical.\n        '
        if is_valid_nat_for_dtype(key, self.categories.dtype):
            return self.isna().any()
        return contains(self, key, container=self._codes)

    def _formatter(self, boxed=False):
        return None

    def _tidy_repr(self, max_vals=10, footer=True):
        '\n        a short repr displaying only max_vals and an optional (but default\n        footer)\n        '
        num = (max_vals // 2)
        head = self[:num]._get_repr(length=False, footer=False)
        tail = self[(- (max_vals - num)):]._get_repr(length=False, footer=False)
        result = f'{head[:(- 1)]}, ..., {tail[1:]}'
        if footer:
            result = f'''{result}
{self._repr_footer()}'''
        return str(result)

    def _repr_categories(self):
        '\n        return the base repr for the categories\n        '
        max_categories = (10 if (get_option('display.max_categories') == 0) else get_option('display.max_categories'))
        from pandas.io.formats import format as fmt
        format_array = partial(fmt.format_array, formatter=None, quoting=QUOTE_NONNUMERIC)
        if (len(self.categories) > max_categories):
            num = (max_categories // 2)
            head = format_array(self.categories[:num])
            tail = format_array(self.categories[(- num):])
            category_strs = ((head + ['...']) + tail)
        else:
            category_strs = format_array(self.categories)
        category_strs = [x.strip() for x in category_strs]
        return category_strs

    def _repr_categories_info(self):
        '\n        Returns a string representation of the footer.\n        '
        category_strs = self._repr_categories()
        dtype = str(self.categories.dtype)
        levheader = f'Categories ({len(self.categories)}, {dtype}): '
        (width, height) = get_terminal_size()
        max_width = (get_option('display.width') or width)
        if console.in_ipython_frontend():
            max_width = 0
        levstring = ''
        start = True
        cur_col_len = len(levheader)
        (sep_len, sep) = ((3, ' < ') if self.ordered else (2, ', '))
        linesep = (sep.rstrip() + '\n')
        for val in category_strs:
            if ((max_width != 0) and (((cur_col_len + sep_len) + len(val)) > max_width)):
                levstring += (linesep + (' ' * (len(levheader) + 1)))
                cur_col_len = (len(levheader) + 1)
            elif (not start):
                levstring += sep
                cur_col_len += len(val)
            levstring += val
            start = False
        return (((levheader + '[') + levstring.replace(' < ... < ', ' ... ')) + ']')

    def _repr_footer(self):
        info = self._repr_categories_info()
        return f'''Length: {len(self)}
{info}'''

    def _get_repr(self, length=True, na_rep='NaN', footer=True):
        from pandas.io.formats import format as fmt
        formatter = fmt.CategoricalFormatter(self, length=length, na_rep=na_rep, footer=footer)
        result = formatter.to_string()
        return str(result)

    def __repr__(self):
        '\n        String representation.\n        '
        _maxlen = 10
        if (len(self._codes) > _maxlen):
            result = self._tidy_repr(_maxlen)
        elif (len(self._codes) > 0):
            result = self._get_repr(length=(len(self) > _maxlen))
        else:
            msg = self._get_repr(length=False, footer=True).replace('\n', ', ')
            result = f'[], {msg}'
        return result

    def __getitem__(self, key):
        '\n        Return an item.\n        '
        result = super().__getitem__(key)
        if (getattr(result, 'ndim', 0) > 1):
            result = result._ndarray
            deprecate_ndim_indexing(result)
        return result

    def _validate_setitem_value(self, value):
        value = extract_array(value, extract_numpy=True)
        if isinstance(value, Categorical):
            if (not is_dtype_equal(self.dtype, value.dtype)):
                raise ValueError('Cannot set a Categorical with another, without identical categories')
            value = self._encode_with_my_categories(value)
            return value._codes
        rvalue = (value if (not is_hashable(value)) else [value])
        from pandas import Index
        to_add = Index(rvalue).difference(self.categories)
        if (len(to_add) and (not isna(to_add).all())):
            raise ValueError('Cannot setitem on a Categorical with a new category, set the categories first')
        codes = self.categories.get_indexer(rvalue)
        return codes.astype(self._ndarray.dtype, copy=False)

    def _reverse_indexer(self):
        "\n        Compute the inverse of a categorical, returning\n        a dict of categories -> indexers.\n\n        *This is an internal function*\n\n        Returns\n        -------\n        dict of categories -> indexers\n\n        Examples\n        --------\n        >>> c = pd.Categorical(list('aabca'))\n        >>> c\n        ['a', 'a', 'b', 'c', 'a']\n        Categories (3, object): ['a', 'b', 'c']\n        >>> c.categories\n        Index(['a', 'b', 'c'], dtype='object')\n        >>> c.codes\n        array([0, 0, 1, 2, 0], dtype=int8)\n        >>> c._reverse_indexer()\n        {'a': array([0, 1, 4]), 'b': array([2]), 'c': array([3])}\n\n        "
        categories = self.categories
        (r, counts) = libalgos.groupsort_indexer(self.codes.astype('int64'), categories.size)
        counts = counts.cumsum()
        _result = (r[start:end] for (start, end) in zip(counts, counts[1:]))
        return dict(zip(categories, _result))

    @deprecate_kwarg(old_arg_name='numeric_only', new_arg_name='skipna')
    def min(self, *, skipna=True, **kwargs):
        '\n        The minimum value of the object.\n\n        Only ordered `Categoricals` have a minimum!\n\n        .. versionchanged:: 1.0.0\n\n           Returns an NA value on empty arrays\n\n        Raises\n        ------\n        TypeError\n            If the `Categorical` is not `ordered`.\n\n        Returns\n        -------\n        min : the minimum of this `Categorical`\n        '
        nv.validate_minmax_axis(kwargs.get('axis', 0))
        nv.validate_min((), kwargs)
        self.check_for_ordered('min')
        if (not len(self._codes)):
            return self.dtype.na_value
        good = (self._codes != (- 1))
        if (not good.all()):
            if (skipna and good.any()):
                pointer = self._codes[good].min()
            else:
                return np.nan
        else:
            pointer = self._codes.min()
        return self._wrap_reduction_result(None, pointer)

    @deprecate_kwarg(old_arg_name='numeric_only', new_arg_name='skipna')
    def max(self, *, skipna=True, **kwargs):
        '\n        The maximum value of the object.\n\n        Only ordered `Categoricals` have a maximum!\n\n        .. versionchanged:: 1.0.0\n\n           Returns an NA value on empty arrays\n\n        Raises\n        ------\n        TypeError\n            If the `Categorical` is not `ordered`.\n\n        Returns\n        -------\n        max : the maximum of this `Categorical`\n        '
        nv.validate_minmax_axis(kwargs.get('axis', 0))
        nv.validate_max((), kwargs)
        self.check_for_ordered('max')
        if (not len(self._codes)):
            return self.dtype.na_value
        good = (self._codes != (- 1))
        if (not good.all()):
            if (skipna and good.any()):
                pointer = self._codes[good].max()
            else:
                return np.nan
        else:
            pointer = self._codes.max()
        return self._wrap_reduction_result(None, pointer)

    def mode(self, dropna=True):
        "\n        Returns the mode(s) of the Categorical.\n\n        Always returns `Categorical` even if only one value.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't consider counts of NaN/NaT.\n\n            .. versionadded:: 0.24.0\n\n        Returns\n        -------\n        modes : `Categorical` (sorted)\n        "
        codes = self._codes
        if dropna:
            good = (self._codes != (- 1))
            codes = self._codes[good]
        codes = sorted(htable.mode_int64(ensure_int64(codes), dropna))
        return self._from_backing_data(codes)

    def unique(self):
        '\n        Return the ``Categorical`` which ``categories`` and ``codes`` are\n        unique. Unused categories are NOT returned.\n\n        - unordered category: values and categories are sorted by appearance\n          order.\n        - ordered category: values are sorted by appearance order, categories\n          keeps existing order.\n\n        Returns\n        -------\n        unique values : ``Categorical``\n\n        See Also\n        --------\n        pandas.unique\n        CategoricalIndex.unique\n        Series.unique : Return unique values of Series object.\n\n        Examples\n        --------\n        An unordered Categorical will return categories in the\n        order of appearance.\n\n        >>> pd.Categorical(list("baabc")).unique()\n        [\'b\', \'a\', \'c\']\n        Categories (3, object): [\'b\', \'a\', \'c\']\n\n        >>> pd.Categorical(list("baabc"), categories=list("abc")).unique()\n        [\'b\', \'a\', \'c\']\n        Categories (3, object): [\'b\', \'a\', \'c\']\n\n        An ordered Categorical preserves the category ordering.\n\n        >>> pd.Categorical(\n        ...     list("baabc"), categories=list("abc"), ordered=True\n        ... ).unique()\n        [\'b\', \'a\', \'c\']\n        Categories (3, object): [\'a\' < \'b\' < \'c\']\n        '
        unique_codes = unique1d(self.codes)
        cat = self.copy()
        cat._codes = unique_codes
        take_codes = unique_codes[(unique_codes != (- 1))]
        if self.ordered:
            take_codes = np.sort(take_codes)
        return cat.set_categories(cat.categories.take(take_codes))

    def _values_for_factorize(self):
        return (self._ndarray, (- 1))

    @classmethod
    def _from_factorized(cls, uniques, original):
        return original._constructor(original.categories.take(uniques), dtype=original.dtype)

    def equals(self, other):
        '\n        Returns True if categorical arrays are equal.\n\n        Parameters\n        ----------\n        other : `Categorical`\n\n        Returns\n        -------\n        bool\n        '
        if (not isinstance(other, Categorical)):
            return False
        elif self._categories_match_up_to_permutation(other):
            other = self._encode_with_my_categories(other)
            return np.array_equal(self._codes, other._codes)
        return False

    @classmethod
    def _concat_same_type(cls, to_concat, axis=0):
        from pandas.core.dtypes.concat import union_categoricals
        return union_categoricals(to_concat)

    def _encode_with_my_categories(self, other):
        "\n        Re-encode another categorical using this Categorical's categories.\n\n        Notes\n        -----\n        This assumes we have already checked\n        self._categories_match_up_to_permutation(other).\n        "
        codes = recode_for_categories(other.codes, other.categories, self.categories, copy=False)
        return self._from_backing_data(codes)

    def _categories_match_up_to_permutation(self, other):
        '\n        Returns True if categoricals are the same dtype\n          same categories, and same ordered\n\n        Parameters\n        ----------\n        other : Categorical\n\n        Returns\n        -------\n        bool\n        '
        return (hash(self.dtype) == hash(other.dtype))

    def is_dtype_equal(self, other):
        warn('Categorical.is_dtype_equal is deprecated and will be removed in a future version', FutureWarning, stacklevel=2)
        try:
            return self._categories_match_up_to_permutation(other)
        except (AttributeError, TypeError):
            return False

    def describe(self):
        '\n        Describes this Categorical\n\n        Returns\n        -------\n        description: `DataFrame`\n            A dataframe with frequency and counts by category.\n        '
        counts = self.value_counts(dropna=False)
        freqs = (counts / float(counts.sum()))
        from pandas.core.reshape.concat import concat
        result = concat([counts, freqs], axis=1)
        result.columns = ['counts', 'freqs']
        result.index.name = 'categories'
        return result

    def isin(self, values):
        "\n        Check whether `values` are contained in Categorical.\n\n        Return a boolean NumPy Array showing whether each element in\n        the Categorical matches an element in the passed sequence of\n        `values` exactly.\n\n        Parameters\n        ----------\n        values : set or list-like\n            The sequence of values to test. Passing in a single string will\n            raise a ``TypeError``. Instead, turn a single string into a\n            list of one element.\n\n        Returns\n        -------\n        isin : numpy.ndarray (bool dtype)\n\n        Raises\n        ------\n        TypeError\n          * If `values` is not a set or list-like\n\n        See Also\n        --------\n        pandas.Series.isin : Equivalent method on Series.\n\n        Examples\n        --------\n        >>> s = pd.Categorical(['lama', 'cow', 'lama', 'beetle', 'lama',\n        ...                'hippo'])\n        >>> s.isin(['cow', 'lama'])\n        array([ True,  True,  True, False,  True, False])\n\n        Passing a single string as ``s.isin('lama')`` will raise an error. Use\n        a list of one element instead:\n\n        >>> s.isin(['lama'])\n        array([ True, False,  True, False,  True, False])\n        "
        if (not is_list_like(values)):
            values_type = type(values).__name__
            raise TypeError(f'only list-like objects are allowed to be passed to isin(), you passed a [{values_type}]')
        values = sanitize_array(values, None, None)
        null_mask = np.asarray(isna(values))
        code_values = self.categories.get_indexer(values)
        code_values = code_values[(null_mask | (code_values >= 0))]
        return algorithms.isin(self.codes, code_values)

    def replace(self, to_replace, value, inplace=False):
        '\n        Replaces all instances of one value with another\n\n        Parameters\n        ----------\n        to_replace: object\n            The value to be replaced\n\n        value: object\n            The value to replace it with\n\n        inplace: bool\n            Whether the operation is done in-place\n\n        Returns\n        -------\n        None if inplace is True, otherwise the new Categorical after replacement\n\n\n        Examples\n        --------\n        >>> s = pd.Categorical([1, 2, 1, 3])\n        >>> s.replace(1, 3)\n        [3, 2, 3, 3]\n        Categories (2, int64): [2, 3]\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        cat = (self if inplace else self.copy())
        if is_list_like(to_replace):
            replace_dict = {replace_value: value for replace_value in to_replace}
        else:
            replace_dict = {to_replace: value}
        for (replace_value, new_value) in replace_dict.items():
            if (new_value == replace_value):
                continue
            if (replace_value in cat.categories):
                if isna(new_value):
                    cat.remove_categories(replace_value, inplace=True)
                    continue
                categories = cat.categories.tolist()
                index = categories.index(replace_value)
                if (new_value in cat.categories):
                    value_index = categories.index(new_value)
                    cat._codes[(cat._codes == index)] = value_index
                    cat.remove_categories(replace_value, inplace=True)
                else:
                    categories[index] = new_value
                    cat.rename_categories(categories, inplace=True)
        if (not inplace):
            return cat

    def _str_map(self, f, na_value=np.nan, dtype=np.dtype(object)):
        from pandas.core.arrays import PandasArray
        categories = self.categories
        codes = self.codes
        result = PandasArray(categories.to_numpy())._str_map(f, na_value, dtype)
        return take_1d(result, codes, fill_value=na_value)

    def _str_get_dummies(self, sep='|'):
        from pandas.core.arrays import PandasArray
        return PandasArray(self.astype(str))._str_get_dummies(sep)

@delegate_names(delegate=Categorical, accessors=['categories', 'ordered'], typ='property')
@delegate_names(delegate=Categorical, accessors=['rename_categories', 'reorder_categories', 'add_categories', 'remove_categories', 'remove_unused_categories', 'set_categories', 'as_ordered', 'as_unordered'], typ='method')
class CategoricalAccessor(PandasDelegate, PandasObject, NoNewAttributesMixin):
    '\n    Accessor object for categorical properties of the Series values.\n\n    Be aware that assigning to `categories` is a inplace operation, while all\n    methods return new categorical data per default (but can be called with\n    `inplace=True`).\n\n    Parameters\n    ----------\n    data : Series or CategoricalIndex\n\n    Examples\n    --------\n    >>> s = pd.Series(list("abbccc")).astype("category")\n    >>> s\n    0    a\n    1    b\n    2    b\n    3    c\n    4    c\n    5    c\n    dtype: category\n    Categories (3, object): [\'a\', \'b\', \'c\']\n\n    >>> s.cat.categories\n    Index([\'a\', \'b\', \'c\'], dtype=\'object\')\n\n    >>> s.cat.rename_categories(list("cba"))\n    0    c\n    1    b\n    2    b\n    3    a\n    4    a\n    5    a\n    dtype: category\n    Categories (3, object): [\'c\', \'b\', \'a\']\n\n    >>> s.cat.reorder_categories(list("cba"))\n    0    a\n    1    b\n    2    b\n    3    c\n    4    c\n    5    c\n    dtype: category\n    Categories (3, object): [\'c\', \'b\', \'a\']\n\n    >>> s.cat.add_categories(["d", "e"])\n    0    a\n    1    b\n    2    b\n    3    c\n    4    c\n    5    c\n    dtype: category\n    Categories (5, object): [\'a\', \'b\', \'c\', \'d\', \'e\']\n\n    >>> s.cat.remove_categories(["a", "c"])\n    0    NaN\n    1      b\n    2      b\n    3    NaN\n    4    NaN\n    5    NaN\n    dtype: category\n    Categories (1, object): [\'b\']\n\n    >>> s1 = s.cat.add_categories(["d", "e"])\n    >>> s1.cat.remove_unused_categories()\n    0    a\n    1    b\n    2    b\n    3    c\n    4    c\n    5    c\n    dtype: category\n    Categories (3, object): [\'a\', \'b\', \'c\']\n\n    >>> s.cat.set_categories(list("abcde"))\n    0    a\n    1    b\n    2    b\n    3    c\n    4    c\n    5    c\n    dtype: category\n    Categories (5, object): [\'a\', \'b\', \'c\', \'d\', \'e\']\n\n    >>> s.cat.as_ordered()\n    0    a\n    1    b\n    2    b\n    3    c\n    4    c\n    5    c\n    dtype: category\n    Categories (3, object): [\'a\' < \'b\' < \'c\']\n\n    >>> s.cat.as_unordered()\n    0    a\n    1    b\n    2    b\n    3    c\n    4    c\n    5    c\n    dtype: category\n    Categories (3, object): [\'a\', \'b\', \'c\']\n    '

    def __init__(self, data):
        self._validate(data)
        self._parent = data.values
        self._index = data.index
        self._name = data.name
        self._freeze()

    @staticmethod
    def _validate(data):
        if (not is_categorical_dtype(data.dtype)):
            raise AttributeError("Can only use .cat accessor with a 'category' dtype")

    def _delegate_property_get(self, name):
        return getattr(self._parent, name)

    def _delegate_property_set(self, name, new_values):
        return setattr(self._parent, name, new_values)

    @property
    def codes(self):
        '\n        Return Series of codes as well as the index.\n        '
        from pandas import Series
        return Series(self._parent.codes, index=self._index)

    def _delegate_method(self, name, *args, **kwargs):
        from pandas import Series
        method = getattr(self._parent, name)
        res = method(*args, **kwargs)
        if (res is not None):
            return Series(res, index=self._index, name=self._name)

def _get_codes_for_values(values, categories):
    '\n    utility routine to turn values into codes given the specified categories\n\n    If `values` is known to be a Categorical, use recode_for_categories instead.\n    '
    dtype_equal = is_dtype_equal(values.dtype, categories.dtype)
    if (is_extension_array_dtype(categories.dtype) and is_object_dtype(values)):
        cls = categories.dtype.construct_array_type()
        values = maybe_cast_to_extension_array(cls, values)
        if (not isinstance(values, cls)):
            values = ensure_object(values)
            categories = ensure_object(categories)
    elif (not dtype_equal):
        values = ensure_object(values)
        categories = ensure_object(categories)
    if isinstance(categories, ABCIndex):
        return coerce_indexer_dtype(categories.get_indexer_for(values), categories)
    (hash_klass, vals) = get_data_algo(values)
    (_, cats) = get_data_algo(categories)
    t = hash_klass(len(cats))
    t.map_locations(cats)
    return coerce_indexer_dtype(t.lookup(vals), cats)

def recode_for_categories(codes, old_categories, new_categories, copy=True):
    "\n    Convert a set of codes for to a new set of categories\n\n    Parameters\n    ----------\n    codes : np.ndarray\n    old_categories, new_categories : Index\n    copy: bool, default True\n        Whether to copy if the codes are unchanged.\n\n    Returns\n    -------\n    new_codes : np.ndarray[np.int64]\n\n    Examples\n    --------\n    >>> old_cat = pd.Index(['b', 'a', 'c'])\n    >>> new_cat = pd.Index(['a', 'b'])\n    >>> codes = np.array([0, 1, 1, 2])\n    >>> recode_for_categories(codes, old_cat, new_cat)\n    array([ 1,  0,  0, -1], dtype=int8)\n    "
    if (len(old_categories) == 0):
        if copy:
            return codes.copy()
        return codes
    elif new_categories.equals(old_categories):
        if copy:
            return codes.copy()
        return codes
    indexer = coerce_indexer_dtype(new_categories.get_indexer(old_categories), new_categories)
    new_codes = take_1d(indexer, codes, fill_value=(- 1))
    return new_codes

def factorize_from_iterable(values):
    '\n    Factorize an input `values` into `categories` and `codes`. Preserves\n    categorical dtype in `categories`.\n\n    *This is an internal function*\n\n    Parameters\n    ----------\n    values : list-like\n\n    Returns\n    -------\n    codes : ndarray\n    categories : Index\n        If `values` has a categorical dtype, then `categories` is\n        a CategoricalIndex keeping the categories and order of `values`.\n    '
    if (not is_list_like(values)):
        raise TypeError('Input must be list-like')
    if is_categorical_dtype(values):
        values = extract_array(values)
        cat_codes = np.arange(len(values.categories), dtype=values.codes.dtype)
        categories = Categorical.from_codes(cat_codes, dtype=values.dtype)
        codes = values.codes
    else:
        cat = Categorical(values, ordered=False)
        categories = cat.categories
        codes = cat.codes
    return (codes, categories)

def factorize_from_iterables(iterables):
    '\n    A higher-level wrapper over `factorize_from_iterable`.\n\n    *This is an internal function*\n\n    Parameters\n    ----------\n    iterables : list-like of list-likes\n\n    Returns\n    -------\n    codes_list : list of ndarrays\n    categories_list : list of Indexes\n\n    Notes\n    -----\n    See `factorize_from_iterable` for more info.\n    '
    if (len(iterables) == 0):
        return [[], []]
    return map(list, zip(*(factorize_from_iterable(it) for it in iterables)))
