
from typing import Any, List, Optional
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import index as libindex
from pandas._libs.lib import no_default
from pandas._typing import ArrayLike, Label
from pandas.util._decorators import Appender, doc
from pandas.core.dtypes.common import ensure_platform_int, is_categorical_dtype, is_scalar
from pandas.core.dtypes.missing import is_valid_nat_for_dtype, isna, notna
from pandas.core import accessor
from pandas.core.arrays.categorical import Categorical, contains
from pandas.core.construction import extract_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, _index_shared_docs, maybe_extract_name
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex, inherit_names
_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({'target_klass': 'CategoricalIndex'})

@inherit_names(['argsort', '_internal_get_values', 'tolist', 'codes', 'categories', 'ordered', '_reverse_indexer', 'searchsorted', 'is_dtype_equal', 'min', 'max'], Categorical)
@accessor.delegate_names(delegate=Categorical, accessors=['rename_categories', 'reorder_categories', 'add_categories', 'remove_categories', 'remove_unused_categories', 'set_categories', 'as_ordered', 'as_unordered'], typ='method', overwrite=True)
class CategoricalIndex(NDArrayBackedExtensionIndex, accessor.PandasDelegate):
    '\n    Index based on an underlying :class:`Categorical`.\n\n    CategoricalIndex, like Categorical, can only take on a limited,\n    and usually fixed, number of possible values (`categories`). Also,\n    like Categorical, it might have an order, but numerical operations\n    (additions, divisions, ...) are not possible.\n\n    Parameters\n    ----------\n    data : array-like (1-dimensional)\n        The values of the categorical. If `categories` are given, values not in\n        `categories` will be replaced with NaN.\n    categories : index-like, optional\n        The categories for the categorical. Items need to be unique.\n        If the categories are not given here (and also not in `dtype`), they\n        will be inferred from the `data`.\n    ordered : bool, optional\n        Whether or not this categorical is treated as an ordered\n        categorical. If not given here or in `dtype`, the resulting\n        categorical will be unordered.\n    dtype : CategoricalDtype or "category", optional\n        If :class:`CategoricalDtype`, cannot be used together with\n        `categories` or `ordered`.\n    copy : bool, default False\n        Make a copy of input ndarray.\n    name : object, optional\n        Name to be stored in the index.\n\n    Attributes\n    ----------\n    codes\n    categories\n    ordered\n\n    Methods\n    -------\n    rename_categories\n    reorder_categories\n    add_categories\n    remove_categories\n    remove_unused_categories\n    set_categories\n    as_ordered\n    as_unordered\n    map\n\n    Raises\n    ------\n    ValueError\n        If the categories do not validate.\n    TypeError\n        If an explicit ``ordered=True`` is given but no `categories` and the\n        `values` are not sortable.\n\n    See Also\n    --------\n    Index : The base pandas Index type.\n    Categorical : A categorical array.\n    CategoricalDtype : Type for categorical data.\n\n    Notes\n    -----\n    See the `user guide\n    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#categoricalindex>`_\n    for more.\n\n    Examples\n    --------\n    >>> pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])\n    CategoricalIndex([\'a\', \'b\', \'c\', \'a\', \'b\', \'c\'],\n                     categories=[\'a\', \'b\', \'c\'], ordered=False, dtype=\'category\')\n\n    ``CategoricalIndex`` can also be instantiated from a ``Categorical``:\n\n    >>> c = pd.Categorical(["a", "b", "c", "a", "b", "c"])\n    >>> pd.CategoricalIndex(c)\n    CategoricalIndex([\'a\', \'b\', \'c\', \'a\', \'b\', \'c\'],\n                     categories=[\'a\', \'b\', \'c\'], ordered=False, dtype=\'category\')\n\n    Ordered ``CategoricalIndex`` can have a min and max value.\n\n    >>> ci = pd.CategoricalIndex(\n    ...     ["a", "b", "c", "a", "b", "c"], ordered=True, categories=["c", "b", "a"]\n    ... )\n    >>> ci\n    CategoricalIndex([\'a\', \'b\', \'c\', \'a\', \'b\', \'c\'],\n                     categories=[\'c\', \'b\', \'a\'], ordered=True, dtype=\'category\')\n    >>> ci.min()\n    \'c\'\n    '
    _typ = 'categoricalindex'

    @property
    def _can_hold_strings(self):
        return self.categories._can_hold_strings

    @property
    def _engine_type(self):
        return {np.int8: libindex.Int8Engine, np.int16: libindex.Int16Engine, np.int32: libindex.Int32Engine, np.int64: libindex.Int64Engine}[self.codes.dtype.type]
    _attributes = ['name']

    def __new__(cls, data=None, categories=None, ordered=None, dtype=None, copy=False, name=None):
        name = maybe_extract_name(name, data, cls)
        if is_scalar(data):
            raise cls._scalar_data_error(data)
        data = Categorical(data, categories=categories, ordered=ordered, dtype=dtype, copy=copy)
        return cls._simple_new(data, name=name)

    @classmethod
    def _simple_new(cls, values, name=None):
        assert isinstance(values, Categorical), type(values)
        result = object.__new__(cls)
        result._data = values
        result.name = name
        result._cache = {}
        result._reset_identity()
        return result

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values=None, name=no_default):
        name = (self.name if (name is no_default) else name)
        if (values is not None):
            values = Categorical(values, dtype=self.dtype)
        return super()._shallow_copy(values=values, name=name)

    def _is_dtype_compat(self, other):
        '\n        *this is an internal non-public method*\n\n        provide a comparison between the dtype of self and other (coercing if\n        needed)\n\n        Parameters\n        ----------\n        other : Index\n\n        Returns\n        -------\n        Categorical\n\n        Raises\n        ------\n        TypeError if the dtypes are not compatible\n        '
        if is_categorical_dtype(other):
            other = extract_array(other)
            if (not other._categories_match_up_to_permutation(self)):
                raise TypeError('categories must match existing categories when appending')
        else:
            values = other
            cat = Categorical(other, dtype=self.dtype)
            other = CategoricalIndex(cat)
            if (not other.isin(values).all()):
                raise TypeError('cannot append a non-category item to a CategoricalIndex')
            other = other._values
            if (not ((other == values) | (isna(other) & isna(values))).all()):
                raise TypeError('categories must match existing categories when appending')
        return other

    def equals(self, other):
        '\n        Determine if two CategoricalIndex objects contain the same elements.\n\n        Returns\n        -------\n        bool\n            If two CategoricalIndex objects have equal elements True,\n            otherwise False.\n        '
        if self.is_(other):
            return True
        if (not isinstance(other, Index)):
            return False
        try:
            other = self._is_dtype_compat(other)
        except (TypeError, ValueError):
            return False
        return self._data.equals(other)

    @property
    def _formatter_func(self):
        return self.categories._formatter_func

    def _format_attrs(self):
        '\n        Return a list of tuples of the (attr,formatted_value)\n        '
        max_categories = (10 if (get_option('display.max_categories') == 0) else get_option('display.max_categories'))
        attrs = [('categories', ibase.default_pprint(self.categories, max_seq_items=max_categories)), ('ordered', self.ordered)]
        if (self.name is not None):
            attrs.append(('name', ibase.default_pprint(self.name)))
        attrs.append(('dtype', f"'{self.dtype.name}'"))
        max_seq_items = (get_option('display.max_seq_items') or len(self))
        if (len(self) > max_seq_items):
            attrs.append(('length', len(self)))
        return attrs

    def _format_with_header(self, header, na_rep='NaN'):
        from pandas.io.formats.printing import pprint_thing
        result = [(pprint_thing(x, escape_chars=('\t', '\r', '\n')) if notna(x) else na_rep) for x in self._values]
        return (header + result)

    @property
    def inferred_type(self):
        return 'categorical'

    @doc(Index.__contains__)
    def __contains__(self, key):
        if is_valid_nat_for_dtype(key, self.categories.dtype):
            return self.hasnans
        return contains(self, key, container=self._engine)

    @doc(Index.fillna)
    def fillna(self, value, downcast=None):
        value = self._require_scalar(value)
        cat = self._data.fillna(value)
        return type(self)._simple_new(cat, name=self.name)

    @doc(Index.unique)
    def unique(self, level=None):
        if (level is not None):
            self._validate_index_level(level)
        result = self._values.unique()
        return type(self)._simple_new(result, name=self.name)

    def reindex(self, target, method=None, level=None, limit=None, tolerance=None):
        "\n        Create index with target's values (move/add/delete values as necessary)\n\n        Returns\n        -------\n        new_index : pd.Index\n            Resulting index\n        indexer : np.ndarray or None\n            Indices of output values in original index\n\n        "
        if (method is not None):
            raise NotImplementedError('argument method is not implemented for CategoricalIndex.reindex')
        if (level is not None):
            raise NotImplementedError('argument level is not implemented for CategoricalIndex.reindex')
        if (limit is not None):
            raise NotImplementedError('argument limit is not implemented for CategoricalIndex.reindex')
        target = ibase.ensure_index(target)
        missing: List[int]
        if self.equals(target):
            indexer = None
            missing = []
        else:
            (indexer, missing) = self.get_indexer_non_unique(np.array(target))
        if (len(self.codes) and (indexer is not None)):
            new_target = self.take(indexer)
        else:
            new_target = target
        if len(missing):
            cats = self.categories.get_indexer(target)
            if ((not isinstance(cats, CategoricalIndex)) or (cats == (- 1)).any()):
                result = Index(np.array(self), name=self.name)
                (new_target, indexer, _) = result._reindex_non_unique(np.array(target))
            else:
                codes = new_target.codes.copy()
                codes[(indexer == (- 1))] = cats[missing]
                cat = self._data._from_backing_data(codes)
                new_target = type(self)._simple_new(cat, name=self.name)
        new_target = np.asarray(new_target)
        if is_categorical_dtype(target):
            new_target = Categorical(new_target, dtype=target.dtype)
            new_target = type(self)._simple_new(new_target, name=self.name)
        else:
            new_target = Index(new_target, name=self.name)
        return (new_target, indexer)

    def _reindex_non_unique(self, target):
        "\n        reindex from a non-unique; which CategoricalIndex's are almost\n        always\n        "
        (new_target, indexer) = self.reindex(target)
        new_indexer = None
        check = (indexer == (- 1))
        if check.any():
            new_indexer = np.arange(len(self.take(indexer)))
            new_indexer[check] = (- 1)
        cats = self.categories.get_indexer(target)
        if (not (cats == (- 1)).any()):
            new_target = Categorical(new_target, dtype=self.dtype)
            new_target = type(self)._simple_new(new_target, name=self.name)
        return (new_target, indexer, new_indexer)

    def _maybe_cast_indexer(self, key):
        return self._data._unbox_scalar(key)

    def _get_indexer(self, target, method=None, limit=None, tolerance=None):
        if self.equals(target):
            return np.arange(len(self), dtype='intp')
        return self._get_indexer_non_unique(target._values)[0]

    @Appender((_index_shared_docs['get_indexer_non_unique'] % _index_doc_kwargs))
    def get_indexer_non_unique(self, target):
        target = ibase.ensure_index(target)
        return self._get_indexer_non_unique(target._values)

    def _get_indexer_non_unique(self, values):
        '\n        get_indexer_non_unique but after unrapping the target Index object.\n        '
        if isinstance(values, Categorical):
            cat = self._data._encode_with_my_categories(values)
            codes = cat._codes
        else:
            codes = self.categories.get_indexer(values)
        (indexer, missing) = self._engine.get_indexer_non_unique(codes)
        return (ensure_platform_int(indexer), missing)

    @doc(Index._convert_list_indexer)
    def _convert_list_indexer(self, keyarr):
        if self.categories._defer_to_indexing:
            indexer = self.categories._convert_list_indexer(keyarr)
            return Index(self.codes).get_indexer_for(indexer)
        return self.get_indexer_for(keyarr)

    @doc(Index._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side, kind):
        if (kind == 'loc'):
            return label
        return super()._maybe_cast_slice_bound(label, side, kind)

    def _is_comparable_dtype(self, dtype):
        return self.categories._is_comparable_dtype(dtype)

    def take_nd(self, *args, **kwargs):
        'Alias for `take`'
        warnings.warn('CategoricalIndex.take_nd is deprecated, use CategoricalIndex.take instead', FutureWarning, stacklevel=2)
        return self.take(*args, **kwargs)

    def map(self, mapper):
        "\n        Map values using input correspondence (a dict, Series, or function).\n\n        Maps the values (their categories, not the codes) of the index to new\n        categories. If the mapping correspondence is one-to-one the result is a\n        :class:`~pandas.CategoricalIndex` which has the same order property as\n        the original, otherwise an :class:`~pandas.Index` is returned.\n\n        If a `dict` or :class:`~pandas.Series` is used any unmapped category is\n        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`\n        will be returned.\n\n        Parameters\n        ----------\n        mapper : function, dict, or Series\n            Mapping correspondence.\n\n        Returns\n        -------\n        pandas.CategoricalIndex or pandas.Index\n            Mapped index.\n\n        See Also\n        --------\n        Index.map : Apply a mapping correspondence on an\n            :class:`~pandas.Index`.\n        Series.map : Apply a mapping correspondence on a\n            :class:`~pandas.Series`.\n        Series.apply : Apply more complex functions on a\n            :class:`~pandas.Series`.\n\n        Examples\n        --------\n        >>> idx = pd.CategoricalIndex(['a', 'b', 'c'])\n        >>> idx\n        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],\n                          ordered=False, dtype='category')\n        >>> idx.map(lambda x: x.upper())\n        CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'],\n                         ordered=False, dtype='category')\n        >>> idx.map({'a': 'first', 'b': 'second', 'c': 'third'})\n        CategoricalIndex(['first', 'second', 'third'], categories=['first',\n                         'second', 'third'], ordered=False, dtype='category')\n\n        If the mapping is one-to-one the ordering of the categories is\n        preserved:\n\n        >>> idx = pd.CategoricalIndex(['a', 'b', 'c'], ordered=True)\n        >>> idx\n        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],\n                         ordered=True, dtype='category')\n        >>> idx.map({'a': 3, 'b': 2, 'c': 1})\n        CategoricalIndex([3, 2, 1], categories=[3, 2, 1], ordered=True,\n                         dtype='category')\n\n        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:\n\n        >>> idx.map({'a': 'first', 'b': 'second', 'c': 'first'})\n        Index(['first', 'second', 'first'], dtype='object')\n\n        If a `dict` is used, all unmapped categories are mapped to `NaN` and\n        the result is an :class:`~pandas.Index`:\n\n        >>> idx.map({'a': 'first', 'b': 'second'})\n        Index(['first', 'second', nan], dtype='object')\n        "
        mapped = self._values.map(mapper)
        return Index(mapped, name=self.name)

    def _concat(self, to_concat, name):
        try:
            codes = np.concatenate([self._is_dtype_compat(c).codes for c in to_concat])
        except TypeError:
            from pandas.core.dtypes.concat import concat_compat
            res = concat_compat(to_concat)
            return Index(res, name=name)
        else:
            cat = self._data._from_backing_data(codes)
            return type(self)._simple_new(cat, name=name)

    def _delegate_method(self, name, *args, **kwargs):
        ' method delegation to the ._values '
        method = getattr(self._values, name)
        if ('inplace' in kwargs):
            raise ValueError('cannot use inplace with CategoricalIndex')
        res = method(*args, **kwargs)
        if is_scalar(res):
            return res
        return CategoricalIndex(res, name=self.name)
