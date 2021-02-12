
import inspect
import re
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union, cast
import numpy as np
from pandas._libs import Interval, Period, Timestamp, algos as libalgos, internals as libinternals, lib, writers
from pandas._libs.internals import BlockPlacement
from pandas._libs.tslibs import conversion
from pandas._typing import ArrayLike, DtypeObj, Scalar, Shape
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import astype_dt64_to_dt64tz, astype_nansafe, convert_scalar_for_putitemlike, find_common_type, infer_dtype_from, infer_dtype_from_scalar, maybe_downcast_numeric, maybe_downcast_to_dtype, maybe_infer_dtype_type, maybe_promote, maybe_upcast, soft_convert_objects
from pandas.core.dtypes.common import DT64NS_DTYPE, TD64NS_DTYPE, is_categorical_dtype, is_datetime64_dtype, is_datetime64tz_dtype, is_dtype_equal, is_extension_array_dtype, is_float, is_integer, is_list_like, is_object_dtype, is_re, is_re_compilable, is_sparse, pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCPandasArray, ABCSeries
from pandas.core.dtypes.missing import is_valid_nat_for_dtype, isna
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import putmask_inplace, putmask_smart, putmask_without_repeat
from pandas.core.array_algos.replace import compare_or_regex_search, replace_regex
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import Categorical, DatetimeArray, ExtensionArray, PandasArray, PandasDtype, TimedeltaArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexers import check_setitem_lengths, is_empty_indexer, is_scalar_indexer
import pandas.core.missing as missing
from pandas.core.nanops import nanpercentile
if TYPE_CHECKING:
    from pandas import Index
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

class Block(PandasObject):
    '\n    Canonical n-dimensional unit of homogeneous dtype contained in a pandas\n    data structure\n\n    Index-ignorant; let the container take care of that\n    '
    __slots__ = ['_mgr_locs', 'values', 'ndim']
    is_numeric = False
    is_float = False
    is_integer = False
    is_complex = False
    is_datetime = False
    is_datetimetz = False
    is_timedelta = False
    is_bool = False
    is_object = False
    is_extension = False
    _can_hold_na = False
    _can_consolidate = True
    _validate_ndim = True

    @classmethod
    def _simple_new(cls, values, placement, ndim):
        '\n        Fastpath constructor, does *no* validation\n        '
        obj = object.__new__(cls)
        obj.ndim = ndim
        obj.values = values
        obj._mgr_locs = placement
        return obj

    def __init__(self, values, placement, ndim):
        '\n        Parameters\n        ----------\n        values : np.ndarray or ExtensionArray\n        placement : BlockPlacement (or castable)\n        ndim : int\n            1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame\n        '
        self.ndim = self._check_ndim(values, ndim)
        self.mgr_locs = placement
        self.values = self._maybe_coerce_values(values)
        if (self._validate_ndim and self.ndim and (len(self.mgr_locs) != len(self.values))):
            raise ValueError(f'Wrong number of items passed {len(self.values)}, placement implies {len(self.mgr_locs)}')

    def _maybe_coerce_values(self, values):
        '\n        Ensure we have correctly-typed values.\n\n        Parameters\n        ----------\n        values : np.ndarray, ExtensionArray, Index\n\n        Returns\n        -------\n        np.ndarray or ExtensionArray\n        '
        return values

    def _check_ndim(self, values, ndim):
        "\n        ndim inference and validation.\n\n        Infers ndim from 'values' if not provided to __init__.\n        Validates that values.ndim and ndim are consistent if and only if\n        the class variable '_validate_ndim' is True.\n\n        Parameters\n        ----------\n        values : array-like\n        ndim : int or None\n\n        Returns\n        -------\n        ndim : int\n\n        Raises\n        ------\n        ValueError : the number of dimensions do not match\n        "
        if (ndim is None):
            ndim = values.ndim
        if (self._validate_ndim and (values.ndim != ndim)):
            raise ValueError(f'Wrong number of dimensions. values.ndim != ndim [{values.ndim} != {ndim}]')
        return ndim

    @property
    def _holder(self):
        "\n        The array-like that can hold the underlying values.\n\n        None for 'Block', overridden by subclasses that don't\n        use an ndarray.\n        "
        return None

    @property
    def _consolidate_key(self):
        return (self._can_consolidate, self.dtype.name)

    @property
    def is_view(self):
        ' return a boolean if I am possibly a view '
        values = self.values
        values = cast(np.ndarray, values)
        return (values.base is not None)

    @property
    def is_categorical(self):
        return (self._holder is Categorical)

    @property
    def is_datelike(self):
        ' return True if I am a non-datelike '
        return (self.is_datetime or self.is_timedelta)

    def external_values(self):
        '\n        The array that Series.values returns (public attribute).\n\n        This has some historical constraints, and is overridden in block\n        subclasses to return the correct array (e.g. period returns\n        object ndarray and datetimetz a datetime64[ns] ndarray instead of\n        proper extension array).\n        '
        return self.values

    def internal_values(self):
        '\n        The array that Series._values returns (internal values).\n        '
        return self.values

    def array_values(self):
        '\n        The array that Series.array returns. Always an ExtensionArray.\n        '
        return PandasArray(self.values)

    def get_values(self, dtype=None):
        '\n        return an internal format, currently just the ndarray\n        this is often overridden to handle to_dense like operations\n        '
        if is_object_dtype(dtype):
            return self.values.astype(object)
        return self.values

    def get_block_values_for_json(self):
        '\n        This is used in the JSON C code.\n        '
        return np.asarray(self.values).reshape(self.shape)

    @property
    def fill_value(self):
        return np.nan

    @property
    def mgr_locs(self):
        return self._mgr_locs

    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs):
        if (not isinstance(new_mgr_locs, libinternals.BlockPlacement)):
            new_mgr_locs = libinternals.BlockPlacement(new_mgr_locs)
        self._mgr_locs = new_mgr_locs

    def make_block(self, values, placement=None):
        '\n        Create a new block, with type inference propagate any values that are\n        not specified\n        '
        if (placement is None):
            placement = self.mgr_locs
        if self.is_extension:
            values = _block_shape(values, ndim=self.ndim)
        return make_block(values, placement=placement, ndim=self.ndim)

    def make_block_same_class(self, values, placement=None, ndim=None):
        ' Wrap given values in a block of same type as self. '
        if (placement is None):
            placement = self.mgr_locs
        if (ndim is None):
            ndim = self.ndim
        return type(self)(values, placement=placement, ndim=ndim)

    def __repr__(self):
        name = type(self).__name__
        if (self.ndim == 1):
            result = f'{name}: {len(self)} dtype: {self.dtype}'
        else:
            shape = ' x '.join((str(s) for s in self.shape))
            result = f'{name}: {self.mgr_locs.indexer}, {shape}, dtype: {self.dtype}'
        return result

    def __len__(self):
        return len(self.values)

    def __getstate__(self):
        return (self.mgr_locs.indexer, self.values)

    def __setstate__(self, state):
        self.mgr_locs = libinternals.BlockPlacement(state[0])
        self.values = state[1]
        self.ndim = self.values.ndim

    def _slice(self, slicer):
        ' return a slice of my values '
        return self.values[slicer]

    def getitem_block(self, slicer, new_mgr_locs=None):
        '\n        Perform __getitem__-like, return result as block.\n\n        As of now, only supports slices that preserve dimensionality.\n        '
        if (new_mgr_locs is None):
            axis0_slicer = (slicer[0] if isinstance(slicer, tuple) else slicer)
            new_mgr_locs = self.mgr_locs[axis0_slicer]
        elif (not isinstance(new_mgr_locs, BlockPlacement)):
            new_mgr_locs = BlockPlacement(new_mgr_locs)
        new_values = self._slice(slicer)
        if (self._validate_ndim and (new_values.ndim != self.ndim)):
            raise ValueError('Only same dim slicing is allowed')
        return type(self)._simple_new(new_values, new_mgr_locs, self.ndim)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def iget(self, i):
        return self.values[i]

    def set_inplace(self, locs, values):
        '\n        Modify block values in-place with new item value.\n\n        Notes\n        -----\n        `set` never creates a new array or new Block, whereas `setitem` _may_\n        create a new array and always creates a new Block.\n        '
        self.values[locs] = values

    def delete(self, loc):
        '\n        Delete given loc(-s) from block in-place.\n        '
        self.values = np.delete(self.values, loc, 0)
        self.mgr_locs = self.mgr_locs.delete(loc)

    def apply(self, func, **kwargs):
        '\n        apply the function to my values; return a block if we are not\n        one\n        '
        with np.errstate(all='ignore'):
            result = func(self.values, **kwargs)
        return self._split_op_result(result)

    def reduce(self, func, ignore_failures=False):
        assert (self.ndim == 2)
        try:
            result = func(self.values)
        except (TypeError, NotImplementedError):
            if ignore_failures:
                return []
            raise
        if (np.ndim(result) == 0):
            res_values = np.array([[result]])
        else:
            res_values = result.reshape((- 1), 1)
        nb = self.make_block(res_values)
        return [nb]

    def _split_op_result(self, result):
        if (is_extension_array_dtype(result) and (result.ndim > 1)):
            nbs = []
            for (i, loc) in enumerate(self.mgr_locs):
                vals = result[i]
                block = self.make_block(values=vals, placement=[loc])
                nbs.append(block)
            return nbs
        if (not isinstance(result, Block)):
            result = self.make_block(result)
        return [result]

    def fillna(self, value, limit=None, inplace=False, downcast=None):
        '\n        fillna on the block with the value. If we fail, then convert to\n        ObjectBlock and try again\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        mask = isna(self.values)
        mask = _extract_bool_array(mask)
        if (limit is not None):
            limit = libalgos.validate_limit(None, limit=limit)
            mask[(mask.cumsum((self.ndim - 1)) > limit)] = False
        if (not self._can_hold_na):
            if inplace:
                return [self]
            else:
                return [self.copy()]
        if self._can_hold_element(value):
            nb = (self if inplace else self.copy())
            putmask_inplace(nb.values, mask, value)
            return self._maybe_downcast([nb], downcast)
        if (not mask.any()):
            return ([self] if inplace else [self.copy()])

        def f(mask, val, idx):
            block = self.coerce_to_target_dtype(value)
            if (idx is not None):
                block = block.getitem_block(slice(idx, (idx + 1)))
            return block.fillna(value, limit=limit, inplace=inplace, downcast=None)
        return self.split_and_operate(None, f, inplace)

    def _split(self):
        '\n        Split a block into a list of single-column blocks.\n        '
        assert (self.ndim == 2)
        new_blocks = []
        for (i, ref_loc) in enumerate(self.mgr_locs):
            vals = self.values[slice(i, (i + 1))]
            nb = self.make_block(vals, [ref_loc])
            new_blocks.append(nb)
        return new_blocks

    def split_and_operate(self, mask, f, inplace, ignore_failures=False):
        '\n        split the block per-column, and apply the callable f\n        per-column, return a new block for each. Handle\n        masking which will not change a block unless needed.\n\n        Parameters\n        ----------\n        mask : 2-d boolean mask\n        f : callable accepting (1d-mask, 1d values, indexer)\n        inplace : bool\n        ignore_failures : bool, default False\n\n        Returns\n        -------\n        list of blocks\n        '
        if (mask is None):
            mask = np.broadcast_to(True, shape=self.shape)
        new_values = self.values

        def make_a_block(nv, ref_loc):
            if isinstance(nv, list):
                assert (len(nv) == 1), nv
                assert isinstance(nv[0], Block)
                block = nv[0]
            else:
                nv = _block_shape(nv, ndim=self.ndim)
                block = self.make_block(values=nv, placement=ref_loc)
            return block
        if (self.ndim == 1):
            if mask.any():
                nv = f(mask, new_values, None)
            else:
                nv = (new_values if inplace else new_values.copy())
            block = make_a_block(nv, self.mgr_locs)
            return [block]
        new_blocks = []
        for (i, ref_loc) in enumerate(self.mgr_locs):
            m = mask[i]
            v = new_values[i]
            if (m.any() or (m.size == 0)):
                try:
                    nv = f(m, v, i)
                except TypeError:
                    if ignore_failures:
                        continue
                    else:
                        raise
            else:
                nv = (v if inplace else v.copy())
            block = make_a_block(nv, [ref_loc])
            new_blocks.append(block)
        return new_blocks

    def _maybe_downcast(self, blocks, downcast=None):
        if ((downcast is None) and (self.is_float or self.is_datelike)):
            return blocks
        return extend_blocks([b.downcast(downcast) for b in blocks])

    def downcast(self, dtypes=None):
        ' try to downcast each item to the dict of dtypes if present '
        if (dtypes is False):
            return [self]
        values = self.values
        if (self.ndim == 1):
            if (dtypes is None):
                dtypes = 'infer'
            nv = maybe_downcast_to_dtype(values, dtypes)
            return [self.make_block(nv)]
        if (dtypes is None):
            return [self]
        if (not ((dtypes == 'infer') or isinstance(dtypes, dict))):
            raise ValueError("downcast must have a dictionary or 'infer' as its argument")
        elif (dtypes != 'infer'):
            raise AssertionError('dtypes as dict is not supported yet')

        def f(mask, val, idx):
            val = maybe_downcast_to_dtype(val, dtype='infer')
            return val
        return self.split_and_operate(None, f, False)

    def astype(self, dtype, copy=False, errors='raise'):
        "\n        Coerce to the new dtype.\n\n        Parameters\n        ----------\n        dtype : str, dtype convertible\n        copy : bool, default False\n            copy if indicated\n        errors : str, {'raise', 'ignore'}, default 'raise'\n            - ``raise`` : allow exceptions to be raised\n            - ``ignore`` : suppress exceptions. On error return original object\n\n        Returns\n        -------\n        Block\n        "
        errors_legal_values = ('raise', 'ignore')
        if (errors not in errors_legal_values):
            invalid_arg = f"Expected value of kwarg 'errors' to be one of {list(errors_legal_values)}. Supplied value is '{errors}'"
            raise ValueError(invalid_arg)
        if (inspect.isclass(dtype) and issubclass(dtype, ExtensionDtype)):
            msg = f"Expected an instance of {dtype.__name__}, but got the class instead. Try instantiating 'dtype'."
            raise TypeError(msg)
        dtype = pandas_dtype(dtype)
        try:
            new_values = self._astype(dtype, copy=copy)
        except (ValueError, TypeError):
            if (errors == 'ignore'):
                new_values = self.values
            else:
                raise
        newb = self.make_block(new_values)
        if (newb.is_numeric and self.is_numeric):
            if (newb.shape != self.shape):
                raise TypeError(f'cannot set astype for copy = [{copy}] for dtype ({self.dtype.name} [{self.shape}]) to different shape ({newb.dtype.name} [{newb.shape}])')
        return newb

    def _astype(self, dtype, copy):
        values = self.values
        if (is_datetime64tz_dtype(dtype) and is_datetime64_dtype(values.dtype)):
            return astype_dt64_to_dt64tz(values, dtype, copy, via_utc=True)
        if is_dtype_equal(values.dtype, dtype):
            if copy:
                return values.copy()
            return values
        if isinstance(values, ExtensionArray):
            values = values.astype(dtype, copy=copy)
        else:
            values = astype_nansafe(values, dtype, copy=True)
        return values

    def convert(self, copy=True, datetime=True, numeric=True, timedelta=True):
        '\n        attempt to coerce any object types to better types return a copy\n        of the block (if copy = True) by definition we are not an ObjectBlock\n        here!\n        '
        return ([self.copy()] if copy else [self])

    def _can_hold_element(self, element):
        ' require the same dtype as ourselves '
        dtype = self.values.dtype.type
        tipo = maybe_infer_dtype_type(element)
        if (tipo is not None):
            return issubclass(tipo.type, dtype)
        return isinstance(element, dtype)

    def should_store(self, value):
        '\n        Should we set self.values[indexer] = value inplace or do we need to cast?\n\n        Parameters\n        ----------\n        value : np.ndarray or ExtensionArray\n\n        Returns\n        -------\n        bool\n        '
        return is_dtype_equal(value.dtype, self.dtype)

    def to_native_types(self, na_rep='nan', quoting=None, **kwargs):
        ' convert to our native types format '
        values = self.values
        mask = isna(values)
        itemsize = writers.word_len(na_rep)
        if ((not self.is_object) and (not quoting) and itemsize):
            values = values.astype(str)
            if ((values.dtype.itemsize / np.dtype('U1').itemsize) < itemsize):
                values = values.astype(f'<U{itemsize}')
        else:
            values = np.array(values, dtype='object')
        values[mask] = na_rep
        return self.make_block(values)

    def copy(self, deep=True):
        ' copy constructor '
        values = self.values
        if deep:
            values = values.copy()
        return self.make_block_same_class(values, ndim=self.ndim)

    def replace(self, to_replace, value, inplace=False, regex=False):
        '\n        replace the to_replace value with value, possible to create new\n        blocks here this is just a call to putmask. regex is not used here.\n        It is used in ObjectBlocks.  It is here for API compatibility.\n        '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        original_to_replace = to_replace
        if (not self._can_hold_element(to_replace)):
            return ([self] if inplace else [self.copy()])
        values = self.values
        mask = missing.mask_missing(values, to_replace)
        if (not mask.any()):
            return ([self] if inplace else [self.copy()])
        if (not self._can_hold_element(value)):
            blk = self.astype(object)
            return blk.replace(to_replace=original_to_replace, value=value, inplace=True, regex=regex)
        blk = (self if inplace else self.copy())
        putmask_inplace(blk.values, mask, value)
        blocks = blk.convert(numeric=False, copy=(not inplace))
        return blocks

    def _replace_regex(self, to_replace, value, inplace=False, convert=True, mask=None):
        '\n        Replace elements by the given value.\n\n        Parameters\n        ----------\n        to_replace : object or pattern\n            Scalar to replace or regular expression to match.\n        value : object\n            Replacement object.\n        inplace : bool, default False\n            Perform inplace modification.\n        convert : bool, default True\n            If true, try to coerce any object types to better types.\n        mask : array-like of bool, optional\n            True indicate corresponding element is ignored.\n\n        Returns\n        -------\n        List[Block]\n        '
        if (not self._can_hold_element(to_replace)):
            return ([self] if inplace else [self.copy()])
        rx = re.compile(to_replace)
        new_values = (self.values if inplace else self.values.copy())
        replace_regex(new_values, rx, value, mask)
        block = self.make_block(new_values)
        if convert:
            nbs = block.convert(numeric=False)
        else:
            nbs = [block]
        return nbs

    def _replace_list(self, src_list, dest_list, inplace=False, regex=False):
        '\n        See BlockManager._replace_list docstring.\n        '
        pairs = [(x, y) for (x, y) in zip(src_list, dest_list) if self._can_hold_element(x)]
        if (not len(pairs)):
            return ([self] if inplace else [self.copy()])
        src_len = (len(pairs) - 1)

        def comp(s: Scalar, mask: np.ndarray, regex: bool=False) -> np.ndarray:
            '\n            Generate a bool array by perform an equality check, or perform\n            an element-wise regular expression matching\n            '
            if isna(s):
                return (~ mask)
            return compare_or_regex_search(self.values, s, regex, mask)
        if self.is_object:
            mask = (~ isna(self.values))
            masks = [comp(s[0], mask, regex) for s in pairs]
        else:
            masks = [missing.mask_missing(self.values, s[0]) for s in pairs]
        masks = [_extract_bool_array(x) for x in masks]
        rb = [(self if inplace else self.copy())]
        for (i, (src, dest)) in enumerate(pairs):
            new_rb: List['Block'] = []
            for blk in rb:
                m = masks[i]
                convert = (i == src_len)
                result = blk._replace_coerce(to_replace=src, value=dest, mask=m, inplace=inplace, regex=regex)
                if (convert and blk.is_object):
                    result = extend_blocks([b.convert(numeric=False, copy=True) for b in result])
                new_rb.extend(result)
            rb = new_rb
        return rb

    def setitem(self, indexer, value):
        '\n        Attempt self.values[indexer] = value, possibly creating a new array.\n\n        Parameters\n        ----------\n        indexer : tuple, list-like, array-like, slice\n            The subset of self.values to set\n        value : object\n            The value being set\n\n        Returns\n        -------\n        Block\n\n        Notes\n        -----\n        `indexer` is a direct slice/positional indexer. `value` must\n        be a compatible shape.\n        '
        transpose = (self.ndim == 2)
        if (isinstance(indexer, np.ndarray) and (indexer.ndim > self.ndim)):
            raise ValueError(f'Cannot set values with ndim > {self.ndim}')
        if (value is None):
            if self.is_numeric:
                value = np.nan
        values = self.values
        if self._can_hold_element(value):
            if (self.dtype.kind in ['m', 'M']):
                arr = self.array_values().T
                arr[indexer] = value
                return self
        else:
            if hasattr(value, 'dtype'):
                dtype = value.dtype
            elif (lib.is_scalar(value) and (not isna(value))):
                (dtype, _) = infer_dtype_from_scalar(value, pandas_dtype=True)
            else:
                (dtype, _) = maybe_promote(np.array(value).dtype)
                return self.astype(dtype).setitem(indexer, value)
            dtype = find_common_type([values.dtype, dtype])
            assert (not is_dtype_equal(self.dtype, dtype))
            return self.astype(dtype).setitem(indexer, value)
        if is_extension_array_dtype(getattr(value, 'dtype', None)):
            is_ea_value = True
            arr_value = value
        else:
            is_ea_value = False
            arr_value = np.array(value)
        if transpose:
            values = values.T
        check_setitem_lengths(indexer, value, values)
        exact_match = (len(arr_value.shape) and (arr_value.shape[0] == values.shape[0]) and (arr_value.size == values.size))
        if is_empty_indexer(indexer, arr_value):
            pass
        elif is_scalar_indexer(indexer, self.ndim):
            values[indexer] = value
        elif (exact_match and is_categorical_dtype(arr_value.dtype)):
            values[indexer] = value
            return self.make_block(Categorical(self.values, dtype=arr_value.dtype))
        elif (exact_match and is_ea_value):
            return self.make_block(arr_value)
        elif exact_match:
            values[indexer] = value
            values = values.astype(arr_value.dtype, copy=False)
        else:
            values[indexer] = value
        if transpose:
            values = values.T
        block = self.make_block(values)
        return block

    def putmask(self, mask, new, axis=0):
        '\n        putmask the data to the block; it is possible that we may create a\n        new dtype of block\n\n        Return the resulting block(s).\n\n        Parameters\n        ----------\n        mask : np.ndarray[bool], SparseArray[bool], or BooleanArray\n        new : a ndarray/object\n        axis : int\n\n        Returns\n        -------\n        List[Block]\n        '
        transpose = (self.ndim == 2)
        mask = _extract_bool_array(mask)
        assert (not isinstance(new, (ABCIndex, ABCSeries, ABCDataFrame)))
        new_values = self.values
        if ((not is_list_like(new)) and isna(new) and (not self.is_object)):
            new = self.fill_value
        if self._can_hold_element(new):
            if (self.dtype.kind in ['m', 'M']):
                arr = self.array_values()
                arr = cast('NDArrayBackedExtensionArray', arr)
                if transpose:
                    arr = arr.T
                arr.putmask(mask, new)
                return [self]
            if transpose:
                new_values = new_values.T
            putmask_without_repeat(new_values, mask, new)
        elif mask.any():
            if transpose:
                mask = mask.T
                if isinstance(new, np.ndarray):
                    new = new.T
                axis = ((new_values.ndim - axis) - 1)

            def f(mask, val, idx):
                if (idx is None):
                    n = new
                else:
                    if isinstance(new, np.ndarray):
                        n = np.squeeze(new[(idx % new.shape[0])])
                    else:
                        n = np.array(new)
                    (dtype, _) = maybe_promote(n.dtype)
                    n = n.astype(dtype)
                nv = putmask_smart(val, mask, n)
                return nv
            new_blocks = self.split_and_operate(mask, f, True)
            return new_blocks
        return [self]

    def coerce_to_target_dtype(self, other):
        '\n        coerce the current block to a dtype compat for other\n        we will return a block, possibly object, and not raise\n\n        we can also safely try to coerce to the same dtype\n        and will receive the same block\n        '
        (dtype, _) = infer_dtype_from(other, pandas_dtype=True)
        new_dtype = find_common_type([self.dtype, dtype])
        return self.astype(new_dtype, copy=False)

    def interpolate(self, method='pad', axis=0, index=None, inplace=False, limit=None, limit_direction='forward', limit_area=None, fill_value=None, coerce=False, downcast=None, **kwargs):
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if (not self._can_hold_na):
            return (self if inplace else self.copy())
        try:
            m = missing.clean_fill_method(method)
        except ValueError:
            m = None
        if (m is not None):
            if (fill_value is not None):
                raise ValueError('Cannot pass both fill_value and method')
            return self._interpolate_with_fill(method=m, axis=axis, inplace=inplace, limit=limit, limit_area=limit_area, downcast=downcast)
        m = missing.clean_interp_method(method, **kwargs)
        assert (index is not None)
        return self._interpolate(method=m, index=index, axis=axis, limit=limit, limit_direction=limit_direction, limit_area=limit_area, fill_value=fill_value, inplace=inplace, downcast=downcast, **kwargs)

    def _interpolate_with_fill(self, method='pad', axis=0, inplace=False, limit=None, limit_area=None, downcast=None):
        ' fillna but using the interpolate machinery '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        assert self._can_hold_na
        values = (self.values if inplace else self.values.copy())
        values = missing.interpolate_2d(values, method=method, axis=axis, limit=limit, limit_area=limit_area)
        blocks = [self.make_block_same_class(values, ndim=self.ndim)]
        return self._maybe_downcast(blocks, downcast)

    def _interpolate(self, method, index, fill_value=None, axis=0, limit=None, limit_direction='forward', limit_area=None, inplace=False, downcast=None, **kwargs):
        ' interpolate using scipy wrappers '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        data = (self.values if inplace else self.values.copy())
        if (not self.is_float):
            if (not self.is_integer):
                return [self]
            data = data.astype(np.float64)
        if (fill_value is None):
            fill_value = self.fill_value
        if (method in ('krogh', 'piecewise_polynomial', 'pchip')):
            if (not index.is_monotonic):
                raise ValueError(f'{method} interpolation requires that the index be monotonic.')

        def func(yvalues: np.ndarray) -> np.ndarray:
            return missing.interpolate_1d(xvalues=index, yvalues=yvalues, method=method, limit=limit, limit_direction=limit_direction, limit_area=limit_area, fill_value=fill_value, bounds_error=False, **kwargs)
        interp_values = np.apply_along_axis(func, axis, data)
        blocks = [self.make_block_same_class(interp_values)]
        return self._maybe_downcast(blocks, downcast)

    def take_nd(self, indexer, axis, new_mgr_locs=None, fill_value=lib.no_default):
        '\n        Take values according to indexer and return them as a block.bb\n\n        '
        values = self.values
        if (fill_value is lib.no_default):
            fill_value = self.fill_value
            allow_fill = False
        else:
            allow_fill = True
        new_values = algos.take_nd(values, indexer, axis=axis, allow_fill=allow_fill, fill_value=fill_value)
        assert (not ((axis == 0) and (new_mgr_locs is None)))
        if (new_mgr_locs is None):
            new_mgr_locs = self.mgr_locs
        if (not is_dtype_equal(new_values.dtype, self.dtype)):
            return self.make_block(new_values, new_mgr_locs)
        else:
            return self.make_block_same_class(new_values, new_mgr_locs)

    def diff(self, n, axis=1):
        ' return block for the diff of the values '
        new_values = algos.diff(self.values, n, axis=axis, stacklevel=7)
        return [self.make_block(values=new_values)]

    def shift(self, periods, axis=0, fill_value=None):
        ' shift the block by periods, possibly upcast '
        (new_values, fill_value) = maybe_upcast(self.values, fill_value)
        new_values = shift(new_values, periods, axis, fill_value)
        return [self.make_block(new_values)]

    def where(self, other, cond, errors='raise', axis=0):
        "\n        evaluate the block; return result block(s) from the result\n\n        Parameters\n        ----------\n        other : a ndarray/object\n        cond : np.ndarray[bool], SparseArray[bool], or BooleanArray\n        errors : str, {'raise', 'ignore'}, default 'raise'\n            - ``raise`` : allow exceptions to be raised\n            - ``ignore`` : suppress exceptions. On error return original object\n        axis : int, default 0\n\n        Returns\n        -------\n        List[Block]\n        "
        import pandas.core.computation.expressions as expressions
        assert (not isinstance(other, (ABCIndex, ABCSeries, ABCDataFrame)))
        assert (errors in ['raise', 'ignore'])
        transpose = (self.ndim == 2)
        values = self.values
        orig_other = other
        if transpose:
            values = values.T
        cond = _extract_bool_array(cond)
        if cond.ravel('K').all():
            result = values
        else:
            if ((self.is_integer or self.is_bool) and lib.is_float(other) and np.isnan(other)):
                pass
            elif (not self._can_hold_element(other)):
                block = self.coerce_to_target_dtype(other)
                blocks = block.where(orig_other, cond, errors=errors, axis=axis)
                return self._maybe_downcast(blocks, 'infer')
            if (not ((self.is_integer or self.is_bool) and lib.is_float(other) and np.isnan(other))):
                other = convert_scalar_for_putitemlike(other, values.dtype)
            result = expressions.where(cond, values, other)
        if (self._can_hold_na or (self.ndim == 1)):
            if transpose:
                result = result.T
            return [self.make_block(result)]
        axis = (cond.ndim - 1)
        cond = cond.swapaxes(axis, 0)
        mask = np.array([cond[i].all() for i in range(cond.shape[0])], dtype=bool)
        result_blocks: List['Block'] = []
        for m in [mask, (~ mask)]:
            if m.any():
                result = cast(np.ndarray, result)
                taken = result.take(m.nonzero()[0], axis=axis)
                r = maybe_downcast_numeric(taken, self.dtype)
                nb = self.make_block(r.T, placement=self.mgr_locs[m])
                result_blocks.append(nb)
        return result_blocks

    def _unstack(self, unstacker, fill_value, new_placement):
        '\n        Return a list of unstacked blocks of self\n\n        Parameters\n        ----------\n        unstacker : reshape._Unstacker\n        fill_value : int\n            Only used in ExtensionBlock._unstack\n\n        Returns\n        -------\n        blocks : list of Block\n            New blocks of unstacked values.\n        mask : array_like of bool\n            The mask of columns of `blocks` we should keep.\n        '
        (new_values, mask) = unstacker.get_new_values(self.values.T, fill_value=fill_value)
        mask = mask.any(0)
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]
        blocks = [make_block(new_values, placement=new_placement)]
        return (blocks, mask)

    def quantile(self, qs, interpolation='linear', axis=0):
        "\n        compute the quantiles of the\n\n        Parameters\n        ----------\n        qs: a scalar or list of the quantiles to be computed\n        interpolation: type of interpolation, default 'linear'\n        axis: axis to compute, default 0\n\n        Returns\n        -------\n        Block\n        "
        assert (self.ndim == 2)
        values = self.get_values()
        is_empty = (values.shape[axis] == 0)
        orig_scalar = (not is_list_like(qs))
        if orig_scalar:
            qs = [qs]
        if is_empty:
            result = np.repeat(np.array(([self.fill_value] * len(qs))), len(values)).reshape(len(values), len(qs))
        else:
            mask = np.asarray(isna(values))
            result = nanpercentile(values, (np.array(qs) * 100), axis=axis, na_value=self.fill_value, mask=mask, ndim=values.ndim, interpolation=interpolation)
            result = np.array(result, copy=False)
            result = result.T
        if (orig_scalar and (not lib.is_scalar(result))):
            assert (result.shape[(- 1)] == 1), result.shape
            result = result[(..., 0)]
            result = lib.item_from_zerodim(result)
        ndim = np.ndim(result)
        return make_block(result, placement=np.arange(len(result)), ndim=ndim)

    def _replace_coerce(self, to_replace, value, mask, inplace=True, regex=False):
        '\n        Replace value corresponding to the given boolean array with another\n        value.\n\n        Parameters\n        ----------\n        to_replace : object or pattern\n            Scalar to replace or regular expression to match.\n        value : object\n            Replacement object.\n        mask : np.ndarray[bool]\n            True indicate corresponding element is ignored.\n        inplace : bool, default True\n            Perform inplace modification.\n        regex : bool, default False\n            If true, perform regular expression substitution.\n\n        Returns\n        -------\n        List[Block]\n        '
        if mask.any():
            if (not regex):
                nb = self.coerce_to_target_dtype(value)
                if ((nb is self) and (not inplace)):
                    nb = nb.copy()
                putmask_inplace(nb.values, mask, value)
                return [nb]
            else:
                regex = _should_use_regex(regex, to_replace)
                if regex:
                    return self._replace_regex(to_replace, value, inplace=inplace, convert=False, mask=mask)
                return self.replace(to_replace, value, inplace=inplace, regex=False)
        return [self]

class ExtensionBlock(Block):
    "\n    Block for holding extension types.\n\n    Notes\n    -----\n    This holds all 3rd-party extension array types. It's also the immediate\n    parent class for our internal extension types' blocks, CategoricalBlock.\n\n    ExtensionArrays are limited to 1-D.\n    "
    _can_consolidate = False
    _validate_ndim = False
    is_extension = True

    def __init__(self, values, placement, ndim):
        "\n        Initialize a non-consolidatable block.\n\n        'ndim' may be inferred from 'placement'.\n\n        This will call continue to call __init__ for the other base\n        classes mixed in with this Mixin.\n        "
        if (not isinstance(placement, libinternals.BlockPlacement)):
            placement = libinternals.BlockPlacement(placement)
        if (ndim is None):
            if (len(placement) != 1):
                ndim = 1
            else:
                ndim = 2
        super().__init__(values, placement, ndim=ndim)
        if ((self.ndim == 2) and (len(self.mgr_locs) != 1)):
            raise AssertionError('block.size != values.size')

    @property
    def shape(self):
        if (self.ndim == 1):
            return (len(self.values),)
        return (len(self.mgr_locs), len(self.values))

    def iget(self, col):
        if ((self.ndim == 2) and isinstance(col, tuple)):
            (col, loc) = col
            if ((not com.is_null_slice(col)) and (col != 0)):
                raise IndexError(f'{self} only contains one item')
            elif isinstance(col, slice):
                if (col != slice(None)):
                    raise NotImplementedError(col)
                return self.values[[loc]]
            return self.values[loc]
        else:
            if (col != 0):
                raise IndexError(f'{self} only contains one item')
            return self.values

    def set_inplace(self, locs, values):
        assert (locs.tolist() == [0])
        self.values = values

    def putmask(self, mask, new, axis=0):
        '\n        See Block.putmask.__doc__\n        '
        mask = _extract_bool_array(mask)
        new_values = self.values
        if (isinstance(new, (np.ndarray, ExtensionArray)) and (len(new) == len(mask))):
            new = new[mask]
        mask = safe_reshape(mask, new_values.shape)
        new_values[mask] = new
        return [self.make_block(values=new_values)]

    def _maybe_coerce_values(self, values):
        '\n        Unbox to an extension array.\n\n        This will unbox an ExtensionArray stored in an Index or Series.\n        ExtensionArrays pass through. No dtype coercion is done.\n\n        Parameters\n        ----------\n        values : Index, Series, ExtensionArray\n\n        Returns\n        -------\n        ExtensionArray\n        '
        return extract_array(values)

    @property
    def _holder(self):
        return type(self.values)

    @property
    def fill_value(self):
        return self.values.dtype.na_value

    @property
    def _can_hold_na(self):
        return self._holder._can_hold_na

    @property
    def is_view(self):
        'Extension arrays are never treated as views.'
        return False

    @property
    def is_numeric(self):
        return self.values.dtype._is_numeric

    def setitem(self, indexer, value):
        '\n        Attempt self.values[indexer] = value, possibly creating a new array.\n\n        This differs from Block.setitem by not allowing setitem to change\n        the dtype of the Block.\n\n        Parameters\n        ----------\n        indexer : tuple, list-like, array-like, slice\n            The subset of self.values to set\n        value : object\n            The value being set\n\n        Returns\n        -------\n        Block\n\n        Notes\n        -----\n        `indexer` is a direct slice/positional indexer. `value` must\n        be a compatible shape.\n        '
        if (not self._can_hold_element(value)):
            return self.astype(object).setitem(indexer, value)
        if isinstance(indexer, tuple):
            indexer = indexer[0]
        check_setitem_lengths(indexer, value, self.values)
        self.values[indexer] = value
        return self

    def get_values(self, dtype=None):
        return np.asarray(self.values).reshape(self.shape)

    def array_values(self):
        return self.values

    def to_native_types(self, na_rep='nan', quoting=None, **kwargs):
        'override to use ExtensionArray astype for the conversion'
        values = self.values
        mask = isna(values)
        values = np.asarray(values.astype(object))
        values[mask] = na_rep
        return self.make_block(values)

    def take_nd(self, indexer, axis=0, new_mgr_locs=None, fill_value=lib.no_default):
        '\n        Take values according to indexer and return them as a block.\n        '
        if (fill_value is lib.no_default):
            fill_value = None
        new_values = self.values.take(indexer, fill_value=fill_value, allow_fill=True)
        assert (not ((self.ndim == 1) and (new_mgr_locs is None)))
        if (new_mgr_locs is None):
            new_mgr_locs = self.mgr_locs
        return self.make_block_same_class(new_values, new_mgr_locs)

    def _can_hold_element(self, element):
        return True

    def _slice(self, slicer):
        '\n        Return a slice of my values.\n\n        Parameters\n        ----------\n        slicer : slice, ndarray[int], or a tuple of these\n            Valid (non-reducing) indexer for self.values.\n\n        Returns\n        -------\n        np.ndarray or ExtensionArray\n        '
        if ((not isinstance(slicer, tuple)) and (self.ndim == 2)):
            slicer = (slicer, slice(None))
        if (isinstance(slicer, tuple) and (len(slicer) == 2)):
            first = slicer[0]
            if (not isinstance(first, slice)):
                raise AssertionError('invalid slicing for a 1-ndim ExtensionArray', first)
            new_locs = self.mgr_locs[first]
            if len(new_locs):
                slicer = slicer[1]
            else:
                raise AssertionError('invalid slicing for a 1-ndim ExtensionArray', slicer)
        return self.values[slicer]

    def fillna(self, value, limit=None, inplace=False, downcast=None):
        values = (self.values if inplace else self.values.copy())
        values = values.fillna(value=value, limit=limit)
        return [self.make_block_same_class(values=values, placement=self.mgr_locs, ndim=self.ndim)]

    def interpolate(self, method='pad', axis=0, inplace=False, limit=None, fill_value=None, **kwargs):
        values = (self.values if inplace else self.values.copy())
        return self.make_block_same_class(values=values.fillna(value=fill_value, method=method, limit=limit), placement=self.mgr_locs)

    def diff(self, n, axis=1):
        if ((axis == 0) and (n != 0)):
            return super().diff(len(self.values), axis=0)
        if (axis == 1):
            axis = 0
        return super().diff(n, axis)

    def shift(self, periods, axis=0, fill_value=None):
        '\n        Shift the block by `periods`.\n\n        Dispatches to underlying ExtensionArray and re-boxes in an\n        ExtensionBlock.\n        '
        return [self.make_block_same_class(self.values.shift(periods=periods, fill_value=fill_value), placement=self.mgr_locs, ndim=self.ndim)]

    def where(self, other, cond, errors='raise', axis=0):
        cond = _extract_bool_array(cond)
        assert (not isinstance(other, (ABCIndex, ABCSeries, ABCDataFrame)))
        if (isinstance(other, np.ndarray) and (other.ndim == 2)):
            assert (other.shape[1] == 1)
            other = other[:, 0]
        if (isinstance(cond, np.ndarray) and (cond.ndim == 2)):
            assert (cond.shape[1] == 1)
            cond = cond[:, 0]
        if (lib.is_scalar(other) and isna(other)):
            other = self.dtype.na_value
        if is_sparse(self.values):
            dtype = None
        else:
            dtype = self.dtype
        result = self.values.copy()
        icond = (~ cond)
        if lib.is_scalar(other):
            set_other = other
        else:
            set_other = other[icond]
        try:
            result[icond] = set_other
        except (NotImplementedError, TypeError):
            result = self._holder._from_sequence(np.where(cond, self.values, other), dtype=dtype)
        return [self.make_block_same_class(result, placement=self.mgr_locs)]

    def _unstack(self, unstacker, fill_value, new_placement):
        n_rows = self.shape[(- 1)]
        dummy_arr = np.arange(n_rows)
        (new_values, mask) = unstacker.get_new_values(dummy_arr, fill_value=(- 1))
        mask = mask.any(0)
        blocks = [self.make_block_same_class(self.values.take(indices, allow_fill=True, fill_value=fill_value), [place]) for (indices, place) in zip(new_values.T, new_placement)]
        return (blocks, mask)

class ObjectValuesExtensionBlock(ExtensionBlock):
    '\n    Block providing backwards-compatibility for `.values`.\n\n    Used by PeriodArray and IntervalArray to ensure that\n    Series[T].values is an ndarray of objects.\n    '

    def external_values(self):
        return self.values.astype(object)

    def _can_hold_element(self, element):
        if is_valid_nat_for_dtype(element, self.dtype):
            return True
        if (isinstance(element, list) and (len(element) == 0)):
            return True
        tipo = maybe_infer_dtype_type(element)
        if (tipo is not None):
            return issubclass(tipo.type, self.dtype.type)
        return isinstance(element, self.dtype.type)

class NumericBlock(Block):
    __slots__ = ()
    is_numeric = True
    _can_hold_na = True

class FloatBlock(NumericBlock):
    __slots__ = ()
    is_float = True

    def _can_hold_element(self, element):
        tipo = maybe_infer_dtype_type(element)
        if (tipo is not None):
            return (issubclass(tipo.type, (np.floating, np.integer)) and (not issubclass(tipo.type, np.timedelta64)))
        return (isinstance(element, (float, int, np.floating, np.int_)) and (not isinstance(element, (bool, np.bool_, np.timedelta64))))

    def to_native_types(self, na_rep='', float_format=None, decimal='.', quoting=None, **kwargs):
        ' convert to our native types format '
        values = self.values
        if ((float_format is None) and (decimal == '.')):
            mask = isna(values)
            if (not quoting):
                values = values.astype(str)
            else:
                values = np.array(values, dtype='object')
            values[mask] = na_rep
            return self.make_block(values)
        from pandas.io.formats.format import FloatArrayFormatter
        formatter = FloatArrayFormatter(values, na_rep=na_rep, float_format=float_format, decimal=decimal, quoting=quoting, fixed_width=False)
        res = formatter.get_result_as_array()
        return self.make_block(res)

class ComplexBlock(NumericBlock):
    __slots__ = ()
    is_complex = True

    def _can_hold_element(self, element):
        tipo = maybe_infer_dtype_type(element)
        if (tipo is not None):
            return issubclass(tipo.type, (np.floating, np.integer, np.complexfloating))
        return (isinstance(element, (float, int, complex, np.float_, np.int_)) and (not isinstance(element, (bool, np.bool_))))

class IntBlock(NumericBlock):
    __slots__ = ()
    is_integer = True
    _can_hold_na = False

    def _can_hold_element(self, element):
        tipo = maybe_infer_dtype_type(element)
        if (tipo is not None):
            return (issubclass(tipo.type, np.integer) and (not issubclass(tipo.type, np.timedelta64)) and (self.dtype.itemsize >= tipo.itemsize))
        return (is_integer(element) or (is_float(element) and element.is_integer()))

class DatetimeLikeBlockMixin(Block):
    'Mixin class for DatetimeBlock, DatetimeTZBlock, and TimedeltaBlock.'
    _can_hold_na = True

    def get_values(self, dtype=None):
        '\n        return object dtype as boxed values, such as Timestamps/Timedelta\n        '
        if is_object_dtype(dtype):
            return self._holder(self.values).astype(object)
        return self.values

    def internal_values(self):
        return self.array_values()

    def array_values(self):
        return self._holder._simple_new(self.values)

    def iget(self, key):
        return self.array_values().reshape(self.shape)[key]

    def diff(self, n, axis=0):
        '\n        1st discrete difference.\n\n        Parameters\n        ----------\n        n : int\n            Number of periods to diff.\n        axis : int, default 0\n            Axis to diff upon.\n\n        Returns\n        -------\n        A list with a new TimeDeltaBlock.\n\n        Notes\n        -----\n        The arguments here are mimicking shift so they are called correctly\n        by apply.\n        '
        values = self.array_values().reshape(self.shape)
        new_values = (values - values.shift(n, axis=axis))
        return [TimeDeltaBlock(new_values, placement=self.mgr_locs.indexer, ndim=self.ndim)]

    def shift(self, periods, axis=0, fill_value=None):
        values = self.array_values()
        new_values = values.shift(periods, fill_value=fill_value, axis=axis)
        return self.make_block_same_class(new_values)

    def to_native_types(self, na_rep='NaT', **kwargs):
        ' convert to our native types format '
        arr = self.array_values()
        result = arr._format_native_types(na_rep=na_rep, **kwargs)
        return self.make_block(result)

    def where(self, other, cond, errors='raise', axis=0):
        arr = self.array_values().reshape(self.shape)
        cond = _extract_bool_array(cond)
        try:
            res_values = arr.T.where(cond, other).T
        except (ValueError, TypeError):
            return super().where(other, cond, errors=errors, axis=axis)
        res_values = res_values.reshape(self.values.shape)
        nb = self.make_block_same_class(res_values)
        return [nb]

    def _can_hold_element(self, element):
        arr = self.array_values()
        try:
            arr._validate_setitem_value(element)
            return True
        except (TypeError, ValueError):
            return False

class DatetimeBlock(DatetimeLikeBlockMixin):
    __slots__ = ()
    is_datetime = True
    _holder = DatetimeArray
    fill_value = np.datetime64('NaT', 'ns')

    def _maybe_coerce_values(self, values):
        '\n        Input validation for values passed to __init__. Ensure that\n        we have datetime64ns, coercing if necessary.\n\n        Parameters\n        ----------\n        values : array-like\n            Must be convertible to datetime64\n\n        Returns\n        -------\n        values : ndarray[datetime64ns]\n\n        Overridden by DatetimeTZBlock.\n        '
        if (values.dtype != DT64NS_DTYPE):
            values = conversion.ensure_datetime64ns(values)
        if isinstance(values, DatetimeArray):
            values = values._data
        assert isinstance(values, np.ndarray), type(values)
        return values

    def set_inplace(self, locs, values):
        '\n        See Block.set.__doc__\n        '
        values = conversion.ensure_datetime64ns(values, copy=False)
        self.values[locs] = values

class DatetimeTZBlock(ExtensionBlock, DatetimeBlock):
    ' implement a datetime64 block with a tz attribute '
    __slots__ = ()
    is_datetimetz = True
    is_extension = True
    internal_values = Block.internal_values
    _holder = DatetimeBlock._holder
    _can_hold_element = DatetimeBlock._can_hold_element
    to_native_types = DatetimeBlock.to_native_types
    diff = DatetimeBlock.diff
    fillna = DatetimeBlock.fillna
    fill_value = DatetimeBlock.fill_value
    _can_hold_na = DatetimeBlock._can_hold_na
    where = DatetimeBlock.where
    array_values = ExtensionBlock.array_values

    def _maybe_coerce_values(self, values):
        '\n        Input validation for values passed to __init__. Ensure that\n        we have datetime64TZ, coercing if necessary.\n\n        Parameters\n        ----------\n        values : array-like\n            Must be convertible to datetime64\n\n        Returns\n        -------\n        values : DatetimeArray\n        '
        if (not isinstance(values, self._holder)):
            values = self._holder(values)
        if (values.tz is None):
            raise ValueError('cannot create a DatetimeTZBlock without a tz')
        return values

    @property
    def is_view(self):
        ' return a boolean if I am possibly a view '
        return (self.values._data.base is not None)

    def get_values(self, dtype=None):
        '\n        Returns an ndarray of values.\n\n        Parameters\n        ----------\n        dtype : np.dtype\n            Only `object`-like dtypes are respected here (not sure\n            why).\n\n        Returns\n        -------\n        values : ndarray\n            When ``dtype=object``, then and object-dtype ndarray of\n            boxed values is returned. Otherwise, an M8[ns] ndarray\n            is returned.\n\n            DatetimeArray is always 1-d. ``get_values`` will reshape\n            the return value to be the same dimensionality as the\n            block.\n        '
        values = self.values
        if is_object_dtype(dtype):
            values = values.astype(object)
        return np.asarray(values).reshape(self.shape)

    def external_values(self):
        return np.asarray(self.values.astype('datetime64[ns]', copy=False))

    def quantile(self, qs, interpolation='linear', axis=0):
        naive = self.values.view('M8[ns]')
        naive = naive.reshape(self.shape)
        blk = self.make_block(naive)
        res_blk = blk.quantile(qs, interpolation=interpolation, axis=axis)
        aware = self._holder(res_blk.values.ravel(), dtype=self.dtype)
        return self.make_block_same_class(aware, ndim=res_blk.ndim)

    def _check_ndim(self, values, ndim):
        '\n        ndim inference and validation.\n\n        This is overridden by the DatetimeTZBlock to check the case of 2D\n        data (values.ndim == 2), which should only be allowed if ndim is\n        also 2.\n        The case of 1D array is still allowed with both ndim of 1 or 2, as\n        if the case for other EAs. Therefore, we are only checking\n        `values.ndim > ndim` instead of `values.ndim != ndim` as for\n        consolidated blocks.\n        '
        if (ndim is None):
            ndim = values.ndim
        if (values.ndim > ndim):
            raise ValueError(f'Wrong number of dimensions. values.ndim != ndim [{values.ndim} != {ndim}]')
        return ndim

class TimeDeltaBlock(DatetimeLikeBlockMixin):
    __slots__ = ()
    is_timedelta = True
    fill_value = np.timedelta64('NaT', 'ns')

    def _maybe_coerce_values(self, values):
        if (values.dtype != TD64NS_DTYPE):
            if (values.dtype.kind != 'm'):
                raise TypeError(values.dtype)
            values = TimedeltaArray._from_sequence(values)._data
        if isinstance(values, TimedeltaArray):
            values = values._data
        assert isinstance(values, np.ndarray), type(values)
        return values

    @property
    def _holder(self):
        return TimedeltaArray

    def fillna(self, value, **kwargs):
        if is_integer(value):
            raise TypeError('Passing integers to fillna for timedelta64[ns] dtype is no longer supported.  To obtain the old behavior, pass `pd.Timedelta(seconds=n)` instead.')
        return super().fillna(value, **kwargs)

class BoolBlock(NumericBlock):
    __slots__ = ()
    is_bool = True
    _can_hold_na = False

    def _can_hold_element(self, element):
        tipo = maybe_infer_dtype_type(element)
        if (tipo is not None):
            return issubclass(tipo.type, np.bool_)
        return isinstance(element, (bool, np.bool_))

class ObjectBlock(Block):
    __slots__ = ()
    is_object = True
    _can_hold_na = True

    def _maybe_coerce_values(self, values):
        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)
        return values

    @property
    def is_bool(self):
        '\n        we can be a bool if we have only bool values but are of type\n        object\n        '
        return lib.is_bool_array(self.values.ravel('K'))

    def reduce(self, func, ignore_failures=False):
        '\n        For object-dtype, we operate column-wise.\n        '
        assert (self.ndim == 2)
        values = self.values
        if (len(values) > 1):

            def mask_func(mask, values, inplace):
                if (values.ndim == 1):
                    values = values.reshape(1, (- 1))
                return func(values)
            return self.split_and_operate(None, mask_func, False, ignore_failures=ignore_failures)
        try:
            res = func(values)
        except TypeError:
            if (not ignore_failures):
                raise
            return []
        assert isinstance(res, np.ndarray)
        assert (res.ndim == 1)
        res = res.reshape(1, (- 1))
        return [self.make_block_same_class(res)]

    def convert(self, copy=True, datetime=True, numeric=True, timedelta=True):
        '\n        attempt to cast any object types to better types return a copy of\n        the block (if copy = True) by definition we ARE an ObjectBlock!!!!!\n        '

        def f(mask, val, idx):
            shape = val.shape
            values = soft_convert_objects(val.ravel(), datetime=datetime, numeric=numeric, timedelta=timedelta, copy=copy)
            if isinstance(values, np.ndarray):
                values = values.reshape(shape)
            return values
        if (self.ndim == 2):
            blocks = self.split_and_operate(None, f, False)
        else:
            values = f(None, self.values.ravel(), None)
            blocks = [self.make_block(values)]
        return blocks

    def _maybe_downcast(self, blocks, downcast=None):
        if (downcast is not None):
            return blocks
        return extend_blocks([b.convert(datetime=True, numeric=False) for b in blocks])

    def _can_hold_element(self, element):
        return True

    def replace(self, to_replace, value, inplace=False, regex=False):
        regex = _should_use_regex(regex, to_replace)
        if regex:
            return self._replace_regex(to_replace, value, inplace=inplace)
        else:
            return super().replace(to_replace, value, inplace=inplace, regex=False)

def _should_use_regex(regex, to_replace):
    '\n    Decide whether to treat `to_replace` as a regular expression.\n    '
    if is_re(to_replace):
        regex = True
    regex = (regex and is_re_compilable(to_replace))
    regex = (regex and (re.compile(to_replace).pattern != ''))
    return regex

class CategoricalBlock(ExtensionBlock):
    __slots__ = ()

    def _replace_list(self, src_list, dest_list, inplace=False, regex=False):
        if (len(algos.unique(dest_list)) == 1):
            return self.replace(src_list, dest_list[0], inplace, regex)
        return super()._replace_list(src_list, dest_list, inplace, regex)

    def replace(self, to_replace, value, inplace=False, regex=False):
        inplace = validate_bool_kwarg(inplace, 'inplace')
        result = (self if inplace else self.copy())
        result.values.replace(to_replace, value, inplace=True)
        return [result]

def get_block_type(values, dtype=None):
    '\n    Find the appropriate Block subclass to use for the given values and dtype.\n\n    Parameters\n    ----------\n    values : ndarray-like\n    dtype : numpy or pandas dtype\n\n    Returns\n    -------\n    cls : class, subclass of Block\n    '
    dtype = (dtype or values.dtype)
    vtype = dtype.type
    kind = dtype.kind
    cls: Type[Block]
    if is_sparse(dtype):
        cls = ExtensionBlock
    elif isinstance(dtype, CategoricalDtype):
        cls = CategoricalBlock
    elif (vtype is Timestamp):
        cls = DatetimeTZBlock
    elif ((vtype is Interval) or (vtype is Period)):
        cls = ObjectValuesExtensionBlock
    elif isinstance(dtype, ExtensionDtype):
        cls = ExtensionBlock
    elif (kind == 'M'):
        cls = DatetimeBlock
    elif (kind == 'm'):
        cls = TimeDeltaBlock
    elif (kind == 'f'):
        cls = FloatBlock
    elif (kind == 'c'):
        cls = ComplexBlock
    elif ((kind == 'i') or (kind == 'u')):
        cls = IntBlock
    elif (kind == 'b'):
        cls = BoolBlock
    else:
        cls = ObjectBlock
    return cls

def make_block(values, placement, klass=None, ndim=None, dtype=None):
    if isinstance(values, ABCPandasArray):
        values = values.to_numpy()
        if (ndim and (ndim > 1)):
            values = np.atleast_2d(values)
    if isinstance(dtype, PandasDtype):
        dtype = dtype.numpy_dtype
    if (klass is None):
        dtype = (dtype or values.dtype)
        klass = get_block_type(values, dtype)
    elif ((klass is DatetimeTZBlock) and (not is_datetime64tz_dtype(values.dtype))):
        values = DatetimeArray._simple_new(values, dtype=dtype)
    return klass(values, ndim=ndim, placement=placement)

def extend_blocks(result, blocks=None):
    ' return a new extended blocks, given the result '
    if (blocks is None):
        blocks = []
    if isinstance(result, list):
        for r in result:
            if isinstance(r, list):
                blocks.extend(r)
            else:
                blocks.append(r)
    else:
        assert isinstance(result, Block), type(result)
        blocks.append(result)
    return blocks

def _block_shape(values, ndim=1):
    ' guarantee the shape of the values to be at least 1 d '
    if (values.ndim < ndim):
        shape = values.shape
        if (not is_extension_array_dtype(values.dtype)):
            values = values.reshape(tuple(((1,) + shape)))
    return values

def safe_reshape(arr, new_shape):
    '\n    Reshape `arr` to have shape `new_shape`, unless it is an ExtensionArray,\n    in which case it will be returned unchanged (see gh-13012).\n\n    Parameters\n    ----------\n    arr : np.ndarray or ExtensionArray\n    new_shape : Tuple[int]\n\n    Returns\n    -------\n    np.ndarray or ExtensionArray\n    '
    if (not is_extension_array_dtype(arr.dtype)):
        arr = np.asarray(arr).reshape(new_shape)
    return arr

def _extract_bool_array(mask):
    '\n    If we have a SparseArray or BooleanArray, convert it to ndarray[bool].\n    '
    if isinstance(mask, ExtensionArray):
        mask = mask.to_numpy(dtype=bool, na_value=False)
    assert isinstance(mask, np.ndarray), type(mask)
    assert (mask.dtype == bool), mask.dtype
    return mask
