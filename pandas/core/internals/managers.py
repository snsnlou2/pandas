
from collections import defaultdict
import itertools
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings
import numpy as np
from pandas._libs import internals as libinternals, lib
from pandas._typing import ArrayLike, DtypeObj, Label, Shape
from pandas.errors import PerformanceWarning
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import find_common_type, infer_dtype_from_scalar, maybe_promote
from pandas.core.dtypes.common import DT64NS_DTYPE, is_dtype_equal, is_extension_array_dtype, is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCPandasArray, ABCSeries
from pandas.core.dtypes.missing import array_equals, isna
import pandas.core.algorithms as algos
from pandas.core.arrays.sparse import SparseDtype
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import Index, ensure_index
from pandas.core.internals.blocks import Block, CategoricalBlock, DatetimeTZBlock, ExtensionBlock, ObjectValuesExtensionBlock, extend_blocks, get_block_type, make_block, safe_reshape
from pandas.core.internals.ops import blockwise_all, operate_blockwise
T = TypeVar('T', bound='BlockManager')

class BlockManager(PandasObject):
    "\n    Core internal data structure to implement DataFrame, Series, etc.\n\n    Manage a bunch of labeled 2D mixed-type ndarrays. Essentially it's a\n    lightweight blocked set of labeled data to be manipulated by the DataFrame\n    public API class\n\n    Attributes\n    ----------\n    shape\n    ndim\n    axes\n    values\n    items\n\n    Methods\n    -------\n    set_axis(axis, new_labels)\n    copy(deep=True)\n\n    get_dtypes\n\n    apply(func, axes, block_filter_fn)\n\n    get_bool_data\n    get_numeric_data\n\n    get_slice(slice_like, axis)\n    get(label)\n    iget(loc)\n\n    take(indexer, axis)\n    reindex_axis(new_labels, axis)\n    reindex_indexer(new_labels, indexer, axis)\n\n    delete(label)\n    insert(loc, label, value)\n    set(label, value)\n\n    Parameters\n    ----------\n    blocks: Sequence of Block\n    axes: Sequence of Index\n    do_integrity_check: bool, default True\n\n    Notes\n    -----\n    This is *not* a public API class\n    "
    __slots__ = ['axes', 'blocks', '_known_consolidated', '_is_consolidated', '_blknos', '_blklocs']

    def __init__(self, blocks, axes, do_integrity_check=True):
        self.axes = [ensure_index(ax) for ax in axes]
        self.blocks: Tuple[(Block, ...)] = tuple(blocks)
        for block in blocks:
            if (self.ndim != block.ndim):
                raise AssertionError(f'Number of Block dimensions ({block.ndim}) must equal number of axes ({self.ndim})')
        if do_integrity_check:
            self._verify_integrity()
        self._known_consolidated = False
        self._blknos = None
        self._blklocs = None

    @classmethod
    def from_blocks(cls, blocks, axes):
        '\n        Constructor for BlockManager and SingleBlockManager with same signature.\n        '
        return cls(blocks, axes, do_integrity_check=False)

    @property
    def blknos(self):
        "\n        Suppose we want to find the array corresponding to our i'th column.\n\n        blknos[i] identifies the block from self.blocks that contains this column.\n\n        blklocs[i] identifies the column of interest within\n        self.blocks[self.blknos[i]]\n        "
        if (self._blknos is None):
            self._rebuild_blknos_and_blklocs()
        return self._blknos

    @property
    def blklocs(self):
        '\n        See blknos.__doc__\n        '
        if (self._blklocs is None):
            self._rebuild_blknos_and_blklocs()
        return self._blklocs

    def make_empty(self, axes=None):
        ' return an empty BlockManager with the items axis of len 0 '
        if (axes is None):
            axes = ([Index([])] + self.axes[1:])
        if (self.ndim == 1):
            assert isinstance(self, SingleBlockManager)
            blk = self.blocks[0]
            arr = blk.values[:0]
            nb = blk.make_block_same_class(arr, placement=slice(0, 0), ndim=1)
            blocks = [nb]
        else:
            blocks = []
        return type(self).from_blocks(blocks, axes)

    def __nonzero__(self):
        return True
    __bool__ = __nonzero__

    @property
    def shape(self):
        return tuple((len(ax) for ax in self.axes))

    @property
    def ndim(self):
        return len(self.axes)

    def set_axis(self, axis, new_labels):
        old_len = len(self.axes[axis])
        new_len = len(new_labels)
        if (new_len != old_len):
            raise ValueError(f'Length mismatch: Expected axis has {old_len} elements, new values have {new_len} elements')
        self.axes[axis] = new_labels

    @property
    def is_single_block(self):
        return (len(self.blocks) == 1)

    def _rebuild_blknos_and_blklocs(self):
        '\n        Update mgr._blknos / mgr._blklocs.\n        '
        new_blknos = np.empty(self.shape[0], dtype=np.intp)
        new_blklocs = np.empty(self.shape[0], dtype=np.intp)
        new_blknos.fill((- 1))
        new_blklocs.fill((- 1))
        for (blkno, blk) in enumerate(self.blocks):
            rl = blk.mgr_locs
            new_blknos[rl.indexer] = blkno
            new_blklocs[rl.indexer] = np.arange(len(rl))
        if (new_blknos == (- 1)).any():
            raise AssertionError('Gaps in blk ref_locs')
        self._blknos = new_blknos
        self._blklocs = new_blklocs

    @property
    def items(self):
        return self.axes[0]

    def get_dtypes(self):
        dtypes = np.array([blk.dtype for blk in self.blocks])
        return algos.take_1d(dtypes, self.blknos, allow_fill=False)

    def __getstate__(self):
        block_values = [b.values for b in self.blocks]
        block_items = [self.items[b.mgr_locs.indexer] for b in self.blocks]
        axes_array = list(self.axes)
        extra_state = {'0.14.1': {'axes': axes_array, 'blocks': [{'values': b.values, 'mgr_locs': b.mgr_locs.indexer} for b in self.blocks]}}
        return (axes_array, block_values, block_items, extra_state)

    def __setstate__(self, state):

        def unpickle_block(values, mgr_locs, ndim: int):
            return make_block(values, placement=mgr_locs, ndim=ndim)
        if (isinstance(state, tuple) and (len(state) >= 4) and ('0.14.1' in state[3])):
            state = state[3]['0.14.1']
            self.axes = [ensure_index(ax) for ax in state['axes']]
            ndim = len(self.axes)
            self.blocks = tuple((unpickle_block(b['values'], b['mgr_locs'], ndim=ndim) for b in state['blocks']))
        else:
            raise NotImplementedError('pre-0.14.1 pickles are no longer supported')
        self._post_setstate()

    def _post_setstate(self):
        self._is_consolidated = False
        self._known_consolidated = False
        self._rebuild_blknos_and_blklocs()

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        output = type(self).__name__
        for (i, ax) in enumerate(self.axes):
            if (i == 0):
                output += f'''
Items: {ax}'''
            else:
                output += f'''
Axis {i}: {ax}'''
        for block in self.blocks:
            output += f'''
{block}'''
        return output

    def _verify_integrity(self):
        mgr_shape = self.shape
        tot_items = sum((len(x.mgr_locs) for x in self.blocks))
        for block in self.blocks:
            if (block.shape[1:] != mgr_shape[1:]):
                raise construction_error(tot_items, block.shape[1:], self.axes)
        if (len(self.items) != tot_items):
            raise AssertionError(f'''Number of manager items must equal union of block items
# manager items: {len(self.items)}, # tot_items: {tot_items}''')

    def reduce(self, func, ignore_failures=False):
        '\n        Apply reduction function blockwise, returning a single-row BlockManager.\n\n        Parameters\n        ----------\n        func : reduction function\n        ignore_failures : bool, default False\n            Whether to drop blocks where func raises TypeError.\n\n        Returns\n        -------\n        BlockManager\n        np.ndarray\n            Indexer of mgr_locs that are retained.\n        '
        assert (self.ndim == 2)
        res_blocks: List[Block] = []
        for blk in self.blocks:
            nbs = blk.reduce(func, ignore_failures)
            res_blocks.extend(nbs)
        index = Index([None])
        if ignore_failures:
            if res_blocks:
                indexer = np.concatenate([blk.mgr_locs.as_array for blk in res_blocks])
                new_mgr = self._combine(res_blocks, copy=False, index=index)
            else:
                indexer = []
                new_mgr = type(self).from_blocks([], [Index([]), index])
        else:
            indexer = np.arange(self.shape[0])
            new_mgr = type(self).from_blocks(res_blocks, [self.items, index])
        return (new_mgr, indexer)

    def operate_blockwise(self, other, array_op):
        '\n        Apply array_op blockwise with another (aligned) BlockManager.\n        '
        return operate_blockwise(self, other, array_op)

    def apply(self, f, align_keys=None, ignore_failures=False, **kwargs):
        '\n        Iterate over the blocks, collect and create a new BlockManager.\n\n        Parameters\n        ----------\n        f : str or callable\n            Name of the Block method to apply.\n        align_keys: List[str] or None, default None\n        ignore_failures: bool, default False\n        **kwargs\n            Keywords to pass to `f`\n\n        Returns\n        -------\n        BlockManager\n        '
        assert ('filter' not in kwargs)
        align_keys = (align_keys or [])
        result_blocks: List[Block] = []
        aligned_args = {k: kwargs[k] for k in align_keys}
        for b in self.blocks:
            if aligned_args:
                for (k, obj) in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        if (obj.ndim == 1):
                            kwargs[k] = obj.iloc[b.mgr_locs.indexer]._values
                        else:
                            kwargs[k] = obj.iloc[:, b.mgr_locs.indexer]._values
                    else:
                        kwargs[k] = obj[b.mgr_locs.indexer]
            try:
                if callable(f):
                    applied = b.apply(f, **kwargs)
                else:
                    applied = getattr(b, f)(**kwargs)
            except (TypeError, NotImplementedError):
                if (not ignore_failures):
                    raise
                continue
            result_blocks = extend_blocks(applied, result_blocks)
        if ignore_failures:
            return self._combine(result_blocks)
        if (len(result_blocks) == 0):
            return self.make_empty(self.axes)
        return type(self).from_blocks(result_blocks, self.axes)

    def quantile(self, axis=0, transposed=False, interpolation='linear', qs=None, numeric_only=None):
        "\n        Iterate over blocks applying quantile reduction.\n        This routine is intended for reduction type operations and\n        will do inference on the generated blocks.\n\n        Parameters\n        ----------\n        axis: reduction axis, default 0\n        consolidate: bool, default True. Join together blocks having same\n            dtype\n        transposed: bool, default False\n            we are holding transposed data\n        interpolation : type of interpolation, default 'linear'\n        qs : a scalar or list of the quantiles to be computed\n        numeric_only : ignored\n\n        Returns\n        -------\n        BlockManager\n        "
        assert (self.ndim >= 2)

        def get_axe(block, qs, axes):
            from pandas import Float64Index
            if is_list_like(qs):
                ax = Float64Index(qs)
            else:
                ax = axes[0]
            return ax
        (axes, blocks) = ([], [])
        for b in self.blocks:
            block = b.quantile(axis=axis, qs=qs, interpolation=interpolation)
            axe = get_axe(b, qs, axes=self.axes)
            axes.append(axe)
            blocks.append(block)
        ndim = {b.ndim for b in blocks}
        assert (0 not in ndim), ndim
        if (2 in ndim):
            new_axes = list(self.axes)
            if (len(blocks) > 1):
                new_axes[1] = axes[0]
                for (b, sb) in zip(blocks, self.blocks):
                    b.mgr_locs = sb.mgr_locs
            else:
                new_axes[axis] = Index(np.concatenate([ax._values for ax in axes]))
            if transposed:
                new_axes = new_axes[::(- 1)]
                blocks = [b.make_block(b.values.T, placement=np.arange(b.shape[1])) for b in blocks]
            return type(self)(blocks, new_axes)
        values = concat_compat([b.values for b in blocks])
        if (len(self.blocks) > 1):
            indexer = np.empty(len(self.axes[0]), dtype=np.intp)
            i = 0
            for b in self.blocks:
                for j in b.mgr_locs:
                    indexer[j] = i
                    i = (i + 1)
            values = values.take(indexer)
        return SingleBlockManager(make_block(values, ndim=1, placement=np.arange(len(values))), axes[0])

    def isna(self, func):
        return self.apply('apply', func=func)

    def where(self, other, cond, align, errors, axis):
        if align:
            align_keys = ['other', 'cond']
        else:
            align_keys = ['cond']
            other = extract_array(other, extract_numpy=True)
        return self.apply('where', align_keys=align_keys, other=other, cond=cond, errors=errors, axis=axis)

    def setitem(self, indexer, value):
        return self.apply('setitem', indexer=indexer, value=value)

    def putmask(self, mask, new, align=True, axis=0):
        if align:
            align_keys = ['new', 'mask']
        else:
            align_keys = ['mask']
            new = extract_array(new, extract_numpy=True)
        return self.apply('putmask', align_keys=align_keys, mask=mask, new=new, axis=axis)

    def diff(self, n, axis):
        return self.apply('diff', n=n, axis=axis)

    def interpolate(self, **kwargs):
        return self.apply('interpolate', **kwargs)

    def shift(self, periods, axis, fill_value):
        if (fill_value is lib.no_default):
            fill_value = None
        if ((axis == 0) and (self.ndim == 2) and (self.nblocks > 1)):
            ncols = self.shape[0]
            if (periods > 0):
                indexer = (([(- 1)] * periods) + list(range((ncols - periods))))
            else:
                nper = abs(periods)
                indexer = (list(range(nper, ncols)) + ([(- 1)] * nper))
            result = self.reindex_indexer(self.items, indexer, axis=0, fill_value=fill_value, allow_dups=True, consolidate=False)
            return result
        return self.apply('shift', periods=periods, axis=axis, fill_value=fill_value)

    def fillna(self, value, limit, inplace, downcast):
        return self.apply('fillna', value=value, limit=limit, inplace=inplace, downcast=downcast)

    def downcast(self):
        return self.apply('downcast')

    def astype(self, dtype, copy=False, errors='raise'):
        return self.apply('astype', dtype=dtype, copy=copy, errors=errors)

    def convert(self, copy=True, datetime=True, numeric=True, timedelta=True):
        return self.apply('convert', copy=copy, datetime=datetime, numeric=numeric, timedelta=timedelta)

    def replace(self, to_replace, value, inplace, regex):
        assert (np.ndim(value) == 0), value
        return self.apply('replace', to_replace=to_replace, value=value, inplace=inplace, regex=regex)

    def replace_list(self, src_list, dest_list, inplace=False, regex=False):
        ' do a list replace '
        inplace = validate_bool_kwarg(inplace, 'inplace')
        bm = self.apply('_replace_list', src_list=src_list, dest_list=dest_list, inplace=inplace, regex=regex)
        bm._consolidate_inplace()
        return bm

    def to_native_types(self, **kwargs):
        '\n        Convert values to native types (strings / python objects) that are used\n        in formatting (repr / csv).\n        '
        return self.apply('to_native_types', **kwargs)

    def is_consolidated(self):
        '\n        Return True if more than one block with the same dtype\n        '
        if (not self._known_consolidated):
            self._consolidate_check()
        return self._is_consolidated

    def _consolidate_check(self):
        dtypes = [blk.dtype for blk in self.blocks if blk._can_consolidate]
        self._is_consolidated = (len(dtypes) == len(set(dtypes)))
        self._known_consolidated = True

    @property
    def is_numeric_mixed_type(self):
        return all((block.is_numeric for block in self.blocks))

    @property
    def any_extension_types(self):
        'Whether any of the blocks in this manager are extension blocks'
        return any((block.is_extension for block in self.blocks))

    @property
    def is_view(self):
        ' return a boolean if we are a single block and are a view '
        if (len(self.blocks) == 1):
            return self.blocks[0].is_view
        return False

    def get_bool_data(self, copy=False):
        '\n        Select blocks that are bool-dtype and columns from object-dtype blocks\n        that are all-bool.\n\n        Parameters\n        ----------\n        copy : bool, default False\n            Whether to copy the blocks\n        '
        new_blocks = []
        for blk in self.blocks:
            if (blk.dtype == bool):
                new_blocks.append(blk)
            elif blk.is_object:
                nbs = blk._split()
                for nb in nbs:
                    if nb.is_bool:
                        new_blocks.append(nb)
        return self._combine(new_blocks, copy)

    def get_numeric_data(self, copy=False):
        '\n        Parameters\n        ----------\n        copy : bool, default False\n            Whether to copy the blocks\n        '
        return self._combine([b for b in self.blocks if b.is_numeric], copy)

    def _combine(self, blocks, copy=True, index=None):
        ' return a new manager with the blocks '
        if (len(blocks) == 0):
            return self.make_empty()
        indexer = np.sort(np.concatenate([b.mgr_locs.as_array for b in blocks]))
        inv_indexer = lib.get_reverse_indexer(indexer, self.shape[0])
        new_blocks: List[Block] = []
        for b in blocks:
            b = b.copy(deep=copy)
            b.mgr_locs = inv_indexer[b.mgr_locs.indexer]
            new_blocks.append(b)
        axes = list(self.axes)
        if (index is not None):
            axes[(- 1)] = index
        axes[0] = self.items.take(indexer)
        return type(self).from_blocks(new_blocks, axes)

    def get_slice(self, slobj, axis=0):
        if (axis == 0):
            new_blocks = self._slice_take_blocks_ax0(slobj)
        elif (axis == 1):
            slicer = (slice(None), slobj)
            new_blocks = [blk.getitem_block(slicer) for blk in self.blocks]
        else:
            raise IndexError('Requested axis not found in manager')
        new_axes = list(self.axes)
        new_axes[axis] = new_axes[axis][slobj]
        bm = type(self)(new_blocks, new_axes, do_integrity_check=False)
        return bm

    @property
    def nblocks(self):
        return len(self.blocks)

    def copy(self, deep=True):
        "\n        Make deep or shallow copy of BlockManager\n\n        Parameters\n        ----------\n        deep : bool or string, default True\n            If False, return shallow copy (do not copy data)\n            If 'all', copy data and a deep copy of the index\n\n        Returns\n        -------\n        BlockManager\n        "
        if deep:

            def copy_func(ax):
                return (ax.copy(deep=True) if (deep == 'all') else ax.view())
            new_axes = [copy_func(ax) for ax in self.axes]
        else:
            new_axes = list(self.axes)
        res = self.apply('copy', deep=deep)
        res.axes = new_axes
        return res

    def as_array(self, transpose=False, dtype=None, copy=False, na_value=lib.no_default):
        '\n        Convert the blockmanager data into an numpy array.\n\n        Parameters\n        ----------\n        transpose : bool, default False\n            If True, transpose the return array.\n        dtype : object, default None\n            Data type of the return array.\n        copy : bool, default False\n            If True then guarantee that a copy is returned. A value of\n            False does not guarantee that the underlying data is not\n            copied.\n        na_value : object, default lib.no_default\n            Value to be used as the missing value sentinel.\n\n        Returns\n        -------\n        arr : ndarray\n        '
        if (len(self.blocks) == 0):
            arr = np.empty(self.shape, dtype=float)
            return (arr.transpose() if transpose else arr)
        copy = (copy or (na_value is not lib.no_default))
        if self.is_single_block:
            blk = self.blocks[0]
            if blk.is_extension:
                arr = blk.values.to_numpy(dtype=dtype, na_value=na_value).reshape(blk.shape)
            else:
                arr = np.asarray(blk.get_values())
                if dtype:
                    arr = arr.astype(dtype, copy=False)
        else:
            arr = self._interleave(dtype=dtype, na_value=na_value)
            copy = False
        if copy:
            arr = arr.copy()
        if (na_value is not lib.no_default):
            arr[isna(arr)] = na_value
        return (arr.transpose() if transpose else arr)

    def _interleave(self, dtype=None, na_value=lib.no_default):
        '\n        Return ndarray from blocks with specified item order\n        Items must be contained in the blocks\n        '
        if (not dtype):
            dtype = _interleaved_dtype(self.blocks)
        if isinstance(dtype, SparseDtype):
            dtype = dtype.subtype
        elif is_extension_array_dtype(dtype):
            dtype = 'object'
        elif is_dtype_equal(dtype, str):
            dtype = 'object'
        result = np.empty(self.shape, dtype=dtype)
        itemmask = np.zeros(self.shape[0])
        for blk in self.blocks:
            rl = blk.mgr_locs
            if blk.is_extension:
                arr = blk.values.to_numpy(dtype=dtype, na_value=na_value)
            else:
                arr = blk.get_values(dtype)
            result[rl.indexer] = arr
            itemmask[rl.indexer] = 1
        if (not itemmask.all()):
            raise AssertionError('Some items were not contained in blocks')
        return result

    def to_dict(self, copy=True):
        '\n        Return a dict of str(dtype) -> BlockManager\n\n        Parameters\n        ----------\n        copy : bool, default True\n\n        Returns\n        -------\n        values : a dict of dtype -> BlockManager\n        '
        bd: Dict[(str, List[Block])] = {}
        for b in self.blocks:
            bd.setdefault(str(b.dtype), []).append(b)
        return {dtype: self._combine(blocks, copy=copy) for (dtype, blocks) in bd.items()}

    def fast_xs(self, loc):
        '\n        Return the array corresponding to `frame.iloc[loc]`.\n\n        Parameters\n        ----------\n        loc : int\n\n        Returns\n        -------\n        np.ndarray or ExtensionArray\n        '
        if (len(self.blocks) == 1):
            return self.blocks[0].iget((slice(None), loc))
        dtype = _interleaved_dtype(self.blocks)
        n = len(self)
        if is_extension_array_dtype(dtype):
            result = np.empty(n, dtype=object)
        else:
            result = np.empty(n, dtype=dtype)
        for blk in self.blocks:
            for (i, rl) in enumerate(blk.mgr_locs):
                result[rl] = blk.iget((i, loc))
        if isinstance(dtype, ExtensionDtype):
            result = dtype.construct_array_type()._from_sequence(result, dtype=dtype)
        return result

    def consolidate(self):
        '\n        Join together blocks having same dtype\n\n        Returns\n        -------\n        y : BlockManager\n        '
        if self.is_consolidated():
            return self
        bm = type(self)(self.blocks, self.axes)
        bm._is_consolidated = False
        bm._consolidate_inplace()
        return bm

    def _consolidate_inplace(self):
        if (not self.is_consolidated()):
            self.blocks = tuple(_consolidate(self.blocks))
            self._is_consolidated = True
            self._known_consolidated = True
            self._rebuild_blknos_and_blklocs()

    def iget(self, i):
        '\n        Return the data as a SingleBlockManager.\n        '
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        return SingleBlockManager(block.make_block_same_class(values, placement=slice(0, len(values)), ndim=1), self.axes[1])

    def iget_values(self, i):
        '\n        Return the data for column i as the values (ndarray or ExtensionArray).\n        '
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        return values

    def idelete(self, indexer):
        '\n        Delete selected locations in-place (new block and array, same BlockManager)\n        '
        is_deleted = np.zeros(self.shape[0], dtype=np.bool_)
        is_deleted[indexer] = True
        ref_loc_offset = (- is_deleted.cumsum())
        is_blk_deleted = ([False] * len(self.blocks))
        if isinstance(indexer, int):
            affected_start = indexer
        else:
            affected_start = is_deleted.nonzero()[0][0]
        for (blkno, _) in _fast_count_smallints(self.blknos[affected_start:]):
            blk = self.blocks[blkno]
            bml = blk.mgr_locs
            blk_del = is_deleted[bml.indexer].nonzero()[0]
            if (len(blk_del) == len(bml)):
                is_blk_deleted[blkno] = True
                continue
            elif (len(blk_del) != 0):
                blk.delete(blk_del)
                bml = blk.mgr_locs
            blk.mgr_locs = bml.add(ref_loc_offset[bml.indexer])
        self.axes[0] = self.items[(~ is_deleted)]
        self.blocks = tuple((b for (blkno, b) in enumerate(self.blocks) if (not is_blk_deleted[blkno])))
        self._rebuild_blknos_and_blklocs()

    def iset(self, loc, value):
        '\n        Set new item in-place. Does not consolidate. Adds new Block if not\n        contained in the current set of items\n        '
        value = extract_array(value, extract_numpy=True)
        if ((self._blklocs is None) and (self.ndim > 1)):
            self._rebuild_blknos_and_blklocs()
        value_is_extension_type = is_extension_array_dtype(value)
        if value_is_extension_type:

            def value_getitem(placement):
                return value
        else:
            if (value.ndim == (self.ndim - 1)):
                value = safe_reshape(value, ((1,) + value.shape))

                def value_getitem(placement):
                    return value
            else:

                def value_getitem(placement):
                    return value[placement.indexer]
            if (value.shape[1:] != self.shape[1:]):
                raise AssertionError('Shape of new values must be compatible with manager shape')
        if lib.is_integer(loc):
            loc = [loc]
        blknos = self.blknos[loc]
        blklocs = self.blklocs[loc].copy()
        unfit_mgr_locs = []
        unfit_val_locs = []
        removed_blknos = []
        for (blkno, val_locs) in libinternals.get_blkno_placements(blknos, group=True):
            blk = self.blocks[blkno]
            blk_locs = blklocs[val_locs.indexer]
            if blk.should_store(value):
                blk.set_inplace(blk_locs, value_getitem(val_locs))
            else:
                unfit_mgr_locs.append(blk.mgr_locs.as_array[blk_locs])
                unfit_val_locs.append(val_locs)
                if (len(val_locs) == len(blk.mgr_locs)):
                    removed_blknos.append(blkno)
                else:
                    blk.delete(blk_locs)
                    self._blklocs[blk.mgr_locs.indexer] = np.arange(len(blk))
        if len(removed_blknos):
            is_deleted = np.zeros(self.nblocks, dtype=np.bool_)
            is_deleted[removed_blknos] = True
            new_blknos = np.empty(self.nblocks, dtype=np.int64)
            new_blknos.fill((- 1))
            new_blknos[(~ is_deleted)] = np.arange((self.nblocks - len(removed_blknos)))
            self._blknos = new_blknos[self._blknos]
            self.blocks = tuple((blk for (i, blk) in enumerate(self.blocks) if (i not in set(removed_blknos))))
        if unfit_val_locs:
            unfit_mgr_locs = np.concatenate(unfit_mgr_locs)
            unfit_count = len(unfit_mgr_locs)
            new_blocks: List[Block] = []
            if value_is_extension_type:
                new_blocks.extend((make_block(values=value, ndim=self.ndim, placement=slice(mgr_loc, (mgr_loc + 1))) for mgr_loc in unfit_mgr_locs))
                self._blknos[unfit_mgr_locs] = (np.arange(unfit_count) + len(self.blocks))
                self._blklocs[unfit_mgr_locs] = 0
            else:
                unfit_val_items = unfit_val_locs[0].append(unfit_val_locs[1:])
                new_blocks.append(make_block(values=value_getitem(unfit_val_items), ndim=self.ndim, placement=unfit_mgr_locs))
                self._blknos[unfit_mgr_locs] = len(self.blocks)
                self._blklocs[unfit_mgr_locs] = np.arange(unfit_count)
            self.blocks += tuple(new_blocks)
            self._known_consolidated = False

    def insert(self, loc, item, value, allow_duplicates=False):
        '\n        Insert item at selected position.\n\n        Parameters\n        ----------\n        loc : int\n        item : hashable\n        value : array_like\n        allow_duplicates: bool\n            If False, trying to insert non-unique item will raise\n\n        '
        if ((not allow_duplicates) and (item in self.items)):
            raise ValueError(f'cannot insert {item}, already exists')
        if (not isinstance(loc, int)):
            raise TypeError('loc must be int')
        new_axis = self.items.insert(loc, item)
        if ((value.ndim == (self.ndim - 1)) and (not is_extension_array_dtype(value.dtype))):
            value = safe_reshape(value, ((1,) + value.shape))
        block = make_block(values=value, ndim=self.ndim, placement=slice(loc, (loc + 1)))
        for (blkno, count) in _fast_count_smallints(self.blknos[loc:]):
            blk = self.blocks[blkno]
            if (count == len(blk.mgr_locs)):
                blk.mgr_locs = blk.mgr_locs.add(1)
            else:
                new_mgr_locs = blk.mgr_locs.as_array.copy()
                new_mgr_locs[(new_mgr_locs >= loc)] += 1
                blk.mgr_locs = new_mgr_locs
        if (loc == self.blklocs.shape[0]):
            self._blklocs = np.append(self._blklocs, 0)
            self._blknos = np.append(self._blknos, len(self.blocks))
        else:
            self._blklocs = np.insert(self._blklocs, loc, 0)
            self._blknos = np.insert(self._blknos, loc, len(self.blocks))
        self.axes[0] = new_axis
        self.blocks += (block,)
        self._known_consolidated = False
        if (len(self.blocks) > 100):
            warnings.warn('DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider using pd.concat instead.  To get a de-fragmented frame, use `newframe = frame.copy()`', PerformanceWarning, stacklevel=5)

    def reindex_axis(self, new_index, axis, method=None, limit=None, fill_value=None, copy=True, consolidate=True, only_slice=False):
        '\n        Conform block manager to new index.\n        '
        new_index = ensure_index(new_index)
        (new_index, indexer) = self.axes[axis].reindex(new_index, method=method, limit=limit)
        return self.reindex_indexer(new_index, indexer, axis=axis, fill_value=fill_value, copy=copy, consolidate=consolidate, only_slice=only_slice)

    def reindex_indexer(self, new_axis, indexer, axis, fill_value=None, allow_dups=False, copy=True, consolidate=True, only_slice=False):
        "\n        Parameters\n        ----------\n        new_axis : Index\n        indexer : ndarray of int64 or None\n        axis : int\n        fill_value : object, default None\n        allow_dups : bool, default False\n        copy : bool, default True\n        consolidate: bool, default True\n            Whether to consolidate inplace before reindexing.\n        only_slice : bool, default False\n            Whether to take views, not copies, along columns.\n\n        pandas-indexer with -1's only.\n        "
        if (indexer is None):
            if ((new_axis is self.axes[axis]) and (not copy)):
                return self
            result = self.copy(deep=copy)
            result.axes = list(self.axes)
            result.axes[axis] = new_axis
            return result
        if consolidate:
            self._consolidate_inplace()
        if (not allow_dups):
            self.axes[axis]._can_reindex(indexer)
        if (axis >= self.ndim):
            raise IndexError('Requested axis not found in manager')
        if (axis == 0):
            new_blocks = self._slice_take_blocks_ax0(indexer, fill_value=fill_value, only_slice=only_slice)
        else:
            new_blocks = [blk.take_nd(indexer, axis=axis, fill_value=(fill_value if (fill_value is not None) else blk.fill_value)) for blk in self.blocks]
        new_axes = list(self.axes)
        new_axes[axis] = new_axis
        return type(self).from_blocks(new_blocks, new_axes)

    def _slice_take_blocks_ax0(self, slice_or_indexer, fill_value=lib.no_default, only_slice=False):
        '\n        Slice/take blocks along axis=0.\n\n        Overloaded for SingleBlock\n\n        Parameters\n        ----------\n        slice_or_indexer : slice, ndarray[bool], or list-like of ints\n        fill_value : scalar, default lib.no_default\n        only_slice : bool, default False\n            If True, we always return views on existing arrays, never copies.\n            This is used when called from ops.blockwise.operate_blockwise.\n\n        Returns\n        -------\n        new_blocks : list of Block\n        '
        allow_fill = (fill_value is not lib.no_default)
        (sl_type, slobj, sllen) = _preprocess_slice_or_indexer(slice_or_indexer, self.shape[0], allow_fill=allow_fill)
        if self.is_single_block:
            blk = self.blocks[0]
            if (sl_type in ('slice', 'mask')):
                if (sllen == 0):
                    return []
                return [blk.getitem_block(slobj, new_mgr_locs=slice(0, sllen))]
            elif ((not allow_fill) or (self.ndim == 1)):
                if (allow_fill and (fill_value is None)):
                    (_, fill_value) = maybe_promote(blk.dtype)
                if ((not allow_fill) and only_slice):
                    blocks = [blk.getitem_block([ml], new_mgr_locs=i) for (i, ml) in enumerate(slobj)]
                    return blocks
                else:
                    return [blk.take_nd(slobj, axis=0, new_mgr_locs=slice(0, sllen), fill_value=fill_value)]
        if (sl_type in ('slice', 'mask')):
            blknos = self.blknos[slobj]
            blklocs = self.blklocs[slobj]
        else:
            blknos = algos.take_1d(self.blknos, slobj, fill_value=(- 1), allow_fill=allow_fill)
            blklocs = algos.take_1d(self.blklocs, slobj, fill_value=(- 1), allow_fill=allow_fill)
        blocks = []
        group = (not only_slice)
        for (blkno, mgr_locs) in libinternals.get_blkno_placements(blknos, group=group):
            if (blkno == (- 1)):
                blocks.append(self._make_na_block(placement=mgr_locs, fill_value=fill_value))
            else:
                blk = self.blocks[blkno]
                if (not blk._can_consolidate):
                    for mgr_loc in mgr_locs:
                        newblk = blk.copy(deep=False)
                        newblk.mgr_locs = slice(mgr_loc, (mgr_loc + 1))
                        blocks.append(newblk)
                else:
                    taker = blklocs[mgr_locs.indexer]
                    max_len = max(len(mgr_locs), (taker.max() + 1))
                    if only_slice:
                        taker = lib.maybe_indices_to_slice(taker, max_len)
                    if isinstance(taker, slice):
                        nb = blk.getitem_block(taker, new_mgr_locs=mgr_locs)
                        blocks.append(nb)
                    elif only_slice:
                        for (i, ml) in zip(taker, mgr_locs):
                            nb = blk.getitem_block([i], new_mgr_locs=ml)
                            blocks.append(nb)
                    else:
                        nb = blk.take_nd(taker, axis=0, new_mgr_locs=mgr_locs)
                        blocks.append(nb)
        return blocks

    def _make_na_block(self, placement, fill_value=None):
        if (fill_value is None):
            fill_value = np.nan
        block_shape = list(self.shape)
        block_shape[0] = len(placement)
        (dtype, fill_value) = infer_dtype_from_scalar(fill_value)
        block_values = np.empty(block_shape, dtype=dtype)
        block_values.fill(fill_value)
        return make_block(block_values, placement=placement, ndim=block_values.ndim)

    def take(self, indexer, axis=1, verify=True, convert=True):
        '\n        Take items along any axis.\n        '
        indexer = (np.arange(indexer.start, indexer.stop, indexer.step, dtype='int64') if isinstance(indexer, slice) else np.asanyarray(indexer, dtype='int64'))
        n = self.shape[axis]
        if convert:
            indexer = maybe_convert_indices(indexer, n)
        if verify:
            if ((indexer == (- 1)) | (indexer >= n)).any():
                raise Exception('Indices must be nonzero and less than the axis length')
        new_labels = self.axes[axis].take(indexer)
        return self.reindex_indexer(new_axis=new_labels, indexer=indexer, axis=axis, allow_dups=True, consolidate=False)

    def equals(self, other):
        if (not isinstance(other, BlockManager)):
            return False
        (self_axes, other_axes) = (self.axes, other.axes)
        if (len(self_axes) != len(other_axes)):
            return False
        if (not all((ax1.equals(ax2) for (ax1, ax2) in zip(self_axes, other_axes)))):
            return False
        if (self.ndim == 1):
            if (other.ndim != 1):
                return False
            left = self.blocks[0].values
            right = other.blocks[0].values
            return array_equals(left, right)
        return blockwise_all(self, other, array_equals)

    def unstack(self, unstacker, fill_value):
        '\n        Return a BlockManager with all blocks unstacked..\n\n        Parameters\n        ----------\n        unstacker : reshape._Unstacker\n        fill_value : Any\n            fill_value for newly introduced missing values.\n\n        Returns\n        -------\n        unstacked : BlockManager\n        '
        new_columns = unstacker.get_new_columns(self.items)
        new_index = unstacker.new_index
        new_blocks: List[Block] = []
        columns_mask: List[np.ndarray] = []
        for blk in self.blocks:
            blk_cols = self.items[blk.mgr_locs.indexer]
            new_items = unstacker.get_new_columns(blk_cols)
            new_placement = new_columns.get_indexer(new_items)
            (blocks, mask) = blk._unstack(unstacker, fill_value, new_placement=new_placement)
            new_blocks.extend(blocks)
            columns_mask.extend(mask)
        new_columns = new_columns[columns_mask]
        bm = BlockManager(new_blocks, [new_columns, new_index])
        return bm

class SingleBlockManager(BlockManager):
    ' manage a single block with '
    ndim = 1
    _is_consolidated = True
    _known_consolidated = True
    __slots__ = ()
    is_single_block = True

    def __init__(self, block, axis, do_integrity_check=False, fastpath=lib.no_default):
        assert isinstance(block, Block), type(block)
        assert isinstance(axis, Index), type(axis)
        if (fastpath is not lib.no_default):
            warnings.warn('The `fastpath` keyword is deprecated and will be removed in a future version.', FutureWarning, stacklevel=2)
        self.axes = [axis]
        self.blocks = (block,)

    @classmethod
    def from_blocks(cls, blocks, axes):
        '\n        Constructor for BlockManager and SingleBlockManager with same signature.\n        '
        assert (len(blocks) == 1)
        assert (len(axes) == 1)
        return cls(blocks[0], axes[0], do_integrity_check=False)

    @classmethod
    def from_array(cls, array, index):
        '\n        Constructor for if we have an array that is not yet a Block.\n        '
        block = make_block(array, placement=slice(0, len(index)), ndim=1)
        return cls(block, index)

    def _post_setstate(self):
        pass

    @property
    def _block(self):
        return self.blocks[0]

    @property
    def _blknos(self):
        ' compat with BlockManager '
        return None

    @property
    def _blklocs(self):
        ' compat with BlockManager '
        return None

    def get_slice(self, slobj, axis=0):
        if (axis >= self.ndim):
            raise IndexError('Requested axis not found in manager')
        blk = self._block
        array = blk._slice(slobj)
        block = blk.make_block_same_class(array, placement=slice(0, len(array)))
        return type(self)(block, self.index[slobj])

    @property
    def index(self):
        return self.axes[0]

    @property
    def dtype(self):
        return self._block.dtype

    def get_dtypes(self):
        return np.array([self._block.dtype])

    def external_values(self):
        'The array that Series.values returns'
        return self._block.external_values()

    def internal_values(self):
        'The array that Series._values returns'
        return self._block.internal_values()

    @property
    def _can_hold_na(self):
        return self._block._can_hold_na

    def is_consolidated(self):
        return True

    def _consolidate_check(self):
        pass

    def _consolidate_inplace(self):
        pass

    def idelete(self, indexer):
        "\n        Delete single location from SingleBlockManager.\n\n        Ensures that self.blocks doesn't become empty.\n        "
        self._block.delete(indexer)
        self.axes[0] = self.axes[0].delete(indexer)

    def fast_xs(self, loc):
        '\n        fast path for getting a cross-section\n        return a view of the data\n        '
        raise NotImplementedError('Use series._values[loc] instead')

def create_block_manager_from_blocks(blocks, axes):
    try:
        if ((len(blocks) == 1) and (not isinstance(blocks[0], Block))):
            if (not len(blocks[0])):
                blocks = []
            else:
                blocks = [make_block(values=blocks[0], placement=slice(0, len(axes[0])), ndim=2)]
        mgr = BlockManager(blocks, axes)
        mgr._consolidate_inplace()
        return mgr
    except ValueError as e:
        blocks = [getattr(b, 'values', b) for b in blocks]
        tot_items = sum((b.shape[0] for b in blocks))
        raise construction_error(tot_items, blocks[0].shape[1:], axes, e)

def create_block_manager_from_arrays(arrays, names, axes):
    assert isinstance(names, Index)
    assert isinstance(axes, list)
    assert all((isinstance(x, Index) for x in axes))
    arrays = [(x if (not isinstance(x, ABCPandasArray)) else x.to_numpy()) for x in arrays]
    try:
        blocks = _form_blocks(arrays, names, axes)
        mgr = BlockManager(blocks, axes)
        mgr._consolidate_inplace()
        return mgr
    except ValueError as e:
        raise construction_error(len(arrays), arrays[0].shape, axes, e)

def construction_error(tot_items, block_shape, axes, e=None):
    ' raise a helpful message about our construction '
    passed = tuple(map(int, ([tot_items] + list(block_shape))))
    if (len(passed) <= 2):
        passed = passed[::(- 1)]
    implied = tuple((len(ax) for ax in axes))
    if (len(implied) <= 2):
        implied = implied[::(- 1)]
    if ((passed == implied) and (e is not None)):
        return e
    if (block_shape[0] == 0):
        return ValueError('Empty data passed with indices specified.')
    return ValueError(f'Shape of passed values is {passed}, indices imply {implied}')

def _form_blocks(arrays, names, axes):
    items_dict: DefaultDict[(str, List)] = defaultdict(list)
    extra_locs = []
    names_idx = names
    if names_idx.equals(axes[0]):
        names_indexer = np.arange(len(names_idx))
    else:
        assert names_idx.intersection(axes[0]).is_unique
        names_indexer = names_idx.get_indexer_for(axes[0])
    for (i, name_idx) in enumerate(names_indexer):
        if (name_idx == (- 1)):
            extra_locs.append(i)
            continue
        k = names[name_idx]
        v = arrays[name_idx]
        block_type = get_block_type(v)
        items_dict[block_type.__name__].append((i, k, v))
    blocks: List[Block] = []
    if len(items_dict['FloatBlock']):
        float_blocks = _multi_blockify(items_dict['FloatBlock'])
        blocks.extend(float_blocks)
    if len(items_dict['ComplexBlock']):
        complex_blocks = _multi_blockify(items_dict['ComplexBlock'])
        blocks.extend(complex_blocks)
    if len(items_dict['TimeDeltaBlock']):
        timedelta_blocks = _multi_blockify(items_dict['TimeDeltaBlock'])
        blocks.extend(timedelta_blocks)
    if len(items_dict['IntBlock']):
        int_blocks = _multi_blockify(items_dict['IntBlock'])
        blocks.extend(int_blocks)
    if len(items_dict['DatetimeBlock']):
        datetime_blocks = _simple_blockify(items_dict['DatetimeBlock'], DT64NS_DTYPE)
        blocks.extend(datetime_blocks)
    if len(items_dict['DatetimeTZBlock']):
        dttz_blocks = [make_block(array, klass=DatetimeTZBlock, placement=i, ndim=2) for (i, _, array) in items_dict['DatetimeTZBlock']]
        blocks.extend(dttz_blocks)
    if len(items_dict['BoolBlock']):
        bool_blocks = _simple_blockify(items_dict['BoolBlock'], np.bool_)
        blocks.extend(bool_blocks)
    if (len(items_dict['ObjectBlock']) > 0):
        object_blocks = _simple_blockify(items_dict['ObjectBlock'], np.object_)
        blocks.extend(object_blocks)
    if (len(items_dict['CategoricalBlock']) > 0):
        cat_blocks = [make_block(array, klass=CategoricalBlock, placement=i, ndim=2) for (i, _, array) in items_dict['CategoricalBlock']]
        blocks.extend(cat_blocks)
    if len(items_dict['ExtensionBlock']):
        external_blocks = [make_block(array, klass=ExtensionBlock, placement=i, ndim=2) for (i, _, array) in items_dict['ExtensionBlock']]
        blocks.extend(external_blocks)
    if len(items_dict['ObjectValuesExtensionBlock']):
        external_blocks = [make_block(array, klass=ObjectValuesExtensionBlock, placement=i, ndim=2) for (i, _, array) in items_dict['ObjectValuesExtensionBlock']]
        blocks.extend(external_blocks)
    if len(extra_locs):
        shape = ((len(extra_locs),) + tuple((len(x) for x in axes[1:])))
        block_values = np.empty(shape, dtype=object)
        block_values.fill(np.nan)
        na_block = make_block(block_values, placement=extra_locs, ndim=2)
        blocks.append(na_block)
    return blocks

def _simple_blockify(tuples, dtype):
    '\n    return a single array of a block that has a single dtype; if dtype is\n    not None, coerce to this dtype\n    '
    (values, placement) = _stack_arrays(tuples, dtype)
    if ((dtype is not None) and (values.dtype != dtype)):
        values = values.astype(dtype)
    block = make_block(values, placement=placement, ndim=2)
    return [block]

def _multi_blockify(tuples, dtype=None):
    ' return an array of blocks that potentially have different dtypes '
    grouper = itertools.groupby(tuples, (lambda x: x[2].dtype))
    new_blocks = []
    for (dtype, tup_block) in grouper:
        (values, placement) = _stack_arrays(list(tup_block), dtype)
        block = make_block(values, placement=placement, ndim=2)
        new_blocks.append(block)
    return new_blocks

def _stack_arrays(tuples, dtype):

    def _asarray_compat(x):
        if isinstance(x, ABCSeries):
            return x._values
        else:
            return np.asarray(x)

    def _shape_compat(x) -> Shape:
        if isinstance(x, ABCSeries):
            return (len(x),)
        else:
            return x.shape
    (placement, names, arrays) = zip(*tuples)
    first = arrays[0]
    shape = ((len(arrays),) + _shape_compat(first))
    stacked = np.empty(shape, dtype=dtype)
    for (i, arr) in enumerate(arrays):
        stacked[i] = _asarray_compat(arr)
    return (stacked, placement)

def _interleaved_dtype(blocks):
    '\n    Find the common dtype for `blocks`.\n\n    Parameters\n    ----------\n    blocks : List[Block]\n\n    Returns\n    -------\n    dtype : np.dtype, ExtensionDtype, or None\n        None is returned when `blocks` is empty.\n    '
    if (not len(blocks)):
        return None
    return find_common_type([b.dtype for b in blocks])

def _consolidate(blocks):
    '\n    Merge blocks having same dtype, exclude non-consolidating blocks\n    '
    gkey = (lambda x: x._consolidate_key)
    grouper = itertools.groupby(sorted(blocks, key=gkey), gkey)
    new_blocks: List[Block] = []
    for ((_can_consolidate, dtype), group_blocks) in grouper:
        merged_blocks = _merge_blocks(list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate)
        new_blocks.extend(merged_blocks)
    return new_blocks

def _merge_blocks(blocks, dtype, can_consolidate):
    if (len(blocks) == 1):
        return blocks
    if can_consolidate:
        if (dtype is None):
            if (len({b.dtype for b in blocks}) != 1):
                raise AssertionError('_merge_blocks are invalid!')
        new_mgr_locs = np.concatenate([b.mgr_locs.as_array for b in blocks])
        new_values = np.vstack([b.values for b in blocks])
        argsort = np.argsort(new_mgr_locs)
        new_values = new_values[argsort]
        new_mgr_locs = new_mgr_locs[argsort]
        return [make_block(new_values, placement=new_mgr_locs, ndim=2)]
    return blocks

def _fast_count_smallints(arr):
    'Faster version of set(arr) for sequences of small numbers.'
    counts = np.bincount(arr.astype(np.int_))
    nz = counts.nonzero()[0]
    return np.c_[(nz, counts[nz])]

def _preprocess_slice_or_indexer(slice_or_indexer, length, allow_fill):
    if isinstance(slice_or_indexer, slice):
        return ('slice', slice_or_indexer, libinternals.slice_len(slice_or_indexer, length))
    elif (isinstance(slice_or_indexer, np.ndarray) and (slice_or_indexer.dtype == np.bool_)):
        return ('mask', slice_or_indexer, slice_or_indexer.sum())
    else:
        indexer = np.asanyarray(slice_or_indexer, dtype=np.int64)
        if (not allow_fill):
            indexer = maybe_convert_indices(indexer, length)
        return ('fancy', indexer, len(indexer))
