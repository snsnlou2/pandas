
from collections import defaultdict
import copy
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple, cast
import numpy as np
from pandas._libs import NaT, internals as libinternals
from pandas._typing import ArrayLike, DtypeObj, Shape
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import get_dtype, is_categorical_dtype, is_datetime64_dtype, is_datetime64tz_dtype, is_extension_array_dtype, is_float_dtype, is_numeric_dtype, is_sparse, is_timedelta64_dtype
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import isna_all
import pandas.core.algorithms as algos
from pandas.core.arrays import DatetimeArray, ExtensionArray
from pandas.core.internals.blocks import make_block
from pandas.core.internals.managers import BlockManager
if TYPE_CHECKING:
    from pandas import Index
    from pandas.core.arrays.sparse.dtype import SparseDtype

def concatenate_block_managers(mgrs_indexers, axes, concat_axis, copy):
    '\n    Concatenate block managers into one.\n\n    Parameters\n    ----------\n    mgrs_indexers : list of (BlockManager, {axis: indexer,...}) tuples\n    axes : list of Index\n    concat_axis : int\n    copy : bool\n\n    Returns\n    -------\n    BlockManager\n    '
    concat_plans = [_get_mgr_concatenation_plan(mgr, indexers) for (mgr, indexers) in mgrs_indexers]
    concat_plan = _combine_concat_plans(concat_plans, concat_axis)
    blocks = []
    for (placement, join_units) in concat_plan:
        if ((len(join_units) == 1) and (not join_units[0].indexers)):
            b = join_units[0].block
            values = b.values
            if copy:
                values = values.copy()
            else:
                values = values.view()
            b = b.make_block_same_class(values, placement=placement)
        elif _is_uniform_join_units(join_units):
            blk = join_units[0].block
            vals = [ju.block.values for ju in join_units]
            if (not blk.is_extension):
                values = np.concatenate(vals, axis=(blk.ndim - 1))
            else:
                values = concat_compat(vals)
                if (not isinstance(values, ExtensionArray)):
                    values = values.reshape(1, len(values))
            if (blk.values.dtype == values.dtype):
                b = blk.make_block_same_class(values, placement=placement)
            else:
                b = make_block(values, placement=placement, ndim=blk.ndim)
        else:
            b = make_block(_concatenate_join_units(join_units, concat_axis, copy=copy), placement=placement, ndim=len(axes))
        blocks.append(b)
    return BlockManager(blocks, axes)

def _get_mgr_concatenation_plan(mgr, indexers):
    '\n    Construct concatenation plan for given block manager and indexers.\n\n    Parameters\n    ----------\n    mgr : BlockManager\n    indexers : dict of {axis: indexer}\n\n    Returns\n    -------\n    plan : list of (BlockPlacement, JoinUnit) tuples\n\n    '
    mgr_shape_list = list(mgr.shape)
    for (ax, indexer) in indexers.items():
        mgr_shape_list[ax] = len(indexer)
    mgr_shape = tuple(mgr_shape_list)
    if (0 in indexers):
        ax0_indexer = indexers.pop(0)
        blknos = algos.take_1d(mgr.blknos, ax0_indexer, fill_value=(- 1))
        blklocs = algos.take_1d(mgr.blklocs, ax0_indexer, fill_value=(- 1))
    else:
        if mgr.is_single_block:
            blk = mgr.blocks[0]
            return [(blk.mgr_locs, JoinUnit(blk, mgr_shape, indexers))]
        ax0_indexer = None
        blknos = mgr.blknos
        blklocs = mgr.blklocs
    plan = []
    for (blkno, placements) in libinternals.get_blkno_placements(blknos, group=False):
        assert placements.is_slice_like
        join_unit_indexers = indexers.copy()
        shape_list = list(mgr_shape)
        shape_list[0] = len(placements)
        shape = tuple(shape_list)
        if (blkno == (- 1)):
            unit = JoinUnit(None, shape)
        else:
            blk = mgr.blocks[blkno]
            ax0_blk_indexer = blklocs[placements.indexer]
            unit_no_ax0_reindexing = ((len(placements) == len(blk.mgr_locs)) and (((ax0_indexer is None) and blk.mgr_locs.is_slice_like and (blk.mgr_locs.as_slice.step == 1)) or (np.diff(ax0_blk_indexer) == 1).all()))
            if unit_no_ax0_reindexing:
                join_unit_indexers.pop(0, None)
            else:
                join_unit_indexers[0] = ax0_blk_indexer
            unit = JoinUnit(blk, shape, join_unit_indexers)
        plan.append((placements, unit))
    return plan

class JoinUnit():

    def __init__(self, block, shape, indexers=None):
        if (indexers is None):
            indexers = {}
        self.block = block
        self.indexers = indexers
        self.shape = shape

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.block)}, {self.indexers})'

    @cache_readonly
    def needs_filling(self):
        for indexer in self.indexers.values():
            if (indexer == (- 1)).any():
                return True
        return False

    @cache_readonly
    def dtype(self):
        if (self.block is None):
            raise AssertionError('Block is None, no dtype')
        if (not self.needs_filling):
            return self.block.dtype
        else:
            return get_dtype(maybe_promote(self.block.dtype, self.block.fill_value)[0])

    @cache_readonly
    def is_na(self):
        if (self.block is None):
            return True
        if (not self.block._can_hold_na):
            return False
        values = self.block.values
        if is_sparse(self.block.values.dtype):
            return False
        elif self.block.is_extension:
            values_flat = values
        else:
            values_flat = values.ravel(order='K')
        return isna_all(values_flat)

    def get_reindexed_values(self, empty_dtype, upcasted_na):
        if (upcasted_na is None):
            fill_value = self.block.fill_value
            values = self.block.get_values()
        else:
            fill_value = upcasted_na
            if self.is_na:
                if getattr(self.block, 'is_object', False):
                    values = self.block.values.ravel(order='K')
                    if (len(values) and (values[0] is None)):
                        fill_value = None
                if (getattr(self.block, 'is_datetimetz', False) or is_datetime64tz_dtype(empty_dtype)):
                    if (self.block is None):
                        return DatetimeArray(np.full(self.shape[1], fill_value.value), dtype=empty_dtype)
                elif getattr(self.block, 'is_categorical', False):
                    pass
                elif getattr(self.block, 'is_extension', False):
                    pass
                elif is_extension_array_dtype(empty_dtype):
                    missing_arr = empty_dtype.construct_array_type()._from_sequence([], dtype=empty_dtype)
                    (ncols, nrows) = self.shape
                    assert (ncols == 1), ncols
                    empty_arr = ((- 1) * np.ones((nrows,), dtype=np.intp))
                    return missing_arr.take(empty_arr, allow_fill=True, fill_value=fill_value)
                else:
                    missing_arr = np.empty(self.shape, dtype=empty_dtype)
                    missing_arr.fill(fill_value)
                    return missing_arr
            if ((not self.indexers) and (not self.block._can_consolidate)):
                return self.block.values
            if (self.block.is_bool and (not self.block.is_categorical)):
                values = self.block.astype(np.object_).values
            elif self.block.is_extension:
                values = self.block.values
            else:
                values = self.block.values
        if (not self.indexers):
            values = values.view()
        else:
            for (ax, indexer) in self.indexers.items():
                values = algos.take_nd(values, indexer, axis=ax, fill_value=fill_value)
        return values

def _concatenate_join_units(join_units, concat_axis, copy):
    '\n    Concatenate values from several join units along selected axis.\n    '
    if ((concat_axis == 0) and (len(join_units) > 1)):
        raise AssertionError('Concatenating join units along axis0')
    (empty_dtype, upcasted_na) = _get_empty_dtype_and_na(join_units)
    to_concat = [ju.get_reindexed_values(empty_dtype=empty_dtype, upcasted_na=upcasted_na) for ju in join_units]
    if (len(to_concat) == 1):
        concat_values = to_concat[0]
        if copy:
            if isinstance(concat_values, np.ndarray):
                if (concat_values.base is not None):
                    concat_values = concat_values.copy()
            else:
                concat_values = concat_values.copy()
    elif any((isinstance(t, ExtensionArray) for t in to_concat)):
        to_concat = [(t if isinstance(t, ExtensionArray) else t[0, :]) for t in to_concat]
        concat_values = concat_compat(to_concat, axis=0)
        if ((not isinstance(concat_values, ExtensionArray)) or (isinstance(concat_values, DatetimeArray) and (concat_values.tz is None))):
            concat_values = np.atleast_2d(concat_values)
    else:
        concat_values = concat_compat(to_concat, axis=concat_axis)
    return concat_values

def _get_empty_dtype_and_na(join_units):
    '\n    Return dtype and N/A values to use when concatenating specified units.\n\n    Returned N/A value may be None which means there was no casting involved.\n\n    Returns\n    -------\n    dtype\n    na\n    '
    if (len(join_units) == 1):
        blk = join_units[0].block
        if (blk is None):
            return (np.dtype(np.float64), np.nan)
    if _is_uniform_reindex(join_units):
        empty_dtype = join_units[0].block.dtype
        upcasted_na = join_units[0].block.fill_value
        return (empty_dtype, upcasted_na)
    has_none_blocks = False
    dtypes = ([None] * len(join_units))
    for (i, unit) in enumerate(join_units):
        if (unit.block is None):
            has_none_blocks = True
        else:
            dtypes[i] = unit.dtype
    upcast_classes = _get_upcast_classes(join_units, dtypes)
    if ('extension' in upcast_classes):
        if (len(upcast_classes) == 1):
            cls = upcast_classes['extension'][0]
            return (cls, cls.na_value)
        else:
            return (np.dtype('object'), np.nan)
    elif ('object' in upcast_classes):
        return (np.dtype(np.object_), np.nan)
    elif ('bool' in upcast_classes):
        if has_none_blocks:
            return (np.dtype(np.object_), np.nan)
        else:
            return (np.dtype(np.bool_), None)
    elif ('category' in upcast_classes):
        return (np.dtype(np.object_), np.nan)
    elif ('datetimetz' in upcast_classes):
        dtype = upcast_classes['datetimetz']
        return (dtype[0], NaT)
    elif ('datetime' in upcast_classes):
        return (np.dtype('M8[ns]'), np.datetime64('NaT', 'ns'))
    elif ('timedelta' in upcast_classes):
        return (np.dtype('m8[ns]'), np.timedelta64('NaT', 'ns'))
    else:
        try:
            common_dtype = np.find_common_type(upcast_classes, [])
        except TypeError:
            return (np.dtype(np.object_), np.nan)
        else:
            if is_float_dtype(common_dtype):
                return (common_dtype, common_dtype.type(np.nan))
            elif is_numeric_dtype(common_dtype):
                if has_none_blocks:
                    return (np.dtype(np.float64), np.nan)
                else:
                    return (common_dtype, None)
    msg = 'invalid dtype determination in get_concat_dtype'
    raise AssertionError(msg)

def _get_upcast_classes(join_units, dtypes):
    'Create mapping between upcast class names and lists of dtypes.'
    upcast_classes: Dict[(str, List[DtypeObj])] = defaultdict(list)
    null_upcast_classes: Dict[(str, List[DtypeObj])] = defaultdict(list)
    for (dtype, unit) in zip(dtypes, join_units):
        if (dtype is None):
            continue
        upcast_cls = _select_upcast_cls_from_dtype(dtype)
        if unit.is_na:
            null_upcast_classes[upcast_cls].append(dtype)
        else:
            upcast_classes[upcast_cls].append(dtype)
    if (not upcast_classes):
        upcast_classes = null_upcast_classes
    return upcast_classes

def _select_upcast_cls_from_dtype(dtype):
    'Select upcast class name based on dtype.'
    if is_categorical_dtype(dtype):
        return 'category'
    elif is_datetime64tz_dtype(dtype):
        return 'datetimetz'
    elif is_extension_array_dtype(dtype):
        return 'extension'
    elif issubclass(dtype.type, np.bool_):
        return 'bool'
    elif issubclass(dtype.type, np.object_):
        return 'object'
    elif is_datetime64_dtype(dtype):
        return 'datetime'
    elif is_timedelta64_dtype(dtype):
        return 'timedelta'
    elif is_sparse(dtype):
        dtype = cast('SparseDtype', dtype)
        return dtype.subtype.name
    elif (is_float_dtype(dtype) or is_numeric_dtype(dtype)):
        return dtype.name
    else:
        return 'float'

def _is_uniform_join_units(join_units):
    '\n    Check if the join units consist of blocks of uniform type that can\n    be concatenated using Block.concat_same_type instead of the generic\n    _concatenate_join_units (which uses `concat_compat`).\n\n    '
    return (all(((type(ju.block) is type(join_units[0].block)) for ju in join_units)) and all((((not ju.is_na) or ju.block.is_extension) for ju in join_units)) and all(((not ju.indexers) for ju in join_units)) and (len(join_units) > 1))

def _is_uniform_reindex(join_units):
    return (all(((ju.block and ju.block.is_extension) for ju in join_units)) and (len({ju.block.dtype.name for ju in join_units}) == 1))

def _trim_join_unit(join_unit, length):
    "\n    Reduce join_unit's shape along item axis to length.\n\n    Extra items that didn't fit are returned as a separate block.\n    "
    if (0 not in join_unit.indexers):
        extra_indexers = join_unit.indexers
        if (join_unit.block is None):
            extra_block = None
        else:
            extra_block = join_unit.block.getitem_block(slice(length, None))
            join_unit.block = join_unit.block.getitem_block(slice(length))
    else:
        extra_block = join_unit.block
        extra_indexers = copy.copy(join_unit.indexers)
        extra_indexers[0] = extra_indexers[0][length:]
        join_unit.indexers[0] = join_unit.indexers[0][:length]
    extra_shape = (((join_unit.shape[0] - length),) + join_unit.shape[1:])
    join_unit.shape = ((length,) + join_unit.shape[1:])
    return JoinUnit(block=extra_block, indexers=extra_indexers, shape=extra_shape)

def _combine_concat_plans(plans, concat_axis):
    '\n    Combine multiple concatenation plans into one.\n\n    existing_plan is updated in-place.\n    '
    if (len(plans) == 1):
        for p in plans[0]:
            (yield (p[0], [p[1]]))
    elif (concat_axis == 0):
        offset = 0
        for plan in plans:
            last_plc = None
            for (plc, unit) in plan:
                (yield (plc.add(offset), [unit]))
                last_plc = plc
            if (last_plc is not None):
                offset += last_plc.as_slice.stop
    else:
        num_ended = [0]

        def _next_or_none(seq):
            retval = next(seq, None)
            if (retval is None):
                num_ended[0] += 1
            return retval
        plans = list(map(iter, plans))
        next_items = list(map(_next_or_none, plans))
        while (num_ended[0] != len(next_items)):
            if (num_ended[0] > 0):
                raise ValueError('Plan shapes are not aligned')
            (placements, units) = zip(*next_items)
            lengths = list(map(len, placements))
            (min_len, max_len) = (min(lengths), max(lengths))
            if (min_len == max_len):
                (yield (placements[0], units))
                next_items[:] = map(_next_or_none, plans)
            else:
                yielded_placement = None
                yielded_units = ([None] * len(next_items))
                for (i, (plc, unit)) in enumerate(next_items):
                    yielded_units[i] = unit
                    if (len(plc) > min_len):
                        next_items[i] = (plc[min_len:], _trim_join_unit(unit, min_len))
                    else:
                        yielded_placement = plc
                        next_items[i] = _next_or_none(plans[i])
                (yield (yielded_placement, yielded_units))
