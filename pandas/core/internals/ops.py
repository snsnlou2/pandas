
from collections import namedtuple
from typing import TYPE_CHECKING, Iterator, List, Tuple
import numpy as np
from pandas._typing import ArrayLike
if TYPE_CHECKING:
    from pandas.core.internals.blocks import Block
    from pandas.core.internals.managers import BlockManager
BlockPairInfo = namedtuple('BlockPairInfo', ['lvals', 'rvals', 'locs', 'left_ea', 'right_ea', 'rblk'])

def _iter_block_pairs(left, right):
    for (n, blk) in enumerate(left.blocks):
        locs = blk.mgr_locs
        blk_vals = blk.values
        left_ea = (not isinstance(blk_vals, np.ndarray))
        rblks = right._slice_take_blocks_ax0(locs.indexer, only_slice=True)
        for (k, rblk) in enumerate(rblks):
            right_ea = (not isinstance(rblk.values, np.ndarray))
            (lvals, rvals) = _get_same_shape_values(blk, rblk, left_ea, right_ea)
            info = BlockPairInfo(lvals, rvals, locs, left_ea, right_ea, rblk)
            (yield info)

def operate_blockwise(left, right, array_op):
    res_blks: List['Block'] = []
    for (lvals, rvals, locs, left_ea, right_ea, rblk) in _iter_block_pairs(left, right):
        res_values = array_op(lvals, rvals)
        if (left_ea and (not right_ea) and hasattr(res_values, 'reshape')):
            res_values = res_values.reshape(1, (- 1))
        nbs = rblk._split_op_result(res_values)
        _reset_block_mgr_locs(nbs, locs)
        res_blks.extend(nbs)
    new_mgr = type(right)(res_blks, axes=right.axes, do_integrity_check=False)
    return new_mgr

def _reset_block_mgr_locs(nbs, locs):
    '\n    Reset mgr_locs to correspond to our original DataFrame.\n    '
    for nb in nbs:
        nblocs = locs.as_array[nb.mgr_locs.indexer]
        nb.mgr_locs = nblocs

def _get_same_shape_values(lblk, rblk, left_ea, right_ea):
    '\n    Slice lblk.values to align with rblk.  Squeeze if we have EAs.\n    '
    lvals = lblk.values
    rvals = rblk.values
    assert rblk.mgr_locs.is_slice_like, rblk.mgr_locs
    if (not (left_ea or right_ea)):
        lvals = lvals[rblk.mgr_locs.indexer, :]
        assert (lvals.shape == rvals.shape), (lvals.shape, rvals.shape)
    elif (left_ea and right_ea):
        assert (lvals.shape == rvals.shape), (lvals.shape, rvals.shape)
    elif right_ea:
        lvals = lvals[rblk.mgr_locs.indexer, :]
        assert (lvals.shape[0] == 1), lvals.shape
        lvals = lvals[0, :]
    else:
        assert (rvals.shape[0] == 1), rvals.shape
        rvals = rvals[0, :]
    return (lvals, rvals)

def blockwise_all(left, right, op):
    '\n    Blockwise `all` reduction.\n    '
    for info in _iter_block_pairs(left, right):
        res = op(info.lvals, info.rvals)
        if (not res):
            return False
    return True
