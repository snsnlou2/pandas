
'\nHelper functions to generate range-like data for DatetimeArray\n(and possibly TimedeltaArray/PeriodArray)\n'
from typing import Union
import numpy as np
from pandas._libs.tslibs import BaseOffset, OutOfBoundsDatetime, Timedelta, Timestamp

def generate_regular_range(start, end, periods, freq):
    '\n    Generate a range of dates or timestamps with the spans between dates\n    described by the given `freq` DateOffset.\n\n    Parameters\n    ----------\n    start : Timedelta, Timestamp or None\n        First point of produced date range.\n    end : Timedelta, Timestamp or None\n        Last point of produced date range.\n    periods : int\n        Number of periods in produced date range.\n    freq : Tick\n        Describes space between dates in produced date range.\n\n    Returns\n    -------\n    ndarray[np.int64] Representing nanoseconds.\n    '
    start = (start.value if (start is not None) else None)
    end = (end.value if (end is not None) else None)
    stride = freq.nanos
    if (periods is None):
        b = start
        e = (((b + (((end - b) // stride) * stride)) + (stride // 2)) + 1)
    elif (start is not None):
        b = start
        e = _generate_range_overflow_safe(b, periods, stride, side='start')
    elif (end is not None):
        e = (end + stride)
        b = _generate_range_overflow_safe(e, periods, stride, side='end')
    else:
        raise ValueError("at least 'start' or 'end' should be specified if a 'period' is given.")
    with np.errstate(over='raise'):
        try:
            values = np.arange(b, e, stride, dtype=np.int64)
        except FloatingPointError:
            xdr = [b]
            while (xdr[(- 1)] != e):
                xdr.append((xdr[(- 1)] + stride))
            values = np.array(xdr[:(- 1)], dtype=np.int64)
    return values

def _generate_range_overflow_safe(endpoint, periods, stride, side='start'):
    "\n    Calculate the second endpoint for passing to np.arange, checking\n    to avoid an integer overflow.  Catch OverflowError and re-raise\n    as OutOfBoundsDatetime.\n\n    Parameters\n    ----------\n    endpoint : int\n        nanosecond timestamp of the known endpoint of the desired range\n    periods : int\n        number of periods in the desired range\n    stride : int\n        nanoseconds between periods in the desired range\n    side : {'start', 'end'}\n        which end of the range `endpoint` refers to\n\n    Returns\n    -------\n    other_end : int\n\n    Raises\n    ------\n    OutOfBoundsDatetime\n    "
    assert (side in ['start', 'end'])
    i64max = np.uint64(np.iinfo(np.int64).max)
    msg = f'Cannot generate range with {side}={endpoint} and periods={periods}'
    with np.errstate(over='raise'):
        try:
            addend = (np.uint64(periods) * np.uint64(np.abs(stride)))
        except FloatingPointError as err:
            raise OutOfBoundsDatetime(msg) from err
    if (np.abs(addend) <= i64max):
        return _generate_range_overflow_safe_signed(endpoint, periods, stride, side)
    elif (((endpoint > 0) and (side == 'start') and (stride > 0)) or ((endpoint < 0) and (side == 'end') and (stride > 0))):
        raise OutOfBoundsDatetime(msg)
    elif ((side == 'end') and (endpoint > i64max) and ((endpoint - stride) <= i64max)):
        return _generate_range_overflow_safe((endpoint - stride), (periods - 1), stride, side)
    mid_periods = (periods // 2)
    remaining = (periods - mid_periods)
    assert (0 < remaining < periods), (remaining, periods, endpoint, stride)
    midpoint = _generate_range_overflow_safe(endpoint, mid_periods, stride, side)
    return _generate_range_overflow_safe(midpoint, remaining, stride, side)

def _generate_range_overflow_safe_signed(endpoint, periods, stride, side):
    '\n    A special case for _generate_range_overflow_safe where `periods * stride`\n    can be calculated without overflowing int64 bounds.\n    '
    assert (side in ['start', 'end'])
    if (side == 'end'):
        stride *= (- 1)
    with np.errstate(over='raise'):
        addend = (np.int64(periods) * np.int64(stride))
        try:
            return (np.int64(endpoint) + addend)
        except (FloatingPointError, OverflowError):
            pass
        assert (((stride > 0) and (endpoint >= 0)) or ((stride < 0) and (endpoint <= 0)))
        if (stride > 0):
            result = (np.uint64(endpoint) + np.uint64(addend))
            i64max = np.uint64(np.iinfo(np.int64).max)
            assert (result > i64max)
            if (result <= (i64max + np.uint64(stride))):
                return result
    raise OutOfBoundsDatetime(f'Cannot generate range with {side}={endpoint} and periods={periods}')
