
from math import ceil
from typing import TYPE_CHECKING, Iterable, List, Sequence, Tuple, Union
import warnings
import matplotlib.table
import matplotlib.ticker as ticker
import numpy as np
from pandas._typing import FrameOrSeriesUnion
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
from pandas.plotting._matplotlib import compat
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.lines import Line2D
    from matplotlib.table import Table

def format_date_labels(ax, rot):
    for label in ax.get_xticklabels():
        label.set_ha('right')
        label.set_rotation(rot)
    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.2)

def table(ax, data, rowLabels=None, colLabels=None, **kwargs):
    if isinstance(data, ABCSeries):
        data = data.to_frame()
    elif isinstance(data, ABCDataFrame):
        pass
    else:
        raise ValueError('Input data must be DataFrame or Series')
    if (rowLabels is None):
        rowLabels = data.index
    if (colLabels is None):
        colLabels = data.columns
    cellText = data.values
    table = matplotlib.table.table(ax, cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, **kwargs)
    return table

def _get_layout(nplots, layout=None, layout_type='box'):
    if (layout is not None):
        if ((not isinstance(layout, (tuple, list))) or (len(layout) != 2)):
            raise ValueError('Layout must be a tuple of (rows, columns)')
        (nrows, ncols) = layout
        ceil_ = (lambda x: int(ceil(x)))
        if ((nrows == (- 1)) and (ncols > 0)):
            layout = (nrows, ncols) = (ceil_((float(nplots) / ncols)), ncols)
        elif ((ncols == (- 1)) and (nrows > 0)):
            layout = (nrows, ncols) = (nrows, ceil_((float(nplots) / nrows)))
        elif ((ncols <= 0) and (nrows <= 0)):
            msg = 'At least one dimension of layout must be positive'
            raise ValueError(msg)
        if ((nrows * ncols) < nplots):
            raise ValueError(f'Layout of {nrows}x{ncols} must be larger than required size {nplots}')
        return layout
    if (layout_type == 'single'):
        return (1, 1)
    elif (layout_type == 'horizontal'):
        return (1, nplots)
    elif (layout_type == 'vertical'):
        return (nplots, 1)
    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    try:
        return layouts[nplots]
    except KeyError:
        k = 1
        while ((k ** 2) < nplots):
            k += 1
        if (((k - 1) * k) >= nplots):
            return (k, (k - 1))
        else:
            return (k, k)

def create_subplots(naxes, sharex=False, sharey=False, squeeze=True, subplot_kw=None, ax=None, layout=None, layout_type='box', **fig_kw):
    "\n    Create a figure with a set of subplots already made.\n\n    This utility wrapper makes it convenient to create common layouts of\n    subplots, including the enclosing figure object, in a single call.\n\n    Parameters\n    ----------\n    naxes : int\n      Number of required axes. Exceeded axes are set invisible. Default is\n      nrows * ncols.\n\n    sharex : bool\n      If True, the X axis will be shared amongst all subplots.\n\n    sharey : bool\n      If True, the Y axis will be shared amongst all subplots.\n\n    squeeze : bool\n\n      If True, extra dimensions are squeezed out from the returned axis object:\n        - if only one subplot is constructed (nrows=ncols=1), the resulting\n        single Axis object is returned as a scalar.\n        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy object\n        array of Axis objects are returned as numpy 1-d arrays.\n        - for NxM subplots with N>1 and M>1 are returned as a 2d array.\n\n      If False, no squeezing is done: the returned axis object is always\n      a 2-d array containing Axis instances, even if it ends up being 1x1.\n\n    subplot_kw : dict\n      Dict with keywords passed to the add_subplot() call used to create each\n      subplots.\n\n    ax : Matplotlib axis object, optional\n\n    layout : tuple\n      Number of rows and columns of the subplot grid.\n      If not specified, calculated from naxes and layout_type\n\n    layout_type : {'box', 'horizontal', 'vertical'}, default 'box'\n      Specify how to layout the subplot grid.\n\n    fig_kw : Other keyword arguments to be passed to the figure() call.\n        Note that all keywords not recognized above will be\n        automatically included here.\n\n    Returns\n    -------\n    fig, ax : tuple\n      - fig is the Matplotlib Figure object\n      - ax can be either a single axis object or an array of axis objects if\n      more than one subplot was created.  The dimensions of the resulting array\n      can be controlled with the squeeze keyword, see above.\n\n    Examples\n    --------\n    x = np.linspace(0, 2*np.pi, 400)\n    y = np.sin(x**2)\n\n    # Just a figure and one subplot\n    f, ax = plt.subplots()\n    ax.plot(x, y)\n    ax.set_title('Simple plot')\n\n    # Two subplots, unpack the output array immediately\n    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)\n    ax1.plot(x, y)\n    ax1.set_title('Sharing Y axis')\n    ax2.scatter(x, y)\n\n    # Four polar axes\n    plt.subplots(2, 2, subplot_kw=dict(polar=True))\n    "
    import matplotlib.pyplot as plt
    if (subplot_kw is None):
        subplot_kw = {}
    if (ax is None):
        fig = plt.figure(**fig_kw)
    else:
        if is_list_like(ax):
            if squeeze:
                ax = flatten_axes(ax)
            if (layout is not None):
                warnings.warn('When passing multiple axes, layout keyword is ignored', UserWarning)
            if (sharex or sharey):
                warnings.warn('When passing multiple axes, sharex and sharey are ignored. These settings must be specified when creating axes', UserWarning, stacklevel=4)
            if (ax.size == naxes):
                fig = ax.flat[0].get_figure()
                return (fig, ax)
            else:
                raise ValueError(f'The number of passed axes must be {naxes}, the same as the output plot')
        fig = ax.get_figure()
        if (naxes == 1):
            if squeeze:
                return (fig, ax)
            else:
                return (fig, flatten_axes(ax))
        else:
            warnings.warn('To output multiple subplots, the figure containing the passed axes is being cleared', UserWarning, stacklevel=4)
            fig.clear()
    (nrows, ncols) = _get_layout(naxes, layout=layout, layout_type=layout_type)
    nplots = (nrows * ncols)
    axarr = np.empty(nplots, dtype=object)
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)
    if sharex:
        subplot_kw['sharex'] = ax0
    if sharey:
        subplot_kw['sharey'] = ax0
    axarr[0] = ax0
    for i in range(1, nplots):
        kwds = subplot_kw.copy()
        if (i >= naxes):
            kwds['sharex'] = None
            kwds['sharey'] = None
        ax = fig.add_subplot(nrows, ncols, (i + 1), **kwds)
        axarr[i] = ax
    if (naxes != nplots):
        for ax in axarr[naxes:]:
            ax.set_visible(False)
    handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey)
    if squeeze:
        if (nplots == 1):
            axes = axarr[0]
        else:
            axes = axarr.reshape(nrows, ncols).squeeze()
    else:
        axes = axarr.reshape(nrows, ncols)
    return (fig, axes)

def _remove_labels_from_axis(axis):
    for t in axis.get_majorticklabels():
        t.set_visible(False)
    if isinstance(axis.get_minor_locator(), ticker.NullLocator):
        axis.set_minor_locator(ticker.AutoLocator())
    if isinstance(axis.get_minor_formatter(), ticker.NullFormatter):
        axis.set_minor_formatter(ticker.FormatStrFormatter(''))
    for t in axis.get_minorticklabels():
        t.set_visible(False)
    axis.get_label().set_visible(False)

def _has_externally_shared_axis(ax1, compare_axis):
    '\n    Return whether an axis is externally shared.\n\n    Parameters\n    ----------\n    ax1 : matplotlib.axes\n        Axis to query.\n    compare_axis : str\n        `"x"` or `"y"` according to whether the X-axis or Y-axis is being\n        compared.\n\n    Returns\n    -------\n    bool\n        `True` if the axis is externally shared. Otherwise `False`.\n\n    Notes\n    -----\n    If two axes with different positions are sharing an axis, they can be\n    referred to as *externally* sharing the common axis.\n\n    If two axes sharing an axis also have the same position, they can be\n    referred to as *internally* sharing the common axis (a.k.a twinning).\n\n    _handle_shared_axes() is only interested in axes externally sharing an\n    axis, regardless of whether either of the axes is also internally sharing\n    with a third axis.\n    '
    if (compare_axis == 'x'):
        axes = ax1.get_shared_x_axes()
    elif (compare_axis == 'y'):
        axes = ax1.get_shared_y_axes()
    else:
        raise ValueError("_has_externally_shared_axis() needs 'x' or 'y' as a second parameter")
    axes = axes.get_siblings(ax1)
    ax1_points = ax1.get_position().get_points()
    for ax2 in axes:
        if (not np.array_equal(ax1_points, ax2.get_position().get_points())):
            return True
    return False

def handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey):
    if (nplots > 1):
        if compat.mpl_ge_3_2_0():
            row_num = (lambda x: x.get_subplotspec().rowspan.start)
            col_num = (lambda x: x.get_subplotspec().colspan.start)
        else:
            row_num = (lambda x: x.rowNum)
            col_num = (lambda x: x.colNum)
        if (nrows > 1):
            try:
                layout = np.zeros(((nrows + 1), (ncols + 1)), dtype=np.bool_)
                for ax in axarr:
                    layout[(row_num(ax), col_num(ax))] = ax.get_visible()
                for ax in axarr:
                    if (not layout[((row_num(ax) + 1), col_num(ax))]):
                        continue
                    if (sharex or _has_externally_shared_axis(ax, 'x')):
                        _remove_labels_from_axis(ax.xaxis)
            except IndexError:
                for ax in axarr:
                    if ax.is_last_row():
                        continue
                    if (sharex or _has_externally_shared_axis(ax, 'x')):
                        _remove_labels_from_axis(ax.xaxis)
        if (ncols > 1):
            for ax in axarr:
                if ax.is_first_col():
                    continue
                if (sharey or _has_externally_shared_axis(ax, 'y')):
                    _remove_labels_from_axis(ax.yaxis)

def flatten_axes(axes):
    if (not is_list_like(axes)):
        return np.array([axes])
    elif isinstance(axes, (np.ndarray, ABCIndex)):
        return np.asarray(axes).ravel()
    return np.array(axes)

def set_ticks_props(axes, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None):
    import matplotlib.pyplot as plt
    for ax in flatten_axes(axes):
        if (xlabelsize is not None):
            plt.setp(ax.get_xticklabels(), fontsize=xlabelsize)
        if (xrot is not None):
            plt.setp(ax.get_xticklabels(), rotation=xrot)
        if (ylabelsize is not None):
            plt.setp(ax.get_yticklabels(), fontsize=ylabelsize)
        if (yrot is not None):
            plt.setp(ax.get_yticklabels(), rotation=yrot)
    return axes

def get_all_lines(ax):
    lines = ax.get_lines()
    if hasattr(ax, 'right_ax'):
        lines += ax.right_ax.get_lines()
    if hasattr(ax, 'left_ax'):
        lines += ax.left_ax.get_lines()
    return lines

def get_xlim(lines):
    (left, right) = (np.inf, (- np.inf))
    for line in lines:
        x = line.get_xdata(orig=False)
        left = min(np.nanmin(x), left)
        right = max(np.nanmax(x), right)
    return (left, right)
