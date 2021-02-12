
import itertools
from typing import TYPE_CHECKING, Collection, Dict, Iterator, List, Optional, Sequence, Union, cast
import warnings
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
from pandas.core.dtypes.common import is_list_like
import pandas.core.common as com
if TYPE_CHECKING:
    from matplotlib.colors import Colormap
Color = Union[(str, Sequence[float])]

def get_standard_colors(num_colors, colormap=None, color_type='default', color=None):
    '\n    Get standard colors based on `colormap`, `color_type` or `color` inputs.\n\n    Parameters\n    ----------\n    num_colors : int\n        Minimum number of colors to be returned.\n        Ignored if `color` is a dictionary.\n    colormap : :py:class:`matplotlib.colors.Colormap`, optional\n        Matplotlib colormap.\n        When provided, the resulting colors will be derived from the colormap.\n    color_type : {"default", "random"}, optional\n        Type of colors to derive. Used if provided `color` and `colormap` are None.\n        Ignored if either `color` or `colormap` are not None.\n    color : dict or str or sequence, optional\n        Color(s) to be used for deriving sequence of colors.\n        Can be either be a dictionary, or a single color (single color string,\n        or sequence of floats representing a single color),\n        or a sequence of colors.\n\n    Returns\n    -------\n    dict or list\n        Standard colors. Can either be a mapping if `color` was a dictionary,\n        or a list of colors with a length of `num_colors` or more.\n\n    Warns\n    -----\n    UserWarning\n        If both `colormap` and `color` are provided.\n        Parameter `color` will override.\n    '
    if isinstance(color, dict):
        return color
    colors = _derive_colors(color=color, colormap=colormap, color_type=color_type, num_colors=num_colors)
    return list(_cycle_colors(colors, num_colors=num_colors))

def _derive_colors(*, color, colormap, color_type, num_colors):
    '\n    Derive colors from either `colormap`, `color_type` or `color` inputs.\n\n    Get a list of colors either from `colormap`, or from `color`,\n    or from `color_type` (if both `colormap` and `color` are None).\n\n    Parameters\n    ----------\n    color : str or sequence, optional\n        Color(s) to be used for deriving sequence of colors.\n        Can be either be a single color (single color string, or sequence of floats\n        representing a single color), or a sequence of colors.\n    colormap : :py:class:`matplotlib.colors.Colormap`, optional\n        Matplotlib colormap.\n        When provided, the resulting colors will be derived from the colormap.\n    color_type : {"default", "random"}, optional\n        Type of colors to derive. Used if provided `color` and `colormap` are None.\n        Ignored if either `color` or `colormap`` are not None.\n    num_colors : int\n        Number of colors to be extracted.\n\n    Returns\n    -------\n    list\n        List of colors extracted.\n\n    Warns\n    -----\n    UserWarning\n        If both `colormap` and `color` are provided.\n        Parameter `color` will override.\n    '
    if ((color is None) and (colormap is not None)):
        return _get_colors_from_colormap(colormap, num_colors=num_colors)
    elif (color is not None):
        if (colormap is not None):
            warnings.warn("'color' and 'colormap' cannot be used simultaneously. Using 'color'")
        return _get_colors_from_color(color)
    else:
        return _get_colors_from_color_type(color_type, num_colors=num_colors)

def _cycle_colors(colors, num_colors):
    'Cycle colors until achieving max of `num_colors` or length of `colors`.\n\n    Extra colors will be ignored by matplotlib if there are more colors\n    than needed and nothing needs to be done here.\n    '
    max_colors = max(num_colors, len(colors))
    (yield from itertools.islice(itertools.cycle(colors), max_colors))

def _get_colors_from_colormap(colormap, num_colors):
    'Get colors from colormap.'
    colormap = _get_cmap_instance(colormap)
    return [colormap(num) for num in np.linspace(0, 1, num=num_colors)]

def _get_cmap_instance(colormap):
    'Get instance of matplotlib colormap.'
    if isinstance(colormap, str):
        cmap = colormap
        colormap = cm.get_cmap(colormap)
        if (colormap is None):
            raise ValueError(f'Colormap {cmap} is not recognized')
    return colormap

def _get_colors_from_color(color):
    'Get colors from user input color.'
    if (len(color) == 0):
        raise ValueError(f'Invalid color argument: {color}')
    if _is_single_color(color):
        color = cast(Color, color)
        return [color]
    color = cast(Collection[Color], color)
    return list(_gen_list_of_colors_from_iterable(color))

def _is_single_color(color):
    'Check if `color` is a single color, not a sequence of colors.\n\n    Single color is of these kinds:\n        - Named color "red", "C0", "firebrick"\n        - Alias "g"\n        - Sequence of floats, such as (0.1, 0.2, 0.3) or (0.1, 0.2, 0.3, 0.4).\n\n    See Also\n    --------\n    _is_single_string_color\n    '
    if (isinstance(color, str) and _is_single_string_color(color)):
        return True
    if _is_floats_color(color):
        return True
    return False

def _gen_list_of_colors_from_iterable(color):
    '\n    Yield colors from string of several letters or from collection of colors.\n    '
    for x in color:
        if _is_single_color(x):
            (yield x)
        else:
            raise ValueError(f'Invalid color {x}')

def _is_floats_color(color):
    'Check if color comprises a sequence of floats representing color.'
    return bool((is_list_like(color) and ((len(color) == 3) or (len(color) == 4)) and all((isinstance(x, (int, float)) for x in color))))

def _get_colors_from_color_type(color_type, num_colors):
    'Get colors from user input color type.'
    if (color_type == 'default'):
        return _get_default_colors(num_colors)
    elif (color_type == 'random'):
        return _get_random_colors(num_colors)
    else:
        raise ValueError("color_type must be either 'default' or 'random'")

def _get_default_colors(num_colors):
    'Get `num_colors` of default colors from matplotlib rc params.'
    import matplotlib.pyplot as plt
    colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    return colors[0:num_colors]

def _get_random_colors(num_colors):
    'Get `num_colors` of random colors.'
    return [_random_color(num) for num in range(num_colors)]

def _random_color(column):
    'Get a random color represented as a list of length 3'
    rs = com.random_state(column)
    return rs.rand(3).tolist()

def _is_single_string_color(color):
    "Check if `color` is a single string color.\n\n    Examples of single string colors:\n        - 'r'\n        - 'g'\n        - 'red'\n        - 'green'\n        - 'C3'\n        - 'firebrick'\n\n    Parameters\n    ----------\n    color : Color\n        Color string or sequence of floats.\n\n    Returns\n    -------\n    bool\n        True if `color` looks like a valid color.\n        False otherwise.\n    "
    conv = matplotlib.colors.ColorConverter()
    try:
        conv.to_rgba(color)
    except ValueError:
        return False
    else:
        return True
