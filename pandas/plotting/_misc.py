
from contextlib import contextmanager
from pandas.plotting._core import _get_plot_backend

def table(ax, data, rowLabels=None, colLabels=None, **kwargs):
    '\n    Helper function to convert DataFrame and Series to matplotlib.table.\n\n    Parameters\n    ----------\n    ax : Matplotlib axes object\n    data : DataFrame or Series\n        Data for table contents.\n    **kwargs\n        Keyword arguments to be passed to matplotlib.table.table.\n        If `rowLabels` or `colLabels` is not specified, data index or column\n        name will be used.\n\n    Returns\n    -------\n    matplotlib table object\n    '
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.table(ax=ax, data=data, rowLabels=None, colLabels=None, **kwargs)

def register():
    '\n    Register pandas formatters and converters with matplotlib.\n\n    This function modifies the global ``matplotlib.units.registry``\n    dictionary. pandas adds custom converters for\n\n    * pd.Timestamp\n    * pd.Period\n    * np.datetime64\n    * datetime.datetime\n    * datetime.date\n    * datetime.time\n\n    See Also\n    --------\n    deregister_matplotlib_converters : Remove pandas formatters and converters.\n    '
    plot_backend = _get_plot_backend('matplotlib')
    plot_backend.register()

def deregister():
    "\n    Remove pandas formatters and converters.\n\n    Removes the custom converters added by :func:`register`. This\n    attempts to set the state of the registry back to the state before\n    pandas registered its own units. Converters for pandas' own types like\n    Timestamp and Period are removed completely. Converters for types\n    pandas overwrites, like ``datetime.datetime``, are restored to their\n    original value.\n\n    See Also\n    --------\n    register_matplotlib_converters : Register pandas formatters and converters\n        with matplotlib.\n    "
    plot_backend = _get_plot_backend('matplotlib')
    plot_backend.deregister()

def scatter_matrix(frame, alpha=0.5, figsize=None, ax=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwargs):
    "\n    Draw a matrix of scatter plots.\n\n    Parameters\n    ----------\n    frame : DataFrame\n    alpha : float, optional\n        Amount of transparency applied.\n    figsize : (float,float), optional\n        A tuple (width, height) in inches.\n    ax : Matplotlib axis object, optional\n    grid : bool, optional\n        Setting this to True will show the grid.\n    diagonal : {'hist', 'kde'}\n        Pick between 'kde' and 'hist' for either Kernel Density Estimation or\n        Histogram plot in the diagonal.\n    marker : str, optional\n        Matplotlib marker type, default '.'.\n    density_kwds : keywords\n        Keyword arguments to be passed to kernel density estimate plot.\n    hist_kwds : keywords\n        Keyword arguments to be passed to hist function.\n    range_padding : float, default 0.05\n        Relative extension of axis range in x and y with respect to\n        (x_max - x_min) or (y_max - y_min).\n    **kwargs\n        Keyword arguments to be passed to scatter function.\n\n    Returns\n    -------\n    numpy.ndarray\n        A matrix of scatter plots.\n\n    Examples\n    --------\n\n    .. plot::\n        :context: close-figs\n\n        >>> df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])\n        >>> pd.plotting.scatter_matrix(df, alpha=0.2)\n    "
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.scatter_matrix(frame=frame, alpha=alpha, figsize=figsize, ax=ax, grid=grid, diagonal=diagonal, marker=marker, density_kwds=density_kwds, hist_kwds=hist_kwds, range_padding=range_padding, **kwargs)

def radviz(frame, class_column, ax=None, color=None, colormap=None, **kwds):
    "\n    Plot a multidimensional dataset in 2D.\n\n    Each Series in the DataFrame is represented as a evenly distributed\n    slice on a circle. Each data point is rendered in the circle according to\n    the value on each Series. Highly correlated `Series` in the `DataFrame`\n    are placed closer on the unit circle.\n\n    RadViz allow to project a N-dimensional data set into a 2D space where the\n    influence of each dimension can be interpreted as a balance between the\n    influence of all dimensions.\n\n    More info available at the `original article\n    <https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.135.889>`_\n    describing RadViz.\n\n    Parameters\n    ----------\n    frame : `DataFrame`\n        Object holding the data.\n    class_column : str\n        Column name containing the name of the data point category.\n    ax : :class:`matplotlib.axes.Axes`, optional\n        A plot instance to which to add the information.\n    color : list[str] or tuple[str], optional\n        Assign a color to each category. Example: ['blue', 'green'].\n    colormap : str or :class:`matplotlib.colors.Colormap`, default None\n        Colormap to select colors from. If string, load colormap with that\n        name from matplotlib.\n    **kwds\n        Options to pass to matplotlib scatter plotting method.\n\n    Returns\n    -------\n    class:`matplotlib.axes.Axes`\n\n    See Also\n    --------\n    plotting.andrews_curves : Plot clustering visualization.\n\n    Examples\n    --------\n\n    .. plot::\n        :context: close-figs\n\n        >>> df = pd.DataFrame(\n        ...     {\n        ...         'SepalLength': [6.5, 7.7, 5.1, 5.8, 7.6, 5.0, 5.4, 4.6, 6.7, 4.6],\n        ...         'SepalWidth': [3.0, 3.8, 3.8, 2.7, 3.0, 2.3, 3.0, 3.2, 3.3, 3.6],\n        ...         'PetalLength': [5.5, 6.7, 1.9, 5.1, 6.6, 3.3, 4.5, 1.4, 5.7, 1.0],\n        ...         'PetalWidth': [1.8, 2.2, 0.4, 1.9, 2.1, 1.0, 1.5, 0.2, 2.1, 0.2],\n        ...         'Category': [\n        ...             'virginica',\n        ...             'virginica',\n        ...             'setosa',\n        ...             'virginica',\n        ...             'virginica',\n        ...             'versicolor',\n        ...             'versicolor',\n        ...             'setosa',\n        ...             'virginica',\n        ...             'setosa'\n        ...         ]\n        ...     }\n        ... )\n        >>> pd.plotting.radviz(df, 'Category')\n    "
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.radviz(frame=frame, class_column=class_column, ax=ax, color=color, colormap=colormap, **kwds)

def andrews_curves(frame, class_column, ax=None, samples=200, color=None, colormap=None, **kwargs):
    "\n    Generate a matplotlib plot of Andrews curves, for visualising clusters of\n    multivariate data.\n\n    Andrews curves have the functional form:\n\n    f(t) = x_1/sqrt(2) + x_2 sin(t) + x_3 cos(t) +\n           x_4 sin(2t) + x_5 cos(2t) + ...\n\n    Where x coefficients correspond to the values of each dimension and t is\n    linearly spaced between -pi and +pi. Each row of frame then corresponds to\n    a single curve.\n\n    Parameters\n    ----------\n    frame : DataFrame\n        Data to be plotted, preferably normalized to (0.0, 1.0).\n    class_column : Name of the column containing class names\n    ax : matplotlib axes object, default None\n    samples : Number of points to plot in each curve\n    color : list or tuple, optional\n        Colors to use for the different classes.\n    colormap : str or matplotlib colormap object, default None\n        Colormap to select colors from. If string, load colormap with that name\n        from matplotlib.\n    **kwargs\n        Options to pass to matplotlib plotting method.\n\n    Returns\n    -------\n    class:`matplotlip.axis.Axes`\n\n    Examples\n    --------\n\n    .. plot::\n        :context: close-figs\n\n        >>> df = pd.read_csv(\n        ...     'https://raw.github.com/pandas-dev/'\n        ...     'pandas/master/pandas/tests/io/data/csv/iris.csv'\n        ... )\n        >>> pd.plotting.andrews_curves(df, 'Name')\n    "
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.andrews_curves(frame=frame, class_column=class_column, ax=ax, samples=samples, color=color, colormap=colormap, **kwargs)

def bootstrap_plot(series, fig=None, size=50, samples=500, **kwds):
    '\n    Bootstrap plot on mean, median and mid-range statistics.\n\n    The bootstrap plot is used to estimate the uncertainty of a statistic\n    by relaying on random sampling with replacement [1]_. This function will\n    generate bootstrapping plots for mean, median and mid-range statistics\n    for the given number of samples of the given size.\n\n    .. [1] "Bootstrapping (statistics)" in     https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29\n\n    Parameters\n    ----------\n    series : pandas.Series\n        Series from where to get the samplings for the bootstrapping.\n    fig : matplotlib.figure.Figure, default None\n        If given, it will use the `fig` reference for plotting instead of\n        creating a new one with default parameters.\n    size : int, default 50\n        Number of data points to consider during each sampling. It must be\n        less than or equal to the length of the `series`.\n    samples : int, default 500\n        Number of times the bootstrap procedure is performed.\n    **kwds\n        Options to pass to matplotlib plotting method.\n\n    Returns\n    -------\n    matplotlib.figure.Figure\n        Matplotlib figure.\n\n    See Also\n    --------\n    DataFrame.plot : Basic plotting for DataFrame objects.\n    Series.plot : Basic plotting for Series objects.\n\n    Examples\n    --------\n    This example draws a basic bootstrap plot for a Series.\n\n    .. plot::\n        :context: close-figs\n\n        >>> s = pd.Series(np.random.uniform(size=100))\n        >>> pd.plotting.bootstrap_plot(s)\n    '
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.bootstrap_plot(series=series, fig=fig, size=size, samples=samples, **kwds)

def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None, use_columns=False, xticks=None, colormap=None, axvlines=True, axvlines_kwds=None, sort_labels=False, **kwargs):
    "\n    Parallel coordinates plotting.\n\n    Parameters\n    ----------\n    frame : DataFrame\n    class_column : str\n        Column name containing class names.\n    cols : list, optional\n        A list of column names to use.\n    ax : matplotlib.axis, optional\n        Matplotlib axis object.\n    color : list or tuple, optional\n        Colors to use for the different classes.\n    use_columns : bool, optional\n        If true, columns will be used as xticks.\n    xticks : list or tuple, optional\n        A list of values to use for xticks.\n    colormap : str or matplotlib colormap, default None\n        Colormap to use for line colors.\n    axvlines : bool, optional\n        If true, vertical lines will be added at each xtick.\n    axvlines_kwds : keywords, optional\n        Options to be passed to axvline method for vertical lines.\n    sort_labels : bool, default False\n        Sort class_column labels, useful when assigning colors.\n    **kwargs\n        Options to pass to matplotlib plotting method.\n\n    Returns\n    -------\n    class:`matplotlib.axis.Axes`\n\n    Examples\n    --------\n\n    .. plot::\n        :context: close-figs\n\n        >>> df = pd.read_csv(\n        ...     'https://raw.github.com/pandas-dev/'\n        ...     'pandas/master/pandas/tests/io/data/csv/iris.csv'\n        ... )\n        >>> pd.plotting.parallel_coordinates(\n        ...     df, 'Name', color=('#556270', '#4ECDC4', '#C7F464')\n        ... )\n    "
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.parallel_coordinates(frame=frame, class_column=class_column, cols=cols, ax=ax, color=color, use_columns=use_columns, xticks=xticks, colormap=colormap, axvlines=axvlines, axvlines_kwds=axvlines_kwds, sort_labels=sort_labels, **kwargs)

def lag_plot(series, lag=1, ax=None, **kwds):
    '\n    Lag plot for time series.\n\n    Parameters\n    ----------\n    series : Time series\n    lag : lag of the scatter plot, default 1\n    ax : Matplotlib axis object, optional\n    **kwds\n        Matplotlib scatter method keyword arguments.\n\n    Returns\n    -------\n    class:`matplotlib.axis.Axes`\n\n    Examples\n    --------\n\n    Lag plots are most commonly used to look for patterns in time series data.\n\n    Given the following time series\n\n    .. plot::\n        :context: close-figs\n\n        >>> np.random.seed(5)\n        >>> x = np.cumsum(np.random.normal(loc=1, scale=5, size=50))\n        >>> s = pd.Series(x)\n        >>> s.plot()\n\n    A lag plot with ``lag=1`` returns\n\n    .. plot::\n        :context: close-figs\n\n        >>> pd.plotting.lag_plot(s, lag=1)\n    '
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.lag_plot(series=series, lag=lag, ax=ax, **kwds)

def autocorrelation_plot(series, ax=None, **kwargs):
    '\n    Autocorrelation plot for time series.\n\n    Parameters\n    ----------\n    series : Time series\n    ax : Matplotlib axis object, optional\n    **kwargs\n        Options to pass to matplotlib plotting method.\n\n    Returns\n    -------\n    class:`matplotlib.axis.Axes`\n\n    Examples\n    --------\n\n    The horizontal lines in the plot correspond to 95% and 99% confidence bands.\n\n    The dashed line is 99% confidence band.\n\n    .. plot::\n        :context: close-figs\n\n        >>> spacing = np.linspace(-9 * np.pi, 9 * np.pi, num=1000)\n        >>> s = pd.Series(0.7 * np.random.rand(1000) + 0.3 * np.sin(spacing))\n        >>> pd.plotting.autocorrelation_plot(s)\n    '
    plot_backend = _get_plot_backend('matplotlib')
    return plot_backend.autocorrelation_plot(series=series, ax=ax, **kwargs)

class _Options(dict):
    '\n    Stores pandas plotting options.\n\n    Allows for parameter aliasing so you can just use parameter names that are\n    the same as the plot function parameters, but is stored in a canonical\n    format that makes it easy to breakdown into groups later.\n    '
    _ALIASES = {'x_compat': 'xaxis.compat'}
    _DEFAULT_KEYS = ['xaxis.compat']

    def __init__(self, deprecated=False):
        self._deprecated = deprecated
        super().__setitem__('xaxis.compat', False)

    def __getitem__(self, key):
        key = self._get_canonical_key(key)
        if (key not in self):
            raise ValueError(f'{key} is not a valid pandas plotting option')
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        key = self._get_canonical_key(key)
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        key = self._get_canonical_key(key)
        if (key in self._DEFAULT_KEYS):
            raise ValueError(f'Cannot remove default parameter {key}')
        return super().__delitem__(key)

    def __contains__(self, key):
        key = self._get_canonical_key(key)
        return super().__contains__(key)

    def reset(self):
        '\n        Reset the option store to its initial state\n\n        Returns\n        -------\n        None\n        '
        self.__init__()

    def _get_canonical_key(self, key):
        return self._ALIASES.get(key, key)

    @contextmanager
    def use(self, key, value):
        '\n        Temporarily set a parameter value using the with statement.\n        Aliasing allowed.\n        '
        old_value = self[key]
        try:
            self[key] = value
            (yield self)
        finally:
            self[key] = old_value
plot_params = _Options()
