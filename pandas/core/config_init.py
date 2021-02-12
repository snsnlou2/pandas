
'\nThis module is imported from the pandas package __init__.py file\nin order to ensure that the core.config options registered here will\nbe available as soon as the user loads the package. if register_option\nis invoked inside specific modules, they will not be registered until that\nmodule is imported, which may or may not be a problem.\n\nIf you need to make sure options are available even before a certain\nmodule is imported, register them here rather than in the module.\n\n'
import warnings
import pandas._config.config as cf
from pandas._config.config import is_bool, is_callable, is_instance_factory, is_int, is_nonnegative_int, is_one_of_factory, is_text
use_bottleneck_doc = '\n: bool\n    Use the bottleneck library to accelerate if it is installed,\n    the default is True\n    Valid values: False,True\n'

def use_bottleneck_cb(key):
    from pandas.core import nanops
    nanops.set_use_bottleneck(cf.get_option(key))
use_numexpr_doc = '\n: bool\n    Use the numexpr library to accelerate computation if it is installed,\n    the default is True\n    Valid values: False,True\n'

def use_numexpr_cb(key):
    from pandas.core.computation import expressions
    expressions.set_use_numexpr(cf.get_option(key))
use_numba_doc = '\n: bool\n    Use the numba engine option for select operations if it is installed,\n    the default is False\n    Valid values: False,True\n'

def use_numba_cb(key):
    from pandas.core.util import numba_
    numba_.set_use_numba(cf.get_option(key))
with cf.config_prefix('compute'):
    cf.register_option('use_bottleneck', True, use_bottleneck_doc, validator=is_bool, cb=use_bottleneck_cb)
    cf.register_option('use_numexpr', True, use_numexpr_doc, validator=is_bool, cb=use_numexpr_cb)
    cf.register_option('use_numba', False, use_numba_doc, validator=is_bool, cb=use_numba_cb)
pc_precision_doc = '\n: int\n    Floating point output precision in terms of number of places after the\n    decimal, for regular formatting as well as scientific notation. Similar\n    to ``precision`` in :meth:`numpy.set_printoptions`.\n'
pc_colspace_doc = '\n: int\n    Default space for DataFrame columns.\n'
pc_max_rows_doc = "\n: int\n    If max_rows is exceeded, switch to truncate view. Depending on\n    `large_repr`, objects are either centrally truncated or printed as\n    a summary view. 'None' value means unlimited.\n\n    In case python/IPython is running in a terminal and `large_repr`\n    equals 'truncate' this can be set to 0 and pandas will auto-detect\n    the height of the terminal and print a truncated object which fits\n    the screen height. The IPython notebook, IPython qtconsole, or\n    IDLE do not run in a terminal and hence it is not possible to do\n    correct auto-detection.\n"
pc_min_rows_doc = '\n: int\n    The numbers of rows to show in a truncated view (when `max_rows` is\n    exceeded). Ignored when `max_rows` is set to None or 0. When set to\n    None, follows the value of `max_rows`.\n'
pc_max_cols_doc = "\n: int\n    If max_cols is exceeded, switch to truncate view. Depending on\n    `large_repr`, objects are either centrally truncated or printed as\n    a summary view. 'None' value means unlimited.\n\n    In case python/IPython is running in a terminal and `large_repr`\n    equals 'truncate' this can be set to 0 and pandas will auto-detect\n    the width of the terminal and print a truncated object which fits\n    the screen width. The IPython notebook, IPython qtconsole, or IDLE\n    do not run in a terminal and hence it is not possible to do\n    correct auto-detection.\n"
pc_max_categories_doc = '\n: int\n    This sets the maximum number of categories pandas should output when\n    printing out a `Categorical` or a Series of dtype "category".\n'
pc_max_info_cols_doc = '\n: int\n    max_info_columns is used in DataFrame.info method to decide if\n    per column information will be printed.\n'
pc_nb_repr_h_doc = '\n: boolean\n    When True, IPython notebook will use html representation for\n    pandas objects (if it is available).\n'
pc_pprint_nest_depth = '\n: int\n    Controls the number of nested levels to process when pretty-printing\n'
pc_multi_sparse_doc = '\n: boolean\n    "sparsify" MultiIndex display (don\'t display repeated\n    elements in outer levels within groups)\n'
float_format_doc = '\n: callable\n    The callable should accept a floating point number and return\n    a string with the desired format of the number. This is used\n    in some places like SeriesFormatter.\n    See formats.format.EngFormatter for an example.\n'
max_colwidth_doc = '\n: int or None\n    The maximum width in characters of a column in the repr of\n    a pandas data structure. When the column overflows, a "..."\n    placeholder is embedded in the output. A \'None\' value means unlimited.\n'
colheader_justify_doc = "\n: 'left'/'right'\n    Controls the justification of column headers. used by DataFrameFormatter.\n"
pc_expand_repr_doc = '\n: boolean\n    Whether to print out the full DataFrame repr for wide DataFrames across\n    multiple lines, `max_columns` is still respected, but the output will\n    wrap-around across multiple "pages" if its width exceeds `display.width`.\n'
pc_show_dimensions_doc = "\n: boolean or 'truncate'\n    Whether to print out dimensions at the end of DataFrame repr.\n    If 'truncate' is specified, only print out the dimensions if the\n    frame is truncated (e.g. not display all rows and/or columns)\n"
pc_east_asian_width_doc = '\n: boolean\n    Whether to use the Unicode East Asian Width to calculate the display text\n    width.\n    Enabling this may affect to the performance (default: False)\n'
pc_ambiguous_as_wide_doc = '\n: boolean\n    Whether to handle Unicode characters belong to Ambiguous as Wide (width=2)\n    (default: False)\n'
pc_latex_repr_doc = '\n: boolean\n    Whether to produce a latex DataFrame representation for jupyter\n    environments that support it.\n    (default: False)\n'
pc_table_schema_doc = '\n: boolean\n    Whether to publish a Table Schema representation for frontends\n    that support it.\n    (default: False)\n'
pc_html_border_doc = '\n: int\n    A ``border=value`` attribute is inserted in the ``<table>`` tag\n    for the DataFrame HTML repr.\n'
pc_html_use_mathjax_doc = ': boolean\n    When True, Jupyter notebook will process table contents using MathJax,\n    rendering mathematical expressions enclosed by the dollar symbol.\n    (default: True)\n'
pc_width_doc = '\n: int\n    Width of the display in characters. In case python/IPython is running in\n    a terminal this can be set to None and pandas will correctly auto-detect\n    the width.\n    Note that the IPython notebook, IPython qtconsole, or IDLE do not run in a\n    terminal and hence it is not possible to correctly detect the width.\n'
pc_chop_threshold_doc = '\n: float or None\n    if set to a float value, all float values smaller then the given threshold\n    will be displayed as exactly 0 by repr and friends.\n'
pc_max_seq_items = '\n: int or None\n    When pretty-printing a long sequence, no more then `max_seq_items`\n    will be printed. If items are omitted, they will be denoted by the\n    addition of "..." to the resulting string.\n\n    If set to None, the number of items to be printed is unlimited.\n'
pc_max_info_rows_doc = '\n: int or None\n    df.info() will usually show null-counts for each column.\n    For large frames this can be quite slow. max_info_rows and max_info_cols\n    limit this null check only to frames with smaller dimensions than\n    specified.\n'
pc_large_repr_doc = "\n: 'truncate'/'info'\n    For DataFrames exceeding max_rows/max_cols, the repr (and HTML repr) can\n    show a truncated table (the default from 0.13), or switch to the view from\n    df.info() (the behaviour in earlier versions of pandas).\n"
pc_memory_usage_doc = "\n: bool, string or None\n    This specifies if the memory usage of a DataFrame should be displayed when\n    df.info() is called. Valid values True,False,'deep'\n"
pc_latex_escape = '\n: bool\n    This specifies if the to_latex method of a Dataframe uses escapes special\n    characters.\n    Valid values: False,True\n'
pc_latex_longtable = '\n:bool\n    This specifies if the to_latex method of a Dataframe uses the longtable\n    format.\n    Valid values: False,True\n'
pc_latex_multicolumn = '\n: bool\n    This specifies if the to_latex method of a Dataframe uses multicolumns\n    to pretty-print MultiIndex columns.\n    Valid values: False,True\n'
pc_latex_multicolumn_format = "\n: string\n    This specifies the format for multicolumn headers.\n    Can be surrounded with '|'.\n    Valid values: 'l', 'c', 'r', 'p{<width>}'\n"
pc_latex_multirow = '\n: bool\n    This specifies if the to_latex method of a Dataframe uses multirows\n    to pretty-print MultiIndex rows.\n    Valid values: False,True\n'

def table_schema_cb(key):
    from pandas.io.formats.printing import enable_data_resource_formatter
    enable_data_resource_formatter(cf.get_option(key))

def is_terminal():
    '\n    Detect if Python is running in a terminal.\n\n    Returns True if Python is running in a terminal or False if not.\n    '
    try:
        ip = get_ipython()
    except NameError:
        return True
    else:
        if hasattr(ip, 'kernel'):
            return False
        else:
            return True
with cf.config_prefix('display'):
    cf.register_option('precision', 6, pc_precision_doc, validator=is_nonnegative_int)
    cf.register_option('float_format', None, float_format_doc, validator=is_one_of_factory([None, is_callable]))
    cf.register_option('column_space', 12, validator=is_int)
    cf.register_option('max_info_rows', 1690785, pc_max_info_rows_doc, validator=is_instance_factory((int, type(None))))
    cf.register_option('max_rows', 60, pc_max_rows_doc, validator=is_nonnegative_int)
    cf.register_option('min_rows', 10, pc_min_rows_doc, validator=is_instance_factory([type(None), int]))
    cf.register_option('max_categories', 8, pc_max_categories_doc, validator=is_int)

    def _deprecate_negative_int_max_colwidth(key):
        value = cf.get_option(key)
        if ((value is not None) and (value < 0)):
            warnings.warn('Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.', FutureWarning, stacklevel=4)
    cf.register_option('max_colwidth', 50, max_colwidth_doc, validator=is_instance_factory([type(None), int]), cb=_deprecate_negative_int_max_colwidth)
    if is_terminal():
        max_cols = 0
    else:
        max_cols = 20
    cf.register_option('max_columns', max_cols, pc_max_cols_doc, validator=is_nonnegative_int)
    cf.register_option('large_repr', 'truncate', pc_large_repr_doc, validator=is_one_of_factory(['truncate', 'info']))
    cf.register_option('max_info_columns', 100, pc_max_info_cols_doc, validator=is_int)
    cf.register_option('colheader_justify', 'right', colheader_justify_doc, validator=is_text)
    cf.register_option('notebook_repr_html', True, pc_nb_repr_h_doc, validator=is_bool)
    cf.register_option('pprint_nest_depth', 3, pc_pprint_nest_depth, validator=is_int)
    cf.register_option('multi_sparse', True, pc_multi_sparse_doc, validator=is_bool)
    cf.register_option('expand_frame_repr', True, pc_expand_repr_doc)
    cf.register_option('show_dimensions', 'truncate', pc_show_dimensions_doc, validator=is_one_of_factory([True, False, 'truncate']))
    cf.register_option('chop_threshold', None, pc_chop_threshold_doc)
    cf.register_option('max_seq_items', 100, pc_max_seq_items)
    cf.register_option('width', 80, pc_width_doc, validator=is_instance_factory([type(None), int]))
    cf.register_option('memory_usage', True, pc_memory_usage_doc, validator=is_one_of_factory([None, True, False, 'deep']))
    cf.register_option('unicode.east_asian_width', False, pc_east_asian_width_doc, validator=is_bool)
    cf.register_option('unicode.ambiguous_as_wide', False, pc_east_asian_width_doc, validator=is_bool)
    cf.register_option('latex.repr', False, pc_latex_repr_doc, validator=is_bool)
    cf.register_option('latex.escape', True, pc_latex_escape, validator=is_bool)
    cf.register_option('latex.longtable', False, pc_latex_longtable, validator=is_bool)
    cf.register_option('latex.multicolumn', True, pc_latex_multicolumn, validator=is_bool)
    cf.register_option('latex.multicolumn_format', 'l', pc_latex_multicolumn, validator=is_text)
    cf.register_option('latex.multirow', False, pc_latex_multirow, validator=is_bool)
    cf.register_option('html.table_schema', False, pc_table_schema_doc, validator=is_bool, cb=table_schema_cb)
    cf.register_option('html.border', 1, pc_html_border_doc, validator=is_int)
    cf.register_option('html.use_mathjax', True, pc_html_use_mathjax_doc, validator=is_bool)
tc_sim_interactive_doc = '\n: boolean\n    Whether to simulate interactive mode for purposes of testing\n'
with cf.config_prefix('mode'):
    cf.register_option('sim_interactive', False, tc_sim_interactive_doc)
use_inf_as_null_doc = '\n: boolean\n    use_inf_as_null had been deprecated and will be removed in a future\n    version. Use `use_inf_as_na` instead.\n'
use_inf_as_na_doc = '\n: boolean\n    True means treat None, NaN, INF, -INF as NA (old way),\n    False means None and NaN are null, but INF, -INF are not NA\n    (new way).\n'

def use_inf_as_na_cb(key):
    from pandas.core.dtypes.missing import _use_inf_as_na
    _use_inf_as_na(key)
with cf.config_prefix('mode'):
    cf.register_option('use_inf_as_na', False, use_inf_as_na_doc, cb=use_inf_as_na_cb)
    cf.register_option('use_inf_as_null', False, use_inf_as_null_doc, cb=use_inf_as_na_cb)
cf.deprecate_option('mode.use_inf_as_null', msg=use_inf_as_null_doc, rkey='mode.use_inf_as_na')
chained_assignment = '\n: string\n    Raise an exception, warn, or no action if trying to use chained assignment,\n    The default is warn\n'
with cf.config_prefix('mode'):
    cf.register_option('chained_assignment', 'warn', chained_assignment, validator=is_one_of_factory([None, 'warn', 'raise']))
reader_engine_doc = "\n: string\n    The default Excel reader engine for '{ext}' files. Available options:\n    auto, {others}.\n"
_xls_options = ['xlrd']
_xlsm_options = ['xlrd', 'openpyxl']
_xlsx_options = ['xlrd', 'openpyxl']
_ods_options = ['odf']
_xlsb_options = ['pyxlsb']
with cf.config_prefix('io.excel.xls'):
    cf.register_option('reader', 'auto', reader_engine_doc.format(ext='xls', others=', '.join(_xls_options)), validator=str)
with cf.config_prefix('io.excel.xlsm'):
    cf.register_option('reader', 'auto', reader_engine_doc.format(ext='xlsm', others=', '.join(_xlsm_options)), validator=str)
with cf.config_prefix('io.excel.xlsx'):
    cf.register_option('reader', 'auto', reader_engine_doc.format(ext='xlsx', others=', '.join(_xlsx_options)), validator=str)
with cf.config_prefix('io.excel.ods'):
    cf.register_option('reader', 'auto', reader_engine_doc.format(ext='ods', others=', '.join(_ods_options)), validator=str)
with cf.config_prefix('io.excel.xlsb'):
    cf.register_option('reader', 'auto', reader_engine_doc.format(ext='xlsb', others=', '.join(_xlsb_options)), validator=str)
writer_engine_doc = "\n: string\n    The default Excel writer engine for '{ext}' files. Available options:\n    auto, {others}.\n"
_xls_options = ['xlwt']
_xlsm_options = ['openpyxl']
_xlsx_options = ['openpyxl', 'xlsxwriter']
_ods_options = ['odf']
with cf.config_prefix('io.excel.xls'):
    cf.register_option('writer', 'auto', writer_engine_doc.format(ext='xls', others=', '.join(_xls_options)), validator=str)
cf.deprecate_option('io.excel.xls.writer', msg='As the xlwt package is no longer maintained, the xlwt engine will be removed in a future version of pandas. This is the only engine in pandas that supports writing in the xls format. Install openpyxl and write to an xlsx file instead.')
with cf.config_prefix('io.excel.xlsm'):
    cf.register_option('writer', 'auto', writer_engine_doc.format(ext='xlsm', others=', '.join(_xlsm_options)), validator=str)
with cf.config_prefix('io.excel.xlsx'):
    cf.register_option('writer', 'auto', writer_engine_doc.format(ext='xlsx', others=', '.join(_xlsx_options)), validator=str)
with cf.config_prefix('io.excel.ods'):
    cf.register_option('writer', 'auto', writer_engine_doc.format(ext='ods', others=', '.join(_ods_options)), validator=str)
parquet_engine_doc = "\n: string\n    The default parquet reader/writer engine. Available options:\n    'auto', 'pyarrow', 'fastparquet', the default is 'auto'\n"
with cf.config_prefix('io.parquet'):
    cf.register_option('engine', 'auto', parquet_engine_doc, validator=is_one_of_factory(['auto', 'pyarrow', 'fastparquet']))
plotting_backend_doc = '\n: str\n    The plotting backend to use. The default value is "matplotlib", the\n    backend provided with pandas. Other backends can be specified by\n    providing the name of the module that implements the backend.\n'

def register_plotting_backend_cb(key):
    if (key == 'matplotlib'):
        return
    from pandas.plotting._core import _get_plot_backend
    _get_plot_backend(key)
with cf.config_prefix('plotting'):
    cf.register_option('backend', defval='matplotlib', doc=plotting_backend_doc, validator=register_plotting_backend_cb)
register_converter_doc = "\n: bool or 'auto'.\n    Whether to register converters with matplotlib's units registry for\n    dates, times, datetimes, and Periods. Toggling to False will remove\n    the converters, restoring any converters that pandas overwrote.\n"

def register_converter_cb(key):
    from pandas.plotting import deregister_matplotlib_converters, register_matplotlib_converters
    if cf.get_option(key):
        register_matplotlib_converters()
    else:
        deregister_matplotlib_converters()
with cf.config_prefix('plotting.matplotlib'):
    cf.register_option('register_converters', 'auto', register_converter_doc, validator=is_one_of_factory(['auto', True, False]), cb=register_converter_cb)
