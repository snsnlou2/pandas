
from datetime import datetime
import importlib
import inspect
import logging
import os
import sys
import jinja2
from numpydoc.docscrape import NumpyDocString
from sphinx.ext.autosummary import _import_by_name
logger = logging.getLogger(__name__)
sys.setrecursionlimit(5000)
sys.path.insert(0, os.path.abspath('../sphinxext'))
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '../..', 'sphinxext')])
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.doctest', 'sphinx.ext.extlinks', 'sphinx.ext.todo', 'numpydoc', 'IPython.sphinxext.ipython_directive', 'IPython.sphinxext.ipython_console_highlighting', 'matplotlib.sphinxext.plot_directive', 'sphinx.ext.intersphinx', 'sphinx.ext.coverage', 'sphinx.ext.mathjax', 'sphinx.ext.ifconfig', 'sphinx.ext.linkcode', 'nbsphinx', 'contributors']
exclude_patterns = ['**.ipynb_checkpoints', '**/includes/**']
try:
    import nbconvert
except ImportError:
    logger.warn('nbconvert not installed. Skipping notebooks.')
    exclude_patterns.append('**/*.ipynb')
else:
    try:
        nbconvert.utils.pandoc.get_pandoc_version()
    except nbconvert.utils.pandoc.PandocMissing:
        logger.warn('Pandoc not installed. Skipping notebooks.')
        exclude_patterns.append('**/*.ipynb')
source_path = os.path.dirname(os.path.abspath(__file__))
pattern = os.environ.get('SPHINX_PATTERN')
if pattern:
    for (dirname, dirs, fnames) in os.walk(source_path):
        for fname in fnames:
            if (os.path.splitext(fname)[(- 1)] in ('.rst', '.ipynb')):
                fname = os.path.relpath(os.path.join(dirname, fname), source_path)
                if ((fname == 'index.rst') and (os.path.abspath(dirname) == source_path)):
                    continue
                elif ((pattern == '-api') and (dirname == 'reference')):
                    exclude_patterns.append(fname)
                elif ((pattern != '-api') and (fname != pattern)):
                    exclude_patterns.append(fname)
with open(os.path.join(source_path, 'index.rst.template')) as f:
    t = jinja2.Template(f.read())
with open(os.path.join(source_path, 'index.rst'), 'w') as f:
    f.write(t.render(include_api=(pattern is None), single_doc=(pattern if ((pattern is not None) and (pattern != '-api')) else None)))
autosummary_generate = (True if (pattern is None) else ['index'])
autodoc_typehints = 'none'
numpydoc_attributes_as_param_list = False
plot_include_source = True
plot_formats = [('png', 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_pre_code = 'import numpy as np\nimport pandas as pd'
nbsphinx_requirejs_path = ''
templates_path = ['../_templates']
source_suffix = ['.rst']
source_encoding = 'utf-8'
master_doc = 'index'
project = 'pandas'
copyright = f'2008-{datetime.now().year}, the pandas development team'
import pandas
version = str(pandas.__version__)
release = version
exclude_trees = []
pygments_style = 'sphinx'
html_theme = 'pydata_sphinx_theme'
html_theme_options = {'external_links': [], 'github_url': 'https://github.com/pandas-dev/pandas', 'twitter_url': 'https://twitter.com/pandas_dev', 'google_analytics_id': 'UA-27880019-2'}
html_logo = '../../web/pandas/static/img/pandas.svg'
html_static_path = ['_static']
html_css_files = ['css/getting_started.css', 'css/pandas.css']
html_favicon = '../../web/pandas/static/img/favicon.ico'
moved_api_pages = [('pandas.core.common.isnull', 'pandas.isna'), ('pandas.core.common.notnull', 'pandas.notna'), ('pandas.core.reshape.get_dummies', 'pandas.get_dummies'), ('pandas.tools.merge.concat', 'pandas.concat'), ('pandas.tools.merge.merge', 'pandas.merge'), ('pandas.tools.pivot.pivot_table', 'pandas.pivot_table'), ('pandas.tseries.tools.to_datetime', 'pandas.to_datetime'), ('pandas.io.clipboard.read_clipboard', 'pandas.read_clipboard'), ('pandas.io.excel.ExcelFile.parse', 'pandas.ExcelFile.parse'), ('pandas.io.excel.read_excel', 'pandas.read_excel'), ('pandas.io.gbq.read_gbq', 'pandas.read_gbq'), ('pandas.io.html.read_html', 'pandas.read_html'), ('pandas.io.json.read_json', 'pandas.read_json'), ('pandas.io.parsers.read_csv', 'pandas.read_csv'), ('pandas.io.parsers.read_fwf', 'pandas.read_fwf'), ('pandas.io.parsers.read_table', 'pandas.read_table'), ('pandas.io.pickle.read_pickle', 'pandas.read_pickle'), ('pandas.io.pytables.HDFStore.append', 'pandas.HDFStore.append'), ('pandas.io.pytables.HDFStore.get', 'pandas.HDFStore.get'), ('pandas.io.pytables.HDFStore.put', 'pandas.HDFStore.put'), ('pandas.io.pytables.HDFStore.select', 'pandas.HDFStore.select'), ('pandas.io.pytables.read_hdf', 'pandas.read_hdf'), ('pandas.io.sql.read_sql', 'pandas.read_sql'), ('pandas.io.sql.read_frame', 'pandas.read_frame'), ('pandas.io.sql.write_frame', 'pandas.write_frame'), ('pandas.io.stata.read_stata', 'pandas.read_stata')]
moved_classes = [('pandas.tseries.resample.Resampler', 'pandas.core.resample.Resampler'), ('pandas.formats.style.Styler', 'pandas.io.formats.style.Styler')]
for (old, new) in moved_classes:
    moved_api_pages.append((old, new))
    (mod, classname) = new.rsplit('.', 1)
    klass = getattr(importlib.import_module(mod), classname)
    methods = [x for x in dir(klass) if ((not x.startswith('_')) or (x in ('__iter__', '__array__')))]
    for method in methods:
        moved_api_pages.append((f'{old}.{method}', f'{new}.{method}'))
if (pattern is None):
    html_additional_pages = {('generated/' + page[0]): 'api_redirect.html' for page in moved_api_pages}
header = f'''.. currentmodule:: pandas

.. ipython:: python
   :suppress:

   import numpy as np
   import pandas as pd

   np.random.seed(123456)
   np.set_printoptions(precision=4, suppress=True)
   pd.options.display.max_rows = 15

   import os
   os.chdir(r'{os.path.dirname(os.path.dirname(__file__))}')
'''
html_context = {'redirects': {old: new for (old, new) in moved_api_pages}, 'header': header}
html_use_modindex = True
htmlhelp_basename = 'pandas'
nbsphinx_allow_errors = True
latex_elements = {}
latex_documents = [('index', 'pandas.tex', 'pandas: powerful Python data analysis toolkit', 'Wes McKinney and the Pandas Development Team', 'manual')]
if (pattern is None):
    intersphinx_mapping = {'dateutil': ('https://dateutil.readthedocs.io/en/latest/', None), 'matplotlib': ('https://matplotlib.org/', None), 'numpy': ('https://numpy.org/doc/stable/', None), 'pandas-gbq': ('https://pandas-gbq.readthedocs.io/en/latest/', None), 'py': ('https://pylib.readthedocs.io/en/latest/', None), 'python': ('https://docs.python.org/3/', None), 'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None), 'statsmodels': ('https://www.statsmodels.org/devel/', None), 'pyarrow': ('https://arrow.apache.org/docs/', None)}
extlinks = {'issue': ('https://github.com/pandas-dev/pandas/issues/%s', 'GH'), 'wiki': ('https://github.com/pandas-dev/pandas/wiki/%s', 'wiki ')}
ipython_warning_is_error = False
ipython_exec_lines = ['import numpy as np', 'import pandas as pd', 'pd.options.display.encoding="utf8"']
import sphinx
from sphinx.util import rpartition
from sphinx.ext.autodoc import AttributeDocumenter, Documenter, MethodDocumenter
from sphinx.ext.autosummary import Autosummary

class AccessorDocumenter(MethodDocumenter):
    '\n    Specialized Documenter subclass for accessors.\n    '
    objtype = 'accessor'
    directivetype = 'method'
    priority = 0.6

    def format_signature(self):
        return ''

class AccessorLevelDocumenter(Documenter):
    '\n    Specialized Documenter subclass for objects on accessor level (methods,\n    attributes).\n    '

    def resolve_name(self, modname, parents, path, base):
        if (modname is None):
            if path:
                mod_cls = path.rstrip('.')
            else:
                mod_cls = None
                mod_cls = self.env.temp_data.get('autodoc:class')
                if (mod_cls is None):
                    mod_cls = self.env.temp_data.get('py:class')
                if (mod_cls is None):
                    return (None, [])
            (modname, accessor) = rpartition(mod_cls, '.')
            (modname, cls) = rpartition(modname, '.')
            parents = [cls, accessor]
            if (not modname):
                modname = self.env.temp_data.get('autodoc:module')
            if (not modname):
                if (sphinx.__version__ > '1.3'):
                    modname = self.env.ref_context.get('py:module')
                else:
                    modname = self.env.temp_data.get('py:module')
        return (modname, (parents + [base]))

class AccessorAttributeDocumenter(AccessorLevelDocumenter, AttributeDocumenter):
    objtype = 'accessorattribute'
    directivetype = 'attribute'
    priority = 0.6

class AccessorMethodDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    objtype = 'accessormethod'
    directivetype = 'method'
    priority = 0.6

class AccessorCallableDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    '\n    This documenter lets us removes .__call__ from the method signature for\n    callable accessors like Series.plot\n    '
    objtype = 'accessorcallable'
    directivetype = 'method'
    priority = 0.5

    def format_name(self):
        return MethodDocumenter.format_name(self).rstrip('.__call__')

class PandasAutosummary(Autosummary):
    '\n    This alternative autosummary class lets us override the table summary for\n    Series.plot and DataFrame.plot in the API docs.\n    '

    def _replace_pandas_items(self, display_name, sig, summary, real_name):
        if (display_name == 'DataFrame.plot'):
            sig = '([x, y, kind, ax, ....])'
            summary = 'DataFrame plotting accessor and method'
        elif (display_name == 'Series.plot'):
            sig = '([kind, ax, figsize, ....])'
            summary = 'Series plotting accessor and method'
        return (display_name, sig, summary, real_name)

    @staticmethod
    def _is_deprecated(real_name):
        try:
            (obj, parent, modname) = _import_by_name(real_name)
        except ImportError:
            return False
        doc = NumpyDocString((obj.__doc__ or ''))
        summary = ''.join((doc['Summary'] + doc['Extended Summary']))
        return ('.. deprecated::' in summary)

    def _add_deprecation_prefixes(self, items):
        for item in items:
            (display_name, sig, summary, real_name) = item
            if self._is_deprecated(real_name):
                summary = f'(DEPRECATED) {summary}'
            (yield (display_name, sig, summary, real_name))

    def get_items(self, names):
        items = Autosummary.get_items(self, names)
        items = [self._replace_pandas_items(*item) for item in items]
        items = list(self._add_deprecation_prefixes(items))
        return items

def linkcode_resolve(domain, info):
    '\n    Determine the URL corresponding to Python object\n    '
    if (domain != 'py'):
        return None
    modname = info['module']
    fullname = info['fullname']
    submod = sys.modules.get(modname)
    if (submod is None):
        return None
    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if (not fn):
        return None
    try:
        (source, lineno) = inspect.getsourcelines(obj)
    except OSError:
        lineno = None
    if lineno:
        linespec = f'#L{lineno}-L{((lineno + len(source)) - 1)}'
    else:
        linespec = ''
    fn = os.path.relpath(fn, start=os.path.dirname(pandas.__file__))
    if ('+' in pandas.__version__):
        return f'https://github.com/pandas-dev/pandas/blob/master/pandas/{fn}{linespec}'
    else:
        return f'https://github.com/pandas-dev/pandas/blob/v{pandas.__version__}/pandas/{fn}{linespec}'

def remove_flags_docstring(app, what, name, obj, options, lines):
    if ((what == 'attribute') and name.endswith('.flags')):
        del lines[:]

def process_class_docstrings(app, what, name, obj, options, lines):
    '\n    For those classes for which we use ::\n\n    :template: autosummary/class_without_autosummary.rst\n\n    the documented attributes/methods have to be listed in the class\n    docstring. However, if one of those lists is empty, we use \'None\',\n    which then generates warnings in sphinx / ugly html output.\n    This "autodoc-process-docstring" event connector removes that part\n    from the processed docstring.\n\n    '
    if (what == 'class'):
        joined = '\n'.join(lines)
        templates = ['.. rubric:: Attributes\n\n.. autosummary::\n   :toctree:\n\n   None\n', '.. rubric:: Methods\n\n.. autosummary::\n   :toctree:\n\n   None\n']
        for template in templates:
            if (template in joined):
                joined = joined.replace(template, '')
        lines[:] = joined.split('\n')
_BUSINED_ALIASES = [('pandas.tseries.offsets.' + name) for name in ['BDay', 'CDay', 'BMonthEnd', 'BMonthBegin', 'CBMonthEnd', 'CBMonthBegin']]

def process_business_alias_docstrings(app, what, name, obj, options, lines):
    '\n    Starting with sphinx 3.4, the "autodoc-process-docstring" event also\n    gets called for alias classes. This results in numpydoc adding the\n    methods/attributes to the docstring, which we don\'t want (+ this\n    causes warnings with sphinx).\n    '
    if (name in _BUSINED_ALIASES):
        lines[:] = []
suppress_warnings = ['app.add_directive']
if pattern:
    suppress_warnings.append('ref.ref')

def rstjinja(app, docname, source):
    '\n    Render our pages as a jinja template for fancy templating goodness.\n    '
    if (app.builder.format != 'html'):
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered

def setup(app):
    app.connect('source-read', rstjinja)
    app.connect('autodoc-process-docstring', remove_flags_docstring)
    app.connect('autodoc-process-docstring', process_class_docstrings)
    app.connect('autodoc-process-docstring', process_business_alias_docstrings)
    app.add_autodocumenter(AccessorDocumenter)
    app.add_autodocumenter(AccessorAttributeDocumenter)
    app.add_autodocumenter(AccessorMethodDocumenter)
    app.add_autodocumenter(AccessorCallableDocumenter)
    app.add_directive('autosummary', PandasAutosummary)
