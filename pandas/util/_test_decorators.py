
'\nThis module provides decorator functions which can be applied to test objects\nin order to skip those objects when certain conditions occur. A sample use case\nis to detect if the platform is missing ``matplotlib``. If so, any test objects\nwhich require ``matplotlib`` and decorated with ``@td.skip_if_no_mpl`` will be\nskipped by ``pytest`` during the execution of the test suite.\n\nTo illustrate, after importing this module:\n\nimport pandas.util._test_decorators as td\n\nThe decorators can be applied to classes:\n\n@td.skip_if_some_reason\nclass Foo:\n    ...\n\nOr individual functions:\n\n@td.skip_if_some_reason\ndef test_foo():\n    ...\n\nFor more information, refer to the ``pytest`` documentation on ``skipif``.\n'
from contextlib import contextmanager
from distutils.version import LooseVersion
import locale
from typing import Callable, Optional
import warnings
import numpy as np
import pytest
from pandas.compat import IS64, is_platform_windows
from pandas.compat._optional import import_optional_dependency
from pandas.core.computation.expressions import NUMEXPR_INSTALLED, USE_NUMEXPR

def safe_import(mod_name, min_version=None):
    '\n    Parameters\n    ----------\n    mod_name : str\n        Name of the module to be imported\n    min_version : str, default None\n        Minimum required version of the specified mod_name\n\n    Returns\n    -------\n    object\n        The imported module if successful, or False\n    '
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='aiohttp', message='.*decorator is deprecated since Python 3.8.*')
        try:
            mod = __import__(mod_name)
        except ImportError:
            return False
    if (not min_version):
        return mod
    else:
        import sys
        try:
            version = getattr(sys.modules[mod_name], '__version__')
        except AttributeError:
            version = getattr(sys.modules[mod_name], '__VERSION__')
        if version:
            from distutils.version import LooseVersion
            if (LooseVersion(version) >= LooseVersion(min_version)):
                return mod
    return False

def _skip_if_no_mpl():
    mod = safe_import('matplotlib')
    if mod:
        mod.use('Agg')
    else:
        return True

def _skip_if_has_locale():
    (lang, _) = locale.getlocale()
    if (lang is not None):
        return True

def _skip_if_not_us_locale():
    (lang, _) = locale.getlocale()
    if (lang != 'en_US'):
        return True

def _skip_if_no_scipy():
    return (not (safe_import('scipy.stats') and safe_import('scipy.sparse') and safe_import('scipy.interpolate') and safe_import('scipy.signal')))

def skip_if_installed(package):
    '\n    Skip a test if a package is installed.\n\n    Parameters\n    ----------\n    package : str\n        The name of the package.\n    '
    return pytest.mark.skipif(safe_import(package), reason=f'Skipping because {package} is installed.')

def skip_if_no(package, min_version=None):
    '\n    Generic function to help skip tests when required packages are not\n    present on the testing system.\n\n    This function returns a pytest mark with a skip condition that will be\n    evaluated during test collection. An attempt will be made to import the\n    specified ``package`` and optionally ensure it meets the ``min_version``\n\n    The mark can be used as either a decorator for a test function or to be\n    applied to parameters in pytest.mark.parametrize calls or parametrized\n    fixtures.\n\n    If the import and version check are unsuccessful, then the test function\n    (or test case when used in conjunction with parametrization) will be\n    skipped.\n\n    Parameters\n    ----------\n    package: str\n        The name of the required package.\n    min_version: str or None, default None\n        Optional minimum version of the package.\n\n    Returns\n    -------\n    _pytest.mark.structures.MarkDecorator\n        a pytest.mark.skipif to use as either a test decorator or a\n        parametrization mark.\n    '
    msg = f"Could not import '{package}'"
    if min_version:
        msg += f' satisfying a min_version of {min_version}'
    return pytest.mark.skipif((not safe_import(package, min_version=min_version)), reason=msg)
skip_if_no_mpl = pytest.mark.skipif(_skip_if_no_mpl(), reason='Missing matplotlib dependency')
skip_if_mpl = pytest.mark.skipif((not _skip_if_no_mpl()), reason='matplotlib is present')
skip_if_32bit = pytest.mark.skipif((not IS64), reason='skipping for 32 bit')
skip_if_windows = pytest.mark.skipif(is_platform_windows(), reason='Running on Windows')
skip_if_windows_python_3 = pytest.mark.skipif(is_platform_windows(), reason='not used on win32')
skip_if_has_locale = pytest.mark.skipif(_skip_if_has_locale(), reason=f'Specific locale is set {locale.getlocale()[0]}')
skip_if_not_us_locale = pytest.mark.skipif(_skip_if_not_us_locale(), reason=f'Specific locale is set {locale.getlocale()[0]}')
skip_if_no_scipy = pytest.mark.skipif(_skip_if_no_scipy(), reason='Missing SciPy requirement')
skip_if_no_ne = pytest.mark.skipif((not USE_NUMEXPR), reason=f'numexpr enabled->{USE_NUMEXPR}, installed->{NUMEXPR_INSTALLED}')

def skip_if_np_lt(ver_str, *args, reason=None):
    if (reason is None):
        reason = f'NumPy {ver_str} or greater required'
    return pytest.mark.skipif((np.__version__ < LooseVersion(ver_str)), *args, reason=reason)

def parametrize_fixture_doc(*args):
    '\n    Intended for use as a decorator for parametrized fixture,\n    this function will wrap the decorated function with a pytest\n    ``parametrize_fixture_doc`` mark. That mark will format\n    initial fixture docstring by replacing placeholders {0}, {1} etc\n    with parameters passed as arguments.\n\n    Parameters\n    ----------\n    args: iterable\n        Positional arguments for docstring.\n\n    Returns\n    -------\n    function\n        The decorated function wrapped within a pytest\n        ``parametrize_fixture_doc`` mark\n    '

    def documented_fixture(fixture):
        fixture.__doc__ = fixture.__doc__.format(*args)
        return fixture
    return documented_fixture

def check_file_leaks(func):
    '\n    Decorate a test function to check that we are not leaking file descriptors.\n    '
    with file_leak_context():
        return func

@contextmanager
def file_leak_context():
    '\n    ContextManager analogue to check_file_leaks.\n    '
    psutil = safe_import('psutil')
    if (not psutil):
        (yield)
    else:
        proc = psutil.Process()
        flist = proc.open_files()
        conns = proc.connections()
        (yield)
        flist2 = proc.open_files()
        flist_ex = [(x.path, x.fd) for x in flist]
        flist2_ex = [(x.path, x.fd) for x in flist2]
        assert (flist2_ex == flist_ex), (flist2, flist)
        conns2 = proc.connections()
        assert (conns2 == conns), (conns2, conns)

def async_mark():
    try:
        import_optional_dependency('pytest_asyncio')
        async_mark = pytest.mark.asyncio
    except ImportError:
        async_mark = pytest.mark.skip(reason='Missing dependency pytest-asyncio')
    return async_mark
