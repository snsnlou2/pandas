
import distutils.version
import importlib
import types
import warnings
VERSIONS = {'bs4': '4.6.0', 'bottleneck': '1.2.1', 'fsspec': '0.7.4', 'fastparquet': '0.3.2', 'gcsfs': '0.6.0', 'lxml.etree': '4.3.0', 'matplotlib': '2.2.3', 'numexpr': '2.6.8', 'odfpy': '1.3.0', 'openpyxl': '2.5.7', 'pandas_gbq': '0.12.0', 'pyarrow': '0.15.0', 'pytest': '5.0.1', 'pyxlsb': '1.0.6', 's3fs': '0.4.0', 'scipy': '1.2.0', 'sqlalchemy': '1.2.8', 'tables': '3.5.1', 'tabulate': '0.8.7', 'xarray': '0.12.3', 'xlrd': '1.2.0', 'xlwt': '1.3.0', 'xlsxwriter': '1.0.2', 'numba': '0.46.0'}
INSTALL_MAPPING = {'bs4': 'beautifulsoup4', 'bottleneck': 'Bottleneck', 'lxml.etree': 'lxml', 'odf': 'odfpy', 'pandas_gbq': 'pandas-gbq', 'sqlalchemy': 'SQLAlchemy', 'jinja2': 'Jinja2'}

def _get_version(module):
    version = getattr(module, '__version__', None)
    if (version is None):
        version = getattr(module, '__VERSION__', None)
    if (version is None):
        raise ImportError(f"Can't determine version for {module.__name__}")
    return version

def import_optional_dependency(name, extra='', raise_on_missing=True, on_version='raise'):
    '\n    Import an optional dependency.\n\n    By default, if a dependency is missing an ImportError with a nice\n    message will be raised. If a dependency is present, but too old,\n    we raise.\n\n    Parameters\n    ----------\n    name : str\n        The module name. This should be top-level only, so that the\n        version may be checked.\n    extra : str\n        Additional text to include in the ImportError message.\n    raise_on_missing : bool, default True\n        Whether to raise if the optional dependency is not found.\n        When False and the module is not present, None is returned.\n    on_version : str {\'raise\', \'warn\'}\n        What to do when a dependency\'s version is too old.\n\n        * raise : Raise an ImportError\n        * warn : Warn that the version is too old. Returns None\n        * ignore: Return the module, even if the version is too old.\n          It\'s expected that users validate the version locally when\n          using ``on_version="ignore"`` (see. ``io/html.py``)\n\n    Returns\n    -------\n    maybe_module : Optional[ModuleType]\n        The imported module, when found and the version is correct.\n        None is returned when the package is not found and `raise_on_missing`\n        is False, or when the package\'s version is too old and `on_version`\n        is ``\'warn\'``.\n    '
    package_name = INSTALL_MAPPING.get(name)
    install_name = (package_name if (package_name is not None) else name)
    msg = f"Missing optional dependency '{install_name}'. {extra} Use pip or conda to install {install_name}."
    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_on_missing:
            raise ImportError(msg) from None
        else:
            return None
    minimum_version = VERSIONS.get(name)
    if minimum_version:
        version = _get_version(module)
        if (distutils.version.LooseVersion(version) < minimum_version):
            assert (on_version in {'warn', 'raise', 'ignore'})
            msg = f"Pandas requires version '{minimum_version}' or newer of '{name}' (version '{version}' currently installed)."
            if (on_version == 'warn'):
                warnings.warn(msg, UserWarning)
                return None
            elif (on_version == 'raise'):
                raise ImportError(msg)
    return module
