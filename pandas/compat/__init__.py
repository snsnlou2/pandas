
'\ncompat\n======\n\nCross-compatible functions for different versions of Python.\n\nOther items:\n* platform checker\n'
import platform
import sys
import warnings
from pandas._typing import F
PY38 = (sys.version_info >= (3, 8))
PY39 = (sys.version_info >= (3, 9))
PYPY = (platform.python_implementation() == 'PyPy')
IS64 = (sys.maxsize > (2 ** 32))

def set_function_name(f, name, cls):
    '\n    Bind the name/qualname attributes of the function.\n    '
    f.__name__ = name
    f.__qualname__ = f'{cls.__name__}.{name}'
    f.__module__ = cls.__module__
    return f

def is_platform_little_endian():
    '\n    Checking if the running platform is little endian.\n\n    Returns\n    -------\n    bool\n        True if the running platform is little endian.\n    '
    return (sys.byteorder == 'little')

def is_platform_windows():
    '\n    Checking if the running platform is windows.\n\n    Returns\n    -------\n    bool\n        True if the running platform is windows.\n    '
    return (sys.platform in ['win32', 'cygwin'])

def is_platform_linux():
    '\n    Checking if the running platform is linux.\n\n    Returns\n    -------\n    bool\n        True if the running platform is linux.\n    '
    return (sys.platform == 'linux')

def is_platform_mac():
    '\n    Checking if the running platform is mac.\n\n    Returns\n    -------\n    bool\n        True if the running platform is mac.\n    '
    return (sys.platform == 'darwin')

def import_lzma():
    '\n    Importing the `lzma` module.\n\n    Warns\n    -----\n    When the `lzma` module is not available.\n    '
    try:
        import lzma
        return lzma
    except ImportError:
        msg = 'Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.'
        warnings.warn(msg)

def get_lzma_file(lzma):
    "\n    Importing the `LZMAFile` class from the `lzma` module.\n\n    Returns\n    -------\n    class\n        The `LZMAFile` class from the `lzma` module.\n\n    Raises\n    ------\n    RuntimeError\n        If the `lzma` module was not imported correctly, or didn't exist.\n    "
    if (lzma is None):
        raise RuntimeError('lzma module not available. A Python re-install with the proper dependencies, might be required to solve this issue.')
    return lzma.LZMAFile
