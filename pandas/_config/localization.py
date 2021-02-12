
'\nHelpers for configuring locale settings.\n\nName `localization` is chosen to avoid overlap with builtin `locale` module.\n'
from contextlib import contextmanager
import locale
import re
import subprocess
from pandas._config.config import options

@contextmanager
def set_locale(new_locale, lc_var=locale.LC_ALL):
    '\n    Context manager for temporarily setting a locale.\n\n    Parameters\n    ----------\n    new_locale : str or tuple\n        A string of the form <language_country>.<encoding>. For example to set\n        the current locale to US English with a UTF8 encoding, you would pass\n        "en_US.UTF-8".\n    lc_var : int, default `locale.LC_ALL`\n        The category of the locale being set.\n\n    Notes\n    -----\n    This is useful when you want to run a particular block of code under a\n    particular locale, without globally setting the locale. This probably isn\'t\n    thread-safe.\n    '
    current_locale = locale.getlocale()
    try:
        locale.setlocale(lc_var, new_locale)
        normalized_locale = locale.getlocale()
        if all(((x is not None) for x in normalized_locale)):
            (yield '.'.join(normalized_locale))
        else:
            (yield new_locale)
    finally:
        locale.setlocale(lc_var, current_locale)

def can_set_locale(lc, lc_var=locale.LC_ALL):
    '\n    Check to see if we can set a locale, and subsequently get the locale,\n    without raising an Exception.\n\n    Parameters\n    ----------\n    lc : str\n        The locale to attempt to set.\n    lc_var : int, default `locale.LC_ALL`\n        The category of the locale being set.\n\n    Returns\n    -------\n    bool\n        Whether the passed locale can be set\n    '
    try:
        with set_locale(lc, lc_var=lc_var):
            pass
    except (ValueError, locale.Error):
        return False
    else:
        return True

def _valid_locales(locales, normalize):
    '\n    Return a list of normalized locales that do not throw an ``Exception``\n    when set.\n\n    Parameters\n    ----------\n    locales : str\n        A string where each locale is separated by a newline.\n    normalize : bool\n        Whether to call ``locale.normalize`` on each locale.\n\n    Returns\n    -------\n    valid_locales : list\n        A list of valid locales.\n    '
    return [loc for loc in ((locale.normalize(loc.strip()) if normalize else loc.strip()) for loc in locales) if can_set_locale(loc)]

def _default_locale_getter():
    return subprocess.check_output(['locale -a'], shell=True)

def get_locales(prefix=None, normalize=True, locale_getter=_default_locale_getter):
    '\n    Get all the locales that are available on the system.\n\n    Parameters\n    ----------\n    prefix : str\n        If not ``None`` then return only those locales with the prefix\n        provided. For example to get all English language locales (those that\n        start with ``"en"``), pass ``prefix="en"``.\n    normalize : bool\n        Call ``locale.normalize`` on the resulting list of available locales.\n        If ``True``, only locales that can be set without throwing an\n        ``Exception`` are returned.\n    locale_getter : callable\n        The function to use to retrieve the current locales. This should return\n        a string with each locale separated by a newline character.\n\n    Returns\n    -------\n    locales : list of strings\n        A list of locale strings that can be set with ``locale.setlocale()``.\n        For example::\n\n            locale.setlocale(locale.LC_ALL, locale_string)\n\n    On error will return None (no locale available, e.g. Windows)\n\n    '
    try:
        raw_locales = locale_getter()
    except subprocess.CalledProcessError:
        return None
    try:
        raw_locales = raw_locales.split(b'\n')
        out_locales = []
        for x in raw_locales:
            try:
                out_locales.append(str(x, encoding=options.display.encoding))
            except UnicodeError:
                out_locales.append(str(x, encoding='windows-1252'))
    except TypeError:
        pass
    if (prefix is None):
        return _valid_locales(out_locales, normalize)
    pattern = re.compile(f'{prefix}.*')
    found = pattern.findall('\n'.join(out_locales))
    return _valid_locales(found, normalize)
