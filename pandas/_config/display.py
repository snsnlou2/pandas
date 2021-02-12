
'\nUnopinionated display configuration.\n'
import locale
import sys
from pandas._config import config as cf
_initial_defencoding = None

def detect_console_encoding():
    '\n    Try to find the most capable encoding supported by the console.\n    slightly modified from the way IPython handles the same issue.\n    '
    global _initial_defencoding
    encoding = None
    try:
        encoding = (sys.stdout.encoding or sys.stdin.encoding)
    except (AttributeError, OSError):
        pass
    if ((not encoding) or ('ascii' in encoding.lower())):
        try:
            encoding = locale.getpreferredencoding()
        except locale.Error:
            pass
    if ((not encoding) or ('ascii' in encoding.lower())):
        encoding = sys.getdefaultencoding()
    if (not _initial_defencoding):
        _initial_defencoding = sys.getdefaultencoding()
    return encoding
pc_encoding_doc = '\n: str/unicode\n    Defaults to the detected encoding of the console.\n    Specifies the encoding to be used for strings returned by to_string,\n    these are generally strings meant to be displayed on the console.\n'
with cf.config_prefix('display'):
    cf.register_option('encoding', detect_console_encoding(), pc_encoding_doc, validator=cf.is_text)
