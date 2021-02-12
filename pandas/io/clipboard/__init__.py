
'\nPyperclip\n\nA cross-platform clipboard module for Python,\nwith copy & paste functions for plain text.\nBy Al Sweigart al@inventwithpython.com\nBSD License\n\nUsage:\n  import pyperclip\n  pyperclip.copy(\'The text to be copied to the clipboard.\')\n  spam = pyperclip.paste()\n\n  if not pyperclip.is_available():\n    print("Copy functionality unavailable!")\n\nOn Windows, no additional modules are needed.\nOn Mac, the pyobjc module is used, falling back to the pbcopy and pbpaste cli\n    commands. (These commands should come with OS X.).\nOn Linux, install xclip or xsel via package manager. For example, in Debian:\n    sudo apt-get install xclip\n    sudo apt-get install xsel\n\nOtherwise on Linux, you will need the PyQt5 modules installed.\n\nThis module does not work with PyGObject yet.\n\nCygwin is currently not supported.\n\nSecurity Note: This module runs programs with these names:\n    - which\n    - where\n    - pbcopy\n    - pbpaste\n    - xclip\n    - xsel\n    - klipper\n    - qdbus\nA malicious user could rename or add programs with these names, tricking\nPyperclip into running them with whatever permissions the Python process has.\n\n'
__version__ = '1.7.0'
import contextlib
import ctypes
from ctypes import c_size_t, c_wchar, c_wchar_p, get_errno, sizeof
import distutils.spawn
import os
import platform
import subprocess
import time
import warnings
HAS_DISPLAY = os.getenv('DISPLAY', False)
EXCEPT_MSG = '\n    Pyperclip could not find a copy/paste mechanism for your system.\n    For more information, please visit\n    https://pyperclip.readthedocs.io/en/latest/introduction.html#not-implemented-error\n    '
ENCODING = 'utf-8'
if (platform.system() == 'Windows'):
    WHICH_CMD = 'where'
else:
    WHICH_CMD = 'which'

def _executable_exists(name):
    return (subprocess.call([WHICH_CMD, name], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0)

class PyperclipException(RuntimeError):
    pass

class PyperclipWindowsException(PyperclipException):

    def __init__(self, message):
        message += f' ({ctypes.WinError()})'
        super().__init__(message)

def _stringifyText(text):
    acceptedTypes = (str, int, float, bool)
    if (not isinstance(text, acceptedTypes)):
        raise PyperclipException(f'only str, int, float, and bool values can be copied to the clipboard, not {type(text).__name__}')
    return str(text)

def init_osx_pbcopy_clipboard():

    def copy_osx_pbcopy(text):
        text = _stringifyText(text)
        p = subprocess.Popen(['pbcopy', 'w'], stdin=subprocess.PIPE, close_fds=True)
        p.communicate(input=text.encode(ENCODING))

    def paste_osx_pbcopy():
        p = subprocess.Popen(['pbpaste', 'r'], stdout=subprocess.PIPE, close_fds=True)
        (stdout, stderr) = p.communicate()
        return stdout.decode(ENCODING)
    return (copy_osx_pbcopy, paste_osx_pbcopy)

def init_osx_pyobjc_clipboard():

    def copy_osx_pyobjc(text):
        'Copy string argument to clipboard'
        text = _stringifyText(text)
        newStr = Foundation.NSString.stringWithString_(text).nsstring()
        newData = newStr.dataUsingEncoding_(Foundation.NSUTF8StringEncoding)
        board = AppKit.NSPasteboard.generalPasteboard()
        board.declareTypes_owner_([AppKit.NSStringPboardType], None)
        board.setData_forType_(newData, AppKit.NSStringPboardType)

    def paste_osx_pyobjc():
        'Returns contents of clipboard'
        board = AppKit.NSPasteboard.generalPasteboard()
        content = board.stringForType_(AppKit.NSStringPboardType)
        return content
    return (copy_osx_pyobjc, paste_osx_pyobjc)

def init_qt_clipboard():
    global QApplication
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            from PyQt4.QtGui import QApplication
    app = QApplication.instance()
    if (app is None):
        app = QApplication([])

    def copy_qt(text):
        text = _stringifyText(text)
        cb = app.clipboard()
        cb.setText(text)

    def paste_qt() -> str:
        cb = app.clipboard()
        return str(cb.text())
    return (copy_qt, paste_qt)

def init_xclip_clipboard():
    DEFAULT_SELECTION = 'c'
    PRIMARY_SELECTION = 'p'

    def copy_xclip(text, primary=False):
        text = _stringifyText(text)
        selection = DEFAULT_SELECTION
        if primary:
            selection = PRIMARY_SELECTION
        p = subprocess.Popen(['xclip', '-selection', selection], stdin=subprocess.PIPE, close_fds=True)
        p.communicate(input=text.encode(ENCODING))

    def paste_xclip(primary=False):
        selection = DEFAULT_SELECTION
        if primary:
            selection = PRIMARY_SELECTION
        p = subprocess.Popen(['xclip', '-selection', selection, '-o'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        (stdout, stderr) = p.communicate()
        return stdout.decode(ENCODING)
    return (copy_xclip, paste_xclip)

def init_xsel_clipboard():
    DEFAULT_SELECTION = '-b'
    PRIMARY_SELECTION = '-p'

    def copy_xsel(text, primary=False):
        text = _stringifyText(text)
        selection_flag = DEFAULT_SELECTION
        if primary:
            selection_flag = PRIMARY_SELECTION
        p = subprocess.Popen(['xsel', selection_flag, '-i'], stdin=subprocess.PIPE, close_fds=True)
        p.communicate(input=text.encode(ENCODING))

    def paste_xsel(primary=False):
        selection_flag = DEFAULT_SELECTION
        if primary:
            selection_flag = PRIMARY_SELECTION
        p = subprocess.Popen(['xsel', selection_flag, '-o'], stdout=subprocess.PIPE, close_fds=True)
        (stdout, stderr) = p.communicate()
        return stdout.decode(ENCODING)
    return (copy_xsel, paste_xsel)

def init_klipper_clipboard():

    def copy_klipper(text):
        text = _stringifyText(text)
        p = subprocess.Popen(['qdbus', 'org.kde.klipper', '/klipper', 'setClipboardContents', text.encode(ENCODING)], stdin=subprocess.PIPE, close_fds=True)
        p.communicate(input=None)

    def paste_klipper():
        p = subprocess.Popen(['qdbus', 'org.kde.klipper', '/klipper', 'getClipboardContents'], stdout=subprocess.PIPE, close_fds=True)
        (stdout, stderr) = p.communicate()
        clipboardContents = stdout.decode(ENCODING)
        assert (len(clipboardContents) > 0)
        assert clipboardContents.endswith('\n')
        if clipboardContents.endswith('\n'):
            clipboardContents = clipboardContents[:(- 1)]
        return clipboardContents
    return (copy_klipper, paste_klipper)

def init_dev_clipboard_clipboard():

    def copy_dev_clipboard(text):
        text = _stringifyText(text)
        if (text == ''):
            warnings.warn('Pyperclip cannot copy a blank string to the clipboard on Cygwin.This is effectively a no-op.')
        if ('\r' in text):
            warnings.warn('Pyperclip cannot handle \\r characters on Cygwin.')
        with open('/dev/clipboard', 'wt') as fo:
            fo.write(text)

    def paste_dev_clipboard() -> str:
        with open('/dev/clipboard') as fo:
            content = fo.read()
        return content
    return (copy_dev_clipboard, paste_dev_clipboard)

def init_no_clipboard():

    class ClipboardUnavailable():

        def __call__(self, *args, **kwargs):
            raise PyperclipException(EXCEPT_MSG)

        def __bool__(self) -> bool:
            return False
    return (ClipboardUnavailable(), ClipboardUnavailable())

class CheckedCall():

    def __init__(self, f):
        super().__setattr__('f', f)

    def __call__(self, *args):
        ret = self.f(*args)
        if ((not ret) and get_errno()):
            raise PyperclipWindowsException(('Error calling ' + self.f.__name__))
        return ret

    def __setattr__(self, key, value):
        setattr(self.f, key, value)

def init_windows_clipboard():
    global HGLOBAL, LPVOID, DWORD, LPCSTR, INT
    global HWND, HINSTANCE, HMENU, BOOL, UINT, HANDLE
    from ctypes.wintypes import BOOL, DWORD, HANDLE, HGLOBAL, HINSTANCE, HMENU, HWND, INT, LPCSTR, LPVOID, UINT
    windll = ctypes.windll
    msvcrt = ctypes.CDLL('msvcrt')
    safeCreateWindowExA = CheckedCall(windll.user32.CreateWindowExA)
    safeCreateWindowExA.argtypes = [DWORD, LPCSTR, LPCSTR, DWORD, INT, INT, INT, INT, HWND, HMENU, HINSTANCE, LPVOID]
    safeCreateWindowExA.restype = HWND
    safeDestroyWindow = CheckedCall(windll.user32.DestroyWindow)
    safeDestroyWindow.argtypes = [HWND]
    safeDestroyWindow.restype = BOOL
    OpenClipboard = windll.user32.OpenClipboard
    OpenClipboard.argtypes = [HWND]
    OpenClipboard.restype = BOOL
    safeCloseClipboard = CheckedCall(windll.user32.CloseClipboard)
    safeCloseClipboard.argtypes = []
    safeCloseClipboard.restype = BOOL
    safeEmptyClipboard = CheckedCall(windll.user32.EmptyClipboard)
    safeEmptyClipboard.argtypes = []
    safeEmptyClipboard.restype = BOOL
    safeGetClipboardData = CheckedCall(windll.user32.GetClipboardData)
    safeGetClipboardData.argtypes = [UINT]
    safeGetClipboardData.restype = HANDLE
    safeSetClipboardData = CheckedCall(windll.user32.SetClipboardData)
    safeSetClipboardData.argtypes = [UINT, HANDLE]
    safeSetClipboardData.restype = HANDLE
    safeGlobalAlloc = CheckedCall(windll.kernel32.GlobalAlloc)
    safeGlobalAlloc.argtypes = [UINT, c_size_t]
    safeGlobalAlloc.restype = HGLOBAL
    safeGlobalLock = CheckedCall(windll.kernel32.GlobalLock)
    safeGlobalLock.argtypes = [HGLOBAL]
    safeGlobalLock.restype = LPVOID
    safeGlobalUnlock = CheckedCall(windll.kernel32.GlobalUnlock)
    safeGlobalUnlock.argtypes = [HGLOBAL]
    safeGlobalUnlock.restype = BOOL
    wcslen = CheckedCall(msvcrt.wcslen)
    wcslen.argtypes = [c_wchar_p]
    wcslen.restype = UINT
    GMEM_MOVEABLE = 2
    CF_UNICODETEXT = 13

    @contextlib.contextmanager
    def window():
        '\n        Context that provides a valid Windows hwnd.\n        '
        hwnd = safeCreateWindowExA(0, b'STATIC', None, 0, 0, 0, 0, 0, None, None, None, None)
        try:
            (yield hwnd)
        finally:
            safeDestroyWindow(hwnd)

    @contextlib.contextmanager
    def clipboard(hwnd):
        '\n        Context manager that opens the clipboard and prevents\n        other applications from modifying the clipboard content.\n        '
        t = (time.time() + 0.5)
        success = False
        while (time.time() < t):
            success = OpenClipboard(hwnd)
            if success:
                break
            time.sleep(0.01)
        if (not success):
            raise PyperclipWindowsException('Error calling OpenClipboard')
        try:
            (yield)
        finally:
            safeCloseClipboard()

    def copy_windows(text):
        text = _stringifyText(text)
        with window() as hwnd:
            with clipboard(hwnd):
                safeEmptyClipboard()
                if text:
                    count = (wcslen(text) + 1)
                    handle = safeGlobalAlloc(GMEM_MOVEABLE, (count * sizeof(c_wchar)))
                    locked_handle = safeGlobalLock(handle)
                    ctypes.memmove(c_wchar_p(locked_handle), c_wchar_p(text), (count * sizeof(c_wchar)))
                    safeGlobalUnlock(handle)
                    safeSetClipboardData(CF_UNICODETEXT, handle)

    def paste_windows():
        with clipboard(None):
            handle = safeGetClipboardData(CF_UNICODETEXT)
            if (not handle):
                return ''
            return c_wchar_p(handle).value
    return (copy_windows, paste_windows)

def init_wsl_clipboard():

    def copy_wsl(text):
        text = _stringifyText(text)
        p = subprocess.Popen(['clip.exe'], stdin=subprocess.PIPE, close_fds=True)
        p.communicate(input=text.encode(ENCODING))

    def paste_wsl():
        p = subprocess.Popen(['powershell.exe', '-command', 'Get-Clipboard'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        (stdout, stderr) = p.communicate()
        return stdout[:(- 2)].decode(ENCODING)
    return (copy_wsl, paste_wsl)

def determine_clipboard():
    '\n    Determine the OS/platform and set the copy() and paste() functions\n    accordingly.\n    '
    global Foundation, AppKit, qtpy, PyQt4, PyQt5
    if ('cygwin' in platform.system().lower()):
        if os.path.exists('/dev/clipboard'):
            warnings.warn("Pyperclip's support for Cygwin is not perfect,see https://github.com/asweigart/pyperclip/issues/55")
            return init_dev_clipboard_clipboard()
    elif ((os.name == 'nt') or (platform.system() == 'Windows')):
        return init_windows_clipboard()
    if (platform.system() == 'Linux'):
        if distutils.spawn.find_executable('wslconfig.exe'):
            return init_wsl_clipboard()
    if ((os.name == 'mac') or (platform.system() == 'Darwin')):
        try:
            import AppKit
            import Foundation
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()
    if HAS_DISPLAY:
        if _executable_exists('xsel'):
            return init_xsel_clipboard()
        if _executable_exists('xclip'):
            return init_xclip_clipboard()
        if (_executable_exists('klipper') and _executable_exists('qdbus')):
            return init_klipper_clipboard()
        try:
            import qtpy
        except ImportError:
            try:
                import PyQt5
            except ImportError:
                try:
                    import PyQt4
                except ImportError:
                    pass
                else:
                    return init_qt_clipboard()
            else:
                return init_qt_clipboard()
        else:
            return init_qt_clipboard()
    return init_no_clipboard()

def set_clipboard(clipboard):
    '\n    Explicitly sets the clipboard mechanism. The "clipboard mechanism" is how\n    the copy() and paste() functions interact with the operating system to\n    implement the copy/paste feature. The clipboard parameter must be one of:\n        - pbcopy\n        - pbobjc (default on Mac OS X)\n        - qt\n        - xclip\n        - xsel\n        - klipper\n        - windows (default on Windows)\n        - no (this is what is set when no clipboard mechanism can be found)\n    '
    global copy, paste
    clipboard_types = {'pbcopy': init_osx_pbcopy_clipboard, 'pyobjc': init_osx_pyobjc_clipboard, 'qt': init_qt_clipboard, 'xclip': init_xclip_clipboard, 'xsel': init_xsel_clipboard, 'klipper': init_klipper_clipboard, 'windows': init_windows_clipboard, 'no': init_no_clipboard}
    if (clipboard not in clipboard_types):
        allowed_clipboard_types = [repr(_) for _ in clipboard_types.keys()]
        raise ValueError(f"Argument must be one of {', '.join(allowed_clipboard_types)}")
    (copy, paste) = clipboard_types[clipboard]()

def lazy_load_stub_copy(text):
    '\n    A stub function for copy(), which will load the real copy() function when\n    called so that the real copy() function is used for later calls.\n\n    This allows users to import pyperclip without having determine_clipboard()\n    automatically run, which will automatically select a clipboard mechanism.\n    This could be a problem if it selects, say, the memory-heavy PyQt4 module\n    but the user was just going to immediately call set_clipboard() to use a\n    different clipboard mechanism.\n\n    The lazy loading this stub function implements gives the user a chance to\n    call set_clipboard() to pick another clipboard mechanism. Or, if the user\n    simply calls copy() or paste() without calling set_clipboard() first,\n    will fall back on whatever clipboard mechanism that determine_clipboard()\n    automatically chooses.\n    '
    global copy, paste
    (copy, paste) = determine_clipboard()
    return copy(text)

def lazy_load_stub_paste():
    '\n    A stub function for paste(), which will load the real paste() function when\n    called so that the real paste() function is used for later calls.\n\n    This allows users to import pyperclip without having determine_clipboard()\n    automatically run, which will automatically select a clipboard mechanism.\n    This could be a problem if it selects, say, the memory-heavy PyQt4 module\n    but the user was just going to immediately call set_clipboard() to use a\n    different clipboard mechanism.\n\n    The lazy loading this stub function implements gives the user a chance to\n    call set_clipboard() to pick another clipboard mechanism. Or, if the user\n    simply calls copy() or paste() without calling set_clipboard() first,\n    will fall back on whatever clipboard mechanism that determine_clipboard()\n    automatically chooses.\n    '
    global copy, paste
    (copy, paste) = determine_clipboard()
    return paste()

def is_available():
    return ((copy != lazy_load_stub_copy) and (paste != lazy_load_stub_paste))
(copy, paste) = (lazy_load_stub_copy, lazy_load_stub_paste)
__all__ = ['copy', 'paste', 'set_clipboard', 'determine_clipboard']
clipboard_get = paste
clipboard_set = copy
