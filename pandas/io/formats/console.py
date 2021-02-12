
'\nInternal module for console introspection\n'
from shutil import get_terminal_size

def get_console_size():
    '\n    Return console size as tuple = (width, height).\n\n    Returns (None,None) in non-interactive session.\n    '
    from pandas import get_option
    display_width = get_option('display.width')
    display_height = get_option('display.max_rows')
    if in_interactive_session():
        if in_ipython_frontend():
            from pandas._config.config import get_default_val
            terminal_width = get_default_val('display.width')
            terminal_height = get_default_val('display.max_rows')
        else:
            (terminal_width, terminal_height) = get_terminal_size()
    else:
        (terminal_width, terminal_height) = (None, None)
    return ((display_width or terminal_width), (display_height or terminal_height))

def in_interactive_session():
    "\n    Check if we're running in an interactive shell.\n\n    Returns\n    -------\n    bool\n        True if running under python/ipython interactive shell.\n    "
    from pandas import get_option

    def check_main():
        try:
            import __main__ as main
        except ModuleNotFoundError:
            return get_option('mode.sim_interactive')
        return ((not hasattr(main, '__file__')) or get_option('mode.sim_interactive'))
    try:
        return (__IPYTHON__ or check_main())
    except NameError:
        return check_main()

def in_ipython_frontend():
    "\n    Check if we're inside an IPython zmq frontend.\n\n    Returns\n    -------\n    bool\n    "
    try:
        ip = get_ipython()
        return ('zmq' in str(type(ip)).lower())
    except NameError:
        pass
    return False
