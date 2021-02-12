
'\npandas._config is considered explicitly upstream of everything else in pandas,\nshould have no intra-pandas dependencies.\n\nimporting `dates` and `display` ensures that keys needed by _libs\nare initialized.\n'
__all__ = ['config', 'detect_console_encoding', 'get_option', 'set_option', 'reset_option', 'describe_option', 'option_context', 'options']
from pandas._config import config
from pandas._config import dates
from pandas._config.config import describe_option, get_option, option_context, options, reset_option, set_option
from pandas._config.display import detect_console_encoding
