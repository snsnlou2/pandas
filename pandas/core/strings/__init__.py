
"\nImplementation of pandas.Series.str and its interface.\n\n* strings.accessor.StringMethods : Accessor for Series.str\n* strings.base.BaseStringArrayMethods: Mixin ABC for EAs to implement str methods\n\nMost methods on the StringMethods accessor follow the pattern:\n\n    1. extract the array from the series (or index)\n    2. Call that array's implementation of the string method\n    3. Wrap the result (in a Series, index, or DataFrame)\n\nPandas extension arrays implementing string methods should inherit from\npandas.core.strings.base.BaseStringArrayMethods. This is an ABC defining\nthe various string methods. To avoid namespace clashes and pollution,\nthese are prefixed with `_str_`. So ``Series.str.upper()`` calls\n``Series.array._str_upper()``. The interface isn't currently public\nto other string extension arrays.\n"
from .accessor import StringMethods
from .base import BaseStringArrayMethods
__all__ = ['StringMethods', 'BaseStringArrayMethods']
