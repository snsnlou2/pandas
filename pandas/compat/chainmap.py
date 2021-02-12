
from typing import ChainMap, MutableMapping, TypeVar, cast
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class DeepChainMap(ChainMap[(_KT, _VT)]):
    '\n    Variant of ChainMap that allows direct updates to inner scopes.\n\n    Only works when all passed mapping are mutable.\n    '

    def __setitem__(self, key, value):
        for mapping in self.maps:
            mutable_mapping = cast(MutableMapping[(_KT, _VT)], mapping)
            if (key in mutable_mapping):
                mutable_mapping[key] = value
                return
        cast(MutableMapping[(_KT, _VT)], self.maps[0])[key] = value

    def __delitem__(self, key):
        "\n        Raises\n        ------\n        KeyError\n            If `key` doesn't exist.\n        "
        for mapping in self.maps:
            mutable_mapping = cast(MutableMapping[(_KT, _VT)], mapping)
            if (key in mapping):
                del mutable_mapping[key]
                return
        raise KeyError(key)
