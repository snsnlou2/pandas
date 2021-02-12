
'\nTests for CategoricalIndex.__repr__ and related methods.\n'
import pandas._config.config as cf
from pandas import CategoricalIndex

class TestCategoricalIndexRepr():

    def test_string_categorical_index_repr(self):
        idx = CategoricalIndex(['a', 'bb', 'ccc'])
        expected = "CategoricalIndex(['a', 'bb', 'ccc'], categories=['a', 'bb', 'ccc'], ordered=False, dtype='category')"
        assert (repr(idx) == expected)
        idx = CategoricalIndex((['a', 'bb', 'ccc'] * 10))
        expected = "CategoricalIndex(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',\n                  'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb',\n                  'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n                 categories=['a', 'bb', 'ccc'], ordered=False, dtype='category')"
        assert (repr(idx) == expected)
        idx = CategoricalIndex((['a', 'bb', 'ccc'] * 100))
        expected = "CategoricalIndex(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',\n                  ...\n                  'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n                 categories=['a', 'bb', 'ccc'], ordered=False, dtype='category', length=300)"
        assert (repr(idx) == expected)
        idx = CategoricalIndex(list('abcdefghijklmmo'))
        expected = "CategoricalIndex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',\n                  'm', 'm', 'o'],\n                 categories=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', ...], ordered=False, dtype='category')"
        assert (repr(idx) == expected)
        idx = CategoricalIndex(['あ', 'いい', 'ううう'])
        expected = "CategoricalIndex(['あ', 'いい', 'ううう'], categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"
        assert (repr(idx) == expected)
        idx = CategoricalIndex((['あ', 'いい', 'ううう'] * 10))
        expected = "CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ',\n                  'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',\n                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],\n                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"
        assert (repr(idx) == expected)
        idx = CategoricalIndex((['あ', 'いい', 'ううう'] * 100))
        expected = "CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ',\n                  ...\n                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],\n                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category', length=300)"
        assert (repr(idx) == expected)
        idx = CategoricalIndex(list('あいうえおかきくけこさしすせそ'))
        expected = "CategoricalIndex(['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し',\n                  'す', 'せ', 'そ'],\n                 categories=['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', ...], ordered=False, dtype='category')"
        assert (repr(idx) == expected)
        with cf.option_context('display.unicode.east_asian_width', True):
            idx = CategoricalIndex(['あ', 'いい', 'ううう'])
            expected = "CategoricalIndex(['あ', 'いい', 'ううう'], categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"
            assert (repr(idx) == expected)
            idx = CategoricalIndex((['あ', 'いい', 'ううう'] * 10))
            expected = "CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',\n                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n                  'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',\n                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],\n                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"
            assert (repr(idx) == expected)
            idx = CategoricalIndex((['あ', 'いい', 'ううう'] * 100))
            expected = "CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',\n                  'ううう', 'あ',\n                  ...\n                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n                  'あ', 'いい', 'ううう'],\n                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category', length=300)"
            assert (repr(idx) == expected)
            idx = CategoricalIndex(list('あいうえおかきくけこさしすせそ'))
            expected = "CategoricalIndex(['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',\n                  'さ', 'し', 'す', 'せ', 'そ'],\n                 categories=['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', ...], ordered=False, dtype='category')"
            assert (repr(idx) == expected)
