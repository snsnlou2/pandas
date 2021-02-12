
import io
import pytest
import validate_unwanted_patterns

class TestBarePytestRaises():

    @pytest.mark.parametrize('data', ['\n    with pytest.raises(ValueError, match="foo"):\n        pass\n    ', '\n    # with pytest.raises(ValueError, match="foo"):\n    #    pass\n    ', '\n    # with pytest.raises(ValueError):\n    #    pass\n    ', '\n    with pytest.raises(\n        ValueError,\n        match="foo"\n    ):\n        pass\n    '])
    def test_pytest_raises(self, data):
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.bare_pytest_raises(fd))
        assert (result == [])

    @pytest.mark.parametrize('data, expected', [('\n    with pytest.raises(ValueError):\n        pass\n    ', [(1, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")]), ('\n    with pytest.raises(ValueError, match="foo"):\n        with pytest.raises(ValueError):\n            pass\n        pass\n    ', [(2, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")]), ('\n    with pytest.raises(ValueError):\n        with pytest.raises(ValueError, match="foo"):\n            pass\n        pass\n    ', [(1, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")]), ('\n    with pytest.raises(\n        ValueError\n    ):\n        pass\n    ', [(1, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")]), ('\n    with pytest.raises(\n        ValueError,\n        # match = "foo"\n    ):\n        pass\n    ', [(1, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")])])
    def test_pytest_raises_raises(self, data, expected):
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.bare_pytest_raises(fd))
        assert (result == expected)

@pytest.mark.parametrize('data, expected', [('msg = ("bar " "baz")', [(1, 'String unnecessarily split in two by black. Please merge them manually.')]), ('msg = ("foo " "bar " "baz")', [(1, 'String unnecessarily split in two by black. Please merge them manually.'), (1, 'String unnecessarily split in two by black. Please merge them manually.')])])
def test_strings_to_concatenate(data, expected):
    fd = io.StringIO(data.strip())
    result = list(validate_unwanted_patterns.strings_to_concatenate(fd))
    assert (result == expected)

class TestStringsWithWrongPlacedWhitespace():

    @pytest.mark.parametrize('data', ['\n    msg = (\n        "foo\n"\n        " bar"\n    )\n    ', '\n    msg = (\n        "foo"\n        "  bar"\n        "baz"\n    )\n    ', '\n    msg = (\n        f"foo"\n        "  bar"\n    )\n    ', '\n    msg = (\n        "foo"\n        f"  bar"\n    )\n    ', '\n    msg = (\n        "foo"\n        rf"  bar"\n    )\n    '])
    def test_strings_with_wrong_placed_whitespace(self, data):
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd))
        assert (result == [])

    @pytest.mark.parametrize('data, expected', [('\n    msg = (\n        "foo"\n        " bar"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        f"foo"\n        " bar"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        "foo"\n        f" bar"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        f"foo"\n        f" bar"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        "foo"\n        rf" bar"\n        " baz"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.'), (4, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        "foo"\n        " bar"\n        rf" baz"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.'), (4, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        "foo"\n        rf" bar"\n        rf" baz"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.'), (4, 'String has a space at the beginning instead of the end of the previous string.')])])
    def test_strings_with_wrong_placed_whitespace_raises(self, data, expected):
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd))
        assert (result == expected)
