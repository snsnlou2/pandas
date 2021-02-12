
from io import BytesIO
import os
import pytest
import pandas.util._test_decorators as td
from pandas import read_csv
import pandas._testing as tm

def test_streaming_s3_objects():
    pytest.importorskip('botocore', minversion='1.10.47')
    from botocore.response import StreamingBody
    data = [b'foo,bar,baz\n1,2,3\n4,5,6\n', b'just,the,header\n']
    for el in data:
        body = StreamingBody(BytesIO(el), content_length=len(el))
        read_csv(body)

@tm.network
@td.skip_if_no('s3fs')
def test_read_without_creds_from_pub_bucket():
    result = read_csv('s3://gdelt-open-data/events/1981.csv', nrows=3)
    assert (len(result) == 3)

@tm.network
@td.skip_if_no('s3fs')
def test_read_with_creds_from_pub_bucket():
    with tm.ensure_safe_environment_variables():
        os.environ.setdefault('AWS_ACCESS_KEY_ID', 'foobar_key')
        os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'foobar_secret')
        df = read_csv('s3://gdelt-open-data/events/1981.csv', nrows=5, sep='\t', header=None)
        assert (len(df) == 5)
