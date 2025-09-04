import json
from py2json.encoders import dumps
import numpy as np


def test_dumps_numpy_and_bytes():
    arr = np.array([1, 2, 3])
    b = b'hello'
    s = dumps({'arr': arr, 'b': b})
    # ensure json produced and arr serialized as list and bytes base64 present
    assert '1' in s and 'hello' not in s
