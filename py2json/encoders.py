import base64
import json
from functools import partial

import numpy as np

class OtoJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return float(obj)
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        if isinstance(obj, np.bool_):
            return True if np.bool_(True) == obj else False

        return json.JSONEncoder.default(self, obj)


json_dump_partial_kwargs = {
    'allow_nan': False,
    'indent': None,
    'separators': (',', ':'),
    'sort_keys': True,
    'cls': OtoJsonEncoder,
}
dump = partial(json.dump, **json_dump_partial_kwargs)
dumps = partial(json.dumps, **json_dump_partial_kwargs)