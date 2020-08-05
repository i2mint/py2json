##### Utils #####################################################################################
## To be able to do partial with positionals too
def partial_positionals(func, fix_args, **fix_kwargs):
    """Like functools.partial, but with positionals as well"""

    def wrapper(*args, **kwargs):
        arg = iter(args)
        return func(*(fix_args[i] if i in fix_args else next(arg)
                      for i in range(len(args) + len(fix_args))),
                    **{**fix_kwargs, **kwargs})

    return wrapper


# Explicit version of partial_positionals(isinstance, {1: types})
def mk_isinstance_cond(*types):
    def isinstance_of_target_types(obj):
        return isinstance(obj, types)

    return isinstance_of_target_types


def mk_scan_map(condition_map, dflt=None):
    def scan_mapping(x):
        for condition, then in condition_map.items():
            if condition(x):
                return then
        return dflt

    return scan_mapping


import numpy as np

isinstance_mapping = mk_isinstance_cond(np.ndarray)
assert isinstance_mapping([1, 2, 3]) == False
assert isinstance_mapping(np.array([1, 2, 3])) == True

##### Use #####################################################################################

import numpy
import pandas
from py2json.fakit import fakit

# TODO: much to factor out into a mini-language here
# TODO: See how the specs complexify if we want to use orient='records' kw in DataFrame (de)serialization
type_cond_map = {
    numpy.ndarray: lambda x: {'$fak': ('numpy.array', (list(x),))},
    pandas.DataFrame: lambda x: {'$fak': {
        'f': 'pandas.DataFrame.from_dict',
        'k': {'data': pandas.DataFrame.to_dict(x, orient='index'),
              'orient': 'index'}}
    }
}

type_conds = {mk_isinstance_cond(types): func for types, func in type_cond_map.items()}

serializer_for_type = mk_scan_map(type_conds, dflt=lambda x: x)
serialize = lambda obj: serializer_for_type(obj)(obj)
deserialize = fakit

arr = numpy.array([1, 2, 3])
serialized_arr = serialize(arr)
assert serialized_arr == {'$fak': ('numpy.array', ([1, 2, 3],))}
assert all(deserialize(serialized_arr) == arr)

df = pandas.DataFrame({'foo': [1, 2, 3], 'bar': [10, 20, 30]})
serialized_df = serialize(df)
assert serialized_df == {'$fak': {
    'f': 'pandas.DataFrame.from_dict',
    'k': {'data': {0: {'foo': 1, 'bar': 10},
                   1: {'foo': 2, 'bar': 20},
                   2: {'foo': 3, 'bar': 30}},
          'orient': 'index'}}}
assert all(deserialize(serialized_df) == df)
