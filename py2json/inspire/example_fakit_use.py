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


def serialized_attr_dict(obj, serializer, attrs=None):
    attrs = attrs or dir(obj)
    return {a: serializer(getattr(obj, a)) for a in attrs}


class Struct:
    def __init__(self, **kwargs):
        for a, val in kwargs.items():
            setattr(self, a, val)


def deserialize_as_obj(attr_dict, deserializer, cls=Struct):
    obj = cls()
    for k, v in attr_dict.items():
        setattr(obj, k, deserializer(v))
    return obj


##### Use #####################################################################################
import numpy
import pandas
from py2json.fakit import fakit_if_marked_for_it

# TODO: much to factor out into a mini-language here
# TODO: See how the specs complexify if we want to use orient='records' kw in DataFrame (de)serialization
type_cond_map = {
    numpy.ndarray: lambda x: {'$fak': ('numpy.array', (numpy.ndarray.tolist(x),))},
    pandas.DataFrame: lambda x: {'$fak': {
        'f': 'pandas.DataFrame.from_dict',
        'k': {'data': pandas.DataFrame.to_dict(x, orient='index'),
              'orient': 'index'}}
    }
}

type_conds = {mk_isinstance_cond(types): func for types, func in type_cond_map.items()}

serializer_for_type = mk_scan_map(type_conds, dflt=lambda x: x)

fak_serialize = lambda obj: serializer_for_type(obj)(obj)
fak_deserialize = fakit_if_marked_for_it

arr = numpy.array([1, 2, 3])
serialized_arr = fak_serialize(arr)
assert serialized_arr == {'$fak': ('numpy.array', ([1, 2, 3],))}
assert all(fak_deserialize(serialized_arr) == arr)

df = pandas.DataFrame({'foo': [1, 2, 3], 'bar': [10, 20, 30]})
serialized_df = fak_serialize(df)
assert serialized_df == {'$fak': {
    'f': 'pandas.DataFrame.from_dict',
    'k': {'data': {0: {'foo': 1, 'bar': 10},
                   1: {'foo': 2, 'bar': 20},
                   2: {'foo': 3, 'bar': 30}},
          'orient': 'index'}}}
assert all(fak_deserialize(serialized_df) == df)


##### Now use with sklearn model! #########################################################


class GenericEstimator(Struct):
    fit = None  # or sklearn will complain


from functools import partial

my_serializer = partial(serialized_attr_dict, serializer=fak_serialize)
my_deserializer = partial(deserialize_as_obj, deserializer=fak_deserialize, cls=GenericEstimator)

from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

X, y = make_blobs()

pca = PCA().fit(X)

jdict = my_serializer(pca, attrs=('components_', 'mean_', 'whiten'))
import json

json.loads(json.dumps(jdict))  # to make sure jdict is json friendly

# Does the deserialized version "work" (for the transform method)?

obj = my_deserializer(jdict)
assert (PCA.transform(obj, X) == PCA.transform(pca, X)).all()
# and in case you forgot the equivalence:
assert (pca.transform(X) == PCA.transform(pca, X)).all()
