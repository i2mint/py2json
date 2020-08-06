##### Utils #####################################################################################
## To be able to do partial with positionals too


# Explicit version of partial_positionals(isinstance, {1: types})
from py2json.util import mk_isinstance_cond, mk_scan_mapper
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
from py2json.fakit import refakit

from py2json.util import is_types_spec, Literal
from py2json.fakit import is_valid_fak
from i2.deco import process_output_with


@process_output_with(dict)
def mk_cond_map_from_types_map(types_map):
    for types, serializer in types_map.items():
        if is_types_spec(types):
            types = mk_isinstance_cond(types)
        assert callable(types), f"types spec should be a callable at this point: {types}"
        # TODO: Would lead to shorter spec language, but needs "arg injection" of sorts
        # if isinstance(serializer, (dict, tuple, list)):
        #     assert is_valid_fak(serializer), f"Should be a valid fak: {serializer}"
        #     fak_spec = serializer
        #
        #     def serializer(x):
        #         return {'$fak': fak_spec}
        yield types, serializer


def asis(x):
    return x


def mk_serializer_and_deserializer_for_types_map(types_map):
    cond_map = mk_cond_map_from_types_map(types_map)
    scan_mapper = mk_scan_mapper(cond_map, dflt=asis)

    def serializer(obj):
        return scan_mapper(obj)(obj)

    return serializer, refakit


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

fak_serialize, fak_deserialize = mk_serializer_and_deserializer_for_types_map(type_cond_map)

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

# make something to serialize
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

X, y = make_blobs()
pca = PCA().fit(X)

# figure out what attributes we need for a target function:
from i2.footprints import attrs_used_by_method

target_func = PCA.transform

needed_attrs = [a for a in attrs_used_by_method(target_func) if hasattr(pca, a)]

jdict = my_serializer(pca, attrs=needed_attrs)

# verify jdict is json friendly
import json

json.loads(json.dumps(jdict))

# Does the deserialized version "work" (for the target function)?
obj = my_deserializer(jdict)
assert (target_func(obj, X) == target_func(pca, X)).all()
# and in case you forgot the equivalence:
assert (pca.transform(X) == target_func(pca, X)).all()
