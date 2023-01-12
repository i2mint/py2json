"""
An attempt at getting a recursive attribute tree
"""
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

from py2json.util import is_types_spec
from py2json.fakit import is_valid_fak
from i2.deco import postprocess, preprocess


@postprocess(dict)
def mk_cond_map_from_types_map(types_map):
    for types, serializer in types_map.items():
        if is_types_spec(types):
            types = mk_isinstance_cond(types)
        assert callable(
            types
        ), f'types spec should be a callable at this point: {types}'
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
    pandas.DataFrame: lambda x: {
        '$fak': {
            'f': 'pandas.DataFrame.from_dict',
            'k': {
                'data': pandas.DataFrame.to_dict(x, orient='index'),
                'orient': 'index',
            },
        }
    },
}

fak_serialize, fak_deserialize = mk_serializer_and_deserializer_for_types_map(
    type_cond_map
)

arr = numpy.array([1, 2, 3])
serialized_arr = fak_serialize(arr)
assert serialized_arr == {'$fak': ('numpy.array', ([1, 2, 3],))}
assert all(fak_deserialize(serialized_arr) == arr)

df = pandas.DataFrame({'foo': [1, 2, 3], 'bar': [10, 20, 30]})
serialized_df = fak_serialize(df)
assert serialized_df == {
    '$fak': {
        'f': 'pandas.DataFrame.from_dict',
        'k': {
            'data': {
                0: {'foo': 1, 'bar': 10},
                1: {'foo': 2, 'bar': 20},
                2: {'foo': 3, 'bar': 30},
            },
            'orient': 'index',
        },
    }
}
assert all(fak_deserialize(serialized_df) == df)


##### Now use with sklearn model! #########################################################


class GenericEstimator(Struct):
    fit = None  # or sklearn will complain


from functools import partial, wraps
from i2.footprints import attrs_used_by_method
import json


# def jsonize_output(func):
#     @wraps(func)
#     def wrapper(x):
#         try:
#             json.dumps()


def mk_serializer_and_deserializer_from_instance_and_methods(
    instance, methods, deserialize_to_cls=None, jsonize=True
):
    cls = instance.__class__
    cls_has_attr = lambda a: hasattr(cls, a)
    instance_has_attr = lambda a: hasattr(instance, a)

    @postprocess(set)
    def get_needed_attrs():
        for method in filter(cls_has_attr, methods):
            target_func = getattr(cls, method)
            for attr in filter(instance_has_attr, attrs_used_by_method(target_func)):
                yield attr

    deserialize_to_cls = deserialize_to_cls or cls
    serializer = partial(
        serialized_attr_dict, serializer=fak_serialize, attrs=get_needed_attrs(),
    )
    deserializer = partial(
        deserialize_as_obj, deserializer=fak_deserialize, cls=deserialize_to_cls,
    )

    if jsonize:
        serializer = postprocess(json.dumps, verbose_error_message=2)(serializer)
        deserializer = preprocess(json.loads)(deserializer)

    return serializer, deserializer


#
#
# my_serializer = partial(serialized_attr_dict, serializer=fak_serialize)
# my_deserializer = partial(deserialize_as_obj, deserializer=fak_deserialize, cls=GenericEstimator)

# make something to serialize
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

X, y = make_blobs()
model = PCA().fit(X)

(serialize, deserialize,) = mk_serializer_and_deserializer_from_instance_and_methods(
    model, methods=['predict', 'transform'], deserialize_to_cls=None
)

serialized_model = serialize(model)
deserialized_model = deserialize(serialized_model)

assert isinstance(serialized_model, str)
if hasattr(model, 'transform'):
    target_func = model.__class__.transform
else:
    target_func = model.__class__.predict

assert numpy.allclose(target_func(deserialized_model, X), target_func(model, X))


# and in case you forgot the equivalence:
# assert (model.transform(X) == target_func(model, X)).all()


def mk_serialize_deserialize_pipeline_for_model(
    model, methods=('predict', 'transform'), deserialize_to_cls=None
):
    (
        serialize,
        deserialize,
    ) = mk_serializer_and_deserializer_from_instance_and_methods(
        model, methods=methods, deserialize_to_cls=deserialize_to_cls
    )

    def alt_model(model):
        return deserialize(serialize(model))

    return alt_model


from py2json.inspire.serializing_sklearn_estimators import estimator_test_df

mk_alt_model = lambda model: mk_serialize_deserialize_pipeline_for_model(model)(model)

kwargs_for_behavioral_test_kwargs_for_estimator = dict(
    mk_alt_model=mk_alt_model,  # Returns an alt object by serializing and deserializing the fitted model
    methods=('predict', 'transform',),  # The methods to use to make behavior_funcs,
)


def test_estimators_serialization(estimator_classes=None, ignore_warnings=True):
    return estimator_test_df(
        estimator_classes,
        ignore_warnings=ignore_warnings,
        **kwargs_for_behavioral_test_kwargs_for_estimator,
    )


def get_estimator_classes_that_are_behaviorally_equivalent_wrt_fakit():
    df = test_estimators_serialization()
    ok_to_test_classes = tuple(df[df['kind'] == 'ok']['cls'].values)
    return ok_to_test_classes


if __name__ == '__main__':
    from collections import Counter
    import pandas as pd
    from py2json.inspire.serializing_sklearn_estimators import all_estimator_classes

    n = len(all_estimator_classes)
    print(f'Testing all {n} estimators...')
    df = test_estimators_serialization()

    c = Counter(df.kind)
    print(pd.Series(c).to_string())

    print(
        f"So {c['ok']}/{n} estimators can be automatically created, "
        f'fitted, json-serialized, and deserialized to an object that is '
        f'behaviorally equivalent to the original '
        f'(as far as predict and transform methods are concerned).'
    )
    print('\nThese are those methods...\n')
    print(*[x.__name__ for x in df.cls], sep='\t')
