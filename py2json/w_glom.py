"""
Navigation and serialization of objects using glom
"""
from glom import glom, Spec
import importlib


def mk_serializer_and_deserializer(spec, mk_inv_spec=None):
    if not isinstance(spec, Spec):
        spec = Spec(spec)

    def serialize(o):
        return glom(o, spec)

    if mk_inv_spec is None:
        return serialize
    else:

        def deserialize(o):
            return glom(o, mk_inv_spec(o))

        return serialize, deserialize


specs_for_kind = {
    'jdict_methods': {
        'description': 'For objects that have from_jdict and to_jdict methods',
        'spec': {
            'module': '__class__.__module__',
            'name': '__class__.__name__',
            'attr': 'from_jdict.__name__',
            'kwargs': lambda x: x.to_jdict(),
        },
        'mk_inv_spec': lambda d: (
            lambda x: importlib.import_module(d['module']),
            lambda x: getattr(x, d['name']),
            lambda x: getattr(x, d['attr']),
            lambda x: x.__call__(d['kwargs']),
        ),
    }
}


def mk_serializer_and_deserializer_for_kind(kind='jdict_methods'):
    specs_dict = specs_for_kind[kind]
    return mk_serializer_and_deserializer(specs_dict['spec'], specs_dict['mk_inv_spec'])
