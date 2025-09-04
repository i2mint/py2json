import json
from py2json.ctor import Ctor
from py2json.fakit import obj_to_dotpath, dotpath_to_obj
from collections import namedtuple


def test_ctor_namedtuple_roundtrip():
    ctor_jdict = Ctor.to_ctor_dict(namedtuple, args=('A', 'x y z'))
    A = Ctor.construct(ctor_jdict)
    inst = A('one', 'two', 'three')
    assert inst.x == 'one' and inst.y == 'two' and inst.z == 'three'


def test_obj_to_dotpath_roundtrip():
    from inspect import Signature

    dot = obj_to_dotpath(Signature.replace)
    assert dotpath_to_obj(dot) is Signature.replace
