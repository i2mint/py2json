import json
from dataclasses import dataclass, astuple
from functools import partial
from pprint import pprint
from typing import Any

import numpy as np
import pytest

from py2json import Ctor


def add(a, b):
    return a + b


class TestClass:
    def __init__(self, value):
        self.value = value

    def __add__(self, other: 'TestClass'):
        return TestToJdict(self.value + other.value)

    def __repr__(self):
        value = self.value
        return f'<{self.__class__.__name__} {value=}>'

    def __eq__(self, other: 'TestClass'):
        return self.value == other.value


class TestToJdict(TestClass):
    def __eq__(self, other: 'TestToJdict'):
        return self.to_jdict() == other.to_jdict()

    def to_jdict(self):
        return {'value': self.value}

    @classmethod
    def from_jdict(cls, jdict):
        return TestToJdict(**jdict)


@dataclass
class TestDataclass:
    a: int = 1
    b: Any = 2


def basic_is_equal(a, b):
    return a == b


def partial_add_is_equal(a, b):
    return a() == b()


def dataclass_with_partial_add_is_equal(a, b):
    return all(
        partial_add_is_equal(x, y) if isinstance(x, partial) else basic_is_equal(x, y)
        for (x, y) in zip(astuple(a), astuple(b))
    )


@pytest.mark.parametrize(
    'test_name,original,deserializer_name,is_equal',
    [
        (
            'functools.partial',
            partial(add, 3, b=4),
            '"_deserialize_ctor_from_jdict"',
            partial_add_is_equal,
        ),
        ('class type', TestClass, '"_deserialize_ctor_from_jdict"', basic_is_equal),
        (
            'function',
            add,
            '"_deserialize_ctor_from_jdict"',
            lambda a, b: basic_is_equal(a, b) and a(1, 2) == b(1, 2),
        ),
        (
            'class method',
            TestToJdict.from_jdict,
            '"_deserialize_ctor_from_jdict"',
            basic_is_equal,
        ),
        (
            'to_jdict and from_jdict',
            TestToJdict([TestToJdict(1)]),
            '"from_jdict"',
            basic_is_equal,
        ),
        (
            'numpy array',
            np.arange(10),
            '{"module": "numpy", "name": "array", "attr": null}',
            np.allclose,
        ),
        ('basic dataclass', TestDataclass(3, 4), '"TestDataclass"', basic_is_equal),
        (
            'dataclass with partial',
            TestDataclass(5, partial(add, 3, b=4)),
            '"TestDataclass"',
            dataclass_with_partial_add_is_equal,
        ),
        (
            'fallback dill',
            TestClass([TestClass(1)]),
            '"dill_load_string"',
            basic_is_equal,
        ),
    ],
)
def test_ctor(test_name, original, deserializer_name, is_equal):
    print(f'\n\n--- Test: {test_name} ---')
    print(f'original    : {original}')
    serialized = Ctor.deconstruct(original, validate_conversion=True)

    assert f'{deserializer_name}' in json.dumps(
        serialized
    ), f'not serialized using: {deserializer_name}'
    deserialized = Ctor.construct(serialized)
    print(f'deserialized: {deserialized}')
    assert is_equal(original, deserialized)
    print('serialized:')
    pprint(serialized)


def test_ctor_dict():
    """Test demonstrating a dict containing many types deconstructed and reconstructed"""
    original = dict(
        function=add,
        cls=TestClass,
        classmethod=TestToJdict.from_jdict,
        to_jdict=TestToJdict([TestToJdict(1)]),
        class_instance=TestClass([TestClass(1)]),
        integer=123,
        string='abc',
        boolean=True,
        none=None,
        list=[1, 2, 3],
        dict=dict(a=1, b=2, c=3),
        numpy_array=np.arange(10),
        partial_func=partial(add, 1, b=2),
        basic_dataclass=TestDataclass(3, 4),
        dataclass_with_partial=TestDataclass(5, partial(add, 3, b=4)),
    )
    print('\n\n------original------')
    pprint(original)
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    print('\n\n-----serialized-----')
    pprint(serialized)
    deserialized = Ctor.construct(serialized)
    print('\n\n----deserialized----')
    pprint(deserialized)
