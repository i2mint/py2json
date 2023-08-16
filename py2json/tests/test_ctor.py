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


class MockClass:
    def __init__(self, value):
        self.value = value

    def __add__(self, other: 'MockClass'):
        return type(self)(self.value + other.value)

    def __repr__(self):
        value = self.value
        return f'<{self.__class__.__name__} {value=}>'

    def __eq__(self, other: 'MockClass'):
        return self.value == other.value


class MockToJdict(MockClass):
    def __eq__(self, other: 'MockToJdict'):
        return self.to_jdict() == other.to_jdict()

    def to_jdict(self):
        return {'value': self.value}

    @classmethod
    def from_jdict(cls, jdict):
        return cls(**jdict)


class MockToDppJdict(MockClass):
    def __eq__(self, other: 'MockToDppJdict'):
        return self.to_dpp_jdict() == other.to_dpp_jdict()

    def to_dpp_jdict(self):
        return {'value': self.value}

    @classmethod
    def from_dpp_jdict(cls, jdict):
        return cls(**jdict)


@Ctor.mk_class_serializable(obj_to_kwargs=lambda obj: {'value': obj.value})
class MockWithDecorator(MockClass):
    pass


@Ctor.mk_class_serializable(
    obj_to_constructor=lambda x: type(x).from_jdict,
    obj_to_args=lambda x: [x.to_jdict()],
)
class MockToJdictWithDecorator(MockToJdict):
    pass


@dataclass
class MockDataclass:
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
        ('class type', MockClass, '"_deserialize_ctor_from_jdict"', basic_is_equal),
        (
            'function',
            add,
            '"_deserialize_ctor_from_jdict"',
            lambda a, b: basic_is_equal(a, b) and a(1, 2) == b(1, 2),
        ),
        (
            'class method',
            MockToJdict.from_jdict,
            '"_deserialize_ctor_from_jdict"',
            basic_is_equal,
        ),
        (
            'to_jdict and from_jdict',
            MockToJdict([MockToJdict(1)]),
            '"from_jdict"',
            basic_is_equal,
        ),
        (
            'numpy array',
            np.arange(10),
            '{"module": "numpy", "name": "array", "attr": null}',
            np.allclose,
        ),
        ('basic dataclass', MockDataclass(3, 4), '"MockDataclass"', basic_is_equal),
        (
            'dataclass with partial',
            MockDataclass(5, partial(add, 3, b=4)),
            '"MockDataclass"',
            dataclass_with_partial_add_is_equal,
        ),
        (
            'fallback dill',
            MockClass([MockClass(1)]),
            '"dill_load_string"',
            basic_is_equal,
        ),
        (
            'make serializable decorator',
            MockWithDecorator(1),
            '"MockWithDecorator"',
            basic_is_equal,
        ),
        (
            'make serializable decorator',
            MockToJdictWithDecorator(1),
            '"from_jdict"',
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
        # function=add,
        # cls=MockClass,
        # cls_with_to_jdict=MockToJdict,
        # classmethod=MockToJdict.from_jdict,
        # to_jdict=MockToJdict([MockToJdict(1)]),
        # to_dpp_jdict=MockToDppJdict([MockToDppJdict(1)]),
        # to_dpp_jdict_nested_with_to_jdict=MockToDppJdict([MockToJdict(1)]),
        # class_instance=MockClass([MockClass(1)]),
        # integer=123,
        # string='abc',
        # boolean=True,
        # none=None,
        # list=[1, 2, 3],
        # dict=dict(a=1, b=2, c=3),
        # numpy_array=np.arange(10),
        # partial_func=partial(add, 1, b=2),
        # basic_dataclass=MockDataclass(3, 4),
        # dataclass_with_partial=MockDataclass(5, partial(add, 3, b=4)),
        cls_with_decorator=MockWithDecorator(1),
        to_jdict_with_decorator=MockToJdictWithDecorator(1),
    )
    print('\n\n------original------')
    pprint(original)
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    print('\n\n-----serialized-----')
    pprint(serialized)
    deserialized = Ctor.construct(serialized)
    print('\n\n----deserialized----')
    pprint(deserialized)


if __name__ == '__main__':
    test_ctor_dict()
