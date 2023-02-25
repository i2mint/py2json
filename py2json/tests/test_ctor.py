import json
from functools import partial
from pprint import pprint

import numpy as np

from py2json import Ctor


def add(a, b):
    print('add called', f'{a} + {b} = ', end='')
    print(f'{a + b}')

    add.count += 1
    return a + b


add.count = 0


def counter(*_a, **_kw):
    counter.count += 1
    return counter.count - 1


counter.count = 0


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


def test_ctor_partial():
    original = partial(add, 3, b=4)
    print(f'\noriginal    : {original}')
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    assert '_deserialize_ctor_from_jdict' in json.dumps(
        serialized
    ), f'not serialized using _deserialize_ctor_from_jdict'
    deserialized = Ctor.construct(serialized)
    print(f'deserialized: {deserialized}')
    assert original() == deserialized()
    print('serialized:')
    pprint(serialized)


def test_ctor_class():
    original = TestClass
    print(f'\noriginal    : {original}')
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    assert '_deserialize_ctor_from_jdict' in json.dumps(
        serialized
    ), f'not serialized using dill'
    deserialized = Ctor.construct(serialized)
    print(f'deserialized: {deserialized}')
    assert original == deserialized
    print('serialized:')
    pprint(serialized)


def test_ctor_function():
    original = add
    print(f'\noriginal    : {original}')
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    assert '_deserialize_ctor_from_jdict' in json.dumps(
        serialized
    ), f'not serialized using _deserialize_ctor_from_jdict'
    deserialized = Ctor.construct(serialized)
    print(f'deserialized: {deserialized}')
    assert original == deserialized
    assert original(1, 2) == deserialized(1, 2)
    print('serialized:')
    pprint(serialized)


def test_ctor_class_method():
    original = TestToJdict.from_jdict
    print(f'\noriginal    : {original}')
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    assert '_deserialize_ctor_from_jdict' in json.dumps(
        serialized
    ), f'not serialized using _deserialize_ctor_from_jdict'
    deserialized = Ctor.construct(serialized)
    print(f'deserialized: {deserialized}')
    assert original == deserialized
    print('serialized:')
    pprint(serialized)


def test_ctor_to_and_from_jdict():
    assert hasattr(TestToJdict, 'to_jdict') and hasattr(TestToJdict, 'from_jdict')
    original = TestToJdict([TestToJdict(1)])
    print(f'\noriginal    : {original}')
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    assert 'from_jdict' in json.dumps(serialized), 'not serialized using to_jdict'
    deserialized = Ctor.construct(serialized)
    print(f'deserialized: {deserialized}')
    assert original == deserialized
    print('serialized:')
    pprint(serialized)


def test_ctor_numpy_array():
    original = np.arange(10)
    print(f'\noriginal    : {original}')
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    assert '{"module": "numpy", "name": "array", "attr": null}' in json.dumps(
        serialized
    ), 'not serialized using to_jdict'
    deserialized = Ctor.construct(serialized)
    print(f'deserialized: {deserialized}')
    assert np.allclose(original, deserialized)
    print('serialized:')
    pprint(serialized)


def test_ctor_fallback_dill_pickle_class_instance():
    original = TestClass([TestClass(1)])
    print(f'\noriginal    : {original}')
    serialized = Ctor.deconstruct(original, validate_conversion=True)

    assert 'dill_load_string' in json.dumps(serialized), f'not serialized using dill'
    deserialized = Ctor.construct(serialized)
    print(f'deserialized: {deserialized}')
    assert original == deserialized
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
    )
    print('\n\n------original------')
    pprint(original)
    serialized = Ctor.deconstruct(original, validate_conversion=True)
    print('\n\n-----serialized-----')
    pprint(serialized)
    deserialized = Ctor.construct(serialized)
    print('\n\n----deserialized----')
    pprint(deserialized)
