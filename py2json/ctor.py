"""Tools for serialization and deserialization by Andie

Ctor short for constructor.

The two main methods to be aware of are Ctor.deconstruct() and Ctor.construct(). Ideally, that's all
you need to use to break down any object into an objest ready for json.dumps() and reconstruct back
to an equivalent object.

Include to_jdict() and from_jdict() for an easy way to make a custom class instance work with this.
Otherwise, it will attempt to pickle any unhandled types.

An example with numpy arrays:

>>> a = np.array([range(i * 10, (i + 1) * 10) for i in range(10)])
>>> a_decon_ctor_dict = Ctor.deconstruct(a, validate_conversion=True, output_type=Ctor.CTOR_DICT)
>>> a_decon_jdict = Ctor.deconstruct(a, validate_conversion=True, output_type=Ctor.JSON_DICT)
>>> a_recon_from_ctor_dict = Ctor.construct(a_decon_ctor_dict)
>>> a_recon_from_jdict = Ctor.construct(a_decon_jdict)
>>> assert np.allclose(a, a_recon_from_ctor_dict, a_recon_from_jdict)

Another example not runnable, since dependent on some third party code

```
from ... .source.audio import PyAudioSourceReader

c = PyAudioSourceReader
c_decon_ctor_dict = Ctor.deconstruct(c, validate_conversion=True, output_type=Ctor.CTOR_DICT)
c_decon_jdict = Ctor.deconstruct(c, validate_conversion=True, output_type=Ctor.JSON_DICT)
c_recon_from_ctor_dict = Ctor.construct(c_decon_ctor_dict)
c_recon_from_jdict = Ctor.construct(c_decon_jdict)
assert c == c_recon_from_jdict == c_recon_from_ctor_dict

The Design

`Ctor.deconstruct()` will traverse through objects supported by remap (list, tuple, dict, set)
to find objects foreign to json and deconstruct them using `Ctor.deconstruction_specs`.
Specified object are deconstructed into a CtorDict containing CONSTRUCTOR, ARGS, and KWARGS.

Converting a function to a CtorDict results in the CONSTRUCTOR as the Ctor deserializing method
and the ARGS as the import path of the deconstructed function.

>>> from pprint import pprint
>>> from py2json.tests.test_ctor import add
>>> serialized = Ctor.deconstruct(add, validate_conversion=True, output_type=Ctor.CTOR_DICT)
>>> pprint(serialized, sort_dicts=False)
{'CONSTRUCTOR': <bound method Ctor._deserialize_ctor_from_jdict of <class 'py2json.ctor.Ctor'>>,
 'ARGS': [{'module': 'py2json.tests.test_ctor', 'name': 'add', 'attr': None}],
 'KWARGS': None}

By default, the output type is a json dict. The CONSTRUCTOR method is reduced to an import path.

>>> serialized = Ctor.deconstruct(add, validate_conversion=True, output_type=Ctor.JSON_DICT)
>>> pprint(serialized, sort_dicts=False)
{'CONSTRUCTOR': {'module': 'py2json.ctor',
                 'name': 'Ctor',
                 'attr': '_deserialize_ctor_from_jdict'},
 'ARGS': [{'module': 'py2json.tests.test_ctor', 'name': 'add', 'attr': None}],
 'KWARGS': None}

Ideally, a custom class will include `to_jdict` and `from_jdict` methods

>>> from py2json.tests.test_ctor import MockToJdict
>>> instance = MockToJdict('test_value')
>>> serialized = Ctor.deconstruct(instance, validate_conversion=True)
>>> pprint(serialized, sort_dicts=False)
{'CONSTRUCTOR': {'module': 'py2json.tests.test_ctor',
                 'name': 'MockToJdict',
                 'attr': 'from_jdict'},
 'ARGS': [{'value': 'test_value'}],
 'KWARGS': None}

As a fallback, dill is used when no matching deconstruction_specs is found.
`CtorException` is thrown if it fails.

>>> from py2json.tests.test_ctor import MockClass
>>> instance = MockClass('test_value')
>>> serialized = Ctor.deconstruct(instance, validate_conversion=True)
>>> pprint(serialized, sort_dicts=False) # doctest: +SKIP
{'CONSTRUCTOR': {'module': 'ctor', 'name': 'dill_load_string', 'attr': None},
 'ARGS': ['\x80\x04\x95E\x00\x00\x00\x00\x00\x00\x00\x8c\x17py2json.tests.test_ctor\x94\x8c\t'
          'MockClass\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c\n'
          'test_value\x94sb.'],
 'KWARGS': None}


Ctor.deconstruct will also check if the deconstructed args need to be deconstructed as well.
In this example, the object deconstructs with ARGS that also contain to_jdict/from_jdict

>>> from py2json.tests.test_ctor import MockToJdict
>>> instance = MockToJdict([MockToJdict('test_value')])
>>> serialized = Ctor.deconstruct(instance, validate_conversion=True)
>>> pprint(serialized, sort_dicts=False)
{'CONSTRUCTOR': {'module': 'py2json.tests.test_ctor',
                 'name': 'MockToJdict',
                 'attr': 'from_jdict'},
 'ARGS': [{'value': [{'CONSTRUCTOR': {'module': 'py2json.tests.test_ctor',
                                      'name': 'MockToJdict',
                                      'attr': 'from_jdict'},
                      'ARGS': [{'value': 'test_value'}],
                      'KWARGS': None}]}],
 'KWARGS': None}

In this example, the partial function deconstructs with ARGS that also contain a function

>>> from functools import partial
>>> partial_func = partial(add, 1, b=2)
>>> serialized = Ctor.deconstruct(partial_func, validate_conversion=True, output_type=Ctor.CTOR_DICT)
>>> pprint(serialized, sort_dicts=False)
{'CONSTRUCTOR': <class 'functools.partial'>,
 'ARGS': [{'CONSTRUCTOR': {'module': 'py2json.ctor',
                           'name': 'Ctor',
                           'attr': '_deserialize_ctor_from_jdict'},
           'ARGS': [{'module': 'py2json.tests.test_ctor',
                     'name': 'add',
                     'attr': None}],
           'KWARGS': None},
          1],
 'KWARGS': {'b': 2}}


Function calls can also be serialized

>>> import os
>>> ctor_jdict = Ctor.to_ctor_dict(constructor=os.path.join, args=['I', 'am', 'a', 'filepath'])
>>> pprint(ctor_jdict, sort_dicts=False)
{'CONSTRUCTOR': {'module': 'posixpath', 'name': 'join', 'attr': None},
 'ARGS': ['I', 'am', 'a', 'filepath'],
 'KWARGS': {}}
>>> Ctor.construct(ctor_jdict)
'I/am/a/filepath'

You can pass args
>>> from collections import namedtuple
>>> ctor_jdict = Ctor.to_ctor_dict(namedtuple, args=('A', 'x y z'))
>>> pprint(ctor_jdict, sort_dicts=False)
{'CONSTRUCTOR': {'module': 'collections', 'name': 'namedtuple', 'attr': None},
 'ARGS': ('A', 'x y z'),
 'KWARGS': {}}
>>> A = Ctor.construct(ctor_jdict)
>>> A('no', 'defaults', 'here')
A(x='no', y='defaults', z='here')

...or kwargs
>>> ctor_jdict = Ctor.to_ctor_dict(namedtuple, kwargs={'typename': 'A', 'field_names': 'x y z'})
>>> pprint(ctor_jdict, sort_dicts=False)
{'CONSTRUCTOR': {'module': 'collections', 'name': 'namedtuple', 'attr': None},
 'ARGS': [],
 'KWARGS': {'typename': 'A', 'field_names': 'x y z'}}
>>> A = Ctor.construct(ctor_jdict)
>>> A('no', 'defaults', 'here')
A(x='no', y='defaults', z='here')

...or both args and kwargs
>>> ctor_jdict = Ctor.to_ctor_dict(namedtuple, ('A', 'x y z'), {'defaults': ('has', 'defaults')})
>>> pprint(ctor_jdict, sort_dicts=False)
{'CONSTRUCTOR': {'module': 'collections', 'name': 'namedtuple', 'attr': None},
 'ARGS': ('A', 'x y z'),
 'KWARGS': {'defaults': ('has', 'defaults')}}
>>> A = Ctor.construct(ctor_jdict)
>>> A('this one')
A(x='this one', y='has', z='defaults')

"""
import dataclasses
import functools
import importlib
import inspect
import json
from typing import Callable, TypedDict, Union, Any, List, Dict

import dill
import numpy as np
from boltons.iterutils import remap, default_enter
from glom import Literal, glom, Spec
from i2 import mk_sentinel

NOT_SET = mk_sentinel('NOT_SET')


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


class classproperty(object):
    """
    Similar to @property decorator except for classes instead of class instances
    >>> class Test:
    ...     @classproperty
    ...     def a(cls):
    ...         return 1
    ...
    ...     @property
    ...     def b(self):
    ...         return 2
    >>> Test.a + 1
    2
    >>> Test.b + 1  # Throws TypeError because Test.b is a "property" object until the class is instantiated.
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for +: 'property' and 'int'
    """

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def if_condition_return_action(obj, condition_action_list):
    for condition, action in condition_action_list:
        if condition(obj):
            return action(obj)


def is_ctor_jdict(obj):
    return isinstance(obj, dict) and set(list(obj)) == {
        'module',
        'name',
        'attr',
    }


ctor_to_jdict_spec = {
    'description': 'Serialize and deserialize Ctor.CONSTRUCTOR to and from jdict',
    'spec': {
        'module': functools.partial(
            if_condition_return_action,
            condition_action_list=[
                (is_ctor_jdict, lambda obj: obj['module']),
                (inspect.ismethod, lambda obj: obj.__self__.__module__),
                (inspect.isclass, lambda obj: obj.__module__),
                (callable, lambda obj: obj.__module__),
            ],
        ),
        'name': functools.partial(
            if_condition_return_action,
            condition_action_list=[
                (is_ctor_jdict, lambda obj: obj['name']),
                (inspect.ismethod, lambda obj: obj.__self__.__name__),
                (inspect.isclass, lambda obj: obj.__name__),
                (callable, lambda obj: obj.__name__),
            ],
        ),
        'attr': functools.partial(
            if_condition_return_action,
            condition_action_list=[
                (is_ctor_jdict, lambda obj: obj['attr']),
                (inspect.ismethod, lambda obj: obj.__name__),
                (inspect.isclass, lambda obj: None),
                (callable, lambda obj: None),
            ],
        ),
    },
    'mk_inv_spec': lambda d: (
        lambda x: importlib.import_module(d['module']),
        lambda x: getattr(x, d['name']),
        lambda x: getattr(x, d['attr']) if d['attr'] else x,
    ),
}

(
    _serialize_ctor_to_jdict,
    _deserialize_ctor_from_jdict,
) = mk_serializer_and_deserializer(
    ctor_to_jdict_spec['spec'], ctor_to_jdict_spec['mk_inv_spec']
)


def dill_dump_string(x):
    return dill.dumps(x).decode('latin-1')


def dill_load_string(x):
    return dill.loads(x.encode('latin-1'))


class CtorException(ValueError):
    pass


class CtorNames:
    # ctor_dict keys
    CONSTRUCTOR = 'CONSTRUCTOR'  # callable
    ARGS = 'ARGS'  # list
    KWARGS = 'KWARGS'  # dict

    # output types
    CTOR_DICT = 'ctor_dict'
    JSON_DICT = 'jdict'
    CONSTRUCTED = 'constructed'


class ConstructorJdict(TypedDict):
    module: str
    name: str
    attr: str


CtorDict = TypedDict(
    'CtorDict',
    {
        CtorNames.CONSTRUCTOR: Union[Callable, ConstructorJdict],
        CtorNames.ARGS: List[Any],
        CtorNames.KWARGS: Dict[str, Any],
    },
)


class Ctor(CtorNames):
    """
    A Base class for serializing any object
    """

    @classmethod
    def _serialize_ctor_to_jdict(cls, ctor_obj):
        return _serialize_ctor_to_jdict(ctor_obj)

    @classmethod
    def _deserialize_ctor_from_jdict(cls, ctor_jdict):
        return _deserialize_ctor_from_jdict(ctor_jdict)

    @classproperty
    def fallback_deconstruction_spec(cls):
        """Try to pickle when no deconstruction spec is found"""
        return {
            'description': 'default to dill pickle',
            'check_type': lambda x: True,
            'spec': {
                Ctor.CONSTRUCTOR: Literal(dill_load_string),
                Ctor.ARGS: lambda x: [dill_dump_string(x)],
                Ctor.KWARGS: Literal(None),
            },
            'validate_conversion': lambda x, serialized_x: (
                x == Ctor._construct_obj(serialized_x)
            ),
        }

    @classproperty
    def deconstruction_specs(cls):
        """
        Return a list of deconstruction_specs=dict(description='doc string',
                                                   check_type='callable for checking if obj satisfies spec',
                                                   spec='describes how to generate ctor_dict',
                                                   validate_conversion='optional callable comparing obj to ctor_dict')
        """
        return [
            {
                'description': 'For functools.partial',
                'check_type': lambda x: isinstance(x, functools.partial),
                'spec': {
                    Ctor.CONSTRUCTOR: Literal(functools.partial),
                    Ctor.ARGS: lambda x: [x.func, *x.args],
                    Ctor.KWARGS: lambda x: dict(x.keywords),
                },
                'validate_conversion': lambda x, serialized_x: (
                    (deserialized_x := Ctor._construct_obj(serialized_x))
                    and all(
                        getattr(x, a, NOT_SET) == getattr(deserialized_x, a, NOT_SET)
                        for a in ('func', 'args', 'keywords')
                    )
                ),
            },
            {
                'description': 'For numpy.ndarray',
                'check_type': lambda x: isinstance(x, np.ndarray),
                'spec': {
                    Ctor.CONSTRUCTOR: Literal(np.array),
                    Ctor.ARGS: lambda x: [x.tolist()],
                    Ctor.KWARGS: Literal(None),
                },
                'validate_conversion': lambda x, serialized_x: (
                    np.allclose(x, Ctor._construct_obj(serialized_x))
                ),
            },
            {
                'description': 'for class methods or class constructors',
                'check_type': lambda x: inspect.ismethod(x) or inspect.isclass(x),
                'spec': {
                    Ctor.CONSTRUCTOR: Literal(cls._deserialize_ctor_from_jdict),
                    Ctor.ARGS: lambda x: [cls._serialize_ctor_to_jdict(x)],
                    Ctor.KWARGS: Literal(None),
                },
            },
            {
                'description': 'for functions',
                'check_type': inspect.isfunction,
                'spec': {
                    Ctor.CONSTRUCTOR: Literal(cls._deserialize_ctor_from_jdict),
                    Ctor.ARGS: lambda x: [cls._serialize_ctor_to_jdict(x)],
                    Ctor.KWARGS: Literal(None),
                },
            },
            {
                'description': 'For instances that have from_jdict and to_jdict methods',
                'check_type': lambda x: not isinstance(x, type)
                and hasattr(x, 'from_jdict')
                and hasattr(x, 'to_jdict'),
                'spec': {
                    Ctor.CONSTRUCTOR: lambda x: type(x).from_jdict,
                    Ctor.ARGS: lambda x: [x.to_jdict()],
                    Ctor.KWARGS: Literal(None),
                },
                'validate_conversion': lambda x, serialized_x: (
                    x.to_jdict() == Ctor._construct_obj(serialized_x).to_jdict()
                ),
            },
            {
                'description': 'For instances that have from_ddp_jdict and to_dpp_jdict methods',
                'check_type': lambda x: not isinstance(x, type)
                and hasattr(x, 'from_dpp_jdict')
                and hasattr(x, 'to_dpp_jdict'),
                'spec': {
                    Ctor.CONSTRUCTOR: lambda x: type(x).from_dpp_jdict,
                    Ctor.ARGS: lambda x: [x.to_dpp_jdict()],
                    Ctor.KWARGS: Literal(None),
                },
                'validate_conversion': lambda x, serialized_x: (
                    x.to_dpp_jdict() == Ctor._construct_obj(serialized_x).to_dpp_jdict()
                ),
            },
            {
                'description': 'For dataclasses.dataclass instance',
                'check_type': lambda x: (
                    dataclasses.is_dataclass(x) and not isinstance(x, type)
                ),
                'spec': {
                    Ctor.CONSTRUCTOR: lambda x: type(x),
                    Ctor.ARGS: Literal(None),
                    Ctor.KWARGS: lambda x: {
                        field.name: getattr(x, field.name)
                        for field in dataclasses.fields(x)
                    },
                },
                'validate_conversion': lambda x, serialized_x: (
                    (deserialized_x := Ctor._construct_obj(serialized_x))
                    and (x_dict := dataclasses.asdict(x))
                    and (deserx_dict := dataclasses.asdict(deserialized_x))
                    and all(  # only validate json compatible types
                        x_dict[k] == deserx_dict[k]
                        for k in x_dict
                        if cls.is_jdict(x_dict[k])
                    )
                ),
            },
        ]

    @classmethod
    def construct(cls, obj):
        """Construct ctor objs. Return obj with no action if obj is str, int, float, bool, or None.

        >>> from pprint import pprint
        >>> from py2json.tests.test_ctor import add
        >>> serialized = Ctor.deconstruct(add, validate_conversion=True, output_type=Ctor.CTOR_DICT)
        >>> pprint(serialized, sort_dicts=False)
        {'CONSTRUCTOR': <bound method Ctor._deserialize_ctor_from_jdict of <class 'py2json.ctor.Ctor'>>,
         'ARGS': [{'module': 'py2json.tests.test_ctor', 'name': 'add', 'attr': None}],
         'KWARGS': None}
        >>> assert Ctor.construct(serialized) == add

        :param obj: ctor_dict or jdict
        :return: remap obj with constructed ctor_dicts
        """
        if cls.is_supported_by_default_remap(obj) and cls.is_jdict(obj):
            obj = cls.deserializer(obj)

        if cls.is_ctor_dict(obj):
            return cls._construct_obj(obj)

        if cls.is_supported_by_default_remap(obj):

            def visit(path, key, value):
                if cls.is_ctor_dict(value):
                    value = cls._construct_obj(value)
                return key, value

            def enter(path, key, value):
                if cls.is_ctor_dict(value):
                    return None, False
                else:
                    return default_enter(path, key, value)

            return remap(obj, visit, enter=enter)
        elif cls.is_python_to_json_basic_type(obj):
            return obj
        else:
            CtorException(f'Cannot construct unknown obj type: {type(obj)}')

    @classmethod
    def deconstruct(
        cls,
        obj,
        validate_conversion: bool = False,
        output_type: str = CtorNames.JSON_DICT,
    ):
        """
        Remap a nested structure by deconstructing nodes of non-basic types into ctor_dicts using the deconstruction_specs

        >>> from pprint import pprint
        >>> from py2json.tests.test_ctor import add
        >>> serialized = Ctor.deconstruct(add, validate_conversion=True, output_type=Ctor.CTOR_DICT)
        >>> pprint(serialized, sort_dicts=False)
        {'CONSTRUCTOR': <bound method Ctor._deserialize_ctor_from_jdict of <class 'py2json.ctor.Ctor'>>,
         'ARGS': [{'module': 'py2json.tests.test_ctor', 'name': 'add', 'attr': None}],
         'KWARGS': None}

        :param obj: any object containing only basic types and those described in the deconstruction_specs
        :param validate_conversion: [boolean] True to compare obj to reconstructed ctor_dict. False to skip validation
        :param output_type: Ctor.JSON_DICT or Ctor.CTOR_DICT
        :return: remapped obj
        """
        decon_ouputs = (Ctor.JSON_DICT, Ctor.CTOR_DICT)
        if output_type not in decon_ouputs:
            raise CtorException(
                f'deconstruct output type must be one of: {decon_ouputs}'
            )

        if not cls.is_supported_by_default_remap(obj):
            if cls.is_python_to_json_basic_type(obj):
                return obj

            ctor_dict = cls._deconstruct_obj(obj)
            if output_type == Ctor.CTOR_DICT:
                return ctor_dict
            elif output_type == Ctor.JSON_DICT:
                return cls.serializer(ctor_dict)

        def visit(path, key, value):
            if not cls.is_python_to_json_basic_type(value) and key != Ctor.CONSTRUCTOR:
                try:
                    value = cls._deconstruct_obj(value, validate_conversion)
                except CtorException:
                    raise CtorException(
                        f'deconstruction spec not found for non-basic type at path={path}, key={key}'
                    )
            return key, value

        def enter(path, key, value):
            if not cls.is_python_to_json_basic_type(value) and key != Ctor.CONSTRUCTOR:
                return None, False
            else:
                return default_enter(path, key, value)

        ctor_dict = remap(obj, visit, enter=enter)
        if output_type == Ctor.CTOR_DICT:
            return ctor_dict
        elif output_type == Ctor.JSON_DICT:
            return cls.serializer(ctor_dict)

    @classmethod
    def serializer(cls, ctor_pydict):
        """Search ctor_pydict and replace any Ctor.CONSTRUCTOR nodes with a JSON equivalent

        :param ctor_pydict: A ctor_dict or Dict containing nested ctor_dicts
        :return: jsonizeable ctor_jdict
        """
        return remap(
            ctor_pydict,
            cls._remap_visit_serialize_to_jdict,
            cls._remap_enter_constructor,
        )

    @classmethod
    def deserializer(cls, ctor_jdict):
        """Search ctor_jdict and replace any Ctor.CONSTRUCTOR nodes with a module object callable

        :param ctor_jdict: A jsonized ctor_dict or Dict containing nested jsonized ctor_dicts
        :return:
        """
        return remap(
            ctor_jdict,
            cls._remap_visit_deserialize_to_pydict,
            cls._remap_enter_constructor,
        )

    @classmethod
    def to_jdict(cls, ctor_dict: CtorDict):
        return cls.serializer(ctor_dict)

    @classmethod
    def from_jdict(cls, jdict):
        ctor_dict = cls.deserializer(jdict)
        return cls._construct_obj(ctor_dict)

    @classmethod
    def to_ctor_dict(
        cls,
        constructor: Callable,
        args: Union[list, tuple] = NOT_SET,
        kwargs: dict = NOT_SET,
        *,
        output_type=CtorNames.JSON_DICT,
    ) -> CtorDict:
        """Function calls can also be serialized

        >>> from pprint import pprint
        >>> from collections import namedtuple
        >>> ctor_jdict = Ctor.to_ctor_dict(namedtuple, ('A', 'x y z'), {'defaults': ('has', 'defaults')})
        >>> pprint(ctor_jdict, sort_dicts=False)
        {'CONSTRUCTOR': {'module': 'collections', 'name': 'namedtuple', 'attr': None},
         'ARGS': ('A', 'x y z'),
         'KWARGS': {'defaults': ('has', 'defaults')}}
        >>> A = Ctor.construct(ctor_jdict)
        >>> A('this one')
        A(x='this one', y='has', z='defaults')

        :param constructor: callable
        :param args: args passed to callable
        :param kwargs: kwargs passed to callable
        :param output_type: JSON_DICT or CTOR_DICT
        :return: CtorDict ready to be constructed with Ctor.construct()
        """
        if args is NOT_SET:
            args = []
        if kwargs is NOT_SET:
            kwargs = {}
        ctor_dict = {
            cls.CONSTRUCTOR: constructor,
            cls.ARGS: args,
            cls.KWARGS: kwargs,
        }
        if output_type == CtorNames.JSON_DICT:
            return cls.to_jdict(ctor_dict)
        return ctor_dict

    # Boolean ..........................................................................................................
    @classmethod
    def is_python_to_json_basic_type(cls, obj):
        """
        Check if JSONEncoder supports the object type by default.

        :param obj: any
        :return: boolean
        """
        return isinstance(obj, (dict, list, tuple, str, int, float, bool, type(None)))

    @classmethod
    def is_ctor_dict(cls, ctor_dict):
        return (
            isinstance(ctor_dict, dict)
            and Ctor.CONSTRUCTOR in ctor_dict
            and callable(ctor_dict[cls.CONSTRUCTOR])
        )

    @classmethod
    def is_supported_by_default_remap(cls, obj):
        return isinstance(obj, (list, tuple, dict, set))

    @classmethod
    def is_jdict(cls, obj):
        try:
            json.dumps(obj)
            is_jdict = True
        except TypeError:
            is_jdict = False
        return is_jdict

    # Private ..........................................................................................................
    @classmethod
    def _construct_obj(cls, ctor_dict: CtorDict):
        """
        Calls the Ctor.CONSTRUCTOR with given Ctor.ARGS and Ctor.KWARGS.
        Checks if ARGS or KWARGS are ctor_dicts and constructs those first

        :param ctor_dict: {Ctor.CONSTRUCTOR: Callable, Ctor.ARGS: List[Any], Ctor.KWARGS: Dict[str, Any]}
        :return: ctor_dict[Ctor.CONSTRUCTOR](*ctor_dict[Ctor.ARGS], **ctor_dict[Ctor.KWARGS])
        """
        if cls.is_ctor_dict(ctor_dict):
            if args := cls._get_value(ctor_dict, cls.ARGS, []):
                args = cls.construct(args)
            if kwargs := cls._get_value(ctor_dict, cls.KWARGS, {}):
                kwargs = cls.construct(kwargs)

            return ctor_dict[cls.CONSTRUCTOR](*args, **kwargs)
        else:
            raise CtorException(
                f"ctor_dict must have a '{cls.CONSTRUCTOR}' key and optionally '{cls.ARGS}' and '{cls.KWARGS}' but got '{ctor_dict}'"
            )

    @classmethod
    def _deconstruct_obj(cls, obj, validate_conversion: bool = False) -> CtorDict:
        """Breakdown an obj into a ctor_dict as described by deconstruction_specs.
        Further breakdowns on deconstructed ARGS and KWARGS if necessary.

        Will use fallback_deconstruction_spec to dill pickle object if no spec is found.
        Validation is always performed in this fallback case.

        :param obj: any object satisfying one of the deconstruction_specs
        :param validate_conversion: [boolean] True to compare obj to reconstructed ctor_dict. False to skip validation
        :return: ctor_dict: {Ctor.CONSTRUCTOR: Callable, Ctor.ARGS: List[Any], Ctor.KWARGS: Dict[str, Any]}
        """
        try:
            s = next(s for s in cls.deconstruction_specs if s['check_type'](obj))
            try:
                serializer = s['serializer']
            except KeyError:
                s['serializer'] = mk_serializer_and_deserializer(s['spec'])
                serializer = s['serializer']

            ctor_dict = serializer(obj)

            if ctor_dict[Ctor.ARGS] is not None:
                ctor_dict[Ctor.ARGS] = cls.deconstruct(
                    ctor_dict[Ctor.ARGS], validate_conversion=validate_conversion
                )
            if ctor_dict[Ctor.KWARGS] is not None:
                ctor_dict[Ctor.KWARGS] = cls.deconstruct(
                    ctor_dict[Ctor.KWARGS], validate_conversion=validate_conversion
                )

            if validate_conversion is True:
                cls._validate_conversion(obj, ctor_dict, s)

            return ctor_dict
        except (StopIteration, KeyError):
            try:
                s = cls.fallback_deconstruction_spec
                serializer = mk_serializer_and_deserializer(s['spec'])
                ctor_dict = serializer(obj)
                cls._validate_conversion(obj, ctor_dict, s)
                return ctor_dict

            except Exception as e:
                raise CtorException(
                    f'deconstruction spec not found for non-basic type: {obj}'
                ) from e

    @classmethod
    def _validate_conversion(cls, obj, ctor_dict, deconstruction_spec):
        try:
            validator = deconstruction_spec['validate_conversion']
        except KeyError:
            deconstruction_spec[
                'validate_conversion'
            ] = cls._default_conversion_validation
            validator = deconstruction_spec['validate_conversion']
        if validator(obj, ctor_dict) is False:
            raise CtorException(f'deconstruction validation test failed for: {obj}')

    @classmethod
    def _get_value(cls, dict_obj, key, default_value):
        if key in dict_obj and dict_obj[key] is not None:
            return dict_obj[key]
        else:
            return default_value

    @classmethod
    def _remap_enter_constructor(cls, p, k, v):
        """Determines which nodes to visit. The default_enter points to leaf nodes.
        This additionally looks for Ctor.CONSTRUCTOR nodes."""
        if k == Ctor.CONSTRUCTOR:
            return p, False
        return default_enter(p, k, v)

    @classmethod
    def _remap_visit_serialize_to_jdict(cls, p, k, v):
        """Serialize when visiting Ctor.CONSTRUCTOR node, otherwise keep the same value"""
        if k == Ctor.CONSTRUCTOR:
            return k, Ctor._serialize_ctor_to_jdict(v)
        return k, v

    @classmethod
    def _remap_visit_deserialize_to_pydict(cls, p, k, v):
        """Deserialize when visiting Ctor.CONSTRUCTOR node, otherwise keep the same value"""
        if k == Ctor.CONSTRUCTOR:
            return k, Ctor._deserialize_ctor_from_jdict(v)
        return k, v

    @classmethod
    def _default_conversion_validation(cls, original, ctor_dict):
        return original == Ctor._construct_obj(ctor_dict)
