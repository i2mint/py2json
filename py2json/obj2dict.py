"""
Utils to encode python object information as a dict
"""

from typing import Optional, Callable, Iterable
from inspect import Parameter
from i2 import Sig, name_of_obj


def func_to_parameters_dict(
    func: Callable,
    *,
    kind: Optional[Callable] = None,  # tip: Use `str`` to get the name of the kind
    default: Optional[Callable] = None,
    annotation: Optional[Callable] = None,
) -> dict:
    """Returns a jsonizable dict of a signature's parameters.

    By default it will only return names, but you can specify a (egress) function
    for any attribute you want to have in your parameter dict (dict giving information
    about each parameter in the signature).
    For example, if you specify ``kind=str``, the ``kind`` attribute of each parameter
    will be included, as a string.

    Similarly, you can specify that you want the default, as is, by specifying
    ``default=lambda x: x``. Note that for ``default`` and ``annotation`` you'll only
    get the field if it existed (i.e. the value was not ``Parameter.empty``).
    The reason for specifying ``kind``, ``default``, and ``annotation`` as functions
    is that you can specify how you want to represent the value of each of these,
    which is needed, for example, to make those values compliant to a targetted system.
    For example, you can specify these functions so that the resulting dict is
    jsonizable.


    >>> def func(a, /, b, c=3, *, d: int = 4):
    ...     pass
    >>> func_to_parameters_dict(func)
    [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}, {'name': 'd'}]
    >>> func_to_parameters_dict(func, kind=str)  # doctest: +NORMALIZE_WHITESPACE
    [{'name': 'a', 'kind': 'POSITIONAL_ONLY'},
    {'name': 'b', 'kind': 'POSITIONAL_OR_KEYWORD'},
    {'name': 'c', 'kind': 'POSITIONAL_OR_KEYWORD'},
    {'name': 'd', 'kind': 'KEYWORD_ONLY'}]

    >>> func_to_parameters_dict(
    ...    func, kind=str, default=lambda x: x, annotation=str
    ... )  # doctest: +NORMALIZE_WHITESPACE
    [{'name': 'a', 'kind': 'POSITIONAL_ONLY'},
    {'name': 'b', 'kind': 'POSITIONAL_OR_KEYWORD'},
    {'name': 'c', 'kind': 'POSITIONAL_OR_KEYWORD', 'default': 3},
    {'name': 'd',
    'kind': 'KEYWORD_ONLY',
    'default': 4,
    'annotation': "<class 'int'>"}]
    """
    parameters = Sig(func).parameters

    def _parameter_dict(p):
        yield 'name', p.name
        if kind:
            yield 'kind', kind(p.kind)
        if default and p.default is not p.empty:
            yield 'default', p.default
        if annotation and p.annotation is not p.empty:
            yield 'annotation', annotation(p.annotation)

    def gen_parameters():
        for p in parameters.values():
            yield dict(_parameter_dict(p))

    # return gen_parameters()
    return list(gen_parameters())


# TODO: Should we specify parameters control via kind, default, annotation, or via
#   a custome `parameters_to_dict` argument? Pros and cons?
def signature_to_dict(
    func: Callable,
    *,
    kind: Optional[Callable] = None,  # tip: Use `str`` to get the name of the kind
    default: Optional[Callable] = None,
    annotation: Optional[Callable] = None,
    return_annotation: Optional[Callable] = None,
) -> dict:
    """Returns a dict version of the signature.
    Includes "parameters" and "return_annotation" field (if requested and existing).
    """
    parameters_dict = func_to_parameters_dict(
        func, kind=kind, default=default, annotation=annotation,
    )
    d = {'parameters': parameters_dict}
    sig = Sig(func)
    if return_annotation and sig.return_annotation is not Parameter.empty:
        d['return_annotation'] = return_annotation(sig.return_annotation)
    return d


# TODO: The "routing" pattern (declarative control flow) applies here again. Could make
#  this into a general "extractor" that would be defined by, possibly nested,
#  object and object info extractors.
def func_info_dict(
    func,
    *,
    name=name_of_obj,
    signature=lambda obj: signature_to_dict(Sig(obj)),
    **func_info_fields,
):
    r"""Returns a dict of information about a function.
    By default, will include `name` and `signature` fields, is extensible to contain
    any number of fields, simply by specifying extra `(field_name, field_func)` pairs.

    Is meant to be used with `functools.partial` to create custom "info dict"
    extractors.

    >>> def func(a, /, b, c=3, *, d: int = 4) -> int:
    ...     '''Docs of the func'''
    ...     return d + c * b ** a
    >>> func_info_dict(func)  # doctest: +NORMALIZE_WHITESPACE
    {'name': 'func',
    'signature':
        {'parameters': [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}, {'name': 'd'}]}}

    >>> from py2json.obj2dict import signature_to_dict
    >>> from functools import partial
    >>> _signature_to_dict = partial(signature_to_dict, kind=str, return_annotation=str)
    >>>
    >>> info_dict = func_info_dict(
    ...     func, signature=_signature_to_dict, doc=lambda x: x.__doc__
    ... ) == {
    ...     'name': 'func',
    ...     'signature': {
    ...         'parameters': [
    ...             {'name': 'a', 'kind': 'POSITIONAL_ONLY'},
    ...             {'name': 'b', 'kind': 'POSITIONAL_OR_KEYWORD'},
    ...             {'name': 'c', 'kind': 'POSITIONAL_OR_KEYWORD'},
    ...             {'name': 'd', 'kind': 'KEYWORD_ONLY'}],
    ...         'return_annotation': "<class 'int'>"
    ...     },
    ...     'doc': 'Docs of the func'
    ... }
    """
    func_info_fields = dict(name=name, signature=signature, **func_info_fields)
    info_dict = dict()
    for field_name, field_func in func_info_fields.items():
        if field_func is not None:
            info_dict[field_name] = field_func(func)
    return info_dict


# --------------------------------------------------------------------------------------
# Old stuff

import re


def kind_of_type(obj_type):
    return obj_type.__module__ + '.' + obj_type.__name__


def kind_of_obj(obj):
    return kind_of_type(type(obj))


def no_dunder_filt(attr):
    return not attr.startswith('__')


def items_with_transformed_keys(d, key_trans=lambda x: x, key_cond=lambda: True):
    """A generator of (transformed_key, value) items.

    :param d: dict (or mapping) to operate on
    :param key_trans: Function that is applied to the key
    :param key_cond: Function specifying whether to change the key or not
    :return: A generator of (transformed_key, value)

    >>> d = {'a': 1, 2: 20}
    >>> dict(items_with_transformed_keys(d, lambda x: x * 100, lambda x: isinstance(x, int)))
    {'a': 1, 200: 20}

    """
    for k, v in d.items():
        if key_cond(k):
            k = key_trans(k)
        yield k, v


class ApplyDictOf(object):
    pass


apply_dict_of = ApplyDictOf()


class Obj2Dict(object):
    """

    >>> import numpy as np
    >>>
    >>> dc = Obj2Dict(
    ...     to_data_for_kind={
    ...         'numpy.ndarray': lambda obj: obj.tolist()
    ...     },
    ...     from_data_for_kind={
    ...         'numpy.ndarray': lambda data: np.array(data)
    ...     },
    ... )
    >>>
    >>> original_obj = np.array([2,4])
    >>> kind, data = dc.kind_and_data_of_obj(original_obj)
    >>> assert kind == 'numpy.ndarray'
    >>> assert type(data) == list
    >>> assert data == [2, 4]
    >>>
    >>> recovered_obj = dc.obj_of_kind_and_data(kind, data)
    >>> assert type(original_obj) == type(recovered_obj)
    >>> assert all(original_obj == recovered_obj)
    >>>
    >>> # But couldn't make it work (yet) with:
    >>> from collections import Counter
    >>>
    >>> class A(object):
    ...     z = Counter({'n': 10, 'k': 5})
    ...     def __init__(self, x=(1,2,3), y=np.array([2,3,4]), z=None):
    ...         self.x = x
    ...         self._y = y
    ...         if z is not None:
    ...             self.z = z
    ...     def __repr__(self):
    ...         return f"A(x={self.x}, y={self._y}, z={self.z})"
    ...
    >>> dc = Obj2Dict(
    ...     to_data_for_kind={
    ...         'numpy.ndarray': lambda obj: obj.tolist(),
    ...         Counter: dict,
    ...     },
    ...     from_data_for_kind={
    ...         '__main__.A': A,
    ...         'numpy.ndarray': lambda data: np.array(data),
    ...         Counter: Counter
    ...     },
    ... )

    """

    def __init__(self, to_data_for_kind=None, from_data_for_kind=None):
        if to_data_for_kind is None:
            to_data_for_kind = {}
        if from_data_for_kind is None:
            from_data_for_kind = {}

        is_type = lambda x: isinstance(x, type)

        self.to_data_for_kind = dict(
            items_with_transformed_keys(
                to_data_for_kind, key_trans=kind_of_type, key_cond=is_type
            )
        )
        self.from_data_for_kind = dict(
            items_with_transformed_keys(
                from_data_for_kind, key_trans=kind_of_type, key_cond=is_type
            )
        )

    def kind_and_data_of_obj(self, obj):
        kind = kind_of_obj(obj)
        if kind in self.to_data_for_kind:
            return kind, self.to_data_for_kind[kind](obj)
        else:
            return kind, obj

    def obj_of_kind_and_data(self, kind, data):
        if kind.startswith('__builtin__'):
            return data
        if (
            isinstance(data, dict)
            and 'data' in data
            and 'kind' in data
            and len(data) == 2
        ):
            data = self.obj_of_kind_and_data(kind=data['kind'], data=data['data'])

        if kind in self.from_data_for_kind:
            return self.from_data_for_kind[kind](data)
        else:
            return data

    def obj_of_kind_data_dict(self, kind_data_dict):
        return self.obj_of_kind_and_data(
            kind=kind_data_dict['kind'], data=kind_data_dict['data']
        )

    def dict_of(self, obj, attr_filt=no_dunder_filt):
        if attr_filt is None:
            attr_filt = lambda attr: True
        elif isinstance(attr_filt, (list, tuple, set)):
            attr_inclusion_set = set(attr_filt)
            attr_filt = lambda attr: attr in attr_inclusion_set
        elif isinstance(attr_filt, str):
            if attr_filt == 'underscore_suffixed':
                attr_filt = lambda attr: attr.endswith('_')
            else:
                attr_pattern = re.compile(attr_filt)
                attr_filt = attr_pattern.match
        else:
            assert callable(
                attr_filt
            ), "Don't know what to do with that kind of attr_filt: {}".format(attr_filt)

        d = dict()
        for k in filter(attr_filt, dir(obj)):
            attr_obj = getattr(obj, k)
            kind, data = self.kind_and_data_of_obj(attr_obj)
            if data is not apply_dict_of:
                d[k] = {'kind': kind, 'data': data}
            else:
                d[k] = {'kind': kind, 'data': self.dict_of(data, attr_filt)}

        return {'kind': kind_of_obj(obj), 'data': d}

        # def obj_of(self, obj_dict):
