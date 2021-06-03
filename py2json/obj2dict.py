"""
f
"""
from __future__ import division

import re
from copy import copy


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
