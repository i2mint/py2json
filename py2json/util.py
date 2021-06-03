"""
py2json utils functions and other helpers
"""
from i2.routing_forest import *
from i2.footprints import *
from i2.signatures import Sig
from dataclasses import dataclass


@dataclass
class Literal:
    obj: object


def missing_args_func(func_to_kwargs=None, ignore=None):
    """Returns a function that returns a set of missing arg names.
    Missing means that that the arg is required for a func (has no default)
    and hasn't been found by the func_to_kwargs policy.

    The returned function can be used to diagnose the coverage of the func_to_kwargs policy,
    or as a filter to find those functions that are not covered by the policy.

    :param func_to_kwargs: Callable that returns valid kwargs for a given func.
    :param ignore: If not None, should be an iterable to names not to check
    :return: A missing_args_func that returns a set of arg names that are missing.

    >>> from collections import namedtuple
    >>> assert missing_args_func()(namedtuple) == {'field_names', 'typename'}
    >>> func_to_kwargs = lambda f: {namedtuple: {'typename': 'Unspecified'}}.get(f, {})
    >>>
    >>> missing_args = missing_args_func(func_to_kwargs)
    >>> missing_args(namedtuple)
    {'field_names'}
    >>> def foo(x=1, y=2): ...  # defaults cover all arguments
    >>> assert list(filter(missing_args, (namedtuple, foo))) == [namedtuple]
    """

    def missing_args_func_(func):
        missing_args = set(Sig(func).without_defaults.parameters) - set(ignore or ())
        if func_to_kwargs is not None:
            missing_args -= func_to_kwargs(func).keys()
        return missing_args

    return missing_args_func_


def mk_func_to_kwargs_from_a_val_for_argname_map(val_for_argname=None):
    """Returns a function func_to_kwargs that returns kwargs for a given callable func.
    The intent being that these kwargs can be used as valid inputs of func as such:
    ```
        func(**func_to_kwargs)
    ```

    Does so by taking the intersection of those arguments of the func that don't have defaults
    and the input val_for_argname mapping.

    Note that if no val_for_argname is given, or non matches the default-less arguments of func,
    then {} is returned.

    >>> val_for_argname = {'typename': 'Unspecified', 'x': 0}
    >>> func_to_kwargs = mk_func_to_kwargs_from_a_val_for_argname_map(val_for_argname)
    >>> missing_args = missing_args_func(func_to_kwargs)
    >>>
    >>> from collections import namedtuple
    >>> def foo(typename, x, y=2): ...
    >>> def bar(x, z=None): ...
    >>> assert missing_args(namedtuple) == {'field_names'}
    >>> assert missing_args(foo) == set()
    >>> assert missing_args(bar) == set()
    >>> assert list(filter(missing_args, (namedtuple, foo, bar))) == [namedtuple]
    """
    val_for_argname = val_for_argname or {}

    def func_to_kwargs(func):
        return {
            k: val_for_argname[k]
            for k in val_for_argname.keys() & set(Sig(func).without_defaults)
        }

    return func_to_kwargs


def is_valid_kwargs(func, kwargs):
    """Test if kwargs constitute a valid input for func simply by trying func(**kwargs) out.

    :param func: A callable
    :param kwargs: A dict of keyword arguments
    :return: True if, and only if `func(**kwargs)` doesn't fail, and False if it does raise an Exception.

    >>> def f(a, b=1):
    ...     return a * b
    >>> is_valid_kwargs(f, {'a': 10})
    True
    >>> is_valid_kwargs(f, {'a': 1, 'b': 10})
    True
    >>> is_valid_kwargs(f, {'b': 2, 'c': 4})  # c is not a valid argument name, so...
    False
    >>> is_valid_kwargs(f, {})  # a has no default value, so you need at least that argument, so...
    False
    """
    try:
        func(**kwargs)
        return True
    except Exception as e:
        return False


from inspect import signature
from functools import wraps, reduce
from i2.signatures import Sig
import warnings


def is_not_none(x):
    return x is not None


def ignore_warnings(func):
    """Couldn't make this work"""
    from i2.signatures import Sig

    raise RuntimeError("Couldn't make this work. Don't use!")

    @Sig.from_objs(func, lambda ignore_warnings=True: ...)
    def _func(*args, ignore_warnings=True, **kwargs):
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter('ignore')
            return func(*args, **kwargs)

    return _func


def catch_errors(errors=(Exception,), on_error=lambda e: print(e)):
    if not callable(on_error):
        on_error_val = on_error
        on_error = lambda e: on_error_val
    else:
        nargs = len(signature(on_error).parameters)
        if nargs > 1:
            raise ValueError(
                f'on_error should be a value or callable with 0 or 1 arguments'
            )
        elif nargs == 0:
            on_error_func = on_error
            on_error = lambda e: on_error_func()

    def wrap_func(func):
        @wraps(func)
        def func_with_errors_caught(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except errors as e:
                return on_error(e)

        return func_with_errors_caught

    return wrap_func


class Nones:
    """
    >>> x, y, z = Nones(3)
    >>> x, y, z
    (None, None, None)
    >>> bool(Nones(3))
    False
    """

    def __init__(self, n_items: int):
        self.n_items = n_items

    def __iter__(self):
        return (None for _ in range(self.n_items))

    def __bool__(self):
        return False


def partial_positionals(func, fix_args, **fix_kwargs):
    """Like functools.partial, but with positionals as well"""

    def wrapper(*args, **kwargs):
        arg = iter(args)
        return func(
            *(
                fix_args[i] if i in fix_args else next(arg)
                for i in range(len(args) + len(fix_args))
            ),
            **{**fix_kwargs, **kwargs},
        )

    return wrapper


def is_types_spec(types) -> bool:
    """Returns True iff input types is a type or an iterable of types"""
    if isinstance(types, type):
        types = (types,)
    try:
        return len(types) > 0 and all(isinstance(x, type) for x in types)
    except Exception:
        return False


def mk_isinstance_cond(types) -> Callable[[Any], bool]:
    """Makes a boolean function that verifies if objects are of a target type (or types)"""

    assert is_types_spec(types), f'types need to be a single or an iterable of types'

    def isinstance_of_target_types(obj):
        return isinstance(obj, types)

    return isinstance_of_target_types


def mk_scan_mapper(condition_map, dflt=None):
    """Make function implementing an if/elif/.../else logic from a {bool_func: x, ...} map"""

    def scan_mapping(x):
        for condition, then in condition_map.items():
            if condition(x):
                return then
        return dflt

    return scan_mapping


def types_to_cond(types_map):
    """Convert a {type(s): x, ...} map into a {is_of_that_type: x, ...} map"""
    return {mk_isinstance_cond(types): x for types, x in types_map.items()}


def types_map_to_scan_mapper(types_map, dflt=None):
    """Make function implementing an if/elif/.../else logic from a {type(s): x, ...} map.
    The returned mapper will be such that `mapper(obj)` will return a value x
    according to the first `isinstance(obj, types)` check that is found in {types: x, ...}
    types_map.

    >>> mapper = types_map_to_scan_mapper({dict: 'a dict!', (list, tuple): 'list-like'},
    ...     dflt='nothing found')
    >>> mapper({'a': 'dict'})
    'a dict!'
    >>> mapper((1, 2, 3))
    'list-like'
    >>> mapper(['a', 'list'])
    'list-like'
    >>> mapper(lambda x: x)  # a function: No match for that!
    'nothing found'
    >>> mapper(mapper)
    'nothing found'
    """
    return mk_scan_mapper(types_to_cond(types_map), dflt)


def compose(*functions):
    """Make a function that is the composition of the input functions"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
