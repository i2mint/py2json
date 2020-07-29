from i2.routing_forest import *
from i2.footprints import *
from i2.signatures import Sig


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
        return {k: val_for_argname[k] for k in val_for_argname.keys() & set(Sig(func).without_defaults)}

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
