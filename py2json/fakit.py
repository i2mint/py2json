"""
A general language for json-serialization of a function call.
- Any construction of a python object needs to go through a function call that makes it so
this approach is general.
- It’s also simple at its base, but open (and intended for) extensions to specialize
and compress the language as well as add layers for security.

Note: "fakit" can be pronounced with the "a" as in "bake" or a
"""
import os
import importlib
from functools import reduce, partial

FAK = '$fak'


# TODO: Make a config_utils.py module to centralize config tools (configs for access is just one -- serializers another)
# TODO: Integrate (external because not standard lib) other safer tools for secrets, such as:
#  https://github.com/SimpleLegal/pocket_protector


def getenv(name, default=None):
    """Like os.getenv, but removes a suffix \\r character if present (problem with some env var systems)"""
    v = os.getenv(name, default)
    if v.endswith('\r'):
        return v[:-1]
    else:
        return v


def assert_callable(f: callable) -> callable:
    assert callable(f), f"Is not callable: {f}"
    return f


def dotpath_to_obj(dotpath: str):
    """Loads and returns the object referenced by the string DOTPATH_TO_MODULE.OBJ_NAME"""
    *module_path, obj_name = dotpath.split('.')
    if len(module_path) > 0:
        return getattr(importlib.import_module('.'.join(module_path)), obj_name)
    else:
        return importlib.import_module(obj_name)


def dotpath_to_func(f: str) -> callable:
    """Loads and returns the function referenced by f,
    which could be a callable or a DOTPATH_TO_MODULE.FUNC_NAME dotpath string to one.

    >>> f = dotpath_to_func('os.path.join')
    >>> assert callable(f)  # I got a callable!
    >>> assert f.__name__ == 'join'  # and indeed, it's name is join
    >>> from inspect import signature
    >>> signature(f)  # and it's signature is indeed that of os.path.join:
    <Signature (a, *p)>
    >>>
    >>> # and just for fun...
    >>> assert signature(dotpath_to_func('inspect.signature')) == signature(signature)
    """

    assert isinstance(f, str) and '.' in f, f"Must be a string with at least one dot in the dot path: {f}"
    *module_path, func_name = f.split('.')
    f = getattr(importlib.import_module('.'.join(module_path)), func_name)
    return assert_callable(f)


def compose(*functions):
    """Make a function that is the composition of the input functions"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def dflt_func_loader(f) -> callable:
    """Loads and returns the function referenced by f,
    which could be a callable or a DOTPATH_TO_MODULE.FUNC_NAME dotpath string to one, or a pipeline of these.
    """
    if callable(f):
        return f
    elif isinstance(f, str):
        return dotpath_to_func(f)
    else:
        return compose(*map(dflt_func_loader, f))


def _fakit(f: callable, a: (tuple, list), k: dict):
    """The function that actually executes the fak command.
    Simply: `f(*(a or ()), **(k or {}))`

    >>> _fakit(print, ('Hello world!',), {})
    Hello world!

    """
    return f(*(a or ()), **(k or {}))  # slightly protected form of f(*a, **k)


def fakit_from_dict(d, func_loader=assert_callable):
    """Execute `f(*a, **k)` where `f`, `a`, and `k` are specified in a dict with those fields.

    This function does two things for you:
    - grabs the `(f, a, k)` triple from a `dict` (where both `'a'` and `'k'` are optional)
    - gives the user control over how the `f` specification resolves to a callable.

    """
    return _fakit(func_loader(d['f']), a=d.get('a', ()), k=d.get('k', {}))


def fakit_from_tuple(t: (tuple, list), func_loader: callable = dflt_func_loader):
    """In this one you specify the `fak` with a tuple (or list).

    You always have to specify a function as the first element of a list (and if you can call it without arguments,
    that's all you need).

    fakit_from_tuple(['builtins.list'])

    But as far as arguments are concerned, you can use a tuple or list
    (which will be understood to be the positional arguments (`*a`)):

    >>> A = fakit_from_tuple(['collections.namedtuple', ('A', 'x y z')])
    >>> A('no', 'defaults', 'here')
    A(x='no', y='defaults', z='here')

    ... you can also use a dict (which will be understood to be the keyword arguments (`**k`)):

    >>> A = fakit_from_tuple(['collections.namedtuple', {'typename': 'A', 'field_names': 'x y z'}])
    >>> A('no', 'defaults', 'here')
    A(x='no', y='defaults', z='here')

    ... or both:

    >>> A = fakit_from_tuple(['collections.namedtuple', ('A', 'x y z'), {'defaults': ('has', 'defaults')}])
    >>> A('this one')
    A(x='this one', y='has', z='defaults')

    :param t: A tuple or list
    :param func_loader: A function that will resolve the function to be called
    :return: What ever the function, called on the given arguments, will return.


    """
    f = func_loader(t[0])
    a = ()
    k = {}
    assert len(t) in {1, 2, 3}, "A tuple fak must be of length 1, 2, or 3. No more, no less."
    if len(t) > 1:
        if isinstance(t[1], dict):
            k = t[1]
        else:
            assert isinstance(t[1], (tuple, list)), "argument specs should be dict, tuple, or list"
            a = t[1]
        if len(t) > 2:
            if isinstance(t[2], dict):
                assert not k, "can only have one kwargs"
                k = t[2]
            else:
                assert isinstance(t[2], (tuple, list)), "argument specs should be dict, tuple, or list"
                assert not a, "can only have one args"
                a = t[2]
    return _fakit(f, a, k)


def fakit(fak, func_loader=dflt_func_loader):
    """Execute a fak with given f, a, k and function loader.

    Essentially returns `func_loader(f)(*a, **k)` where `(f, a, k)` are flexibly specified by `fak`.

    The `func_loader` is where you specify any validation of func specification and/or how to get a callable from it.
    The default `func_loader` will produce a callable from a dot path (e.g. `'os.path.join'`),
    But note that the intended use is for the user to use their own `func_loader`.
    The user should do this, amongst other things:
    - For security purposes, like not allowing `subprocess.call` or such.
    - For expressivity purposes, like to create their own domain specific mini-language
     that maps function specification to actual function.

    Args:
        fak: A (f, a, k) specification. Could be a tuple or a dict (with 'f', 'a', 'k' keys). All but f are optional.
        func_loader: A function returning a function.

    Returns: A python object.

    >>> fak = {'f': 'os.path.join', 'a': ['I', 'am', 'a', 'filepath']}
    >>> assert fakit(fak) =='I/am/a/filepath' or fakit(fak) == 'I\am\a\filepath'


    >>> from inspect import signature
    >>>
    >>> A = fakit({'f': 'collections.namedtuple', 'a': ['A', 'x y z'], 'k': {'defaults': (2, 3)}})
    >>> # A should be equivalent to `collections.namedtuple('A', 'x y z', defaults=(2, 3))`
    >>> signature(A)
    <Signature (x, y=2, z=3)>
    >>> A(1)
    A(x=1, y=2, z=3)
    >>> A(42, z='forty two')
    A(x=42, y=2, z='forty two')

    >>> def foo(x, y, z=3):
    ...     return x + y * z
    >>> func_map = {
    ...     'foo': foo,
    ...     'bar': (lambda a, b='world': f"{a} {b}!"),
    ...     'sig': signature}
    >>> from functools import partial
    >>> call_func = partial(fakit, func_loader=func_map.get)
    >>> call_func({'f': 'foo', 'a': (1, 10)})
    31
    """

    if isinstance(fak, dict):
        return fakit_from_dict(fak, func_loader=func_loader)
    else:
        assert isinstance(fak, (tuple, list)), "fak should be dict, tuple, or list"
        return fakit_from_tuple(fak, func_loader=func_loader)


fakit.from_dict = fakit_from_dict
fakit.from_tuple = fakit_from_tuple
fakit.w_func_loader = lambda func_loader: partial(fakit, func_loader=func_loader)
