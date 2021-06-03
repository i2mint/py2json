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
from functools import partial
from collections.abc import Mapping

from py2json.util import compose

FAK = '$fak'


# TODO: Make a config_utils.py module to centralize config tools (configs for access is just one
#  -- serializers another)
# TODO: Integrate (external because not standard lib) other safer tools for secrets, such as:
#  https://github.com/SimpleLegal/pocket_protector


def getenv(name, default=None):
    """Like os.getenv, but removes a suffix \\r character if present (problem with some env var
    systems)"""
    v = os.getenv(name, default)
    if v.endswith('\r'):
        return v[:-1]
    else:
        return v


def assert_callable(f: callable) -> callable:
    assert callable(f), f'Is not callable: {f}'
    return f


def dotpath_to_obj(dotpath: str):
    """Loads and returns the object referenced by the string DOTPATH_TO_MODULE.OBJ_NAME"""
    first, *remaining = dotpath.split('.')
    obj = importlib.import_module(first)  # assume it's a module
    for item in remaining:
        obj = getattr(obj, item)
    return obj


def obj_to_dotpath(obj):
    """Get the dotpath reference for an object

    >>> from inspect import Signature
    >>> obj_to_dotpath(Signature.replace)
    'inspect.Signature.replace'
    >>> # note that below, it's not a "full path"
    >>> obj_to_dotpath(func_to_dotpath)[-21:]  # the :21 is because the full string is sys dependent
    'fakit.func_to_dotpath'

    func_to_dotpath is the inverse of dotpath_to_func

    >>> assert dotpath_to_obj(obj_to_dotpath(Signature.replace)) == Signature.replace

    """
    return '.'.join((obj.__module__, obj.__qualname__))


def func_to_dotpath(func: callable) -> str:
    return obj_to_dotpath(assert_callable(func))


def dotpath_to_func(dotpath: str) -> callable:
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

    dotpath_to_func is the inverse of func_to_dotpath
    >>> assert dotpath_to_func(func_to_dotpath(signature)) == signature

    """
    obj = dotpath_to_obj(dotpath)
    return assert_callable(obj)


def dflt_func_loader(f) -> callable:
    """Loads and returns the function referenced by f,
    which could be a callable or a DOTPATH_TO_MODULE.FUNC_NAME dotpath string to one,
    or a pipeline of these.
    """
    if callable(f):
        return f
    elif isinstance(f, str):
        return dotpath_to_func(f)
    elif isinstance(f, dict) and len(f) == 1 and FAK in f:
        return fakit(f[FAK])
    else:
        return compose(*map(dflt_func_loader, f))


def _fakit(f: callable, a: (tuple, list), k: dict):
    """The function that actually executes the fak command.
    Simply: `f(*(a or ()), **(k or {}))`

    >>> _fakit(print, ('Hello world!',), {})
    Hello world!

    """
    return f(*(a or ()), **(k or {}))  # slightly protected form of f(*a, **k)


def extract_fak(fak):
    """Extracts the (raw) (f, a, k) triple from a dict or tuple/list fak.
    Also asserts the validity of input fak.

    >>> extract_fak(('func', (1, 2), {'keyword': 3}))
    ('func', (1, 2), {'keyword': 3})

    If fak has only two input items and the second is a dict, the second output will be an empty
    tuple.
    >>> extract_fak(('func', {'keyword': 3}))
    ('func', (), {'keyword': 3})

    If fak has only two input items and the second is a tuple, the second output will be an empty
    dict.
    >>> extract_fak(['func', (1, 2)])
    ('func', (1, 2), {})

    If you only have only one element in your list/tuple input...
    >>> extract_fak(['func'])
    ('func', (), {})

    If your input is a dict
    >>> extract_fak({'f': 'func', 'a': (1, 2), 'k': {'keyword': 3}})
    ('func', (1, 2), {'keyword': 3})
    >>> extract_fak({'f': 'func', 'k': {'keyword': 3}})
    ('func', (), {'keyword': 3})
    >>> extract_fak({'f': 'func', 'a': (1, 2)})
    ('func', (1, 2), {})
    """
    if isinstance(fak, dict):
        if 'f' not in fak:
            raise ValueError(f'There needs to be an `f` key, was not: {fak}')
        f = fak['f']
        a = fak.get('a', ())
        k = fak.get('k', {})
    else:
        assert isinstance(
            fak, (tuple, list)
        ), f'fak should be dict, tuple, or list, was not: {fak}'
        assert (
            len(fak) >= 1
        ), f'fak should have at least one element (the function component): {fak}'
        f = fak[0]
        a = ()
        k = {}
        assert len(fak) in {
            1,
            2,
            3,
        }, 'A tuple fak must be of length 1, 2, or 3. No more, no less.'
        if len(fak) > 1:
            if isinstance(fak[1], dict):
                k = fak[1]
            else:
                a = fak[1]
                assert isinstance(
                    a, (tuple, list)
                ), 'argument specs should be dict, tuple, or list'
            if len(fak) > 2:
                if isinstance(fak[2], dict):
                    assert not k, 'can only have one kwargs'
                    k = fak[2]
                else:
                    assert isinstance(
                        fak[2], (tuple, list)
                    ), 'argument specs should be dict, tuple, or list'
                    assert not a, 'can only have one args'
                    a = fak[2]

    assert isinstance(a, (tuple, list)), f'a kind is not a tuple or list: {fak}'
    assert isinstance(k, Mapping), f'k kind is not a mapping: {fak}'
    return f, a, k


def validate_fak(fak):
    """Returns the input iff (f, a, k) could be extracted and validated from input fak"""
    _ = extract_fak(fak)
    return fak


def is_valid_fak(fak):
    try:
        validate_fak(fak)
        return True
    except Exception:
        return False


def extract_and_load(fak, func_loader=dflt_func_loader):
    f, a, k = extract_fak(fak)
    f = func_loader(f)
    return f, a, k


def fakit(fak, func_loader=dflt_func_loader):
    """Execute a fak with given (f, a, k) tuple or {f: f, a: a, k: k} dict, and a function loader.

    Essentially returns `func_loader(f)(*a, **k)` where `(f, a, k)` are flexibly specified by `fak`.

    The `func_loader` is where you specify any validation of func specification and/or how to get
    a callable from it.
    The default `func_loader` will produce a callable from a dot path (e.g. `'os.path.join'`),
    But note that the intended use is for the user to use their own `func_loader`.
    The user should do this, amongst other things:
    - For security purposes, like not allowing `subprocess.call` or such.
    - For expressivity purposes, like to create their own domain specific mini-language
     that maps function specification to actual function.

    Args:
        fak: A (f, a, k) specification. Could be a tuple or a dict (with 'f', 'a', 'k' keys). All
        but f are optional.
        func_loader: A function returning a function.

    Returns: A python object.

    >>> fak = {'f': 'os.path.join', 'a': ['I', 'am', 'a', 'filepath']}
    >>> assert fakit(fak) =='I/am/a/filepath' or fakit(fak) == 'I\am\a\filepath'

    >>> A = fakit(['collections.namedtuple', ('A', 'x y z')])
    >>> A('no', 'defaults', 'here')
    A(x='no', y='defaults', z='here')

    ... you can also use a dict (which will be understood to be the keyword arguments (`**k`)):

    >>> A = fakit(['collections.namedtuple', {'typename': 'A', 'field_names': 'x y z'}])
    >>> A('no', 'defaults', 'here')
    A(x='no', y='defaults', z='here')

    ... or both:

    >>> A = fakit(['collections.namedtuple', ('A', 'x y z'), {'defaults': ('has', 'defaults')}])
    >>> A('this one')
    A(x='this one', y='has', z='defaults')

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

    Common gotcha: Forgetting that `a` is iterpreted as an iterable of function args. For example

    >>> fakit(('builtins.print', 'hello'))  # not correct
    Traceback (most recent call last):
      ...
    AssertionError: argument specs should be dict, tuple, or list
    >>> fakit(('builtins.print', ['hello']))  # correct
    hello

    >>> fakit(('builtins.sum', [1, 2, 3]))  # not correct
    Traceback (most recent call last):
      ...
    TypeError: sum() takes at most 2 arguments (3 given)
    >>> fakit(('builtins.sum', ([1, 2, 3],)))  # correct
    6

    """
    if isinstance(fak, dict) and FAK in fak:
        fak = fak[FAK]
    return _fakit(*extract_and_load(fak, func_loader))


fakit.w_func_loader = lambda func_loader: partial(fakit, func_loader=func_loader)


def fakit_if_marked_for_it(x, func_loader=dflt_func_loader):
    if isinstance(x, dict) and FAK in x:
        return fakit(x[FAK], func_loader)
    else:
        return x


inf = float('infinity')


def refakit(x, func_loader=dflt_func_loader, max_levels=inf):
    """Fakit recursively looking for nested {'$fak': ...} specifications of python objects

    :param x:
    :param func_loader:
    :return:

    >>> t = {'$fak': ('builtins.sum', ([1,2,3],))}
    >>> refakit(t)  # it works with one level
    6

    >>> ttt = {'$fak': ('builtins.sum', ([t, t],))}
    >>> refakit(ttt)
    12

    But this recursive interpretation of the the fakit elemnts in [t, t] would not
    happen if we restricted the max_levels to be 2 for example.

    The max levels is there to be able to specify that the refakit shouldn't go too deep in
    nested lists (and thus spare some computation.
    TODO: Perhaps we could include this max_levels as a specification in fakit?

    See also: `fakit`, the one level only version of `refakit`.
    """
    if max_levels == 0:
        return x
    if isinstance(x, dict) and FAK in x:
        f, a, k = extract_fak(x[FAK])

        # recurse over inputs to see if there's some that are expressed with a $fak dict
        a = [refakit(aa, func_loader, max_levels - 1) for aa in a]
        k = {kk: refakit(vv, func_loader, max_levels - 1) for kk, vv in k.items()}

        return fakit((f, a, k), func_loader)
    elif isinstance(x, list):
        return [refakit(xx, func_loader, max_levels - 1) for xx in x]
    else:
        return x
