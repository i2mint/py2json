import os
import importlib
from warnings import warn
from functools import reduce
from py2store.util import DictAttr, str_to_var_str

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


def dotpath_to_obj(dotpath):
    """Loads and returns the object referenced by the string DOTPATH_TO_MODULE.OBJ_NAME"""
    *module_path, obj_name = dotpath.split('.')
    if len(module_path) > 0:
        return getattr(importlib.import_module('.'.join(module_path)), obj_name)
    else:
        return importlib.import_module(obj_name)


def dotpath_to_func(f: (str, callable)) -> callable:
    """Loads and returns the function referenced by f,
    which could be a callable or a DOTPATH_TO_MODULE.FUNC_NAME dotpath string to one.
    """

    if isinstance(f, str):
        if '.' in f:
            *module_path, func_name = f.split('.')
            f = getattr(importlib.import_module('.'.join(module_path)), func_name)
        else:
            f = getattr(importlib.import_module('py2store'), f)

    return assert_callable(f)


def compose(*functions):
    """Make a function that is the composition of the input functions"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def dflt_func_loader(f) -> callable:
    """Loads and returns the function referenced by f,
    which could be a callable or a DOTPATH_TO_MODULE.FUNC_NAME dotpath string to one, or a pipeline of these
    """
    if isinstance(f, str) or callable(f):
        return dotpath_to_func(f)
    else:
        return compose(*map(dflt_func_loader, f))


def _fakit(f: callable, a: (tuple, list), k: dict):
    return f(*(a or ()), **(k or {}))


def fakit_from_dict(d, func_loader=assert_callable):
    return _fakit(func_loader(d['f']), a=d.get('a', ()), k=d.get('k', {}))


def fakit_from_tuple(t: (tuple, list), func_loader: callable = dflt_func_loader):
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

    Essentially returns func_loader(f)(*a, **k)

    Args:
        fak: A (f, a, k) specification. Could be a tuple or a dict (with 'f', 'a', 'k' keys). All but f are optional.
        func_loader: A function returning a function. This is where you specify any validation of func specification f,
            and/or how to get a callable from it.

    Returns: A python object.

    >>> fak = {'f': 'os.path.join', 'a': ['I', 'am', 'a', 'filepath']}
    >>> fakit(fak)
    'I/am/a/filepath'
    """

    if isinstance(fak, dict):
        return fakit_from_dict(fak, func_loader=func_loader)
    else:
        assert isinstance(fak, (tuple, list)), "fak should be dict, tuple, or list"
        return fakit_from_tuple(fak, func_loader=func_loader)


fakit.from_dict = fakit_from_dict
fakit.from_tuple = fakit_from_tuple

if __name__ == '__main__':
    fak = {'f': 'os.path.join', 'a': ['I', 'am', 'a', 'filepath']}
    fakit(fak)
    'I/am/a/filepath'
