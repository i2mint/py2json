"""
f
"""
import inspect


def func_to_jdict(func):
    """
    >>> def multiplier(a, b):
    ...     return a * b
    >>> jdict = func_to_jdict(multiplier)
    >>> assert jdict == {'$py_source_lines': 'def multiplier(a, b):\\n    return a * b\\n'}
    """
    lines = inspect.getsource(func)
    return {'$py_source_lines': lines}


def jdict_to_func(jdict):
    """
    >>> jdict = {'$py_source_lines': 'def multiplier(a, b):\\n    return a * b\\n'}
    >>> deserialized_func = jdict_to_func(jdict)
    >>> deserialized_func(7, 6)
    42
    """
    _locals = {}
    exec(jdict['$py_source_lines'], None, _locals)
    func_name, func_obj = next(iter(_locals.items()))
    return func_obj


def test_this():
    def multiplier(a, b):
        return a * b

    func = multiplier

    jdict = func_to_jdict(func)
    assert jdict == {'$py_source_lines': 'def multiplier(a, b):\n    return a * b\n'}

    deserialized_func = jdict_to_func(jdict)
    assert deserialized_func != func  # not equal, but....
    assert deserialized_func(7, 6) == 42
