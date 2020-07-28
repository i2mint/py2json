from operator import eq

import numpy as np


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


from sklearn.utils import all_estimators
from i2.signatures import Sig

estimator_classes = [obj for name, obj in all_estimators()]


def funcs_that_need_args(funcs, func_to_kwargs=None, self_name=None):
    """
    >>> from sklearn.utils import all_estimators
    >>> estimator_classes = [obj for name, obj in all_estimators()]
    >>> estimator_names = list(map(lambda x: x.__name__, funcs_that_need_args(estimator_classes)))
    >>> expected = [
    ... 'ClassifierChain', 'ColumnTransformer', 'FeatureUnion', 'GridSearchCV', 'MultiOutputClassifier',
    ... 'MultiOutputRegressor', 'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier',
    ... 'Pipeline', 'RFE', 'RFECV', 'RandomizedSearchCV', 'RegressorChain', 'SelectFromModel',
    ... 'SparseCoder', 'StackingClassifier', 'StackingRegressor', 'VotingClassifier', 'VotingRegressor']
    >>>
    >>> import sklearn
    >>> if sklearn.__version__ == '0.23.1':
    ...      assert estimator_names == expected
    """
    for func in funcs:
        required_args = set(Sig(func).without_defaults.parameters) - {self_name}
        if func_to_kwargs is not None:
            required_args -= func_to_kwargs(func).keys()
        if required_args:
            yield func


from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import Normalizer
from scipy.stats import uniform
from collections import UserDict  # using UserDict instead of dict because want the dict to have a doc

estimator_cls_val_for_argname = {
    'estimator': ElasticNetCV,
    'base_estimator': ElasticNetCV,
    'estimators': (PCA, ElasticNetCV),
    'transformers': [("norm1", Normalizer(norm='l1'), [0, 1]),
                     ("norm2", Normalizer(norm='l1'), slice(2, 4))],
    'transformer_list': [("pca", PCA(n_components=1)),
                         ("svd", TruncatedSVD(n_components=2))],
    'steps': [('PCA', PCA), ('ElasticNetCV', ElasticNetCV)],
    'param_grid': {'kernel': ('linear', 'rbf'), 'C': [1, 10]},
    'param_distributions': dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1']),
}

estimator_cls_val_for_argname = UserDict(estimator_cls_val_for_argname)
estimator_cls_val_for_argname.__doc__ = """Valid values for default-less arguments of sklearn estimators.
A dict whose keys are all (but one -- sklearn.decomposition._dict_learning.SparseCoder)
the default-less arguments of all sklearn estimators and 
the values provide a valid default for them.

How did you do it, you ask?

Well, I 
>>> val_for_argname = {}  # a mapping from argname and val I updated at every "iteration"
>>> # make a func_to_kwargs from it:
>>> func_to_kwargs = sse.mk_func_to_kwargs_from_a_val_for_argname_map(val_for_argname)
>>>
>>> # make a filter function for that func_to_kwargs
>>> not_covered_by_our_policy = sse.missing_args_func(func_to_kwargs)
>>>
>>> # get the next class that's not covered
>>> not_covered_cls = next(filter(not_covered_by_our_policy, sse.estimator_classes), None)
>>>
>>> if not_covered_cls is not None:
...     from i2.doc_mint import doctest_string_print
...     doctest_string_print(not_covered_cls)
>>> # then I read that example code, figured out a valid input, and re-iterated.
"""

estimator_cls_to_kwargs = mk_func_to_kwargs_from_a_val_for_argname_map(estimator_cls_val_for_argname)

estimator_classes_without_resolved_kwargs = set(funcs_that_need_args(
    estimator_classes, estimator_cls_to_kwargs))


def is_behaviorally_equivalent(obj1, obj2, func, output_comp=eq):
    """Checks if two objects obj1 and obj2 are behaviorally equivalent,
    according to behavior func (a callable that will be called on the objects) and
    output_comp (the function that will be called on the outputs of these calls to
    decide whether they can be considered equivalent.

    Suggestions:
        output_comp=np.isclose
    """
    return output_comp(func(obj1), func(obj2))


def init_params_for_cls(estimator_cls):
    """Gets valid params to initialize an estimator_cls
    :param estimator_cls: A (sklearn estimator) class
    :return: init_params such that estimator_cls(**init_params) is valid
    """
    return estimator_cls_to_kwargs(estimator_cls)
    # raise NotImplementedError("")


def xy_for_learner(learner):
    """Gets valid params to initialize an estimator_cls
    :param learner: A (sklearn estimator) INSTANCE
    :return: (X, y) pair such that learner.fit(X, y) works in a meaningful way
    """
    raise NotImplementedError("")


def xy_and_fitted_model_for_estimator_cls(estimator_cls):
    """
    """
    init_params = init_params_for_cls(estimator_cls)
    learner = estimator_cls(**init_params)
    X, y = xy_for_learner(learner)
    return X, y, learner.fit(X, y)  # assuming sklearn style


def test_estimator(estimator_cls, serializer, deserializer,
                   methods=('predict', 'transform'), comp_func=np.isclose):
    X, y, model = xy_and_fitted_model_for_estimator_cls(estimator_cls)

    deserialized_model = deserializer(serializer(model))

    for method in methods:
        if hasattr(estimator_cls, method):
            method_func = getattr(estimator_cls, method)
            yield method, is_behaviorally_equivalent(model, deserialized_model, method_func, comp_func)
