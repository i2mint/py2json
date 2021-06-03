"""
Functions to automatically make json serializers/deserializers for sklearn estimators
"""
from operator import eq
from functools import partial, wraps

import numpy as np
from py2json.util import mk_func_to_kwargs_from_a_val_for_argname_map

from sklearn.utils import all_estimators
from i2.signatures import Sig

all_estimator_classes = tuple([obj for name, obj in all_estimators()])


def funcs_that_need_args(funcs, func_to_kwargs=None, self_name=None):
    """
    >>> from sklearn.utils import all_estimators
    >>> estimator_classes = [obj for name, obj in all_estimators()]
    >>> estimator_names = list(map(lambda x: x.__name__, funcs_that_need_args(all_estimator_classes)))
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
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import Normalizer
from scipy.stats import uniform
from collections import (
    UserDict,
)  # using UserDict instead of dict because want the dict to have a doc

estimator_cls_val_for_argname = {
    'estimator': ElasticNetCV,
    'base_estimator': ElasticNetCV,
    'estimators': (PCA, ElasticNetCV),
    'transformers': [
        ('norm1', Normalizer(norm='l1'), [0, 1]),
        ('norm2', Normalizer(norm='l1'), slice(2, 4)),
    ],
    'transformer_list': [
        ('pca', PCA(n_components=1)),
        ('svd', TruncatedSVD(n_components=2)),
    ],
    'steps': [('PCA', PCA), ('ElasticNetCV', ElasticNetCV)],
    'param_grid': {'kernel': ('linear', 'rbf'), 'C': [1, 10]},
    'param_distributions': dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1']),
}

estimator_cls_val_for_argname = UserDict(estimator_cls_val_for_argname)
estimator_cls_val_for_argname.__doc__ = '''Valid values for default-less arguments of sklearn estimators.
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
'''

estimator_cls_to_kwargs = mk_func_to_kwargs_from_a_val_for_argname_map(
    estimator_cls_val_for_argname
)

estimator_classes_without_resolved_kwargs = set(
    funcs_that_need_args(all_estimator_classes, estimator_cls_to_kwargs)
)


def is_behaviorally_equivalent(obj1, obj2, behavior_func, output_comp=eq):
    """Checks if two objects obj1 and obj2 are behaviorally equivalent,
    according to behavior func (a callable that will be called on the objects) and
    output_comp (the function that will be called on the outputs of these calls to
    decide whether they can be considered equivalent.

    You'll notice that `behavior_func` takes no other input than `obj` (or `other_obj`).
    Most behaviors depend on other variables you say? We can hardly say that two objects are equivalent
    if they agree on only one data point you say? Yes, I agree. But `is_behaviorally_equivalent` doesn't rule these out.

    What to do if you want to validate behavioral equivalence like the following?
    ```python
    def many_validation_pts():
        for x, y, z in some_large_set_of_xyz_combinations:
            yield output_comp(behavior_func(x, y, obj, z), behavior_func(x, y, other_obj, z))
    assert(all(many_validation_pts()))
    ```

    Well, instead of loading the `is_behaviorally_equivalent` with `args`, `kwargs`, and `argname_of_obj`
    to be able to acheive the above, we simply ask the user to encompass it all in the provided `behavior_func` itself,
    using standard tools such as `functools.partial` or custom decorators.
    For example:

    ```python
    from functools import partial
    def many_validation_pts():
        for x, y, z in some_large_set_of_xyz_combinations:
            _behavior_func = partial(behavior_func, x=x, y=y, z=z)
            yield is_behaviorally_equivalent(obj, other_obj, _behavior_func, output_comp)
    assert(all(many_validation_pts()))
    ```

    Suggestions:
        output_comp=np.allclose or np.isclose
    """
    return output_comp(behavior_func(obj1), behavior_func(obj2))


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
    raise NotImplementedError('')


def xy_and_fitted_model_for_estimator_cls(estimator_cls):
    """"""
    init_params = init_params_for_cls(estimator_cls)
    learner = estimator_cls(**init_params)
    X, y = xy_for_learner(learner)
    return X, y, learner.fit(X, y)  # assuming sklearn style


def compose(func1, func2):
    sig1 = Sig(func1)
    sig2 = Sig(func2)

    @sig1
    def composed_funcs(*args, **kwargs):
        return func2(func1(*args, **kwargs))

    composed_funcs.__return_annotation__ = sig2.return_annotation
    return composed_funcs


import pickle

dflt_mk_alt_model = compose(pickle.dumps, pickle.loads)

from sklearn.datasets import make_blobs

_X, _y = make_blobs(center_box=(0, 20.0))


def dflt_xy_for_learner(learner):
    """Returns some random (but fixed after first import) (X, y) pair, insensitive to the learner"""
    return _X, _y


def behavioral_test_kwargs_for_estimator(
    estimator_cls,
    init_params_for_cls=estimator_cls_to_kwargs,  # Returns valid params to initialize an estimator_cls
    xy_for_learner=dflt_xy_for_learner,  # Returns an (X, y) pair for the learner to fit on
    mk_alt_model=dflt_mk_alt_model,  # Returns an alt object by serializing and deserializing the fitted model
    methods=('predict', 'transform',),  # The methods to use to make behavior_funcs,
    output_comp=np.allclose,
):
    """Generate `is_behaviorally_equivalent` kwargs from `estimator_cls`

    :param estimator_cls:
    :param init_params_for_cls: Returns valid params to initialize an estimator_cls
    :param xy_for_learner: Returns an (X, y) pair for the learner to fit on
    :param mk_alt_obj: Returns an alt object by serializing and deserializing the fitted model
    :param methods: The methods to use to make behavior_funcs
    :return Generator of (obj, alt_obj, behavior_func, output_comp) tuples
    """
    model, X, y = model_X_y(estimator_cls, init_params_for_cls, xy_for_learner)
    alt_model = mk_alt_model(model)
    for method in methods:
        if hasattr(estimator_cls, method):
            method_func = getattr(estimator_cls, method)
            yield dict(
                obj1=model,
                obj2=alt_model,
                behavior_func=partial(method_func, X=X),
                output_comp=output_comp,
            )


def model_X_y(estimator_cls, init_params_for_cls, xy_for_learner):
    estimator_params = init_params_for_cls(estimator_cls)
    learner = estimator_cls(**estimator_params)
    X, y = xy_for_learner(learner)
    model = learner.fit(X, y)
    return model, X, y


from py2json.util import catch_errors


def fitting_x_y_works(learner, X, y):
    try:
        learner.fit(X, y)
        return True
    except Exception as e:
        return False


@catch_errors(on_error=None)
def cls_to_learner_or_none(cls, init_params_for_cls=init_params_for_cls):
    return cls(**init_params_for_cls(cls))


@catch_errors(on_error=None)
def fit_or_return_none(learner, X, y):
    learner.fit(X, y)
    return learner


def fit_learners(learners, X, y, ignore_warnings=True):
    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter('ignore')
        for i, learner in enumerate(learners):
            try:
                yield fit_or_return_none(learner, X, y)
            except Exception as e:
                print(f'{i}: {learner.__class__.__name__}: {e}')
                yield None


import warnings


class ReprControlledTuple(tuple):
    def __repr__(self):
        t = super().__repr__()
        n = len(t)
        if n > 100:
            return f'{t[:100]}... ({len(self)} elements)'
        else:
            return t


def model_X_y_gen(
    estimator_classes=None,
    init_params_for_cls=estimator_cls_to_kwargs,  # Returns valid params to initialize an estimator_cls
    xy_for_learner=dflt_xy_for_learner,  # Returns an (X, y) pair for the learner to fit on
    on_error='raise',
):
    if estimator_classes is None:
        estimator_classes = all_estimator_classes
    for cls in estimator_classes:
        try:
            yield model_X_y(cls, init_params_for_cls, xy_for_learner)
        except Exception as e:
            if on_error == 'raise':
                raise
            elif on_error == 'ignore':
                continue
            elif on_error == 'yield_context':
                dict(kind='error_in_model_X_y', cls=cls, error=e)
            else:
                raise ValueError(f'Unrecognized on_error: {on_error}')


def test_estimators(
    estimator_classes=None,
    ignore_warnings=True,
    **kwargs_for_behavioral_test_kwargs_for_estimator,
):
    if estimator_classes is None:
        estimator_classes = all_estimator_classes
    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter('ignore')

        for estimator_cls in estimator_classes:
            try:
                for kws in behavioral_test_kwargs_for_estimator(
                    estimator_cls, **kwargs_for_behavioral_test_kwargs_for_estimator,
                ):
                    try:
                        if is_behaviorally_equivalent(**kws):
                            yield dict(kind='ok', cls=estimator_cls)
                        else:
                            yield dict(
                                kind='behaviorally_different',
                                cls=estimator_cls,
                                kws=kws,
                            )
                    except Exception as e:
                        yield dict(
                            kind='error_in_test', cls=estimator_cls, kws=kws, error=e,
                        )
            except Exception as e:
                yield dict(kind='error_making_test_kws', cls=estimator_cls, error=e)


def estimator_test_df(
    estimator_classes=None,
    ignore_warnings=True,
    **kwargs_for_behavioral_test_kwargs_for_estimator,
):
    import pandas as pd

    if estimator_classes is None:
        estimator_classes = all_estimator_classes
    return pd.DataFrame(
        list(
            test_estimators(
                estimator_classes,
                ignore_warnings,
                **kwargs_for_behavioral_test_kwargs_for_estimator,
            )
        )
    )


def get_estimator_classes_that_are_behaviorally_equivalent_wrt_pickle():
    df = estimator_test_df()
    ok_to_test_classes = tuple(df[df['kind'] == 'ok']['cls'].values)
    len(ok_to_test_classes)


from py2json.inspire.example_fakit_use import (
    mk_serializer_and_deserializer_from_instance_and_methods,
)
