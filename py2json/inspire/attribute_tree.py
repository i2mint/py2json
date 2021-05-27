"""
f
"""
from anytree import Node, AnyNode, RenderTree, ContStyle, NodeMixin
from warnings import warn

warn(
    'Keeping around until final kv_walk based version is done',
    PendingDeprecationWarning,
)

NodeMixin.separator = '.'

DFLT_ON_ERROR = 'warn'


def handle_error(e, on_error=DFLT_ON_ERROR):
    if on_error == 'warn':
        warn(str(e))
    elif on_error == 'skip':
        pass
    elif on_error == 'raise':
        raise
    else:
        raise ValueError(f'Unknown on_error value: {on_error}')


def get_attr_and_vals(obj, on_error=DFLT_ON_ERROR):
    for k in dir(obj):
        try:
            yield k, getattr(obj, k)
        except Exception as e:
            handle_error(e, on_error)


def attr_and_vals_dict(obj, filt=None):
    return {k: v for k, v in filter(filt, get_attr_and_vals(obj))}


# Filters
def value_not_callable(kv):
    return not callable(kv[1])


class AttrTreeImporter:
    _base_types = (int, float, str, bytes, bool, list)

    def __init__(self, val_types=(), max_levels=99, on_error=DFLT_ON_ERROR):
        self.val_types = tuple(set(val_types).union(self._base_types))
        self.on_error = on_error
        self.max_levels = max_levels

    def attr_and_vals(self, obj, level=0):
        if level < self.max_levels:
            for k, v in get_attr_and_vals(obj):
                if isinstance(v, self.val_types):
                    yield k, v
                else:
                    yield k, self.dict_of_obj(v, level + 1)

    def dict_of_obj(self, obj, level=0):
        return dict(self.attr_and_vals(obj, level))

    def tree_of_dict(self, d, parent=None, root_name='root'):
        if parent is None:
            parent = Node(root_name)
        for k, v in d.items():
            if isinstance(v, dict):
                n = Node(k, parent=parent)
                self.tree_of_dict(v, parent=n)
            else:
                n = Node(k, parent=parent, val=v)
        return parent

    def tree_of_obj(self, obj, root_name=None):
        if root_name is None:
            if hasattr(obj, '__name__'):
                root_name = obj.__name__
            elif hasattr(obj, '__class__'):
                root_name = obj.__class__.__name__
            else:
                root_name = 'root'
        return self.tree_of_dict(self.dict_of_obj(obj), root_name=root_name)
