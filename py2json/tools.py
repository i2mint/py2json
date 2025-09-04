"""Higher-level tools to create python<->json codecs using py2json primitives.

This module provides a small `Codec` object and `make_json_codec` factory to wire
py2json's `Ctor` and `fakit` into a simple encode/decode interface. It also
offers configurable "path protocols" for how dotted references are expressed.

Design goals:
- Keep defaults simple: dotted paths resolved with `fakit.dotpath_to_obj`.
- Support alternative path formats (e.g. colon-separated `module:attr`).
- Use `Ctor.deconstruct`/`Ctor.construct` for python<->json conversions and
  `fakit.refakit` for evaluating $fak expressions.
"""

from typing import Any, Callable, Optional, Union

from py2json.ctor import Ctor
from py2json.fakit import FAK, refakit, dotpath_to_obj


def _default_colon_parser(s: str):
    """Parse strings like 'package.module:attr' into a dotted import form.

    This simple parser returns a string suitable for `dotpath_to_obj` by
    converting the colon into a dot join: 'pkg.mod:Name.attr' -> 'pkg.mod.Name.attr'
    """
    if ':' not in s:
        return s
    module, rest = s.split(':', 1)
    # rest may contain dots already
    return module + '.' + rest


def _mk_parser_from_sep(sep: str) -> Callable[[str], str]:
    """Return a parser function interpreting `sep` as module/object separator.

    If sep == '.', treat strings as already dotted and return them verbatim.
    Otherwise split on the first occurrence of sep and join with a dot so that
    e.g. 'pkg.mod:Name.attr' with sep=':' -> 'pkg.mod.Name.attr'.
    """
    if sep == '.':
        return lambda s: s

    def parser(s: str) -> str:
        if sep not in s:
            return s
        module, rest = s.split(sep, 1)
        return module + '.' + rest

    return parser


class JsonCodec:
    """A tiny codec that converts Python -> JSON-friendly dicts and back.

    encode(obj) -> jsonable
    decode(j) -> python object

    Options allow customizing the function loader used by `refakit` via
    `func_loader` and a `path_parser` to normalize string references.
    """

    def __init__(
        self,
        *,
        func_loader: Optional[Callable[[Any], Callable]] = None,
        path_parser: Optional[Union[Callable[[str], str], str]] = '.',
    ):
        self.func_loader = func_loader
        # path_parser may be a callable or a string separator; default is '.'
        if callable(path_parser):
            self.path_parser = path_parser
        elif isinstance(path_parser, str):
            # interpret the string as the separator between module and object
            self.path_parser = _mk_parser_from_sep(path_parser)
        else:
            raise TypeError('path_parser must be a callable or a separator string')

    # encode python -> json-friendly structure
    def encode(self, obj: Any) -> Any:
        """Encode a Python object into a JSON-friendly structure.

        For non-basic objects we delegate to `Ctor.deconstruct(..., output_type=Ctor.JSON_DICT)`.
        Basic json types pass through unchanged.
        """
        if Ctor.is_python_to_json_basic_type(obj):
            return obj
        # allow passing a path string that should be normalized by path_parser
        if isinstance(obj, str):
            return self.path_parser(obj)
        # otherwise use Ctor to deconstruct complex objects
        return Ctor.deconstruct(obj, output_type=Ctor.JSON_DICT)

    # decode json-like -> python
    def decode(self, j: Any) -> Any:
        """Decode a JSON-friendly structure back into Python.

        Behaviour:
        - If it's a dict with the fak key (FAK), use refakit to evaluate it
          (respecting the provided func_loader).
        - Otherwise, attempt to construct via `Ctor.construct`, which will
          handle nested ctor-jdicts produced by `Ctor.deconstruct`.
        """
        # handle fak expressions first
        if isinstance(j, dict) and FAK in j:
            if self.func_loader is None:
                return refakit(j)
            return refakit(j, func_loader=self.func_loader)

        # otherwise use Ctor.construct which handles ctor-jdicts and nested structures
        try:
            return Ctor.construct(j)
        except Exception:
            # If construct fails, return original structure (best-effort)
            return j


def make_json_codec(
    *,
    func_loader: Optional[Callable[[Any], Callable]] = None,
    path_parser: Optional[Union[Callable[[str], str], str]] = '.',
) -> JsonCodec:
    """Factory for a Codec configured with a func_loader and path protocol.

    Args:
        func_loader: Optional callable used by `refakit` to resolve function names.
        path_protocol: 'dot' (default) or 'colon' — controls default path parsing.
        path_parser: optional custom parser for string references.
    """
    return JsonCodec(func_loader=func_loader, path_parser=path_parser)


__all__ = ['JsonCodec', 'make_json_codec']
