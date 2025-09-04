from py2json.tools import make_json_codec


def test_colon_path_parser_and_codec():
    codec = make_json_codec(path_parser=':')
    s = 'collections.namedtuple:Z.a'
    enc = codec.encode(s)
    # colon parser should convert into 'collections.namedtuple.Z.a'
    assert isinstance(enc, str) and 'collections.namedtuple' in enc


def test_codec_decode_fak():
    codec = make_json_codec()
    fak_expr = {'$fak': ('builtins.sum', ([1, 2, 3],))}
    assert codec.decode(fak_expr) == 6
