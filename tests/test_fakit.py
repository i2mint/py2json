from py2json.fakit import fakit, refakit


def test_fakit_join():
    res = fakit({'f': 'os.path.join', 'a': ['I', 'am', 'a', 'filepath']})
    # platform dependent separator, just ensure parts are present
    assert 'I' in res and 'filepath' in res


def test_refakit_nested():
    t = {'$fak': ('builtins.sum', ([1, 2, 3],))}
    assert refakit(t) == 6
