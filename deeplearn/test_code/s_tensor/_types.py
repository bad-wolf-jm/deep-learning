class _symbol(str):
    _inst = {}

    @classmethod
    def new(cls, name):
        print (cls._inst)
        if name in cls._inst:
            return cls._inst[name]
        new_symbol = cls(name)
        cls._inst[name]=new_symbol
        return new_symbol

class _keyword(_symbol):
    _inst = {}

class _pair(object):
    def __init__(self, car=None, cdr=None):
        self._car = car
        self._cdr = cdr

class _empty_list_type(object):
    pass
_empty_list_instance = _empty_list_type()

def _empty_list():



if __name__ == '__main__':
    s1 = _symbol.new('a')
    s2 = _symbol.new('b')
    s3 = _symbol.new('a')
    k1 = _keyword.new('a')
    k2 = _keyword.new('b')
    k3 = _keyword.new('a')

    assert s1 is not s2
    assert s1 is s3
    assert k1 is not k2
    assert k1 is k3
    assert s1 is not k1
    assert s1 is not k2
    assert s1 is not k3
    assert s2 is not k1
    assert s2 is not k2
    assert s2 is not k3
    assert s3 is not k1
    assert s3 is not k2
    assert s3 is not k3
