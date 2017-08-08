

class N:
    foo = 1
    bar = 2


print(N.foo, N.bar)

N.foo = 3
N.bar = 4
print(N.foo, N.bar)

def fun():
    N.foo = 'a'
    N.bar = 'b'

fun()
print(N.foo, N.bar)
