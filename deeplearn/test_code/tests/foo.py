
program = """
def function(a, b):
    return a + b + c
"""

env = {}
exec(program, env)
#env['c'] = 4
#print (eval("function(2,3)", env))
f = env['function']
print (f(1,1))
