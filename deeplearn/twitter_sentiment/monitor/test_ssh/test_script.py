import sys

print "FOOBAR"

line = sys.stdin.read(5)
while line != '':
#for i in range(100):
    sys.stdout.write('[REPLY]' + line + '\n')
    sys.stdout.flush()
    #line = sys.stdin.read(5)
    #print line
