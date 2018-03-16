import datetime
import re
import subprocess
import time



def is_zombie(p_uuid):
    x = subprocess.check_output(["ps", '-axf'])
    lines = x.split('\n')
    uuid_re = re.compile(p_uuid)
    for l in lines:
        if uuid_re.search(l):
            return False
    return True



def FOO(l):
    number_re = re.compile(r"\d+")
    elements = l.split(' ')
    i = 1
    for e in elements:
        if len(e) > 0 and number_re.match(e):
            if i == 0:
                return int(e)
            else:
                i -= 1
        #else:
        #    return None



#        print elemen

#print(len(lines))