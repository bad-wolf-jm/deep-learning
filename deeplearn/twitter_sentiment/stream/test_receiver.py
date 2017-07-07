from stream import DataReceiver


foo = DataReceiver()

def pp(train_x = None, train_y = None, **kw):
    print('train', list(zip(train_x, train_y)))
def qq(train_x = None, train_y = None, **kw):
    print('validate', list(zip(train_x, train_y)))

foo.register_action_handler('train', pp)
foo.register_action_handler('validate', qq)
foo.start(False)
