stream_port = 9999
stream_ip = "*"
command_port = 99887
command_ip = "*"
learning_rate = 0.01
optimizer = 'adagrad'
batch_size = 128
validation_size = 64
checkpoint_folder = ""
checkpoint_interval = 50

def fill_arg_parser(parser):
    parser.add_argument('-s', '--stream-to', dest='stream_to',
                        type=str,
                        default=stream_dest,
                        help='The IP address of the training loop to send the data to, in the format 0.0.0.0:port')
    parser.add_argument('-b', '--batch-size', dest='batch_size',
                        type=int,
                        default=stream_batch_size,
                        help='The size of the mini-batches to sent to the training server')
    parser.add_argument('-V', '--validation-size',
                        dest='validation_size',
                        type=int,
                        default=stream_validation_size,
                        help='The number of validation samples to send to the training server')
    parser.add_argument('-I', '--validation-interval',
                        dest='validation_interval',
                        type=int,
                        default=stream_validation_interval,
                        help='Validate every N batches')
    parser.add_argument('-e', '--epochs', dest='epochs',
                        type=int,
                        default=stream_epochs,
                        help='The number of epochs, i.e. the number of times the loop should see each elmenet of the data set')
