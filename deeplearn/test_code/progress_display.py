import sys

max_line_length = 0

def basic_callback(**args):
    global max_line_length
    remaining_time_est = args['batch_time'] * (args['total_batches'] - args['batch_index'])
    line = "\rEpoch {0} of {1} --- A: {2:.2f}  - L: {3:.2f} --- VA: {4:.2f}  - VL: {5:.2f}--- Remaining time: {6}"
    line = line.format(args['epoch_number'] + 1, args['total_epochs'],
                       args['batch_accuracy'], args['batch_loss'],
                       args['validation_accuracy'], args['validation_loss'],
                       '{0:02d}:{1:02d}'.format(int(remaining_time_est) // 60, int(remaining_time_est) % 60),
                       )
    if len(line) <= max_line_length:
        line += " "*(len(line) - max_line_length + 1)
        max_line_length = len(line)
    sys.stdout.write(line)
    sys.stdout.flush()
