import os

state_folder = os.path.expanduser('~/.sentiment_analysis/state')
data_folder = os.path.expanduser('~/.sentiment_analysis/datasets')
checkpoint_folder = os.path.expanduser('~/.sentiment_analysis/checkpoints')

for p in [data_folder, checkpoint_folder, state_folder]:
    if not os.path.exists(p):
        os.makedirs(p)
