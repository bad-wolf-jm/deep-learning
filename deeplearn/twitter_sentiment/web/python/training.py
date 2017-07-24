import time
import os
import math
import json
import tensorflow as tf
import glob
from models.tf_session import tf_session
from train.summary import StreamSummary
import datetime
from train.supervisor import TrainingSupervisor


class PersistentTrainingSupervisor(TrainingSupervisor):
    def __init__(self, train_root=None):
        super(PersistentTrainingSupervisor).__init__(**kwargs)
        self.train_root = train_root
        self.train_session_root = None

        self.test_folder = os.path.join(self.train_root, 'tests')
        self.checkpoint_folder = os.path.join(self.train_root, 'checkpoints')
        for p in [self.test_folder, self.checkpoint_folder]:
            if not os.path.exists(p):
                os.makedirs(p)

    def _clean_test_folder(self):
        files = [[f, os.stat(f).st_ctime]for f in glob.glob("{root}/*.json".format(root=self.test_root))]
        files = sorted(files, key=lambda x: x[1], reverse=True)[self.test_keep or 10:]
        for f in files:
            os.unlink(f[0])

    def get_test_results(self):
        files = [[f, os.stat(f).st_ctime]for f in glob.glob("{root}/*.json".format(root=self.test_root))]
        files = sorted(files, key=lambda x: x[1], reverse=True)
        return [f[0] for f in files]

    def save_test(self, train=None, test=None):
        test_output_file = "{model_name}-test-{index}-loss:{loss:.4f}-accuracy:{accuracy:.2f}.json"
        test_output_file = test_output_file.format(model_name=type(self.model).__name__,
                                                   index=test_index, loss=result['loss'],
                                                   accuracy=100 * result['accuracy'])
        test_output_path = os.path.join(self.test_folder, test_output_file)
        output_string = json.dumps({'train': test_result_on_train, 'test': result})
        with open(test_output_path, 'w') as to_file:
            to_file.write(output_string)
        self._clean_test_folder()
