import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from config.rnn import default as default_config
from models import RNN
from utils import DataManager
from utils.validation import (Accumulator, AggregateMetric, calcu_metrics)

from functional import seq
import json
from multiprocessing import Process, Pipe
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorpack import (TrainConfig, SyncMultiGPUTrainerParameterServer as Trainer,
                        PredictConfig, SaverRestore, logger)
from tensorpack.predict import SimpleDatasetPredictor as Predictor
from tensorpack.callbacks import (
    ScheduledHyperParamSetter, MaxSaver, ModelSaver)
from tensorpack.tfutils.common import get_default_sess_config
from tensorpack.tfutils.sesscreate import ReuseSessionCreator

RESNET_LOC = "../data/resnet_v2_101/resnet_v2_101.ckpt"
MODEL_NAME = "max-training-ap.tfmodel"
MAIN_LOG_LOC = "./single_label/"
PROGRESS_FILE = "progress.pickle"
TEST_RESULT = "test_run.pickle"
METRICS_FILE = "metrics.json"


# name: (stage, annotation_number)
'''
DATA_SETS = {
    'D1':  (2, 10),
    'D2':  (2, 20),
    'D3':  (2, 30),
    'D4':  (3, 10),
    'D5':  (3, 20),
    'D6':  (4, 10),
    'D7':  (4, 20),
    'D8':  (5, 10),
    'D9':  (5, 20),
    'D10': (5, 30),
    'D11': (5, 40),
    'D12': (5, 50),
    'D13': (6, 10),
    'D14': (6, 20),
    'D15': (6, 30),
    'D16': (6, 40),
    'D17': (6, 50),
    'D18': (6, 60),
}
'''
DATA_SET = {
    'D1': (2, 2)
}


def run_for_label(config, train_stream, log_dir):
    def train(config, data, log_dir):
        logger.set_logger_dir(log_dir, action='d')
        ignore_restore = ['learning_rate', 'global_step', 'logits/weights',
                          'logits/biases', 'hidden_fc/weights', 'hidden_fc/biases']
        model = RNN(config, is_finetuning=False)

        tf.reset_default_graph()
        train_config = TrainConfig(model=model, dataflow=train_stream,
                                   callbacks=[
                                       ScheduledHyperParamSetter(
                                           'learning_rate', [(0, 1e-4), (15, 1e-5)]),
                                       ModelSaver(max_to_keep=5),
                                       MaxSaver('training_ap', MODEL_NAME),
                                   ],
                                   session_init=SaverRestore(
                                       model_path=RESNET_LOC, ignore=ignore_restore),
                                   max_epoch=1, tower=[0, 1])
        trainer = Trainer(train_config)
        trainer.train()

    child = Process(target=train, args=(config, train_stream, log_dir))
    child.start()
    child.join()


def combine_and_calcu_metrics(logits, labels, queries):
    logits = np.stack(logits)
    labels = np.stack(labels)
    values = calcu_metrics(logits, labels, queries, 0.4)
    return seq(queries).zip(seq(values)).dict()


def run_test(config, test_stream, log_dir):
    def test(config, data, log_dir, pipe):
        model = RNN(config, is_finetuning=False)
        tf.reset_default_graph()
        pred_config = PredictConfig(model=model,
                                    session_init=SaverRestore(
                                        model_path=log_dir + MODEL_NAME),
                                    output_names=['logits_export', 'label'])
        data.reset_state()
        predictor = Predictor(pred_config, data)
        result = list(predictor.get_result())
        pipe.send(zip(*result))

    rx, tx = Pipe(duplex=False)
    child = Process(target=test, args=(config, test_stream, log_dir, tx))
    child.start()
    child.join()
    return rx.recv()


def run_for_dataset(config, train_set, test_set, log_dir):
    log_obj = {}
    log_obj['stages'] = config.stages
    log_obj['annotation_number'] = config.annotation_number
    print(f"Stage: {config.stages}")
    print(f"Annotation number: {config.annotation_number}")

    config.proportion = {'train': 1.0, 'val': 0.0, 'test': 0.0}
    dm = DataManager.from_dataset(train_set, test_set, config)
    log_obj['data_size'] = dm.get_num_info()
    print(dm.get_num_info())

    train_all_labels = dm.get_train_set()
    test_all_labels = dm.get_test_set()

    progress = None
    test_result = None
    prog_name = log_dir + PROGRESS_FILE
    result_name = log_dir + TEST_RESULT
    if not Path(prog_name).is_file():
        progress = set()
        test_result = []
    else:
        with open(prog_name, 'rb') as f:
            progress = pickle.load(f)
        with open(result_name, 'rb') as f:
            test_result = pickle.load(f)

    vocabulary = dm.get_vocabulary()
    for index, label in enumerate(vocabulary):
        if index in progress:
            continue

        dm = DataManager.from_dataset(train_all_labels, test_all_labels,
                                      config, [label])

        label_log_dir = log_dir + f"{index}/"
        run_for_label(config, dm.get_train_stream(), label_log_dir)
        label_result = run_test(config, dm.get_test_stream(), label_log_dir)

        test_result.append(label_result)
        with open(result_name, 'wb') as f:
            pickle.dump(test_result, f)
        progress.add(index)
        with open(prog_name, 'wb') as f:
            pickle.dump(progress, f)

    logits, labels = zip(*test_result)
    metrics = combine_and_calcu_metrics(logits, labels, config.validation_metrics)
    print(metrics)
    log_obj['metrics'] = metrics

    return log_obj


def update_metrics_collection(metrics):
    collection = None
    file_name = MAIN_LOG_LOC + METRICS_FILE
    if not Path(file_name).is_file():
        collection = []
    else:
        with open(file_name, 'r') as f:
            collection = json.load(f)

    collection.append(metrics)
    with open(file_name, 'w') as f:
        json.dump(collection, f)


def run():
    progress = None
    file_name = MAIN_LOG_LOC + PROGRESS_FILE
    if not Path(file_name).is_file():
        progress = set()
    else:
        with open(file_name, 'rb') as f:
            progress = pickle.load(f)

    config = default_config
    config.use_hidden_dense = True
    config.dropout_keep_prob = 0.5
    config.weight_decay = 0.0
    config.stages = [2, 3, 4, 5, 6]
    config.proportion = {'train': 0.55, 'val': 0.0, 'test': 0.45}
    config.annotation_number = None
    dm = DataManager.from_config(config)
    train_set = dm.get_train_set()
    test_set = dm.get_test_set()

    for set_name, (stage, annot_num) in tqdm(DATA_SETS.items()):
        # this data set has already been tested.
        if set_name in progress:
            continue
            
        # config.weight_decay = 5e-4
        config.stages = [stage]
        config.annotation_number = annot_num
        log_dir = MAIN_LOG_LOC + set_name + '/'
        
        # build process
        metrics = run_for_dataset(config, train_set, test_set, log_dir)
        update_metrics_collection(metrics)

        progress.add(set_name)
        with open(file_name, 'wb') as f:
            pickle.dump(progress, f)


if __name__ == "__main__":
    run()