import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from config.rnn import default as default_config
from models import RNN
from utils import DataManager
from utils.validation import (Accumulator, AggerateMetric, calcu_metrics)


from functional import seq
import json
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
MAIN_LOG_LOC = "./log/"
PROGRESS_FILE = "progress.pickle"
METRICS_FILE = "metrics.json"

# name: (stage, annotation_number)
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


def run_for_dataset(config, log_dir):
    logger.set_logger_dir(log_dir, action='d')
    ignore_restore = ['learning_rate', 'global_step']
    save_name = "rnn-max-micro_auc.tfmodel"
    threshold = 0.5

    log_obj = {}
    log_obj['stages'] = config.stages
    log_obj['annotation_number'] = config.annotation_number
    print(f"Stage: {config.stages}")
    print(f"Annotation number: {config.annotation_number}")

    config.proportion = {'train': 0.55, 'val': 0.0, 'test': 0.45}
    data_manager = DataManager(config)
    log_obj['set_size'] = data_manager.get_num_info()
    train_data = data_manager.get_train_stream()
    test_data = data_manager.get_test_stream()
    model = RNN(config)

    tf.reset_default_graph()
    train_config = TrainConfig(model=model, dataflow=train_data,
                               callbacks=[
                                   ScheduledHyperParamSetter(
                                       'learning_rate', [(0, 1e-4), (18, 1e-5)]),
                                   ModelSaver(),
                                   MaxSaver('training_auc', save_name),
                               ],
                               session_init=SaverRestore(
                                   model_path=RESNET_LOC, ignore=ignore_restore),
                               max_epoch=1, nr_tower=2)
    trainer = Trainer(train_config)
    trainer.train()
    trainer.sess.close()

    tf.reset_default_graph()
    pred_config = PredictConfig(model=model,
                                session_init=SaverRestore(
                                    model_path=log_dir + save_name),
                                output_names=['logits_export', 'label'])
    predictor = Predictor(pred_config, test_data)
    accumulator = seq(predictor.get_result()) \
        .smap(lambda a, b: (a.shape[0],
                            calcu_metrics(a, b, config.validation_metrics, threshold))) \
        .aggregate(Accumulator(*config.validation_metrics),
                   lambda accu, args: accu.feed(args[0], *args[1]))
    predictor.predictor.sess.close()
    metrics = accumulator.retrive()
    
    print(f"Test metrics: {metrics}")
    log_obj['metrics'] = metrics

    return log_obj


def dummy_run_for_dataset(config, _logdir):
    from time import sleep

    log_obj = {}
    log_obj['stages'] = config.stages
    log_obj['annotation_number'] = config.annotation_number
    print(f"Stage: {config.stages}")
    print(f"Annotation number: {config.annotation_number}")

    sleep(3)
    metrics = {'macro_auc': 0.9222, "micro_f1": 0.6032}
    print(f"Test metrics: {metrics}")
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

    for set_name, (stage, annot_num) in tqdm(DATA_SETS.items()):
        # this data set has already been tested.
        if set_name in progress:
            continue

        config = default_config
        config.stages = [stage]
        config.annotation_number = annot_num
        log_dir = MAIN_LOG_LOC + set_name + '/'
        metrics = run_for_dataset(config, log_dir)

        progress.add(set_name)
        update_metrics_collection(metrics)
        with open(file_name, 'wb') as f:
            pickle.dump(progress, f)


if __name__ == "__main__":
    run()
