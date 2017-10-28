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

TRANSFER_LOC = "./log/all-stages-max-micro-auc.tfmodel"
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


def run_for_dataset(config, train_set, test_set, log_dir, pipe):
    logger.set_logger_dir(log_dir, action='d')
    ignore_restore = ['learning_rate', 'global_step', 'logits/weights', 'logits/biases', 
                      'hidden_fc/weights', 'hidden_fc/biases']
    save_name = "max-training-auc.tfmodel"
    threshold = 0.5

    log_obj = {}
    log_obj['stages'] = config.stages
    log_obj['annotation_number'] = config.annotation_number
    print(f"Stage: {config.stages}")
    print(f"Annotation number: {config.annotation_number}")

    config.proportion = {'train': 1.0, 'val': 0.0, 'test': 0.0}
    dm = DataManager.from_dataset(train_set, test_set, config)
     
    log_obj['data_size'] = dm.get_num_info()
    print(dm.get_num_info())
    train_data = dm.get_train_stream()
    test_data = dm.get_test_stream()
    model = RNN(config, is_finetuning=True)

    tf.reset_default_graph()
    train_config = TrainConfig(model=model, dataflow=train_data,
                               callbacks=[
                                   ScheduledHyperParamSetter(
                                       'learning_rate', [(0, 1e-4), (15, 1e-5)]),
                                   ModelSaver(),
                                   MaxSaver('training_auc', save_name),
                               ],
                               session_init=SaverRestore(
                                   model_path=TRANSFER_LOC, ignore=ignore_restore),
                               max_epoch=20, tower=[1])
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

    pipe.send(log_obj)
    pipe.close()


def dummy_run_for_dataset(config, _logdir, pipe):
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
    pipe.send(log_obj)
    pipe.close()

    
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
    config.use_hidden_dense = False
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
        receiver, sender = Pipe(duplex=False)
        child = Process(target=run_for_dataset, args=(config, train_set, test_set, log_dir, sender))
        child.start()
        child.join()
        metrics = receiver.recv()
        receiver.close()
        update_metrics_collection(metrics)
        
        progress.add(set_name)
        with open(file_name, 'wb') as f:
            pickle.dump(progress, f)


if __name__ == "__main__":
    run()
