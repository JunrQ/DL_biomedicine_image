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
from tensorpack.train import launch_train_with_config
from tensorpack.predict import SimpleDatasetPredictor as Predictor
from tensorpack.callbacks import (
    ScheduledHyperParamSetter, MaxSaver, ModelSaver, DataParallelInferenceRunner as InfRunner)
from tensorpack.tfutils.common import get_default_sess_config
from tensorpack.tfutils.sesscreate import ReuseSessionCreator

RESNET_LOC = "../data/resnet_v2_101/resnet_v2_101.ckpt"
MAIN_LOG_LOC = "./multiple_rnn_10_notshuffle/"
PROGRESS_FILE = "progress.pickle"
METRICS_FILE = "metrics.json"

# name: (stage, annotation_number)
DATA_SETS = {
    'D1': 342143214,
    'D2': 574326524,
    'D3': 465785678,
    'D4': 843274321,
    'D5': 457283457,
    'D6': 367894532,
    'D7': 895634612,
    'D8': 231647343,
    'D9': 923462325,
    'D10': 193547546,
}


def run_for_dataset(config, train_set, test_set, log_dir, pipe):
    logger.set_logger_dir(log_dir, action='d')
    ignore_restore = ['learning_rate', 'global_step', 'logits/weights', 'logits/biases', 
                      'hidden_fc/weights', 'hidden_fc/biases']
    save_name = "max-training-macro-f1.tfmodel"
    threshold = 0.4

    log_obj = {}
    log_obj['stages'] = config.stages
    log_obj['annotation_number'] = config.annotation_number
    print(f"Stage: {config.stages}")
    print(f"Annotation number: {config.annotation_number}")

    config.proportion = {'train': 0.8, 'val': 0.2, 'test': 0.0}
    dm = DataManager.from_dataset(train_set, test_set, config)
     
    log_obj['data_size'] = dm.get_num_info()
    print(dm.get_num_info())
    
    train_data = dm.get_train_stream()
    validation_data = dm.get_validation_stream(batch_size=32)
    test_data = dm.get_test_stream(batch_size=32)
    model = RNN(config, is_finetuning=True, label_weights=dm.get_imbalance_ratio().train.values)

    tf.reset_default_graph()
    train_config = TrainConfig(model=model, dataflow=train_data,
                               callbacks=[
                                   ScheduledHyperParamSetter(
                                       'learning_rate', [(0, 1e-4), (40, 1e-5)]),
                                   InfRunner(validation_data, [AggregateMetric(config.validation_metrics, threshold)],
                                         [0, 1]),
                                   ModelSaver(max_to_keep=5),
                                   MaxSaver('macro_f1', save_name),
                               ],
                               session_init=SaverRestore(
                                   model_path=RESNET_LOC, ignore=ignore_restore),
                               max_epoch=65)
    launch_train_with_config(train_config, Trainer(gpus=[0, 1]))

    tf.reset_default_graph()
    pred_config = PredictConfig(model=model,
                                session_init=SaverRestore(
                                    model_path=log_dir + save_name),
                                input_names=['image', 'length', 'label'],
                                output_names=['logits_export', 'label'])
    test_data.reset_state()
    predictor = Predictor(pred_config, test_data)
    
    accumulator = Accumulator(*config.validation_metrics)
    for a, b in predictor.get_result():
        try:
            r = calcu_metrics(a, b, config.validation_metrics, threshold)
            accumulator.feed(a.shape[0], *r)
        except:
            pass
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
    config.shuffle_group = False
    config.read_times = 5
    config.use_glimpse = True
    config.use_hidden_dense = True
    config.dropout_keep_prob = 0.5
    config.weight_decay = 0.0
    config.gamma = 0
    config.stages = [2, 3, 4, 5, 6]
    config.proportion = {'train': 0.55, 'val': 0.0, 'test': 0.45}
    config.annotation_number = 10
    config.batch_size = 10

    for set_name, seed in tqdm(DATA_SETS.items()):
        # this data set has already been tested.
        if set_name in progress:
            continue
            
        dm = DataManager.from_config(config, seed)
        train_set = dm.get_train_set()
        test_set = dm.get_test_set()
            
        # config.weight_decay = 5e-4
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