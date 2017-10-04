# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorpack.dataflow.base import DataFlow

class UrlDataFlow(DataFlow):
    def get_data(self):
        pass

class DataManager(object):
    def __init__(self, image_location, image_manifest, annotation_manifest, separation_scheme):
        pass

    def get_train_epoch(self, batch_size):
        pass

    def get_validation_epoch(self, batch_size):
        pass

    def get_test_epoch(self, atch_size):
        pass
