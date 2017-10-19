# -*- coding: utf-8 -*-
""" Provide data via DataFlow

    This module transforms image urls and string tags into tensors.
"""
from .csv_loader import (load_annot_table, load_image_table)
from .seperation import SeparationScheme

import cv2
from functional import seq
import numpy as np
from scipy.misc import imread
from sklearn.preprocessing import MultiLabelBinarizer
from tensorpack.dataflow import (BatchData, CacheData, DataFlow, MapData,
                                 MapDataComponent, ThreadedMapData, PrefetchDataZMQ)

class UrlDataFlow(DataFlow):
    """ Entry point of data pipeline.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_data(self):
        """ Required by base class DataFlow
        """
        for _, row in self.dataframe.iterrows():
            urls = row.image_url
            annots = row.annotation
            yield [urls, annots]

    def size(self):
        """ Required by base class DataFlow
        """
        return len(self.dataframe)

    
class DataManager(object):
    """ Complete data pipeline.
    """

    def __init__(self, config):
        image_table = load_image_table(config.image_table_location)
        annot_table = load_annot_table(config.annotation_table_location)
        sep_scheme = SeparationScheme(config)
        separated, vocabulary = sep_scheme.separate(image_table, annot_table)
        self.train_set = separated.train
        self.val_set = separated.validation
        self.test_set = separated.test
        self.binarizer = MultiLabelBinarizer(classes=vocabulary)
        self.config = config

    def get_train_stream(self):
        """ Data stream for training.

            A stream is a generator of batches. 
            Usually it is managed by the train module in tensorpack. We 
            don not need to tough it directly.
        """
        stream = self._build_basic_stream(self.train_set)
        return stream

    def get_validation_stream(self):
        """ Data stream for validation.

            The data is cached for frequent reuse.
        """
        stream = self._build_basic_stream(self.val_set)
        # validation set is small, so we cache it
        stream = CacheData(stream)
        return stream

    def get_test_stream(self):
        """ Data stream for test.
        """
        stream = self._build_basic_stream(self.test_set)
        return stream

    def recover_label(self, encoding):
        """ Turn one-hot encoding back to string labels.

            Could be useful for demonstration.
        """
        return self.binarizer.inverse_transform(encoding)

    def _encode_label(self, labels):
        encoding = self.binarizer.fit_transform([labels])
        return np.squeeze(encoding)

    def _build_basic_stream(self, data_set):
        stream = UrlDataFlow(data_set)
        
        # trim image sequence to max length, also shuffle squence
        stream = MapDataComponent(stream,
                                  lambda urls: _cut_to_max_length(urls, self.config.max_sequence_length), 0)
        
        # add length info of image sequence into data points
        stream = MapData(stream, lambda dp: [dp[0], len(dp[0]), dp[1]])
        # read image multithreadedly
        
        
        stream = ThreadedMapData(
            stream, nr_thread=5,
            map_func=lambda dp: [_load_image(
                dp[0], self.config.image_directory, self.config.image_size),
                dp[1], dp[2]],
            buffer_size=100)

        # pad and stack images to Tensor(shape=[T, C, H, W])
        stream = MapDataComponent(stream,
                                  lambda imgs: _pad_input(imgs, self.config.max_sequence_length), 0) 
        # one-hot encode labels
        stream = MapDataComponent(stream, self._encode_label, 2)
        stream = BatchData(stream, self.config.batch_size)
        return stream


def _load_image(url_list, img_dir, img_size):
    imgs = seq(url_list) \
        .map(lambda url: img_dir + url) \
        .map(lambda loc: cv2.imread(loc, cv2.IMREAD_COLOR)) \
        .map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) \
        .map(lambda img: cv2.resize(img, img_size)) \
        .list()
    return imgs


def _cut_to_max_length(url_list, max_len):
    select_len = min(len(url_list), max_len)
    return np.random.choice(url_list, select_len)


def _pad_input(img_list, max_len):
    additional = max_len - len(img_list)
    tensor = np.stack(img_list, axis=0)
    paddings = [[0, additional], [0, 0], [0, 0], [0, 0]]
    padded = np.pad(tensor, paddings, mode='constant')
    return padded

