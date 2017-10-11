# -*- coding: utf-8 -*-
""" Provide data via DataFlow

    This module transforms image urls and string tags into tensors.
"""
from .csv_loader import (load_annot_table, load_image_table)
from functional import seq
import numpy as np
from pathlib import Path
from scipy.misc import imread
from skimage.transform import resize
from .seperation import SeparationScheme
from sklearn.preprocessing import MultiLabelBinarizer
from tensorpack.dataflow import (BatchData, CacheData, DataFlow, MapData,
                                 MapDataComponent, ThreadedMapData)


# TODO: configuration interface

_IMAGE_DIR = str(Path.home()) + \
    "/Documents/flyexpress/DL_biomedicine_image/data/pic_data/"
_MAX_SEQ_LENGTH = 10
_IMAGE_SIZE = (128, 320)


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

    def __init__(self,
                 image_manifest, annotation_manifest):
        image_table = load_image_table(image_manifest)
        annot_table = load_annot_table(annotation_manifest)
        # TODO: Add configuration interface
        sep_scheme = SeparationScheme()
        # TODO: Add configuration interface
        separated, vocabulary = sep_scheme.separate(image_table, annot_table)
        self.train_set = separated.train
        self.val_set = separated.validation
        self.test_set = separated.test
        self.binarizer = MultiLabelBinarizer(classes=vocabulary)

    def get_train_stream(self, bs):
        """ Data stream for training.

            A stream is a generator of batches. 
            Usually it is managed by the train module in tensorpack. We 
            don not need to tough it directly.
        """
        stream = self._build_basic_stream(self.train_set)
        stream = BatchData(stream, batch_size=bs)
        return stream

    def get_validation_stream(self, bs):
        """ Data stream for validation.

            The data is cached for frequent reuse.
        """
        stream = self._build_basic_stream(self.val_set)
        # validation set is small, so we cache it
        stream = CacheData(stream)
        stream = BatchData(stream, batch_size=bs)
        return stream

    def get_test_stream(self, bs):
        """ Data stream for test.
        """
        stream = self._build_basic_stream(self.test_set)
        stream = BatchData(stream, batch_size=bs)
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
                                  lambda urls: _cut_to_max_length(urls, _MAX_SEQ_LENGTH), 0)
        # add length info of image sequence into data points
        stream = MapData(stream, lambda dp: [dp[0], len(dp[0]), dp[1]])
        # read image multithreadedly
        stream = ThreadedMapData(
            stream, nr_thread=10,
            map_func=lambda dp: [_load_image(dp[0], _IMAGE_DIR), dp[1], dp[2]],
            buffer_size=40)
        # pad and stack images to Tensor(shape=[T, C, H, W])
        stream = MapDataComponent(stream,
                                  lambda imgs: _pad_input(imgs, _MAX_SEQ_LENGTH), 0)
        # one-hot encode labels
        stream = MapDataComponent(stream, self._encode_label, 2)
        return stream


def _load_image(url_list, img_dir):
    imgs = seq(url_list) \
        .map(lambda url: img_dir + url) \
        .map(imread) \
        .map(lambda img: resize(img, _IMAGE_SIZE, mode='constant')) \
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
