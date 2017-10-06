# -*- coding: utf-8 -*-
""" Provide data via DataFlow

    This module transforms image urls and string tags into tensors.
"""
from csv_loader import (load_annot_table, load_image_table)
from functional import seq
import numpy as np
from pathlib import Path
from scipy.misc import imread
from seperation import SeparationScheme
from sklearn.preprocessing import MultiLabelBinarizer
from tensorpack.dataflow import (BatchData, CacheData, DataFlow, MapData,
                                 MapDataComponent, ThreadedMapData)

# TODO: configuration interface

_IMAGE_DIR = str(Path.home()) + \
    "/Documents/flyexpress/DL_biomedicine_image/data/pic_data/"
_MAX_SEQ_LENGTH = 10
_BATCH_SIZE = 32


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

    def reset_state(self):
        """ Required by base class DataFlow
        """
        super(UrlDataFlow, self).reset_state()


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

    def get_train_stream(self):
        """ Data stream for training.

            A stream is a generator of batches. 
            Usually it is managed by the train module in tensorpack. We 
            don not need to tough it directly.
        """
        stream = self._build_basic_stream(self.train_set)
        stream = BatchData(stream, batch_size=_BATCH_SIZE)
        return stream

    def get_validation_stream(self):
        """ Data stream for validation.

            The data is cached for frequent reuse.
        """
        stream = self._build_basic_stream(self.val_set)
        # validation set is small, so we cache it
        stream = CacheData(stream)
        stream = BatchData(stream, batch_size=_BATCH_SIZE)
        return stream

    def get_test_stream(self):
        """ Data stream for test.
        """
        stream = self._build_basic_stream(self.test_set)
        stream = BatchData(stream, batch_size=_BATCH_SIZE)
        return stream

    def recover_label(self, encoding):
        """ Turn one-hot encoding back to string labels.

            Could be useful for demonstration.
        """
        return self.binarizer.inverse_transform(encoding)

    def _encode_label(self, labels):
        return self.binarizer.fit_transform([labels])

    def _build_basic_stream(self, data_set):
        stream = UrlDataFlow(data_set)
        # trim image sequence to max length
        stream = MapDataComponent(stream,
                                  lambda urls: _cut_to_max_length(urls, _MAX_SEQ_LENGTH), 0)
        # add length info of image sequence into data points
        stream = MapData(stream, lambda dp: [dp[0], len(dp[0]), dp[1]])
        # read image multithreadedly
        stream = ThreadedMapData(
            stream, nr_thread=10,
            map_func=lambda dp: [_load_image(dp[0], _IMAGE_DIR), dp[1], dp[2]])
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
        .map(lambda img: np.transpose(img, [2, 0, 1])) \
        .list()
    return imgs


def _cut_to_max_length(url_list, max_len):
    if len(url_list) > max_len:
        return np.random.choice(url_list, max_len)
    else:
        return url_list


def _pad_input(img_list, max_len):
    additional = max_len - len(img_list)
    tensor = np.stack(img_list, axis=0)
    paddings = [[0, additional], [0, 0], [0, 0], [0, 0]]
    padded = np.pad(tensor, paddings, mode='constant')
    return padded
