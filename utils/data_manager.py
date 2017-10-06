# -*- coding: utf-8 -*-
from csv_loader import *
from functional import seq
import numpy as np
from pathlib import Path
from seperation import SeparationScheme
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.misc import imread
from tensorpack.dataflow import (CacheData, MapData, MapDataComponent, 
                                 BatchData, ThreadedMapData, DataFlow)

# TODO: configuration interface

_IMAGE_DIR = str(Path.home()) + "/Documents/flyexpress/DL_biomedicine_image/data/pic_data/"
_MAX_SEQ_LENGTH = 10
_BATCH_SIZE = 32

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


def _encode_label(annots, vocab):
    binarizer = MultiLabelBinarizer(classes=vocab)
    return binarizer.fit_transform([annots])


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
        self.vocabulary = vocabulary
        
    def _build_basic_stream(self, data_set):
        ds = UrlDataFlow(data_set)
        # trim image sequence to max length
        ds = MapDataComponent(ds,
                              lambda urls: _cut_to_max_length(urls, _MAX_SEQ_LENGTH), 0)
        # add length info of image sequence into data points
        ds = MapData(ds, lambda dp: [dp[0], len(dp[0]), dp[1]])
        # read image multithreadedly
        ds = ThreadedMapData(
            ds, nr_thread=10,
            map_func=lambda dp: [_load_image(dp[0], _IMAGE_DIR), dp[1], dp[2]])
        # pad and stack images to Tensor(shape=[T, C, H, W])
        ds = MapDataComponent(ds, 
                              lambda imgs: _pad_input(imgs, _MAX_SEQ_LENGTH), 0)
        # one-hot encode labels
        ds = MapDataComponent(ds,
                              lambda annots: _encode_label(annots, self.vocabulary), 2)
        return ds

    def get_train_stream(self):
        ds = self._build_basic_stream(self.train_set)
        ds = BatchData(ds, batch_size=_BATCH_SIZE)
        return ds

    def get_validation_stream(self):
        ds = self._build_basic_stream(self.val_set)
        # validation set is small, so we cache it
        ds = CacheData(ds)
        ds = BatchData(ds, batch_size=_BATCH_SIZE)
        return ds

    def get_test_stream(self):
        ds = self._build_basic_stream(self.test_set)
        ds = BatchData(ds, batch_size=_BATCH_SIZE)
        return ds
