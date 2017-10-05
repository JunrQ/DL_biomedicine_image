# -*- coding: utf-8 -*-
from csv_loader import *
from functional import seq
from seperation import SeparationScheme
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorpack.dataflow.common import (MapDataComponent, BatchData)
from tensorpack.dataflow.prefetch import (ThreadedMapData, PrefetchDataZMQ)
from tensorpack.dataflow.imgaug import AugmentorList
from tensorpack.dataflow.imgaug import Resize
from tensorpack.dataflow.base import DataFlow

# TODO: configuration interface

_IMAGE_DIR = "~/Download/"
_MAX_SEQ_LENGTH = 10
_BATCH_SIZE = 32


def _load_image(url_list, img_dir):
    imgs = seq(url_list) \
        .map(lambda url: img_dir + url) \
        .map(tf.read_file) \
        .map(tf.image.decode_bmp) \
        .map(lambda img: tf.transpose(img, perm=[2, 0, 1])) \
        .list()
    return imgs


def _pad_input(img_list, max_len):
    additional = max_len - len(img_list)
    tensor = tf.stack(img_list)
    paddings = [[0, additional], [0, 0], [0, 0], [0, 0]]
    padded = tf.pad(tensor, paddings)
    return padded


def _encode_label(annots, vocab):
    binarizer = MultiLabelBinarizer(classes=vocab)
    return binarizer.fit_transform(annots)


class UrlDataFlow(DataFlow):
    """ Entry point of data pipeline.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_data(self):
        """ Required by base class DataFlow
        """
        for _, row in self.dataframe:
            urls = row.image_url
            annots = row.annotation
            yield urls, len(urls), annots

    def size(self):
        """ Required by base class DataFlow
        """
        return len(self.dataframe)


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

    def get_train_stream(self):
        ds = UrlDataFlow(self.train_set)
        ds = ThreadedMapData(ds, nr_thread=10,
                             map_func=lambda dp: (_load_image(dp[0], _IMAGE_DIR), dp[1]))
        augmentor = AugmentorList([Resize((320, 128))])
        ds = MapDataComponent(ds, augmentor.augment, 0)
        ds = MapDataComponent(
            ds, lambda imgs: _pad_input(imgs, _MAX_SEQ_LENGTH), 0)
        ds = MapDataComponent(ds,
                              lambda annots: _encode_label(annots, self.vocabulary), 2)
        ds = BatchData(ds, _BATCH_SIZE)
        return ds

    def get_validation_stream(self):
        pass

    def get_test_stream(self):
        pass
