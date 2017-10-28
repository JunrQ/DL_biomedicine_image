# -*- coding: utf-8 -*-
""" Provide data via DataFlow

    This module transforms image urls and string tags into tensors.
"""
from .csv_loader import (load_annot_table, load_image_table)
from .seperation import SeparationScheme

import cv2
from functional import seq
import numpy as np
import pandas as pd
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
    
    def __init__(self, train_set, val_set, test_set, vocab, config):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.binarizer = MultiLabelBinarizer(classes=vocab)
        self.config = config

    @classmethod
    def from_config(cls, config):
        """ Construct DataManager solely from config.

        The image table and annotation table are obtained from the csv files specified in config.
        """
        image_table = load_image_table(config.image_table_location)
        annot_table = load_annot_table(config.annotation_table_location)
        vocab = _extract_top_vocab(annot_table.annotation, config.annotation_number)
        scheme = SeparationScheme(config, vocab)
        sep = scheme.separate(image_table, annot_table)
        return cls(sep.train, sep.validation, sep.test, vocab, config)

    @classmethod
    def from_dataset(cls, train_set, test_set, config):
        """ Construct DataManager from train set and test set.
        
        The train set can be further subdivided into a smaller train set and validation set.
        The proportion of the subdivision is specified by the `proportion` field in `config`.
        """
        vocab = _extract_common_top_vocab(train_set.annotation, test_set.annotation, 
                                          config.annotation_number)
        assert config.proportion['test'] == 0, \
            "Subdivision will not perform on test set, please make sure test proportion is zero"
            
        scheme = SeparationScheme(config, vocab)
        image_table = pd.DataFrame(train_set.image_url)
        annot_table = pd.DataFrame(train_set.annotation)
        train_sep = scheme.separate(image_table, annot_table)
        
        config.proportion = {'train': 0.0, 'val': 0.0, 'test': 1.0}
        scheme = SeparationScheme(config, vocab)
        image_table = pd.DataFrame(test_set.image_url)
        annot_table = pd.DataFrame(test_set.annotation)
        # still need to filter test set
        test_sep = scheme.separate(image_table, annot_table)
        
        return cls(train_sep.train, train_sep.validation, test_sep.test, vocab, config)

    def get_train_set(self):
        """ Get train set as pandas DataFrame
        """
        return self.train_set

    def get_validation_set(self):
        """ Get validation set as pandas DataFrame
        """
        return self.val_set

    def get_test_set(self):
        """ Get test set as pandas DataFrame
        """
        return self.test_set

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
        # stream = CacheData(stream)
        return stream

    def get_test_stream(self):
        """ Data stream for test.
        """
        stream = self._build_basic_stream(self.test_set)
        return stream

    def get_imbalance_ratio(self):
        """ Get the ratio of imbalance of each label.
        """
        df = pd.DataFrame(index=self.binarizer.classes)
        df['train'] = self._imbalance_ratio(self.train_set)
        df['val'] = self._imbalance_ratio(self.val_set)
        df['test'] = self._imbalance_ratio(self.test_set)
        return df

    def get_num_info(self):
        img_nums = seq((self.train_set, self.val_set, self.test_set)) \
            .map(lambda s: seq(s.image_url).flat_map(lambda l: l).list()) \
            .map(len) \
            .list()
            
        return {'train': (len(self.train_set), img_nums[0]), 
                'val': (len(self.val_set), img_nums[1]),
                'test': (len(self.test_set), img_nums[2]) }

    def recover_label(self, encoding):
        """ Turn one-hot encoding back to string labels.

            Could be useful for demonstration.
        """
        return self.binarizer.inverse_transform(encoding)

    def _imbalance_ratio(self, data_set):
        binary_annot = self._encode_labels(data_set.annotation)
        binary_annot = np.array(binary_annot)
        posi_ratio = np.sum(binary_annot, axis=0) / binary_annot.shape[0]
        return (1 - posi_ratio) / posi_ratio

    def _build_basic_stream(self, data_set):
        imbalance_ratio = self._imbalance_ratio(data_set)
        data_set = data_set.copy(deep=True)
        data_set.annotation = self._encode_labels(data_set.annotation)
        stream = UrlDataFlow(data_set)

        # trim image sequence to max length, also shuffle squence
        max_len = self.config.max_sequence_length
        stream = MapDataComponent(stream,
                                  lambda urls: _cut_to_max_length(urls, max_len), 0)

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
        stream = BatchData(stream, self.config.batch_size)

        stream = MapData(stream, lambda dp: dp + [imbalance_ratio])
        return stream

    def _encode_labels(self, labels):
        mat = self.binarizer.fit_transform(labels)
        return list(iter(mat))

    
def _extract_top_vocab(annots, num):
    all_words = pd.Series(seq(annots).flat_map(lambda l: l).list())
    if num is None:
        vocab = all_words.unique()
    else:
        vocab = all_words.value_counts().nlargest(num).index.values
    return vocab


def _extract_common_top_vocab(annots_one, annots_two, num):
    fuse = 0
    while True:
        vocab_one = set(_extract_top_vocab(annots_one, num))
        vocab_two = set(_extract_top_vocab(annots_two, num))
        common = vocab_one.intersection(vocab_two)
        if len(common) >= num:
            return seq(common).take(num).list()
        num += 1
        fuse += 1
        if fuse >= 5:
            raise ValueError(f"After {fuse} times operation "
                             "still can not find enough common labels.")
        

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
