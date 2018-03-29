# -*- coding: utf-8 -*-
""" Provide data via DataFlow

    This module transforms image urls and string tags into tensors.
"""
import cv2
from functional import seq
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorpack.dataflow import (BatchData, CacheData, DataFlow, MapData,
                                 MapDataComponent, ThreadedMapData, LocallyShuffleData)

from .csv_loader import load_annot_table, load_image_table
from .filter import filter_stages_and_directions, filter_labels
from .seperation import separate


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
    
    
class UrlDataFlowSi(DataFlow):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def get_data(self):
        for _, row in self.dataframe.iterrows():
            urls = row.image_url
            annots = row.annotation
            for url in urls:
                yield [url, annots]
                
    def size(self):
        return seq(self.dataframe.image_url.values).map(len).sum()

class DataManager(object):
    """ Complete data pipeline.
    """

    def __init__(self, train_set, val_set, test_set, vocab, config):
        vocab.sort()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.binarizer = MultiLabelBinarizer(classes=vocab)
        self.config = config
        self.overrided = False
        self.image_features = None

    @classmethod
    def from_config(cls, config):
        """ Construct DataManager solely from config.

        The image table and annotation table are obtained from the csv files specified in config.
        """
        image_table = load_image_table(config.image_table_location)
        annot_table = load_annot_table(config.annotation_table_location)
        image_table, annot_table = filter_stages_and_directions(
            image_table, annot_table, config.stages, config.directions)
        vocab = _extract_top_vocab(
            annot_table.annotation, config.annotation_number)
        image_table, annot_table = filter_labels(
            image_table, annot_table, vocab)
        sep = separate(image_table, annot_table, config)
        return cls(sep.train, sep.validation, sep.test, vocab, config)

    @classmethod
    def from_dataset(cls, train_set, test_set, config, vocab=None):
        """ Construct DataManager from train set and test set.

        The train set can be further subdivided into a smaller train set and validation set.
        The proportion of the subdivision is specified by the `proportion` field in `config`.
        """
        assert config.proportion['test'] == 0, \
            "Subdivision will not perform on test set, please make sure test proportion is zero"

        train_imgs = pd.DataFrame(train_set.image_url)
        train_annots = pd.DataFrame(train_set.annotation)
        train_imgs, train_annots = filter_stages_and_directions(
            train_imgs, train_annots, config.stages, config.directions)

        test_imgs = pd.DataFrame(test_set.image_url)
        test_annots = pd.DataFrame(test_set.annotation)
        test_imgs, test_annots = filter_stages_and_directions(
            test_imgs, test_annots, config.stages, config.directions)

        if vocab is None:
            vocab = _extract_common_top_vocab(
                train_annots.annotation, test_annots.annotation,
                config.annotation_number)

        train_imgs, train_annots = filter_labels(
            train_imgs, train_annots, vocab)
        test_imgs, test_annots = filter_labels(test_imgs, test_annots, vocab)

        train_sep = separate(train_imgs, train_annots, config)
        config.proportion = {'train': 0.0, 'val': 0.0, 'test': 1.0}
        test_sep = separate(test_imgs, test_annots, config)

        return cls(train_sep.train, train_sep.validation, test_sep.test, vocab, config)
    
    def override_feature(self, feature):
        self.overrided = True
        self.image_features = feature
    
    def get_vocabulary(self):
        return self.binarizer.classes

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
    
    def get_train_stream_si(self):
        """ Single instance train stream
        """
        stream = self._build_basic_stream_si(self.train_set)
        return stream
    
    def get_validation_stream_si(self):
        """ Sinle instance train stream
        """
        stream = self._build_basic_stream_si(self.val_set)
        return stream
    
    def get_test_stream_si(self):
        """ Single instance test stream
        """
        stream = self._build_basic_stream_si(self.test_set)
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
                'test': (len(self.test_set), img_nums[2])}

    def recover_label(self, encoding):
        """ Turn one-hot encoding back to string labels.

            Could be useful for demonstration.
        """
        return self.binarizer.inverse_transform(encoding)

    def _imbalance_ratio(self, data_set):
        binary_annot = self.encode_labels(data_set.annotation)
        binary_annot = np.array(binary_annot)
        posi_ratio = np.sum(binary_annot, axis=0) / binary_annot.shape[0]
        return (1 - posi_ratio) / posi_ratio
    
    def _build_basic_stream_si(self, data_set):
        assert not self.overrided, "single instance stream must be built from image"
        
        data_set = data_set.copy(deep=True)
        data_set.annotation = self.encode_labels(data_set.annotation)
        stream = UrlDataFlowSi(data_set)
        
        stream = MapDataComponent(stream,
                                  lambda url: load_image_si(url, self.config.image_directory,
                                                             self.config.image_size), 0)
        stream = ThreadedMapData(stream, nr_thread=10, map_func=lambda i: i, buffer_size=1000)
        stream = LocallyShuffleData(stream, 1000)
        stream = BatchData(stream, self.config.batch_size, remainder=True)
        
        return stream
        
    
    def _build_basic_stream(self, data_set):
        if self.overrided:
            return self._stream_from_feature(data_set)
        else:
            return self._stream_from_url(data_set)

    def _stream_from_feature(self, data_set):
        data_set = data_set.copy(deep=True)
        data_set.annotation = self.encode_labels(data_set.annotation)
        stream = UrlDataFlow(data_set)

        # trim image sequence to max length, also shuffle squence
        max_len = self.config.max_sequence_length
        stream = MapDataComponent(stream,
                                  lambda urls: cut_to_max_length(urls, max_len), 0)

        # add length info of image sequence into data points
        stream = MapData(stream, lambda dp: [dp[0], len(dp[0]), dp[1]])
        # read image multithreadedly
        stream = MapDataComponent(stream,
                                  lambda urls: seq(urls).map(lambda url: self.image_features[url]).list(), 0)
        # pad and stack images to Tensor(shape=[T, C, H, W])
        stream = MapDataComponent(stream,
                                  lambda imgs: pad_feature_input(imgs, self.config.max_sequence_length), 0)
        stream = BatchData(stream, self.config.batch_size, remainder=True)
        return stream   
        
    def _stream_from_url(self, data_set):
        data_set = data_set.copy(deep=True)
        data_set.annotation = self.encode_labels(data_set.annotation)
        stream = UrlDataFlow(data_set)

        # trim image sequence to max length, also shuffle squence
        max_len = self.config.max_sequence_length
        stream = MapDataComponent(stream,
                                  lambda urls: cut_to_max_length(urls, max_len), 0)

        # add length info of image sequence into data points
        stream = MapData(stream, lambda dp: [dp[0], len(dp[0]), dp[1]])
        # read image multithreadedly

        stream = ThreadedMapData(
            stream, nr_thread=5,
            map_func=lambda dp: [load_image(
                dp[0], self.config.image_directory, self.config.image_size),
                dp[1], dp[2]],
            buffer_size=100)

        # pad and stack images to Tensor(shape=[T, C, H, W])
        stream = MapDataComponent(stream,
                                  lambda imgs: pad_image_input(imgs, self.config.max_sequence_length), 0)
        stream = BatchData(stream, self.config.batch_size, remainder=True)
        return stream

    def encode_labels(self, labels):
        mat = self.binarizer.fit_transform(labels)
        return list(iter(mat))

    
def select_label(ds, index):
    stream = MapDataComponent(ds,
                              lambda lbs: lbs[:, index:(index+1)], 2)
    return stream

def _extract_top_vocab(annots, num):
    all_words = pd.Series(seq(annots).flat_map(lambda l: l).list())
    if num is None:
        vocab = all_words.unique()
    else:
        vocab = all_words.value_counts().nlargest(num).index.values
    return vocab


def _extract_common_top_vocab(annots_one, annots_two, num):
    try_num = num
    while True:
        vocab_one = set(_extract_top_vocab(annots_one, try_num))
        vocab_two = set(_extract_top_vocab(annots_two, try_num))
        common = vocab_one.intersection(vocab_two)
        if len(common) >= num:
            return seq(common).take(num).list()
        try_num += 1
        if try_num - num >= 5:
            raise ValueError(f"After {try_num - num} times operation "
                             "still can not find enough common labels.")


def load_image(url_list, img_dir, img_size):
    imgs = seq(url_list) \
        .map(lambda url: load_image_si(url, img_dir, img_size)) \
        .list()
    return imgs
                                  
def load_image_si(url, img_dir, img_size):
    img = cv2.imread(img_dir + url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img


def cut_to_max_length(url_list, max_len):
    select_len = min(len(url_list), max_len)
    return np.random.choice(url_list, select_len)


def pad_feature_input(feature_list, max_len):
    additional = max_len - len(feature_list)
    tensor = np.stack(feature_list, axis=0)
    paddings = [[0, additional], [0, 0]]
    padded = np.pad(tensor, paddings, mode='constant')
    return padded

def pad_image_input(img_list, max_len):
    additional = max_len - len(img_list)
    tensor = np.stack(img_list, axis=0)
    paddings = [[0, additional], [0, 0], [0, 0], [0, 0]]
    padded = np.pad(tensor, paddings, mode='constant')
    return padded
