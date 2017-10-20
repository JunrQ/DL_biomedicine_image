
import os
import tensorflow as tf

import csv

import datetime
import skimage.io
import skimage.transform
import skimage.util

import numpy as np
from urllib import request

DOWNIAMGE_PARURL = 'http://www.flyexpress.net/fx_images/BDGP/standardized/'

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
list_stopWords = list(set(stopwords.words('english')))
lemmatizer = WordNetLemmatizer()
def propress_annot(annot):
    """
    word lemmatization
    """
    annot_to_return = ""
    for ele in annot.split():
        tmp = lemmatizer.lemmatize(ele) + " "
        if tmp not in list_stopWords:
            annot_to_return += tmp
    return annot_to_return

def csvfile2iterator(csvfile_path, parent_path):
    """Given a csvfile path, return a iterator
        used for csvfile.csv in E:csvfile.csv
        e.g. AlkB1,['131902_s.bmp'],"['maternal', 'ubiquitous']"

        with element: {'gene stage': gene stage,
                   'urls': urls list,
                   'labels': label list}
    """
    with open(csvfile_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            urls = []
            gene_stage = row[0]
            labels = []
            # print(row)
            for ele in row[2][1:-1].split(','):
                # print(ele)
                labels.append(ele.split('\'')[1])

            for ele in row[1][1:-1].split():
                urls.append(os.path.join(parent_path, ele.split('\'')[1]))

            yield {'gene stage': gene_stage,
                   'urls': urls,
                   'labels': labels}



def whiten(img):
    s = np.std(img, axis=(0,))
    m = np.mean(img, axis=(0,))
    return (img - m) / s


def annot2vec(label, vocab):
    """
    根据输入的label 和对应的字典 返回ont-hot表示
    Args:
      label 是一个string，用空格分开
      vocab 是一个 list，元素为word
    """
    label_list = np.zeros(len(vocab), dtype=np.int)
    for w in label:
        try:
            idx = vocab.index(w)
            label_list[idx] = 1
        except ValueError:
            continue

    return label_list

def download_from_url(image_url, filename_path):
    print("Redownloading %s" % filename_path)
    with request.urlopen(image_url) as web:
        # 为保险起见使用二进制写文件模式，防止编码错误
        with open(filename_path, 'wb') as outfile:
            outfile.write(web.read())

def read_image_from_single_file(filename, dtype='float64', redownload=False, try_times=2):
    """
    Args:
        filename: a path of an image
    Return:
        im: numpy.ndarray, dtype: float32
    """
    flag = True
    for tmp in range(try_times):
        try:
            im = skimage.io.imread(filename)
            # print(im.shape)
            flag = True
            break
        except IOError:
            flag = False
            if redownload:
              if os.path.exists(filename):
                  os.remove(filename)
              img_url = DOWNIAMGE_PARURL + '\\' + filename.split('\\')[-1]
              download_from_url(img_url, filename)
            else:
              break
            continue

    if not flag:
        # print('[Error: %s do not exist!]' % filename)
        return -1
    else:
        im = skimage.util.img_as_float(im).astype(dtype)
        return im

def get_image_from_urls_list_concat_dim0(urls_list,
                                         ignore_diff_size=False,
                                         shape=None,
                                         dtype='float64'):
    """
    根据urls_list里面的图片的位置，返回读入后的图片
    图片使用numpy.concatenate连接
    """
    if shape is None:
        print("Wrong shape parameters, shape should not be None.")

    first = -1
    for idx in range(0, len(urls_list)):
        im = read_image_from_single_file(urls_list[idx], dtype=dtype)
        if isinstance(im, int):
            continue

        # print(im.shape)
        if im.shape != shape:
            if ignore_diff_size:
                continue
            else:
                im = skimage.transform.resize(im, shape)
                first = idx
                break
        else:
          first = idx
          break
    # print(im.shape)
    if first == len(urls_list):
        return im
    if first == -1:
        # print("All urls failed.")
        return -1

    for idx in range(first, len(urls_list)):
        temp = read_image_from_single_file(urls_list[idx], dtype=dtype)
        if isinstance(temp, int):
            continue
        if temp.shape != shape:
            if ignore_diff_size:
                continue
            else:
                temp = skimage.transform.resize(temp, shape)

        im = np.concatenate((im, temp))
    return im

def get_image_from_urls_list_concat_dimNone(urls_list,
                                            shape=None,
                                            ignore_diff_size=True):
    """
    根据urls_list里面的图片的位置，返回读入后的图片

    """
    if shape is None:
        print("Wrong shape parameters, shape should not be None.")

    first = 0
    for idx in range(0, len(urls_list)):
        im = read_image_from_single_file(urls_list[idx], dtype=dtype)
        if isinstance(im, int):
            continue
        if im.shape != shape:
            if ignore_diff_size:
                continue
            else:
                im = skimage.transform.resize(im, shape)
                first = idx
                break

    if first == len(urls_list):
        return im
    if first == 0:
        # print("All urls failed.")
        return -1

    im = im[None, :]

    for idx in range(first, len(urls_list)):
        temp = read_image_from_single_file(urls_list[idx], dtype=dtype)
        if isinstance(temp, int):
            continue
        if temp.shape != shape:
            if ignore_diff_size:
                continue
            else:
                temp = skimage.transform.resize(temp, shape)
        temp = temp[None, :]
        im = np.concatenate((im, temp))
    return im

def parse_sequence_example(serialized, image_feature, caption_feature):
    """Parses a tensorflow.SequenceExample into an image and caption.

    Args:
      serialized: A scalar string Tensor; a single serialized SequenceExample.
      image_feature: Name of SequenceExample context feature containing image
        data.
      caption_feature: Name of SequenceExample feature list containing integer
        captions.

    Returns:
      encoded_image: A scalar string Tensor containing a JPEG encoded image.
      caption: A 1-D uint64 Tensor with dynamically specified length.
    """
    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            image_feature: tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

    encoded_image = context[image_feature]
    caption = sequence[caption_feature]
    return encoded_image, caption


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
    """Batches input images and captions.

    This function splits the caption into an input sequence and a target sequence,
    where the target sequence is the input sequence right-shifted by 1. Input and
    target sequences are batched and padded up to the maximum length of sequences
    in the batch. A mask is created to distinguish real words from padding words.

    Example:
      Actual captions in the batch ('-' denotes padded character):
        [
          [ 1 2 5 4 5 ],
          [ 1 2 3 4 - ],
          [ 1 2 3 - - ],
        ]

      input_seqs:
        [
          [ 1 2 3 4 ],
          [ 1 2 3 - ],
          [ 1 2 - - ],
        ]

      target_seqs:
        [
          [ 2 3 4 5 ],
          [ 2 3 4 - ],
          [ 2 3 - - ],
        ]

      mask:
        [
          [ 1 1 1 1 ],
          [ 1 1 1 0 ],
          [ 1 1 0 0 ],
        ]

    Args:
      images_and_captions: A list of pairs [image, caption], where image is a
        Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
        any length. Each pair will be processed and added to the queue in a
        separate thread.
      batch_size: Batch size.
      queue_capacity: Queue capacity.
      add_summaries: If true, add caption length summaries.

    Returns:
      images: A Tensor of shape [batch_size, height, width, channels].
      input_seqs: An int32 Tensor of shape [batch_size, padded_length].
      target_seqs: An int32 Tensor of shape [batch_size, padded_length].
      mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
    """
    enqueue_list = []
    for image, caption in images_and_captions:
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)
        enqueue_list.append([image, input_seq, target_seq, indicator])

    images, input_seqs, target_seqs, mask = tf.train.batch_join(
        enqueue_list,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch_and_pad")

    if add_summaries:
        lengths = tf.add(tf.reduce_sum(mask, 1), 1)
        tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
        tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
        tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

    return images, input_seqs, target_seqs, mask
