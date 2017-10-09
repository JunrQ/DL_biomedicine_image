import tensorflow as tf
import tensorflow.contrib.slim as slim
from .nets import (inception_resnet_v2, resnet_v2_101, resnet_arg_scope)


def image_preprocess(image):
    """ Centralize pixel distribution
    """
    with tf.name_scope('image_preprocess'):
        if image.dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std
        return image


def extract_feature(images, is_training, weight_decay):
    normalized = image_preprocess(images)
    _, T, H, W, C = normalized.get_shape().as_list()
    merge_dim = tf.reshape(normalized, shape=[-1, H, W, C], name='flatten_timestep')
    with slim.arg_scope(resnet_arg_scope()):
        with slim.arg_scope([slim.conv2d], trainable=False, weights_regularizer=None):
            with slim.arg_scope([slim.batch_norm], trainable=False):
                _, end_points = resnet_v2_101(merge_dim, is_training=is_training)
                
    feature = end_points['resnet_v2_101/block3']
    with slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(weight_decay)):
        conv = slim.conv2d(feature, 512, (3, 3), stride=2, scope='conv1')
        conv = slim.conv2d(conv, 512, (3, 3), stride=1, scope='conv2')
        conv = slim.conv2d(conv, 512, (3, 3), stride=1, scope='conv3')
    avg = tf.reduce_mean(conv, [1, 2], keep_dims=False)
    _, F = avg.get_shape().as_list()
    recover_ts = tf.reshape(avg, [-1, T, F], name='recover_timestep')
    return recover_ts