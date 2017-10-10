import tensorflow as tf
import tensorflow.contrib.slim as slim
from .nets import (inception_resnet_v2, resnet_v2_101, resnet_arg_scope)


def image_preprocess(image):
    """ Centralize pixel distribution
    """
    with tf.name_scope('image_preprocess'):
        image = tf.cast(image, tf.float32)
        image = (image - 0.5) * 2
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
    with slim.arg_scope(resnet_arg_scope(use_batch_norm=False)):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            conv = slim.conv2d(feature, 512, (3, 3), stride=2, scope='conv1')
            bn = slim.batch_norm(conv, scope='batch_norm1')
            conv = slim.conv2d(bn, 512, (3, 3), stride=1, scope='conv2')
            bn = slim.batch_norm(conv, scope='batch_norm2')
            conv = slim.conv2d(bn, 512, (3, 3), stride=1, scope='conv3')
            bn = slim.batch_norm(conv, scope='batch_norm3')
    avg = tf.reduce_mean(bn, [1, 2], keep_dims=False)
    
    _, F = avg.get_shape().as_list()
    recover_ts = tf.reshape(avg, [-1, T, F], name='recover_timestep')
    return recover_ts
