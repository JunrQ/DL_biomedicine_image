import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2


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


def extract_feature(images, is_training):
    normalized = image_preprocess(images)
    ori_shape = tf.shape(normalized)
    merge_dim = tf.reshape(normalized,
                           [-1, ori_shape[2], ori_shape[3], ori_shape[4]],
                           name='flatten_timestep')
    with slim.arg_scope([slim.batch_norm, slim.conv2d], trainable=False):
        net, _ = resnet_v2.resnet_v2_101(merge_dim, is_training=is_training)
    feature_shape = tf.shape(net)
    recover_dim = tf.reshape(net,
                             [ori_shape[0], ori_shape[1],
                                 feature_shape[1], feature_shape[2], -1],
                             name='recover_timestep')
    return recover_dim
