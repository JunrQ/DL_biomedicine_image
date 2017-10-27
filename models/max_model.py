from .image_utils import extract_feature_vgg
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import (ModelDesc, InputDesc, get_current_tower_context)
from tensorpack.tfutils.summary import add_moving_summary

"""
Input:
  Parameter:
    max_img: a group with images more than max_img will be abandoned,
             otherswise, will be repeated to max_img
             Example: max_img = 5, a group of images [[img1], [img2], [img3]], will be
                      [[img1], [img2], [img3], [img1], [img2]]
    stage: gene stage
    top_k_label: only take care of most frequent labels
  shape: [max_img, 128, 320, 3]
  dtype: tf.float32
Label:
  Parameters:
    top_k_label: top_k_label is also number of classes(labels)
  shape: [1, top_k_laebls]
  dtype: tf.float32
"""

class MaxModel(ModelDesc):
  """Model.
  extract_feature will return vgg features,
  """

  def __init__(self, config):
    """Model constructor.
    Args:
    """
    self.config = config

  def _get_inputs(self):
    return [InputDesc(tf.float32, [None, self.config.max_sequence_length, 128, 320, 3], 'image'),
            InputDesc(tf.int32, [None], 'length'),
            InputDesc(tf.int32, [None, self.config.annotation_number], 'label'),
            InputDesc(tf.float32, [self.config.annotation_number], 'scale')]

  def _build_graph(self, inputs):
    """
    Args:
      inputs: list of input tensors, matching _get_inputs
    """
    images, _, labels, _ = inputs
    # lables_expand is just an repeat of labels
            # i.e. batch_size = 3, max_img = 3, num_classes = 3
            #     labels: [[1, 1, 0], [0 ,1, 0], [0, 1, 1]]
            #    labels_expand: [
            #                    [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],
            #                    [[0 ,1, 0], [0 ,1, 0], [0 ,1, 0], [0 ,1, 0]],
            #                    [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
            #                    ]
            # 1-labels_expand means negative label for every single image in a group
            # np.repeat(labels, max_img, axis=0).reshape(batch_size, max_img, -1)
    labels_expand = tf.reshape(labels, [-1, 1, self.config.annotation_number])
    labels_expand = tf.tile(labels_expand, [1, self.config.max_sequence_length, 1])

    ctx = get_current_tower_context()

    # reshape
    images = tf.reshape(images, [-1, 128, 320, 3])
    feature = extract_feature_vgg(images, ctx.is_training, self.config.weight_decay)

    # [batch_size * max_img, 128, 320, 3]
    net = feature

    # for single in tf.unstack(net, axis=0, name='unstack'):

    with tf.variable_scope("adaption", values=[net]) as scope:
      # pool0 -- 8 * 20
      # net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      # net = slim.max_pool2d(self.vgg_output, [2, 2], scope='pool0')
      # conv1
      for tmp_idx in range(len(self.config.adaption_layer_filters)):
        net = tf.layers.conv2d(net, self.config.adaption_layer_filters[tmp_idx],
                               self.config.adaption_kernels_size[tmp_idx], self.config.adaption_layer_strides[tmp_idx],
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv' + str(tmp_idx + 1))

        net = tf.layers.dropout(net, training=ctx.is_training)

      if self.config.adaption_fc_layers_num:
        if self.config.adaption_fc_layers_num != len(self.config.adaption_fc_filters):
          raise ValueError("adaption_fc_layers_num should equal len()")
        for tmp_idx in range(self.config.adaption_fc_layers_num):
          net = tf.layers.conv2d(net, self.config.adaption_fc_filters[tmp_idx], [1, 1],
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 activation=tf.nn.relu,
                                 padding='same',
                                 name='fc' + str(tmp_idx + 1))
          net = tf.layers.dropout(net, training=self.is_training)


      if self.config.plus_global_feature:
        net_reduce_dim = tf.layers.conv2d(net, self.config.net_global_dim[0], [1, 1],
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                          activation=tf.nn.relu, name='net_reduce_fc')

        # reduce area from 16*40 to 6 * 14
        net_reduce_area = tf.nn.avg_pool(net_reduce_dim, ksize=[1, 4, 4, 1], strides=[1, 3, 3, 1], padding='SAME')
        net_reduce_area = tf.contrib.layers.flatten(net_reduce_area)

        # unreshape
        # B, F = net_reduce_area.get_shape().as_list()
        # [batch_size, max_img, None]
        # net_reduce_area = tf.reshape(net_reduce_area, [self.batch_size * self.config.max_img, -1])

        '''
        W_global_1 = tf.get_variable('W_global_1',
                                     shape=[],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                     trainable=True)
        b_gloabl_1 = tf.get_variable('b_global_1',
                                     shape=[],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                     trainable=True)
        '''
        # fully connect layer, make dim to self.net_global_dim[1]
        net_global_feature = tf.layers.dense(net_reduce_area, units=self.config.net_global_dim[1],
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                             activation=tf.nn.relu)

        # max feature, net to a conv2d layers
        net_max_feature = tf.layers.conv2d(net, self.config.net_max_features_nums, [1, 1],
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                           activation=tf.nn.relu, name='net_max_feature')

        # [batch_size, max_img, net_global_dim[-1]]
        # net_max_feature = tf.reshape(net_max_feature, [self.config.batch_size, self.config.max_img, -1])

        # get the max in axis (1, 2)
        # shape [self.max_img, self.adaption_filters[-1]]
        # _, F = net_max_feature.get_shape().as_list()
        # net_max_feature = tf.reshape(net_max_feature, [-1, self.config.max_img, F])
        # [batch_size, max_img, net_max_features_nums[-1]]
        fc_o = tf.reduce_max(net_max_feature, axis=(1, 2), keep_dims=False)

        # concatenate them
        # print(net_global_feature.get_shape())
        # print(self.fc_o.get_shape())
        # [batch_size, max_img, net_global_dim[-1] + net_max_features_nums]
        fc_o = tf.concat([net_global_feature, fc_o], 1)

        adaption_output = tf.layers.dense(fc_o, self.config.annotation_number,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                                           activation=None, name='adaption_output')

        self.adaption_output = tf.reshape(adaption_output, 
                                          [self.config.batch_size, self.config.max_sequence_length,
                                           self.config.annotation_number])

        self.output = tf.reduce_max(self.adaption_output, axis=[1], keep_dims=False)
    self.output = tf.identity(self.output, name='logits_export')
    self.cost = self._loss(labels, labels_expand)

  def _loss(self, labels, labels_expand):
    output_prob = tf.sigmoid(self.output)

    adaption_prob = tf.sigmoid(self.adaption_output)
    # Note, this is adaption_prob
    # labels: [batch_size, num_classes]
    # labels_expand: [batch_size, max_img, num_classes]
    self.logits_neg = tf.where(tf.greater(adaption_prob, self.config.neg_threshold),
                               tf.subtract(tf.ones_like(adaption_prob), tf.cast(labels_expand, dtype=tf.float32)),
                               tf.zeros_like(adaption_prob))

    self.logits_pos = tf.where(tf.less(output_prob, self.config.pos_threshold),
                               tf.cast(labels, dtype=tf.float32),
                               tf.zeros_like(labels, dtype=tf.float32))

    self.cross_entropy = -(tf.reduce_sum(tf.multiply(self.logits_neg, tf.log(1. - adaption_prob + 1e-10))) +
                           self.config.loss_ratio * tf.reduce_sum(
                             tf.multiply(self.logits_pos, tf.log(output_prob + 1e-10)))
                           )

    add_moving_summary(self.cross_entropy)
    auc, _ = tf.metrics.auc(labels, output_prob, updates_collections=[tf.GraphKeys.UPDATE_OPS])
    tf.summary.scalar('training_auc', auc)    
    return self.cross_entropy

  def _get_optimizer(self):
    lr = tf.get_variable('learning_rate', shape=(),
                         dtype=tf.float32, trainable=False)
    tf.summary.scalar('learning_rate-summary', lr)
    return tf.train.AdamOptimizer(learning_rate=lr)


