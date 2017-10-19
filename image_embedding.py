# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.
These model definitions were introduced in the following technical report:
  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0
More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)
@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg16_base_layer(images,
                     output_layer='conv4/conv4_3',
                     trainable=False,
                     is_training=True,
                     weight_decay=0.00004,
                     stddev=0.1,
                     dropout_keep_prob=0.5,
                     add_summaries=False,
                     scope='base'):
  """
  """
  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None
    is_training = False

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=weights_regularizer,
      trainable=trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                        biases_initializer=tf.zeros_initializer()):
      with slim.arg_scope([slim.conv2d], padding='SAME'):
        net, end_points = vgg_16(images, is_training=is_training, scope='vgg_16')
        # print(end_points)

        with tf.variable_scope("logits"):
          output = end_points['vgg_16/' + output_layer]

          # Add summaries.
          # if add_summaries:
          # for v in end_points.values():
          # tf.contrib.layers.summaries.summarize_activation(v)

  return output


def adaption_layer(inputs,
                   num_output,
                   filters=[4096, 4096],
                   kernels_size=[[5, 5], [1, 1]],
                   fc_layers_num=2,
                   is_training=True,
                   trainable=True,
                   weight_decay=0.00004,
                   stddev=0.1,
                   dropout_keep_prob=0.5,
                   add_summaries=True,
                   scope='adaption'):
  """
  The total layer: max_pooling([2,2]) --> conv2d(filters[0], kernels_size[0], stride=(2,2))
               --> ## avg_pool2d(shape=input_shape[1:3]) which make output shape
                    [batch_size, 1, channles] ##
               --> fc(filters[1], kernels_size[1]) * fc_layers_num --> conv2d(output_num)
               --> flatten

  """

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "adaption", [inputs]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer):
      # pool0
      net = slim.max_pool2d(inputs, [2, 2], scope='pool0')
      # conv1
      net = slim.conv2d(net, filters[0], kernels_size[0], stride=(2, 2), scope='conv1')
      # mean pooling pool1
      shape = net.get_shape()
      net = tf.reduce_mean(net, axis=(1, 2), keep_dims=True)

      for idx in range(fc_layers_num):
        net = slim.conv2d(net, filters[1], kernels_size[1], scope='fc' + str(idx + 1))
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout' + str(idx + 1))

      net = slim.flatten(net, scope="flatten")
  return net


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points
