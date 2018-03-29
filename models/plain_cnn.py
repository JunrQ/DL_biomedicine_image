from .nets import (resnet_v2_101, resnet_arg_scope)
from .image_utils import image_preprocess
from .rnn import focal_loss

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import (ModelDesc, InputDesc, get_current_tower_context)
from tensorpack.tfutils.summary import add_moving_summary

class PlainCnn(ModelDesc):
    def __init__(self, config, label_weights=None, is_finetuning=False):
        self.config = config
        self.cost = None
        self.is_finetuning = is_finetuning
        self.scale = label_weights
        
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 128, 320, 3], 'image'),
                InputDesc(tf.int32, [None, self.config.annotation_number], 'label')]
    
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', shape=(),
                             dtype=tf.float32, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.AdamOptimizer(learning_rate=lr)
    
    def _build_graph(self, inputs):
        ctx = get_current_tower_context()
        image, label = inputs
        
        normalized = image_preprocess(image)
        _, H, W, C = normalized.get_shape().as_list()
        F = 512
        
        with slim.arg_scope(resnet_arg_scope()):
            with slim.arg_scope([slim.conv2d], trainable=False, weights_regularizer=None):
                with slim.arg_scope([slim.batch_norm], trainable=False):
                    _, end_points = resnet_v2_101(
                        normalized, is_training=ctx.is_training)
                    
        resnet_feature = end_points['resnet_v2_101/block3']
        resnet_feature = tf.nn.dropout(
            resnet_feature, self.config.dropout_keep_prob, name='dropout')
        
        with tf.variable_scope('custom_cnn'):
            with slim.arg_scope(resnet_arg_scope(use_batch_norm=False)):
                with slim.arg_scope([slim.conv2d], trainable=not self.is_finetuning,
                                    weights_regularizer=slim.l2_regularizer(
                                        self.config.weight_decay)):
                    with slim.arg_scope([slim.batch_norm], is_training=ctx.is_training,
                                        trainable=not self.is_finetuning):
                        conv = slim.conv2d(resnet_feature, F, (3, 3), stride=2, scope='conv1')
                        bn = slim.batch_norm(conv, scope='bn1')
                        
        feature = tf.reduce_mean(bn, [1, 2], keep_dims=False)
        
        if self.config.use_hidden_dense:
            feature = slim.fully_connected(
                feature, F, weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                scope='hidden_fc')
            
        dropout = tf.nn.dropout(
            feature, self.config.dropout_keep_prob, name='dropout')
        
        logits = slim.fully_connected(dropout, self.config.annotation_number, 
                                      activation_fn=None,
                                      weights_regularizer=slim.l2_regularizer(
                                          self.config.weight_decay),
                                      scope='logits')
        
        logits = tf.identity(logits, name='logits_export')
        loss = focal_loss(logits, label, self.config.gamma, self.scale)
        loss = tf.identity(loss, name='loss/value')
        add_moving_summary(loss)
        auc, _ = tf.metrics.auc(label, tf.sigmoid(logits), 
                                updates_collections=[tf.GraphKeys.UPDATE_OPS])
        tf.summary.scalar('training_auc', auc)
        ap, _ = tf.metrics.auc(label, tf.sigmoid(logits), curve='PR', 
                               updates_collections=[tf.GraphKeys.UPDATE_OPS])
        tf.summary.scalar('training_ap', ap)
        
        self.cost = loss