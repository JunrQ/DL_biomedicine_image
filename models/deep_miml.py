from .image_utils import extract_feature_vgg
from .rnn import focal_loss

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import (ModelDesc, InputDesc, get_current_tower_context)
from tensorpack.tfutils.summary import add_moving_summary

class DeepMiml(ModelDesc):
    def __init__(self, 
                 config, 
                 subconcept,
                 label_weights=None,
                 is_finetuning=False):
        self.config = config
        self.subconcept = subconcept
        self.cost = None
        self.is_finetuning = is_finetuning
        self.scale = label_weights
        
    def _get_inputs(self):
        return [InputDesc(tf.float32,
                          [None,
                           self.config.max_sequence_length,
                           128 // self.config.downsample,
                           320 // self.config.downsample, 
                           3], 
                          'image'),
                InputDesc(tf.int32, [None], 'length'),
                InputDesc(tf.int32, [None, self.config.annotation_number], 'label')]
    
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', shape=(),
                             dtype=tf.float32, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.AdamOptimizer(learning_rate=lr)
    
    def _build_graph(self, inputs):
        ctx = get_current_tower_context()
        
        image, length, label = inputs
        tf.summary.histogram('images', image)
        
        instances = extract_feature_vgg(
            image, ctx.is_training, self.config.weight_decay)
        _, I, C = instances.get_shape().as_list()
        print(f"Instance shape: {instances.get_shape().as_list()}")
        instances = tf.reshape(instances, [-1, C], name='instances')
        
        tf.summary.histogram('instances', instances)
        
        subconcept = slim.fully_connected(
            instances,
            num_outputs=self.subconcept * self.config.annotation_number,
            weights_regularizer = slim.l2_regularizer(self.config.weight_decay))
        '''
        subconcept = tf.reshape(
            subconcept, [-1, I, self.subconcept * self.config.annotation_number])
        '''
        subconcept = tf.reshape(
            subconcept,
            [-1, I, self.subconcept, self.config.annotation_number],
            name='subconcept')
        tf.summary.histogram('subconcept', subconcept)
        
        subconcept = tf.layers.dropout(
            subconcept, 
            rate=1 - self.config.dropout_keep_prob,
            training = ctx.is_training)
        
        instance_score = tf.reduce_max(subconcept, axis=1, keep_dims=False)
        label_score = tf.reduce_max(instance_score, axis=1, keep_dims=False)
        
        logits = slim.fully_connected(
            label_score,
            num_outputs=self.config.annotation_number,
            weights_regularizer = slim.l2_regularizer(self.config.weight_decay),
            activation_fn=tf.identity)
        
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
