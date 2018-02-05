from .image_utils import extract_feature_resnet
from .rnn import focal_loss
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import (ModelDesc, InputDesc, get_current_tower_context)
from tensorpack.tfutils.summary import add_moving_summary

class PlainLstm(ModelDesc):
    def __init__(self, config, label_weights=None, is_finetuning=False):
        self.config = config
        self.cost = None
        self.is_finetuning= is_finetuning
        self.scale = label_weights
        
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, self.config.max_sequence_length, 128, 320, 3], 'image'),
                InputDesc(tf.int32, [None], 'length'),
                InputDesc(tf.int32, [None, self.config.annotation_number], 'label')]
    
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', shape=(),
                             dtype=tf.float32, trainable=False)
        tf.summary.scalar('learning_rate-summay', lr)
        return tf.train.AdamOptimizer(learning_rate=lr)
    
    def _build_graph(self, inputs):
        ctx = get_current_tower_context()
        
        image, length, label = inputs
        feature = extract_feature_resnet(
            image, ctx.is_training, self.is_finetuning, self.config.weight_decay)
        feature = tf.identity(feature, name='feature')
        
        N = tf.shape(feature)[0]
        _, T, F = feature.get_shape().as_list()
        dropout_keep_prob = self.config.dropout_keep_prob if ctx.is_training else 1.0
        
        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.LSTMCell(F, use_peepholes=True,
                                                initializer=slim.xavier_initializer())
            dropout_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, state_keep_prob=dropout_keep_prob)
            init_state = self._get_initial_state(feature, length)
            #lift_seq_dim = tf.transpose(feature, [1, 0, 2])
            _, (final_encode, _) = tf.nn.dynamic_rnn(dropout_cell, feature, 
                                                    initial_state=init_state, sequence_length=length)
        
        with slim.arg_scope(
            [slim.fully_connected], 
            weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
            if self.config.use_hidden_dense:
                final_encode = slim.fully_connected(
                    final_encode, F, scope='hidden_fc')
                
            logits = slim.fully_connected(final_encode, self.config.annotation_number,
                                          activation_fn=None, scope='logits')
            
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
        
    def _get_initial_state(self, feature, length):
        N = tf.shape(feature)[0]
        _, _, F = feature.get_shape().as_list()

        if self.config.use_glimpse:
            lstm_init = self._calcu_glimpse(feature, length)
        else:
            zeros = tf.zeros([N, F], dtype=tf.float32)
            lstm_init = (zeros, zeros)
            
        return lstm_init
            
    def _calcu_glimpse(self, feature, length):
        _, _, F = feature.get_shape().as_list()
        T = self.config.max_sequence_length
        
        mask = tf.sequence_mask(length, T)
        mask = tf.tile(tf.reshape(mask, [-1, T, 1]), [1, 1, F])
        mask = tf.cast(mask, tf.float32)
        
        with tf.variable_scope('glimpse'):
            sum = tf.reduce_sum(
                feature * mask, axis=1, keep_dims=False, name='sum_sequence')
            length = tf.cast(length, tf.float32, name='cast_length_to_float')
            expand = tf.tile(tf.expand_dims(length, 1), [1, F])
            avg = sum / expand
            with slim.arg_scope(
                [slim.fully_connected], num_outputs=F,
                weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                c_glimpse = slim.fully_connected(avg, scope='fc_c_state_init')
                m_glimpse = slim.fully_connected(avg, scope='fc_m_state_init')
        state_tuple = tf.contrib.rnn.LSTMStateTuple(c_glimpse, m_glimpse)
        
        return state_tuple
        