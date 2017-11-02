""" LSTM RNN for set input.

    From:
    [1] Vinyals, Oriol; Bengio, Samy and Kudlur, Manjunath
        Order Matters: Sequence to sequence for sets. arXiv:1511.06391
"""

from .image_utils import extract_feature_resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import (ModelDesc, InputDesc, get_current_tower_context)
from tensorpack.tfutils.summary import add_moving_summary

class MemCell(tf.contrib.rnn.RNNCell):
    """ LSTM cell with memory.
    """

    def __init__(self, mem, length, weight_decay, max_sequence_length):
        """
        Args:
            mem: Feature tensor of shape [N, T, F].
            length: Tensor of shape [N]. Sequences of variant length are 
                zero padded to MAX_LENGTH. We need this additional information
                to decide the effective sequence.
            weight_decay: l2 regularizer parameter.
        """
        self.memory = mem
        self.length = length
        _, _, F = mem.get_shape().as_list()

        self._mem_size = max_sequence_length
        self._feature_size = F
        self.weight_decay = weight_decay

    def __call__(self, inputs, state, scope=None):
        """ Required by the base class. Move one time step.

        Args:
            _inputs: A Dummy inputs whose content is irrelevant. 
            state: Tensor of shape [F] (last state).
        """
        N = tf.shape(inputs)[0]
        read = self._read_memory(state)
        gate_feature = tf.concat(
            [read, state], 1, name='concat_read_and_state')

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.sigmoid,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay)):
            forget_gate = slim.fully_connected(
                gate_feature, self._feature_size, scope='fc_fg')
            input_gate = slim.fully_connected(
                gate_feature, self._feature_size, scope='fc_ig')
        new_state = forget_gate * state + input_gate * read
        return (), new_state

    def _read_memory(self, state):
        """ Gather information from cell memory with attention.
        """

        flatten_mem = tf.reshape(
            self.memory, [-1, self._feature_size], name='flatten_mem')
        expanded_state = tf.tile(
            state, [self._mem_size, 1], name='expand_state')
        # lstm with peephole
        concated = tf.concat([flatten_mem, expanded_state],
                             1, name='concat_memory_and_state')
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(self.weight_decay)):
            #dense = slim.fully_connected(concated, self._feature_size, scope='read_dense')
            logits = slim.fully_connected(concated, 1, activation_fn=None, scope='read_logits')
        # [N, T]
        logits = tf.reshape(
            logits, shape=[-1, self._mem_size], name='recover_time_of_logits')
        attention = self._calcu_attention(logits)
        expanded_attention = tf.tile(tf.reshape(attention, [-1, 1], name='reshape_attention'),
                                     [1, self._feature_size], name='expand_attention')
        weighted = expanded_attention * flatten_mem
        weighted = tf.reshape(
            weighted, [-1, self._mem_size, self._feature_size])
        read = tf.reduce_sum(weighted, axis=1, keep_dims=False)
        return read
    
    def _stable_exp(self, logits):
        sub = tf.reduce_max(logits, axis=1, keep_dims=True, name='max_logits')
        return tf.exp(logits - sub, name='exp_logits')
    
    def _calcu_attention(self, logits):
        exp_logits = self._stable_exp(logits)
        mask = tf.sequence_mask(self.length, self._mem_size)
        eff_exp_logits = exp_logits * tf.cast(mask, logits.dtype)
        divisor = tf.reduce_sum(eff_exp_logits, axis=1, keep_dims=True, name='softmax_divisor')
        att = eff_exp_logits / (divisor + 1e-10)
        att_debug = tf.reduce_sum(att, axis=1, keep_dims=False, name='att_debug')
        return att
    
    def _calcu_attention_bug(self, logits):
        """ Caution: This function is not correct, it is merely a backup.
        """
        mask = tf.sequence_mask(self.length, self._mem_size)
        eff_logits = logits * tf.cast(mask, logits.dtype)
        att = tf.nn.softmax(eff_logits)
        att_debug = att * tf.cast(mask, logits.dtype) 
        att_debug = tf.reduce_sum(att, axis=1, keep_dims=False, name='att_debug')
        return att
    
    @property
    def state_size(self):
        """ Required by the base class. 
        """
        return self._feature_size

    @property
    def output_size(self):
        return 0


class RNN(ModelDesc):
    """ RNN model for image sequence annotation.
    """

    def __init__(self, config, is_finetuning=False, label_scale=None, ):
        """
        Args:
            read_time: How many times should the lstm run.
            label_num: Number of classes.
            weight_decay: l2 regularizer parameter.
        """
        self.config = config
        self.cost = None
        self.is_finetuning = is_finetuning
        self.label_scale = label_scale

    def _get_inputs(self):
        """ Required by the base class.
        """
        return [InputDesc(tf.float32, [None, self.config.max_sequence_length, 128, 320, 3], 'image'),
                InputDesc(tf.int32, [None], 'length'),
                InputDesc(tf.int32, [None, self.config.annotation_number], 'label')]
    
    def _homo_loss(self, logits, labels, _scale):
        loss = tf.losses.sigmoid_cross_entropy(labels, logits,
                                               reduction=tf.losses.Reduction.MEAN, scope='loss')
        return loss
    
    def _focal_loss(self, logits, labels, label_scale):
        """ Focal loss. arxiv:1708:02002
        """
        gamma = self.config.gamma
        p_t = tf.sigmoid(logits)
        loss_posi = -tf.log(p_t)
        if gamma > 0:
            loss_posi *= (1 - p_t)**gamma
        p_t = 1.0 - p_t
        loss_nega = -tf.log(p_t)
        if gamma > 0:
            loss_nega *= (1 - p_t)**gamma
        mask = tf.cast(labels, tf.float32)
        loss = loss_posi * mask + loss_nega * (1 - mask)
        return tf.reduce_mean(loss, axis=[0, 1], keep_dims=False, name='loss/value')
    
    def _weighted_loss(self, logits, labels, ratio):
        """ Scale loss per-label by weights
        
        Args:
            logits: [N, C].
            labels: [N, C].
            ratio: [C].
        Return:
            loss: Reduce by weighted sum.
        """
        N = tf.shape(logits)[0]
        C = self.config.annotation_number

        ratio = tf.reshape(ratio, [1, C])
        expanded_ratio = tf.tile(ratio, [N, 1], name='expand_imbalace_ratio')
        identity = tf.cast(1 - labels, tf.float32)
        scaled = expanded_ratio * tf.cast(labels, tf.float32)
        weights = identity + scaled
        loss = tf.losses.sigmoid_cross_entropy(labels, logits, weights=weights,
                                               reduction=tf.losses.Reduction.MEAN, scope='loss')
        return loss
        
    def _build_graph(self, inputs):
        """ Required by the base class.
        """
        image, length, label = inputs
        N = tf.shape(image)[0]
        ctx = get_current_tower_context()
        feature = extract_feature_resnet(image, ctx.is_training, self.is_finetuning, self.config.weight_decay)
        dropout_keep_prob = self.config.dropout_keep_prob if ctx.is_training else 1.0

        with tf.variable_scope('rnn'):
            with slim.arg_scope([slim.fully_connected], trainable=not self.is_finetuning):
                rnn_cell = MemCell(feature, length, self.config.weight_decay, self.config.max_sequence_length)
                dropout_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, state_keep_prob=dropout_keep_prob)
                # the content of input sequence for the lstm cell is irrelevant, but we need its length
                # information to deduce read_time
                dummy_input = [tf.zeros([N, 1])] * self.config.read_time
                initial_state = self._calcu_glimpse(
                    feature, length) if self.config.use_glimpse else None
                _, final_encoding = tf.nn.static_rnn(
                    dropout_cell, dummy_input, initial_state=initial_state, dtype=tf.float32, scope='process')
                
        if self.config.use_hidden_dense:
            _, F = final_encoding.get_shape().as_list()
            final_encoding = slim.fully_connected(final_encoding, F,
                                                  weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                                  scope='hidden_fc')

        #final_encoding = tf.nn.dropout(final_encoding, self.config.drop_out_keep, name='dropout')
        logits = slim.fully_connected(final_encoding, self.config.annotation_number, activation_fn=None,
                                      weights_regularizer=slim.l2_regularizer(
                                          self.config.weight_decay),
                                      scope='logits')
        # gave logits a reasonable name, so one can access it easily. (e.g. via get_variable(name))
        logits = tf.identity(logits, name='logits_export')
        loss = self._focal_loss(logits, label, self.label_scale)
        add_moving_summary(loss)
        # export loss for easy access
        loss = tf.identity(loss, name='loss_export')
        # training metric
        auc, _ = tf.metrics.auc(label, tf.sigmoid(logits), curve='ROC', 
                                updates_collections=[tf.GraphKeys.UPDATE_OPS])
        tf.summary.scalar('training_auc', auc)
        ap, _ = tf.metrics.auc(label, tf.sigmoid(logits), curve='PR',
                               updates_collections=[tf.GraphKeys.UPDATE_OPS])
        tf.summary.scalar('training_ap', ap)
        self.cost = loss

    def _calcu_glimpse(self, feature, length):
        """ Calculate initial state for recurrent layer. 

        The initial state is calculated as an average over all images.

        Args: 
            feature: A tensor of shape [N, T, F].
            length: A tensor of shape [N].

        Return:
            glimpse: A tensor of shape [N, F].
        """
        sum = tf.reduce_sum(
            feature, axis=1, keep_dims=False, name='sum_sequence')
        F = tf.shape(sum)[-1]
        length = tf.cast(length, tf.float32, name='cast_length_to_float')
        expand = tf.tile(tf.expand_dims(length, 1), [1, F])
        return sum / expand

    def _pad_to_max_len(self, feature, max_len):
        """ Pad a image sequence to the length of the internal memory.

        Args:
            feature: Tensor of shape [N, T, F].
            max_len: Size of internal memory.

        Return:
            Tensor of shape [N, T, F] where T equals max_len.
        """
        pad_len = max_len - tf.shape(feature)[1]
        paddings = [[0, 0], [0, pad_len], [0, 0]]
        return tf.pad(feature, paddings, name='zero_pad_input')

    def _get_optimizer(self):
        """ Required by base class ModelDesc.
        """
        lr = tf.get_variable('learning_rate', shape=(),
                             dtype=tf.float32, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.AdamOptimizer(learning_rate=lr)
