from .image_utils import extract_feature_resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import (ModelDesc, InputDesc, get_current_tower_context)
from tensorpack.tfutils.summary import add_moving_summary


class MemCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, mem, length, max_length, weight_decay, dropout):
        """ 
        Args:
            mem: (feature_mem, att_mem), feature_mem: shape [N, T, F],
                att_mem: shape [N, T, H, W, C]
        """
        feature_mem, att_mem = mem
        self.feature_mem = feature_mem
        self.att_mem = att_mem
        _, _, F = feature_mem.get_shape().as_list()
        self.mem_size = max_length
        self.feature_size = F
        self.length = length
        self.weight_decay = weight_decay
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            F, use_peepholes=True, initializer=slim.xavier_initializer(), name='lstm')
        dropout_wrapped = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, input_keep_prob=dropout, state_keep_prob=dropout)
        self.lstm_cell = dropout_wrapped

    def __call__(self, inputs, state, scope=None):
        read_state, accu_att = state
        read, att = self._read_memory(state)
        _, new_read_state = self.lstm_cell(read, state)
        return (), (new_read_state, accu_att + att)

    def _read_memory(self, state):
        attention = self._calcu_attention_cnn(state)
        expanded_attention = tf.tile(tf.reshape(
            attention, [-1, 1]), [1, self.feature_size])
        flatten_mem = tf.reshape(self.feature_mem, [-1, self.feature_size])
        weighted = expanded_attention * flatten_mem
        weighted = tf.reshape(weighted, [-1, self.mem_size, self.feature_size])
        read = tf.reduce_sum(weighted, axis=1, keep_dims=False)
        return read, attention

    def _calcu_attention_cnn(self, state):
        with tf.variable_scope('att_cnn'):
            memory = self._attention_net(
                self.att_mem, self.feature_size)
        with tf.variable_scope('att_func'):
            return self._attention_func(memory, state)

    def _attention_func(self, memory, state):
        """ 
        Args: concated, shape [N x T, F]
        """
        _, F = memory.get_shape().as_list()
        flatten_mem = tf.reshape(memory, [-1, F], name='flatten_mem')
        expanded_state = tf.tile(
            state, [self.mem_size, 1], name='expand_state')
        concated = tf.concat([flatten_mem, expanded_state],
                             axis=1, name='concate_mem_and_state')
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(self.weight_decay)):
            dense = slim.fully_connected(
                concated, self.feature_size, scope='att_fc')
            logits = slim.fully_connected(
                dense, 1, activation_fn=None, scope='att_logits')
        logits = tf.reshape(
            logits, shape=[-1, self.mem_size], name='recover_mem')

        return self._normalize_attention(logits)

    def _attention_net(self, att_mem, hidden_dim):
        _, _, H, W, C = att_mem.get_shape().as_list()
        merge_dim = tf.reshape(
            att_mem, shape=[-1, H, W, C], name='flatten_mem')
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(self.weight_decay), scope='conv'):
            with slim.arg_scope([slim.batch_norm], scale=True, scope='batch_norm'):
                conv = slim.conv2d(merge_dim, hidden_dim, (1, 1), stride=2)
                bn = slim.batch_norm(conv)
                conv = slim.conv2d(bn, hidden_dim, (3, 3), stride=1)
                bn = slim.batch_norm(conv)
                conv = slim.conv2d(bn, 256, (1, 1), stride=1)
                bn = slim.batch_norm(conv)
        avg = rf.reduce_mean(bn, [1, 2], keep_dims=False)
        _, F = avg.get_shape().as_list()
        recover = tf.reshape(avg, [-1, self.mem_size, F], name='recover_mem')
        return recover

    def _normalize_attention(self, logits):
        """
        Args:
            logits: shape [N, T]
        """
        exp_logits = self._stable_exp(logits)
        mask = tf.sequence_mask(self.length, self.mem_size)
        eff_exp_logits = exp_logits * tf.cast(mask, logits.dtype)
        divisor = tf.reduce_sum(
            eff_exp_logits, axis=1, keep_dims=True, name='softmax_divisor')
        att = eff_exp_logits / (divisor + 1e-10)
        return att

    def _stable_exp(self, logits):
        sub = tf.reduce_max(logits, axis=1, keep_dims=True, name='max_logits')
        return tf.exp(logits - sub, name='exp_logits')

    @property
    def state_size(self):
        return (self.feature_size, self.mem_size)


class RNNV2(ModelDesc):
    """ RNN model for image sequence annotation.
    """

    def __init__(self, config, is_finetuning=False):
        """
        Args:
            read_time: How many times should the lstm run.
            label_num: Number of classes.
            weight_decay: l2 regularizer parameter.
        """
        self.config = config
        self.cost = None
        self.is_finetuning = is_finetuning

    def _get_inputs(self):
        """ Required by the base class.
        """
        return [InputDesc(tf.float32, [None, self.config.max_sequence_length, 128, 320, 3], 'image'),
                InputDesc(tf.int32, [None], 'length'),
                InputDesc(
                    tf.int32, [None, self.config.annotation_number], 'label'),
                InputDesc(tf.float32, [self.config.annotation_number], 'scale')]

    def _focal_loss(self, logits, labels, scale):
        """ Focal loss. arxiv:1708:02002
        """
        p_t = tf.sigmoid(logits)
        loss_posi = -(1.0 - p_t)**self.config.gamma * tf.log(p_t) * scale
        p_t = 1.0 - p_t
        loss_nega = -(1.0 - p_t)**self.config.gamma * tf.log(p_t)
        mask = tf.cast(labels, tf.float32)
        loss = loss_posi * mask + loss_nega * (1 - mask)
        return tf.reduce_mean(loss, axis=[0, 1], keep_dims=False, name='loss/value')

    def _build_graph(self, inputs):
        """ Required by the base class.
        """
        image, length, label, scale = inputs
        N = tf.shape(image)[0]
        ctx = get_current_tower_context()
        low_feature, high_feature = extract_feature_resnet_v2(
            image, ctx.is_training, self.is_finetuning, self.config.weight_decay)
        dropout_keep_prob = self.config.dropout_keep_prob if ctx.is_training else 1.0

        with tf.variable_scope('rnn'):
            with slim.arg_scope([slim.fully_connected], trainable=not self.is_finetuning):
                rnn_cell = MemCell(
                    feature, length, self.config.max_sequence_length,
                    self.config.weight_decay, dropout_keep_prob)
                # the content of input sequence for the lstm cell is irrelevant, but we need its length
                # information to deduce read_time
                dummy_input = [tf.zeros([N, 1])] * self.config.read_time
                initial_state = self._get_initial_state(high_feature, length)
                _, final_encoding = tf.nn.static_rnn(
                    rnn_cell, dummy_input, initial_state=initial_state, dtype=tf.float32, scope='process')

        logits = slim.fully_connected(
            final_encoding, self.config.annotation_number, activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
            scope='logits')
        # gave logits a reasonable name, so one can access it easily. (e.g. via get_variable(name))
        logits = tf.identity(logits, name='logits_export')
        loss = self._focal_loss(logits, label, scale)
        add_moving_summary(loss)
        # export loss for easy access
        loss = tf.identity(loss, name='loss_export')
        # training metric
        auc, _ = tf.metrics.auc(label, tf.sigmoid(
            logits), updates_collections=[tf.GraphKeys.UPDATE_OPS])
        tf.summary.scalar('training_auc', auc)
        self.cost = loss

    def _get_initial_state(self, feature, length):
        N = tf.shape(feature)[0]
        F = tf.shape(feature)[1]

        if self.config.use_glimpse:
            context_init = self._calcu_glimpse(feature, length)
        else:
            context_init = tf.zeros([N, F], dtype=tf.float32)
        accu_init = tf.zeros(
            [N, self.config.max_sequence_length], dtype=tf.float32)
        return context_init, accu_init

    def _calcu_glimpse(self, feature, length):
        """ Calculate initial state for recurrent layer. 

        The initial state is calculated as an average over all images.

        Args: 
            feature: A tensor of shape [N, T, F].
            length: A tensor of shape [N].

        Return:
            glimpse: A tensor of shape [N, F].
        """
        with tf.variable_scope('glimpse'):
            sum = tf.reduce_sum(
                feature, axis=1, keep_dims=False, name='sum_sequence')
            F = tf.shape(sum)[-1]
            length = tf.cast(length, tf.float32, name='cast_length_to_float')
            expand = tf.tile(tf.expand_dims(length, 1), [1, F])
            avg = sum / expand
            with slim.arg_scope([slim.fully_connected], num_outputs=F
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                glimpse = slim.repeat(avg, 2, slim.fully_connected, scope='fc')
        return glimpse

    def _get_optimizer(self):
        """ Required by base class ModelDesc.
        """
        lr = tf.get_variable('learning_rate', shape=(),
                             dtype=tf.float32, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.AdamOptimizer(learning_rate=lr)
