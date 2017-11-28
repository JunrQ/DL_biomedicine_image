from .image_utils import extract_feature_resnet, extract_feature_resnet_v2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import (ModelDesc, InputDesc, get_current_tower_context)
from tensorpack.tfutils.summary import add_moving_summary


class MemCell(tf.contrib.rnn.RNNCell):
    def __init__(self, mem, length, max_length, weight_decay, dropout, 
                 doubly_stochastic_lambda):
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
        self.doubly_stochastic_lambda = doubly_stochastic_lambda
        lstm_cell = tf.contrib.rnn.LSTMCell(
            F, use_peepholes=True, initializer=slim.xavier_initializer())
        dropout_wrapped = tf.contrib.rnn.DropoutWrapper(
            lstm_cell, input_keep_prob=dropout, state_keep_prob=dropout)
        self.lstm_cell = dropout_wrapped

    def __call__(self, inputs, state, scope=None):
        lstm_state, accu_att = state
        read_state, _ = lstm_state
        read, att = self._read_memory(read_state)
        _, new_lstm_state = self.lstm_cell(read, lstm_state)
        return (), (new_lstm_state, accu_att + att)

    def _read_memory(self, state):
        attention = self._calcu_attention(state)
        attention = tf.identity(attention, name='attention')
        expanded_attention = tf.tile(tf.reshape(
            attention, [-1, 1]), [1, self.feature_size])
        flatten_mem = tf.reshape(self.feature_mem, [-1, self.feature_size])
        weighted = expanded_attention * flatten_mem
        weighted = tf.reshape(weighted, [-1, self.mem_size, self.feature_size])
        read = tf.reduce_sum(weighted, axis=1, keep_dims=False)
        return read, attention
    
    def _calcu_attention(self, state):
        memory = self.feature_mem
        with tf.variable_scope('att_func'):
            return self._attention_func(memory, state)

    def _calcu_attention_cnn(self, state):
        with tf.variable_scope('att_cnn'):
            memory = self._attention_conv_net(
                self.att_mem, self.feature_size)
        with tf.variable_scope('att_func'):
            return self._attention_func(memory, state)

    def _attention_func(self, memory, state):
        """ 
        Args: concated, shape [N x T, F]
        """
        _, _, F = memory.get_shape().as_list()
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

        att = self._normalize_attention(logits)
        return self._dim_attention(att, state)
    
    def _dim_attention(self, att, state):
        if self.doubly_stochastic_lambda == 0:
            return att
        
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(self.weight_decay)):
            gate = slim.fully_connected(
                state, 1, scope='gate_att', activation_fn=tf.sigmoid)
        return gate * att

    def _attention_conv_net(self, att_mem, hidden_dim):
        _, _, H, W, C = att_mem.get_shape().as_list()
        merge_dim = tf.reshape(
            att_mem, shape=[-1, H, W, C], name='flatten_mem')
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(self.weight_decay)):
            with slim.arg_scope([slim.batch_norm], scale=True):
                conv = slim.conv2d(merge_dim, hidden_dim, (1, 1), stride=2)
                bn = slim.batch_norm(conv)
                conv = slim.conv2d(bn, hidden_dim, (3, 3), stride=1)
                bn = slim.batch_norm(conv)
                conv = slim.conv2d(bn, 256, (1, 1), stride=1)
                bn = slim.batch_norm(conv)
        avg = tf.reduce_mean(bn, [1, 2], keep_dims=False)
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
    
    @property
    def output_size(self):
        return 0


class RNNV2(ModelDesc):
    """ RNN model for image sequence annotation.
    """

    def __init__(self, config, is_finetuning=False, label_scale=None):
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
                InputDesc(
                    tf.int32, [None, self.config.annotation_number], 'label')]
    
    def _homo_loss(self, logits, labels, _scale):
        loss = tf.losses.sigmoid_cross_entropy(
            labels, logits, reduction=tf.losses.Reduction.MEAN, scope='loss')
        return loss

    def _focal_loss(self, logits, labels, scale):
        """ Focal loss. arxiv:1708:02002
        """
        gamma = self.config.gamma
        N = tf.shape(logits)[0]
        p_t = tf.sigmoid(logits)
        loss_posi = -tf.log(p_t) * scale
        if gamma >= 0:
            loss_posi *= (1 - p_t)**gamma
        p_t = 1.0 - p_t
        loss_nega = -tf.log(p_t)
        if gamma >= 0:
            loss_nega *= (1 - p_t)**gamma
        mask = tf.cast(labels, tf.float32)
        combine = loss_posi * mask + loss_nega * (1 - mask)
        reduce = tf.reduce_num(combine, axis=[0, 1], keep_dims=False)
        loss = reduce / combine
        return loss
    
    def _doubly_stochastic_att_loss(self, accu_att, length):
        """ Doubly stochastic attention.
        Show, Attend, and Tell.
        """
        # [N, T]
        mask = tf.sequence_mask(length, self.config.max_sequence_length)
        penalty = ((1 - accu_att) * tf.cast(mask, dtype=tf.float32))**2
        label_mean = tf.reduce_mean(penalty, axis=[0, 1], keep_dims=False)
        loss = self.config.doubly_stochastic_lambda * label_mean
        return loss

    def _build_graph(self, inputs):
        """ Required by the base class.
        """
        image, length, label = inputs
        N = tf.shape(image)[0]
        ctx = get_current_tower_context()
        low_feature, high_feature = extract_feature_resnet_v2(
            image, ctx.is_training, self.is_finetuning, self.config.weight_decay)
        dropout_keep_prob = self.config.dropout_keep_prob if ctx.is_training else 1.0

        with tf.variable_scope('rnn'):
            with slim.arg_scope([slim.fully_connected, slim.conv2d], 
                                trainable=not self.is_finetuning):
                rnn_cell = MemCell(
                    (high_feature, low_feature), length, 
                    self.config.max_sequence_length,
                    self.config.weight_decay, dropout_keep_prob, 
                    self.config.doubly_stochastic_lambda
                )
                # the content of input sequence for the lstm cell is irrelevant, but we need its length
                # information to deduce read_time
        dummy_input = [tf.zeros([N, 1])] * self.config.read_time
        initial_state = self._get_initial_state(high_feature, length)
        _, ((_, final_encoding), accu_att) = tf.nn.static_rnn(
            rnn_cell, dummy_input, initial_state=initial_state, 
            dtype=tf.float32)

        logits = slim.fully_connected(
            final_encoding, self.config.annotation_number, activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
            scope='logits')
        # gave logits a reasonable name, so one can access it easily. (e.g. via get_variable(name))
        logits = tf.identity(logits, name='logits_export')
        loss = self._homo_loss(logits, label, self.label_scale)
        loss += self._doubly_stochastic_att_loss(accu_att, length)
        loss = tf.identity(loss, name='loss/value')
        add_moving_summary(loss)
        # export loss for easy access
        # training metric
        auc, _ = tf.metrics.auc(label, tf.sigmoid(
            logits), curve='ROC', updates_collections=[tf.GraphKeys.UPDATE_OPS])
        tf.summary.scalar('training_auc', auc)
        ap, _ = tf.metrics.auc(label, tf.sigmoid(
            logits), curve='PR', updates_collections=[tf.GraphKeys.UPDATE_OPS])
        tf.summary.scalar('train_ap', ap)
        self.cost = loss

    def _get_initial_state(self, feature, length):
        N = tf.shape(feature)[0]
        _, _, F = feature.get_shape().as_list()

        if self.config.use_glimpse:
            lstm_init = self._calcu_glimpse(feature, length)
        else:
            zeros = tf.zeros([N, F], dtype=tf.float32)
            lstm_init = (zeros, zeros)
        accu_init = tf.zeros(
            [N, self.config.max_sequence_length], dtype=tf.float32)
        return lstm_init, accu_init

    def _calcu_glimpse(self, feature, length):
        """ Calculate initial state for recurrent layer. 

        The initial state is calculated as an average over all images.

        Args: 
            feature: A tensor of shape [N, T, F].
            length: A tensor of shape [N].

        Return:
            glimpse: A tensor of shape [N, F].
        """
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
        return (c_glimpse, m_glimpse)

    def _get_optimizer(self):
        """ Required by base class ModelDesc.
        """
        lr = tf.get_variable('learning_rate', shape=(),
                             dtype=tf.float32, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.AdamOptimizer(learning_rate=lr)
