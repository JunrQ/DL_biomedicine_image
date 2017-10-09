from .image_utils import extract_feature
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorpack import (ModelDesc, InputDesc, get_current_tower_context)


class MemCell(tf.contrib.rnn.RNNCell):
    def __init__(self, mem, length, batch_size, weight_decay):
        """
        Args:
            mem: Tensor of shape [N, T, F]
            length: Tensor of shape [N]
        """
        self.memory = mem
        self.length = length
        _, T, F = mem.get_shape().as_list()
        assert T is not None and F is not None, "T and F is unknown"
        self._mem_size = T
        self._feature_size = F
        self._batch_size = batch_size
        self.weight_decay = weight_decay

    def __call__(self, _inputs, state, scope=None):
        read = self._read_memory(state)
        gate_feature = tf.concat(
            [read, state], 1, name='concat_read_and_state')
        regularizer = slim.regularizers.l2_regularizer(self.weight_decay)

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.sigmoid,
                            weights_regularizer=regularizer):
            forget_gate = slim.fully_connected(
                gate_feature, self._feature_size, scope='fc_fg')
            input_gate = slim.fully_connected(
                gate_feature, self._feature_size, scope='fc_ig')
        new_state = tf.tanh(forget_gate * state + input_gate * read)
        return (), new_state

    def _read_memory(self, state):
        flatten_mem = tf.reshape(
            self.memory, [-1, self._feature_size], name='flatten_mem')
        expanded_state = tf.tile(
            state, [self._mem_size, 1], name='expand_state')
        concated = tf.concat([flatten_mem, expanded_state],
                             1, name='concat_memory_and_state')
        logits = slim.fully_connected(concated, 1, activation_fn=None,
                                      weights_regularizer=slim.regularizers.l2_regularizer(
                                          self.weight_decay),
                                      scope='fc_read')
        logits = tf.reshape(logits, shape=[-1, self._mem_size], name='recover_time_of_logits')
        mask = tf.sequence_mask(self.length, self._mem_size)
        effective_logits = logits * tf.cast(mask, logits.dtype)
        attention = tf.nn.softmax(effective_logits)
        expanded_attention = tf.tile(tf.reshape(attention, [-1, 1], name='reshape_attention'),
                                     [1, self._feature_size], name='expand_attention')
        weighted = expanded_attention * flatten_mem
        weighted = tf.reshape(weighted, [-1, self._mem_size, self._feature_size])
        read = tf.reduce_sum(weighted, axis=1, keep_dims=False)
        return read

    @property
    def state_size(self):
        return self._feature_size

    @property
    def output_size(self):
        return 0


class RNN(ModelDesc):
    def __init__(self, read_time, label_num, batch_size, weight_decay=5e-4):
        self.weight_decay = weight_decay
        self.read_time = read_time
        self.label_num = label_num
        self.batch_size = batch_size
        self.cost = None

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 10, 128, 320, 3], 'image'),
                InputDesc(tf.int32, [None], 'length'),
                InputDesc(tf.int32, [None, 40], 'label')]

    def _build_graph(self, inputs):
        image, length, label = inputs
        N = tf.shape(image)[0]
        ctx = get_current_tower_context()
        feature = extract_feature(image, ctx.is_training, self.weight_decay)
        rnn_cell = MemCell(feature, length, self.batch_size, self.weight_decay)
        dummy_input = [tf.zeros([N, 1])] * self.read_time
        _, final_encoding = tf.nn.static_rnn(
            rnn_cell, dummy_input, dtype=tf.float32, scope='process')
        logits = slim.fully_connected(final_encoding, self.label_num, activation_fn=None,
                                      weights_regularizer=slim.regularizers.l2_regularizer(
                                          self.weight_decay),
                                      scope='logits')
        loss = tf.losses.sigmoid_cross_entropy(label, logits, scope='loss')
        tf.summary.scalar('loss-summary', loss)
        self.cost = loss

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', shape=(), dtype=tf.float32, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.AdamOptimizer(learning_rate=lr)
