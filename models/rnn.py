""" LSTM RNN for set input.

    From:
    [1] Vinyals, Oriol; Bengio, Samy and Kudlur, Manjunath
        Order Matters: Sequence to sequence for sets. arXiv:1511.06391
"""

from .image_utils import extract_feature
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
        logits = slim.fully_connected(concated, 1, activation_fn=None,
                                      weights_regularizer=slim.regularizers.l2_regularizer(
                                          self.weight_decay),
                                      scope='fc_read')
        logits = tf.reshape(
            logits, shape=[-1, self._mem_size], name='recover_time_of_logits')
        mask = tf.sequence_mask(self.length, self._mem_size)
        effective_logits = logits * tf.cast(mask, logits.dtype)
        attention = tf.nn.softmax(effective_logits)
        expanded_attention = tf.tile(tf.reshape(attention, [-1, 1], name='reshape_attention'),
                                     [1, self._feature_size], name='expand_attention')
        weighted = expanded_attention * flatten_mem
        weighted = tf.reshape(
            weighted, [-1, self._mem_size, self._feature_size])
        read = tf.reduce_sum(weighted, axis=1, keep_dims=False)
        return read

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

    def __init__(self, config):
        """
        Args:
            read_time: How many times should the lstm run.
            label_num: Number of classes.
            weight_decay: l2 regularizer parameter.
        """
        self.weight_decay = config.weight_decay
        self.read_time = config.read_time
        self.label_num = config.label_num
        self.max_sequence_length = config.max_sequence_length
        self.use_glimpse = config.use_glimpse
        self.cost = None

    def _get_inputs(self):
        """ Required by the base class.
        """
        return [InputDesc(tf.float32, [None, None, 128, 320, 3], 'image'),
                InputDesc(tf.int32, [None], 'length'),
                InputDesc(tf.int32, [None, self.label_num], 'label')]

    def _build_graph(self, inputs):
        """ Required by the base class.
        """
        image, length, label = inputs
        N = tf.shape(image)[0]
        ctx = get_current_tower_context()
        feature = extract_feature(image, ctx.is_training, self.weight_decay)
        feature = self._pad_to_max_len(feature, self.max_sequence_length)

        with tf.variable_scope('rnn'):
            rnn_cell = MemCell(feature, length, self.weight_decay)
            # the content of input sequence for the lstm cell is irrelevant, but we need its length
            # information to deduce read_time
            dummy_input = [tf.zeros([N, 1])] * self.read_time
            initial_state = self._calcu_glimpse(
                feature, length) if self.use_glimpse else None
            _, final_encoding = tf.nn.static_rnn(
                rnn_cell, dummy_input, initial_state=initial_state, dtype=tf.float32, scope='process')

        logits = slim.fully_connected(final_encoding, self.label_num, activation_fn=None,
                                      weights_regularizer=slim.l2_regularizer(
                                          self.weight_decay),
                                      scope='logits')
        # gave logits a reasonable name, so one can access it easily. (e.g. via get_variable(name))
        logits = tf.identity(logits, name='logits_export')
        loss = tf.losses.sigmoid_cross_entropy(label, logits,
                                               reduction=tf.losses.Reduction.MEAN, scope='loss')
        add_moving_summary(loss)
        # export loss for easy access
        loss = tf.identity(loss, name='loss_export')
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
        lr = tf.get_variable('learning_rate', shape=(),
                             dtype=tf.float32, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.AdamOptimizer(learning_rate=lr)
