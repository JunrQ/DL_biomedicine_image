"""
Build the model.
"""

from image_embedding import vgg16_base_layer, adaption_layer
import image_processing
import ops

import os
import pickle
import numpy as np
import skimage.io
import tensorflow as tf
slim = tf.contrib.slim

from file_path import *

DATASET_ITERATOR = ops.csvfile2iterator(csvfile_path=CSVFILE_PATH, parent_path=IMAGE_PARENT_PATH)


class Model(object):
  """
  """

  def __init__(self,
               ckpt_path,
               gpu=0,
               model_ckpt_path=MODEL_CKPT_PATH,
               image_foramt='bmp',
               first_run=True,
               predict_way='fc',
               adaption_output_dim=1024,
               mode="train",
               vgg_trainable=False,
               vgg_output_layer='conv4/conv4_3',
               adaption_layer_filters=[4096, 4096],
               adaption_kernels_size=[[5, 5], [3, 3]],
               adaption_layer_strides=[(2, 2), (1, 1)],
               adaption_fc_layers_num=1,
               adaption_fc_filters=[1024],
               rnn_state_dim=256,
               height=128,
               width=320,
               channels=3,
               image_concatnate_way='col',# below params used for filter input
               top_k_labels=None,
               concatenate_input=True,
               deprecated_word=None,
               stage_allowed=[5, 6],
               min_annot_num=40,
               max_img=15,
               annot_min_per_group=0,
               only_word=None,
               num_preprocess_threads=4,
               batch_size=5,
               weight_decay=0.00004,
               cnn_input_length=2048,
               valid_ratio=0.2,
               split_labels=False,
               neg_threshold=0.2,
               pos_threshold=0.9,
               loss_ratio=10,
               save_way='equal_batch'
               ):
    """
    Args:
      first_run: if first_run, original image and label need to convert to pkl file,
                 with multi images in a group concatnate through cols. Just make it True.
                 Note:
                  **If you change the parameters, you should delete the RAW_DATASET_PATH file**
      image_foramt: Used when saving concatenated images
      predict_way:
          'fc': avgpool2d, if images are concatenate(concatenate_input=True), a group images for labels
                else(concatenate_input=False), a single image for labels
                besides, after pool layer, there a self.adaption_fc_layers_num fc
          'cnn':
          'batch_max': if concatenate_input == False, then do max pooling in batch after in dim(1, 2)
                        else(concatenate_input == True), then do max pooling in dim(1, 2)
                        the lase layer is maxpooling layer
      image_concatnate_way: Original should be concatnate through columns
      tok_k_labels: if not None, e.g. 5, then top 5 labels are concerned
      concatenate_input: if True, input are concatenate, else, batch
      deprecated_word=None,
      stage_allowed=[5, 6],
      max_img=
      concatenate_input: if True, the input(a group: gene stage) for the model is concatenate,
                        and batch is composed of those concatenated images, so batch size can be
                        fixed to a number like 5.
                        if False, the input(a group: gene stage) for the model is treated as a batch
                        and batch is composed of those single images and same labels, so batch size is
                        not fixed, and depends on number of images in a gene stage group.

      valid_ratio: divide the total into train, valid. #valid = #total dataset * valid_ratio

      split_labels: if True, label is split.
    """
    self.ckpt_path = ckpt_path
    self.model_ckpt_path = model_ckpt_path
    self.mode = mode
    self.height = height
    self.width = width
    self.channels = channels
    self.vgg_output_layer = vgg_output_layer
    self.adaption_layer_filters = adaption_layer_filters
    self.adaption_kernels_size = adaption_kernels_size
    self.adaption_layer_strides = adaption_layer_strides
    self.adaption_fc_layers_num = adaption_fc_layers_num
    self.adaption_output_dim = adaption_output_dim
    self.vgg_trainable = vgg_trainable
    self.image_concatnate_way = image_concatnate_way
    self.first_run = first_run
    self.image_foramt = image_foramt
    self.predict_way = predict_way
    self.top_k_labels = top_k_labels
    self.concatenate_input = concatenate_input
    self.deprecated_word = deprecated_word
    self.stage_allowed = stage_allowed
    self.max_img = max_img
    self.min_annot_num = min_annot_num
    self.annot_min_per_group = annot_min_per_group
    self.only_word = only_word
    self.num_preprocess_threads = num_preprocess_threads
    self.batch_size = batch_size
    self.weight_decay = weight_decay
    self.is_training = tf.Variable(True, dtype=tf.bool)
    self.assing_is_training_true_op = tf.assign(self.is_training, True)
    self.assing_is_training_false_op = tf.assign(self.is_training, False)
    self.cnn_input_length = cnn_input_length
    self.valid_ratio = valid_ratio
    self.split_labels = split_labels
    self.adaption_fc_filters = adaption_fc_filters
    self.neg_threshold = neg_threshold
    self.pos_threshold = pos_threshold
    self.loss_ratio = loss_ratio
    self.rnn_state_dim = rnn_state_dim
    self.gpu = gpu
    self.save_way = save_way

  def load_data(self):
    """
    Output:
      self.raw_dataset: should be a list of dictionary whose element
                        should be {'filename':
                                   'label_index':}
    Usage:
      filenames = [d['filename'] for d in data]
      label_indexes = [d['label_index'] for d in data]
    """
    if self.first_run:
      # if exist, get it.
      if os.path.exists(RAW_DATASET_PATH):
        with open(RAW_DATASET_PATH, 'rb') as f:
          self.vocab = pickle.load(f)
          self.raw_dataset = pickle.load(f)
          # print(self.raw_dataset)

        with open(VALID_DATASET_PATH, 'rb') as f:
          self.valid_dataset = pickle.load(f)
          # print(self.raw_dataset)
      else:
        print("Preparing dataset, please wait")
        self.raw_dataset = []
        self.vocab = []

        tmp_dataset = []
        tmp_vocab = []
        tmp_vocab_count = []
        for ele in DATASET_ITERATOR:
          # print(ele)
          gene_stage = ele['gene stage']
          urls_list = ele['urls']
          label = ele['labels']

          if self.split_labels:
            raise ValueError("Model parameter split_labels should be False")
            tmp_label = []
            for tmp in label:
              for tmp_ in tmp.split():
                tmp_label.append(tmp_)
            label = tmp_label

          # choose stage
          stage = int(gene_stage[-1])
          if stage not in self.stage_allowed:
            continue
          #
          if len(urls_list) > self.max_img:
            continue
          #
          if len(label) < self.annot_min_per_group:
            continue

          # if don not have only_word in label, continue
          tmp_flag = True
          if not self.only_word is None:
            tmp_flag = True
            for _word_ele in self.only_word:
              if _word_ele not in label:
                tmp_flag = False
          if not tmp_flag:
            continue

          # remove self.deprecated_word
          if self.deprecated_word is not None:
            tmp_label = []
            for _tmp in label:
              if _tmp in self.deprecated_word:
                continue
              else:
                tmp_label.append(_tmp)
            label = tmp_label

          tmp_dataset.append({'urls': urls_list,
                            'annot': label,
                            'gene stage': gene_stage})

        tmp_list = []
        for ele in tmp_dataset:
          label = ele['annot']
          tmp_list += label
        tmp_list = np.array(tmp_list)
        tmp_vocab, tmp_vocab_count = np.unique(tmp_list, return_counts=True)

        # print(tmp_dataset_0)
        # only the top k labels
        tmp_dataset_0 = tmp_dataset
        if not self.top_k_labels is None:
          max_arg = np.argsort(tmp_vocab_count)
          top_k_labels_idx = max_arg[-self.top_k_labels:]
          # print(top_k_labels_idx.dtype)
          allowed_word = tmp_vocab[top_k_labels_idx]
          # print(allowed_word)
          tmp_dataset_1 = []
          for ele in tmp_dataset_0:
            label = ele['annot']
            tmp_label = []
            for ele_word in label:
              if ele_word in allowed_word:
                tmp_label.append(ele_word)
              # print(tmp_label)
            if len(tmp_label) > 0:
              # print(tmp_label)
              tmp_dataset_1.append({'urls': ele['urls'],
                              'annot': tmp_label,
                              'gene stage': ele['gene stage']})

          #####################
          # NOTE: May be WRONG!
          # actual vocab may not be allowed_word
          #####################
          # self.vocab = list(allowed_word)
        else:
          tmp_dataset_0 = []
          tmp_vocab = list(tmp_vocab)
          tmp_vocab_count = list(tmp_vocab_count)
          for tmp in tmp_dataset:
            urls = tmp['urls']
            annot = tmp['annot']
            gene_stage = tmp['gene stage']
            annot_new = []
            for tmp_label in annot:
              tmp_idx = tmp_vocab.index(tmp_label)
              if tmp_vocab_count[tmp_idx] >= self.min_annot_num:
                annot_new.append(tmp_label)
            if len(annot_new) < self.annot_min_per_group:
              continue
            tmp_dataset_0.append({'urls': urls,
                              'annot': annot_new,
                              'gene stage': gene_stage})

          tmp_dataset_1 = tmp_dataset_0

          # Done filters input data

        # build self.vocab
        tmp_list = []
        for ele in tmp_dataset_1:
          tmp_list += ele['annot']
        tmp_list = np.array(tmp_list)
        self.vocab = list(np.unique(tmp_list))

        # print(tmp_dataset_1)

        for ele in tmp_dataset_1:
          gene_stage = ele['gene stage']
          urls_list = ele['urls']
          # print(urls_list)
          label = ele['annot']
          # print(gene_stage)
          # print(urls_list)
          # print(label)
          if self.save_way == 'equal_batch':
            img_file_name = os.path.join(DATASET_PAR_PATH, 'r_' + str(len(urls_list)) + '_' +
                                         gene_stage + '.' + self.image_foramt)
          else:
            img_file_name = os.path.join(DATASET_PAR_PATH, gene_stage + '.' + self.image_foramt)
          label_index = ops.annot2vec(label, self.vocab)
          if os.path.exists(img_file_name):
            self.raw_dataset.append({'filename': img_file_name,
                                   'label_index': label_index})
            continue
          # read in image and concatenate through cols
          _image = ops.get_image_from_urls_list_concat_dim0(
                                  urls_list,
                                  shape=(self.height, self.width, self.channels))
          #print(_image.shape)
          if isinstance(_image, int):
            continue
          if self.save_way == 'equal_batch':
            temp_image = _image
            for tmp_idx in range(int(self.max_img / (_image.shape[0] / self.height) + 1)):
              temp_image = np.concatenate((temp_image, _image))
            _image = temp_image[:self.max_img * self.height]

          skimage.io.imsave(img_file_name, _image)
          self.raw_dataset.append({'filename': img_file_name,
                                   'label_index': label_index})
        np.random.shuffle(self.raw_dataset)
        valid_num = int(len(self.raw_dataset) * self.valid_ratio)
        self.valid_dataset = self.raw_dataset[:valid_num]
        self.raw_dataset = self.raw_dataset[(valid_num + 1):]
        # save it
        with open(RAW_DATASET_PATH, 'wb') as f:
          pickle.dump(self.vocab, f, True)
          pickle.dump(self.raw_dataset, f, True)

        with open(VALID_DATASET_PATH, 'wb') as f:
          pickle.dump(self.valid_dataset, f, True)


  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
    """
    # _decoder = {'bmp': tf.image.decode_bmp,
    #             'jpg': tf.image.decode_jpeg,
    #             'jpeg': tf.image.decode_jpeg,
    #             'png': tf.image.decode_pbg
    #             }
    # if self.image_foramt not in _decoder=.keys:
    #   decoder = tf.image.decode_image
    # else:
    #   decoder = _decoder[self.image_foramt]

    if self.mode == 'supervise':
      if self.predict_way == 'batch_max':
        # In supervise mode, images and inputs are fed via placeholders.
        # and after a certain number of steps, information will be printed
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channels], name="image_feed")
        self.targets = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None],  # batch_size
                                    name="input_feed")
      elif self.predict_way == 'rnn':
        self.images = tf.placeholder(dtype=tf.float32, shape=[self.max_img, self.height, self.width, self.channels], name="image_feed")
        self.targets = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None],  # batch_size
                                    name="input_feed")
    elif self.mode == 'inference':
      # In supervise mode, images and inputs are fed via placeholders.
      # and after a certain number of steps, information will be printed
      self.images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channels], name="image_feed")
      self.targets = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None],  # batch_size
                                    name="input_feed")
    elif self.mode == 'train':
      image_path_list = [d['filename'] for d in self.raw_dataset]
      label_index_list = [tf.convert_to_tensor(d['label_index'], dtype=tf.float32) for d in self.raw_dataset]

      filename, label_index = tf.train.slice_input_producer([image_path_list, label_index_list], shuffle=True)
      # label_index = np.fromstring(label_index, dtype='float32')
      # images_and_labels = []
      # for thread_id in range(self.num_preprocess_threads):
      image_buffer = tf.read_file(filename)

      image = tf.image.decode_bmp(image_buffer, channels=3)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      # images_and_labels.append([image, label_index])

      image = tf.cast(image, tf.float32)
      # image = tf.image.resize_images(image, image.get_shape()[0])

      self.images = tf.reshape(image, shape=[self.max_img, self.height, self.width, 3])
      self.targets = tf.expand_dims(label_index, 0)



    else:
      raise ValueError('Wrong mode!')


  def build_base_model(self):
    """
    VGG16 + adaption layers
    see image_embedding for details.
    Args:

    Output:
      self.adaption_output: shape(batch_size, adaption_output_dim)
    """
    self.classes_num = len(self.vocab)
    #with tf.device('/gpu:0'):
    self.vgg_output = vgg16_base_layer(self.images,
                                       is_training=self.vgg_trainable,
                                       trainable=self.vgg_trainable,
                                       output_layer=self.vgg_output_layer,
                                       weight_decay=self.weight_decay)

  def build_finetune_model(self):
    if self.predict_way == 'rnn':
      #with tf.device('/gpu:1'):
      net = tf.nn.max_pool(self.vgg_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      # print(net.get_shape())
      with tf.variable_scope("adaption", values=[net]) as scope:
        # pool0
        # net = slim.max_pool2d(self.vgg_output, [2, 2], scope='pool0')
        # conv1
        for tmp_idx in range(len(self.adaption_layer_filters)):
          net = tf.layers.conv2d(net, self.adaption_layer_filters[tmp_idx],
                          self.adaption_kernels_size[tmp_idx], self.adaption_layer_strides[tmp_idx],
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                          activation=tf.nn.relu,
                          padding='same',
                          name='conv' + str(tmp_idx + 1))

          net = tf.layers.dropout(net, training=self.is_training)
      # print(net.get_shape())
      net = tf.contrib.layers.flatten(net)
      # print(net.get_shape())
      _output_l = int(self.vgg_output_layer.split('/')[0][-1])
      _area = self.height * self.width / ((2**(_output_l - 1 + 1))**2)
      _area = int(_area) * self.adaption_layer_filters[-1]
      self.memory_dim = _area
      # print(_area)
      # _area = tf.cast(net.get_shape()[-1], tf.int32)

      with tf.variable_scope("rnn", values=[net]) as scope:
        W_input = tf.Variable(tf.truncated_normal(shape=[_area, _area + self.rnn_state_dim]))
        b_input = tf.Variable(tf.zeros(shape=[_area, 1]))
        W_s = tf.Variable(tf.truncated_normal(shape=[self.rnn_state_dim, _area + self.rnn_state_dim]))
        b_s = tf.Variable(tf.zeros(shape=[self.rnn_state_dim, 1]))

        # W_m = tf.Variable(tf.truncated_normal(shape=[_area, _area + self.rnn_state_dim]))

        # initial state
        self.initial_state = tf.placeholder(tf.float32, [self.rnn_state_dim, 1])
        self.initial_memory = tf.placeholder(tf.float32, [_area, 1])
        state = self.initial_state
        memory = self.initial_memory
        input_series = tf.unstack(net)
        for idx in range(len(input_series)):
          input_ = tf.reshape(input_series[idx], [_area, 1])
          current_input = tf.concat([state, input_], axis=0)

          # print(current_input.get_shape())
          state = tf.multiply(tf.tanh(tf.matmul(W_s, current_input) + b_s), state) + state

          m_input = tf.concat([state, input_], axis=0)

          input_to_rnn = tf.multiply(tf.sigmoid(tf.matmul(W_input, m_input) + b_input), input_)

          memory = memory + input_to_rnn


          # tf.matmul(Wi2s, input_to_rnn) + bi2s
        memory = tf.divide(memory, self.max_img)
        memory = tf.reshape(memory, [1, _area])
        self.rnn_output = tf.layers.dense(memory,
              units=self.classes_num,
              activation=None,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
              name='rnn_output')



    elif self.predict_way == 'batch_max':

      # with tf.device('/gpu:1'):
      net = self.vgg_output
      with tf.variable_scope("adaption", values=[net]) as scope:
        # pool0
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # net = slim.max_pool2d(self.vgg_output, [2, 2], scope='pool0')
        # conv1
        for tmp_idx in range(len(self.adaption_layer_filters)):
          net = tf.layers.conv2d(net, self.adaption_layer_filters[tmp_idx],
                          self.adaption_kernels_size[tmp_idx], self.adaption_layer_strides[tmp_idx],
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                          activation=tf.nn.relu,
                          name='conv' + str(tmp_idx + 1))

          net = tf.layers.dropout(net, training=self.is_training)

        if self.adaption_fc_layers_num:
          if self.adaption_fc_layers_num != len(self.adaption_fc_filters):
            raise ValueError("adaption_fc_layers_num should equal len()")
          for tmp_idx in range(self.adaption_fc_layers_num):
            net = tf.layers.conv2d(net, self.adaption_fc_filters[tmp_idx], [1, 1],
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    activation=tf.nn.relu,
                    name='fc' + str(tmp_idx + 1))
            net = tf.layers.dropout(net, training=self.is_training)
        # fc
        self.fc1 = tf.layers.conv2d(net, self.classes_num, [1, 1],
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                activation=None, name='fc_output')
        # max pooling pool1
        # shape = net.get_shape()
        self.fc_o = tf.reduce_max(self.fc1, axis=(1, 2), keep_dims=False)

      self.adaption_output = self.fc_o

    else:
      raise ValueError('Wrong predict_way!')

    self.all_vars = tf.global_variables()
    self.vgg_variables = [v for v in self.all_vars if v.name.startswith('vgg_16')]
    # self.vgg_variables = tf.get_collection(
    #     tf.GraphKeys.GLOBAL_VARIABLES, scope="base")

  def build_output_layer(self):
    """
    Output layer
    """
    if self.predict_way == 'rnn':
      self.output = self.rnn_output
    elif self.predict_way == 'batch_max':
      if self.concatenate_input == True:
        self.output = self.adaption_output
      else:
        self.batch_max = tf.reduce_max(self.adaption_output, axis=(0, ), keep_dims=True)
        self.output =self.batch_max
    else:
      raise ValueError('Wrong predict_way!')



  def build_model(self):
    """Build model.
    Build loss function
    """
    #
    if self.predict_way == 'rnn':
      logits = self.output
      # labels = self.targets
      self.output_prob = tf.sigmoid(logits)
      self.cross_entropy = -(tf.reduce_sum(tf.multiply((1 - self.targets), tf.log(1. - self.output_prob + 1e-10))) +
                             self.loss_ratio * tf.reduce_sum(
                               tf.multiply(self.targets, tf.log(self.output_prob + 1e-10)))
                             )
      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      self.total_loss = tf.add_n([self.cross_entropy] + regularization_losses)
    elif self.predict_way == 'batch_max':
      logits = self.output
      labels = self.targets
      self.output_prob = tf.sigmoid(logits)

      self.logits_neg = tf.where(tf.greater(self.output_prob, self.neg_threshold),
                                    tf.subtract(1., labels),
                                    tf.zeros_like(labels))

      self.logits_pos = tf.where(tf.less(self.output_prob, self.pos_threshold),
                              labels,
                              tf.zeros_like(labels))


      '''
      self.cross_entropy = -(tf.reduce_sum(tf.multiply((1 - labels), tf.log(1. - self.output_prob + 1e-10))) +
                             tf.reduce_sum(tf.multiply(labels, tf.log(self.output_prob + 1e-10)))
                              )
      '''
      self.cross_entropy = -(tf.reduce_sum(tf.multiply(self.logits_neg, tf.log(1. - self.output_prob + 1e-10))) +
                             self.loss_ratio * tf.reduce_sum(tf.multiply(self.logits_pos, tf.log(self.output_prob + 1e-10)))
                              )
      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      self.total_loss = tf.add_n([self.cross_entropy] + regularization_losses)
      # tf.summary.scalar('total_loss', self.total_loss)

    else:
      raise ValueError('Wrong predict_way!')


  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    self.global_step = global_step

  def setup_finetune_model_initializer(self):
    """
    """
    # Restore inception variables only
    saver = tf.train.Saver(self.vgg_variables)
    def restore_fn(sess):
      tf.logging.info("Restoring vgg variables from checkpoint file %s",
                      self.ckpt_path)
      saver.restore(sess, self.ckpt_path)
    self.init_fn = restore_fn

  def setup_model_initializer(self):
    """
    """
    # Restore inception variables only
    saver = tf.train.Saver(self.all_vars)
    def restore_fn(sess):
      tf.logging.info("Restoring trained variables from checkpoint file %s",
                      self.model_ckpt_path)
      saver.restore(sess, self.model_ckpt_path)
    self.model_init_fn = restore_fn

  def build(self):
    """Build total model.
    """
    self.load_data()
    self.build_inputs()
    if self.gpu:
      with tf.device('/gpu:0'):
        self.build_base_model()
      with tf.device('/gpu:1'):
        self.build_finetune_model()
        self.build_output_layer()
        self.build_model()

    else:
      self.build_base_model()
      self.build_finetune_model()
      self.build_output_layer()
      self.build_model()

    self.setup_finetune_model_initializer()
    if self.model_ckpt_path:
      self.setup_model_initializer()
    self.setup_global_step()



