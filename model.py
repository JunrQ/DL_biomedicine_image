

from image_embedding import vgg16_base_layer, adaption_layer
import image_processing
import ops

import os
import cPickle as pickle
import numpy as np
import skimage.io
import tensorflow as tf

# where to save .pkl, after concatnate images
PKL_PATH = r''
# where to save dataset: self.raw_dataset, self.vocab, after load_data
RAW_DATASET_PATH = r''
# saved image parent path, parent_path + gene_stage + .format, give the image file path
DATASET_PAR_PATH = PKL_PATH
# csvfile path
CSVFILE_PATH = r'E:\csfile.csv'
# original image parent path
IMAGE_PARENT_PATH = r'E:\pic_data'

DATASET_ITERATOR = ops.csvfile2iterator(csvfile_path=CSVFILE_PATH, parent_path=IMAGE_PARENT_PATH)


class Model(object):
  """
  """

  def __init__(self,
               ckpt_path,
               image_foramt='bmp',
               first_run=True,
               predict_way='fc',
               adaption_output_dim=1024,
               mode="train",
               vgg_trainable=False,
               vgg_output_layer='conv4',
               adaption_layer_filters=[4096, 4096],
               adaption_kernels_size=[[5, 5], [1, 1]],
               adaption_fc_layers_num=2,
               height=128,
               width=320,
               channels=3,
               image_concatnate_way='col',# below params used for filter input
               top_k_labels=None,
               concatenate_input=True,
               deprecated_word=None,
               stage_allowed=[5, 6],
               max_img=15,
               annot_min_per_group=0,
               only_word=None,
               num_preprocess_threads=4,
               batch_size=5,
               weight_decay=0.00004
               ):
    """
    Args:
      first_run: if first_run, original image and label need to convert to pkl file,
                 with multi images in a group concatnate through cols. Just make it True.
                 Note:
                  **If you change the parameters, you should delete the RAW_DATASET_PATH file**
      image_foramt: Used when saving concatenated images
      predict_way:
          'fc'
          'cnn'
      image_concatnate_way: Original should be concatnate through columns
      tok_k_labels: if not None, e.g. 5, then top 5 labels are concerned
      concatenate_input: if True, input are concatenate, else, batch
      deprecated_word=None,
      stage_allowed=[5, 6],
      max_img=
      concatenate_input: if True, the input(a group: gene stage) for the model is concatenate,
                        if False, the input(a group: gene stage) for the model is treated as a batch
    """
    self.ckpt_path = ckpt_path
    self.mode = mode
    self.height = height
    self.width = width
    self.channels = channels
    self.vgg_output_layer = vgg_output_layer
    self.adaption_layer_filters = adaption_layer_filters
    self.adaption_kernels_size = adaption_kernels_size
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
    self.annot_min_per_group = annot_min_per_group
    self.only_word = only_word
    self.num_preprocess_threads = num_preprocess_threads
    self.batch_size = batch_size
    self.weight_decay = weight_decay
    self.is_training = tf.Variable(True, dtype=tf.bool)
    self.assing_is_training_true_op = tf.assign(self.is_training, True)
    self.assing_is_training_false_op = tf.assign(self.is_training, False)

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
          self.raw_dataset = pickle.load(f)
          self.vocab = pickle.load(f)
      else:
        self.raw_dataset = []
        self.vocab = []

        tmp_dataset = []
        tmp_vocab = []
        tmp_vocab_count = []
        # [TODO]: fill in keys and **DATASET_ITERATOR**
        for ele in DATASET_ITERATOR:
          gene_stage = ele['gene stage']
          urls_list = ele['urls']
          label = ele['labels']

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

          # count vocab
          for tmp_word in label:
            if not (tmp_word in tmp_vocab):
              tmp_vocab.append(tmp_word)
              tmp_vocab_count.append(0)
            else:
              idx = tmp_vocab.index(tmp_vocab)
              tmp_vocab_count[idx] += 1

        # only the top k labels
        if not self.top_k_labels is None:
          max_arg = np.argsort(np.array(tmp_vocab_count))
          top_k_labels_idx = max_arg[:self.top_k_labels]
          allowed_word = tmp_vocab[top_k_labels_idx]

          tmp_dataset_1 = []
          for ele in tmp_dataset:
            label = ele['annot']
            for ele_word in label:
              tmp_label = []
              if ele_word in allowed_word:
                tmp_label.append(ele_word)
            if len(tmp_label) > 0:
              tmp_dataset_1.append({'urls': ele['urls'],
                              'annot': tmp_label,
                              'gene stage': ele['gene_stage']})
        else:
          tmp_dataset_1 = tmp_dataset

        # Done filters input data

        # build self.vocab
        for ele in tmp_dataset_1:
          label = ele['annot']
          for tmp_word in label:
            if tmp_word not in self.vocab:
              self.vocab.append(tmp_word)

        for ele in tmp_dataset_1:
          gene_stage = ele['gene stage']
          urls_list = ele['urls']
          label = ele['annot']
          # read in image and concatenate through cols
          _image = ops.get_image_from_urls_list_concat_dim0(
                                  urls_list, 
                                  shape=(self.height, self.width, self.channels))

          img_file_name = os.path.join(DATASET_PAR_PATH, gene_stage + '.' + self.image_foramt)
          label_index = ops.annot2vec(label, self.vocab)
          skimage.io.imsave(img_file_name, _image)
          self.raw_dataset.append({'filename': img_file_name,
                                   'label_index': label_index})
        # save it
        with open(RAW_DATASET_PATH, 'wb') as f:
          pickle.dump(self.vocab, f, True)
          pickle.dump(self.raw_dataset, f, True)


  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
    """
    _decoder = {'bmp': tf.image.decode_bmp,
                'jpg': tf.image.decode_jpeg,
                'jpeg': tf.image.decode_jpeg,
                'png': tf.image.decode_pbg
                }
    if self.image_foramt not in _decoder.keys:
      decoder = tf.image.decode_image
    else:
      decoder = _decoder[self.image_foramt]


    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")


    else:
      # Train mode
      data = self.raw_dataset
      filenames = [d['filename'] for d in data]
      label_indexes = [d['label_index'] for d in data]
      filename, label_index = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

      images_and_labels = []
      for thread_in in range(self.num_preprocess_threads):
        encoded_image = tf.read_file(filename)

        image = image_processing.read_image(encoded_image,
                                decoder,
                                self.concatenate_input,
                                height=self.height,
                                width=self.width,
                                train=True,
                                concatnate_way=self.image_concatnate_way)
        if self.concatenate_input:
          images_and_labels.append([image, label_index])

          images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=self.batch_size,
            capacity=2 * self.num_preprocess_threads * self.batch_size)
        else:
          images = image
          batch_size = image.get_shape()[0]
          label_index_batch = tf.reshape(tf.tile(label_index, batch_size), shape=[batch_size, -1])


        self.images = images
        self.targets =label_index_batch


  def build_finetune_model(self):
    """
    VGG16 + adaption layers
    see image_embedding for details.
    Args:
    
    Output:
      self.adaption_output: shape(batch_size, adaption_output_dim)
    """
    self.vgg_output = vgg16_base_layer(self.images,
                                      is_training= self.is_training,
                                      trainable=self.vgg_trainable,
                                      output_layer=self.vgg_output_layer,
                                      weight_decay=self.weight_decay)
    self.adaption_output = adaption_layer(self.vgg_output,
                                          is_training=self.is_training,
                                          num_output=self.adaption_output_dim,
                                          weight_decay=self.weight_decay)

    self.all_vars = tf.global_variables()
    # self.vgg_variables = [v for v in self.all_vars if v.name.startswith('vgg_16')]
    self.vgg_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16")

  def build_output_layer(self):
    """

    """
    self.classes_num = len(self.vocab)

    self.output = tf.layers.conv2d(self.adaption_output, self.classes_num, [1, 1],
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                      activation_fn=None,
                      normalizer_fn=None,
                      scope='fc_output')



  def build_model(self):
    """Build model.
    Build loss function

    """
    # 
    net_output = self.adaption_output
    if self.predict_way == 'fc':
      logits = self.output
      labels = self.targets
      cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
      self.cross_entropy_loss = tf.reduce_mean(cross_entropy)
      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      self.total_loss = tf.add_n([self.cross_entropy_loss] + regularization_losses)

    elif self.predict_way == 'cnn':
      # TODO
      pass

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
    # Restore inception variables only.
    saver = tf.train.Saver(self.vgg_variables)

    def restore_fn(sess):
      tf.logging.info("Restoring vgg variables from checkpoint file %s",
                      self.ckpt_path)
      saver.restore(sess, self.ckpt_path)

    self.init_fn = restore_fn

  def build(self):
    """Build total model.
    """
    self.load_data()
    self.build_inputs()
    self.build_finetune_model()
    self.build_model()
    self.setup_finetune_model_initializer()
    self.setup_global_step()



