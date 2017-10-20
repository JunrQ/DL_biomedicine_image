"""
Train the model.
[IMPORTANT]
if you change parameters of model, please do delete file:
  E:\zcq\codes\pkl\raw_dataset.pkl
"""
from model import Model
import ops
import image_processing

import datetime
import numpy as np
import os
import time

import tensorflow as tf

import itertools

from file_path import *

import math

import pickle

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

train_sample_num = 2093 # test 572
def main(initial_learning_rate=0.001,
          gpu=1,
          optimizer=tf.train.AdamOptimizer(1e-4),
          max_steps=train_sample_num * 40,
          print_every_steps=int(train_sample_num / 4),
          save_frequence=train_sample_num * 10,
          num_pred=6,
          shuffle=True,
          batch_size=5,
          top_k_labels=30,
          min_annot_num=20,
          concatenate_input=False,
          weight_decay=0.000005,
          predict_way='rnn',
          mode='train',
          input_queue_length=80,
          stage_allowed=[6],
          adaption_layer_filters=[1024, 512, 64],
          adaption_kernels_size=[[3, 3], [3, 3], [3, 3]],
          adaption_layer_strides=[(1, 1), (1, 1), (1, 1)],
          adaption_fc_layers_num=0,
          adaption_fc_filters=[],
          neg_threshold=0.3,
          pos_threshold=0.9,
          loss_ratio=5.0
          ):
  """
  Args:
    concatenate_input: if True: batch_size is used,  concatenated images and corresponding labels are input
                            if False: a group are a batch
  Return:

  """
  config_dict = {'adaption_layer_filters': adaption_layer_filters,
                 'adaption_kernels_size': adaption_kernels_size,
                 'adaption_layer_strides': adaption_layer_strides,
                 'adaption_fc_layers_num': adaption_fc_layers_num,
                 'adaption_fc_filters': adaption_fc_filters,
                 'stage_allowed': stage_allowed}
  with open(CONFIG_PATH, 'wb') as f:
    pickle.dump(config_dict, f, True)


  g = tf.Graph()
  with g.as_default():

    # Default parameters
    """
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
     channels=3
    """
    model = Model(ckpt_path=CKPT_PATH,
                  gpu=gpu,
                  model_ckpt_path=MODEL_CKPT_PATH,
                  mode=mode,
                  concatenate_input=concatenate_input,
                  predict_way=predict_way,
                  min_annot_num=min_annot_num,
                  top_k_labels=top_k_labels,
                  weight_decay=weight_decay,
                  stage_allowed=stage_allowed,
                  adaption_layer_filters=adaption_layer_filters,
                  adaption_kernels_size=adaption_kernels_size,
                  adaption_layer_strides=adaption_layer_strides,
                  adaption_fc_layers_num=adaption_fc_layers_num,
                  adaption_fc_filters=adaption_fc_filters,
                  neg_threshold=neg_threshold,
                  pos_threshold=pos_threshold,
                  loss_ratio=loss_ratio
                  )

    model.build()
    vocab = np.array(model.vocab)

    if model.mode == 'train':
      if predict_way == 'batch_max':
        train_op = optimizer.minimize(
          model.total_loss
        )
        # summary_op = tf.merge_all_summaries()
        init = tf.global_variables_initializer()
        saver_model = tf.train.Saver()

        config = tf.ConfigProto()
        if model.gpu:
          config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
          print("Number of dataset: %d"%len(model.raw_dataset))
          print("Number of test dataset: %d"%len(model.valid_dataset))
          print("Number of classes: %d"%model.classes_num)
          print("Vocab: ", model.vocab)
          sess.run(init)
          if model.model_ckpt_path:
            model.model_init_fn(sess)
          else:
            model.init_fn(sess)
          tf.train.start_queue_runners(sess=sess)
          start_time = time.time()
          for x in range(max_steps + 1):

            i = [train_op, model.total_loss]

            o = sess.run(i)
            loss_value = o[1]

            localtime = time.asctime(time.localtime(time.time()))

            if (x % 1000 == 0) and (x > 0):
              duration = time.time() - start_time
              format_str = ('step %d, loss = %.2f ,%.3f sec/1000 samples. Time: %s')
              print(format_str % (x, loss_value, duration, localtime))
              start_time = time.time()


            if (x > 1) and (x % save_frequence == 0):
              saver_model.save(sess, SAVE_PATH, global_step=x)

      elif predict_way == 'rnn':
        train_op = optimizer.minimize(
          model.total_loss
        )
        # summary_op = tf.merge_all_summaries()
        init = tf.global_variables_initializer()
        saver_model = tf.train.Saver()

        config = tf.ConfigProto()
        if model.gpu:
          config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
          print("Number of dataset: %d" % len(model.raw_dataset))
          print("Number of test dataset: %d" % len(model.valid_dataset))
          print("Number of classes: %d" % model.classes_num)
          print("Vocab: ", model.vocab)
          sess.run(init)
          if model.model_ckpt_path:
            model.model_init_fn(sess)
          else:
            model.init_fn(sess)
          tf.train.start_queue_runners(sess=sess)
          start_time = time.time()
          for x in range(max_steps + 1):

            i = [train_op, model.total_loss]

            o = sess.run(i, {model.initial_state: np.zeros((model.rnn_state_dim, 1), dtype='float32'),
                             model.initial_memory: np.zeros((model.memory_dim, 1), dtype='float32')})
            loss_value = o[1]

            localtime = time.asctime(time.localtime(time.time()))

            if (x % 1000 == 0) and (x > 0):
              duration = time.time() - start_time
              format_str = ('step %d, loss = %.2f ,%.3f sec/1000 samples. Time: %s')
              print(format_str % (x, loss_value, duration, localtime))
              start_time = time.time()

            if (x > 1) and (x % save_frequence == 0):
              saver_model.save(sess, SAVE_PATH, global_step=x)

    elif model.mode == 'supervise':
      if model.predict_way == 'batch_max':
        # Set up the learning rate.
        learning_rate_decay_fn = None

        train_op = optimizer.minimize(
            model.total_loss
            )

        init = tf.global_variables_initializer()
        saver_model = tf.train.Saver()
        if shuffle:
          np.random.shuffle(model.raw_dataset)
        dataset = itertools.cycle(iter(model.raw_dataset))

        dataset_queue = []
        for _ in range(input_queue_length):
          dataset_queue.append(dataset.__next__())

        def queue_atom(dataset_queue):
          np.random.shuffle(dataset_queue)
          data = dataset_queue.pop()
          dataset_queue.append(dataset.__next__())
          return data

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:
          print("Number of dataset: %d"%len(model.raw_dataset))
          print("Number of test dataset: %d"%len(model.valid_dataset))
          print("Number of classes: %d"%model.classes_num)
          print("Vocab: ", model.vocab)
          sess.run(init)
          if model.model_ckpt_path:
            model.model_init_fn(sess)
          else:
            model.init_fn(sess)
          # tf.train.start_queue_runners(sess=sess)

          for x_step in range(max_steps + 1):

            if (x_step > 1) and (x_step % save_frequence == 0):
              saver_model.save(sess, SAVE_PATH, global_step=x_step)

            while True:
              single_data = queue_atom(dataset_queue)
              i = ops.read_image_from_single_file(single_data['filename'])
              l = single_data['label_index']
              if not isinstance(i, int):
                # i = np.reshape(i, (-1, model.height, model.width))
                i = i[None, :]
                l = l[None, :]
                break
            if model.concatenate_input == True:
              # !!!!NOTE!!!!:
              # for the fact shape might not be the same, this way might not work
              for tmp in range(batch_size - 1):
                while True:
                  single_data = queue_atom(dataset_queue)
                  i0 = ops.read_image_from_single_file(single_data['filename'])
                  l0 = single_data['label_index']
                  if not isinstance(i0, int):
                    i0 = i0[None, :]
                    l0 = l0[None, :]
                    break
                i = np.concatenate((i, i0))
                l = np.concatenate((l, l0))
            else:
              i = np.reshape(i, (-1, model.height, model.width, model.channels))
              i_new = np.zeros(i.shape, dtype='float32')
              for tmp in range(i.shape[0]):
                i_new[tmp] = image_processing.np_image_random(i[tmp],
                                              rotate_angle=[-15, 15],
                                              rescale=None,
                                              whiten=False,
                                              normalize=False,
                                              hsv=False)
                # print(i_new[tmp])
              i = i_new
              if len(i) > 1:
                if shuffle:
                  np.random.shuffle(i)

            if (x_step > 0) and (x_step % print_every_steps == 0):
              sess.run(model.assing_is_training_false_op)

              prob = sess.run([model.output,
                               model.output_prob,
                               model.targets,
                               model.logits_neg,
                               model.logits_pos,
                               model.total_loss,
                               model.cross_entropy],
                              feed_dict={model.images: i,
                                         model.targets: l})

              print('loss: ', prob[-1], prob[-2])
              sess.run(model.assing_is_training_true_op)

              for single_batch in range(len(prob[0])):
                target = ''
                prediction = ''
                pred_result = np.argsort(prob[0][single_batch])

                for s in vocab[l[single_batch] == 1.]:
                  target += (s + ' \n')
                print('Target: \n%s' % target)

                for s in range(num_pred):
                  # print(pred_result)
                  prediction += (str(vocab[pred_result[-(s+1)]]) + ': '
                                + str(prob[1][single_batch][pred_result[-(s+1)]]) + '\n')
                print('Prediction: \n%s' % prediction)

            train_op.run(feed_dict={model.images: i,
                                    model.targets: l})

      elif model.predict_way == 'rnn':

        train_op = optimizer.minimize(
          model.total_loss
        )

        init = tf.global_variables_initializer()
        saver_model = tf.train.Saver()
        if shuffle:
          np.random.shuffle(model.raw_dataset)
        dataset = itertools.cycle(iter(model.raw_dataset))

        dataset_queue = []
        for _ in range(input_queue_length):
          dataset_queue.append(dataset.__next__())

        def queue_atom(dataset_queue):
          np.random.shuffle(dataset_queue)
          data = dataset_queue.pop()
          dataset_queue.append(dataset.__next__())
          return data

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
          print("Number of dataset: %d" % len(model.raw_dataset))
          print("Number of test dataset: %d" % len(model.valid_dataset))
          print("Number of classes: %d" % model.classes_num)
          print("Vocab: ", model.vocab)
          sess.run(init)
          if model.model_ckpt_path:
            model.model_init_fn(sess)
          else:
            model.init_fn(sess)
          # tf.train.start_queue_runners(sess=sess)

          for x_step in range(max_steps + 1):

            if (x_step > 1) and (x_step % save_frequence == 0):
              saver_model.save(sess, SAVE_PATH, global_step=x_step)

            while True:
              single_data = queue_atom(dataset_queue)
              i = ops.read_image_from_single_file(single_data['filename'])
              l = single_data['label_index']
              if not isinstance(i, int):
                # i = np.reshape(i, (-1, model.height, model.width))
                i = i[None, :]
                l = l[None, :]
                break
            if model.concatenate_input == True:
              # !!!!NOTE!!!!:
              # for the fact shape might not be the same, this way might not work
              for tmp in range(batch_size - 1):
                while True:
                  single_data = queue_atom(dataset_queue)
                  i0 = ops.read_image_from_single_file(single_data['filename'])
                  l0 = single_data['label_index']
                  if not isinstance(i0, int):
                    i0 = i0[None, :]
                    l0 = l0[None, :]
                    break
                i = np.concatenate((i, i0))
                l = np.concatenate((l, l0))
            else:
              i = np.reshape(i, (-1, model.height, model.width, model.channels))
              i_new = np.zeros(i.shape, dtype='float32')
              for tmp in range(i.shape[0]):
                i_new[tmp] = image_processing.np_image_random(i[tmp],
                                                              rotate_angle=[-15, 15],
                                                              rescale=None,
                                                              whiten=False,
                                                              normalize=False,
                                                              hsv=False)
                # print(i_new[tmp])

              i = i_new
              if model.save_way != 'equal_batch':
                i = np.repeat(i, int(model.max_img / len(i)) + 1, axis=0)
                i = i[:model.max_img]
              if shuffle:
                np.random.shuffle(i)

            if (x_step > 0) and (x_step % print_every_steps == 0):
              sess.run(model.assing_is_training_false_op)

              prob = sess.run([model.output,
                               model.output_prob,
                               model.targets,
                               model.total_loss,
                               model.cross_entropy],
                              feed_dict={model.images: i,
                                         model.targets: l,
                                         model.initial_state: np.zeros((model.rnn_state_dim, 1), dtype='float32'),
                                         model.initial_memory: np.zeros((model.memory_dim, 1), dtype='float32')})

              print('loss: ', prob[-1], prob[-2])
              sess.run(model.assing_is_training_true_op)

              for single_batch in range(len(prob[0])):
                target = ''
                prediction = ''
                pred_result = np.argsort(prob[0][single_batch])

                for s in vocab[l[single_batch] == 1.]:
                  target += (s + ' \n')
                print('Target: \n%s' % target)

                for s in range(num_pred):
                  # print(pred_result)
                  prediction += (str(vocab[pred_result[-(s + 1)]]) + ': '
                                 + str(prob[1][single_batch][pred_result[-(s + 1)]]) + '\n')
                print('Prediction: \n%s' % prediction)

            train_op.run(feed_dict={model.images: i,
                                    model.targets: l,
                                    model.initial_state: np.zeros((model.rnn_state_dim, 1), dtype='float32'),
                                    model.initial_memory: np.zeros((model.memory_dim, 1), dtype='float32')})

if __name__ == '__main__':
    main()




