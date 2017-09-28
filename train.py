"""
Train the model.
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

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def main(initial_learning_rate=0.001,
         optimizer=tf.train.AdamOptimizer(0.0001),
         max_steps=99999999999999,
         print_every_steps=500,
         num_pred=10,
         shuffle=True,
         batch_size=5,
         top_k_labels=20,
         min_annot_num=40,
         concatenate_input=False,
         predict_way='batch_max'
          ):
  """

  :param initial_learning_rate:
  :param optimizer:
  :param max_steps:
  :param print_every_steps:
  :param num_pred:
  :param shuffle:
  :param batch_size:
  :param concatenate_input: if True: batch_size concatenated images and corresponding labels are input
                            if False: a group are a batch

  :return:
  """
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
                  mode='supervise',
                  concatenate_input=concatenate_input,
                  predict_way=predict_way,
                  min_annot_num=min_annot_num,
                  top_k_labels=top_k_labels
                  )

    model.build()
    vocab = np.array(model.vocab)

    if model.mode == 'supervise' and model.predict_way == 'batch_max':
      # Set up the learning rate.
      learning_rate_decay_fn = None

      train_op = optimizer.minimize(
          model.total_loss
          )

      init = tf.initialize_all_variables()
      saver_model = tf.train.Saver()
      if shuffle:
        np.random.shuffle(model.raw_dataset)
      dataset = itertools.cycle(iter(model.raw_dataset))

      with tf.Session() as sess:
        print("Number of classes: %d"%model.classes_num)
        sess.run(init)
        model.init_fn(sess)
        tf.train.start_queue_runners(sess=sess)

        for x_step in range(max_steps + 1):
          # print(single_data)
          # read in images
          while True:
            single_data = dataset.__next__()
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
                single_data = dataset.__next__()
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
                                            rotate_angle=[-20, 20],
                                            rescale=None,
                                            whiten=False,
                                            normalize=True)
            i = i_new
          # print(i.shape)
          # print(l.shape)
          # print(i)
          # print(l)
          # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          # step = sess.run(model.global_step)

          if (x_step > 1) and (x_step % print_every_steps == 0):
            sess.run(model.assing_is_training_false_op)

            prob = sess.run([model.output,],
                            feed_dict={model.images: i,
                                       model.targets: l})

            # print(prob)
            sess.run(model.assing_is_training_true_op)

            for single_batch in range(len(prob[0])):
              target = ''
              prediction = ''
              pred_result = np.argsort(prob[0][single_batch])
              # print(prob[0].shape)
              # print(l[single_batch] == 1)
              # print(type(model.vocab))
              # print(pred_result)

              for s in vocab[l[single_batch] == 1.]:
                target += (s + ' \n')
              print('Target: %s' % target)

              for s in range(num_pred):
                # print(pred_result)
                prediction += (str(vocab[pred_result[s]]) + ' \n')
              print('Prediction: %s' % prediction)

          train_op.run(feed_dict={model.images: i,
                                  model.targets: l})

if __name__ == '__main__':
    main()



