"""
Train the model.
"""
from model import Model
import ops

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
         max_steps=99999999,
         print_every_steps=10,
         num_pred=10,
         shuffle=True,
         batch_size=5,
         concatenate_input=False
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
                  concatenate_input=concatenate_input)

    model.build()

    if model.mode == 'supervise':
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
        sess.run(init)
        model.init_fn(sess)
        tf.train.start_queue_runners(sess=sess)

        for x in range(max_steps + 1):
          single_data = dataset.__next__()
          # print(single_data)
          # read in images
          while True:
            i = ops.read_image_from_single_file(single_data['filename'])
            l = single_data['label_index']
            if not isinstance(i, int):
              # i = np.reshape(i, (-1, model.height, model.width))
              i = i[None, :]
              l = l[None, :]
              break
          if model.concatenate_input == True:
            for tmp in range(batch_size - 1):
              while True:
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

          print(i.shape)
          print(l.shape)
          # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          step = sess.run(model.global_step)

          if step > 1 and step % print_every_steps:
            sess.run(model.assing_is_training_false_op)

            prob = sess.run([model.output,],
                            feed_dict={model.images: i,
                                       model.targets: l})

            sess.run(model.assing_is_training_true_op)

            for single_batch in range(len(prob)):
              target = ''
              prediction = ''
              pred_result = np.argsort(prob[single_batch])
              for s in model.vocab[l[single_batch] == 1.]:
                target += (s + ' ')
              print('Target: %s' % target)

              for s in range(num_pred):
                # print(pred_result)
                prediction += (str(model.vocab[pred_result[s]]) + ' ')
              print('Prediction: %s' % prediction)

          train_op.run(feed_dict={model.images: i,
                                  model.targets: l})

if __name__ == '__main__':
    main()



