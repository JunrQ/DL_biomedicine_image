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

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def main(initial_learning_rate=0.001,
         optimizer=tf.train.AdamOptimizer(1e-4),
         max_steps=999999999999,
         print_every_steps=666,
         save_frequence=2500,
         num_pred=10,
         shuffle=True,
         batch_size=5,
         top_k_labels=60,
         min_annot_num=40,
         concatenate_input=False,
         weight_decay=0.00004,
         predict_way='batch_max',
         input_queue_length=40
          ):
  """
  Args:
    concatenate_input: if True: batch_size is used,  concatenated images and corresponding labels are input
                            if False: a group are a batch
  Return:

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
                  top_k_labels=top_k_labels,
                  weight_decay=weight_decay
                  )

    model.build()
    vocab = np.array(model.vocab)

    if model.mode == 'supervise' and model.predict_way == 'batch_max':
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

      with tf.Session() as sess:
        print("Number of classes: %d"%model.classes_num)
        sess.run(init)
        model.init_fn(sess)
        # model.model_init_fn(sess)
        # tf.train.start_queue_runners(sess=sess)

        for x_step in range(max_steps + 1):
          if (x_step > 1) and (x_step % save_frequence == 0):
            saver_model.save(sess, SAVE_PATH, global_step=x_step+13332)
          # print(single_data)
          # read in images
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
                                            rotate_angle=[-20, 20],
                                            rescale=None,
                                            whiten=False,
                                            normalize=False)
              # print(i_new[tmp])
            i = i_new
          # print(i.shape)
          # print(l.shape)
          # print(i)
          # print(l)
          # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          # step = sess.run(model.global_step)

          if (x_step > 0) and (x_step % print_every_steps == 0):
            sess.run(model.assing_is_training_false_op)

            prob = sess.run([model.output,
                             model.output_prob,
                             model.targets,
                             model.fc0,
                             model.logits_neg,
                             model.logits_pos,
                             # model.fc1,
                             model.total_loss,
                             model.cross_entropy],
                            feed_dict={model.images: i,
                                       model.targets: l})

            print(prob[1].shape, prob[2].shape)
            # print(i)
            print(prob[1], '\n', prob[4], '\n', prob[5], '\n', prob[-1], prob[-2])
            # print(prob[3])
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
                prediction += (str(vocab[pred_result[-(s+1)]]) + ' \n')
              print('Prediction: %s' % prediction)

          train_op.run(feed_dict={model.images: i,
                                  model.targets: l})

if __name__ == '__main__':
    main()




