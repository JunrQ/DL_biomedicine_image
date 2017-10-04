"""
Inference the mode
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
         save_frequence=6666,
         num_pred=10,
         shuffle=True,
         batch_size=5,
         top_k_labels=20,
         min_annot_num=20,
         concatenate_input=False,
         weight_decay=0.00004,
         predict_way='batch_max',
         top_k_accuracy=3
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
                  model_ckpt_path=MODEL_CKPT_PATH,
                  mode='inference',
                  concatenate_input=concatenate_input,
                  predict_way=predict_way,
                  min_annot_num=min_annot_num,
                  top_k_labels=top_k_labels,
                  weight_decay=weight_decay
                  )

    model.build()
    vocab = np.array(model.vocab)

    if model.mode == 'inference' and model.predict_way == 'batch_max':

      init = tf.global_variables_initializer()
      if shuffle:
        np.random.shuffle(model.valid_dataset)
      dataset = iter(model.valid_dataset)

      data_num = 0
      data_total_num = len(model.valid_dataset)

      accuracy = 0
      total_sample = 0

      with tf.Session() as sess:
        print("Number of classes: %d"%model.classes_num)
        sess.run(init)
        # model.init_fn(sess)
        model.model_init_fn(sess)

        for x_step in range(max_steps + 1):
          # print(single_data)
          # read in images
          while True:
            if data_num >= data_total_num:
              break
            single_data = dataset.__next__()
            data_num += 1
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
            raise ValueError
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

          sess.run(model.assing_is_training_false_op)
          if 1:
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

              total_sample += 1
              tmp_t = np.sum(l[0])
              tmp_acc =  (np.sum(l[0][pred_result[-1:-(1+tmp_t):-1]])/ tmp_t)
              accuracy += tmp_acc

              print("Current accuracy:  ", tmp_acc)


      print("Average accuracy: ", accuracy / total_sample)


if __name__ == '__main__':
    main()




