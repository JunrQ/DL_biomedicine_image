
from model import Model

import datetime
import numpy as np
import os
import time

import tensorflow as tf

CKPT_PATH = r'E:\zcq\codes\weakcnn\theano\vgg_16.ckpt'



def main(initial_learning_rate=0.001,
         optimizer=tf.train.AdamOptimizer(0.0001),
         max_steps=99999999,
         print_every_steps=100,
         num_pred=5
          ):




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
    model = Model(ckpt_path=CKPT_PATH)

    model.build()

    # Set up the learning rate.
    learning_rate_decay_fn = None

    learning_rate = tf.constant(initial_learning_rate)


    train_op = optimizer.minimize(
        model.total_loss
        )


    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(init)
      sess.run(model.init_fn)
      tf.train.start_queue_runners(sess=sess)




      for x in xrange(max_steps + 1):
        start_time = time.time()

        i = [train_op, model.total_loss]

        o = sess.run(i)

        loss_value = o[1]

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        step = sess.run(model.global_step)

        if step > 1 and step % print_every_steps:
          sess.run(model.assing_is_training_false_op)
          pred = model.output.eval()
          targets = model.targets.eval()
          sess.run(model.assing_is_training_true_op)

          for single_batch in range(len(pred)):
            target = ''
            prediction = ''
            pred_result = np.argsort(pred[single_batch])
            for s in model.vocab[targets[single_batch] == 1.]:
              target += (s + ' ')
            print('Target: %s' % target)

            for s in range(num_pred):
              # print(pred_result)
              prediction += (str(model.vocab[pred_result[s]]) + ' ')
            print('Prediction: %s' % prediction)

if __name__ == '__main__':
    main()



