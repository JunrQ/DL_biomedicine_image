"""
Inference the mode

Performance Evaluation Criteria:
    1) criteria that are directly extended from traditional single-label
        evaluation measures to multi-labels [16], [23], [24],
        such as the *macro F1*, *micro F1*, and *AUC (the Area Under ROC Curve)*;
    2) criteria that are specifically designed for the setting of multi-labels [48], [56],
        such as the *hamming loss*, *one-error*, *coverage*, *average precision*,
        and *ranking loss*.
    exclude *hamming loss*


"""

from model import Model
import ops
import image_processing

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score,\
                            auc, coverage_error, label_ranking_loss


from file_path import *

import math
import pickle

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def main(initial_learning_rate=0.001,
          optimizer=tf.train.AdamOptimizer(1e-4),
          max_steps=999999999999,
          print_every_steps=500,
          save_frequence=1000,
          num_pred=5,
          shuffle=True,
          batch_size=5,
          top_k_labels=10,
          min_annot_num=150,
          concatenate_input=False,
          weight_decay=0.000005,
          predict_way='batch_max',
          input_queue_length=80,
          stage_allowed=[6],
          adaption_layer_filters=[4096, 4096],
          adaption_kernels_size=[[5, 5], [3, 3]],
          adaption_layer_strides=[(2, 2), (1, 1)],
          adaption_fc_layers_num=1,
          adaption_fc_filters=[1024],
          top_k_accuracy=3,
          threshold=0.8,
          calculusN=100
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
                  weight_decay=weight_decay,
                  stage_allowed=stage_allowed,
                  adaption_layer_filters=adaption_layer_filters,
                  adaption_kernels_size=adaption_kernels_size,
                  adaption_layer_strides=adaption_layer_strides,
                  adaption_fc_layers_num=adaption_fc_layers_num,
                  adaption_fc_filters=adaption_fc_filters
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
      # data_total_num = 5

      # accuracy = 0
      # total_sample = 0

      y_true = [] # shape [n_samples, n_classes]
      y_score = [] # shape [n_samples, n_classes]

      with tf.Session() as sess:
        print("Number of test dataset: %d"%len(model.valid_dataset))
        print("Number of classes: %d"%model.classes_num)
        sess.run(init)
        model.model_init_fn(sess)

        for x_step in range(max_steps + 1):
          if data_num >= data_total_num:
            break
          while True:
            if data_num >= data_total_num:
              break
            single_data = dataset.__next__()
            data_num += 1
            i = ops.read_image_from_single_file(single_data['filename'])
            l = single_data['label_index']
            if not isinstance(i, int):
              i = i[None, :]
              l = l[None, :]
              break
          if model.concatenate_input == True:
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

            # print(prob[1].shape, prob[2].shape)
            # print(prob[1], '\n', prob[4], '\n', prob[5], '\n', prob[-1], prob[-2])

            for single_batch in range(len(prob[0])):
              #
              y_true.append(l[single_batch])
              y_score.append(prob[1][single_batch])

              target = ''
              prediction = ''
              pred_result = np.argsort(prob[0][single_batch])

              for s in vocab[l[single_batch] == 1.]:
                target += (s + ' \n')
              print('Target: %s' % target)

              for s in range(num_pred):
                prediction += (str(vocab[pred_result[-(s+1)]]) + ' \n')
              print('Prediction: %s' % prediction)

              '''
              total_sample += 1
              tmp_t = np.sum(l[0])
              tmp_acc =  (np.sum(l[0][pred_result[-1:-(1+tmp_t):-1]])/ tmp_t)
              accuracy += tmp_acc

              print("Current accuracy:  ", tmp_acc)
              '''


      # print("Average accuracy: ", accuracy / total_sample)

      y_true = np.array(y_true)
      y_score = np.array(y_score)
      f1_micro = 0.0
      f1_macro = 0.0
      # print(model.classes_num)
      # print(y_true.shape)
      # print(y_true)
      # print(y_score)

      average_precision = average_precision_score(y_true[:, 0], y_score[:, 0])
      print("Average precision: ", average_precision)

      # f1 score
      y_pred = np.where(y_score >= threshold, 1, 0)
      sk_f1_micro = f1_score(y_true, y_pred, average='micro')
      sk_f1_macro = f1_score(y_true, y_pred, average='macro')
      print("sklearn f1 micro score: ", sk_f1_micro)
      print("sklearn f1 macro score: ", sk_f1_macro)

      TP = np.multiply(y_true, y_pred)
      FP = np.multiply(1 - y_true, y_pred)
      FN = np.multiply(y_true, 1 - y_pred)
      precision = 1.0 * np.sum(TP) / (np.sum(FP) + np.sum(TP))
      recall = 1.0 * np.sum(TP) / (np.sum(TP) + np.sum(FN))
      f1_micro += (2.0 * precision * recall / (precision + recall))

      for idx in range(y_true.shape[1]):
        y_true_ = y_true[:, idx]
        y_pred_ = y_pred[:, idx]
        tp = np.multiply(y_true_, y_pred_)
        fp = np.multiply(1 - y_true_, y_pred_)
        fn = np.multiply(y_true_, 1 - y_pred_)
        precision_ = 1.0 * np.sum(tp) / (np.sum(fp) + np.sum(tp))
        recall_ = 1.0 * np.sum(tp) / (np.sum(tp) + np.sum(fn))
        f1_macro += (2.0 * precision_ * recall_ / (precision_ + recall_))

      '''
      for threshold in np.linspace(0.0, 1.0, num=calculusN, endpoint=False):
        y_pred = np.where(y_score >= threshold, 1, 0)
        TP = np.multiply(y_true, y_pred)
        FP = np.multiply(1 - y_true, y_pred)
        FN = np.multiply(y_true, 1 - y_pred)
        precision = 1.0 * np.sum(TP) / (np.sum(FP) + np.sum(TP))
        recall = 1.0 * np.sum(TP) / (np.sum(TP) + np.sum(FN))
        f1_micro += (2.0 * precision * recall / (precision + recall) / calculusN)

        for idx in range(y_true.shape[1]):
          y_true_ = y_true[:, idx]
          y_pred_ = y_pred[:, idx]
          tp = np.multiply(y_true_, y_pred_)
          fp = np.multiply(1 - y_true_, y_pred_)
          fn = np.multiply(y_true_, 1 - y_pred_)
          precision_ = 1.0 * np.sum(tp) / (np.sum(fp) + np.sum(tp))
          recall_ = 1.0 * np.sum(tp) / (np.sum(tp) + np.sum(fn))
          f1_macro += (2.0 * precision_ * recall_ / (precision_ + recall_) / calculusN)
      '''

      f1_macro /= model.classes_num
      print("f1 micro score: ", f1_micro)
      print("f1 macro score: ", f1_macro)

      TPR = []
      FPR = []
      for threshold_ in np.linspace(0.0, 1.0, num=calculusN, endpoint=False):
        tmp_y_pred = np.where(y_score >= threshold_, 1, 0)
        TP = np.multiply(y_true, tmp_y_pred)
        FP = np.multiply(1 - y_true, tmp_y_pred)
        FN = np.multiply(y_true, 1 - tmp_y_pred)
        TN = np.multiply(1 - y_true, 1 - tmp_y_pred)
        TPR.append(1.0 * np.sum(TP) / (np.sum(FN) + np.sum(TP)))
        FPR.append(1.0 * np.sum(FP) / (np.sum(FP) + np.sum(TN)))

      auc_score = auc(FPR, TPR)
      print("AUC score: ", auc_score)

      sk_coverage = coverage_error(y_true, y_score)
      print("sklearn coverage score: ", sk_coverage)
      sk_ranking_loss = label_ranking_loss(y_true, y_score)
      print("sklearn ranking loss: ", sk_ranking_loss)


      max_idx = np.argmax(y_score, axis=1)
      max_list_idx = zip(range(len(max_idx)), max_idx)
      max_list = []
      for ele in max_list_idx:
        # print(ele)
        max_list.append(y_true[ele[0], ele[1]])

      one_error = np.sum(max_list) / data_total_num
      print("One error: ", one_error)

      result = {'f1 micro': f1_micro,
                'f1 macro': f1_macro,
                'sk f1 micro': sk_f1_micro,
                'sk f1 macro': sk_f1_macro,
                'coverage': sk_coverage,
                'ranking loss': sk_ranking_loss,
                'one error': one_error,
               # 'sk auc': sk_auc,
                'auc': auc_score}

      with open(SAVE_RESULT_PATH, 'wb') as f:
        pickle.dump(result, f, True)

      print(result)


if __name__ == '__main__':
    main()




