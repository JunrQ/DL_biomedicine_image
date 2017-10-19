

import tensorflow as tf

import skimage.transform, skimage.color
import numpy as np

def np_image_random(image,
                    rotate_angle=[-10, 10],
                    rescale=[0.8, 1.2],
                    whiten=False,
                    normalize=True,
                    hsv=True
                    ):
    """
    image should be a ndarray, dtype float
    """
    o_shape = image.shape
    if rotate_angle is not None:
        rotate_angle = np.random.random() * (rotate_angle[1] - rotate_angle[0]) + rotate_angle[0]
        image = skimage.transform.rotate(image, rotate_angle, mode='edge')
    if rescale is not None:
        rescale = np.random.random() * (rescale[1] - rescale[0]) + rescale[0]
        image = skimage.transform.rescale(image, scale=rescale, mode='edge')

    if whiten:
        image -= np.mean(image, axis=0)
        cov = np.dot(image.T, image) / image.shape[0]
        U, S, V = np.linalg.svd(cov)
        Xrot = np.dot(image, U)
        image = Xrot / np.sqrt(S + 1e-5)
    if normalize:
        image = image - np.mean(image, axis=0)
        image = image / np.std(image, axis=0)

        image = (image - np.min(image, axis=0)) / (np.max(image, axis=0) - np.min(image, axis=0))

    if hsv:
      image = skimage.color.rgb2hsv(image)

    return image




def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_image(image, height, width, bbox, thread_id=0, scope=None):
    """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
    with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):

        # NOTE(ry) I unceremoniously removed all the bounding box code.
        # Original here: https://github.com/tensorflow/models/blob/148a15fb043dacdd1595eb4c5267705fbd362c6a/inception/inception/image_processing.py

        distorted_image = image

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(distorted_image, height,
                                                 width, resize_method)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.image_summary('cropped_resized_image',
                             tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)

        if not thread_id:
            tf.image_summary('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        return distorted_image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
    with tf.op_scope([image, height, width], scope, 'eval_image'):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        return image


def read_image(image_buffer, decoder, concatenate_input, height, width, train, concatnate_way='col'):
    """Decode and preprocess one image for evaluation or training.

  Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """

    image = decoder(image_buffer)
    if concatenate_input:
      return image

    image_shape = image.get_shape()
    if concatnate_way == 'col':
      image_group = tf.reshape(image, shape=(image_shape[0] / height, height, width))
    else:
      raise ValueError("Wrong concatnate_way! Original images should cancatante through cols")
    # if train:

    # Finally, rescale to [-1,1] instead of [0, 1)
    # image = tf.sub(image, 0.5)
    # image = tf.mul(image, 2.0)

    return image_group


