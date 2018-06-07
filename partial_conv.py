import tensorflow as tf

"""Arguments
   tensor: Tensor input.
   binary_mask: Tensor, a mask with the same size as tensor, channel size = 1
   filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
   kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
   strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
   l2_scale: float, A scalar multiplier Tensor. 0.0 disables the regularizer.

 Returns:
   Output tensor, binary mask.
 """


def sparse_conv(tensor, binary_mask=None, filters=32, kernel_size=3, strides=2, l2_scale=0.0):
    if binary_mask == None:  # first layer has no binary mask
        b, h, w, c = tensor.get_shape()
        channels = tf.split(tensor, c, axis=3)
        # assume that if one channel has no information, all channels have no information
        binary_mask = tf.where(tf.equal(channels[0], 0), tf.zeros_like(channels[0]),
                               tf.ones_like(channels[0]))  # mask should only have the size of (B,H,W,1)

    features = tf.multiply(tensor, binary_mask)
    features = tf.layers.conv2d(features, filters=filters, kernel_size=kernel_size, strides=(strides, strides),
                                trainable=True, use_bias=False, padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale))

    norm = tf.layers.conv2d(binary_mask, filters=filters, kernel_size=kernel_size, strides=(strides, strides),
                            kernel_initializer=tf.ones_initializer(), trainable=False, use_bias=False, padding="same")
    norm = tf.where(tf.equal(norm, 0), tf.zeros_like(norm), tf.reciprocal(norm))
    _, _, _, bias_size = norm.get_shape()

    b = tf.Variable(tf.constant(0.0, shape=[bias_size]), trainable=True)
    feature = tf.multiply(features, norm) + b
    mask = tf.layers.max_pooling2d(binary_mask, strides=strides, pool_size=3, padding="same")

    return feature, mask


image = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name="input_image")
b_mask = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="binary_mask")
features, b_mask = sparse_conv(image)
features, b_mask = sparse_conv(features, binary_mask=b_mask)

sess = tf.Session()
sess.run(tf.global_variables_initializer())