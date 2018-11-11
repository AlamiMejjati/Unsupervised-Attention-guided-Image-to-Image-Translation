import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out

def instance_norm_bis(x,mask):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        for i in range(x.shape[-1]):
            slice = tf.gather(x, i, axis=3)
            slice_mask = tf.gather(mask, i, axis=3)
            tmp = tf.boolean_mask(slice,slice_mask)
            mean, var = tf.nn.moments_bis(x, [1, 2], keep_dims=False)

        mean, var = tf.nn.moments_bis(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def general_conv2d_(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv

def general_conv2d(inputconv, do_norm, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_relu=True,
                   relufactor=0):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=tf.constant_initializer(0.0)
        )

        conv = tf.cond(do_norm, lambda: instance_norm(conv), lambda: conv)


        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")


        return conv

def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(
            inputconv, o_d, [f_h, f_w],
            [s_h, s_w], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv

def upsamplingDeconv(inputconv, size, is_scale, method,align_corners, name):
    if len(inputconv.get_shape()) == 3:
        if is_scale:
            size_h = size[0] * int(inputconv.get_shape()[0])
            size_w = size[1] * int(inputconv.get_shape()[1])
            size = [int(size_h), int(size_w)]
    elif len(inputconv.get_shape()) == 4:
        if is_scale:
            size_h = size[0] * int(inputconv.get_shape()[1])
            size_w = size[1] * int(inputconv.get_shape()[2])
            size = [int(size_h), int(size_w)]
    else:
        raise Exception("Donot support shape %s" % inputconv.get_shape())
    print("  [TL] UpSampling2dLayer %s: is_scale:%s size:%s method:%d align_corners:%s" %
          (name, is_scale, size, method, align_corners))
    with tf.variable_scope(name) as vs:
        try:
            out = tf.image.resize_images(inputconv, size=size, method=method, align_corners=align_corners)
        except:  # for TF 0.10
            out = tf.image.resize_images(inputconv, new_height=size[0], new_width=size[1], method=method,
                                                  align_corners=align_corners)
    return out

def general_fc_layers(inpfc, outshape, name):
    with tf.variable_scope(name):

        fcw = tf.Variable(tf.truncated_normal(outshape,
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fcb = tf.Variable(tf.constant(1.0, shape=[outshape[-1]], dtype=tf.float32),
                           trainable=True, name='biases')

        fcl = tf.nn.bias_add(tf.matmul(inpfc, fcw), fcb)
        fc_out = tf.nn.relu(fcl)

        return fc_out
