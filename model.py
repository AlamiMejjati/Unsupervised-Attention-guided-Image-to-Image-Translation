"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf
import numpy as np
import layers

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

# The number of color channels per image.
IMG_CHANNELS = 3

POOL_SIZE = 50
ngf = 32
ndf = 64


def get_outputs(inputs, skip=False):

    images_a = inputs['images_a']
    images_b = inputs['images_b']
    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']
    fake_pool_a_mask = inputs['fake_pool_a_mask']
    fake_pool_b_mask = inputs['fake_pool_b_mask']
    transition_rate = inputs['transition_rate']
    donorm = inputs['donorm']
    with tf.variable_scope("Model") as scope:

        current_autoenc = autoenc_upsample
        current_discriminator = discriminator
        current_generator = build_generator_resnet_9blocks

        mask_a = current_autoenc(images_a, "g_A_ae")
        mask_b = current_autoenc(images_b, "g_B_ae")
        mask_a = tf.concat([mask_a] * 3, axis=3)
        mask_b = tf.concat([mask_b] * 3, axis=3)

        mask_a_on_a = tf.multiply(images_a, mask_a)
        mask_b_on_b = tf.multiply(images_b, mask_b)

        prob_real_a_is_real = current_discriminator(images_a, mask_a, transition_rate, donorm, "d_A")
        prob_real_b_is_real = current_discriminator(images_b, mask_b, transition_rate, donorm, "d_B")

        fake_images_b_from_g = current_generator(images_a, name="g_A", skip=skip)
        fake_images_b = tf.multiply(fake_images_b_from_g, mask_a) + tf.multiply(images_a, 1-mask_a)

        fake_images_a_from_g = current_generator(images_b, name="g_B", skip=skip)
        fake_images_a = tf.multiply(fake_images_a_from_g, mask_b) + tf.multiply(images_b, 1-mask_b)
        scope.reuse_variables()

        prob_fake_a_is_real = current_discriminator(fake_images_a, mask_b, transition_rate, donorm, "d_A")
        prob_fake_b_is_real = current_discriminator(fake_images_b, mask_a, transition_rate, donorm, "d_B")

        mask_acycle = current_autoenc(fake_images_a, "g_A_ae")
        mask_bcycle = current_autoenc(fake_images_b, "g_B_ae")
        mask_bcycle = tf.concat([mask_bcycle] * 3, axis=3)
        mask_acycle = tf.concat([mask_acycle] * 3, axis=3)

        mask_acycle_on_fakeA = tf.multiply(fake_images_a, mask_acycle)
        mask_bcycle_on_fakeB = tf.multiply(fake_images_b, mask_bcycle)

        cycle_images_a_from_g = current_generator(fake_images_b, name="g_B", skip=skip)
        cycle_images_b_from_g = current_generator(fake_images_a, name="g_A", skip=skip)

        cycle_images_a = tf.multiply(cycle_images_a_from_g,
                                     mask_bcycle) + tf.multiply(fake_images_b, 1 - mask_bcycle)

        cycle_images_b = tf.multiply(cycle_images_b_from_g,
                                     mask_acycle) + tf.multiply(fake_images_a, 1 - mask_acycle)

        scope.reuse_variables()

        prob_fake_pool_a_is_real = current_discriminator(fake_pool_a, fake_pool_a_mask, transition_rate, donorm, "d_A")
        prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, fake_pool_b_mask, transition_rate, donorm, "d_B")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
        'masked_ims': [mask_a_on_a, mask_b_on_b, mask_acycle_on_fakeA, mask_bcycle_on_fakeB],
        'masks': [mask_a, mask_b, mask_acycle, mask_bcycle],
        'masked_gen_ims' : [fake_images_b_from_g, fake_images_a_from_g , cycle_images_a_from_g, cycle_images_b_from_g],
        'mask_tmp' : mask_a,
    }

def autoenc_upsample(inputae, name):

    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "REFLECT"

        pad_input = tf.pad(inputae, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d(
            pad_input, tf.constant(True, dtype=bool), ngf, f, f, 2, 2, 0.02, name="c1")
        o_c2 = layers.general_conv2d(
            o_c1, tf.constant(True, dtype=bool), ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")

        o_r1 = build_resnet_block_Att(o_c2, ngf * 2, "r1", padding)

        size_d1 = o_r1.get_shape().as_list()
        o_c4 = layers.upsamplingDeconv(o_r1, size=[size_d1[1] * 2, size_d1[2] * 2], is_scale=False, method=1,
                                   align_corners=False,name= 'up1')
        # o_c4_pad = tf.pad(o_c4, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='padup1')
        o_c4_end = layers.general_conv2d(o_c4, tf.constant(True, dtype=bool), ngf * 2, (3, 3), (1, 1), padding='VALID', name='c4')

        size_d2 = o_c4_end.get_shape().as_list()
        o_c5 = layers.upsamplingDeconv(o_c4_end, size=[size_d2[1] * 2, size_d2[2] * 2], is_scale=False, method=1,
                                       align_corners=False, name='up2')
        # o_c5_pad = tf.pad(o_c5, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='padup2')
        oc5_end = layers.general_conv2d(o_c5, tf.constant(True, dtype=bool), ngf , (3, 3), (1, 1), padding='VALID', name='c5')

        # o_c6 = tf.pad(oc5_end, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='padup3')
        o_c6_end = layers.general_conv2d(oc5_end, tf.constant(False, dtype=bool),
                                         1 , (f, f), (1, 1), padding='VALID', name='c6', do_relu=False)

        return tf.nn.sigmoid(o_c6_end,'sigmoid')

def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    """build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)

def build_resnet_block_Att(inputres, dim, name="resnet", padding="REFLECT"):
    """build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)

def build_generator_resnet_9blocks(inputgen, name="generator", skip=False):

    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"
        inputgen = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)

        o_c1 = layers.general_conv2d(
            inputgen, tf.constant(True, dtype=bool), ngf, f, f, 1, 1, 0.02, name="c1")

        o_c2 = layers.general_conv2d(
            o_c1, tf.constant(True, dtype=bool),ngf * 2, ks, ks, 2, 2, 0.02, padding='same', name="c2")

        o_c3 = layers.general_conv2d(
            o_c2, tf.constant(True, dtype=bool), ngf * 4, ks, ks, 2, 2, 0.02, padding='same', name="c3")


        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)

        o_c4 = layers.general_deconv2d(
            o_r9, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
            "SAME", "c4")

        o_c5 = layers.general_deconv2d(
            o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,
            "SAME", "c5")

        o_c6 = layers.general_conv2d(o_c5, tf.constant(False, dtype=bool), IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c6", do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen

def discriminator(inputdisc,  mask, transition_rate, donorm,  name="discriminator"):

    with tf.variable_scope(name):
        mask = tf.cast(tf.greater_equal(mask, transition_rate), tf.float32)
        inputdisc = tf.multiply(inputdisc, mask)
        f = 4
        padw = 2
        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c1 = layers.general_conv2d(pad_input, donorm, ndf, f, f, 2, 2,
                                     0.02, "VALID", "c1",
                                     relufactor=0.2)

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c2 = layers.general_conv2d(pad_o_c1, donorm, ndf * 2, f, f, 2, 2,
                                     0.02, "VALID", "c2",  relufactor=0.2)

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c3 = layers.general_conv2d(pad_o_c2, donorm, ndf * 4, f, f, 2, 2,
                                     0.02, "VALID", "c3", relufactor=0.2)

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c4 = layers.general_conv2d(pad_o_c3, donorm, ndf * 8, f, f, 1, 1,
                                     0.02, "VALID", "c4", relufactor=0.2)
        # o_c4 = tf.multiply(o_c4, mask_4)
        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c5 = layers.general_conv2d(
            pad_o_c4, tf.constant(False, dtype=bool), 1, f, f, 1, 1, 0.02, "VALID", "c5", do_relu=False)


        return o_c5
