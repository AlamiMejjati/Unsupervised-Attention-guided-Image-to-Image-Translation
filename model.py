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


def get_outputs(inputs, network="tensorflow", skip=False):

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
        current_discriminator = discriminator_bis
        current_generator = build_generator_resnet_9blocks_bis

        mask_a = current_autoenc(images_a, "g_A_ae")
        mask_b = current_autoenc(images_b, "g_B_ae")
        mask_a = tf.concat([mask_a] * 3, axis=3)
        mask_b = tf.concat([mask_b] * 3, axis=3)

        mask_a_on_a = tf.multiply(images_a, mask_a)
        mask_b_on_b = tf.multiply(images_b, mask_b)

        prob_real_a_is_real = current_discriminator(images_a, mask_a, transition_rate, donorm, "d_A")
        prob_real_b_is_real = current_discriminator(images_b, mask_b, transition_rate, donorm, "d_B")

        fake_images_b_from_g = current_generator(mask_a_on_a, mask_a, transition_rate, name="g_A", skip=skip)
        fake_images_b = tf.multiply(fake_images_b_from_g, mask_a) + tf.multiply(images_a, 1-mask_a)

        fake_images_a_from_g = current_generator(mask_b_on_b, mask_b, transition_rate, name="g_B", skip=skip)
        fake_images_a = tf.multiply(fake_images_a_from_g, mask_b) + tf.multiply(images_b, 1-mask_b)
        scope.reuse_variables()

        prob_fake_a_is_real= current_discriminator(fake_images_a, mask_b, transition_rate, donorm, "d_A")
        prob_fake_b_is_real= current_discriminator(fake_images_b, mask_a, transition_rate, donorm, "d_B")

        mask_acycle = current_autoenc(fake_images_a, "g_A_ae")
        mask_bcycle = current_autoenc(fake_images_b, "g_B_ae")
        mask_bcycle = tf.concat([mask_bcycle] * 3, axis=3)
        mask_acycle = tf.concat([mask_acycle] * 3, axis=3)

        mask_acycle_on_fakeA = tf.multiply(fake_images_a, mask_acycle)
        mask_bcycle_on_fakeB = tf.multiply(fake_images_b, mask_bcycle)

        cycle_images_a_from_g = current_generator(mask_bcycle_on_fakeB, mask_bcycle, transition_rate, "g_B",
                                                     skip=skip)
        cycle_images_b_from_g = current_generator(mask_acycle_on_fakeA, mask_acycle, transition_rate,"g_A",
                                                     skip=skip)

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
            pad_input, ngf, f, f, 2, 2, 0.02, name="c1")
        o_c2 = layers.general_conv2d(
            o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")

        o_r1 = build_resnet_block(o_c2, ngf * 2, "r1", padding)

        size_d1 = o_r1.get_shape().as_list()
        o_c4 = layers.upsamplingDeconv(o_r1, size=[size_d1[1] * 2, size_d1[2] * 2], is_scale=False, method=1,
                                   align_corners=False,name= 'up1')
        # o_c4_pad = tf.pad(o_c4, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='padup1')
        o_c4_end = layers.general_conv2d(o_c4, ngf * 2, (3, 3), (1, 1), padding='VALID', name='c4')

        size_d2 = o_c4_end.get_shape().as_list()
        o_c5 = layers.upsamplingDeconv(o_c4_end, size=[size_d2[1] * 2, size_d2[2] * 2], is_scale=False, method=1,
                                       align_corners=False, name='up2')
        # o_c5_pad = tf.pad(o_c5, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='padup2')
        oc5_end = layers.general_conv2d(o_c5, ngf , (3, 3), (1, 1), padding='VALID', name='c5')

        # o_c6 = tf.pad(oc5_end, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='padup3')
        o_c6_end = layers.general_conv2d(oc5_end, 1 , (f, f), (1, 1), padding='VALID', name='c6',
                                         do_norm=False, do_relu=False)

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
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1", do_norm = False)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_norm = False, do_relu=False)

        return tf.nn.relu(out_res + inputres)


def build_partial_resnet_block(inputres, mask_in, dim, name="resnet"):
    """build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    padding = "CONSTANT"
    with tf.variable_scope(name):
        inputres_ = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        mask_in_ = tf.pad(mask_in, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)

        out_res, mask = layers.general_partial_conv2d(
            inputres_, mask_in_, tf.constant(False, dtype=bool), dim, 3, 3, 1, 1, 0.02, name="c1")

        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res_in = tf.multiply(out_res, mask)

        out_res, mask = layers.general_partial_conv2d(
            out_res_in, mask, tf.constant(False, dtype=bool), dim, 3, 3, 1, 1, 0.02, name="c2", do_relu=False)

        out_res_in = tf.multiply(out_res, mask)
        mask_end = tf.cast(tf.greater(mask + mask_in , 0.), tf.float32)
        return tf.nn.relu(out_res_in + inputres), mask_end

def build_generator_resnet_9blocks_bis(inputgen, mask, transition_rate, name="generator", skip=False):

    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"
        inputgen = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        mask = tf.pad(mask, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)

        o_c1, mask = layers.general_partial_conv2d(
            inputgen, mask, tf.constant(False, dtype=bool), ngf, f, f, 1, 1, 0.02, name="c1")

        # o_c1_in = tf.multiply(o_c1, mask)

        o_c2, mask = layers.general_partial_conv2d(
            o_c1, mask, tf.constant(False, dtype=bool),ngf * 2, ks, ks, 2, 2, 0.02, padding='same', name="c2")

        # o_c2_in = tf.multiply(o_c2, mask)

        o_c3, mask = layers.general_partial_conv2d(
            o_c2, mask, tf.constant(False, dtype=bool), ngf * 4, ks, ks, 2, 2, 0.02, padding='same', name="c3")

        # o_c3_in = tf.multiply(o_c3, mask)

        o_r1, mask_r1 = build_partial_resnet_block(o_c3, mask, ngf * 4, "r1")
        o_r2, mask_r2 = build_partial_resnet_block(o_r1, mask_r1, ngf * 4, "r2")
        o_r3, mask_r3 = build_partial_resnet_block(o_r2, mask_r2, ngf * 4, "r3")
        o_r4, mask_r4 = build_partial_resnet_block(o_r3, mask_r3, ngf * 4, "r4")
        o_r5, mask_r5 = build_partial_resnet_block(o_r4, mask_r4, ngf * 4, "r5")
        o_r6, mask_r6 = build_partial_resnet_block(o_r5, mask_r5, ngf * 4, "r6")
        o_r7, mask_r7 = build_partial_resnet_block(o_r6, mask_r6, ngf * 4, "r7")
        o_r8, mask_r8 = build_partial_resnet_block(o_r7, mask_r7, ngf * 4, "r8")
        o_r9, mask_r9 = build_partial_resnet_block(o_r8, mask_r8, ngf * 4, "r9")

        o_c4 = layers.general_deconv2d(
            o_r9, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
            "SAME", "c4")
        o_c5 = layers.general_deconv2d(
            o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,
            "SAME", "c5")
        o_c6 = layers.general_conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c6",
                                     do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen

def build_generator_resnet_9blocks(inputgen, mask, transition_rate, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d(
            pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
        o_c2 = layers.general_conv2d(
            o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        o_c3 = layers.general_conv2d(
            o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

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
        o_c6 = layers.general_conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c6",
                                     do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen

def discriminator(inputdisc,  mask, transition_rate,  name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        padw = 2

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2,
                                     0.02, "VALID", "c1", do_norm=False,
                                     relufactor=0.2)

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "VALID", "c2", relufactor=0.2)

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "VALID", "c3", relufactor=0.2)

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1,
                                     0.02, "VALID", "c4", relufactor=0.2)

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(
            pad_o_c4, 1, f, f, 1, 1, 0.02, "VALID", "c5",
            do_norm=False, do_relu=False)

        return o_c5

def discriminator_bis(inputdisc,  mask, transition_rate, donorm,  name="discriminator"):

    with tf.variable_scope(name):
        mask = tf.cast(tf.greater_equal(mask, transition_rate), tf.float32)
        inputdisc = tf.multiply(inputdisc, mask)
        f = 4
        padw = 2
        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        pad_mask = tf.pad(mask, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c1, mask_1 = layers.general_partial_conv2d(pad_input, pad_mask, donorm, ndf, f, f, 2, 2,
                                     0.02, "VALID", "c1",
                                     relufactor=0.2)

        # o_c1 = tf.multiply(o_c1, mask_1)
        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        pad_mask = tf.pad(mask_1, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c2, mask_2 = layers.general_partial_conv2d(pad_o_c1, pad_mask, donorm, ndf * 2, f, f, 2, 2,
                                     0.02, "VALID", "c2",  relufactor=0.2)

        # o_c2 = tf.multiply(o_c2, mask_2)
        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        pad_mask = tf.pad(mask_2, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c3, mask_3 = layers.general_partial_conv2d(pad_o_c2, pad_mask, donorm ,ndf * 4, f, f, 2, 2,
                                     0.02, "VALID", "c3", relufactor=0.2)
        # o_c3 = tf.multiply(o_c3, mask_3)
        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        pad_mask = tf.pad(mask_3, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c4, mask_4 = layers.general_partial_conv2d(pad_o_c3, pad_mask, donorm, ndf * 8, f, f, 1, 1,
                                     0.02, "VALID", "c4", relufactor=0.2)
        # o_c4 = tf.multiply(o_c4, mask_4)
        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        pad_mask = tf.pad(mask_4, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")

        o_c5, mask_5 = layers.general_partial_conv2d(
            pad_o_c4, pad_mask, tf.constant(False, dtype=bool), 1, f, f, 1, 1, 0.02, "VALID", "c5", do_relu=False)


        return o_c5
