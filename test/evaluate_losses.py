import numpy as np
import tensorflow as tf

from .. import losses


def test_evaluate_g_losses(sess):

    _LAMBDA_A = 10
    _LAMBDA_B = 10

    input_a = tf.random_uniform((5, 7), maxval=1)
    cycle_images_a = input_a + 1
    input_b = tf.random_uniform((5, 7), maxval=1)
    cycle_images_b = input_b - 2

    cycle_consistency_loss_a = _LAMBDA_A * losses.cycle_consistency_loss(
        real_images=input_a, generated_images=cycle_images_a,
    )
    cycle_consistency_loss_b = _LAMBDA_B * losses.cycle_consistency_loss(
        real_images=input_b, generated_images=cycle_images_b,
    )

    prob_fake_a_is_real = tf.constant([0, 1.0, 0])
    prob_fake_b_is_real = tf.constant([1.0, 1.0, 0])

    lsgan_loss_a = losses.lsgan_loss_generator(prob_fake_a_is_real)
    lsgan_loss_b = losses.lsgan_loss_generator(prob_fake_b_is_real)

    assert np.isclose(sess.run(lsgan_loss_a), 0.66666669) and \
        np.isclose(sess.run(lsgan_loss_b), 0.3333333) and \
        np.isclose(sess.run(cycle_consistency_loss_a), 10) and \
        np.isclose(sess.run(cycle_consistency_loss_b), 20)


def test_evaluate_d_losses(sess):

    prob_real_a_is_real = tf.constant([1.0, 1.0, 0])
    prob_fake_pool_a_is_real = tf.constant([1.0, 0, 0])
    d_loss_A = losses.lsgan_loss_discriminator(
        prob_real_is_real=prob_real_a_is_real,
        prob_fake_is_real=prob_fake_pool_a_is_real)
    assert np.isclose(sess.run(d_loss_A), 0.3333333)
