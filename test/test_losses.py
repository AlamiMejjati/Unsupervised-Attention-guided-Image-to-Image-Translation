import numpy as np
import tensorflow as tf

from .. import losses


def test_cycle_consistency_loss_is_none_with_perfect_fakes(sess):
    batch_size, height, width, channels = [16, 2, 3, 1]

    tf.set_random_seed(0)

    images = tf.random_uniform((batch_size, height, width, channels), maxval=1)

    loss = losses.cycle_consistency_loss(
        real_images=images,
        generated_images=images,
    )

    assert sess.run(loss) == 0


def test_cycle_consistency_loss_is_positive_with_imperfect_fake_x(sess):
    batch_size, height, width, channels = [16, 2, 3, 1]

    tf.set_random_seed(0)

    real_images = tf.random_uniform(
        (batch_size, height, width, channels), maxval=1,
    )
    generated_images = real_images + 1

    loss = losses.cycle_consistency_loss(
        real_images=real_images,
        generated_images=generated_images,
    )

    assert sess.run(loss) == 1


def test_lsgan_loss_discrim_is_none_with_perfect_discrimination(sess):
    batch_size = 100
    prob_real_is_real = tf.ones((batch_size))
    prob_fake_is_real = tf.zeros((batch_size))
    loss = losses.lsgan_loss_discriminator(
        prob_real_is_real, prob_fake_is_real,
    )
    assert sess.run(loss) == 0


def test_lsgan_loss_discrim_is_positive_with_imperfect_discrimination(sess):
    batch_size = 100
    prob_real_is_real = tf.ones((batch_size)) * 0.4
    prob_fake_is_real = tf.ones((batch_size)) * 0.7
    loss = losses.lsgan_loss_discriminator(
        prob_real_is_real, prob_fake_is_real,
    )
    loss = sess.run(loss)

    np.testing.assert_almost_equal(loss, (0.6 * 0.6 + 0.7 * 0.7) / 2)
