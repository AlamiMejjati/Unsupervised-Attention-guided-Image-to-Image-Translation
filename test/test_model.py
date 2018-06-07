import tensorflow as tf

from dl_research.testing import slow

from .. import model

# -----------------------------------------------------------------------------


@slow
def test_output_sizes(sess):
    images_size = [
        model.BATCH_SIZE,
        model.IMG_HEIGHT,
        model.IMG_WIDTH,
        model.IMG_CHANNELS,
    ]

    pool_size = [
        model.POOL_SIZE,
        model.IMG_HEIGHT,
        model.IMG_WIDTH,
        model.IMG_CHANNELS,
    ]

    inputs = {
        'images_a': tf.ones(images_size),
        'images_b': tf.ones(images_size),
        'fake_pool_a': tf.ones(pool_size),
        'fake_pool_b': tf.ones(pool_size),
    }

    outputs = model.get_outputs(inputs)

    assert outputs['prob_real_a_is_real'].get_shape().as_list() == [
        model.BATCH_SIZE, 32, 32, 1,
    ]
    assert outputs['prob_real_b_is_real'].get_shape().as_list() == [
        model.BATCH_SIZE, 32, 32, 1,
    ]
    assert outputs['prob_fake_a_is_real'].get_shape().as_list() == [
        model.BATCH_SIZE, 32, 32, 1,
    ]
    assert outputs['prob_fake_b_is_real'].get_shape().as_list() == [
        model.BATCH_SIZE, 32, 32, 1,
    ]
    assert outputs['prob_fake_pool_a_is_real'].get_shape().as_list() == [
        model.POOL_SIZE, 32, 32, 1,
    ]
    assert outputs['prob_fake_pool_b_is_real'].get_shape().as_list() == [
        model.POOL_SIZE, 32, 32, 1,
    ]
    assert outputs['cycle_images_a'].get_shape().as_list() == images_size
    assert outputs['cycle_images_b'].get_shape().as_list() == images_size
