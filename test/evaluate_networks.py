import numpy as np
import tensorflow as tf

from .. import model


def test_evaluate_g(sess):
    x_val = np.ones_like(np.random.randn(1, 16, 16, 3)).astype(np.float32)
    for i in range(16):
        for j in range(16):
            for k in range(3):
                x_val[0][i][j][k] = ((i + j + k) % 2) / 2.0
    inputs = {
        'images_a': tf.stack(x_val),
        'images_b': tf.stack(x_val),
        'fake_pool_a': tf.zeros([1, 16, 16, 3]),
        'fake_pool_b': tf.zeros([1, 16, 16, 3]),
    }

    outputs = model.get_outputs(inputs)

    sess.run(tf.global_variables_initializer())
    assert sess.run(outputs['fake_images_a'][0][5][7][0]) == 5


def test_evaluate_d(sess):
    x_val = np.ones_like(np.random.randn(1, 16, 16, 3)).astype(np.float32)
    for i in range(16):
        for j in range(16):
            for k in range(3):
                x_val[0][i][j][k] = ((i + j + k) % 2) / 2.0
    inputs = {
        'images_a': tf.stack(x_val),
        'images_b': tf.stack(x_val),
        'fake_pool_a': tf.zeros([1, 16, 16, 3]),
        'fake_pool_b': tf.zeros([1, 16, 16, 3]),
    }

    outputs = model.get_outputs(inputs)

    sess.run(tf.global_variables_initializer())
    assert sess.run(outputs['prob_real_a_is_real'][0][3][3][0]) == 5
