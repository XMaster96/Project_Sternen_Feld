import tensorflow as tf
import numpy as np

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer



def Policy_gradient_output(backend_output, action_size, policy_normalized_columns_initializer_value=0.01, value_normalized_columns_initializer_value=1):

    policy = tf.layers.Dense(inputs=backend_output, units=action_size, activation=tf.nn.softmax, weights_initializer=normalized_columns_initializer(policy_normalized_columns_initializer_value), biases_initializer=None)
    value = tf.layers.Dense(inputs=backend_output, units=1, activation=None, weights_initializer=normalized_columns_initializer(value_normalized_columns_initializer_value), biases_initializer=None)

    return policy, value











