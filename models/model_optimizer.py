import tensorflow as tf
import numpy as np


def A2C(policy_layer, value_layer, scope, action_size, entropy_coefficient, value_coefficient, max_grad_norm):


    actions = tf.placeholder(tf.int32, [None], name="actions")
    advantages = tf.placeholder(tf.float32, [None], name="advantages")
    rewards = tf.placeholder(tf.float32, [None], name="rewards")
    lr = tf.placeholder(tf.float32, name="learning_rate")

    actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)

    # Policy loss
    # Output -log(pi)
    neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_layer, labels=actions_onehot)

    # 1/n * sum A(si,ai) * -logpi(ai|si)
    policy_loss = tf.reduce_mean(advantages * neglogpac)

    # Value loss 1/2 SUM [R - V(s)]^2
    value_loss = 0.5 * tf.reduce_sum(tf.square(rewards - tf.reshape(value_layer, [-1])))

    # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
    entropy = tf.reduce_mean(actions_onehot.entropy())


    loss = policy_loss - entropy * entropy_coefficient + value_loss * value_coefficient

    # Update parameters using loss
    # 1. Get the model parameters
    # in Keras params = model.output
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    # 2. Calculate the gradients
    grads = tf.gradients(loss, params)
    if max_grad_norm is not None:
        # Clip the gradients (normalize)
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    # zip aggregate each gradient with parameters associated
    # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

    # 3. Build our trainer
    trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)

    # 4. Backpropagation
    _train = trainer.apply_gradients(grads)


    return [actions, advantages, rewards, lr], [policy_loss, value_loss, entropy, _train]






