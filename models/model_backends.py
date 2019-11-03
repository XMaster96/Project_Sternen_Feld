import tensorflow as tf
import numpy as np

def Conv3D_normal_LSTM(input_shape, lstm_size=128, orthogonal_initializer_gain=1):

    input = tf.placeholder(shape=input_shape, dtype=tf.float32)

    conv1 = tf.layers.conv2d(inputs=input, filters=12, kernel_size=(7, 7), padding='valid', activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(gain=orthogonal_initializer_gain))
    conv2 = tf.layers.conv2d(inputs=conv1, filters=22, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(gain=orthogonal_initializer_gain))
    conv3 = tf.layers.conv2d(inputs=conv2, filters=44, kernel_size=(3, 3), padding='valid', activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(gain=orthogonal_initializer_gain))


    flatt = tf.layers.flatten(conv3)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)

    #
    init_state_c = np.zeros((1, lstm_cell.state_size.c), np.float32)
    init_state_h = np.zeros((1, lstm_cell.state_size.h), np.float32)

    # List of inital LSTM stats.
    state_init = [init_state_c, init_state_h]

    # LSTM state placeholder.
    c_1_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
    h_1_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])

    # Tumbel for LSTM placeholder.
    state_in = (c_1_in, h_1_in)

    # Creat initial_state tumbelt for LSTM executen.
    state_1_in = tf.contrib.rnn.LSTMStateTuple(c_1_in, h_1_in)

    rnn_in = tf.expand_dims(flatt, [0])
    #inpur_len = tf.shape(input)[:1]

    # Execute LSTM.
    lstm_1_outputs, lstm_1_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=rnn_in, initial_state=state_1_in)

    lstm_1_c, lstm_1_h = lstm_1_state

    state_out = (lstm_1_c[:1, :], lstm_1_h[:1, :])

    lstm_out = tf.reshape(lstm_1_outputs, [-1, lstm_size])




    return [input, state_in], [state_out, state_init], lstm_out




