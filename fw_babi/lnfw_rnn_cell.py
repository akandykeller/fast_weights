"""
Fast Weights Cell.
Ba et al. Using Fast Weights to Attend to the Recent Past
https://arxiv.org/abs/1610.06258
"""

from tensorflow.contrib.rnn import RNNCell
import tensorflow as tf
import numpy as np

class FastWeightsRNNCell(RNNCell):

    def __init__(self, num_hidden_units, batch_size, loop_steps=1, 
                 decay_rate=0.95, eta=0.5, dropout_keep_prob=1.0):
        super(FastWeightsRNNCell, self).__init__()
        self._num_hidden_units = num_hidden_units
        self._keep_prob = dropout_keep_prob
        self._batch_size = batch_size
        self._S = loop_steps
        self._e = eta
        self._l = decay_rate

    @property
    def state_size(self):
        return self._num_hidden_units

    @property
    def output_size(self):
        return self._num_hidden_units

    def zero_state(self, batch_size=None, dtype=None):
        A = tf.zeros(
            [self._batch_size, self._num_hidden_units, self._num_hidden_units],
            dtype=tf.float32)
        
        h = tf.zeros(
            [self._batch_size, self._num_hidden_units],
            dtype=tf.float32)

        return (h, A)

    def __call__(self, inputs, state, scope=None):
        # Split recurrent input into state and FW
        state, A = state

        # Recover vairables from scope
        W_x = tf.get_variable(name='W_x')
        b_x = tf.get_variable(name='b_x')
        W_h = tf.get_variable(name='W_h')
        gain = tf.get_variable(name='gain')
        bias = tf.get_variable(name='bias')
        
        state = tf.nn.dropout(state, self._keep_prob)
        state = tf.nn.relu((tf.matmul(inputs, W_x) + b_x) + tf.matmul(state, W_h))
        
        h_s = tf.reshape(state, [self._batch_size, 1, self._num_hidden_units])

        A = tf.add(tf.scalar_mul(self._l, A),
                   tf.scalar_mul(self._e, tf.matmul(tf.transpose(h_s, [0, 2, 1]), h_s)))

        for _ in range(self._S):
            h_s = tf.reshape(tf.matmul(inputs, W_x) + b_x, tf.shape(h_s)) \
                          + tf.reshape(tf.matmul(state, W_h), tf.shape(h_s)) \
                          + tf.matmul(h_s, A)

            # Apply layernorm
            mu = tf.reduce_mean(h_s, axis=2) # each sample
            sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(h_s) - mu), axis=1))

            h_s = tf.divide(tf.multiply(gain, (tf.squeeze(h_s) - mu)), tf.expand_dims(sigma, -1)) + bias

            # Apply nonlinearity
            h_s = tf.nn.relu(h_s)
            # Expand for future S steps
            h_s = tf.expand_dims(h_s, 1)

        # Reshape h_s into h
        h = tf.reshape(h_s, [self._batch_size, self._num_hidden_units])

        return h, (h, A)



class FastWeightsRNNCell_Experimental(RNNCell):

    def __init__(self, num_hidden_units, batch_size, loop_steps=1, 
                 decay_rate=0.95, eta=0.5, dropout_keep_prob=1.0):
        super(FastWeightsRNNCell_Experimental, self).__init__()
        self._num_hidden_units = num_hidden_units
        self._keep_prob = dropout_keep_prob
        self._batch_size = batch_size
        self._S = loop_steps
        self._e = eta
        self._l = decay_rate

    @property
    def state_size(self):
        return self._num_hidden_units

    @property
    def output_size(self):
        return self._num_hidden_units

    def zero_state(self, batch_size=None, dtype=None):
        A = tf.zeros(
            [self._batch_size, self._num_hidden_units, self._num_hidden_units],
            dtype=tf.float32)
        
        A_deconv = tf.zeros(
            [self._batch_size, self._num_hidden_units, self._num_hidden_units],
            dtype=tf.float32)

        h = tf.zeros(
            [self._batch_size, self._num_hidden_units],
            dtype=tf.float32)

        return (h, A, A_deconv)

    def __call__(self, inputs, state, scope=None):
        # Split recurrent input into state and FW
        state, A, A_deconv = state

        # Recover vairables from scope
        W_x = tf.get_variable(name='W_x')
        b_x = tf.get_variable(name='b_x')
        W_conv = tf.get_variable(name='W_conv')
        W_h = tf.get_variable(name='W_h')
        gain = tf.get_variable(name='gain')
        bias = tf.get_variable(name='bias')
        
        state = tf.nn.dropout(state, self._keep_prob)
        state = tf.nn.relu((tf.matmul(inputs, W_x) + b_x) + tf.matmul(state, W_h))
        
        h_s = tf.reshape(state, [self._batch_size, 1, self._num_hidden_units])

        A_deconv_temp = tf.nn.conv2d_transpose(tf.expand_dims(h_s, -1), W_conv, 
                    output_shape=[self._batch_size, 1, self._num_hidden_units, self._num_hidden_units], 
                    strides=[1,1,1,1], padding='SAME')

        A_deconv = tf.add(tf.scalar_mul(self._l, A_deconv),
                           tf.scalar_mul(self._e, tf.squeeze(A_deconv_temp)))

        A = tf.add(tf.scalar_mul(self._l, A),
                   tf.scalar_mul(self._e, tf.matmul(tf.transpose(h_s, [0, 2, 1]), h_s)))

        for _ in range(self._S):
            h_s = tf.reshape(tf.matmul(inputs, W_x) + b_x, tf.shape(h_s)) \
                          + tf.reshape(tf.matmul(state, W_h), tf.shape(h_s)) \
                          + tf.matmul(h_s, A) \
                           + tf.matmul(h_s, A_deconv)


            # Apply layernorm
            mu = tf.reduce_mean(h_s, axis=2) # each sample
            sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(h_s) - mu), axis=1))

            h_s = tf.divide(tf.multiply(gain, (tf.squeeze(h_s) - mu)), tf.expand_dims(sigma, -1)) + bias

            # Apply nonlinearity
            h_s = tf.nn.relu(h_s)
            # Expand for future S steps
            h_s = tf.expand_dims(h_s, 1)

        # Reshape h_s into h
        h = tf.reshape(h_s, [self._batch_size, self._num_hidden_units])

        return h, (h, A, A_deconv)