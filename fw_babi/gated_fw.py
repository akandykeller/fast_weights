"""Gated Fast-Weights Q&A Network

Unrolled implementation of Gated Fast-Weights network applied in the bAbI Q&A setting.

The framework for this implementation (including sentence and question encoders)
is based on the end-to-end memory network, but the actual memory network module
is replaced with a Gated Fast-Weights RNN. 
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 of End to end Memory Networks
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    encoding[:, -1] = 1.0

    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class Gated_FWQA(object):
    """Fast Weights Q&A network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, 
        embedding_size=15, 
        num_hidden_units=40,
        intermediate_size=100,
        num_layers=1,
        tied_output=False,
        dropout_before_bn=True,
        s1_ident=False, 
        max_grad_norm=40.0,
        nonlin=None,
        initializer=tf.random_normal_initializer(stddev=0.1),
        encoding=position_encoding,
        log_dir='./results/',
        session=tf.Session(),
        name='FWQA'):
        """Creates a Fast-Weights Q&A network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._num_hidden_units = num_hidden_units
        self._num_layers = num_layers
        self._tied_output = tied_output
        self._dropout_before_bn = dropout_before_bn
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._s1_ident = s1_ident
        self._init = initializer
        self._log_dir = log_dir
        self._name = name

        self._s1_nin = self._embedding_size + self._num_hidden_units
        self._s1_nout = intermediate_size

        self._delta_1_size = 2 * (self._s1_nin  + self._num_hidden_units)
        self._delta_2_size = 4 * (self._num_hidden_units)

        self._s2_nout = self._num_hidden_units + self._delta_1_size + self._delta_2_size

        self._build_inputs()
        self._build_vars()

        self._opt = tf.train.AdamOptimizer(learning_rate=self._lr)

        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        self.logits = self._inference(self._stories_wq, self._sq_lens) # (batch_size, vocab_size)

        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.cast(tf.argmax(self._answers, 1), tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum
        tf.summary.scalar('cross_entropy_sum', loss_op)

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(self.logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(self.logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        self.merged_train_summaries = tf.summary.merge_all()

        self.all_correct_predictions = tf.equal(self.all_answers_pred, self.all_answers_true)
        self.mean_accuracy_all = tf.reduce_mean(tf.cast(self.all_correct_predictions, tf.float32))
        self.mean_acc_all_summary = tf.summary.scalar('mean_epoch_accuracy', self.mean_accuracy_all)

        self.train_writer = tf.summary.FileWriter(self._log_dir + 'train', session.graph)
        self.val_writer = tf.summary.FileWriter(self._log_dir + 'val')
        self.test_writer = tf.summary.FileWriter(self._log_dir + 'test')

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories_wq = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories_queries")
        self._sq_lens = tf.placeholder(tf.int32, [None], name="sq_lens")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")

        self._keep_prob = tf.placeholder_with_default(1.0, [], name="keep_prob") 

        # For the summary:
        self.all_answers_pred = tf.placeholder(tf.int32, [None], name="all_answers_pred")
        self.all_answers_true = tf.placeholder(tf.int32, [None], name="all_answers_true")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            LUT = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_size-1, self._embedding_size])])
            self.LUT = tf.Variable(LUT, name="LUT")

        with tf.variable_scope("fast_weights_rnn"):
            # first do make the s-RNN weights
            if self._s1_ident:
                assert self._s1_nin == self._s1_nout
                self.S1 = tf.get_variable(name="S1",
                              initializer=tf.constant(0.05 * np.identity(self._s1_nout), dtype=tf.float32),
                              dtype=tf.float32)
            else:
                self.S1 = tf.get_variable(name="S1",
                          initializer=tf.random_uniform([self._s1_nin, self._s1_nout],               
                                      -np.sqrt(2.0/self._s1_nin), np.sqrt(2.0/self._s1_nin)),
                          dtype=tf.float32)

            self.b1 = tf.get_variable(name="b1",
                                      initializer=tf.zeros([self._s1_nout]),
                                      dtype=tf.float32)
            
            self.S2 = tf.get_variable(name="S2",
                                      initializer=tf.random_uniform([self._s1_nout, self._s2_nout],               
                                                  -np.sqrt(2.0/self._s1_nout), np.sqrt(2.0/self._s1_nout)),
                                      dtype=tf.float32)

            self.b2 = tf.get_variable(name="b2",
                                      initializer=tf.zeros([self._s2_nout]),
                                      dtype=tf.float32)

            # Then make the FW RNN weights
            # self.F1 = 55 x 40 
            # self.F2 = 40 x 40

            self.W = tf.get_variable(name="W",
                                      initializer=tf.random_uniform([self._embedding_size, self._num_hidden_units],               
                                                  -np.sqrt(2.0/self._embedding_size), np.sqrt(2.0/self._embedding_size)),
                                      dtype=tf.float32)


            # scale and shift for layernorm of F1
            self.gain_1 = tf.get_variable(name='gain_1',
                # shape=[self._num_hidden_units],
                initializer=tf.ones([self._num_hidden_units]),
                dtype=tf.float32)

            self.bias_1 = tf.get_variable(name='bias_1',
                # shape=[self._num_hidden_units],
                initializer=tf.zeros([self._num_hidden_units]),
                dtype=tf.float32)


            # scale and shift for layernorm of F2
            self.gain_2 = tf.get_variable(name='gain_2',
                initializer=tf.ones([self._num_hidden_units]),
                dtype=tf.float32)

            self.bias_2 = tf.get_variable(name='bias_2',
                initializer=tf.zeros([self._num_hidden_units]),
                dtype=tf.float32)


        with tf.variable_scope(self._name):
            # Final output layer weights
            if not self._tied_output:
                self.W_softmax = tf.Variable(tf.random_uniform(
                    [self._num_hidden_units, self._vocab_size],
                    -np.sqrt(2.0/self._vocab_size),
                    np.sqrt(2.0/self._vocab_size)),
                    dtype=tf.float32, name="W_softmax")

                self.b_softmax = tf.Variable(tf.zeros(
                    [self._vocab_size]),
                    dtype=tf.float32, name="b_softmax")

        self._nil_vars = set([self.LUT.name])

    def _inference(self, stories_wq, sq_lens):
        with tf.variable_scope(self._name):
            # Use LUT for all word embeddings
            sq_emb = tf.nn.embedding_lookup(self.LUT, stories_wq)
            input_seq = tf.reduce_sum(sq_emb * self._encoding, axis=2)

        seq_len = input_seq.get_shape().as_list()[1]

        # initialize hidden states and fast weights
        self.h_s = tf.zeros([self._batch_size, self._num_hidden_units], dtype=tf.float32)
        self.h_f = tf.zeros([self._batch_size, self._num_hidden_units], dtype=tf.float32)
        self.F1 = tf.zeros([self._batch_size, self._num_hidden_units, self._s1_nin], dtype=tf.float32)
        self.F2 = tf.zeros([self._batch_size, self._num_hidden_units, self._num_hidden_units], dtype=tf.float32)

        for t in range(0, seq_len):
            with tf.variable_scope("fast_weights_rnn"):
                # Compute output of S-RNN
                
                s1_out = tf.nn.tanh(
                            tf.matmul(
                                tf.concat(axis=1, values=[self.h_s, input_seq[:, t, :]]), 
                                self.S1)
                            )

                s1_out = tf.nn.dropout(s1_out, self._keep_prob)

                s2_out = tf.matmul(
                            s1_out,
                            self.S2 
                            )

                z_s, delta_1, delta_2 = self.unpack_s_out(s2_out)

                # Compute h_s_t+1
                self.h_s = tf.nn.tanh(z_s)

                # Break deltas in constituent parts
                a_1, b_1, g_1, d_1 = self.unpack_delta(delta_1, self._num_hidden_units, self._s1_nin)
                a_2, b_2, g_2, d_2 = self.unpack_delta(delta_2, self._num_hidden_units, self._num_hidden_units)

                # Compute fast weight update matrix H
                H_1 = tf.matmul(tf.expand_dims(tf.nn.tanh(a_1), -1), tf.expand_dims(tf.nn.tanh(b_1), 1))
                H_2 = tf.matmul(tf.expand_dims(tf.nn.tanh(a_2), -1), tf.expand_dims(tf.nn.tanh(b_2), 1))

                # Compute gate matrix T
                T_1 = tf.matmul(tf.expand_dims(tf.nn.tanh(g_1), -1), tf.expand_dims(tf.nn.tanh(d_1), 1))
                T_2 = tf.matmul(tf.expand_dims(tf.nn.tanh(g_2), -1), tf.expand_dims(tf.nn.tanh(d_2), 1))

                # consider not tied gates? input & forget gate?

                # Update fast weights via gated combination
                self.F1 = tf.multiply(T_1, H_1) + tf.multiply((1 - T_1), self.F1)
                self.F2 = tf.multiply(T_2, H_2) + tf.multiply((1 - T_2), self.F2)

                # Compute h_f_t+1
                # Compute f1_out
                f1_out = tf.nn.tanh(
                            tf.matmul(
                                self.F1,
                                tf.expand_dims(tf.concat(axis=1, values=[self.h_f, input_seq[:, t, :]]), -1)
                                )
                            )

                if self._dropout_before_bn:
                    f1_out = tf.nn.dropout(f1_out, self._keep_prob)

                # Apply layernorm 1
                mu_1 = tf.reduce_mean(f1_out, axis=1) # each sample
                sigma_1 = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(f1_out) - mu_1), axis=1))
                f1_out = tf.divide(tf.multiply(self.gain_1, (tf.squeeze(f1_out) - mu_1)), tf.expand_dims(sigma_1, -1)) + self.bias_1

                if not self._dropout_before_bn:
                    f1_out = tf.nn.dropout(f1_out, self._keep_prob)

                f2_out = tf.nn.tanh(
                            tf.matmul(
                                self.F2,
                                tf.expand_dims(f1_out, -1)
                                )
                            )

                # Apply layernorm 2
                mu_2 = tf.reduce_mean(f2_out, axis=1) # each sample
                sigma_2 = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(f2_out) - mu_2), axis=1))
                f2_out = tf.divide(tf.multiply(self.gain_2, (tf.squeeze(f2_out) - mu_2)), tf.expand_dims(sigma_2, -1)) + self.bias_2

                self.h_f = f2_out

        if self._tied_output:
            with tf.variable_scope("fast_weights_rnn"):
                out_emb = tf.matmul(self.h_f, tf.transpose(self.W, [1, 0]))

            with tf.variable_scope(self._name):
                logits = tf.matmul(out_emb, tf.transpose(self.LUT, [1, 0]))

        else:
            with tf.variable_scope(self._name):
                logits = tf.matmul(self.h_f, self.W_softmax) + self.b_softmax

        return logits

    def unpack_s_out(self, s_out):
        z_s = s_out[:, 0 : self._num_hidden_units]
        delta_1 = s_out[:, self._num_hidden_units : self._num_hidden_units + self._delta_1_size]
        delta_2 = s_out[:, self._num_hidden_units + self._delta_1_size :]

        return z_s, delta_1, delta_2


    def unpack_delta(self, delta, m, n):
        a = delta[:, 0 : m]
        b = delta[:, m : m + n]
        g = delta[:, m + n : m + n + m]
        d = delta[:, m + n + m :]

        return a, b, g, d


    def batch_fit(self, stories_wq, sq_lens, answers, learning_rate, keep_prob):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories_wq: stories_wq, self._sq_lens: sq_lens, 
                     self._answers: answers, self._lr: learning_rate, 
                     self._keep_prob: keep_prob}
        summary, loss, _ = self._sess.run([self.merged_train_summaries, self.loss_op, self.train_op], feed_dict=feed_dict)
        
        return summary, loss

    def predict(self, stories_wq, sq_lens, answers):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories_wq: stories_wq, self._sq_lens: sq_lens, self._answers: answers}
        summary, preds = self._sess.run([self.merged_train_summaries, self.predict_op], feed_dict=feed_dict)
        return summary, preds

    def predict_proba(self, stories_wq, sq_lens, answers):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories_wq: stories_wq, self._sq_lens: sq_lens, self._answers: answers}

        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories_wq, sq_lens, answers):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories_wq: stories_wq, self._sq_lens: sq_lens, self._answers: answers}

        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)

    def compute_mean_accuracy(self, answers_pred, answers_true):
        """Computes the mean accuracy of all batches for summary writer

        Args:
            answers_pred: 
            answers_true:
        """
        feed_dict = {self.all_answers_pred: answers_pred, self.all_answers_true: answers_true}
        return self._sess.run([self.mean_acc_all_summary, self.mean_accuracy_all], feed_dict=feed_dict)
