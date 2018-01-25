"""Fast-Weights Q&A Network

Multi-layered bi-directional implementation of the Fast-Weights RNN using 
the cell from lnfw_rnn_cell.py.

The framework for this implementation (including sentence and question encoders)
is based on the end-to-end memory network, but the actual memory network module
is replaced with a Fast-Weights RNN. 
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from . import FastWeightsLSTMCell

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

class FWQA_DeepBiLSTM(object):
    """Fast Weights Q&A network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size, num_hidden_units,
        S=1,
        S_Q=0,
        num_layers=1,
        tied_output=False,
        max_grad_norm=40.0,
        max_pooling=True,
        layer_norm=True,
        forget_bias=1.0,
        nonlin=None,
        initializer=tf.random_normal_initializer(stddev=0.1),
        encoding=position_encoding,
        log_dir='./results/',
        session=tf.Session(),
        name='FWQA_LSTM'):
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
        self._max_pooling = max_pooling
        self._layer_norm = layer_norm
        self._forget_bias = forget_bias

        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._log_dir = log_dir
        self._name = name

        self._S = S
        self._S_Q = S_Q

        self._build_inputs()
        self._build_vars()

        self._opt = tf.train.AdamOptimizer(learning_rate=self._lr)

        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        logits = self._inference(self._stories_wq, self._sq_lens) # (batch_size, vocab_size)

        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.cast(tf.argmax(self._answers, 1), tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._answers, tf.float32), name="cross_entropy")
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
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
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
        self._l_1 = tf.placeholder(tf.float32, [], name="decay_lambda_l1") # Decay lambda
        self._e_1 = tf.placeholder(tf.float32, [], name="FW_learning_rate_l1") # fast weights learning rate (eta)
        self._l_2 = tf.placeholder(tf.float32, [], name="decay_lambda_l2") # Decay lambda
        self._e_2 = tf.placeholder(tf.float32, [], name="FW_learning_rate_l2") # fast weights learning rate (eta)
        self._l_3 = tf.placeholder(tf.float32, [], name="decay_lambda_l3") # Decay lambda
        self._e_3 = tf.placeholder(tf.float32, [], name="FW_learning_rate_l3") # fast weights learning rate (eta)
        self._keep_prob = tf.placeholder_with_default(1.0, [], name="keep_prob") 
        
        self.etas = [self._e_1, self._e_2, self._e_3]
        self.lambdas = [self._l_1, self._l_2, self._l_3]

        # For the summary:
        self.all_answers_pred = tf.placeholder(tf.int32, [None], name="all_answers_pred")
        self.all_answers_true = tf.placeholder(tf.int32, [None], name="all_answers_true")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            LUT = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_size-1, self._embedding_size])])
            self.LUT = tf.Variable(LUT, name="LUT")

        self.fw_cells_list = []
        self.bw_cells_list = []

        with tf.variable_scope("fast_weights_rnn"):
            for layernum in range(self._num_layers):
                if layernum == 0:
                    W_ifoj_shape = [self._embedding_size + self._num_hidden_units, self._num_hidden_units * 4]
                else:
                    W_ifoj_shape = [self._num_hidden_units * 4, self._num_hidden_units * 4]

                for scope in ["fw/multi_rnn_cell/cell_{}".format(layernum), "bw/multi_rnn_cell/cell_{}".format(layernum)]:
                    with tf.variable_scope(scope):
                        # input weights (proper initialization)
                        self.W_ifoj = tf.get_variable(name="W_ifoj",
                                        # shape=W_ifoj_shape,
                                        initializer=tf.random_uniform(W_ifoj_shape,               
                                                    -np.sqrt(2.0/W_ifoj_shape[0]), np.sqrt(2.0/W_ifoj_shape[0])),
                                        dtype=tf.float32)

                        self.b_ifoj = tf.get_variable(name="b_ifoj",
                                    # shape=[self._num_hidden_units],
                                    initializer=tf.zeros([self._num_hidden_units * 4]),
                                    dtype=tf.float32)

                        # # hidden weights (See Hinton's video @ 21:20)
                        # self.W_h = tf.get_variable(name="W_h",
                        #     # shape=[self._num_hidden_units, self._num_hidden_units],
                        #     initializer=tf.constant(0.05 * np.identity(self._num_hidden_units), dtype=tf.float32),
                        #     dtype=tf.float32)

                        # scale and shift for new_c layernorm
                        self.gain_state = tf.get_variable(name='gain_state',
                            # shape=[self._num_hidden_units],
                            initializer=tf.ones([self._num_hidden_units]),
                            dtype=tf.float32)

                        self.bias_state = tf.get_variable(name='bias_state',
                            # shape=[self._num_hidden_units],
                            initializer=tf.zeros([self._num_hidden_units]),
                            dtype=tf.float32)

                        # scale and shift for ifoj layernorm
                        self.gain_ifoj = tf.get_variable(name='gain_ifoj',
                            # shape=[self._num_hidden_units],
                            initializer=tf.ones([self._num_hidden_units * 4]),
                            dtype=tf.float32)

                        self.bias_ifoj = tf.get_variable(name='bias_ifoj',
                            # shape=[self._num_hidden_units],
                            initializer=tf.zeros([self._num_hidden_units * 4]),
                            dtype=tf.float32)

                self.fw_cells_list.append(FastWeightsLSTMCell(num_hidden_units=self._num_hidden_units, 
                                                               batch_size=self._batch_size, loop_steps=self._S, 
                                                               forget_bias=self._forget_bias, layer_norm=self._layer_norm,
                                                               decay_rate=self.lambdas[layernum], eta=self.etas[layernum], 
                                                               dropout_keep_prob=self._keep_prob))

                self.bw_cells_list.append(FastWeightsLSTMCell(num_hidden_units=self._num_hidden_units, 
                                                               batch_size=self._batch_size, loop_steps=self._S,
                                                               forget_bias=self._forget_bias, layer_norm=self._layer_norm,
                                                               decay_rate=self.lambdas[layernum], eta=self.etas[layernum], 
                                                               dropout_keep_prob=self._keep_prob))

            self.fw_cells = tf.nn.rnn_cell.MultiRNNCell(self.fw_cells_list)
            self.bw_cells = tf.nn.rnn_cell.MultiRNNCell(self.bw_cells_list)


        with tf.variable_scope(self._name):
            # Final output layer weights
            if not self._tied_output:
                self.W_softmax = tf.Variable(tf.random_uniform(
                    [2 * self._num_hidden_units, self._vocab_size],
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

        import ipdb
        ipdb.set_trace()

        with tf.variable_scope("fast_weights_rnn", reuse=True) as scope:
            outputs, states = tf.nn.bidirectional_dynamic_rnn(self.fw_cells, self.bw_cells,
                                    input_seq, sequence_length=sq_lens, 
                                    initial_state_fw=self.fw_cells.zero_state(batch_size=self._batch_size, dtype=tf.float32), 
                                    initial_state_bw=self.bw_cells.zero_state(batch_size=self._batch_size, dtype=tf.float32), 
                                    time_major=False, scope=scope)

            fw_states, bw_states = states
            fw_outputs, bw_outputs = outputs

            if self._max_pooling:
                pooled_fw_state = tf.layers.max_pooling1d(fw_outputs, self._memory_size, (1,))
                pooled_bw_state = tf.layers.max_pooling1d(bw_outputs, self._memory_size, (1,))

                final_fw_state = tf.squeeze(pooled_fw_state)
                final_bw_state = tf.squeeze(pooled_bw_state)

            else:
                final_fw_state = fw_states[-1][0]
                final_bw_state = bw_states[-1][0]

        if self._tied_output:
            with tf.variable_scope("fast_weights_rnn/fw/multi_rnn_cell/cell_0", reuse=True):
                fw_emb = tf.matmul(final_fw_state, tf.transpose(self.W_x, [1, 0]))

            with tf.variable_scope("fast_weights_rnn/bw/multi_rnn_cell/cell_0", reuse=True):
                bw_emb = tf.matmul(final_bw_state, tf.transpose(self.W_x, [1, 0]))
            
            with tf.variable_scope(self._name):
                logits_fw = tf.matmul(fw_emb, tf.transpose(self.LUT, [1, 0]))
                logits_bw = tf.matmul(bw_emb, tf.transpose(self.LUT, [1, 0]))

                logits = tf.maximum(logits_fw, logits_bw)

        else:
            with tf.variable_scope(self._name):
                final_output = tf.concat([final_fw_state, final_bw_state], 1)
                logits = tf.matmul(final_output, self.W_softmax) + self.b_softmax

        return logits

    def batch_fit(self, stories_wq, sq_lens, answers, learning_rate, etas, decay_lambdas, keep_prob):
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
                     self._e_1: etas[0], self._l_1: decay_lambdas[0], 
                     self._e_2: etas[1], self._l_2: decay_lambdas[1], 
                     self._e_3: etas[2], self._l_3: decay_lambdas[2], 
                     self._keep_prob: keep_prob}
        summary, loss, _ = self._sess.run([self.merged_train_summaries, self.loss_op, self.train_op], feed_dict=feed_dict)
        
        return summary, loss

    def predict(self, stories_wq, sq_lens, answers, etas, decay_lambdas):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories_wq: stories_wq, self._sq_lens: sq_lens, self._answers: answers, 
                     self._e_1: etas[0], self._l_1: decay_lambdas[0], 
                     self._e_2: etas[1], self._l_2: decay_lambdas[1], 
                     self._e_3: etas[2], self._l_3: decay_lambdas[2]}
        summary, loss = self._sess.run([self.merged_train_summaries, self.predict_op], feed_dict=feed_dict)
        return summary, loss

    def predict_proba(self, stories_wq, sq_lens, answers, etas, decay_lambdas):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories_wq: stories_wq, self._sq_lens: sq_lens, self._answers: answers, 
                     self._e_1: etas[0], self._l_1: decay_lambdas[0], 
                     self._e_2: etas[1], self._l_2: decay_lambdas[1], 
                     self._e_3: etas[2], self._l_3: decay_lambdas[2]}

        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories_wq, sq_lens, answers, etas, decay_lambdas):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories_wq: stories_wq, self._sq_lens: sq_lens, self._answers: answers, 
                     self._e_1: etas[0], self._l_1: decay_lambdas[0], 
                     self._e_2: etas[1], self._l_2: decay_lambdas[1], 
                     self._e_3: etas[2], self._l_3: decay_lambdas[2]}

        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)

    def compute_mean_accuracy(self, answers_pred, answers_true):
        """Computes the mean accuracy of all batches for summary writer

        Args:
            answers_pred: 
            answers_true:
        """
        feed_dict = {self.all_answers_pred: answers_pred, self.all_answers_true: answers_true}
        return self._sess.run([self.mean_acc_all_summary, self.mean_accuracy_all], feed_dict=feed_dict)
