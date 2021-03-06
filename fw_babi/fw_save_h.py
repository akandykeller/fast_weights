"""Fast-Weights Q&A Network

Alternate implementation of the FW-RNN which stores all hidden states and re-computes
the fast-weight matrix A, allowing for explicit computation of the attention over each
hidden state during inference. 

The framework for this implementation (including sentence and question encoders)
is based on the end-to-end memory network, but the actual memory network module
is replaced with a Fast-Weights RNN. 
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

def print_attentions(attentions, s, q, rev_word_idx, pred, answer, b_idx=0):
    print("Question: {}?".format(' '.join([rev_word_idx[x] for x in q[b_idx, :]])))
    
    inputs = np.concatenate((s[b_idx, :, :], np.reshape(q[b_idx, :], (1, q.shape[1]))), axis=0)

    for seq_idx, attn in enumerate(attentions):
        print("RNN Step: {}".format(seq_idx))

        for t in range(len(attn)):
            val = attn[t][b_idx]
            sent = " ".join([rev_word_idx[x] for x in inputs[t, :]])
            print("{0:.2f} || {1}".format(val, sent))

    print("Predicted Answer: {}".format(rev_word_idx[pred[b_idx]]))
    print("Correct Answer: {}".format(rev_word_idx[np.argmax(answer[b_idx])]))


class FWQA_save_h(object):
    """Fast Weights Q&A network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size, num_hidden_units,
        S=1,
        S_Q=0,
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
        logits = self._inference(self._stories, self._queries) # (batch_size, vocab_size)

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
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")
        self._l = tf.placeholder(tf.float32, [], name="decay_lambda") # Decay lambda
        self._e = tf.placeholder(tf.float32, [], name="FW_learning_rate") # fast weights learning rate (eta)
        self._keep_prob = tf.placeholder_with_default(1.0, [], name="keep_prob") 
        
        # For the summary:
        self.all_answers_pred = tf.placeholder(tf.int32, [None], name="all_answers_pred")
        self.all_answers_true = tf.placeholder(tf.int32, [None], name="all_answers_true")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            LUT = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_size-1, self._embedding_size])])
            self.LUT = tf.Variable(LUT, name="LUT")

            LUT_Q = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_size-1, self._embedding_size])])
            self.LUT_Q = tf.Variable(LUT_Q, name="LUT_Q")

        with tf.variable_scope("fast_weights"):
            # input weights (proper initialization)
            self.W_x = tf.Variable(tf.random_uniform(
                [self._embedding_size, self._num_hidden_units],
                -np.sqrt(2.0/self._embedding_size),
                np.sqrt(2.0/self._embedding_size)),
                dtype=tf.float32, name="W_x")
            
            self.b_x = tf.Variable(tf.zeros(
                [self._num_hidden_units]),
                dtype=tf.float32, name="b_x")

            # hidden weights (See Hinton's video @ 21:20)
            self.W_h = tf.Variable(
                initial_value=0.05 * np.identity(self._num_hidden_units),
                dtype=tf.float32, name="W_h")

            # Final output layer weights (could be removed & tie input weights)
            self.W_softmax = tf.Variable(tf.random_uniform(
                [self._num_hidden_units, self._vocab_size],
                -np.sqrt(2.0/self._vocab_size),
                np.sqrt(2.0/self._vocab_size)),
                dtype=tf.float32, name="W_softmax")

            self.b_softmax = tf.Variable(tf.zeros(
                [self._vocab_size]),
                dtype=tf.float32, name="b_softmax")

            # scale and shift for layernorm
            self.gain = tf.Variable(tf.ones(
                [self._num_hidden_units]),
                dtype=tf.float32, name="gain")

            self.bias = tf.Variable(tf.zeros(
                [self._num_hidden_units]),
                dtype=tf.float32, name="ln_bias")

        self._nil_vars = set([self.LUT.name, self.LUT_Q.name])

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name):
            # Use different query and memory LUT
            q_emb = tf.nn.embedding_lookup(self.LUT_Q, queries)
            q_enc  = tf.expand_dims(tf.reduce_sum(q_emb * self._encoding, 1), 1)
            
            m_emb = tf.nn.embedding_lookup(self.LUT, stories)
            m_enc = tf.reduce_sum(m_emb * self._encoding, 2)

            input_seq = tf.concat(axis=1, values=[m_enc, q_enc])

        seq_len = input_seq.get_shape().as_list()[1]

        self.attentions = []

        # fast weights and hidden state initialization
        self.A = tf.zeros(
            [self._batch_size, self._num_hidden_units],
            dtype=tf.float32)
        
        self.h_list = [tf.zeros(
                        [self._batch_size, self._num_hidden_units],
                        dtype=tf.float32)]

        with tf.variable_scope("fast_weights"):
            # NOTE:inputs are batch-major
            # Process batch by time-major
            for t in range(0, seq_len):
                self.h_list[t] = tf.nn.dropout(self.h_list[t], self._keep_prob)

                # hidden state (preliminary vector)
                self.h_list[t] = tf.nn.relu((tf.matmul(input_seq[:, t, :], self.W_x) + self.b_x) + tf.matmul(self.h_list[t], self.W_h))

                self.h_s = self.h_list[t]

                # Loop for S steps
                S_steps = self._S if t < self._memory_size else self._S + self._S_Q
                for _ in range(S_steps):
                    self.attentions.append([tf.squeeze(tf.matmul(tf.expand_dims(h, 1), tf.expand_dims(self.h_s, -1))) for h in self.h_list])

                    # Create the fixed A for this time step
                    self.A_hs = tf.scalar_mul(self._e,
                                         tf.add_n([tf.pow(self._l, tf.constant(float(t - idx))) * tf.multiply(tf.expand_dims(self.attentions[-1][idx],-1), h)
                                                    for idx, h in enumerate(self.h_list)]))

                    self.h_s = tf.matmul(input_seq[:, t, :], self.W_x) + self.b_x \
                                + tf.matmul(self.h_list[t], self.W_h) \
                                + self.A_hs

                    # Apply layernorm
                    mu = tf.expand_dims(tf.reduce_mean(self.h_s, axis=1), -1) # each sample
                    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(self.h_s) - mu), axis=1))

                    self.h_s = tf.divide(tf.multiply(self.gain, (tf.squeeze(self.h_s) - mu)), tf.expand_dims(sigma, -1)) + \
                        self.bias

                    # Apply nonlinearity
                    self.h_s = tf.nn.relu(self.h_s)

                # Reshape h_s into h
                self.h_list.append(self.h_s)

            # self.h_out = tf.matmul(self.h, tf.transpose(self.W_x, [1,0]))

            # with tf.variable_scope(self._name):
            #     return tf.matmul(self.h_out, tf.transpose(self.LUT, [1,0]))

            return tf.matmul(self.h_list[-1], self.W_softmax) + self.b_softmax

    def batch_fit(self, stories, queries, answers, learning_rate, eta, decay_lambda, keep_prob):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, 
                     self._lr: learning_rate, self._e: eta, self._l: decay_lambda, self._keep_prob: keep_prob}
        summary, loss, _ = self._sess.run([self.merged_train_summaries, self.loss_op, self.train_op], feed_dict=feed_dict)
        
        return summary, loss

    def predict(self, stories, queries, answers, eta, decay_lambda):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._e: eta, self._l: decay_lambda}
        summary, loss, attentions = self._sess.run([self.merged_train_summaries, self.predict_op, self.attentions], feed_dict=feed_dict)
        return summary, loss, attentions

    def predict_proba(self, stories, queries, answers, eta, decay_lambda):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._e: eta, self._l: decay_lambda}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries, answers, eta, decay_lambda):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._e: eta, self._l: decay_lambda}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)

    def compute_mean_accuracy(self, answers_pred, answers_true):
        """Computes the mean accuracy of all batches for summary writer

        Args:
            answers_pred: 
            answers_true:
        """
        feed_dict = {self.all_answers_pred: answers_pred, self.all_answers_true: answers_true}
        return self._sess.run([self.mean_acc_all_summary, self.mean_accuracy_all], feed_dict=feed_dict)
