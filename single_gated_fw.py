"""Example running Fast-Weights QA model on a single babi task
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data_SQ
from sklearn import cross_validation, metrics
from fw_babi import Gated_FWQA
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for SGD.")
tf.flags.DEFINE_integer("intermediate_size", 100, "Shape 1 of S^1")
tf.flags.DEFINE_float("anneal_rate", 50, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 400, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 1.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 15, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("num_hidden_units", 40, "Number of hidden units for RNN.")
tf.flags.DEFINE_integer("num_layers", 1, "Number of layers for RNN.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Dropout keep probabilitiy.")
tf.flags.DEFINE_integer("memory_size", 70, "Maximum size of memory (all stories are padded to this).")
tf.flags.DEFINE_boolean("tied_output", True, "Tie output weights to input weights")
tf.flags.DEFINE_boolean("dropout_before_bn", True, "Droupout before batchnorm or after")
tf.flags.DEFINE_boolean("s1_ident", False, "Initiaize S1 with the identity matrix")
tf.flags.DEFINE_boolean("prepend_q", False, "Append question before story sentences in addition to after.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("log_dir", "results/", "Directory to store training results")

FLAGS = tf.flags.FLAGS
print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
batch_size = FLAGS.batch_size

# Add time words/indexes
for i in range(memory_size):
    word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position
sentence_size += 1  # +1 for time words

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
SQ, SQ_lens, A = vectorize_data_SQ(train, word_idx, sentence_size, memory_size, prepend_q=FLAGS.prepend_q)
trainSQ, valSQ, trainSQ_lens, valSQ_lens, trainA, valA = cross_validation.train_test_split(SQ, SQ_lens, A, test_size=.1, random_state=FLAGS.random_state)
testSQ, testSQ_lens, testA = vectorize_data_SQ(test, word_idx, sentence_size, memory_size, prepend_q=FLAGS.prepend_q)

print(testSQ[0])
print(testSQ_lens[0])

print("Training set shape", trainSQ.shape)

if valSQ.shape[0] < batch_size:
    valSQ = np.resize(valSQ, (batch_size + 1,) +  valSQ.shape[1:])
    valSQ_lens = np.resize(valSQ_lens, (batch_size + 1,) +  valSQ_lens.shape[1:])
    valA = np.resize(valA, (batch_size + 1,) +  valA.shape[1:])

# params
n_train = trainSQ.shape[0]
n_test = testSQ.shape[0]
n_val = valSQ.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

val_batches = zip(range(0, n_val-batch_size, batch_size), range(batch_size, n_val, batch_size))
val_batches = [(start, end) for start, end in val_batches]

test_batches = zip(range(0, n_test-batch_size, batch_size), range(batch_size, n_test, batch_size))
test_batches = [(start, end) for start, end in test_batches]

with tf.Session() as sess:
    model = Gated_FWQA(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, FLAGS.num_hidden_units, FLAGS.intermediate_size,
                 num_layers=FLAGS.num_layers, tied_output=FLAGS.tied_output, dropout_before_bn=FLAGS.dropout_before_bn, s1_ident=FLAGS.s1_ident,
                 session=sess, max_grad_norm=FLAGS.max_grad_norm, log_dir=FLAGS.log_dir)
    
    print("Model Stats")
    print("S1 shape: {}".format(model.S1.shape))
    print("S2 shape: {}".format(model.S2.shape))

    model.train_writer.add_summary(sess.run(tf.summary.text("FLAGS", tf.convert_to_tensor("{}".format(FLAGS.__flags)))))

    for t in range(1, FLAGS.epochs+1):
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        for b_idx, (start, end) in enumerate(batches):
            sq = trainSQ[start:end]
            sq_lens = trainSQ_lens[start:end]
            a = trainA[start:end]
            # import ipdb
            # ipdb.set_trace()
            # summary, pred, extra = model.predict(sq, sq_lens, a)
            summary, cost_t = model.batch_fit(sq, sq_lens, a, lr, FLAGS.keep_prob)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for b_idx, start in enumerate(range(0, n_train-batch_size, batch_size)):
                end = start + batch_size
                sq = trainSQ[start:end]
                sq_lens = trainSQ_lens[start:end]
                a = trainA[start:end]
                summary, pred = model.predict(sq, sq_lens, a)
                model.train_writer.add_summary(summary, t * len(batches) + b_idx)
                train_preds += list(pred)

            train_acc_summary, acc = model.compute_mean_accuracy(np.array(train_preds), train_labels[:len(train_preds)])
            model.train_writer.add_summary(train_acc_summary, t)

            val_preds = []
            for b_idx, (start, end) in enumerate(val_batches):
                sq = valSQ[start:end]
                sq_lens = valSQ_lens[start:end]
                a = valA[start:end]
                summary, pred = model.predict(sq, sq_lens, a)
                model.val_writer.add_summary(summary, t * len(val_batches) + b_idx)
                val_preds += list(pred)

            val_acc_summary, acc = model.compute_mean_accuracy(np.array(val_preds), val_labels[:len(val_preds)])
            model.val_writer.add_summary(val_acc_summary, t)

            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels[:len(train_preds)])
            val_acc = metrics.accuracy_score(np.array(val_preds), val_labels[:len(val_preds)])

            eval_str = ("-----------------------\n" + 
                        "Epoch {}\n".format(t) + 
                        "Total Cost: {}\n".format(total_cost) + 
                        "Training Accuracy: {}\n".format(train_acc) +
                        "Validation Accuracy: {}\n".format(val_acc) + 
                        "-----------------------\n")

            print(eval_str)
            model.train_writer.add_summary(sess.run(tf.summary.text("eval_str_{}".format(t), tf.convert_to_tensor(eval_str))))


    test_preds = []
    for b_idx, (start, end) in enumerate(test_batches):
        sq = testSQ[start:end]
        sq_lens = testSQ_lens[start:end]
        a = testA[start:end]
        summary, pred = model.predict(sq, sq_lens, a)
        model.test_writer.add_summary(summary, t * len(test_batches) + b_idx)
        test_preds += list(pred)

    test_acc_summary, acc = model.compute_mean_accuracy(np.array(test_preds), test_labels[:len(test_preds)])
    model.test_writer.add_summary(test_acc_summary, t)

    test_acc = metrics.accuracy_score(np.array(test_preds), test_labels[:len(test_preds)])
    print("Testing Accuracy:", test_acc)
    model.test_writer.add_summary(sess.run(tf.summary.text("test_eval_str_{}".format(t), tf.convert_to_tensor("Test Accuracy: {}".format(test_acc)))))