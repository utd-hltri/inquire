from __future__ import print_function

import random
from itertools import izip
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tabulate import tabulate

import data_utils
import gan_model
import ir_evals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '',
                           'Path expression to folder containing train/vali/test data')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Number of queries per batch')
tf.app.flags.DEFINE_string('max_list_length', 128,
                           'Maximum number of documents to consider for each query')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_integer('max_run_steps', 10000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('eval_epoch_steps', 10, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_integer('random_seed', 1337, 'A seed value for randomness.')
tf.app.flags.DEFINE_integer('d_steps_per_g_step', 3,
                            'How many times the discriminator will be trained for every generator training step.')


def _running_avg(name, running_avg_loss, loss, summary_writer, step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)
    loss_sum = tf.Summary()
    loss_sum.value.add(tag='running_avg_' + name, simple_value=running_avg_loss.item())
    summary_writer.add_summary(loss_sum, step)
    # print('running_avg_%s: %f\n' % (name, running_avg_loss))
    return running_avg_loss


def _evaluate_list(estimated_scores, true_scores):
    # print "Estimated", len(estimated_scores), "==", estimated_scores
    # print "True", len(true_scores), "==", true_scores
    rs = [x[1] for x in sorted(izip(estimated_scores, true_scores), key=lambda x: -x[0])]
    # print rs
    ndcg_5 = ir_evals.ndcg_at_k(rs, 5)
    ndcg_10 = ir_evals.ndcg_at_k(rs, 10)
    ndcg_15 = ir_evals.ndcg_at_k(rs, 15)
    ndcg_20 = ir_evals.ndcg_at_k(rs, 20)

    p_5 = ir_evals.precision_at_k(rs, 5)
    p_10 = ir_evals.precision_at_k(rs, 10)
    p_15 = ir_evals.precision_at_k(rs, 15)
    p_20 = ir_evals.precision_at_k(rs, 20)

    mean_ap = ir_evals.average_precision(rs)
    mrr = ir_evals.mean_reciprocal_rank(rs)
    return [mean_ap, mrr, p_5, p_10, p_15, p_20, ndcg_5, ndcg_10, ndcg_15, ndcg_20]


def _evaluate_lists(guess_lists, gold_lists):
    evaluations = []
    for guess, gold in izip(guess_lists, gold_lists):
        evaluations.append(_evaluate_list(guess, gold))
    return np.mean(evaluations, axis=0)


def _train(model, batch_size, vectors_train,
           scores_train, vectors_vali, scores_vali, vectors_test, scores_test):
    print('Building model...')
    start = timer()
    model.build_graph()
    saver = tf.train.Saver()

    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    train_writer = tf.summary.FileWriter(FLAGS.log_root + '/train/')
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model._g_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True))
    end = timer()
    print('Model constructed in %6.3fs' % (end - start))

    print('Pre-training D...')
    for k in xrange(10):
        running_avg_loss_d = 0
        running_avg_d1 = 0
        for i in xrange(len(vectors_train)):
            x_vectors = vectors_train[i]
            x_scores = scores_train[i]
            d1, loss_d, step_d = model.run_pretrain_step(sess, x_vectors, x_scores)
            running_avg_d1 = _running_avg('D1', running_avg_d1, np.mean(d1), train_writer, step_d)
            running_avg_loss_d = _running_avg('loss_d', running_avg_loss_d, loss_d, train_writer, step_d)
        t = list(zip(vectors_train, scores_train))
        random.shuffle(t)
        vectors_train, scores_train = zip(*t)
        print('Pre-training epoch %d; Loss = %6.3f; D1 = %6.3f' % (k, running_avg_loss_d, running_avg_d1))

    print('Training...')
    running_avg_loss_d = 0
    running_avg_loss_g = 0
    running_avg_d1 = 0
    running_avg_d2 = 0
    step = 0
    while not sv.should_stop() and step < FLAGS.max_run_steps:
        table = []
        guess_scores = []
        gold_scores = []
        start = timer()
        for i in xrange(0, len(vectors_train) - (2 * FLAGS.d_steps_per_g_step + 1), 2 * FLAGS.d_steps_per_g_step + 1):
            for j in xrange(FLAGS.d_steps_per_g_step):
                x_vectors = vectors_train[i + j]
                x_scores = scores_train[i + j]
                z_vectors = vectors_train[i + j + 1]
                d1, _, loss_d, step_d = model.run_train_d_step(sess, x_vectors, x_scores, z_vectors)
                running_avg_loss_d = _running_avg('loss_d', running_avg_loss_d, loss_d, train_writer, step_d)

            z_vectors = vectors_train[i + j + 1]
            d2, z_scores, _, loss_g, step_g = model.run_train_g_step(sess, z_vectors)
            running_avg_loss_g = _running_avg('loss_g', running_avg_loss_g, loss_g, train_writer, step_g)
            running_avg_d1 = _running_avg('D1', running_avg_d1, np.mean(d1), train_writer, step_g)
            running_avg_d2 = _running_avg('D2', running_avg_d2, np.mean(d2), train_writer, step_g)
            guess_scores.extend(z_scores)
            gold_scores.extend(scores_train[i + j + 1])
        end = timer()
        step += 1
        train_writer.flush()
        evals = _evaluate_lists(guess_scores, gold_scores).tolist()
        evals.insert(0, 'Train')
        table.append(evals)

        # Evaluate validation set
        guess_scores = []
        gold_scores = []
        for i in xrange(len(vectors_vali)):
            z_vectors = vectors_vali[i]
            z_scores_gold = scores_vali[i]
            z_scores_guess = model.run_eval_step(sess, z_vectors)
            guess_scores.extend(z_scores_guess)
            gold_scores.extend(z_scores_gold)
        evals = _evaluate_lists(guess_scores, gold_scores).tolist()
        evals.insert(0, 'Valid')
        table.append(evals)

        if step % FLAGS.eval_epoch_steps == 0:
            train_writer.flush()
            # Evaluate test set
            guess_scores = []
            gold_scores = []
            for i in xrange(len(vectors_test)):
                z_vectors = vectors_test[i]
                z_scores_gold = scores_test[i]
                z_scores_guess = model.run_eval_step(sess, z_vectors)
                guess_scores.extend(z_scores_guess)
                gold_scores.extend(z_scores_gold)
            evals = _evaluate_lists(guess_scores, gold_scores).tolist()
            evals.insert(0, 'Test')
            table.append(evals)
        print("\nEpoch %d; Time = %6.3fs; Loss[D] = %6.3f; Loss[G] = %6.3f; Performance:" %
              (step, end - start, running_avg_loss_d, running_avg_loss_g))
        print(tabulate(table, headers=['', 'MAP', 'MRR', 'P5', 'P10', 'P15', 'P20', 'N5', 'N10', 'N15', 'N20'],
                       numalign='right', tablefmt='grid'))

        # Shuffle training data around after each epoch
        t = list(zip(vectors_train, scores_train))
        random.shuffle(t)
        vectors_train, scores_train = zip(*t)

    sv.Stop()
    return running_avg_loss_d, running_avg_loss_g, running_avg_d1, running_avg_d2


def main(unused_argv):
    batch_size = FLAGS.batch_size
    max_list_len = FLAGS.max_list_length

    vectors_train, scores_train, _, _, feature_vector_dim, max_score = data_utils.load_letor_data(
        FLAGS.data_path + '/train.txt', batch_size, max_list_len)

    print('Maximum relevance score = %d' % max_score)
    print('Feature vector dimensionality = %d' % feature_vector_dim)

    vectors_devel, scores_devel, _, _, _, _ = data_utils.load_letor_data(
        FLAGS.data_path + '/vali.txt', batch_size, max_list_len)

    vectors_test, scores_test, _, _, _, _ = data_utils.load_letor_data(
        FLAGS.data_path + '/test.txt', batch_size, max_list_len)

    hps = gan_model.HParams(
        g_relu1_dim=64,
        g_relu2_dim=16,
        d_rnn_dim=128,
        d_relu_dim=16,
        d_min_lr=0.01,
        d_lr=0.15,
        d_ranking_dim=8,
        mu=0,
        sigma=1,
        keep_prob=.50)

    tf.set_random_seed(FLAGS.random_seed)

    model = gan_model.GANModel(max_score,
                               feature_vector_dim,
                               max_list_len,
                               batch_size,
                               hps)

    _train(model, batch_size,
           vectors_train, scores_train, vectors_devel, scores_devel, vectors_test, scores_test)


if __name__ == '__main__':
    tf.app.run()
