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
tf.app.flags.DEFINE_integer('pretrain_epochs', 10, 'Number of epochs to pretrain the discriminator')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Number of queries per batch')
tf.app.flags.DEFINE_string('max_list_length', 128,
                           'Maximum number of documents to consider for each query')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_integer('max_run_steps', 10000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('eval_epoch_steps', 300, 'How often to run eval.')
tf.app.flags.DEFINE_integer('validate_epoch_steps', 100, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 600, 'How often to checkpoint.')
tf.app.flags.DEFINE_integer('random_seed', 1337, 'A seed value for randomness.')
tf.app.flags.DEFINE_integer('d_steps_per_g_step', 4,
                            'How many times the discriminator will be trained for every generator training step.')
tf.app.flags.DEFINE_boolean('smooth_ranks', True, 'Smooth ranks by introducing noise (e.g. 5 -> sample(4.7, 6.12)')
tf.app.flags.DEFINE_float('smooth_rank_noise', 1, 'Noise used to smooth ranks')
tf.app.flags.DEFINE_boolean('batch_norm', False, 'use batch normalization on discriminator')
tf.app.flags.DEFINE_boolean('layer_norm', False, 'use layer normalization on discriminator')
tf.app.flags.DEFINE_boolean('dual_disc_loss', False, 'use dual discriminator loss')
tf.app.flags.DEFINE_boolean('dual_gen_loss', False, 'use dual generator loss')
tf.app.flags.DEFINE_string('mode', 'Train', 'whether to train or evaluate')
tf.app.flags.DEFINE_string('runtag', 'GAN', 'runtag to use for evaluation')
tf.app.flags.DEFINE_string('eval_out', 'output', 'path to output')

def _running_avg(name, running_avg_loss, loss, summary_writer, step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12.0)
    loss_sum = tf.Summary()
    loss_sum.value.add(tag='running_avg_' + name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    # print('running_avg_%s: %f\n' % (name, running_avg_loss))
    return running_avg_loss


def _evaluate_ranking(rs):
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


def _evaluate_list_scores(guess_lists, gold_lists):
    evaluations = []
    for guess, gold in izip(guess_lists, gold_lists):
        rs = [x[1] for x in sorted(izip(guess, gold), key=lambda xt: -xt[0])]
        evaluations.append(_evaluate_ranking(rs))
    return np.mean(evaluations, axis=0)


def _evaluate_list_ranks(guess_ranks, gold_lists):
    evaluations = []
    for guess, gold in izip(guess_ranks, gold_lists):
        rs = [x[1] for x in sorted(izip(guess, gold), key=lambda xt: xt[0])]
        evaluations.append(_evaluate_ranking(rs))
    return np.mean(evaluations, axis=0)


def _get_ranks(scores):
    seeds = np.random.random(scores.shape)
    idx = np.lexsort((seeds, -scores))
    ranks = np.zeros_like(scores)
    for i in xrange(ranks.shape[0]):
        ranks[i, idx[i]] = np.asarray(range(len(idx[i]))) + 1
    if FLAGS.smooth_ranks:
        noise = np.random.uniform(low=-FLAGS.smooth_rank_noise, high=FLAGS.smooth_rank_noise, size=scores.shape)
        noisy_ranks = noise + ranks
        return noisy_ranks
    else:
        return ranks


def _evaluate(name, guess_scores, gold_scores, table):
    evals = _evaluate_list_ranks(guess_scores, gold_scores).tolist()
    # noinspection PyUnresolvedReferences
    evals.insert(0, name)
    table.append(evals)


def _train(model, train, devel, test):
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
        log_device_placement=False))
    end = timer()
    print('Model constructed in %6.3fs' % (end - start))

    print('Pre-training D...')
    for k in xrange(FLAGS.pretrain_epochs):
        running_avg_loss_d = 0
        running_avg_d1 = 0
        for i in xrange(len(train.vectors)):
            d1, loss_d, step_d = model.run_pretrain_step(sess, train.vectors[i],
                                                         _get_ranks(train.scores[i]), train.lens[i])
            running_avg_d1 = _running_avg('D1', running_avg_d1, np.mean(d1), train_writer, step_d)
            running_avg_loss_d = _running_avg('loss_d', running_avg_loss_d, loss_d, train_writer, step_d)
        t = list(zip(train.vectors, train.scores))
        random.shuffle(t)
        train.vectors, train.scores = zip(*t)
        print('Pre-training epoch %d; Loss = %6.3f; D1 = %6.3f' % (k + 1, running_avg_loss_d, running_avg_d1))

    print('Training...')
    running_avg_loss_d = 0
    running_avg_loss_g = 0
    running_avg_d1 = 0
    running_avg_d2 = 0
    step = 0
    while not sv.should_stop() and step < FLAGS.max_run_steps:
        table = []
        guess_ranks = []
        guess_scores = []
        gold_scores = []
        start = timer()
        for i in xrange(0, len(train.vectors) - (2 * FLAGS.d_steps_per_g_step + 1), 2 * FLAGS.d_steps_per_g_step + 1):
            d1 = 0
            for j in xrange(FLAGS.d_steps_per_g_step):
                d1, _, loss_d, step_d = model.run_train_d_step(sess,
                                                               x_vectors=train.vectors[i + j],
                                                               x_ranks=_get_ranks(train.scores[i + j]),
                                                               x_lens=train.lens[i + j],
                                                               z_vectors=train.vectors[i + j + 1],
                                                               z_lens=train.lens[i + j + 1])
                running_avg_loss_d = _running_avg('loss_d', running_avg_loss_d, loss_d, train_writer, step_d)
            d2, z_ranks, z_scores, _, loss_g, step_g = model.run_train_g_step(sess,
                                                                     train.vectors[i + j + 1],
                                                                     train.lens[i + j + 1],
                                                                     train.scores[i + j + 1])
            running_avg_loss_g = _running_avg('loss_g', running_avg_loss_g, loss_g, train_writer, step_g)
            running_avg_d1 = _running_avg('D1', running_avg_d1, np.mean(d1), train_writer, step_g)
            running_avg_d2 = _running_avg('D2', running_avg_d2, np.mean(d2), train_writer, step_g)
            guess_ranks.extend(z_ranks)
            guess_scores.extend(z_scores)
            gold_scores.extend(train.scores[i + j + 1])
        end = timer()
        step += 1
        train_writer.flush()
        _evaluate('Train', guess_ranks, gold_scores, table)

        idx = np.argsort(np.asarray(gold_scores[0]))[:-11:-1]
        print('Guess Scores:', ['{:.3f}'.format(i) for i in guess_scores[0][idx]])
        print('Guess Ranks:', guess_ranks[0][idx])
        print('Gold Scores:', gold_scores[0][idx])
        print('Gold Ranks:', ['{:.3f}'.format(i) for i in _get_ranks(np.asarray(gold_scores))[0, idx]])

        # Evaluate validation set
        if step % FLAGS.validate_epoch_steps == 0:
            train_writer.flush()
            guess_ranks = []
            gold_scores = []
            for i in xrange(len(devel.vectors)):
                z_scores_gold = devel.scores[i]
                z_ranks_guess, _ = model.run_eval_step(sess, devel.vectors[i], devel.lens[i])
                guess_ranks.extend(z_ranks_guess)
                gold_scores.extend(z_scores_gold)
            _evaluate('Devel', guess_ranks, gold_scores, table)

        if step % FLAGS.eval_epoch_steps == 0:
            train_writer.flush()
            # Evaluate test set
            guess_ranks = []
            gold_scores = []
            for i in xrange(len(test.vectors)):
                z_scores_gold = test.scores[i]
                z_ranks_guess, _ = model.run_eval_step(sess, test.vectors[i], test.lens[i])
                guess_ranks.extend(z_ranks_guess)
                gold_scores.extend(z_scores_gold)
            _evaluate('Test', guess_ranks, gold_scores, table)
        print("\nEpoch %d; Time = %6.4fs; Loss[D] = %6.4f; Loss[G] = %6.4f; Performance:" %
              (step, end - start, running_avg_loss_d, running_avg_loss_g))
        print(tabulate(table, headers=['', 'MAP', 'MRR', 'P5', 'P10', 'P15', 'P20', 'N5', 'N10', 'N15', 'N20'],
                       numalign='right', tablefmt='simple'))

        # Shuffle training data around after each epoch
        t = list(zip(train.vectors, train.scores, train.lens))
        random.shuffle(t)
        train.vectors, train.scores, train.lens = zip(*t)

    sv.Stop()
    return running_avg_loss_d, running_avg_loss_g, running_avg_d1, running_avg_d2

def evaluate_model(model, test, ckpt, file, runtag='gan'):
    dids = []
    guess_scores = []
    guess_ranks = []
    qids = []
    gold_scores = []

    model.build_graph()

    sv = tf.train.Supervisor(logdir=FLAGS.log_root)

    with sv.managed_session() as sess:
        for i in xrange(len(test.vectors)):
            b_ranks, b_scores = model.run_eval_step(sess, test.vectors[i], test.lens[i])
            for b in xrange(b_ranks.shape[0]):
                for rank, score, rel, did in zip(b_ranks[b], b_scores[b], test.scores[i][b], test.dids[i][b]):
                    dids.append(did)
                    guess_scores.append(score)
                    guess_ranks.append(rank)
                    gold_scores.append(int(rel))
                    qids.append(test.qids[i][b])

    import trec_utils

    trec_utils.write_trec_submission_file(gold_scores, qids, dids, file + '.run', runtag)
    trec_utils.write_trec_qrels_file(qids, dids, gold_scores, file + '.qrels')

    import time

    time.sleep(1)
    trec_utils.run_trec_eval(file + '.qrels', file + '.run')


def main(unused_argv):
    train = data_utils.LetorLoader(os.path.join(FLAGS.data_path, 'train.txt'))
    devel = data_utils.LetorLoader(os.path.join(FLAGS.data_path, 'vali.txt'))
    test = data_utils.LetorLoader(os.path.join(FLAGS.data_path, 'test.txt'))

    max_score = train.max_score
    print('Maximum relevance score = %d' % max_score)

    num_feats = train.num_feats
    print('Feature vector dimensionality = %d' % num_feats)

    if train.max_len < FLAGS.max_list_length:
        print('Lowering max list length to %d (down from %d)' % (train.max_len, FLAGS.max_list_length))
        max_len = train.max_len
    else:
        max_len = FLAGS.max_list_length
        print('Maximum list length: %d' % max_len)

    hps = gan_model.HParams(
        g_relu1_dim=64,
        g_relu2_dim=32,
        d_rnn_dim=128,
        d_relu_dim=64,
        d_min_lr=1e-6,
        d_lr=1e-2,
        d_ranking_dim=12,
        d_batch_norm=FLAGS.batch_norm,
        d_layer_norm=FLAGS.layer_norm,
        mu=0,
        sigma=1,
        keep_prob=.50)

    tf.set_random_seed(FLAGS.random_seed)

    batch_size = FLAGS.batch_size

    model = gan_model.GANModel(max_score,
                               num_feats,
                               max_len,
                               batch_size,
                               hps,
                               FLAGS.dual_gen_loss,
                               FLAGS.dual_disc_loss)

    if (FLAGS.mode.lower() == 'eval'):
        evaluate_model(model, test.make_batches(batch_size, max_len), FLAGS.log_root, FLAGS.eval_out, FLAGS.runtag)
    else:
        _train(model,
               train.make_batches(batch_size, max_len),
               devel.make_batches(batch_size, max_len),
               test.make_batches(batch_size, max_len))


if __name__ == '__main__':
    tf.app.run()
