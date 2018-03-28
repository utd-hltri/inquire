from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from collections import namedtuple

HParams = namedtuple('HParams',
                     'g_relu1_dim, g_relu2_dim, ' +
                     'd_rnn_dim, d_relu_dim, d_min_lr, d_lr, d_ranking_dim, ' +
                     'mu, sigma, keep_prob')


# Designed for tensorflow v 1.0?

def _clamp_as_probability(tensor):
    return tf.maximum(tf.minimum(tensor, .99), .01)


def _leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)


class GANModel(object):
    def __init__(self,
                 max_score,
                 feature_vector_dim,
                 max_list_len,
                 batch_size,
                 hps):
        self._max_score = max_score
        self._feature_vector_dim = feature_vector_dim
        self._max_list_len = max_list_len
        self._batch_size = batch_size
        self._hps = hps
        self._g_step = tf.Variable(0, trainable=False)
        self._d_step = tf.Variable(0, trainable=False)
        self._keep_prob = tf.constant(hps.keep_prob)

    def _add_placeholders(self):
        batch_size = self._batch_size
        max_list_len = self._max_list_len
        feature_vector_dim = self._feature_vector_dim
        self._z_seeds = tf.placeholder(tf.float32, [batch_size, 1], name='z_seeds')
        self._z_vectors_stacked = tf.placeholder(tf.float32,
                                                 [batch_size, max_list_len, feature_vector_dim],
                                                 name='z_vectors_stacked')
        self._x_vectors_stacked = tf.placeholder(tf.float32,
                                                 [batch_size, max_list_len, feature_vector_dim],
                                                 name='x_vectors_stacked')
        self._x_scores_stacked = tf.placeholder(tf.float32,
                                                [batch_size, max_list_len],
                                                name='x_scores_stacked')
        self._is_training = tf.placeholder(tf.bool, [], name='is_training')

    def _add_generator(self):
        hps = self._hps
        max_score = self._max_score

        with tf.variable_scope('generator') as scope:
            # List of [BATCH_SIZE] x [VECTOR_SIZE]
            z_vectors = tf.unstack(self._z_vectors_stacked, axis=1)
            z_scores = []
            batch_norm_params = {'is_training': self._is_training, 'decay': 0.9, 'updates_collections': None}
            with tf.contrib.slim.arg_scope([tf.contrib.slim.fully_connected],
                                           activation_fn=_leaky_relu,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.constant_initializer(0.01),
                                           normalizer_fn=tf.contrib.slim.batch_norm,
                                           normalizer_params=batch_norm_params):
                for i, z_vector in enumerate(z_vectors):
                    if i > 0:
                        scope.reuse_variables()
                    z_vector_with_seed = tf.concat(axis=1, values=[z_vector, self._z_seeds])

                    g_layer_1 = tf.contrib.slim.fully_connected(z_vector_with_seed, hps.g_relu1_dim,
                                                                normalizer_fn=None, scope='g_layer_1')
                    g_layer_2 = tf.contrib.slim.fully_connected(g_layer_1, hps.g_relu2_dim, scope='g_layer_2')
                    g_layer_2_dropout = tf.contrib.slim.dropout(g_layer_2, hps.keep_prob, scope='dropout')
                    g_layer_3 = tf.contrib.slim.fully_connected(g_layer_2_dropout, 1, activation_fn=tf.nn.sigmoid,
                                                                normalizer_fn=None, scope='g_layer_3')
                    z_score = _clamp_as_probability(g_layer_3) * max_score
                    z_scores.append(tf.squeeze(z_score))

            self._z_scores_stacked = tf.stack(z_scores, axis=1, name='z_scores_stacked')
            self._theta_g = [v for v in tf.global_variables() if v.name.startswith(scope.name)]

    def _add_discriminator_copy(self, vectors_stacked, scores_stacked):
        max_list_len = self._max_list_len
        hps = self._hps

        # Sort stacked vector and score tensors by scores
        sorted_rolled_scores, indices = tf.nn.top_k(scores_stacked, k=max_list_len, sorted=True, name="sorted_scores")
        vector_indices = tf.concat(axis=2, values=[self._indices_helper, tf.expand_dims(indices, -1)], name="rolled_vector_indices")
        vectors_stacked_sorted = tf.gather_nd(vectors_stacked, vector_indices, 'sorted_vectors_stacked')

        # Layer 1: RNN over unstacked (vector+score) inputs
        # inputs_stacked = tf.concat(2,
        #     [vectors_stacked_sorted, tf.expand_dims(sorted_rolled_scores, -1)],
        #     name='inputs_stacked')
        # inputs = tf.unstack(inputs_stacked, axis=1)
        inputs = tf.unstack(vectors_stacked_sorted, axis=1)
        pairwise_stacked_vectors = vectors_stacked_sorted[:, :max_list_len - 1, :] - vectors_stacked_sorted[:, 1:, :]
        pairwise_stacked_scores = sorted_rolled_scores[:, :max_list_len - 1] - sorted_rolled_scores[:, 1:]
        # scores = tf.unstack(sorted_rolled_scores, axis=1)
        # ranking_matrix = tf.stack([self._ranking_weights for _ in range(batch_size)])
        # ranking_vectors = tf.unstack(ranking_matrix, axis=1)

        batch_norm_params = {'is_training': self._is_training, 'decay': 0.9, 'updates_collections': None}
        with tf.contrib.slim.arg_scope([tf.contrib.slim.fully_connected],
                                       activation_fn=_leaky_relu,
                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       biases_initializer=tf.constant_initializer(0.01),
                                       normalizer_fn=tf.contrib.slim.batch_norm,
                                       normalizer_params=batch_norm_params):
            with tf.variable_scope("flattening") as scope:
                pairwise_inputs = []
                for i, (v, s) in enumerate(
                        zip(tf.unstack(pairwise_stacked_vectors, axis=1), tf.unstack(pairwise_stacked_scores, axis=1))):
                    if i > 0:
                        scope.reuse_variables()
                    d_layer_0a = tf.contrib.slim.fully_connected(v, hps.g_relu1_dim,
                                                                 normalizer_fn=None, scope='d_layer0a')
                    d_layer_0b = tf.contrib.slim.fully_connected(d_layer_0a, 1, scope='d_layer0b')
                    pairwise_inputs.append(tf.tanh(tf.scalar_mul(self._ranking_weights[i], d_layer_0b - s)))
                    # print(pairwise_score)
                    # print(ranking_vectors[i])
                    # pairwise_input = tf.concat(1, [pairwise_vector, pairwise_score])#, ranking_vectors[i]])
                    # pairwise_inputs.append(pairwise_input)
                    # pairwise_inputs.append(pairwise_vector)
            # d_input = tf.concat(2, [tf.stack(pairwise_inputs, axis=1), ranking_matrix])
            # print(d_input)
            # print(ranking_matrix)
            d_input = tf.concat(axis=1, values=pairwise_inputs)  # pairwise_inputs, axis=1)
            # d_input = tf.reshape(d_input, [batch_size, (max_list_len - 1) * (feature_vector_dim + 1)])

            # outputs, _, _ = tf.nn.bidirectional_rnn(self._rnn_cell_fw, self._rnn_cell_bw, pairwise_inputs, scope="d_rnn", dtype=tf.float32)
            # d_layer_1 = outputs[0]

            # batch_norm_params = {'is_training': self._is_training, 'decay': 0.9, 'updates_collections': None}
            # with tf.contrib.slim.arg_scope([tf.contrib.slim.fully_connected],
            #                                activation_fn=_leaky_relu,
            #                                weights_initializer=tf.contrib.layers.xavier_initializer(),
            #                                biases_initializer=tf.constant_initializer(0.01),
            #                                normalizer_fn=tf.contrib.slim.batch_norm,
            #                                normalizer_params=batch_norm_params):
            d_layer_1 = tf.contrib.slim.fully_connected(d_input, hps.d_relu_dim, normalizer_fn=None, scope='d_layer_1')
            d_layer_2 = tf.contrib.slim.fully_connected(d_layer_1, hps.d_relu_dim, scope='d_layer_2')
            d_layer_3 = tf.contrib.slim.fully_connected(d_layer_2, 1, activation_fn=tf.nn.sigmoid,
                                                        normalizer_fn=None, scope='d_layer_3')
        return _clamp_as_probability(d_layer_3)

    def _add_discriminator(self):
        hps = self._hps
        batch_size = self._batch_size
        max_list_len = self._max_list_len

        with tf.variable_scope('discriminator'):
            self._ranking_weights = [1.0 / i for i in range(1, max_list_len)]

            # Helper tensor used to sort stacked feature vector tensor
            rows = []
            for i in xrange(batch_size):
                row = tf.fill([max_list_len, 1], i)
                rows.append(row)
            self._indices_helper = tf.stack(rows)

            # RNN cell shared by discriminators
            # self._rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(hps.d_rnn_dim)
            # self._rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(hps.d_rnn_dim)

            # Discriminator likelihoods
            with tf.name_scope('d1'):
                self._d1 = self._add_discriminator_copy(self._x_vectors_stacked, self._x_scores_stacked)
        with tf.variable_scope('discriminator', reuse=True), tf.name_scope('d2'):
                self._d2 = self._add_discriminator_copy(self._z_vectors_stacked, self._z_scores_stacked)

        with tf.variable_scope('discriminator') as scope:
            self._theta_d = [v for v in tf.global_variables() if v.name.startswith(scope.name)]

            # Discriminator pre-training
            self._d_pre_step = tf.Variable(0, trainable=False)
            self._x_labels_stacked = tf.placeholder(tf.float32, [batch_size], name='x_labels_stacked')
            self._d_pre_loss = tf.reduce_mean(tf.square(self._d1 - self._x_labels_stacked))
            self._d_pre_opt = tf.train.AdamOptimizer().minimize(self._d_pre_loss,
                                                                global_step=self._d_pre_step,
                                                                var_list=self._theta_d)

    def _add_train_ops(self):
        hps = self._hps
        self._obj_d = tf.reduce_mean(tf.log(self._d1) + tf.log(1 - self._d2))
        self._loss_d = 1 - self._obj_d
        self._lr_rate_d = tf.maximum(
            hps.d_min_lr,  # min_lr_rate.
            tf.train.exponential_decay(hps.d_lr, self._d_step, 30000, 0.98))
        self._opt_d = tf.train.GradientDescentOptimizer(0.01).minimize(self._loss_d,
                                                                       global_step=self._d_step,
                                                                       var_list=self._theta_d)
        self._obj_g = tf.reduce_mean(tf.log(self._d2))
        # In GAN papers, the loss function to optimize G is min (log 1-D),
        # but in practice folks practically use max log D
        # because the first formulation has vanishing gradients early on
        # Goodfellow et. al (2014)
        # from: https://github.com/soumith/ganhacks
        self._loss_g = -self._obj_g
        self._opt_g = tf.train.AdamOptimizer().minimize(self._loss_g,
                                                        global_step=self._g_step,
                                                        var_list=self._theta_g)

    def build_graph(self):
        self._add_placeholders()
        self._add_generator()
        self._add_discriminator()
        self._add_train_ops()

    def run_pretrain_step(self, sess, x_vectors, x_scores):
        hps = self._hps
        batch_size = self._batch_size
        z_vectors = np.zeros((self._batch_size, self._max_list_len, self._feature_vector_dim))
        z_seeds = np.zeros((batch_size, 1))

        x_vectors_pre = []
        x_scores_pre = []
        x_labels_pre = []
        for i in xrange(len(x_scores)):
            x_vectors_pre.append(x_vectors[i])
            if i % 2 == 0:
                x_scores_pre.append(np.random.permutation(x_scores[i]))
                x_labels_pre.append(0)
            else:
                x_scores_pre.append(x_scores[i])
                x_labels_pre.append(1)
        _, d1, loss_d, step_d = sess.run(
            [self._d_pre_opt, self._d1, self._d_pre_loss, self._d_pre_step],
            feed_dict={self._is_training: True,
                       self._x_vectors_stacked: x_vectors,
                       self._x_scores_stacked: x_scores,
                       self._z_vectors_stacked: z_vectors,
                       self._z_seeds: z_seeds,
                       self._x_labels_stacked: np.asarray(x_labels_pre)})
        return d1, loss_d, step_d

    def run_train_d_step(self, sess, x_vectors, x_scores, z_vectors):
        hps = self._hps
        batch_size = self._batch_size
        x_labels = np.zeros(batch_size)
        z_seeds = np.random.normal(hps.mu, hps.sigma, (batch_size, 1))
        _, d1, obj_d, loss_d, step_d = sess.run(
            [self._opt_d, self._d1, self._obj_d, self._loss_d, self._d_step],
            feed_dict={self._is_training: True,
                       self._x_vectors_stacked: x_vectors,
                       self._x_scores_stacked: x_scores,
                       self._z_vectors_stacked: z_vectors,
                       self._z_seeds: z_seeds,
                       self._x_labels_stacked: x_labels})
        return d1, obj_d, loss_d, step_d

    def run_train_g_step(self, sess, z_vectors):
        hps = self._hps
        batch_size = self._batch_size
        x_labels = np.zeros(batch_size)
        x_vectors = np.zeros((self._batch_size, self._max_list_len, self._feature_vector_dim))
        x_scores = np.zeros((self._batch_size, self._max_list_len))
        z_seeds = np.random.normal(hps.mu, hps.sigma, (batch_size, 1))
        _, d2, z_scores, obj_g, loss_g, step_g = sess.run(
            [self._opt_g, self._d2, self._z_scores_stacked, self._obj_g, self._loss_g, self._g_step],
            feed_dict={
                self._is_training: True,
                self._x_vectors_stacked: x_vectors,
                self._x_scores_stacked: x_scores,
                self._z_vectors_stacked: z_vectors,
                self._z_seeds: z_seeds,
                self._x_labels_stacked: x_labels})
        return d2, z_scores, obj_g, loss_g, step_g

    def run_train_step(self, sess, x_vectors1, x_scores1, z_vectors1, z_vectors2):
        d1, obj_d, loss_d, step_d = self.run_train_d_step(sess, x_vectors1, x_scores1, z_vectors1)
        d2, z_scores, obj_g, loss_g, step_g = self.run_train_g_step(sess, z_vectors2)
        return d1, d2, loss_d, loss_g, z_scores, step_d, step_g

    def run_eval_step(self, sess, z_vectors):
        hps = self._hps
        batch_size = self._batch_size
        z_seeds = np.random.normal(hps.mu, hps.sigma, (batch_size, 1))
        # The generator doesn't use x_vectors or x_scores, but Tensorflow requires
        # all placeholders to be provided, so we give blank tensors
        x_vectors = np.zeros((self._batch_size, self._max_list_len, self._feature_vector_dim))
        x_scores = np.zeros((self._batch_size, self._max_list_len))
        return sess.run(self._z_scores_stacked,
                        feed_dict={self._is_training: False,
                                   self._x_vectors_stacked: x_vectors,
                                   self._x_scores_stacked: x_scores,
                                   self._z_vectors_stacked: z_vectors,
                                   self._z_seeds: z_seeds})
