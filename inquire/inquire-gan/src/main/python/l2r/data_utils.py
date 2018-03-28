from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from sklearn.datasets import load_svmlight_file
from timeit import default_timer as timer
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin


def _group_into_batches(x, y, qid, batch_size, max_len, num_feats):
    x_batched = []
    y_batched = []
    qid_batched = []
    for i in xrange(0, len(x) - batch_size, batch_size):
        X_batch = []
        y_batch = []
        qid_batch = qid[i:i + batch_size]
        for j in xrange(batch_size):
            _X = np.zeros([max_len, num_feats], dtype=np.float32)
            _y = np.zeros([max_len], dtype=np.float32)
            n = min(x[i + j].shape[0], max_len)
            assert x[i + j].shape[0] == y[i + j].shape[0]
            indices = np.random.permutation(n)
            _X[indices] = x[i + j][0:n]
            _y[indices] = y[i + j][0:n]
            X_batch.append(_X)
            y_batch.append(_y)
        x_batched.append(X_batch)
        y_batched.append(y_batch)
        qid_batched.append(qid_batch)
    num_batches = len(x_batched)
    assert len(x_batched) == len(y_batched) == len(qid_batched)
    print("Number of batches = %d" % num_batches)
    x_batched = np.asarray(x_batched)
    y_batched = np.asarray(y_batched)
    qid_batched = np.asarray(qid_batched)
    #    X_batched.resize(num_batches, batch_size, max_len, num_feats)
    #    y_batched.resize(num_batches, batch_size, max_len)
    return x_batched, y_batched, qid_batched


def load_letor_data(path, batch_size, max_len):
    # Initially, vectors are flat
    start = timer()
    x_flat, y_flat, qid_flat = load_svmlight_file(path, dtype=np.float32, query_id=True)
    end = timer()
    print("Loaded SVMLight file '%s' in %6.3f seconds" % (path, end - start))

    max_score = y_flat.max()

    print(x_flat.shape)
    print(y_flat.shape)
    print(qid_flat.shape)

    # First, sort instances by list length
    start = timer()
    t = list(zip(x_flat.toarray(), y_flat, qid_flat))
    t.sort(key=lambda p: p[0].shape[0])
    x_flat, y_flat, qid_flat = zip(*t)
    x_flat = np.asarray(x_flat)
    y_flat = np.asarray(y_flat)
    qid_flat = np.asarray(qid_flat)

    # Flatten to numpy arrays
    feature_vector_dim = x_flat.shape[1]
    end = timer()
    print(" - Converted to flat numpy arrays in %6.3f seconds" % (end - start))

    # We need to group them into buckets, one for each qid
    start = timer()
    qids, index, counts = np.unique(qid_flat, return_index=True, return_counts=True)
    n_qids = len(qids)
    x = []
    y = []
    qid = []
    lens = []
    for i in xrange(n_qids):
        start_index = index[i]
        end_index = index[i] + counts[i]
        x.append(x_flat[start_index:end_index])
        y.append(y_flat[start_index:end_index])
        qid.append(qids[i])
        lens.append(len(x) for x in x_flat[start_index:end_index])
    end = timer()
    print(" - Bucketed SVMLight vectors in %6.3f seconds" % (end - start))

    x_batched, y_batched, qid_batched = _group_into_batches(np.array(x), np.array(y), np.array(qid),
                                                            batch_size, max_len, feature_vector_dim)
    return x_batched, y_batched, qid_batched, lens, feature_vector_dim, max_score
