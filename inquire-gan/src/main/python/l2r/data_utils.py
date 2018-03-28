from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function

from sklearn.datasets import load_svmlight_file
from timeit import default_timer as timer
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin


class LetorBatcher(object):
    def __init__(self, x, y, lens, qid, dids, batch_size, max_len, num_feats):
        x_batched = []
        y_batched = []
        qid_batched = []
        lens_batched = []
        dids_batched = []
        for i in xrange(0, len(x) - batch_size, batch_size):
            x_batch = []
            y_batch = []
            qid_batch = qid[i:i + batch_size]
            lens_batch = lens[i:i + batch_size]
            dids_batch = []
            for j in xrange(batch_size):
                _dids = [None for _ in range(max_len)]
                _X = np.zeros([max_len, num_feats], dtype=np.float32)
                _y = np.zeros([max_len], dtype=np.float32)
                n = min(x[i + j].shape[0], max_len)
                assert x[i + j].shape[0] == y[i + j].shape[0]
                indices = np.random.permutation(n)
                _X[indices] = x[i + j][0:n]
                _y[indices] = y[i + j][0:n]
                for p, q in enumerate(indices):
                    _dids[q] = dids[i + j][p]
                x_batch.append(_X)
                y_batch.append(_y)
                dids_batch.append(_dids)
                # print(dids_batch)
            x_batched.append(x_batch)
            y_batched.append(y_batch)
            qid_batched.append(qid_batch)
            lens_batched.append(lens_batch)
            dids_batched.append(dids_batch)
        num_batches = len(x_batched)
        assert len(x_batched) == len(y_batched) == len(qid_batched)

        print("Number of batches = %d" % num_batches)
        self.vectors = np.asarray(x_batched)
        self.scores = np.asarray(y_batched)
        self.qids = np.asarray(qid_batched)
        self.lens = np.asarray(lens_batched)
        self.dids = np.asarray(dids_batched)


class LetorLoader(object):

    def _load_dids(self, path):
        import re

        pattern = re.compile('#\s*docid\s*=\s*([a-zA-Z0-9_-]+)')

        dids = []
        with open(path, 'rb') as f:
            for line in f:
                match = pattern.search(line)
                did = match.group(1)
                dids.append(did)
        return dids



    def __init__(self, path):
        # Initially, vectors are flat
        start = timer()
        x_flat, y_flat, qid_flat = load_svmlight_file(path, dtype=np.float32, query_id=True)
        end = timer()
        print("Loaded SVMLight file '%s' in %6.3f seconds" % (path, end - start))

        did_flat = np.asarray(self._load_dids(path))

        self.max_score = y_flat.max()

        # First, sort instances by list length
        start = timer()
        t = list(zip(x_flat.toarray(), y_flat, qid_flat, did_flat))
        t.sort(key=lambda p: p[0].shape[0])
        x_flat, y_flat, qid_flat, diod_flat = zip(*t)
        x_flat = np.asarray(x_flat)
        y_flat = np.asarray(y_flat)
        qid_flat = np.asarray(qid_flat)


        # Flatten to numpy arrays
        self.num_feats = x_flat.shape[1]
        end = timer()
        print(" - Converted to flat numpy arrays in %6.3f seconds" % (end - start))

        # We need to group them into buckets, one for each qid
        start = timer()
        qids, index, counts = np.unique(qid_flat, return_index=True, return_counts=True)
        n_qids = len(qids)
        x = []
        y = []
        qid = []
        dids = []
        lens = []
        for i in xrange(n_qids):
            start_index = index[i]
            end_index = index[i] + counts[i]
            xs = x_flat[start_index:end_index]
            x.append(xs)
            y.append(y_flat[start_index:end_index])
            qid.append(qids[i])
            dids.append(did_flat[start_index:end_index])
            lens.append(len(xs))
        end = timer()
        print(" - Bucketed SVMLight vectors in %6.3f seconds" % (end - start))

        self._x = np.asarray(x)
        self._y = np.asarray(y)
        self._qid = np.asarray(qid)
        self._lens = np.asarray(lens)
        self._dids = np.asarray(dids)
        self.max_len = self._lens.max()

    def make_batches(self, batch_size, max_len):
        return LetorBatcher(self._x, self._y, self._lens, self._qid, self._dids, batch_size, max_len, self.num_feats)
