from __future__ import print_function

import subprocess


# Run an external command pipe it to stdout
def _run_command(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(*p.stdout.readlines())
    return p.wait()


# Produce a TREC submission file
def write_trec_submission_file(scores, qids, dids, path, runtag='gan'):
    with open(path, 'wb') as dest:
        for score, qid, did, in zip(scores, qids, dids):
            if did:
                print(qid, 'Q0', did, '1', score, runtag, file=dest)

# Produce a TREC qrels file
def write_trec_qrels_file(qids, dids, relevances, path):
    with open(path, 'wb') as dest:
        for qid, did, relevance in zip(qids, dids, relevances):
            if did:
                print(qid, '0', did, relevance, file=dest)


def run_trec_eval(qrels, submission, options=''):
    command = 'trec_eval %s %s %s' % (options, qrels, submission)
    _run_command(command)