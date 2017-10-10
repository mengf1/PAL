import numpy as np
import re
import itertools
from collections import Counter
import tensorflow as tf

# map a label to a string
label2str = {1: "PER", 2: "LOC", 3: "ORG", 4: "MISC", 5: "O"}

# predefine a label_set: PER - 1, LOC - 2, ORG - 3, MISC - 4, O - 5
# 0 is for padding
labels_map = {'B-ORG': 3, 'O': 5, 'B-MISC': 4, 'B-PER': 1, 'I-PER': 1, 'B-LOC': 2, 'I-ORG': 3, 'I-MISC': 4, 'I-LOC': 2}

def load_data2labels(input_file):
    seq_set = []
    seq = []
    seq_set_label = []
    seq_label = []
    seq_set_len = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                seq_set.append(" ".join(seq))
                seq_set_label.append(seq_label)
                seq_set_len.append(len(seq_label))
                seq = []
                seq_label = []
            else:
                tok, label = line.split()
                seq.append(tok)
                seq_label.append(labels_map[label])
    return [seq_set, seq_set_label, seq_set_len]


def load_crosslingual_embeddings(input_file, vocab, max_vocab_size=20000):
    embeddings = list(open(input_file, "r").readlines())
    pre_w2v = {}
    emb_size = 0
    for emb in embeddings:
        parts = emb.strip().split()
        if emb_size != (len(parts) - 1):
            if emb_size == 0:
                emb_size = len(parts) - 1
            else:
                print "Different embedding size!"
                break

        w = parts[0]
        w_parts = w.split(":")
        if len(w_parts) != 2:
            w = ":"
        else:
            w = w_parts[1]
        vec = []
        for i in range(1, len(parts)):
            vec.append(float(parts[i]))
        # print w, vec
        pre_w2v[w] = vec

    n_dict = len(vocab._mapping)
    vocab_w2v = [None] * n_dict
    # vocab_w2v[0]=np.random.uniform(-0.25,0.25,100)
    for w, i in vocab._mapping.iteritems():
        if w in pre_w2v:
            vocab_w2v[i] = pre_w2v[w]
        else:
            vocab_w2v[i] = list(np.random.uniform(-0.25, 0.25, emb_size))

    cur_i = len(vocab_w2v)
    if len(vocab_w2v) > max_vocab_size:
        print "Vocabulary size is larger than", max_vocab_size
        raise SystemExit
    while cur_i < max_vocab_size:
        cur_i += 1
        padding = [0] * emb_size
        vocab_w2v.append(padding)
    print "Vocabulary", n_dict, "Embedding size", emb_size
    return vocab_w2v


def data2sents(X, Y):
    data = []
    for i in range(len(Y)):
        sent = []
        text = X[i]
        items = text.split()
        for j in range(len(Y[i])):
            sent.append((items[j], str(Y[i][j])))
        data.append(sent)
    return data
