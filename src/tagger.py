import pycrfsuite
import os
import utilities
import random
import math


class CRFTagger(object):

    def __init__(self, modelfile):
        print("CRF Tagger")
        self.modelfile = modelfile
        self.name = "CRF"

    def word2features(self, sent, i):
        word = sent[i][0]
        #postag = sent[i][1]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            #'postag=' + postag,
            #'postag[:2]=' + postag[:2],
        ]
        if i > 0:
            word1 = sent[i - 1][0]
            #postag1 = sent[i - 1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                #'-1:postag=' + postag1,
                #'-1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('BOS')

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            #postag1 = sent[i + 1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                #'+1:postag=' + postag1,
                #'+1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('EOS')

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        # print sent
        return [label for token, label in sent]

    def sent2tokens(self, sent):
        return [token for token, label in sent]

    def train(self, train_sents):
        X_train = [self.sent2features(s) for s in train_sents]
        Y_train = [self.sent2labels(s) for s in train_sents]
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, Y_train):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train(self.modelfile)
        if len(trainer.logparser.iterations) != 0:
            print len(trainer.logparser.iterations), trainer.logparser.iterations[-1]
        else:
            # todo
            print len(trainer.logparser.iterations)
            print("There is no loss to present")

    # different lens
    def get_predictions(self, sent):
        x = self.sent2features(sent)
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.modelfile):
            y_marginals = []
            for i in range(len(x)):
                y_marginals.append([0.2] * 5)
            return y_marginals

        tagger.open(self.modelfile)
        tagger.set(x)
        y_marginals = []
        print tagger.labels()
        # if len(tagger.labels) < 5
        for i in range(len(x)):
            y_i = []
            for y in range(1, 6):
                if str(y) in tagger.labels():
                    y_i.append(tagger.marginal(str(y), i))
                else:
                    y_i.append(0.)
            y_marginals.append(y_i)
        return y_marginals

    # use P(yseq|xseq)
    def get_confidence(self, sent):
        x = self.sent2features(sent)
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.modelfile):
            confidence = 0.2
            return [confidence]

        tagger.open(self.modelfile)
        tagger.set(x)
        y_pred = tagger.tag()
        p_y_pred = tagger.probability(y_pred)
        confidence = pow(p_y_pred, 1. / len(y_pred))
        return [confidence]

    def get_uncertainty(self, sent):
        x = self.sent2features(sent)
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.modelfile):
            unc = random.random()
            return unc
        tagger.open(self.modelfile)
        tagger.set(x)
        ttk = 0.
        for i in range(len(x)):
            y_probs = []
            for y in range(1, 6):
                if str(y) in tagger.labels():
                    y_probs.append(tagger.marginal(str(y), i))
                else:
                    y_probs.append(0.)
            ent = 0.
            for y_i in y_probs:
                if y_i > 0:
                    ent -= y_i * math.log(y_i, 5)
            ttk += ent
        return ttk

    def test(self, test_sents):
        X_test = [self.sent2features(s) for s in test_sents]
        Y_true = [self.sent2labels(s) for s in test_sents]
        tagger = pycrfsuite.Tagger()
        tagger.open(self.modelfile)
        y_pred = [tagger.tag(xseq) for xseq in X_test]
        pre = 0
        pre_tot = 0
        rec = 0
        rec_tot = 0
        corr = 0
        total = 0
        for i in range(len(Y_true)):
            for j in range(len(Y_true[i])):
                total += 1
                if y_pred[i][j] == Y_true[i][j]:
                    corr += 1
                if utilities.label2str[int(y_pred[i][j])] != 'O':  # not 'O'
                    pre_tot += 1
                    if y_pred[i][j] == Y_true[i][j]:
                        pre += 1
                if utilities.label2str[int(Y_true[i][j])] != 'O':
                    rec_tot += 1
                    if y_pred[i][j] == Y_true[i][j]:
                        rec += 1

        res = corr * 1. / total
        print "Accuracy (token level)", res
        if pre_tot == 0:
            pre = 0
        else:
            pre = 1. * pre / pre_tot
        rec = 1. * rec / rec_tot
        print pre, rec

        beta = 1
        f1score = 0
        if pre != 0 or rec != 0:
            f1score = (beta * beta + 1) * pre * rec / \
                (beta * beta * pre + rec)
        print "F1", f1score
        return f1score
