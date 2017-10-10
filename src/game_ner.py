import numpy as np
import sys
import random
from tagger import CRFTagger
import helpers
#from gensim.models import doc2vec, word2vec
import tensorflow as tf


class NERGame:

    def __init__(self, story, test, dev, max_len, w2v, budget):
        # build environment
        # load data as story
        print("Initilizing the game:")
        # import story
        self.train_x, self.train_y, self.train_idx = story
        self.test_x, self.test_y, self.test_idx = test
        self.dev_x, self.dev_y, self.dev_idx = dev
        self.max_len = max_len
        self.w2v = w2v

        print "Story: length = ", len(self.train_x)
        self.order = range(0, len(self.train_x))
        # if re-order, use random.shuffle(self.order)
        # load word embeddings, pretrained - w2v
        # print "Dictionary size", len(self.w2v), "Embedding size",
        # len(self.w2v[0])

        # when queried times is 100, then stop
        self.budget = budget
        self.queried_times = 0

        # select pool
        self.queried_set_x = []
        self.queried_set_y = []
        self.queried_set_idx = []

        # let's start
        self.episode = 1
        # story frame
        self.current_frame = 0
        #self.nextFrame = self.current_frame + 1
        self.terminal = False
        self.make_query = False
        self.performance = 0

    def get_frame(self, model):
        self.make_query = False
        sentence = self.train_x[self.order[self.current_frame]]
        sentence_idx = self.train_idx[self.order[self.current_frame]]
        confidence = 0.
        predictions = []
        if model.name == "CRF":
            confidence = model.get_confidence(sentence)
            predictions = model.get_predictions(sentence)
        else:
            confidence = model.get_confidence(sentence_idx)
            predictions = model.get_predictions(sentence_idx)
        preds_padding = []
        orig_len = len(predictions)
        if orig_len < self.max_len:
            preds_padding.extend(predictions)
            for i in range(self.max_len - orig_len):
                preds_padding.append([0] * 5)
        elif orig_len > self.max_len:
            preds_padding = predictions[0:self.max_len]
        else:
            preds_padding = predictions

        obervation = [sentence_idx, confidence, preds_padding]
        return obervation

    # tagger = crf model
    def feedback(self, action, model):
        reward = 0.
        if action[1] == 1:
            self.make_query = True
            self.query()
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance
            if new_performance != self.performance:
                #reward = 3.
                self.performance = new_performance
            # else:
                #reward = -1.
        else:
            reward = 0.

        # next frame
        is_terminal = False
        if self.queried_times == self.budget:
            self.terminal = True
            is_terminal = True
            # update special reward
            # reward = new_performance * 100
            self.reboot()  # set the current frame = -1
        else:
            self.terminal = False
        next_sentence = self.train_x[self.order[self.current_frame + 1]]
        next_sentence_idx = self.train_idx[self.order[self.current_frame + 1]]

        confidence = 0.
        predictions = []
        if model.name == "CRF":
            confidence = model.get_confidence(next_sentence)
            predictions = model.get_predictions(next_sentence)
        else:
            confidence = model.get_confidence(next_sentence_idx)
            predictions = model.get_predictions(next_sentence_idx)
        preds_padding = []
        orig_len = len(predictions)
        if orig_len < self.max_len:
            preds_padding.extend(predictions)
            for i in range(self.max_len - orig_len):
                preds_padding.append([0] * 5)
        elif orig_len > self.max_len:
            preds_padding = predictions[0:self.max_len]
        else:
            preds_padding = predictions

        next_observation = [next_sentence_idx, confidence, preds_padding]
        self.current_frame += 1
        return reward, next_observation, is_terminal

    def query(self):
        if self.make_query == True:
            sentence = self.train_x[self.order[self.current_frame]]
            # simulate: obtain the labels
            labels = self.train_y[self.order[self.current_frame]]
            self.queried_times += 1
            # print "Select:", sentence, labels
            self.queried_set_x.append(sentence)
            self.queried_set_y.append(labels)
            self.queried_set_idx.append(
                self.train_idx[self.order[self.current_frame]])
            print "> Queried times", len(self.queried_set_x)

    # tagger = model
    def get_performance(self, tagger):
        # train with {queried_set_x, queried_set_y}
        # train with examples: self.model.train(self.queried_set_x,
        # self.queried_set_y)
        if tagger.name == "RNN":
            tagger.train(self.queried_set_idx, self.queried_set_y)
            performance = tagger.test(self.dev_idx, self.dev_y)
            return performance

        print len(self.queried_set_x), len(self.queried_set_y)
        train_sents = helpers.data2sents(
            self.queried_set_x, self.queried_set_y)
        # print train_sents
        tagger.train(train_sents)
        # test on development data
        test_sents = helpers.data2sents(self.dev_x, self.dev_y)
        performance = tagger.test(test_sents)
        #performance = self.model.test2conlleval(self.dev_x, self.dev_y)
        return performance

    def reboot(self):
        # resort story
        # why not use docvecs? TypeError: 'DocvecsArray' object does not
        # support item assignment
        random.shuffle(self.order)
        self.queried_times = 0
        self.terminal = False
        self.queried_set_x = []
        self.queried_set_y = []
        self.queried_set_idx = []
        self.current_frame = -1
        self.episode += 1
        print "> Next episode", self.episode