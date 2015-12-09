from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
from collections import defaultdict, Counter

class Word2Vec(object):
    def __init__(self, config):
        self.gpu = config.gpu
        self.alpha = config.alpha
        self.stream = config.stream
        self.neg_sample_size = config.neg_sample_size
        self.min_frequency = config.min_frequency
        self.window = config.window
        self.lr = config.lr
        self.min_lr = config.min_lr
        self.table_size = config.table_size # unigram table size

    def build_vocab(self, filename):
        start_time = time.time()
        with open(filename) as f
            words = [word for line in f.readlines() for word in line.split()]
        self.total_count = len(words)
        self.counter = []
        self.counter.extend([item for item in collections.Counter(words).most_common()
                          if item[0] > self.min_frequency])
        self.vocab_size = len(self.counter)
        word2idx = dict()
        for word, _ in self.counter:
            word2idx[word] = len(word2idx)
        data = list()
        unk_count = 0
        for word in words:
            if word in word2idx:
                index = word2idx[word]
            else:
                index = 0 # word2idx['UNK']
                unk_count = unk_count + 1
            data.append(index)
        self.count[0][1] = unk_count
        idx2word = dict(zip(word2idx.values(), word2idx.keys()))
        duration = time.time() - start_time
        print("%d words processed in %.2f seconds", % (self.total_count, duration))
        print("Vocab size after eliminating words occuring less than %d times: %d", % (self.min_frequency, self.vocab_size))
        self.data = data
        self.words = words
        self.word2idx = word2idx 
        self.idx2word = idx2word
        self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)

    def build_table(self):
        start_time = time.time()
        total_count_pow = 0
        for _, count in self.counter:
            total_count_pow += math.pow(count, self.alpha)
        word_index = 0
        word_prob = math.pow(self.counter[word_index], self.alph) / total_count_pow
        for idx in xrange(self.table_size):
            self.table[idx] = word_index
            if idx / self.table_size > word_prob:
                word_index += 1
                word_prob += math.pow(self.counter[word_index], self.alph) / total_count_pow
            if word_index > self.vocab_size:
                word_index = word_index - 1
        print(string.format("Done in %.2f seconds.", time.time() - start))

    def train_pair(word, contexts):
        self.sess.run(self.train, feed_dict={self.word: word, self.contexts: contexts})

    def sample_contexts(context):
        self.contexts[0] = context
        for idx in xrange(sef.neg_smaple_size):
            neg_context = self.table[random.rand(self.table_size)]
