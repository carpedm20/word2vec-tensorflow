from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import random
import numpy as np
import tensorflow as tf
from collections import Counter

class Word2Vec(object):
    def __init__(self, config, sess):
        self.sess = sess

        self.alpha = config['alpha']
        self.embed_size = config['embed_size']
        self.neg_sample_size = config['neg_sample_size']
        self.min_frequency = config['min_frequency']
        self.window = config['window']
        self.lr = config['lr']
        self.min_lr = config['min_lr']
        self.table_size = config['table_size'] # unigram table size

    def build_vocab(self, filename):
        start_time = time.time()
        with open(filename) as f:
            words = [word for line in f.readlines() for word in line.split()]
        self.total_count = len(words)
        self.counter = [['UNK', 0]]
        self.counter.extend([list(item) for item in Counter(words).most_common()
                          if item[0] > self.min_frequency])
        self.vocab_size = len(self.counter)
        word2idx = dict()
        for word, _ in self.counter:
            word2idx[word] = len(word2idx)
        data = list()
        unk_count = 0
        for word in words:
            if word in word2idx:
                idx = word2idx[word]
            else:
                idx = 0 # word2idx['UNK']
                unk_count = unk_count + 1
            data.append(idx)
        self.counter[0][1] = unk_count
        idx2word = dict(zip(word2idx.values(), word2idx.keys()))
        duration = time.time() - start_time

        print("%d words processed in %.2f seconds" % (self.total_count, duration))
        print("Vocab size after eliminating words occuring less than %d times: %d" % (self.min_frequency, self.vocab_size))

        self.data = data
        self.words = words
        self.word2idx = word2idx 
        self.idx2word = idx2word

        self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)
        self.labels = np.zeros([1, 1+self.neg_sample_size], dtype=np.float32); self.labels[0][0] = 1
        self.contexts = np.ndarray(1 + self.neg_sample_size, dtype=np.int32)
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.int32, [1], name='pos_x')
        self.y = tf.placeholder(tf.int32, [1 + self.neg_sample_size], name='pos_x')

        init_width = 0.5 / self.embed_size
        self.embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -init_width, init_width), name='embed')
        self.w = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1.0 / math.sqrt(self.embed_size)), name='w')

        self.x_embed = tf.nn.embedding_lookup(self.embed, self.x, name='pos_embed')
        self.y_w = tf.nn.embedding_lookup(self.w, self.y, name='pos_embed')

        self.mul = tf.matmul(self.x_embed, self.y_w, transpose_b=True)
        self.p = tf.nn.sigmoid(self.mul)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(self.p, self.labels)
        self.train = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)

        self.sess.run(tf.initialize_all_variables())

    def train_pair(self, word_idx, contexts):
        self.sess.run(self.train, feed_dict={self.x: [word_idx], self.y: contexts})

    def build_table(self):
        start_time = time.time()
        total_count_pow = 0
        for _, count in self.counter:
            total_count_pow += math.pow(count, self.alpha)
        word_idx = 1
        self.table = np.zeros([self.table_size], dtype=np.int32)
        word_prob = math.pow(self.counter[word_idx][1], self.alpha) / total_count_pow
        for idx in xrange(self.table_size):
            self.table[idx] = word_idx
            if idx / self.table_size > word_prob:
                word_idx += 1
                word_prob += math.pow(self.counter[word_idx][1], self.alpha) / total_count_pow
            if word_idx > self.vocab_size:
                word_idx = word_idx - 1
        print("Done in %.2f seconds.", time.time() - start_time)

    def sample_contexts(self, context):
        self.contexts[0] = context
        idx = 0
        while idx < self.neg_sample_size - 1:
            neg_context = self.table[random.randrange(self.table_size)]
            if context != neg_context:
                self.contexts[idx+2] = neg_context
                idx += 1

    def train_stream(self, filename):
        print("Training...")

        start_time = time.time()
        c = 0
        with open(filename) as f:
            words = [word for line in f.readlines() for word in line.split()]
            for idx, word in enumerate(words):
                try:
                    word_idx = self.word2idx[word]
                    reduced_window = random.randrange(self.window)
                    self.words[0] = word_idx
                    for jdx in xrange(idx - reduced_window, idx + reduced_window + 1):
                        context = words[jdx]
                        if jdx != idx:
                            try:
                                context_idx = self.word2idx[context]
                                self.sample_contexts(context_idx)
                                self.train_pair(word_idx, self.contexts)
                                self.lr = max(self.min_lr, self.lr + self.decay)
                                c += 1
                                if c % 100000 == 0:
                                    loss = self.sess.run(self.loss, feed_dict={self.x: [word_idx], self.y: self.contexts})
                                    print("%d words trained in %.2f seconds. Learning rate: %.4f, Loss : %.4f" % (c, time.time() - start_time, self.lr, loss))
                            except:
                                continue
                except:
                    continue

    def get_sim_Words(self, idxs, k):
        if type(idxs[0]) == str:
            idxs = np.array([self.word2idx[word] for word in idxs])
        else:
            idxs = np.array(idxs)

        vals, idxs = sess.run(
            [nearby_val, nearby_idx], {nearby_character: idxs})
        for i in xrange(len(idxs)):
            print(idx2word[idxs[i]])
            print()
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print("%-20s %6.4f" % (idx2word[neighbor], distance))

    def train_model(self, corpus):
        self.train_stream(corpus)

