import os
import tensorflow as tf
from model import Word2Vec

config = {}
config['corpus'] = "corpus.txt" # input data
config['window'] = 5 # (maximum) window size
config['embed_size'] = 100 # dimensionality of word embeddings
config['alpha'] = 0.75 # smooth out unigram frequencies
config['table_size'] = int(1E8) # table size from which to sample neg samples
config['neg_sample_size'] = 5 # number of negative samples for each positive sample
config['min_frequency'] = 10 #threshold for vocab frequency
config['lr'] = 0.025 # initial learning rate
config['min_lr'] = 0.001 # min learning rate
config['epochs'] = 3 # number of epochs to train
config['gpu'] = 0 # 1 = use gpu, 0 = use cpu
config['stream'] = 1 # 1 = stream from hard drive 0 = copy to memory first

with tf.Session() as sess:
    w2v = Word2Vec(config, sess)
    w2v.build_vocab(config['corpus'])
    w2v.build_table()

    for idx in xrange(config['epochs']):
        w2v.lr = config['lr']
        w2v.train_model(config['corpus'])

    w2v.print_sim_words(['the', 'he', 'can'], 5)
