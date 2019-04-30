from gensim.models import FastText, Word2Vec
from gensim.utils import simple_preprocess
from scipy.spatial.distance import cosine
from io import open
import numpy as np
import os
import re


class Embedding(object):

    def __init__(self, path_to_files, **kwargs):
        self.config = {
            'size': 128,
            'window': 3,
            'batch_words': 100,
            'workers': 4,
            'epochs': 100,
            'tokenize': lambda x: re.split('\s+', x),
            'sentencize': lambda x: x.split("."),
            'model': 'fasttext',
        }
        self.config.update(kwargs)
        self.n_samples = 0
        self.path_to_files = path_to_files
        self.model = self.train_embedding()

    def process_files(self):
        corpus = []
        for f in os.listdir(self.path_to_files):
            lines = ""
            for line in open(os.path.join(self.path_to_files, f), encoding="utf-8", errors='replace'):
                lines += line
            for s in self.config['sentencize'](lines):
                self.n_samples += 1
                corpus.append(self.config['tokenize'](s))
        return corpus

    def train_embedding(self):
        data = self.process_files()
        if self.config['model'] == "word2vec":
            model = Word2Vec(data, min_count=1, size=self.config['size'], 
                         window=self.config['window'], workers=self.config['workers'], 
                         batch_words=self.config['batch_words'])
        else:
            model = FastText(data, min_count=1, size=self.config['size'], 
                             window=self.config['window'], workers=self.config['workers'], 
                             batch_words=self.config['batch_words'])
        model.train(data, total_examples=len(data), epochs=self.config['epochs'])
        return model

    def get_vectors(self, array):
        if self.config['model'] == "word2vec":
            return np.mean([self.model.wv[w] for w in array.split(' ')], axis=0)
        else:
            return self.model.wv[array]
        
    def get_similarity(self, a, b):
        return 1 - cosine(self.get_vectors(a), self.get_vectors(b))

