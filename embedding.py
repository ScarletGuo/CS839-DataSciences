from gensim.models import FastText
from gensim.utils import simple_preprocess
from scipy.spatial.distance import cosine
from io import open
import os

def get_similarity(a, b):
    return 1 - cosine(a, b)


class Embedding(object):

    def __init__(self, path_to_files, **kwargs):
        self.config = {
            'size': 128,
            'window': 3,
            'batch_words': 100,
            'workers': 4,
            'epochs': 100,
            'tokenize': simple_preprocess,
            'sentencize': lambda x: x.split("."),

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
        model = FastText(data, min_count=1, size=self.config['size'], 
                         window=self.config['window'], workers=self.config['workers'], 
                         batch_words=self.config['batch_words'])
        model.train(data, total_examples=len(data), epochs=self.config['epochs'])
        return model

    def get_vectors(self, array):
        return self.model.wv[array]

