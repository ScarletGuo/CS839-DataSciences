from os import listdir
from os.path import isfile, join
import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from io import open
from features import *
import pandas as pd
import itertools
from tqdm import tqdm
#import ipdb
import string
import spacy
import re


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class NgramFeature(object):
    def __init__(self, doc_id, ngram, sts_id, span_id, n):
        self.features = {
            'doc_id': doc_id,
            'ngram': ngram,
            'sentence_id': sts_id,
            'span_id': span_id,
            'length': n,
        }

    def get_table_row(self):
        return self.features

    def add_feature(self, name, value):
        self.features[name] = value
        
    def add_features(self, feat):
        self.features.update(feat)


"""
@return list of sentences in a file
"""
def get_sentances(fname):
    ## Note: if need download punkt model use: nltk.download('punkt')
    with open(fname, "r", encoding='utf-8', errors='replace') as fp:
        data = fp.read()
        sts_list = tokenizer.tokenize(data)
        # print(sts)
    return sts_list


def get_linguistic_features(sts):
    nlp = spacy.load('en_core_web_sm')
    return nlp(sts)


def find_ngram(L, comb_len=3):
    return [L[i:i+j] for i in range(0,comb_len) for j in range(1,len(L)-i+1)]


def find_ngram_index(doc_id, sts_list, comb_len):
    fobj_list = []
    exclude = set(string.punctuation)
    for sts_id, sts in enumerate(sts_list):
        sts_fobj_list = []
        # sentence level
        cur_sts = re.sub('\s+', ' ', sts).strip()
        ling_feat = get_linguistic_features(cur_sts)
        tokens_in_sts = [t for t in ling_feat]
        # get candidates
        for span_id, span in enumerate(ling_feat.noun_chunks):
            span_fobj_list = []
            tokens_in_span = [t for t in span]
            span_txt = " ".join([t.text for t in span])
            for tokens_in_gram in find_ngram(tokens_in_span, comb_len):
                gram_idx = map(lambda x: x.i, tokens_in_gram)
                gram = map(lambda x: x.text, tokens_in_gram)
                ngram_fobj = NgramFeature(doc_id, " ".join(gram), sts_id, span_id, len(gram))
                # add features of the gram
                ngram_fobj.add_features(get_features(tokens_in_sts, tokens_in_gram, gram_idx, 
                                                     gram, cur_sts, span_txt))
                span_fobj_list.append(ngram_fobj)
            sts_fobj_list.append(span_fobj_list)
        # merge each nonchunck span and get features of span
        for span_id, span in enumerate(ling_feat.noun_chunks):
            span_token = span.merge()
            span_feature = get_span_features(span_token)
            for fobj in sts_fobj_list[span_id]:
                fobj.add_features(span_feature)
                fobj_list.append(fobj)
    return fobj_list


def find_ngram_features(dir_name, comb_len=3):
    dir_name += '/'
    txt_names = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    fobj_list = []
    sts_list = []
    for txt_name in tqdm(txt_names):
        cur_sts_list = get_sentances(join(dir_name, txt_name))
        sts_list += cur_sts_list
        fobj_list += find_ngram_index(get_doc_id(txt_name), sts_list, comb_len)
        #return find_ngram_index(get_doc_id(txt_name), sts_list, comb_len)
    return pd.DataFrame(data=[fobj.get_table_row() for fobj in fobj_list])
        
        
def get_doc_id(txt_name):
    return int(txt_name.replace('.txt',''))


def get_instances():
    path = "./data/original/train/"
    txt_names = [f for f in listdir(path) if isfile(join(path, f))]
    count = 0
    table = []  # all word combinations of all files
    combination = [1, 2, 3]
    sts_list = []
    for txt_name in txt_names:
        sts_list = sts_list + get_sentances(join(path, txt_name))
        for i in combination:  # length of combination
            count += 1
            text = list(open(join(path, txt_name), "r", encoding='utf-8', errors='replace').readlines())
            assert len(text) == 1  # each file should have only one line
            text = text[0].replace(",", "")
            text = text.replace(";", "")
            text = text.replace(".", "")
            text = text.replace("(", "")
            text = text.replace(")", "")
            text = text.replace("!", "")
            text = text.replace("?", "")
            text = text.replace("<", "")
            text = text.replace(">", "")
            text = text.replace("/", "")
            text = text.replace("\'", "")
            text = text.replace("\"", "")
            text = text.replace("\`", "")
            tokens = word_tokenize(text)  # single token
            this_grams = ngrams(tokens, i)  # token combinations
            for item in this_grams:  # item is a tuple
                table.append([txt_name.split(".")[0], item, 0])
    
    # table: list, containing (file_id, tuple of words combination, label)
    return table
    """
    with open("tabel.txt", "w") as f:
        for instance in table:
            temp = "".join(str(s.strip()) + " " for s in instance[1])
            temp = temp.rstrip()
            f.write(instance[0] + ":" + temp + ";" + str(instance[2]) + "\n")
    """

'''
already implemented by SpaCy
'''


def find_word_pos(sts, word):
    a = re.search(re.compile(r'\b%s\b' % word), sts)
    if(a is None):
        print(sts)
        print(word)
        return None
    return a.start()

    
if __name__ == "__main__":
    #get_instances()
    X = find_ngram_features('./data/original/t/')




