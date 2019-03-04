from os import listdir
from os.path import isfile, join
import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from io import open
from copy import deepcopy
import pandas as pd
#import ipdb
import string
import spacy
import re


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class NgramFeature(object):
    def __init__(self, doc_id, ngram, sts_id, upcase_ratio):
        self.features = {
            'doc_id': doc_id,
            'ngram': ngram,
            'sentence_id': sts_id,
            'upcase_ratio': upcase_ratio,
        }

    def get_table_row(self):
        return self.features

    def add_feature(self, name, value):
        self.features[name] = value


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


def find_word_index(sts, word):
    return re.split('\s+', sts).index(word)


def find_word_pos(sts, word):
    a = re.search(re.compile(r'\b%s\b' % word), sts)
    if(a is None):
        print(sts)
        print(word)
        return None
    return a.start()

def find_ngram_upcase(items):
    assert(len(items) > 0)
    total = 0
    for item in items:
        if 'A' <= item[0] <= 'Z':
            total += 1
    return float(total) / len(items)


def get_linguistic_features(sts):
    nlp = spacy.load('en_core_web_sm')
    return nlp(sts)


def find_ngram_index(doc_id, sts_list, comb_len):
    fobj_list = []
    exclude = set(string.punctuation)
    for sts_id, sts in enumerate(sts_list):
        # sentence level
        cur_sts = re.sub('\s+', ' ', sts).strip()
        ling_feat = get_linguistic_features(cur_sts)
        # merge spans
        for span in ling_feat.ents:
            #copy_feat = deepcopy(ling_feat)
            token = span.merge()
            upcase_ratio = find_ngram_upcase(token.text)
            ngram_fobj = NgramFeature(doc_id, token.text, sts_id, upcase_ratio)
            for attr in dir(token):
                if attr.startswith("__") or attr.endswith('_') or len(attr)<=1:
                    continue
                val = getattr(token, attr)
                if isinstance(val, float) or isinstance(val, int) or isinstance(val, long):
                    ngram_fobj.add_feature(attr, val)
            fobj_list.append(ngram_fobj)
#         cur_sts = ''.join(ch for ch in sts if ch not in exclude)
#         for i in range(1, comb_len + 1):
#             tokens = word_tokenize(cur_sts)
#             this_grams = ngrams(tokens, i)
#             for item in this_grams:
#                 idx = find_word_index(cur_sts, item[0])
#                 if(idx is None):
#                     continue
#                 upcase_ratio = find_ngram_upcase(item)
#                 ngram_fobj = NgramFeature(doc_id, item, cur_sts, idx, upcase_ratio)
#                 fobj_list.append(ngram_fobj)
    return fobj_list


def find_ngram_features(dir_name, comb_len=3):
    dir_name += '/'
    txt_names = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    fobj_list = []
    sts_list = []
    for txt_name in txt_names:
        cur_sts_list = get_sentances(join(dir_name, txt_name))
        sts_list += cur_sts_list
        fobj_list += find_ngram_index(get_doc_id(txt_name), sts_list, comb_len)
    return pd.DataFrame(data=[fobj.get_table_row() for fobj in fobj_list]).set_index(['ngram','sentence_id','doc_id'])
        
        
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

    
if __name__ == "__main__":
    #get_instances()
    X = find_ngram_features('./data/original/t/')




