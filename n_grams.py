from os import listdir
from os.path import isfile, join
import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import ipdb
import string
import re


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class NgramFeature(object):
    def __init__(self, ngram, sts, idx, upcase_ratio):
        self.ngram = ngram
        self.sentence = sts
        self.idx = idx
        self.upcase_ratio = upcase_ratio

    def get_table_row(self):
        return [self.ngram, self.sentence, self.idx, self.upcase_ratio]


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
        if item[0] > 'A':
            total += 1
    return float(total) / len(items)


def find_ngram_index(sts_list, comb_len):
    fobj_list = []
    exclude = set(string.punctuation)
    for sts in sts_list:
        cur_sts = ''.join(ch for ch in sts if ch not in exclude)
        for i in range(1, comb_len + 1):
            tokens = word_tokenize(cur_sts)
            this_grams = ngrams(tokens, i)
            for item in this_grams:
                idx = find_word_index(cur_sts, item[0])
                if(idx is None):
                    continue
                upcase_ratio = find_ngram_upcase(item)
                ngram_fobj = NgramFeature(item, cur_sts, idx, upcase_ratio)
                fobj_list.append(ngram_fobj)
    return fobj_list


def find_ngram_feathures(dir_name, comb_len=3):
    dir_name += '/'
    txt_names = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    fobj_list = []
    sts_list = []
    for txt_name in txt_names:
        cur_sts_list = get_sentances(join(dir_name, txt_name))
        sts_list += cur_sts_list
        fobj_list += find_ngram_index(sts_list, comb_len)
    for fobj in fobj_list:
        print(fobj.get_table_row())


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
    find_ngram_feathures('./data/original/t/')




