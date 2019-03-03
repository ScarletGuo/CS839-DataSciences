from os import listdir
from os.path import isfile, join
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import ipdb

def get_instances():
    path = "./data/original"
    txt_names = [f for f in listdir(path) if isfile(join(path, f))]
    count = 0
    table = []  # all word combinations of all files
    combination = [1, 2, 3]
    for txt_name in txt_names:
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
    get_instances()




