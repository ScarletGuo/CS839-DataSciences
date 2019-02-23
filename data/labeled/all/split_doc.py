# -*- coding: UTF-8 -*-

import os
from io import open


def validate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def write_docs(lines, ID, folder="labeled"):
    for doc in lines.split("\n"):
        if len(doc) < 5:
            continue
        validate_path(folder)
        out = open("{folder}/{ID}.txt".format(folder=folder, ID=ID), 'w+')
        try:
            out.write(doc)
        except:
            print(doc)
            raise
        out.close()
        ID += 3
    return ID


path = "labeled/all/"
ID = 0
for f in os.listdir(path):
    lines = " ".join([line for line in open(os.path.join(path, f), encoding="utf-8")])
    original = lines.replace("<>", "").replace("</>", "")
    write_docs(lines, ID)
    ID = write_docs(original, ID, folder="original")


