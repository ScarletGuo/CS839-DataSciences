#! /usr/bin/env python3

import os
import re

train_dir = './data/labeled/train/'
test_dir = './data/labeled/test/'

def extract_single_file(fname):
    result_list = []
    with open(fname, "r",encoding='utf-8', errors='replace') as fp:
        for line in fp:
            result =  re.findall('<>(.*?)<', line,  re.S)
            result_list = result_list + result 
    return result_list


def extract_gt(dir_name):
    dirs = os.listdir(dir_name)
    result_dict = {}
    for f in dirs:
        if 'txt' not in f:
            continue
        items = f.split('.')
        result_dict[int(items[0])] = extract_single_file(dir_name + '/' + f)
    return result_dict


def test():
    l = extract_gt(train_dir)
    l = extract_gt(test_dir)
    print(l)


if __name__ == '__main__':
    test()
    pass

