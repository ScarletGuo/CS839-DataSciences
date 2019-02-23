import sys
import os
import numpy as np

txtlist = list(open("79.txt", "r", encoding='utf-8').readlines())
i = 2

for line in txtlist:
    name = str(i) + ".txt"
    with open(os.path.join(name), 'w') as t:
        t.write(line) 
    i += 3


