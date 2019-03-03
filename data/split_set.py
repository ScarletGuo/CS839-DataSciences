import numpy as np
import shutil
import os

def move_to(ID, folder):
	try:
		os.mkdir('labeled/%s/'%folder)
		os.mkdir('original/%s/'%folder)
	except:
		pass
	for i in ID:
		shutil.move('labeled/%d.txt'%i, 'labeled/%s/'%(folder))
		shutil.move('original/%d.txt'%i, 'original/%s'%(folder))

f = open("splits.txt", 'w+')
x = [int(i.replace('.txt','')) for i in os.listdir('labeled') if ".txt" in i]
f.write("total number of files: %d\n"%len(x))
np.random.shuffle(x)
training, test = x[0:200], x[200:300]
f.write("training: {}\n".format(training))
f.write("test: {}\n".format(test))
move_to(training, 'train')
move_to(test, 'test')
f.close()
