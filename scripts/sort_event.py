import sys
import os
sys.path.append("..")
from util import *

if __name__ == '__main__':
	id2event_value = load_id2event_value()
	l = []
	for idx, line in enumerate(file("../cof.txt")):
		l.append((idx, float(line)))
	l.sort(key = lambda x: x[1])
	for idx, value in l:
		if idx < 2:
			continue
		print "%s\t%5f" %(id2event_value[idx], value)


