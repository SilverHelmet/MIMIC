import sys
import os
sys.path.append("..")
from util import *
from build_sample import Sample


outf = file(os.path.join(exper_dir, "sample.word2vec"), 'w')
cnt = 0
for line in file(sys.argv[1]):
    cnt += 1
    if cnt % 100 == 0:
        print "cnt =", cnt
    sample = Sample.load_from_line(line)
    for event in sample.events:
        if event.index != 1:
            outf.write(str(event.index) + " ")
    outf.write("\n")
outf.close()
