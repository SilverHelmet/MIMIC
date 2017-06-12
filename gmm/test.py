from multiprocessing import Process, Queue
import h5py
import numpy as np
import time

def write_segs(X, output_file):
    X.sort(key = lambda x: x[0])
    X = [row[1] for row in X]
    X = np.array(X)
    f = h5py.File(output_file, 'w')
    f['segment'] = X
    f.close()

def collect(name, max_item, output_file, write_func, q):
    X = []
    for i in xrange(max_item):
        value = q.get(block = True)
        print "collect item", value
        X.append(value)
    write_func(X, output_file)

def work(st, ed, q):
    time.sleep(3)
    for x in xrange(st, ed):
        q.put((x, [x] * 10))





q = Queue()
prs = []
for x in range(5):
    st = x * 10
    ed = (x+1) * 10
    pw = Process(target = work, args=(st, ed, q))
    pw.start()
pc = Process(target = collect, args=("collect", 50, "out_test.h5", write_segs, q))
pc.start()
pc.join()
    
