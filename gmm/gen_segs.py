
from multiprocessing import Process, Queue
from numba import jit
import numpy as np
import h5py
import datetime


oo = 1e6
def dp_wrapper(cost, max_segs):
    global oo
    nb_events = len(cost)
    f = np.zeros((nb_events + 1, max_segs + 1))
    f[:] = -oo
    last_seg = np.zeros((nb_events + 1, max_segs+1), dtype = int)
    last_seg[:] = -1
    f[0][0] = 0
    dp_jit(cost, max_segs, f, last_seg)
    ed = nb_events
    L = np.argmax(f[-1,:])
    seg = []
    while L > 0:
        seg.append(ed)
        ed = last_seg[ed][L]
        L -= 1
    seg.reverse()
    seg.extend([0] * (max_segs - len(seg)))
    return seg

@jit(cache = True)
def dp_jit(cost, max_segs, f, last_seg):   
    nb_events = len(cost)  
    for i in range(1, nb_events + 1):
        for L in range(1, max_segs + 1):
            for k in range(0, i):
                tmp = f[k][L-1] + cost[k][i-1]
                if tmp > f[i][L]:
                    f[i][L] = tmp
                    last_seg[i][L] = k  

def dp(cost, max_segs):
    global oo

    nb_events = len(cost)
    f = [[-oo] * (max_segs + 1) for i in xrange(nb_events + 1)]
    last_seg = [[-1] * (max_segs+1) for i in xrange(nb_events + 1)]
    f[0][0] = 0

    for i in xrange(1, nb_events + 1):
        for L in range(1, max_segs + 1):
            for k in xrange(0, i):
                tmp = f[k][L-1] + cost[k][i-1]
                if tmp > f[i][L]:
                    f[i][L] = tmp
                    last_seg[i][L] = k       
    ed = nb_events
    f = np.array(f)
    L = np.argmax(f[-1,:])
    seg = []
    while L > 0:
        seg.append(ed)
        ed = last_seg[ed][L]
        L -= 1
    assert ed == 0
    seg.reverse()
    seg.extend([0] * (max_segs - len(seg)))
    return seg

def gen_seg_by_dp(event, gmm, emds, aggre_mode, max_segs):
    event = [i for i in event if i > 0]
    nb_events = len(event)
    # cost = np.zeros((nb_events, nb_events))
    cost = [[0] * nb_events for i in xrange(nb_events)]
    Xs = []
    for i in xrange(len(cost)):
        embedding = np.zeros(emds.shape[1])
        for j in xrange(i, len(cost)):
            embedding += emds[event[j]]
            if aggre_mode == "sum":
                Xs.append(embedding.copy())
            elif aggre_mode == "ave":
                Xs.append(embedding / (j-i+1))
    Xs = np.array(Xs)
    log_probs = gmm.score(Xs)
    idx = 0
    for i in xrange(len(cost)):
        for j in xrange(i, len(cost)):
            cost[i][j] = log_probs[idx]
            idx += 1
    seg = dp_wrapper(np.array(cost), max_segs)
    # seg = dp(cost, max_segs)
    return seg

def gen_segs_by_dp(events, gmm, emds, aggre_mode, max_segs):
    segs = []
    idx = 0
    for event in events:
        
        if idx % 100 == 0:
            now_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
            print "\t %s collect %d" %(now_time, idx)
        idx += 1
        seg = gen_seg_by_dp(event, gmm, emds, aggre_mode, max_segs)
        segs.append(seg)
    return seg
    
def work(name, base, result, events, gmm, emds, aggre_mode, max_segs):
    for event in events:
        seg = gen_seg_by_dp(event, gmm, emds, aggre_mode, max_segs)
        result.put((name, (base, seg)))
        base += 1

def write_segs(X, output_file):
    print "write segment to [%s]" %output_file
    X.sort(key = lambda x:x[0])
    X = [row[1] for row in X]
    f = h5py.File(output_file, 'w')
    f['segment'] = X
    f.close()


def collect(nb_items, result, output_file, write_func):
    X = []
    for i in xrange(nb_items):
        pid, item = result.get(block = True)
        idx = item[0]
        X.append(item)
        # print "\tget from [%d] item %d" %(pid, idx)
        if i % 100 == 0:
            now_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
            print "\t %s collect %d" %(now_time, i)
    write_func(X, output_file)
     

def gen_segs_by_dp_mp(events, gmm, emds, aggre_mode, max_segs, nb_processor, output_file):
    num = len(events) / nb_processor + 10
    st = 0
    result = Queue()
    procs = []
    for x in range(nb_processor):
        ed = min(len(events), st + num)
        proc = Process(target = work, args=(x, st, result, events[st:ed], gmm, emds, aggre_mode, max_segs))
        procs.append(proc)
        print "gen processor from %d to %d" %(st, ed)
        st = ed
    assert st == len(events)
    for proc in procs:
        proc.start()
    collector = Process(target = collect, args = (len(events), result, output_file, write_segs))
    collector.start()
    collector.join()
