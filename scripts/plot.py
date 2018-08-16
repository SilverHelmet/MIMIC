import sys
import os
from util import result_dir
import matplotlib.pyplot as plt
from stat_value_dis import FValueStat
import numpy as np
import math

def calc_value_distance(fv):
    pos_values = fv.value_dis(1)
    neg_values = fv.value_dis(0)
    diff = pos_values - neg_values
    diff_s = (diff * diff).mean()
    return float(diff_s)

def KLD(p,q):
    p,q=zip(*filter(lambda (x,y): x!=0 or y!=0, zip(p,q)))
    p=p+np.spacing(1)
    q=q+np.spacing(1)
    return sum([_p * math.log(_p/_q,2) for (_p,_q) in zip(p,q)])

def calc_event_distance(fv):
    pos_values = fv.event_dis(1)
    neg_values = fv.event_dis(0)
    pos_dis = pos_values / (pos_values.sum() + .0)
    neg_dis = neg_values / (neg_values.sum() + .0)
    return KLD(pos_dis, neg_dis)

def plot_one_value_dis(eidx, fidx, path):
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        _eidx = fv.eidx
        _fidx = fv.fidx
        if eidx != _eidx or fidx != _fidx:
            continue
        pos_v = fv.value_dis(1)
        neg_v = fv.value_dis(0)
        x = range(24)
        plt.plot(x, pos_v, 'r')
        plt.plot(x, neg_v, 'b')
        outpath = os.path.join(result_dir, 'graph/' + "%d_%d_value.png" %(eidx, fidx))
        plt.savefig(outpath)
        plt.close('all')

def plot_value_dis(path):
    x = []
    key2dis = {}
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        if not fv.check_size(100):
            continue
        eidx = fv.eidx
        fidx = fv.fidx
        key = (eidx, fidx)
        distance = calc_value_distance(fv)
        key2dis[key] = distance
        x.append(distance)
    outf = file(path + ".value_dis.tsv", 'w')
    for fe in sorted(key2dis.keys(), key = lambda x:key2dis[x], reverse = True):
        outf.write("%d\t%d\t%.4f\n" %(fe[0], fe[1], key2dis[fe]))
    outf.close()
    x = np.array(x)
    print 'size = %d' %len(x)
    x = sorted(x)
    r = x[-1] - x[0]
    step = r / 10
    bins = np.arange(x[0], x[-1], step)
    # bins = bins[[0] + range(3, len(bins))]
    print bins
    plt.hist(x, bins = bins, normed=False)
    # plt.show()
    graph_outpath = os.path.join(result_dir, 'graph/' + os.path.basename(path) + '.value_dis.png')
    if not os.path.exists(result_dir + '/graph'):
        os.mkdir(result_dir + '/graph')
    plt.savefig(graph_outpath)

def plot_one_event_dis(eidx, path):
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        _eidx = fv.eidx
        if _eidx != eidx:
            continue
        pos_cnts = fv.event_dis(1)
        pos_cnts /= pos_cnts.sum()
        neg_cnts = fv.event_dis(0)
        neg_cnts /= neg_cnts.sum()
        x = range(24)
        plt.plot(x, pos_cnts, 'r')
        plt.plot(x, neg_cnts, 'b')
        outpath = os.path.join(result_dir, 'graph/' + "%d_event.png" %(eidx,))
        plt.savefig(outpath)
        plt.close('all')

def plot_event_dis(path):
    x = []
    key2dis = {}
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        if not fv.check_size(100):
            continue
        eidx = fv.eidx
        fidx = fv.fidx
        key = eidx
        if key in key2dis:
            continue
        distance = calc_event_distance(fv)
        key2dis[key] = distance
        x.append(distance)
    outf = file(path + ".event_dis.tsv", 'w')
    for eidx in sorted(key2dis.keys(), key = lambda x:key2dis[x], reverse = True):
        outf.write("%d\t%.4f\n" %(eidx, key2dis[eidx]))
    outf.close()

    x = np.array(x)
    print 'size = %d' %len(x)
    x = sorted(x)[:-3]
    r = x[-1] - x[0]
    bins = [x[0], 0.01, 0.02, 0.03, 0.04, 0.06, .1, .2, .3, .4, .5, .6, .7]
    print bins
    plt.hist(x, bins = bins, normed=False)
    graph_outpath = os.path.join(result_dir, 'graph/' + os.path.basename(path) + '.event_dis.png')
    if not os.path.exists(result_dir + '/graph'):
        os.mkdir(result_dir + '/graph')
    plt.savefig(graph_outpath)


if __name__ == "__main__":
    plt.style.use('ggplot')
    death_stat_path = os.path.join(result_dir, 'death_value.stat.json')    

    # plot_value_dis(death_stat_path)
    # value_list = [(2996, 569),
    #     (2844, 535)]
    # for eidx, fidx in value_list:
    #     plot_one_value_dis(eidx, fidx, death_stat_path)

    plot_event_dis(death_stat_path)
    # event_list = [2731, 19, 1936, 1514, 2208, 1047, 1748, 1224, 875, 2521, 1092, 3362]
    # for eidx in event_list:
    #     plot_one_event_dis(eidx, death_stat_path)

