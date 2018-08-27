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

def plot_one_value_dis(eidx, fidx, path, ax):
    filename = os.path.basename(path)
    if 'death' in filename:
        dataset = 'death'
    else:
        dataset = 'labtest'
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        dis = calc_event_distance(fv)
        _eidx = fv.eidx
        _fidx = fv.fidx
        if eidx != _eidx or fidx != _fidx:
            continue
        pos_v = fv.value_dis(1)
        neg_v = fv.value_dis(0)
        x = range(24)
        ax.plot(x, pos_v, 'r', label = 'postive samples')
        ax.plot(x, neg_v, 'b', label = 'negtive samples')
        ax.legend()
        ax.set_xlabel('hour')
        ax.set_ylabel('mean value')
        ax.set_title('distribution of attributes')
        # outpath = os.path.join(result_dir, 'graph/' + "valueDist_data=%s_eid=%d_fid=%d_dis=%.3f.png" %(dataset, eidx, fidx, dis))
        # plt.savefig(outpath)
        # plt.close('all')

def plot_value_dis(path, ax):
    x = []
    key2dis = {}
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        if not fv.check_size(100):
            continue
        eidx = fv.eidx
        fidx = fv.fidx
        if fidx == 0:
            continue
        key = (eidx, fidx)
        distance = calc_value_distance(fv)
        key2dis[key] = distance
        x.append(distance)
    outf = file(path + ".value_dis.tsv", 'w')
    for fe in sorted(key2dis.keys(), key = lambda x:key2dis[x], reverse = True):
        outf.write("%d\t%d\t%.4f\n" %(fe[0], fe[1], key2dis[fe]))
    outf.close()
    x = np.array(x)
    x[x > 1.0] = 1
    print 'value_dis size = %d' %len(x)
    x = sorted(x)
    # x = sorted(x)[:int(0.99 * len(x))]
    r = x[-1] - x[0]
    step = r / 10
    bins = np.arange(x[0], x[-1] + step, step)
    # bins = bins[[0] + range(3, len(bins))]
    print bins
    if ax is None:
        ax = plt.subplot(1, 1, 1)
    ax.hist(x, bins = bins, normed=True)
    ax.set_xlabel('MSE')
    ax.set_title('distribution of MSE of attributes')
    # plt.show()
    # graph_outpath = os.path.join(result_dir, 'graph/' + os.path.basename(path) + '.value_dis.png')
    # if not os.path.exists(result_dir + '/graph'):
    #     os.mkdir(result_dir + '/graph')
    # plt.savefig(graph_outpath)
    # plt.close('all')


def plot_one_event_dis(eidx, path, ax):
    filename = os.path.basename(path)
    if 'death' in filename:
        dataset = 'death'
    else:
        dataset = 'labtest'
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        dis = calc_event_distance(fv)
        _eidx = fv.eidx
        if _eidx != eidx or fv.fidx != 0:
            continue
        pos_cnts = fv.event_dis(1)
        pos_cnts /= pos_cnts.sum()
        neg_cnts = fv.event_dis(0)
        neg_cnts /= neg_cnts.sum()
        x = range(24)
        ax.plot(x, pos_cnts, 'r', label = 'postive samples')
        ax.plot(x, neg_cnts, 'b', label = 'negtive samples')
        ax.legend()
        ax.set_xlabel('hour')
        ax.set_ylabel('occur prob')
        ax.set_title('distrbution of event')
        # outpath = os.path.join(result_dir, 'graph/' + "eventDist_data=%s_eid=%d_dis=%.3f.png" %(dataset, eidx, dis))
        # plt.savefig(outpath)
        # plt.close('all')

def plot_event_dis(path, ax = None):
    x = []
    key2dis = {}
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        if not fv.check_size(100):
            continue
        eidx = fv.eidx
        fidx = fv.fidx
        if fidx != 0:
            continue
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
    x[x > 1.0] = 1.0
    print 'event_dis size = %d' %len(x)
    x = sorted(x)
    # x = sorted(x)[:int(0.99 * len(x))]
    r = x[-1] - x[0]
    # bins = [x[0], 0.01, 0.02, 0.03, 0.04, 0.06, .1, .2, .3, .4, .5, .6, .7]
    step = r / 10
    bins = np.arange(x[0], x[-1] + step, step)
    print bins
    if ax is None:
        ax = plt.subplot(1, 1, 1)
    ax.hist(x, bins = bins, normed=True)
    ax.set_xlabel('KL-distance')
    ax.set_title('distribution of KL-distance of events')
    # graph_outpath = os.path.join(result_dir, 'graph/' + os.path.basename(path) + '.event_dis.png')
    # if not os.path.exists(result_dir + '/graph'):
    #     os.mkdir(result_dir + '/graph')
    # plt.savefig(graph_outpath)
    # plt.close('all')


def plot_label_event():
    death_stat_path = os.path.join(result_dir, 'death_value.stat.json')    
    labtest_stat_path = os.path.join(result_dir, 'labtest_value.stat.json')

    stat_path = death_stat_path
    # stat_path = labtest_stat_path

    plot_value_dis(stat_path)
    value_list = [(2996, 569),
        (2054, 375),
        (2251, 420),
        (1129, 143),
        (2041, 366),
        (3234, 606),
        (1755, 247),
        (3313, 625),
        (3270, 614),
        (2728, 521)]
    for eidx, fidx in value_list:
        plot_one_value_dis(eidx, fidx, stat_path)

    plot_event_dis(stat_path)
    event_list = [1936, 2699, 2730, 2726, 2729, 2633, 1889, 1883, 1827]
    for eidx in event_list:
        plot_one_event_dis(eidx, death_stat_path)

def plot_time_event_effect_dis(fv, diff):
    pos_value = fv.value_dis(1)
    neg_value = fv.value_dis(0)
    x = range(24)
    plt.plot(x, pos_value, 'r', label ='+time')
    plt.plot(x, neg_value, 'b', label ='-time')
    plt.legend()
    plt.savefig('result/graph/effectDist_data=death_eid={}_dis={}.png'.format(fv.eidx, diff))
    plt.close('all')   

def plot_one_event_effect_dis(eidx, path):
    for line in file(path):
        fv = FValueStat.load_from_line_onlye(line)
        if fv.eidx != eidx:
            continue
        pos_value = fv.value_dis(1)
        neg_value = fv.value_dis(0)
        x = range(24)
        ax = plt.subplot(1, 1, 1)
        ax.plot(x, pos_value, 'r', label ='with time feature')
        ax.plot(x, neg_value, 'b', label ='without time feature')
        ax.legend()
        ax.set_xlabel('hour')
        ax.set_ylabel('effect')
        ax.set_title('mean effect of event {}'.format(eidx))
        plt.savefig('result/graph/effect_eid={}'.format(eidx))
        plt.close('all')

def plot_event_effect_dis(stat_path, ax):
    outpath = stat_path + ".effect_dis.tsv"
    outf = file(outpath, 'w')
    outs = []
    x = []
    error_cnt = 0
    for line in file(stat_path):
        fv = FValueStat.load_from_line_onlye(line)
        eidx = fv.eidx
        if fv.get_size(1) < 100:
            error_cnt += 1
        time_mean = fv.mean(1)
        no_time_mean = fv.mean(0)
        if time_mean is None or no_time_mean is None:
            error_cnt += 1
            continue
        diff = time_mean - no_time_mean
        x.append(diff)
        out = [eidx, time_mean, no_time_mean, diff, fv]
        outs.append(out)
    print 'error size = %d' %error_cnt
    print 'event effect size = %d' %len(x)
    x = np.array(x)
    x[x < -0.05] = -0.05
    x[x > 0.05] = 0.05
    x.sort()

    r = x[-1] - x[0]
    step = r / 10
    bins = np.arange(x[0] - 0.0001, x[-1] + step, step)
    print bins
    ax.hist(x, bins = bins, normed=True)
    ax.set_xlabel('mean effect')
    ax.set_title('distribution of difference between events with/without time_feature')
    outs.sort(key = lambda x: x[3], reverse = True)
    step = 50
    idx = 0
    for eidx, time_mean, no_time_mean, diff, fv in outs:
        # if idx % step == 0 or idx == len(outs) - 1:
        #     plot_time_event_effect_dis(fv, diff)
        idx += 1
        
        out = [eidx, round(time_mean, 4), round(no_time_mean, 4), round(diff, 4)]
        outf.write('\t'.join(map(str, out)) + '\n')
    outf.close()

def plot_time_event():
    death_stat_path = 'result/death_event_time_effect.json'
    plot_event_effect_dis(death_stat_path)

def plot_one_event(eidx, fidx, death_stat_path):
    plot_one_event_dis(eidx, death_stat_path, plt.subplot(1 ,2, 1))
    plot_one_value_dis(eidx, fidx, death_stat_path, plt.subplot(1, 2, 2))
    plt.savefig('result/graph/event_dis_eid=%d,fidx=%d.png' %(eidx, fidx))
    plt.close('all')


def plot_total_effect():
    death_stat_path = os.path.join(result_dir, 'death_value.stat.json')
    death_effect_path = os.path.join(result_dir,'')
    plt.rcParams['figure.figsize'] = (12.0, 4.0)
    plot_event_dis(death_stat_path, plt.subplot(1,2,1))
    plot_value_dis(death_stat_path, plt.subplot(1,2,2))
    plt.savefig('result/graph/event&attr_dis.png')
    plt.close('all')

    ef_pairs = [(71, 28), (1514, 199), (43, 15), (1500, 189), (1756, 247)]
    for eidx, fdix in ef_pairs:
        plot_one_event(eidx, fdix,death_stat_path)

    effect_path = 'result/death_event_time_effect.json'
    plot_event_effect_dis(effect_path, plt.subplot(1,1,1))
    plt.savefig('result/graph/event_effect_dis.png')
    plt.close('all')

    eidxs = [253, 2018, 2842, 2527]
    for eidx in eidxs:
        plot_one_event_effect_dis(eidx, effect_path)

    
    
    # plt.show() 


if __name__ == "__main__":
    plt.style.use('ggplot')
    plot_total_effect()
    # plot_label_event()
    # plot_time_event()



