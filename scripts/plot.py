import sys
import os
from util import result_dir
import matplotlib.pyplot as plt
from stat_value_dis import FValueStat
import numpy as np

def calc_value_distance(fv):
    pos_values = fv.value_dis(1)
    neg_values = fv.value_dis(0)
    diff = pos_values - neg_values
    diff_s = (diff * diff).mean()
    return float(diff_s)

def get_bins(x, num_bins):
    x = sorted(x)
    splits = []
    for i in range(num_bins):
        r = (i + .0) / num_bins
        splits.append(x[int(r * len(x))])
    splits.append(x[-1])
    return splits

def get_bins_r(x, ratios):
    x = sorted(x)
    splits = []
    for r in ratios:
        p = min(len(x) -1 , int(r * len(x)))
        splits.append(x[p])
    splits.append(x[-1])
    return splits

def plot_value_dis(path):
    outf = file(path + ".value_dis.tsv", 'w')
    x = []
    for line in file(path):
        fv = FValueStat.load_from_line(line)
        eidx = fv.eidx
        fidx = fv.fidx
        key = (eidx, fidx)
        distance = calc_value_distance(fv)
        if eidx == 2527 and fidx == 478:
            print fv.value_dis(1)
            print fv.value_dis(0)
        x.append(distance)
        outf.write("%d,%d\t%s\n" %(eidx, fidx, distance))
    x = np.array(x)
    x[x == .0] = 1e-33
    x[x < 0.1] = 0.1
    x = np.log(x)
    
    # rs = [0, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    bins = get_bins(x, 30)
    bins = bins[0] + bins[8:]
    # bins = get_bins_r(x, rs)
    print bins
    plt.hist(x, bins = 30, normed=False)
    plt.show()

if __name__ == "__main__":
    plt.style.use('ggplot')
    death_stat_path = os.path.join(result_dir, 'death_value.stat.json')    

    plot_value_dis(death_stat_path)

