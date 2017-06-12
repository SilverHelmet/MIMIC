import scripts_util
from util import *
import os
import sys
import h5py

class FeatureStat:
    def __init__(self, idx):
        self.idx = idx
        self.values = []

    def add_value(self, value):
        self.values.append(value)

    def finish_norm(self):
        if len(self.values) == 0:
            self.values = [0]
        x = np.array(self.values)
        self.min = x.min()
        self.max = x.max()
        self.std = x.std() + 0.0
        if self.std == 0.0:
            self.std = 1.0
        self.mean = x.mean()
        del self.values

    def norm_value(self, value):
        return (value - self.mean) / self.std
    
def infer_outpath(inf):
    dirpath = os.path.dirname(inf)
    filename = os.path.basename(inf)
    outf = os.path.join(dirpath, "normed_" + filename)
    return outf

def stat_feature_of_one(path, features_stats):
    f = h5py.File(path, 'r')
    for feature_mat in f['feature'][:]:
        for feature_pair in feature_mat:
            idx = 0
            while idx < 6:
                if feature_pair[idx+1] != 0:
                    value = float(feature_pair[idx + 1])
                    index = int(feature_pair[idx])
                    features_stats[index].add_value(value)
                idx += 2
    f.close()

def stat_feature(paths):
    feature_stats = [FeatureStat(i) for i in range(649)]

    for path in paths:
        stat_feature_of_one(path, feature_stats)
    return feature_stats

def norm_feature_of_one(path, feature_stats):
    print path
    inf = h5py.File(path, 'r')
    outf = h5py.File(infer_outpath(path), 'w')
    outf['event'] = inf['event'][:]
    outf['label'] = inf['label'][:]
    outf['sample_id'] = inf['sample_id'][:]
    features = inf['feature'][:]
    inf.close()
    cnt = 0
    for feature_mat in features:
        cnt += 1
        if cnt % 1000 == 0:
            print "\tcnt = %d" %cnt
        for feature_pair in feature_mat:
            idx = 0
            while idx < 6:
                if feature_pair[idx + 1] != 0.0:
                    value = float(feature_pair[idx + 1])
                    index = int(feature_pair[idx])
                    feature_pair[idx + 1] = feature_stats[index].norm_value(value)
                idx += 2
    outf['feature'] = features
    outf.close()

def norm_feature(paths, feature_stats):
    for feature_stat in feature_stats:
        feature_stat.finish_norm()
    for path in paths:
        norm_feature_of_one(path, feature_stats)

    

if __name__ == "__main__":
    infs = sys.argv[1:]
    outf = [infer_outpath(inf) for inf in infs]
    print "outpath = ", outf

    feature_stats = stat_feature(infs)
    norm_feature(infs, feature_stats)

