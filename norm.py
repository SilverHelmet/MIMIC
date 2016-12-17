from util import *
import h5py
import sys
import numpy as np
import math

class Stat:
    mode = "0-1"
    def __init__(self, ID):
        self.ID = ID
        self.min = None
        self.max = None
        self.added = False
        self.cnt = 0
        self.values = []

    def clear(self):
        self.added = False

    def add_value(self, value):
        self.add(value)
        self.values.append(value)

    def add(self, value):
        if not self.added:
            self.cnt += 1
        self.added = True
        if self.min == None:
            self.min = value
            self.max = value
        else:
            self.min = min(self.min, value)
            self.max = max(self.max, value)

    def norm(self):
        self.range = self.max - self.min
        x = np.array(self.values)
        self.mean = x.mean()
        self.std = x.std() + 0.0  
        self.min = x.min()
        self.max = x.max()  
        self.range = self.max - self.min + 0.0
        if self.std == 0.0:
            self.std = 1.0
        if self.range == 0.0:
            self.range = 1.0
        del self.values

    def norm_value(self, value):
        if Stat.mode == "0-1":
            return (value - self.min) / self.range
        elif Stat.mode == "mean=0":
            return (value - self.mean) / self.std
        else:
            print "error"

def add_norm_suffix(path):
    parts = path.split(".")
    ext = parts[-1]
    filename = ".".join(parts[:-1]) + ".normed" + "." + Stat.mode + "." + ext
    return filename

def stat(stat_filepath, feature_stat, event_stat):
    inf = h5py.File(stat_filepath, 'r')
    print "stat %s" %stat_filepath
    for idx, feature_matrix in enumerate(inf['feature'][:]):
        if idx % 1000 == 0:
            print "idx = %d" %idx
        for feature in feature_matrix:
            length = len(feature)
            i = 0
            while i < length:
                if feature[i + 1] != 0:
                    index = int(feature[i])
                    value = feature[i + 1]
                    if index not in feature_stat:
                        feature_stat[index] = Stat(index)
                    feature_stat[index].add_value(value)
                for stat in feature_stat.values():
                    stat.clear()
                i += 2

    for event_seq in inf['event'][:]:
        for event in event_seq:
            if not event in event_stat:
                event_stat[event] = Stat(event)
            event_stat[event].add(1)
        for stat in event_stat.values():
            stat.clear()


def norm_feature(feature, f_stat):
    i = 0
    while i < len(feature):
        index = int(feature[i])
        value = feature[i + 1] + 0.0
        if value != 0.0:
            stat = f_stat[index]
            value = stat.norm_value(value)
            feature[i+1] = value
        i += 2



def norm(outf, feature_stat, event_stat, feature_seqs, event_seqs, bad_events):
    for i in range(len(outf['label'][:])):
        if i % 1000 == 0:
            print 'i = %d' %i
        event_seq = event_seqs[i]
        feature_matrix = feature_seqs[i]
        for j in range(len(event_seq)):
            if event_seq[j] in bad_events:
                feature_matrix[j] = 0
                event_seq[j] = 0
            else:
                norm_feature(feature_matrix[j], feature_stat)

    outf['event'] = event_seqs
    outf['feature'] = feature_seqs








input_paths = sys.argv[1:]
output_paths = [add_norm_suffix(input_path) for input_path in input_paths]
print "load from %s\nwrite to %s" %(input_paths, output_paths)
feature_stat = {}
event_stat = {}

stat(input_paths[0], feature_stat, event_stat)
min_cnt = 5
bad_events = set()
for f_stat in event_stat.values():
    if f_stat.cnt <= min_cnt:
        bad_events.add(f_stat.ID)
print "norm mode =", Stat.mode
print "#bad =", len(bad_events)
print bad_events

for path in input_paths[1:]:
    stat(path, feature_stat, event_stat)

for f_stat in feature_stat.values():
    size = len(f_stat.values)
    # print f_stat.values
    f_stat.norm()  
    print f_stat.ID, 'size =', size, "min =", f_stat.min, "max =", f_stat.max
    # print f_stat.ID, "size =", size, "mean =",f_stat.mean, "std =",f_stat.std


for idx, path in enumerate(input_paths[:]):
    print "handling %s" %path
    inf = h5py.File(path, 'r')
    outf = h5py.File(output_paths[idx], 'w')
    outf['label'] = inf['label'][:]
    norm(outf, feature_stat, event_stat, inf['feature'][:], inf['event'][:], bad_events)
    inf.close()
    outf.close()

