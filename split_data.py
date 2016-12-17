import sys
import h5py
from build_sample import Sample
from util import *
import os
import numpy as np


def add_padding(l, max_len, padding_value = 0):
    assert max_len >= len(l)
    for i in range(max_len - len(l)):
        l.append(padding_value)

def print_to_local(samples, filepath, max_len):
    global ignore_1
    f = h5py.File(filepath, 'w')
    labels = []
    events = []
    event_times = []
    features = []
    max_feature_len = 6
    feature_padding = [0] * max_feature_len
    for sample in samples:
        label = sample.label
        event_seq = []
        
        feature_seqs = []
        for event in sample.events:
            if event.index == 1 and ignore_1:
                continue
            event_seq.append(event.index)
            feature_seq = []
            for feature in event.features:
                feature_seq.append(feature.index)
                feature_seq.append(feature.value)
            add_padding(feature_seq, max_feature_len)
            feature_seqs.append(feature_seq)

        add_padding(event_seq, max_len)
        add_padding(feature_seqs, max_len, feature_padding)
        events.append(event_seq)
        event_times.append(event_time)
        features.append(feature_seqs)

        labels.append(label)

    f['label'] = np.array(labels)
    f['feature'] = np.array(features)
    f['event'] = np.array(events)
    f.close()

def adjust(limits, ratio):
    min_limit = reduce(min, limits)
    up_cell = ratio * min_limit
    for i in range(len(limits)):
        limits[i] = min(up_cell, limits[i])


if __name__ == '__main__':
    max_len = 1000
    ignore_1 = True
    sample_file = sys.argv[1]
    normalize = sys.argv[2] == 'norm'
    print "load from [%s], normalization Flag = [%s]" %(sample_file, normalize)
    samples = []
    tot_cnt = [0] * 2
    ratio = 0.8
    nb_error = 0
    for line in file(sample_file):
        sample = Sample.load_from_line(line)
        if sample.valid():
            samples.append(sample)
            tot_cnt[sample.label] += 1
        else:
            nb_error += 1
    train_limits = [round(cnt * ratio) for cnt in tot_cnt]
    test_limits = [round(cnt * (1-ratio)) for cnt in tot_cnt]
    label_ratio = 100
    adjust(train_limits, label_ratio)
    adjust(test_limits, label_ratio)
    print "#error =", nb_error
    print "tot_cnt =", tot_cnt
    print "train_limit =", train_limits
    print "test_limit =", test_limits
    train_samples = []
    test_samples = []
    for sample in samples:
        label = sample.label
        if train_limits[label] > 0:
            train_samples.append(sample)
            train_limits[label] -= 1
        elif test_limits[label] > 0:
            test_limits[label] -= 1
            test_samples.append(sample)

    print_to_local(train_samples, os.path.join(exper_dir, "emergency_train_%d_%d_%s.h5" %(label_ratio, max_len, ignore_1)), max_len)
    print_to_local(test_samples, os.path.join(exper_dir, "emergency_test_%d_%d_%s.h5" %(label_ratio, max_len, ignore_1)), max_len)
    print train_limits
    print test_limits



