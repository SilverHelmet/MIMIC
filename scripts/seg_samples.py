import scripts_util
import util
import sys
import h5py
import numpy as np


event_dim = 3391
feature_dim = 668
def sequence2bow(event):
    global event_dim
    size = len(event)
    ret = np.zeros((size, event_dim))
    for i in range(size):
        seq = event[i]
        for event_idx in seq:
            if event_idx != 0:
                ret[i][event_idx] += 1
    return ret

def merge_event(event_seqs, seg_splits):
    ret = []
    for i in range(len(seg_splits)):
        seg = seg_splits[i]
        event_seq = event_seqs[i]
        event = []
        st = 0
        for ed in seg:
            if ed == 0:
                event.append([])
            else:
                event.append(event_seq[st:ed])
                st = ed
        ret.append(sequence2bow(event))
    ret = np.array(ret)
    print "event shape =", ret.shape

def collect_feature(feature_seqs, st, ed):
    length = len(feature_seqs[0])
    vec = np.zeros(feature_dim)
    for i in range(st, ed):
        feature_pairs = feature_seqs[i]
        idx = 0
        while idx < length:
            if feature_pairs[idx+1] != 0:
                index = int(feature_pairs[idx])
                value = feature_pairs[idx+1]
                vec[index] += value
            idx += 2
    return vec


def merge_feature(features, seg_splits):
    global feature_dim
    ret = []
    for i in range(len(seg_splits)):
        feature_seqs = features[i]
        seg = seg_splits[i]
        feature = []
        for ed in seg:
            if ed == 0:
                feature.append(np.zeros(feature_dim))
            else:
                feature.append(collect_feature(feature_seqs, st, ed))
                st = ed
        ret.append(feature)
    ret = np.array(ret)
    print "feature shape =", ret.shape
    return ret


if __name__ == "__main__":
    sample_file = sys.argv[1]
    seg_file = sys.argv[2]
    dataset = h5py.File(sample_file, 'r')
    seg_dataset = h5py.File(seg_file, 'a')
    seg_splits = seg_dataset['segment']
    event = merge_event(dataset['event'][:], seg_splits)
    feature = merge_feature(dataset['feature'][:], seg_splits)
    seg_dataset['event'] = event
    seg_dataset['feature'] = feature
    seg_dataset['sample_id'] = dataset['sample_id'][:]
    seg_dataset.close()
