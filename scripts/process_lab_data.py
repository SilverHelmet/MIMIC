import h5py
import cPickle as pickle
import numpy as np
import time
from tqdm import tqdm
from util import lab_exper_dir, parse_time
import os

def time_to_stamp(time_string, start, time_format='%Y-%m-%d %H:%M:%S'):
    if (time_string==""):
        return -10800.
    return time.mktime(time.strptime(time_string, time_format)) - start

def feature_norm(features):
    global mv_dict
    fshape = features.shape
    assert fshape[2]==6
    for i in range(fshape[0]):
        for j in range(fshape[1]):
            for k in range(3):
                tmp = features[i,j,2*k]
                if tmp>0:
                    mean, std = mv_dict[tmp]
                    features[i,j,2*k+1] = (features[i,j,2*k+1]-mean)*1.0/std
    return features

def load_data(path, start, end):
    f = h5py.File(path)
    labels = f['label'][start:end]
    events = f['event'][start:end]
    times = f['time'][start:end]
    features = f['feature'][start:end]
    features = feature_norm(features)

    time_shift = []
    for id in xrange(times.shape[0]):

        event_time = times[id]
        st = parse_time(event_time[0])
        if not st:
            print id
        assert st
        new_event_time = []
        for time in event_time:
            if len(time) > 0:
                t = parse_time(time)
                new_event_time.append(t.hour + t.minute / 60.0 + t.second / 3600.0)
            else:
                new_event_time.append(-1)

        time_shift.append(new_event_time)
    times = np.asarray(time_shift, dtype='float32')

    chosen_event = []
    chosen_time = []
    chosen_label = []
    chosen_feature = []
    # chosen_feature_id = []
    # chosen_feature_value = []
    # tic = time.time()
    for id in tqdm(xrange(labels.shape[0]), total = labels.shape[0]):
        this_label = labels[id]
        this_event = events[id]
        this_feature = features[id]
        # this_feature_id = features[id][:,(0,2,4)]
        # this_feature_value = features[id][:,(1,3,5)]
        this_time = times[id]

        chosen = this_label[(this_label[:,0]==34)+(this_label[:,0]==35)]#choose id 34 or 35
        for tmp in chosen:
            this_start = int(tmp[-1])
            this_end = int(tmp[-2])
            if this_end<=this_start+10:#ignore seq whose len<10
                continue
            if this_end>this_start+1000:#cut max seq len to 100
                this_start = this_end-1000
            pad_num = 1000-this_end+this_start
            chosen_event.append(
                np.pad(this_event[this_start+1:this_end+1], ((0,pad_num),), 'constant'))
            # ap_time = this_time[this_start+1:this_end+1]-this_time[this_start+1]
            ap_time = this_time[this_start+1:this_end+1]
            chosen_time.append(
                np.pad(ap_time, ((0, pad_num),), 'constant'))
            chosen_feature.append(
                np.pad(this_feature[this_start + 1: this_end + 1], ((0, pad_num), (0,0)), 'constant')
            )
            # chosen_feature_id.append(
            #     np.pad(this_feature_id[this_start+1:this_end+1], ((0,pad_num),(0,0)), 'constant'))
            # chosen_feature_value.append(
            #     np.pad(this_feature_value[this_start+1:this_end+1], ((0,pad_num),(0,0)), 'constant'))
            chosen_label.append(tmp[2])#use value as label

    chosen_event = np.asarray(chosen_event, dtype='int16')
    chosen_time = np.asarray(chosen_time)
    chosen_feature = np.array(chosen_feature)
    # chosen_feature_id = np.asarray(chosen_feature_id, dtype='int16')
    # chosen_feature_value = np.asarray(chosen_feature_value, dtype='float32')
    chosen_label = np.asarray(chosen_label, dtype='float32')
    f.close()
    return chosen_event, chosen_time, chosen_feature, chosen_label

def load_data_all(name, start, end):
    lab_path = os.path.join(lab_exper_dir, 'Lab.h5')
    f = h5py.File(lab_exper_dir + '/labtest_{}_1000.h5'.format(name), 'w')
    events, times, chosen_feature, labels = load_data(lab_path, start, end)
    f['event'] = events
    f['time'] = times
    f['normed_feature'] = chosen_feature
    f['label'] = labels
    f.close()
    
with open(os.path.join(lab_exper_dir, 'mv_dict.pkl'), 'rb') as f:
    mv_dict = pickle.load(f)
size = 1139
load_data_all('test', 0, 0 + size * 2)
load_data_all('train', 9112, 9112 + size * 7)
load_data_all('valid', 41006, 41006 + size)
# load_data_all('test', 0, 9112)
# load_data_all('train', 9112, 41006)
# load_data_all('valid', 41006, 45563)