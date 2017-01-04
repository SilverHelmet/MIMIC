from util import *
import sys
from scripts.sample_setting import Sample
import json
import h5py
import numpy as np

def add_padding(l, max_len, padding_value = 0):
    assert max_len >= len(l)
    for i in range(max_len - len(l)):
        l.append(padding_value)

def s_generator(filepath, index_set):
    cnt = 0
    for idx, line in enumerate(file(sample_file)):
        if idx in index_set:
            if cnt % 10000 == 0:
                print "\t%s generator %d" %(now(), cnt)
            cnt += 1
            yield Sample.load_from_json(json.loads(line))
        
max_feature_len = 6
def print_to_local_generator(generator, filepath, max_len, event_map):
    global max_feature_len
    f = h5py.File(filepath, 'w')
    labels = []
    events = []
    event_times = []
    features = []
    max_feature_len = 6
    sample_ids = []
    feature_padding = [0] * max_feature_len
    for sample in generator:
        label = sample.sample_setting.label
        sid = sample.sample_setting.sample_id
        event_seq = []
        event_time_seq = []

        feature_seqs = []
        for event in sample.events:
            # may map event to no text retyping event
            # or the map is the identify map
            event.index = event_map[event.index]
            event_seq.append(event.index)
            event_time_seq.append(str(event.time))
            feature_seq = []
            for feature in event.features:
                feature_seq.append(feature.index)
                feature_seq.append(feature.value)
            add_padding(feature_seq, max_feature_len)
            feature_seqs.append(feature_seq)

        add_padding(event_seq, max_len)
        add_padding(event_time_seq, max_len, padding_value = "")
        add_padding(feature_seqs, max_len, feature_padding)
        events.append(event_seq)
        event_times.append(event_time_seq)
        features.append(feature_seqs)
        labels.append(label)
        sample_ids.append(sid)

    f['label'] = np.array(labels)
    f['feature'] = np.array(features)
    f['event'] = np.array(events)
    f['time'] = np.array(event_times)
    f['sample_id'] = np.array(sample_ids)
    f.close()

def adjust(limits, ratio):
    min_limit = reduce(min, limits)
    up_cell = ratio * min_limit
    for i in range(len(limits)):
        limits[i] = min(up_cell, limits[i])

if __name__ == "__main__":
    '''
        python split_ICU_samples_generator.py sample_file [event_des_text]
    '''
    max_len = 1000
    sample_file = sys.argv[1] 
    if len(sys.argv) > 3:
        print "----- merge event to no text retyping event -----"
        event_map = merge_event_map(os.path.join(result_dir, "event_des_text.tsv"))
    else:
        event_map = lambda x: x
    print "load samples from [%s]" %sample_file
    tot_cnt = [0] * 2
    valid_ratio = 0.1
    test_ratio = 0.2
    nb_error = 0
    labels = []
    for idx, line in enumerate(file(sample_file)):
        if idx % 10000 == 0:
            print "\t %s idx = %d" %(now(), idx)
        label = Sample.load_label(json.loads(line))
        tot_cnt[label] += 1
        labels.append(label)
    
    print tot_cnt
    test_limits = [round(cnt * test_ratio) for cnt in tot_cnt]
    valid_limits = [round(cnt * valid_ratio) for cnt in tot_cnt]
    train_limits = [cnt - test_cnt - valid_cnt for cnt, test_cnt, valid_cnt in zip(tot_cnt, test_limits, valid_limits)]
    print "train_limits =", train_limits
    print "valid_limits =", valid_limits
    print "test_limitrs =", test_limits
    train_idx = set()
    valid_idx = set()
    test_idx = set()
    for idx, label in enumerate(labels):
        if train_limits[label] > 0:
            train_idx.add(idx)
            train_limits[label] -= 1
        elif valid_limits[label] > 0:
            valid_idx.add(idx)
            valid_limits[label] -= 1 
        elif test_limits[label] > 0:
            test_limits[label] -= 1
            test_idx.add(idx)
    print train_limits
    print valid_limits
    print test_limits
    print_to_local_generator(s_generator(sample_file, valid_idx), os.path.join(ICU_exper_dir, "ICUIn_valid_%d.h5" %max_len), max_len, event_map)
    print_to_local_generator(s_generator(sample_file, train_idx), os.path.join(ICU_exper_dir, "ICUIn_train_%d.h5" %max_len), max_len, event_map)
    print_to_local_generator(s_generator(sample_file, test_idx), os.path.join(ICU_exper_dir, "ICUIn_test_%d.h5" %max_len), max_len, event_map)

        
        

    
