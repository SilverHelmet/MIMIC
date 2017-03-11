#encoding:utf-8
import os
import sys
import zhu_util
from util import *
from build_sample_setting import DiagnosisSampleSetting
import h5py
import numpy as np

class Dataset:
    max_event_len = 1000
    def __init__(self, outpath):
        self.outpath = outpath
        self.event = []
        self.sample_id = []
        self.label = []
        self.time = []
        self.feature = []
        self.max_len = 0

    def add_sample(self, sample_id, event, label, time, feature):
        if len(event) <= 10:
            return
        self.max_len = max(self.max_len, len(event))
        assert len(event) <= Dataset.max_event_len
        event.extend([0] * (Dataset.max_event_len - len(event)))
        time = map(str, time)
        time.extend([""] * (Dataset.max_event_len - len(time)))
        f_len = feature.shape[1]
        feature = list(feature)
        feature.extend([[0] * f_len for i in range(Dataset.max_event_len - len(feature))])

        self.event.append(event)
        self.label.append(label)
        self.sample_id.append(sample_id)
        self.time.append(time)
        self.feature.extend(feature)

    def save(self):
        print 'size', len(self.label)
        print 'max len', self.max_len
        f = h5py.File(self.outpath, 'w')
        f['event'] = np.array(self.event)
        f['sample_id'] = np.array(self.sample_id)
        f['label'] = np.array(self.label)
        f['time'] = np.array(self.time)
        f['feature'] = np.array(self.feature)
        f.close()
        

def find_point(time_list, point):
    for i in range(len(time_list)):
        if time_list[i] >= point:
            return i
    return len(time_list)
        

def build_sample(event_list, time_list, feature_list, setting):
    ed = setting.ed
    fi = find_point(time_list, ed)
    st = max(0, fi - Dataset.max_event_len)
    event = event_list[st:fi]
    time = time_list[st:fi]
    feature = feature_list[st:fi]
    return event, time, feature

    
        

def load_setting(filepath):
    setting_map = {}
    for line in file(filepath):
        setting = DiagnosisSampleSetting.load_from_json(line)
        pid = setting.pid
        if not pid in setting_map:
            setting_map[pid] = []
        setting_map[pid].append(setting)
    return setting_map

def split_setting(setting_map, ratios):
    cnts = [0] * 2
    limits = []
    for pid in setting_map:
        for setting in setting_map[pid]:
            cnts[setting.label] += 1
    for cnt in cnts:
        limit = [int(ratio * cnt) for ratio in ratios]
        limits.append(limit)

    print "cnts, ", cnts
    print"limits, ", limits

    for pid in setting_map:
        for setting in setting_map[pid]:
            label = setting.label
            limit = limits[label]
            setting.did = 0
            for i in range(len(limit)):
                if limit[i] > 0:
                    limit[i] -= 1
                    setting.did = i
                    break
                
    print "limits,", limits  

def load_data(filepath):
    f = h5py.File(filepath)
    patients = f['patient'][:]
    events = f['event'][:]
    times = f['time'][:]
    features = f['feature'][:]
    return patients, events, times, features



setting_map = load_setting("zhu_data/sample_settings.json")
patients, events, times, features = load_data("zhu_data/HeartDisease.h5")


dataset_train = Dataset("zhu_exper/train.h5")
dataset_valid = Dataset("zhu_exper/valid.h5")
dataset_test = Dataset("zhu_exper/test.h5")
ds = [dataset_train, dataset_valid, dataset_test]
ratios = [0.7, 0.1, 0.2]
split_setting(setting_map, ratios)

for pid, event_list, time_list, feature_list in zip(patients, events, times, features):
    settings = setting_map.get(pid, [])
    event_list = [e for e in event_list if e != 0]
    time_list = time_list[:len(event_list)]
    feature_list = feature_list[:len(event_list)]
    time_list = [parse_time(time) for time in time_list]
    # for i in range(1, len(time_list)):
    #     if time_list[i] == None:
    #         time_list[i] = time_list[i-1]
    # for i in range(len(time_list)-1, -1, -1):
    #     if time_list[i] == None:
    #         time_list[i] = time_list[i+1]
    for time in time_list:
        assert time
    # time_list = [time if time else datetime.datetime.min for time in time_list]
    # time_list = [parse_time(time) for time in time_list]


    for setting in settings:
        sample_event, sample_time, sample_feature = build_sample(event_list, time_list, feature_list, setting)

        ds[setting.did].add_sample(setting.sample_id, sample_event, setting.label, sample_time, sample_feature)

for d in ds:
    d.save()