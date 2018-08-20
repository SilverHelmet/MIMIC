from collections import defaultdict
import glob
from models.dataset import Dataset
from tqdm import tqdm
from util import Print, death_exper_dir, result_dir
import json
import os
import numpy as np

class FValueStat:
    def __init__(self, eidx, fidx):
        self.eidx = eidx
        self.fidx = fidx
        self.label2hour2sum = {}
        self.label2hour2cnt = {}

    def add(self, time, label, value):
        label = int(label)
        if not label in self.label2hour2sum:
            self.label2hour2sum[label] = defaultdict(float)
            self.label2hour2cnt[label] = defaultdict(float)
        hour2sum = self.label2hour2sum[label]
        hour2cnt = self.label2hour2cnt[label]
        hour = int(time)
        hour2sum[hour] += value
        hour2cnt[hour] += 1

    def get_size(self, label):
        label = str(label)
        hour2cnt = self.label2hour2cnt.get(label, {})
        size = sum(hour2cnt.values())
        return size

    def check_size(self, size):
        size0 = self.get_size(0)
        size1 = self.get_size(1)
        return size0 >= size and size1 >= size

    def event_dis(self, label):
        label = str(label)
        if not label in self.label2hour2sum:
            hour2cnt = {}
            print 'error'
        else:
            hour2cnt = self.label2hour2cnt[label]
    
        cnts = []
        for hour in range(24):
            hour = str(hour)
            cnt = hour2cnt.get(hour, .0)
            cnts.append(cnt)
        return np.array(cnts)

    def value_dis(self, label):
        label = str(label)
        if not label in self.label2hour2sum:
            hour2sum = {}
            hour2cnt = {}
            print 'error'
        else:
            hour2sum = self.label2hour2sum[label]
            hour2cnt = self.label2hour2cnt[label]
        mean_v = []
        for hour in range(24):
            hour = str(hour)
            cnt = hour2cnt.get(hour, .1)
            sum = hour2sum.get(hour, .0)
            mean_v.append(sum / cnt)
        return np.array(mean_v)

    def to_json(self):
        return {
            'sum_dict': self.label2hour2sum,
            'cnt_dict': self.label2hour2cnt,
        }

    @staticmethod
    def load_from_line(line):
        eidx, fidx, json_str = line.split('\t')
        obj = json.loads(json_str)
        fv = FValueStat(int(eidx), int(fidx))
        fv.label2hour2sum = obj['sum_dict']
        fv.label2hour2cnt = obj['cnt_dict']
        return fv

def make_key(eidx, f_idx):
    return (eidx, f_idx)

def load_filepath(filepath, stat_dict):
    d = Dataset(filepath)
    setting = {
        'time_off': 1.0,
        'time_base': 'abs'
        }
    d.load(True, False, True, True, None, setting = setting)
    # d.load(True, False, True, False, setting = setting)

    Print("load %s" %filepath)
    for idx in tqdm(range(d.size), total = d.size):
        features = d.features[idx]
        times = d.times[idx]
        events = d.events[idx]
        label = d.labels[idx]
        for i in range(len(times)):
            time = times[i]
            feature_pair = features[i]
            eidx = events[i]
            if eidx == 0:
                continue
            for j in range(3):
                fidx = feature_pair[j * 2]
                if fidx == 0:
                    continue
                fv = feature_pair[j * 2 + 1]
                key = make_key(eidx, fidx)
                if not key in stat_dict:
                    stat_dict[key] = FValueStat(eidx, fidx)
                stat_dict[key].add(time, label, fv)

def load_lab_filepath(filepath, stat_dict):
    d = Dataset(filepath)
    setting = {
        'time_off': 1.0,
        'time_base': 'abs'
        }
    d.load(True, False, True, setting = setting)

    Print("load %s" %filepath)
    for idx in tqdm(range(d.size), total = d.size):
        feature_idxs = d.feature_idxs[idx]
        feature_values = d.feature_values[idx]
        times = d.times[idx]
        events = d.events[idx]
        label = d.labels[idx]
        for i in range(len(times)):
            time = times[i]
            idxs = features_idxs[i]
            values = feature_values[i]
            eidx = events[i]
            if eidx == 0:
                continue
            for fidx, fv in zip(idxs, values):
                if fidx == 0:
                    continue
                key = make_key(eidx, fidx)
                if not key in stat_dict:
                    stat_dict[key] = FValueStat(eidx, fidx)
                stat_dict[key].add(time, label, fv)
    
def stat_death():
    stat_dict = {}
    parttern = death_exper_dir + '/death_*_1000.h5'
    # parttern = "death_exper/sample/samples.h5"
    for filepath in glob.glob(parttern):
        # print filepath
        load_filepath(filepath, stat_dict)
    outpath = os.path.join(result_dir, 'death_value.stat.json')
    Print("write result to [%s]" %outpath)
    outf = file(outpath, 'w')
    for key in sorted(stat_dict.keys()):
        eidx, fidx = key
        json_obj = stat_dict[key].to_json()
        outf.write("%d\t%d\t%s\n" %(eidx, fidx, json.dumps(json_obj)))
    outf.close()

def stat_labtest():
    stat_dict = {}
    parttern = 'lab_exper/labtest_*_1000.h5'
    for filepath in glob.glob(parttern):
        load_lab_filepath(filepath, stat_dict)
    outpath = os.path.join(result_dir, 'labtest_value.stat.json')
    Print("write result to [%s]" %outpath)
    outf = file(outpath, 'w')
    for key in sorted(stat_dict.keys()):
        eidx, fidx = key
        json_obj = stat_dict[key].to_json()
        outf.write("%d\t%d\t%s\n" %(eidx, fidx, json.dumps(json_obj)))    
    outf.close()

if __name__ == "__main__":
    # stat_death()
    stat_labtest()
