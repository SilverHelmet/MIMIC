from collections import defaultdict
import glob
from models.dataset import Dataset
from tqdm import tqdm
from util import Print, death_exper_dir, result_dir
import json
import os

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

    def to_json(self):
        return {
            'sum_dict': self.label2hour2sum,
            'cnt_dict': self.label2hour2cnt,
        }

def make_key(eidx, f_idx):
    return (eidx, f_idx)

def load_filepath(filepath, stat_dict):
    d = Dataset(filepath)
    setting = {
        'time_off': 1.0,
        'time_base': 'abs'
        }
    # d.load(True, False, True, True, setting = setting)
    d.load(True, False, True, False, setting = setting)

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
        

if __name__ == "__main__":
    stat_death()
