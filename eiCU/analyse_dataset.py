import os
import math
import numpy as np
from .dataset import Dataset, Sample
from util import Print, eiCU_data_dir
from collections import defaultdict
import json
import matplotlib.pyplot as plt

class DatasetStat:
    def __init__(self):
        self.ori_len = []
        self.sample_len = []
        self.label_num = []
        self.diag_cnt = defaultdict(int)
    
    def analyse(self, d):
        ori_len = d.ori_len
        self.ori_len.extend(list(ori_len))

        sample_len = map(lambda x: min(x, Sample.MAX_EVENT_LEN), ori_len)
        self.sample_len.extend(list(sample_len))

        for diags in d.label:
            self.label_num.append(len(diags))
            for diag in diags:
                self.diag_cnt[diag] += 1

    def save(self, outpath):
        with file(outpath, 'w') as wf:
            lists = [self.ori_len, self.sample_len, self.label_num, ["{}:{}".format(key, value) for key, value in self.diag_cnt.iteritems()]] 
            for l in lists:
                wf.write(','.join(map(str, l)))
                wf.write('\n')
    
    @staticmethod
    def load(inpath):
        def parse_line_of_list(line):
            return map(int, line.rstrip().split(","))

        def parse_line_of_dict(line):
            p = line.rstrip().split(',')
            d = {}
            for key_value in p:
                key, value = map(int, key_value.split(':'))
                d[key] = value
            return d

        with file(inpath, 'r') as rf:
            ds = DatasetStat()
            ds.ori_len = parse_line_of_list(rf.readline())
            ds.sample_len = parse_line_of_list(rf.readline())
            ds.label_num = parse_line_of_list(rf.readline())
            ds.diag_cnt = parse_line_of_dict(rf.readline())
            return ds

def plot_distribution(X, outpath, maxvalue, nb_bins = 20):
    X = np.array(X)
    X[X > maxvalue] = maxvalue
    plt.figure(0)
    _, bins, _ = plt.hist(X, bins = nb_bins)
    bins = map(int, map(math.ceil, bins))
    bins = sorted(set(bins))
    labels = map(str, bins)
    labels[-1] = ">"  + labels[-1]
    plt.xticks(bins, labels, rotation = 60)
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    plt.savefig(outpath)
    plt.close(0)

def plot(ds):
    outdir = os.path.join(eiCU_data_dir, 'result/statistics')
    values = [ds.ori_len, ds.sample_len, ds.label_num, ds.diag_cnt.values()]
    names = ['ori_len', 'sample_len', 'label_num', 'diag_cnt']
    maxxs = [10000, 1000, 20, 20000]
    for value, name,maxx in zip(values, names, maxxs):
        outpath = os.path.join(outdir, "{}_dist.png".format(name))
        plot_distribution(value, outpath, maxx)


def analyse_datasets(filepaths, diag_setting_path):
    outdir = os.path.join(eiCU_data_dir, 'result/statistics')
    outpath = os.path.join(outdir, 'dataset_statistics.csv')
    if os.path.exists(outpath):
        return outpath

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    ds = DatasetStat()

    for filepath in filepaths:
        Print("---- analyse [%s] ----" %filepath)
        d = Dataset.load(filepath, diag_setting_path)
        ds.analyse(d)
    
    ds.save(outpath)
    return outpath

if __name__ == "__main__":
    dataset_dir = os.path.join(eiCU_data_dir, 'dataset')
    names = ['train', 'valid', 'test']
    # names = ['valid']
    filepaths = [os.path.join(dataset_dir, 'eiCU_diagnosis_{}.h5'.format(name))for name in names]
    diag_setting_path = os.path.join(eiCU_data_dir, 'result/diagnosis_set.json')
    outpath = analyse_datasets(filepaths, diag_setting_path)

    ds = DatasetStat.load(outpath)
    plt.style.use('ggplot')
    plot(ds)
    
