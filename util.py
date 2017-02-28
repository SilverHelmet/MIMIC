from pg import DB
from pandas import DataFrame
import sys
import os
import datetime
import time
import re
import numpy as np


def connect():
    host = '162.105.146.245'
    host = 'localhost'
    schema = 'mimiciii'
    db = DB(host = host, user = 'mimic', passwd = 'mimic')
    db.query('set search_path to %s' %(schema))
    return db

class Patient():

    bs_attrs = []

    def __init__(self, row):
        self.values = {}
        self.names = []
        for field in Patient.bs_attrs:
            self.values[field] = row[field]
            self.names.append(field)

    def to_row(self):
        ret = []
        for name in self.names:
            ret.append(self.values[name])
        return ret
        
    @staticmethod
    def set_attrs(columns):
        Patient.bs_attrs = []   
        for field in columns:
            Patient.bs_attrs.append(field)

    @staticmethod
    def write_to_local(patients, path):
        columns = None
        data = []
        index = []
        for pid, patient in patients.iteritems():
            if columns is None:
                columns = patient.names
            data.append(patient.to_row())
            index.append(pid)

        dt = DataFrame(data = data, index = index, columns = columns)
        dt.sort_index()
        dt.to_csv(path)

def date2str(date):
    return date.strftime('%Y-%m-%d')

def time2str(time):
    return time.strftime('%Y-%m-%d %H:%M:%S')

def parse_time(time_str):
    if len(time_str) == 19:
        try:
            return datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except Exception, e:
            return None
    elif len(time_str) == 10:
        try:
            return datetime.datetime.strptime(time_str, '%Y-%m-%d')
        except Exception, e:
            return None
    elif len(time_str) in [12, 13, 14]:
        try:
            return datetime.datetime.strptime(time_str, '%m/%d/%y %H:%M')
        except Exception, e:
            return None
    return None

def parse_number(number_str):
    try:
        return float(number_str)
    except Exception, e:
        return None

def is_time(time_str):
    time = parse_time(time_str)
    return time is not None

def is_number(number_str):
    number = parse_number(number_str)
    return number is not None

def load_reg(filepath):
    regs = []
    for line in file(filepath):
        line = line.strip()
        if line.startswith("#"):
            continue
        if line == "":
            continue
        regs.append(re.compile(line))
    return regs

def load_id2event_value():
    ret = {}
    for line in file(os.path.join(result_dir, "event_des_text.tsv")):
        parts = line.strip("\n").split(" ")
        event_id = int(parts[0])
        event_type = parts[1]
        value = " ".join(parts[2:])
        ret[event_id] = event_type + '.' + value
    return ret

def load_id2event_rtype():
    ret = {}
    for line in file(os.path.join(result_dir, "event_des_text.tsv")):
        parts = line.strip("\n").split(" ")
        event_id = int(parts[0])
        event_type = parts[1]
        value = " ".join(parts[2:])
        ret[event_id] = event_type
    return ret

    

def merge_prob(probs, ids, func):
    prob_map = {}
    assert len(probs) == len(ids)
    for i in range(len(probs)):
        prob = probs[i]
        sid = ids[i]
        if not sid in prob_map:
            prob_map[sid] = prob
        else:
            prob_map[sid] = func(prob_map[sid], prob)
    probs = []
    for sid in sorted(prob_map.keys()):
        probs.append(prob_map[sid])
    return np.array(probs)

def merge_label(labels, ids):
    label_map = {}
    for i in range(len(labels)):
        label = labels[i]
        sid = ids[i]
        if not sid in label_map:
            label_map[sid] = label
    return np.array([label_map[sid] for sid in sorted(label_map.keys())])

def norm_to_prob(X):
    y = np.expand_dims(X.sum(-1), -1)
    y[y == 0] = 1
    return X / y

def load_numpy_array(filepath):
    return np.load(filepath)

def now():
    return datetime.datetime.now().strftime('%m-%d %H:%M:%S')

def merge_event_map(filepath):
    print "load event des from [%s]" %filepath
    new_idx_cnt = 2
    new_events_idx = {}
    old2new = {0: 0, 1: 1}
    for line in file(filepath):
        line = line.strip()
        if line == "":
            conitnue
        parts = line.split(" ")
        old_idx = int(parts[0])
        rtype = parts[1]
        if not rtype in new_events_idx:
            new_events_idx[rtype] = new_idx_cnt
            new_idx_cnt += 1
        old2new[old_idx] = new_events_idx[rtype]
    return old2new

def load_items(filepath):
    items = {}
    for line in file(filepath):
        line = line.strip()
        if line == "":
            continue
        p = line.split('\t')
        code = int(p[0])
        if len(p) == 1:
            des = ""
        else:
            des = p[1]
        items[code] = des
    return items

        


    

def load_setting(filepath, default_setting):
    setting = default_setting if default_setting else {}

    if filepath.startswith("#"):
        lines = filepath[1:].split("|")
    else:
        lines = file(filepath).readlines()
    for line in lines:
        line = line.rstrip()
        if line == "":
            continue
        if line.startswith("#"):
            continue
        parts = line.split("|")
        for key_value in parts:
            x = key_value.strip().split("=")
            if len(x) >= 2:
                key = x[0]
                if x[1] == "True":
                    value = True
                elif x[1] == "False":
                    value = False
                elif is_number(x[1]):
                    if x[1].isdigit():
                        value = int(x[1])
                    else:
                        value = float(x[1])
                else:
                    value = "=".join(x[1:])
                setting[key] = value
                print "load arg %s = %s" %(key, value)
    return setting



script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, 'data')
stat_dir = os.path.join(script_dir, 'stat')
result_dir = os.path.join(script_dir, 'result')
static_data_dir = os.path.join(script_dir, "static_data")
config_dir = os.path.join(script_dir, 'config')
event_dir = os.path.join(script_dir, 'event')
event_stat_dir = os.path.join(script_dir, "event_stat")
# exper_dir = os.path.join(script_dir, "exper")
death_exper_dir = os.path.join(script_dir, 'death_exper')
death_seg_dir = os.path.join(death_exper_dir, 'segs')
death_merged_exper_dir = os.path.join(script_dir, 'death_merged_exper')
ICU_exper_dir = os.path.join(script_dir, "ICU_exper")
ICU_merged_exper_dir = os.path.join(script_dir, "ICU_merged_exper")
ICU_seg_dir = os.path.join(ICU_exper_dir, 'segs')
ICU_emd_dir = os.path.join(ICU_exper_dir, 'embeddings')
event_seq_stat_dir = os.path.join(script_dir, "event_seq_stat")
graph_dir = os.path.join(script_dir, 'graph')
time_dis_graph_dir = os.path.join(graph_dir, "time_dis")

if __name__ == "__main__":
    s = "01/3/14 00:00"
    print parse_time(s)
    # print event_des_pattern['inputevents_cv.30026']
    