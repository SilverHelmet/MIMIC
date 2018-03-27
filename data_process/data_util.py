import os
from util import event_seq_stat_dir, static_data_dir, parse_time
import json

def load_feature_map(filepath = None):
    if filepath is None:
        filepath = os.path.join(static_data_dir, 'static_feature_map.tsv')

    feature_map = {}
    max_idx = 0
    for line in file(filepath):
        key, idx = line.split('\t')
        feature_map[key] = int(idx)
        max_idx = max(max_idx, int(idx))
    return feature_map, max_idx + 1

def load_sample_info(filepath):
    sample_info_map = {}
    for line in file(filepath):
        obj = json.loads(line)
        pid = obj['pid']
        settings = obj['settings']
        for setting in settings:
            sid = setting['sample_id']
            hid = setting['hid']
            st = parse_time(setting['st'])
            sample_info_map[sid] = {'pid': pid, hid': hid, 'st': st}
    return sample_info_map


    