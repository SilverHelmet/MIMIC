import os
from util import Print, event_seq_stat_dir, static_data_dir, parse_time
from .data_util import load_feature_map, load_sample_info
import glob
import json

class StaticFeature:
    def __init__(self, feature_map, feature_size, base_dir):
        self.feature_map = feature_map
        self.feature_size = feature_size
        self.base_dir = base_dir
        self.feature_name_set = self.get_feature_name_set(self.feature_map)
        self.p_info_map = {}
        self.h_info_map = {}

    def get_feature_name_set(self, feature_map):
        name_set = set()
        for name_value in feature_map:
            p = name_value.split("#")
            assert len(p) <= 3
            feature_name = p[0] + "#" + p[1]
            name_set.add(feature_name)
        return name_set
    
    def load(self):
        self.load_static_info_from_file(os.path.join(self.base_dir, 'admissions.json'), 'hadm_id', self.h_info_map)
        self.load_static_info_from_file(os.path.join(self.base_dir, 'diagnoses_icd.json'), 'hadm_id', self.h_info_map)
        self.load_static_info_from_file(os.path.join(self.base_dir, 'patients.json'), 'subject_id', self.p_info_map)
        self.load_static_info_from_file(os.path.join(self.base_dir, 'procedures_icd.json'), 'hadm_id', self.h_info_map)

    def load_static_info_from_file(self, filepath, key_name, info_map):
        table = os.path.basename(filepath).split('.')[0]
        for line in file(filepath):
            obj = json.loads(line)
            key = obj[key_name]
            new_obj = {}
            if not key in info_map:
                info_map[key] = {}
            info = info_map[key]
            for name in obj:
                if not name in self.feature_name_set:
                    continue
                if obj[name] != ""
                    info[table + "#" + name] = obj[name]

    def get_static_feature(self, info, vec, sample_info):

        for name in info:
            value = info[value]
            if name == 'patients#dob':
                idx = self.feature_map[name]
                x = sample_info['st'] - 
            else:
                idx = self.feature_map[name + "#" + value]
                x = 1 
            
            
            feature = name + "#" + value

    def gen_feature_of_sample(self, sample_info):
        vec = [0] * self.feature_size
        self.get_static_feature(self.p_info_map[pid], vec, sample_info)
        self.get_static_feature(self.h_info_map[hid], vec, sample_info)
        return vec

if __name__ == "__main__":
    death_sample_setting_path = os.path.join(event_seq_stat_dir, 'death_sample_setting.txt')
    icu_sample_setting_path = os.path.join(event_seq_stat_dir, 'ICUIn_sample_setting.txt')

    sample_setting_path = death_sample_setting_path

    feature_map, n_features = load_feature_map()
    sample_info_map = load_sample_info(sample_setting_path)

    static_feature = StaticFeature(feature_map, n_features, os.path.join(static_data_dir, 'static_feature'))
    static_feature.load()

    sample_info = {
        'pid': 2,
        'hid': 163353,
        'st': parse_time('2138-07-17 19:04:00')
    }

    static_feature.gen_feature_of_sample(sample_info)


