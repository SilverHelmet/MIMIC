import os
from util import Print, event_seq_stat_dir, static_data_dir, parse_time, death_exper_dir, ICU_exper_dir
from .data_util import load_feature_map, load_sample_info
import glob
import json
import h5py
import numpy as np
from tqdm import tqdm

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
                feature_name = table + '#' + name
                if not feature_name in self.feature_name_set:
                    continue
                if obj[name] != "":
                    info[feature_name] = obj[name]

    def get_static_feature(self, info, vec, sample_info, sparse):
        for name in info:
            value = info[name]
            if name == 'patients#dob':
                idx = self.feature_map[name]
                x = (sample_info['st'] - parse_time(value)).days / 365.0
                if x > 90:
                    x = 90
                x /= 90.0
            else:
                name_value = name + "#" + value
                if name_value in self.feature_map:
                    idx = self.feature_map[name_value]
                else:
                    idx = -1
                x = 1 
            if idx >= 0:
                if sparse:
                    vec.append(idx)
                    vec.append(x)
                else:
                    vec[idx] = x


    def gen_feature_of_sample(self, sample_info):
        vec = [0] * self.feature_size
        pid = sample_info['pid']
        hid = sample_info['hid']
        self.get_static_feature(self.p_info_map[pid], vec, sample_info, False)
        self.get_static_feature(self.h_info_map[hid], vec, sample_info, False)
        return vec

    def gen_sparse_feature_of_sample(self, sample_info):
        vec = []
        pid = sample_info['pid']
        hid = sample_info['hid']
        self.get_static_feature(self.p_info_map[pid], vec, sample_info, True)
        self.get_static_feature(self.h_info_map[hid], vec, sample_info, True)
        return vec
    
def generate_static_feature(samples_h5, sample_info_map, static_feature):
    Print('generate static feature of [%s]' %(os.path.basename(samples_h5)))
    f = h5py.File(samples_h5)
    sample_ids = f['sample_id'][:]
    vecs = []
    for sid in tqdm(sample_ids, total = len(sample_ids)):
        info = sample_info_map[sid]
        vec = static_feature.gen_sparse_feature_of_sample(info)
        vecs.append(vec)
    f.close()

    outpath = os.path.dirname(samples_h5) + '/' +  os.path.basename(samples_h5).split('.')[0] + "_static"
    Print('write static feature to [%s]' %os.path.basename(outpath))
    np.save(outpath, np.array(vecs))



if __name__ == "__main__":
    death_sample_setting_path = os.path.join(event_seq_stat_dir, 'death_sample_setting.txt')
    icu_sample_setting_path = os.path.join(event_seq_stat_dir, 'ICUIn_sample_setting.txt')

    sample_setting_path = death_sample_setting_path

    feature_map, n_features = load_feature_map()
    death_sample_info_map = load_sample_info(death_sample_setting_path)
    icu_sample_info_map = load_sample_info(icu_sample_setting_path)

    static_feature = StaticFeature(feature_map, n_features, os.path.join(static_data_dir, 'static_feature'))
    static_feature.load()

    dataset_info_pairs = [
        (os.path.join(death_exper_dir, 'death_train_1000.h5'), death_sample_info_map),
        (os.path.join(death_exper_dir, 'death_valid_1000.h5'), death_sample_info_map),
        (os.path.join(death_exper_dir, 'death_test_1000.h5'), death_sample_info_map),
        (os.path.join(ICU_exper_dir, 'ICUIn_train_1000.h5'), icu_sample_info_map),
        (os.path.join(ICU_exper_dir, 'ICUIn_valid_1000.h5'), icu_sample_info_map),
        (os.path.join(ICU_exper_dir, 'ICUIn_test_1000.h5'), icu_sample_info_map),
    ]

    for dataset, sample_info_map in dataset_info_pairs:
        generate_static_feature(dataset, death_sample_info_map, static_feature)

    

