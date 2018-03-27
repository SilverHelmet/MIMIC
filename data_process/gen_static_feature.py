import os
from util import Print, event_seq_stat_dir, static_data_dir
from .data_util import load_feature_map, load_sample_info
import glob


class StaticFeature:
    
    def __init__(self, feature_map, base_dir = None):
        if base_dir is None:
            base_dir = static_data_dir
        self.feature_map = feature_map
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
        self.load_static_info_from_file(os.path.join(self.base_dir, 'diagnoses_icd.json'), 'hadm_id', self.h_info_map)

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
                info[table + "#" + name] = obj[name]

            
                


    def load_admission(self):
        path = os.path.join(self.base_dir, 'admissions.json')




if __name__ == "__main__":
    death_sample_setting_path = os.path.join(event_seq_stat_dir, 'death_sample_setting.txt')
    icu_sample_setting_path = os.path.join(event_seq_stat_dir, 'ICUIn_sample_setting.txt')

    sample_setting_path = death_sample_setting_path

    feature_map = load_feature_map()
    sample_info_map = load_sample_info(sample_setting_path)

    static_info_map = load_static_info()
