#encoding: utf-8
import os
import pandas
import json
from tqdm import tqdm
from util import eiCU_data_dir, Print
from collections import defaultdict
import glob

class DiagnosisList:
    def __init__(self):
        self.diags = list()
    
    def add(self, diag_idx):
        self.diags.append(diag_idx)
    
    def to_obj(self):
        obj = sorted(set(self.diags))
        return obj
    
    @staticmethod
    def load_from_obj(obj):
        diag_set = DiagnosisList()
        diag_set.diags = list(obj)
        return diag_set


class ICUDiagSetting:
    def __init__(self):
        self.diags_for_offset = {}

    def add(self, offset, diag_idx):
        offset = int(offset)
        if not offset in self.diags_for_offset:
            self.diags_for_offset[offset] = DiagnosisList()
        self.diags_for_offset[offset].add(index)
    
    def to_obj(self):
        obj = {}
        for offset in self.diags_for_offset:
            obj[offset] = self.diags_for_offset[offset].to_obj()
        return obj
    
    @staticmethod
    def load_from_obj(obj):
        icu_diag_setting = ICUDiagSetting()
        for offset in obj:
            icu_diag_setting.diags_for_offset[int(offset)] = \
                DiagnosisList.load_from_obj(obj[offset])
        return icu_diag_setting

def collect_puid(puid_path):
    if os.path.exists(puid_path):
        return
    coded_event_dir = os.path.join(eiCU_data_dir, 'coded_event')
    puids = set()
    Print('---- collect puid ----')
    for filepath in glob.glob(coded_event_dir + '/*csv'):
        filename = os.path.basename(filepath)
        table_name = filename[:filename.find('_event_coded.csv')]
        Print('\tcollect puid from [%s]' %table_name)
        t = pandas.read_csv(filepath)
        table_puids = set(t['puid'])

        puids.update(table_puids)
    Print('---- write to [%s] ----' %puid_path)
    with file(puid_path, 'w') as wf:
        for puid in sorted(puids):
            wf.write(str(puid) + '\n')

def gen_diagnosis_set(puid_path, d_map_path, d_set_path):
    if os.path.exists(d_map_path) and os.path.exists(d_set_path):
        return 
    data_path = os.path.join(eiCU_data_dir, 'raw_event/diagnosis_event.csv')

    if not os.path.exists(d_map_path):
        Print('generate diagnosis map to [%s]' %d_map_path)
        stat_path = os.path.join(eiCU_data_dir, 'statistic/diagnosis_event_statistic.txt')
        code_rate = []
        for line in file(stat_path):
            p = line.split('$')
            code = p[0]
            rate = float(p[-1])
            code_rate.append((code, rate))
        code_rate.sort(key=lambda x: x[1], reverse=True)
        with file(d_map_path, 'w') as wf:
            obj = {}
            idx = 0
            for code, rate in code_rate:
                if len(code.split(',')) != 2:
                    continue
                if code == 'nan':
                    continue
                obj[code] = {
                    'index': idx,
                    'rate': rate
                }
                idx += 1
            json.dump(obj, wf, indent=3)

    d_map = json.load(file(d_map_path, 'r'))
    if not os.path.exists(d_set_path):
        Print('generate diagnosis set to [%s]' %d_set_path)
        icu_diag_setting_map = defaultdict(ICUDiagSetting)
        raw_data_path = os.path.join(eiCU_data_dir, 'raw_event/diagnosis_event.csv')
        t = pandas.read_csv(raw_data_path)
        for idx, row in t.iterrows():
            puid = row['puid']
            offset = row['eventoffset']
            event_code = row['event_code']
            if code not in d_map:
                continue
            diag_idx = d_map[code]['index']
            icu_diag_setting_map[puid].add(offset, diag_idx)

        obj = {}
        for puid in sorted(icu_diag_setting_map.keys()):
            obj[puid] = icu_diag_setting_map[puid].to_obj()
        with file(d_set_path) as wf:
            json.dump(obj, wf, indent=3)

if __name__ == "__main__":
    result_dir = os.path.join(eiCU_data_dir, 'result')

    puid_path = os.path.join(result_dir, 'puids.txt')
    collect_puid(puid_path)

    diagnosis_map_path = os.path.join(result_dir, 'diagnosis_map.json')
    diagnosis_set_path = os.path.join(result_dir, 'diagnosis_set.csv')
    gen_diagnosis_set(puid_path, diagnosis_map_path, diagnosis_set_path)


    