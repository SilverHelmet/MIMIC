#encoding: utf-8
import os
import pandas
import json
from tqdm import tqdm
from util import eiCU_data_dir, Print
import glob

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
        code_rate.sort(key = lambda x: x[1], reverse = True)
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
            json.dump(obj, wf, indent = 3)

if __name__ == "__main__":
    result_dir = os.path.join(eiCU_data_dir, 'result')

    puid_path = os.path.join(result_dir, 'puids.txt')
    collect_puid(puid_path)

    diagnosis_map_path = os.path.join(result_dir, 'diagnosis_map.json')
    diagnosis_set_path = os.path.join(result_dir, 'diagnosis_set.csv')
    gen_diagnosis_set(puid_path, diagnosis_map_path, diagnosis_set_path)


    