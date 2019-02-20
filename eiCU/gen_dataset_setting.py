import os
import pandas
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
        puids.union(table_puids)
        break
    Print('---- write to [%s] ----' %puid_path)
    with file(puid_path, 'w') as wf:
        for puid in sorted(puids):
            wf.write(str(puid) + '\n')

if __name__ == "__main__":
    result_dir = os.path.join(eiCU_data_dir, 'result')
    puid_path = os.path.join(result_dir, 'puids.txt')
    
    collect_puid(puid_path)


    