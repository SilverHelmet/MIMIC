import os
import sys
import json
import glob
from tqdm import tqdm
from util import eiCU_data_dir, Print, get_nb_lines
from collections import defaultdict

class Event:
    def __init__(self, offset, code, fea_idxs, fea_values):
        self.offset = offset
        self.code = code
        self.fea_idxs = fea_idxs
        self.fea_values = fea_values

    @staticmethod
    def parse_line(line):
        p = line.rstrip().split(',')
        _, offset, puid, num_fea_str, code = p
        fea_str_list = num_fea_str.split('$')
        fea_idxs = []
        fea_values = []
        for idx_value in fea_str_list:
            idx, value = idx_value.split(":")
            if value == 'nan':
                continue
            fea_idxs.append(int(idx))
            fea_values.append(float(value))
        event = Event(offset, code, fea_idxs, fea_values)
        return puid, event
    
    def __str__(self):
        return ",".join(map(str, [self.offset, self.code] + self.fea_idxs  + self.fea_values))


def gen_event_list():
    event_list_map = defaultdict(list)

    event_dir = os.path.join(eiCU_data_dir, 'coded_event')
    for filepath in glob.glob(event_dir + '/*csv'):
        Print('load event from [%s]' %os.path.basename(filepath))
        for line in tqdm(file(filepath), total = get_nb_lines(filepath)):
            if line.startswith(','):
                continue
            puid, event = Event.parse_line(line)
            event_list_map[puid].append(event)

    dataset_dir = os.path.join(eiCU_data_dir, 'dataset')
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    outpath = os.path.join(dataset_dir, 'event_list.txt')
    wf = file(outpath, 'w')
    for puid in sorted(event_list_map.keys()):
        event_list = event_list_map[puid]
        event_list.sort(key=lambda x: (x.offset, x.code))
        event_list_str = map(str, event_list)
        wf.write(str(puid) + "|")
        wf.write("|".join(event_list_str))
        wf.write('\n')
    wf.close()

if __name__ == "__main__":
    gen_event_list()
