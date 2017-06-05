import scripts_util
import os
import glob
from ..util import *
from tqdm import tqdm
from ..extractor import parse_line

class ItemDes:
    def __init__(self):
        self.item_map = self.load_d_item_table()
        self.labitem_map = self.load_d_labitem_table()

    @staticmethod
    def load_d_item_table():
        filepath = os.path.join(static_data_dir, 'item_code.tsv')
        des_map = {}
        for line in file(filepath):
            code, label = line.strip('\n').split('\t')
            assert code not in des_map
            des_map[code] = {'label': label}
        return des_map

    @staticmethod
    def load_d_labitem_table():
        filepath = os.path.join(static_data_dir, 'labitem_code.tsv')
        des_map = {}
        for line in file(filepath):
            code, des = line.strip('\n').split("\t")
            assert code not in des_map
            label, fluid = des.split(' | ')
            des_map[code] = {'label': label, "fluid": fluid}
        return des_map

    def get_des(self, table, code = None):
        if code is None:
            return {}
        if table.upper() in ['CHARTEVENTS', 'DATETIMEEVENTS', 'INPUTEVENTS_CV', 'INPUTEVENTS_MV', 'MICROBIOLOGYEVENTS', 'OUTPUTEVENTS', 'PROCEDUREEVENTS_MV']:
            return self.item_map[code]
        elif table.upper() == 'LABEVENTS':
            return self.labitem_map[code]
        else:
            return {}

    def get_event_des(self, event):
        p = event.split(".")
        assert len(p) < 3
        if len(p) == 1:
            table = p[0]
            code = None
        else:
            table, code = p
        return self.get_des(table, code)

def load_data(filepath, event_cnt):
    for line in tqdm(file(filepath), total = get_nb_lines(filepath)):
        p = parse_line(line)
        pid = int(p[0].split("_")[0])
        event  = p[2]
        if not event in event_cnt:
            event_cnt[event] = 0
        event_cnt[event] += 1
    return event_cnt
        

def get_top_freq_event():
    Print("get top freq event")
    nb_files = get_nb_files(data_dir + "/*tsv")
    event_cnt = {}
    for idx, filepath in enumerate(glob.glob(data_dir + "/*tsv"), start = 1):
        Print("process %d/%d %s" %(idx, nb_files, os.path.basename(filepath)))
        load_data(filepath, event_cnt)
        break
    return event_cnt




if __name__ == "__main__":
    item_des = ItemDes()
    event_cnt = get_top_freq_event()

    for event in sorted(event_cnt.keys(), key = lambda x:event_cnt[x], reverse = True):
        print event, event_cnt[event], item_des.get_des(event)


