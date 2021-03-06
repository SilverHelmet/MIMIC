import scripts_util
import os
import glob
from util import *
from tqdm import tqdm
from extractor import parse_line
import json
from event_des import column_name_map

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
    return event_cnt

def count_coverage(filepath, event_cnt):
    for line in tqdm(file(filepath), total = get_nb_lines(filepath)):
        p = parse_line(line)
        
        event = p[2]
        if event in event_cnt:
            pid = int(p[0].split("_")[0])
            event_cnt[event].add(pid)

def get_coverage(top_events):
    
    Print('get coverage')
    nb_files = get_nb_files(data_dir + "/*tsv")
    event_cnt = {}
    for e in top_events:
        event_cnt[e] = set()
    for idx, filepath in enumerate(glob.glob(data_dir + "/*tsv"), start = 1):
        Print("process %d/%d %s" %(idx, nb_files, os.path.basename(filepath)))
        count_coverage(filepath, event_cnt)
    return event_cnt





if __name__ == "__main__":
    
    item_des = ItemDes()
    event_cnt = get_top_freq_event()
    top_events = []
    for event in sorted(event_cnt.keys(), key = lambda x:event_cnt[x], reverse = True)[:2000]:
        # print event, event_cnt[event], item_des.get_event_des(event)
        top_events.append(event)


    event_p_cnt = get_coverage(top_events)
    nb_patients = 46520
    outf = file(os.path.join(result_dir, "top_freq_event.json"), 'w')
    column_map = column_name_map()
    for e in top_events:
        coverage = round((len(event_p_cnt[e]) + 0.0)/ nb_patients, 3)
        table = e.split('.')[0]
        attrs = column_map.get(table, [])
        if len(attrs) == 0: 
            attrs = column_map.get(e, [])
        obj = {
            'event': e,
            'description': item_des.get_event_des(e),
            'table': table,
            'attrs': attrs,
            'frequency': event_cnt[e],
            'coverage': coverage
        }
        outf.write(json.dumps(obj) + "\n")
    outf.close()


