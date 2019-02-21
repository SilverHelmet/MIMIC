import sys
from util import eiCU_data_dir
import os
import glob
import json

def load_decode_info(filepath_pattern):
    info = {}
    for filepath in glob.glob(filepath_pattern):
        filename = os.path.basename(filepath)
        table = filename[:filename.find('event_decoded') - 1]
        f = file(filepath, 'r')
        t_info = json.load(f)
        f.close()
        for key in t_info:
            key = int(key)
            info[key] = [table] + t_info[key]
    return info

def merge_index_info():
    data_dir = eiCU_data_dir
    info_dir = os.path.join(data_dir, 'decoded')
    out_dir = os.path.join(data_dir, 'result')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    type_info = load_decode_info(info_dir + "/*type.json")
    fidx_info = load_decode_info(info_dir + "/*fidx.json")
    with file(os.path.join(out_dir, 'event_decoded_type.json'), 'w') as wf:
        json.dump(type_info, wf, indent = 3)
    with file(os.path.join(out_dir, 'event_decoded_fidx.json'), 'w') as wf:
        json.dump(fidx_info, wf, indent = 3)



if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "merge_index_info":
        merge_index_info()
    else:
        assert False
    