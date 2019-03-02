import os
from .gen_event_list import Event
from .gen_dataset_setting import load_diag_setting
from .dataset import Dataset, Sample
from util import eiCU_data_dir, Print, get_nb_lines
from tqdm import tqdm
import numpy as np


def puid_offset_key(puid, offset):
    return str(puid) + "#" + str(offset)

def split_dataset(diag_setting_map, ratios, datasets):
    puid_offset_list = diag_setting_map.get_puid_offset_list()
    size = len(puid_offset_list)
    sample_list = [puid_offset_key for puid, offset in puid_offset_list]
    shuffle_idxs = np.random.permutation(size)
    st = 0
    sample2dataset = {}

    for ratio, dataset in zip(ratios, datasets):
        ed = st + int(size * ratio)
        ed = min(ed, size)

        slice_idxs = shuffle_idxs
        for idx in slice_idxs:
            sample2dataset[sample_list[idx]] = dataset

        st = ed
    assert st == szie
    return sample2dataset


def gen_samples_for_puid(line, diag_setting_map, dataset):
    p = line.rstrip().split('|')
    puid = int(p[0])
    diag_setting = diag_setting_map.get_diags_for_puid(puid)
    offset_list = diag_setting.get_offset_list()
    event_list = []
    for idx in range(1, len(p)):
        event_str = p[idx]
        event = Event.load_from_str(event_str)
        event_list.append(event)
    
    for offset in offset_list:
        diags = diag_setting.get_diags_for_offset(offset)
        sample = Sample(puid, offset, diags)
        dataset.add_sample(sample, event_list)        

def gen_dataset(diag_setting_path, event_list_path, out_dir):
    diag_setting_map = load_diag_setting(diag_setting_path)
    # names = ['train', 'valid', 'test']
    # datasets = []
    # for name in names:
    #     outpath = os.path.join(out_dir, 'eiCU_diagnosis_{}.h5'.format(name))
    #     d = Dataset(outpath=outpath)
    #     datasets.append(d)

    # split_dataset(diag_setting_map, [0.7, 0.1, 0.2], datasets)


    Print('---- generate dataset from [%s] ----' %event_list_path)
    dataset = Dataset()
    total = get_nb_lines(event_list_path)
    # total = 200304
    for line in tqdm(file(event_list_path), total=total):
        gen_samples_for_puid(line, diag_setting_map, dataset)
    
    sub_datasets = dataset.split([0.7, 0.1, 0.2])
    names = ['train', 'valid', 'test']
    for name, d in zip(names, sub_datasets):
        outpath = os.path.join(out_dir, "eiCU_diagnosis_{}.h5".format(name))
        size = len(d.event)
        Print('save {} samples to {}'.format(size, outpath))
        d.save(outpath)






if __name__ == "__main__":
    dataset_dir = os.path.join(eiCU_data_dir, 'dataset')
    diag_setting_path = os.path.join(eiCU_data_dir, 'result/diagnosis_set.json')
    event_list_path = os.path.join(dataset_dir, 'event_list.txt')

    gen_dataset(diag_setting_path, event_list_path, dataset_dir)
    