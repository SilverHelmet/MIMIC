import pickle
import h5py
import os
import datetime
from tqdm import tqdm
import numpy as np

def parse_time(time_str):
    if len(time_str) in [18, 19]:
        try:
            return datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            return None
    elif len(time_str) == 10:
        try:
            return datetime.datetime.strptime(time_str, '%Y-%m-%d')
        except Exception as e:
            return None
    elif len(time_str) in [12, 13, 14]:
        try:
            return datetime.datetime.strptime(time_str, '%m/%d/%y %H:%M')
        except Exception as e:
            return None
    elif len(time_str) == 16:
        try:
            return datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M")
        except Exception as e:
            return None
    return None


def process_icd_data(idxs, data, name, st, ed, chosen_label, seq_len):
    d_idxs = idxs[st:ed]
    print ('process {}'.format(name))
    t = data['time'][d_idxs, :seq_len]
    l = data['label'][d_idxs]
    feature = data['feature'][d_idxs]
    feature_idx = np.asarray(feature[:, :seq_len, [0,2,4]], dtype = 'int16')
    feature_value = np.asarray(feature[:, :seq_len, [1,3,5]], dtype = 'float32')
    event = np.asarray(data['event'][d_idxs, :seq_len], dtype = 'int16')
    time_hour = np.zeros(event.shape, dtype = 'float32')
    print feature_idx.shape
    print feature_value.shape
    print event.shape
    print time_hour.shape
    print t.shape

    for i in tqdm(range(len(t)), total = len(t)):
        row = t[i]
        for j in range(len(row)):
            time_str = str(row[j])
            if len(time_str) > 0:
                date = parse_time(time_str)
                if date is None:
                    print ''
                    print  row[j]
                    print time_str
                    print ''
                assert date is not None
                hour = date.hour + date.minute / 60.0 + date.second / 3600.0
                time_hour[i][j] = hour
            else:
                time_hour[i][j] = .0
    time_hour = np.asarray(time_hour, dtype = 'float32')

    new_labels = []
    for labels in l:
        if chosen_label in  labels:
            new_labels.append(1)
        else:
            new_labels.append(0)
    new_labels = np.asarray(new_labels, dtype='int8')
    num_pos = new_labels.sum()
    num_neg = (new_labels == 0).sum()
    print("#pos = %d, #neg = %d" %(num_pos, num_neg))

    outpath = 'icd_exper/icd9_{}_{}.h5'.format(name, seq_len)
    if not os.path.exists('icd_exper'):
        os.mkdir('icd_exper')
    f = h5py.File(outpath, 'w')
    f['label'] = new_labels
    f['time'] = time_hour
    f['event'] = event
    f['feature_idx'] = feature_idx
    f['feature_value'] = feature_value

    f.close()

    

if __name__ == "__main__":
    size = 35017
    label = 8
    seq_len = 1000
    data_path = 'icd_exper/icd.pkl'
    f = open(data_path, "rb") 
    data = pickle.load(f)
    f.close()
    idxs = np.random.permutation(size)
    process_icd_data(idxs, data, 'valid', 0, 3500, label, seq_len)
    process_icd_data(idxs, data, 'test', 3500, 3500*3, label, seq_len)
    process_icd_data(idxs, data, 'train', 3500*3, 35017, label, seq_len)

