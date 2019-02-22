import numpy as np
import h5py
import math

class Event:
    MAX_FEA_LEN = 4
    def __init__(self, offset, code, fea_idxs, fea_values):
        self.offset = offset
        self.code = code
        self.fea_idxs = fea_idxs
        self.fea_values = fea_values

    @staticmethod
    def parse_line(line):
        p = line.rstrip().split(',')
        _, offset, puid, num_fea_str, code = p
        puid = int(puid)
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

    def pad_feature(self):
        while len(self.fea_idxs) < Event.MAX_FEA_LEN:
            self.fea_idxs.append(0)
            self.fea_values.append(0)
    
    @staticmethod
    def padded_event():
        return Event(0, 0, [0] * Event.MAX_FEA_LEN, [0] * Event.MAX_FEA_LEN)

    @staticmethod
    def load_from_str(s):
        p = s.split(',')
        offset, code = map(int, p[:2])
        n_fea = (len(p) - 2) / 2
        fea_idxs = p[2: 2+n_fea]
        fea_values = p[2+n_fea:]
        assert len(fea_idxs) == len(fea_values)
        return Event(offset, code, fea_idxs, fea_values)
    
    

class Sample:
    MAX_EVENT_LEN = 1000
    def __init__(self, puid, offset, diag_list):
        self.puid = puid
        self.offset = offset
        self.diag_list = list(diag_list)

    def find_end(self, event_list):
        if event_list[-1] < self.offset:
            return len(event_list)
        l = 0
        r = len(event_list) - 1
        while l < r:
            mid = (l + r) / 2
            if event_list[mid] >= self.offset:
                r = mid
            else:
                l = mid + 1
        return l
    
    @staticmethod
    def get_padded_event_list(event_list):
        s_event = []
        s_time = []
        s_fea_idx = []
        s_fea_value = []
        padded_event = Event.padded_event()
        for idx in range(Sample.MAX_EVENT_LEN):
            if idx < len(event_list):
                event = event_list[idx]
            else:
                event = padded_event
            event.pad_feature()
            s_event.append(event.code)
            s_time.append(event.offset)
            s_fea_idx.append(event.fea_idxs)
            s_fea_value.append(event.fea_values)
        return s_event, s_time, s_fea_idx, s_fea_value


class Dataset:
    def __init__(self, outpath = None):
        self.event = []
        self.time = []
        self.feature_idx = []
        self.feature_value = []
        self.label = []
        self.ori_len = []
        self.offset = []
        self.puid = []
        self.outpath = outpath
    
    def add_sample(self, sample, event_list):
        ed = sample.find_end(event_list)
        st = max(0, ed - Sample.MAX_EVENT_LEN)
        if ed - st <= 10:
            return

        self.puid.append(sample.puid)
        self.offset.append(sample.offset)
        self.ori_len.append(ed)
        self.label.append(sample.diag_list)
        s_event, s_time, s_fea_idx, s_fea_value = Sample.get_padded_event_list(event_list[st: ed])
        self.feature_idx.append(s_fea_idx)
        self.feature_value.append(s_fea_value)
        self.event.append(s_event)
        self.time.append(s_time)
    
    def split(self, ratios):
        self.event = np.array(self.event, dtype = 'int32')
        self.time = np.array(self.time, dtype = 'float32')
        self.feature_idx = np.array(self.feature_idx, dtype = 'int32')
        self.feature_value = np.array(self.feature_value, dtype = 'float32')
        self.ori_len = np.array(self.ori_len, dtype = 'int32')
        self.offset = np.array(self.offset, dtype = 'int32')
        self.puid = np.array(self.puid, dtype = 'int32')
        self.label = np.array(self.label)
        

        size = len(self.event)
        shuffle_idxs = np.random.permutation(size)
        new_datasets = []
        st = 0
        for ratio in ratios:
            ed = int(math.ceil(size * ratio) + st)
            ed = min(size, ed)
            slice_idxs = shuffle_idxs[st:ed]
            d = Dataset()
            d.event = self.event[slice_idxs]
            d.feature_idx = self.feature_idx[slice_idxs]
            d.feature_value = self.feature_value[slice_idxs]
            d.time = self.time[slice_idxs]
            d.puid = self.puid[slice_idxs]
            d.offset = self.offset[slice_idxs]
            d.label = self.label[slice_idxs]
            d.ori_len = self.ori_len[slice_idxs]

            new_datasets.append(d)
            st = ed

        assert st == size
        return new_datasets
    
    def save(self, outpath):
        f = h5py.File(outpath, 'w')
        f['event'] = self.event
        f['feature_idx'] = self.feature_idx
        f['feature_value'] = self.feature_value
        f['time'] = self.time
        f['puid'] = self.puid
        f['offset'] = self.offset
        # f['label'] = self.label
        f['ori_len'] = self.ori_len
        f.close()


