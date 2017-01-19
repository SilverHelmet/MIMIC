from util import *
import os
import h5py
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

class Dataset:
    
    def __init__(self, file, seg = None):
        self.dataset_file = file
        self.seg_file = seg
    
    @staticmethod
    def create_datasets(files ,dataset_dir = None, segs = None, seg_dir = None):
        assert len(files) == 3
        if dataset_dir is not None:
            files = [os.path.join(dataset_dir ,file) for file in files]
        
        if seg_dir is not None:
            assert len(segs) == 3
            segs = [os.path.join(seg_dir, seg) for seg in segs]
        if segs is None:
            segs = [None] * 3
        datasets = []
        for file, seg in zip(files, segs):
            datasets.append(Dataset(file, seg))
        return tuple(datasets)

    def load(self, load_time = False):
        f = h5py.File(self.dataset_file, 'r')
        self.labels = f['label'][:]
        self.size = len(self.labels)
        self.events = f['event'][:]
        self.features = f['feature'][:]
        self.ids = f['sample_id'][:]
        self.merged_labels = merge_label(self.labels, self.ids)
        if load_time:
            self.times = f['time'][:]
        f.close()
        if self.seg_file is not None:
            f = h5py.File(self.seg_file)
            self.segs = f['segment'][:]
            f.close()

    def eval(self, model, setting):
        prediction = model.predict_generator(sample_generator(self, setting), val_samples = self.size)

        auROC = roc_auc_score(self.labels, prediction)

        merged_prediction = merge_prob(prediction, self.ids, max)
        merged_auROC = roc_auc_score(self.merged_labels, merged_prediction)

        fpr, tpr, thresholds = roc_curve(self.labels, prediction)
        auPRC = auc(fpr, tpr)
        fpr, tpr, thresholds = roc_curve(self.merged_labels, merged_prediction)
        merged_auPRC = auc(fpt, tpr)


        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] =0
        acc = accuracy_score(self.labels, prediction)

        merged_prediction[merged_prediction >= 0.5] = 1
        merged_prediction[merged_prediction < 0.5] = 0
        merged_acc = accuracy_score(self.merged_labels, merged_prediction)

        


        return (acc, auROC, auPRC, merged_acc, merged_auROC, merged_auPRC)

def print_eval(prefix, result):
    out = [prefix, *result]
    print "%s  acc = %f, auROC = %f, auPRC merged_acc = %f, merged_auc = %f"
    

def add_padding(l, max_len, padding_value = 0):
    assert max_len >= len(l)
    for i in range(max_len - len(l)):
        l.append(padding_value)
    return l

def gen_seged_event_seq(event_seq, split, max_seg_length):
    st = 0
    event_seqs = []
    for ed in split:
        if ed == 0:
            event_seqs.append([0] * max_seg_length)
        else:    
            event_seqs.append(add_padding(list(event_seq[st:ed]), max_seg_length))
            st = ed
    return event_seqs

def sequence2bow(event, event_dim):
    size = len(event)
    ret = np.zeros((size, event_dim))
    for i in range(size):
        seq = event[i]
        for event_idx in seq:
            if event_idx != 0:
                ret[i][event_idx] += 1
    return ret

def merge_event_by_seg(event_seq, split, event_dim, aggre_mode):
    st = 0
    event_seqs = []
    for ed in split:
        if ed == 0:
            event_seqs.append([])
        else:
            event_seqs.append(event_seq[st:ed])
            st = ed
    event_cnts = sequence2bow(event_seqs, event_dim)
    if aggre_mode == "one":
        event_cnts[event_cnts > 1] = 1
    elif aggre_mode == "ave":
        event_cnts = norm_to_prob(event_cnts)
    return event_cnts

def sample_generator(dataset, setting):
    labels = dataset.labels
    features = dataset.features
    events = dataset.events
    segs = dataset.segs
    nb_sample = len(labels)
    batch_size = setting['batch_size']
    disturbance = setting['disturbance']
    segment_flag = setting['segment_flag']
    max_seg_length = setting.get('max_seg_length', None)
    event_dim = setting['event_dim']
    rnn = setting['rnn']
    while  True:
        i = 0
        while i < nb_sample:
            st = i
            ed = min(i + batch_size, nb_sample)
            # print st, ed
            label = labels[st:ed]
            event = events[st:ed]
            if rnn == 'attlstm':
                # output shape (nb_sample, max_segs, max_seg_length)
                seged_event = []
                for j in range(st, ed):
                    split_seg = segs[j]
                    seged_event.append(gen_seged_event_seq(events[j], split_seg, max_seg_length))
                seged_event = np.array(seged_event)
                yield (seged_event, label)
            else:
                # output shape (nb_sample, max_segs, event_dim)
                aggre_mode = setting['aggregation']
                seg_event = []
                for j in range(st, ed):
                    split_seg = segs[j]
                    seg_event.append(merge_event_by_seg(events[j], split_seg, event_dim, aggre_mode))
                seg_event = np.array(seg_event)
                yield(seg_event, label)

            i += batch_size 
            if i >= nb_sample:
                i = 0

        


            
    
        
        
        
            

# def pair_to_vec(feature_pairs):
#     global feature_dim
#     i = 0
#     length = len(feature_pairs)
#     vec = np.zeros(feature_dim)
#     while i < length:
#         if feature_pairs[i+1] != 0:
#             index = int(feature_pairs[i])
#             value = feature_pairs[i+1]
#             vec[index] = value
#         i += 2
#     return vec


# def collect_feature(feature_matrix, st, ed):
#     global aggre_mode
#     length = len(feature_matrix[0])
#     vec = np.zeros(feature_dim)
#     for i in range(st, ed):
#         feature_pairs = feature_matrix[i]
#         idx = 0
#         while idx < length:
#             if feature_pairs[idx+1] != 0:
#                 index = int(feature_pairs[idx])
#                 value = feature_pairs[idx+1]
#                 vec[index] += value
#             idx += 2
#     dim = ed - st + 0.0
#     if aggre_mode == "ave":
#         vec /= dim
#     return vec

# def merge_fea_by_seg(feature_matrix, split):
#     global feature_dim
#     seg_fea_matrix = []
#     st = 0
#     for ed in split:
#         if ed == 0:
#             seg_fea_matrix.append(np.zeros(feature_dim))
#         else:
#             seg_fea_matrix.append(collect_feature(feature_matrix, st, ed))
#             st = ed
#     return seg_fea_matrix