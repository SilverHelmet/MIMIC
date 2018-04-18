from util import *
import os
import h5py
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, precision_recall_curve
from gcn.graph import build_time_graph

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

    def load(self, load_time = False, load_static_feature = False):
        self.load_time = load_time
        self.load_static_feature = load_static_feature
        f = h5py.File(self.dataset_file, 'r')
        self.labels = f['label'][:]
        self.size = len(self.labels)
        self.events = f['event'][:]
        self.label_times = f['label_time'][:]
        self.predicting_times = f['predicting_time'][:]
        if 'feature' in f:
            self.features = f['feature'][:]
        else:
            self.features = np.zeros((1,1))
        self.ids = f['sample_id'][:]
        self.merged_labels = merge_label(self.labels, self.ids)
        if load_time:
            self.times = f['time'][:]
        f.close()
        if self.seg_file is not None:
            f = h5py.File(self.seg_file)
            self.segs = f['segment'][:]
            self.max_segs = f['max_segs'].value
            self.max_seg_length = f['max_seg_length'].value
            f.close()
        
        # load static feature
        if load_static_feature:
            path = self.dataset_file.replace('_merged_', '_')
            base_dir = os.path.dirname(path)
            static_filename = os.path.basename(self.dataset_file).replace('.h5', '_static.npy')
            static_feature_path = os.path.join(base_dir, static_filename)

            Print('load static feature from [%s]' %static_feature_path)
            self.static_features = np.load(static_feature_path)
        else:
            self.static_features = np.zeros((1,1))

    def trans_time(self):
        offset_hours = np.ones_like(self.events) * -1.0
        n, m = self.times.shape
        for i in range(n):
            st = parse_time(self.times[i][0])
            offset_hours[i][0] = 0.0
            for j in range(1, m):
                time_s = self.times[i][j]
                if len(time_s) > 0:
                    offset_hours[i][j] = (parse_time(time_s) - st).total_seconds()/3600.0
        self.times = offset_hours


    def sample(self, sample_list = None):
        if sample_list is None:
            sample_list = np.arange(10000, 11000, 1)
        s_dataset = Dataset(file = None)
        s_dataset.labels = self.labels[sample_list]
        s_dataset.events = self.events[sample_list]
        s_dataset.features = self.features[sample_list]
        s_dataset.ids = self.ids[sample_list]
        s_dataset.label_times = self.label_times[sample_list]
        s_dataset.predicting_times = self.predicting_times[sample_list]
        s_dataset.segs = self.segs[sample_list]
        s_dataset.max_seg_length = self.max_seg_length
        s_dataset.max_segs = self.max_segs
        s_dataset.load_time = self.load_time
        if self.load_time:
            s_dataset.times = self.times[sample_list]

        s_dataset.load_static_feature = self.load_static_feature
        s_dataset.static_features = self.static_features[sample_list]
        return s_dataset

    def label_time_at(self, i):
        return self.label_times[i]
    
    def predicting_time_at(self, i):
        return self.predicting_times[i]

    def label_at(self, i):
        return self.labels[i]

    def event_mat(self, row):
        return np.array(gen_seged_event_seq(self.events[row], self.segs[row], self.max_seg_length))

    def feature_mat(self, row, feature_dim = 649):
        return np.array(gen_seged_feature_seq(self.features[row], self.segs[row], self.max_seg_length, feature_dim))

    def save(self, sample_dir):
        outf = h5py.File(os.path.join(sample_dir, 'samples.h5'), 'w')        
        outf['label'] = self.labels
        outf['event'] = self.events
        outf['feature'] = self.features
        outf['sample_id'] = self.ids
        outf['time'] = self.times
        outf['label_time'] = self.label_times
        outf['predicting_time'] = self.predicting_times
        outf.close()

        outf = h5py.File(os.path.join(sample_dir, 'samples_seg.h5'), "w")
        outf['segment'] = self.segs
        outf['max_segs'] = self.max_segs
        outf['max_seg_length'] = self.max_seg_length
        outf.close()

        if self.load_static_feature:
            np.save(os.path.join(sample_dir, 'samples_static'), self.static_features)


    def print_shape(self):
        for k, v in self.__dict__.items():
            if hasattr(v, 'shape'):
                print "%s's shape is %s" %(k, str(v.shape))
            else:
                print "%s = %s" %(k, v)

    def eval(self, model, setting):
        prediction = model.predict_generator(sample_generator(self, setting), val_samples = self.size)

        auROC = roc_auc_score(self.labels, prediction)

        merged_prediction = merge_prob(prediction, self.ids, max)
        merged_auROC = roc_auc_score(self.merged_labels, merged_prediction)

        precision, recall, thresholds = precision_recall_curve(self.labels, prediction)
        auPRC = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(self.merged_labels, merged_prediction)
        merged_auPRC = auc(recall, precision)


        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] =0
        acc = accuracy_score(self.labels, prediction)

        merged_prediction[merged_prediction >= 0.5] = 1
        merged_prediction[merged_prediction < 0.5] = 0
        merged_acc = accuracy_score(self.merged_labels, merged_prediction)

        return (acc, auROC, auPRC, merged_acc, merged_auROC, merged_auPRC)

def print_eval(prefix, result):
    out = [prefix]
    out.extend(result)
    print "%s acc = %.4f, auROC = %.4f, auPRC =%.4f, merged_acc = %.4f, merged_auROC = %.4f, merged_auPRC = %.4f" %(tuple(out))
    

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


def collect_feature(feature_matrix, st, ed, feature_dim):
    length = len(feature_matrix[0])
    vec = np.zeros(feature_dim)
    for i in range(st, ed):
        feature_pairs = feature_matrix[i]
        idx = 0
        while idx < length:
            if feature_pairs[idx+1] != 0:
                index = int(feature_pairs[idx])
                value = feature_pairs[idx+1]
                vec[index] += value
            idx += 2
    return vec

def merge_fea_by_seg(feature_matrix, split, feature_dim):
    '''
        return shape: (max_segs, feature_dim)
    '''

    seg_fea_matrix = []
    st = 0
    for ed in split:
        if ed == 0:
            seg_fea_matrix.append(np.zeros(feature_dim))
        else:
            seg_fea_matrix.append(collect_feature(feature_matrix, st, ed, feature_dim))
            st = ed
    return seg_fea_matrix

def gen_feature(feature_matrix, st, ed, max_seg_length, feature_dim):
    length = len(feature_matrix[0])
    vec = np.zeros((max_seg_length, feature_dim))
    for j in range(st, ed):
        feature_pairs = feature_matrix[j]
        i = j - st
        idx = 0
        while idx < length:
            if feature_pairs[idx+1] != 0:
                index = int(feature_pairs[idx])
                value = feature_pairs[idx+1]
                vec[i][index] += value
            idx += 2
    return vec


def gen_seged_feature_seq(feature_matrix, split, max_seg_length, feature_dim):
    '''
        return shape: (max_segs, max_seg_length, feature_dim)
    '''
    seg_fea_matrix = []
    st = 0
    for ed in split:
        if ed == 0:
            seg_fea_matrix.append(np.zeros((max_seg_length, feature_dim)))
        else:
            seg_fea_matrix.append(gen_feature(feature_matrix, st, ed, max_seg_length, feature_dim))
            st = ed
    return seg_fea_matrix

def parse_sparse_static_feature(static_feature, size):
    vec = np.zeros((size, ))
    i = 0
    length = len(static_feature)
    while i < length:
        idx = static_feature[i]
        value = static_feature[i+1]
        vec[idx] += value
        i += 2
    return vec


def sample_generator(dataset, setting, shuffle = False):
    Print("start generate samples")
    labels = dataset.labels
    features = dataset.features
    events = dataset.events
    segs = dataset.segs
    nb_sample = len(labels)
    batch_size = setting['batch_size']
    disturbance = setting['disturbance']
    segment_flag = setting['segment_flag']
    max_seg_length = setting['max_seg_length']
    event_dim = setting['event_dim']
    rnn = setting['rnn']
    feature_dim = setting.get('feature_dim', None)
    gcn = setting['GCN']
    if gcn:
        times = dataset.times


    static_features = dataset.static_features
    use_static_feature = setting['static_feature']
    static_feature_size = setting.get('static_feature_size', 0)
    if shuffle:
        indices = np.random.permutation(nb_sample)
    else:
        indices = np.arange(nb_sample)
    while  True:
        i = 0
        while i < nb_sample:
            st = i
            ed = min(i + batch_size, nb_sample)
            # print st, ed
            batch_indices = indices[st:ed]

            label = labels[batch_indices]
            event = events[batch_indices]
            seg = segs[batch_indices]

            if gcn:
                seged_event = event
                time = times[batch_indices]
                As = []
                for sample_time in time:
                    As.append(build_time_graph(sample_time, 0.5))
                As = np.array(As)

            elif rnn == 'attlstm' or rnn == 'attgru':
                # output shape (nb_sample, max_segs, max_seg_length)
                seged_event = []
                for j in range(ed - st):
                    split_seg = seg[j]
                    seged_event.append(gen_seged_event_seq(event[j], split_seg, max_seg_length))                    
                seged_event = np.array(seged_event)

                # output shape (nb_sample, max_segs, max_seg_length, feature_dim)
                if disturbance:
                    seg_feature_matrixes = []
                    for j in range(ed - st):
                        split_seg = seg[j]
                        seg_feature_matrixes.append(gen_seged_feature_seq(features[batch_indices[j]], split_seg, max_seg_length, feature_dim))
                    seg_feature_matrixes = np.array(seg_feature_matrixes)
            else:
                aggre_mode = setting['aggregation']
                # output shape (nb_sample, max_segs, event_dim)
                seged_event = []
                for j in range(ed - st):
                    split_seg = seg[j]
                    seged_event.append(merge_event_by_seg(event[j], split_seg, event_dim, aggre_mode))
                seged_event = np.array(seged_event)

                # output shape (nb_sample, max_segs, feature_dim)
                if disturbance:
                    seg_feature_matrixes = []
                    for j in range(ed - st):
                        split_seg = seg[j]
                        seg_feature_matrixes.append(merge_fea_by_seg(features[batch_indices[j]], split_seg, feature_dim))
                    seg_feature_matrixes = np.array(seg_feature_matrixes)

            if use_static_feature:
                static_feature_mat = []
                for j in range(ed - st):
                    static_feature = static_features[batch_indices[j]]
                    static_feature_mat.append(parse_sparse_static_feature(static_feature, static_feature_size))
                static_feature_mat = np.array(static_feature_mat)

            inputs = [seged_event]
            if disturbance:
                inputs.append(seg_feature_matrixes)
            if gcn:
                inputs.append(As)
            if use_static_feature:
                inputs.append(static_feature_mat)

            if len(inputs) == 1:
                inputs = inputs[0]

            yield (inputs, label)
            i += batch_size 
            if i >= nb_sample:
                i = 0


if __name__ == "__main__":
    sample_dir = 'death_exper/sample'
    # dataset = Dataset('death_exper/death_valid_1000.h5', seg = 'death_exper/segs/death_valid_1000_segmode=fixLength_maxchunk=32_length=32.h5')
    # dataset.load(load_static_feature = True, load_time = True)
    # s_dataset = dataset.sample()
    # if not os.path.exists(sample_dir):
    #     os.mkdir(sample_dir)
    # s_dataset.save(sample_dir)

    
    s_dataset = Dataset('death_exper/sample/samples.h5', 'death_exper/sample/samples_seg.h5')
    s_dataset.load(True, True)
    s_dataset.print_shape()
    s_dataset.trans_time()
    print s_dataset.times[1][:10]
    print s_dataset.times[1][-10:]
    A = build_time_graph(s_dataset.times[1], 0.5)
    print A[4][:10]