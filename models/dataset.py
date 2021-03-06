from util import *
import os
import h5py
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, precision_recall_curve
from gcn.graph import build_time_graph_2, get_seg_time, time_funcs, get_seg_time, time_funcs
import glob
from collections import defaultdict

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

    def load(self, load_time = False, load_static_feature = False, load_transfer_time = False, event_set = None, setting = {}):
        self.load_time = load_time
        self.load_static_feature = load_static_feature
        f = h5py.File(self.dataset_file, 'r')
        self.feature_set = f.keys()
        if 'label' in self.feature_set:
            self.labels = f['label'][:]
        else:
            self.labels = f['labels'][:]
        self.size = len(self.labels)

        if setting['use_merged_event']:
            merged_f = h5py.File(self.dataset_file.replace('.h5', '_merged.h5'), 'r')
            self.events = merged_f['event'][:]
            merged_f.close()
        else:
            if 'event' in self.feature_set:
                self.events = f['event'][:]
            else:
                self.events = f['events'][:]
        if 'label_time' in f:
            self.label_times = f['label_time'][:]
        if 'predicting_time' in f:
            self.predicting_times = f['predicting_time'][:]
        
        if 'feature_idx' in self.feature_set:
            self.feature_idxs = f['feature_idx'][:]
            self.feature_values = f['feature_value'][:]
        else:
            nf_path = os.path.dirname(self.dataset_file) + "/normed_" + os.path.basename(self.dataset_file)
            if os.path.exists(nf_path):
                print 'load normed feature from [%s]' %nf_path
                nf = h5py.File(nf_path, 'r')
                features = nf['feature'][:]
                nf.close()
            else:
                features = f['feature']
            self.feature_idxs = features[:, :, [0,2,4]]
            self.feature_values = features[:, :, [1,3,5]]
        if 'sample_id' in f:
            self.ids = f['sample_id'][:]
            self.merged_labels = merge_label(self.labels, self.ids)
        if load_time:
            time_off = setting.get('time_off', 3.0)
            time_key = 'time'
            if time_key not in self.feature_set:
                time_key = 'times'
            if load_transfer_time and '|S' in str(f[time_key].dtype):
                time_base = setting.get('time_base', 'first_event')
                print 'load time diff / %.1f' %time_off
                print 'user time base: %s' %time_base
                if time_base == 'first_event':
                    time_path = self.dataset_file.replace('.h5', '_time.npy').replace('normed_', "")
                else:
                    time_path = self.dataset_file.replace('.h5', '_abstime.npy').replace('normed_', "")
                if not os.path.exists(time_path):
                    Print('%s tranfer time format' % self.dataset_file)
                    self.times = f[time_key][:]
                    self.trans_time(time_path, time_base)
                    self.times = self.times / time_off
                else:
                    self.times = np.load(time_path) / time_off
                print "time max", self.times.max()
            else:
                self.times = f[time_key][:] / time_off
                print "time max", self.times.max()
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

        # mask other event in: events, features, static_feature
        if event_set is not None:
            mask_index = []
            total = (self.events > 0).sum()
            for i, event_seq in enumerate(self.events):
                for j, eid in enumerate(event_seq):
                    if eid not in event_set:
                        mask_index.append((i,j))
            print('#masked event = %d/%.4f%%' %(len(mask_index), len(mask_index) * 100.0 / total) )
            self.events[mask_index] = 0
            
            if 'feature' in self.feature_set:
                self.features[mask_index] = 0
            if self.load_static_feature:
                self.static_features[mask_index] = 0



    def trans_time(self, outpath, time_base):
        offset_hours = np.ones_like(self.events) * -1.0
        n, m = self.times.shape
        for i in range(n):
            if time_base == 'first_event':
                st = parse_time(self.times[i][0])
                offset_hours[i][0] = 0.0
                for j in range(1, m):
                    time_s = self.times[i][j]
                    if len(time_s) > 0:
                        offset_hours[i][j] = (parse_time(time_s) - st).total_seconds()/3600.0
            else:
                for j in range(0, m):
                    time_s = self.times[i][j]
                    if len(time_s) > 0:
                        t = parse_time(time_s)
                        offset_hours[i][j] = t.hour + t.minute / 60.0 + t.second / 3600.0
        self.times = offset_hours
        np.save(outpath, offset_hours)


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

    def generate_model_input(self, setting):
        batch_size = setting['batch_size']
        setting['batch_size'] = self.size
        for x, y in sample_generator(self, setting):
            self.inputs = x
            self.i_labels = y
            assert len(self.i_labels) == self.size
            assert (self.i_labels != self.labels).sum() == 0
            break
        setting['batch_size'] = batch_size

    def get_inputs(self, idxs):
        if type(self.inputs) is list:
            ret_inputs = [m_input[idxs] for m_input in self.inputs]
        else:
            ret_inputs = self.inputs[idxs]
        return ret_inputs

    def get_labels(self, idxs):
        return self.i_labels[idxs]


    def eval(self, model, setting, event_set = None, info = None, verbose = False):
        if setting['sample_generator']:
            prediction = model.predict_generator(sample_generator(self, setting = setting, shuffle = False, event_set = event_set, info = info, verbose = verbose), val_samples = self.size)
        else:
            prediction = model.predict(self.inputs)
        calc_merged_score = 'sample_id' in self.feature_set

        auROC = roc_auc_score(self.labels, prediction)

        if calc_merged_score:
            merged_prediction = merge_prob(prediction, self.ids, max)
            merged_auROC = roc_auc_score(self.merged_labels, merged_prediction)

        precision, recall, thresholds = precision_recall_curve(self.labels, prediction)
        auPRC = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(1 - self.labels, 1 - prediction)
        reverse_auPRC = auc(recall, precision)


        if calc_merged_score:
            precision, recall, thresholds = precision_recall_curve(self.merged_labels, merged_prediction)
            merged_auPRC = auc(recall, precision)

            precision, recall, thresholds = precision_recall_curve(1 - self.merged_labels, 1 - merged_prediction)
            reverse_merged_auPRC = auc(recall, precision)


        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] =0
        acc = accuracy_score(self.labels, prediction)

        if calc_merged_score:
            merged_prediction[merged_prediction >= 0.5] = 1
            merged_prediction[merged_prediction < 0.5] = 0
            merged_acc = accuracy_score(self.merged_labels, merged_prediction)

        if calc_merged_score:
            return (acc, auROC, auPRC, reverse_auPRC, merged_acc, merged_auROC, merged_auPRC, reverse_merged_auPRC)
        else:
            return (acc, auROC, auPRC, reverse_auPRC)

def print_eval(prefix, result):
    out = [prefix]
    out.extend(result)
    if len(out) == 5:
        print "%s acc = %.4f, auROC = %.4f, auPRC = %.4f, reverse_auPRC = %.4f" %(tuple(out))
    else:
        print "%s acc = %.4f, auROC = %.4f, auPRC = %.4f, reverse_auPRC = %.4f, merged_acc = %.4f, merged_auROC = %.4f, merged_auPRC = %.4f, merged_reverse_auPRC = %.4f" %(tuple(out))
    

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

def make_gcn_seg_mat(segment, seg_mat):
    # seg_mat = np.zeros((n, m, k))
    seg_group_idx = 0
    st = 0
    for ed in segment:
        if ed == 0:
            continue
        for i in range(st, ed):
            seg_mat[seg_group_idx][i - st][i] = 1
        st = ed
        seg_group_idx += 1
    # return seg_mat



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

def fill_feature(feature_pairs, row, base):
    idx = 0
    length = len(feature_pairs)
    while idx < length:
        if feature_pairs[idx + 1] != 0:
            index = int(feature_pairs[idx])
            value = feature_pairs[idx + 1]
            row[base + idx] += value
        else:
            break
        idx += 2

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

def gen_gcn_feature_mat(feature_matrix, width, feature_dim, event_seq, gcn_feature_mat):
    pre_map = {}
    event_map = {}
    next_map = {}
    for idx in range(len(event_seq)):
        event = event_seq[idx]
        pre = event_map.get(event, -1)
        event_map[event] = idx
        pre_map[idx] = pre

    event_map = {}
    for idx in range(len(event_seq) - 1, -1, -1):
        event = event_seq[idx]
        next = event_map.get(event, -1)
        event_map[event] = idx
        next_map[idx] = next

    for idx, event in enumerate(event_seq):
        base = width * feature_dim
        row = gcn_feature_mat[idx]
        e_idx = idx
        for _ in range(width):
            e_idx = pre_map[e_idx]
            if e_idx == -1:
                break
            base -= feature_dim
            fill_feature(feature_matrix[e_idx], row, base)

        base = width * feature_dim
        fill_feature(feature_matrix[idx], row, base)

        e_idx = idx
        for _ in range(width):
            e_idx = next_map[e_idx]
            if e_idx == -1:
                break
            base += feature_dim
            fill_feature(feature_matrix[e_idx], row, base)
    # return gcn_feature_mat


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

def sample_generator_tf(dataset, setting):
    batch_size = setting['batch_size']
    size = dataset.size
    idxs = np.random.permutation(size)
    events = dataset.events
    labels = dataset.labels
    feature_idxs = dataset.feature_idxs
    feature_values = dataset.feature_values

    st = 0
    while True:
        ed = st + batch_size
        batch_idxs = idxs[st: ed]
        event = events[batch_idxs]
        feature_idx = feature_idxs[batch_idxs]
        feature_value = feature_values[batch_idxs]
        label = labels[batch_idxs]
        yield event, feature_idx, feature_value, label
        st = ed
        if st > size:
            st = 0

def sample_generator(dataset, setting, shuffle = False, train_index = None, event_set = None, info = None, verbose = False):
    # Print("start generate samples")
    if train_index is None:
        if shuffle:
            train_index = np.random.permutation(dataset.size)
        else:
            train_index = np.arange(dataset.size)
    labels = dataset.labels
    # features = dataset.features
    feature_idxs = dataset.feature_idxs
    feature_values = dataset.feature_values
    events = dataset.events
    try:
        segs = dataset.segs
    except:
        segs = np.zeros((len(labels, )))

    # nb_sample = len(labels)
    nb_sample = len(train_index)
    event_len = setting['event_len']
    batch_size = setting['batch_size']
    segment_flag = setting['segment_flag']
    max_segs = setting.get('max_segs', -1)
    max_seg_length = setting.get('max_seg_length', -1)
    event_dim = setting['event_dim']
    rnn = setting['rnn']
    feature_dim = setting.get('feature_dim', None)
    gcn = setting['GCN']
    gcn_seg = setting['GCN_Seg']
    gcn_numeric_feature = setting['gcn_numeric_feature']
    gcn_numeric_width = setting.get('gcn_numeric_width', 1)
    gcn_time_width = setting['gcn_time_width']
    post_model = setting['post_model']    
    gcn_time_func = time_funcs[setting.get('gcn_time_func', 'one')]
    post_gcn_time_func = time_funcs[setting.get('post_gcn_time_func', 'invert')]
    time_feature_flag = setting['time_feature']
    time_feature_type = setting['time_feature_type']
    if setting['load_time']:
        times = dataset.times


    static_features = dataset.static_features
    use_static_feature = setting['static_feature']
    static_feature_size = setting.get('static_feature_size', 0)
    if time_feature_flag or setting['eventxtime']:
        mul = setting.get('time_mul', 3.0)
        time_hours = (times * mul).astype(int)

    if info is None:
        info = defaultdict(int)
    
    while  True:
        i = 0
        while i < nb_sample:
            if verbose and ((i / batch_size) % 50 == 0):
                Print('generate at %d/%d' %(i, nb_sample))

            
            st = i
            ed = min(i + batch_size, nb_sample)
            batch_indices = train_index[st:ed]

            label = labels[batch_indices]
            event = events[batch_indices]
            seg = segs[batch_indices]

            # feature = features[batch_indices]
            # if event_set is not None:
            #     info['total'] += (event > 0).sum()
            #     for i1, event_seq in enumerate(event):
            #         for i2, eid in enumerate(event_seq):
            #             if eid not in event_set and eid > 0:
            #                 event[i1,i2] = 0
            #                 feature[i1,i2] = 0
            #                 info['mask'] += 1

            seged_event = event
            if event_set is not None:
                info['total'] += (seged_event > 0).sum()
                for i1, event_seq in enumerate(seged_event):
                    for i2, eid in enumerate(event_seq):
                        if eid > 0 and eid not in event_set:
                            event[i1, i2] = 0
                            info['mask'] += 1
            if setting['eventxtime']:
                mask_idx = seged_event == 0
                time_hour = time_hours[batch_indices]
                time_hour[mask_idx] = 0
                seged_event = seged_event * 24 + time_hour
            if gcn:
                time = times[batch_indices]
                As = np.zeros((ed - st, event_len, event_len))
                for idx, sample_time in enumerate(time):
                    build_time_graph_2(sample_time, gcn_time_width, As[idx])
    

            if use_static_feature:
                static_feature_mat = []
                for j in range(ed - st):
                    static_feature = static_features[batch_indices[j]]
                    static_feature_mat.append(parse_sparse_static_feature(static_feature, static_feature_size))
                static_feature_mat = np.array(static_feature_mat)

            if gcn_numeric_feature:
                if setting['numeric_feature_type'] == "HELSTM":
                    feature_idx = feature_idxs[batch_indices]
                    feature_value = feature_values[batch_indices]
                else:
                    assert False
                    feature_size = feature_dim * (gcn_numeric_width * 2 + 1)
                    gcn_num_feature_matries = np.zeros((ed - st, event_len, feature_size))
                    for j in range(ed - st):
                        gen_gcn_feature_mat(feature[j], gcn_numeric_width, feature_dim, event[j], gcn_num_feature_matries[j])

            if time_feature_flag:
                time_hour = time_hours[batch_indices]

            if gcn_seg:
                gcn_seg_mat = np.zeros((ed - st, max_segs, max_seg_length, event_len))
                for idx, batch_seg in enumerate(seg):
                    make_gcn_seg_mat(batch_seg, gcn_seg_mat[idx])

            if post_model == 'gcn':
                gcn_seg_edge_mat = np.zeros((ed -st, max_segs, max_segs))
                time = times[batch_indices]
                for j in range(ed - st):
                    split_seg = seg[j] 
                    sample_time = time[j]
                    seg_sample_time = get_seg_time(sample_time, split_seg)
                    build_time_graph_2(seg_sample_time, 9999999999, gcn_seg_edge_mat[j], post_gcn_time_func)
            elif post_model == 'LSTM':
                pass
            elif post_model == 'HELSTM':
                event_time = np.expand_dims(times[batch_indices], -1)
            else:
                assert False



            inputs = [seged_event]
            if gcn_numeric_feature:
                if setting['numeric_feature_type'] == "HELSTM":
                    inputs.append(feature_idx)
                    inputs.append(feature_value)
                else:
                    inputs.append(gcn_num_feature_matries)
            if time_feature_flag:
                inputs.append(time_hour)
            if gcn:
                inputs.append(As)
            if gcn_seg:
                inputs.append(gcn_seg_mat)
            if post_model == 'gcn':
                inputs.append(gcn_seg_edge_mat)
            elif post_model == 'HELSTM':
                inputs.append(event_time)            
            if use_static_feature:
                inputs.append(static_feature_mat)
            if len(inputs) == 1:
                inputs = inputs[0]
            yield (inputs, label)
            i += batch_size 
            if i >= nb_sample:
                i = 0
                # break
        if i == 0:
            break


if __name__ == "__main__":
    sample_dir = 'death_exper/sample'
    # dataset = Dataset('death_exper/death_valid_1000.h5', seg = 'death_exper/segs/death_valid_1000_segmode=fixLength_maxchunk=32_length=32.h5')
    # dataset.load(load_static_feature = True, load_time = True)
    # s_dataset = dataset.sample()
    # if not os.path.exists(sample_dir):
    #     os.mkdir(sample_dir)
    # s_dataset.save(sample_dir)

    
    # for filepath in glob.glob(death_exper_dir + "/death*_1000.h5"):
    #     s_dataset = Dataset(filepath)
    #     s_dataset.load(True, False, True)
    s_dataset = Dataset('death_exper/sample/samples.h5', 'death_exper/sample/samples_seg.h5')
    s_dataset.load(True, True, True)
    s_dataset.print_shape()
    # s_dataset.trans_time()
    print s_dataset.times[1][:10]
    As = np.zeros((2, 1000, 1000))
    # build_time_graph(s_dataset.times[1], 1.0, As[0])
    print s_dataset.times[1][-10:]
    split_seg = s_dataset.segs[1]
    print split_seg
    seg_time = get_seg_time(s_dataset.times[1], split_seg)
    
    x = np.zeros((len(split_seg), len(split_seg)))
    print seg_time
    build_time_graph_2(seg_time, 99999, x, time_func = time_funcs.get('invert'))
    print x[-9][:]