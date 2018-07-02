from util import model_dir, death_exper_dir, Print, result_dir, add_to_cnt_dict
import os
import sys
from keras.models import load_model,Model
from models.models import get_custom_objects, np_softmax
from models.dataset import Dataset, print_eval
from tqdm import tqdm
import numpy as np
import json


class FeatureStat:
    def __init__(self, event2feature):
        e2f = load_event2feature()
        self.features_cnt = {}
        self.features_sum = {}
        for e in e2f:
            for f in e2f[e]:
                self.features_cnt['%d#%d' %(e, f)] = 0
                self.features_sum['%d#%d' %(e, f)] = 0


    def stat(self, event_mat, feature_mat):
        Print("start stat feature")
        for i in tqdm(range(len(event_mat)), total = len(event_mat)):
            event_seq = event_mat[i]
            feature_pairs = feature_mat[i]
            self.add(event_seq, feature_pairs)
        return self.calc_average()

    def add(self, event_seq, feature_pairs):
        for sid in range(len(event_seq)):
            eid = event_seq[sid]
            if eid == 0:
                continue
            pairs = feature_pairs[sid]

            i = 0
            length = len(pairs)
            while i < length:
                idx = int(pairs[i])
                value = pairs[i+1]
                if idx != 0:
                    key = "%d#%d" %(eid, idx)
                    add_to_cnt_dict(self.features_cnt, key, 1)
                    add_to_cnt_dict(self.features_sum, key, value)
                i += 2

    def calc_average(self):
        averages = {}
        for key in self.features_cnt:
            cnt = self.features_cnt[key]
            fsum = self.features_sum[key]
            if cnt == 0:
                cnt = 1.0
            averages[key] = fsum / cnt
        return averages

def get_weights(model, keys):
    keys = set(keys)
    substrs = set()
    for key in keys:
        if key.endswith('*'):
            substrs.add(key[:-1])
    weights_map = {}
    for weight in model.weights:
        if weight.name in keys:
            weights_map[weight.name] = weight.get_value()
            continue
        for substr in substrs:
            if substr in weight.name:
                weights_map[weight.name] = weight.get_value()
    return weights_map

def load_sample():
    model_path = os.path.join(model_dir, 'sample.model')
    exper_dir = os.path.join(death_exper_dir, 'sample')
    dataset_path = os.path.join(exper_dir, 'samples.h5')
    seg_path = os.path.join(exper_dir,  'samples_seg.h5')

    dataset = Dataset(dataset_path, seg_path)
    dataset.load(False, False, False)

    weights_name = ['embedding_W', 'numeric feature embedding_W', 
        'numeric feature embedding_b', 'kernel*']
    model = load_model(model_path, get_custom_objects())
    weights_map = get_weights(model, weights_name)

    return dataset, model, weights_map

def load_death_fixlength16():
    model_path = os.path.join(model_dir, 'death_fixLength16_fea_catAtt_gcn_mode-1_12X8.model2')
    exper_dir = death_exper_dir
    dataset_path = os.path.join(exper_dir, 'death_train_1000.h5')
    seg_path = os.path.join(exper_dir,  'segs/death_train_1000_segmode=fixLength_maxchunk=63_length=16.h5')

    dataset = Dataset(dataset_path, seg_path)
    dataset.load(False, False, False)

    weights_name = ['embedding_W', 'numeric feature embedding_W', 
        'numeric feature embedding_b', 'kernel*']
    model = load_model(model_path, get_custom_objects())
    weights_map = get_weights(model, weights_name)

    return dataset, model, weights_map

def load_event2feature():
    path = os.path.join(result_dir, 'feature_info.tsv')
    e2f = {}
    fs = set()
    for line in file(path):
        event = json.loads(line)
        features = event['feature']
        eid = event['event_id']
        e2f[eid] = []
        for feature in features:
            idx = int(feature.split(" at ")[1])
            e2f[eid].append(idx)
            fs.add(idx)
    return e2f

def calc_event_score_by_rank(event_emd, limit):
    def value_argsort(seq, values):
        return sorted(seq, key = lambda x: values[x], reverse = True)
    
    size = event_emd.shape[0]
    scores = [0.0] * size 
    for col_idx in range(event_emd.shape[1]):
        seq = range(size)
        values = event_emd[:, col_idx]
        sorted_eid = value_argsort(seq, values)
        for rank, eid in enumerate(sorted_eid, start = 0):
            if rank >= limit:
                continue
            scores[eid] += 1.0  - rank / limit
    return scores




def calc_event_score(weights_map, features_ave, feature_width, out_dir):
    event_size = len(weights_map['embedding_W'])
    feature_size = len(weights_map['numeric feature embedding_W']) / feature_width
    event2fea = load_event2feature()
    event = weights_map['embedding_W']
    event_feature = np.zeros((event_size, feature_size * feature_width))
    Print('calc event feature embedding')
    for eid in range(2, event_size):
        base = 0
        for i in range(feature_width):
            for fidx in event2fea[eid]:
                event_feature[eid][base + fidx] = features_ave["%d#%d" %(eid, fidx)]
            base += feature_size

    feature_emd = np.matmul(event_feature, weights_map['numeric feature embedding_W']) + weights_map['numeric feature embedding_b']
    feature_emd = np.tanh(feature_emd)
    Print("calc event feature embedding over")

    Print("calc event embedding")
    event_ori_emd = np.concatenate((weights_map['embedding_W'], feature_emd), axis = -1)
    
    # event_kernel_emd = np.matmul(event_ori_emd, weights_map['kernel_0_0'])
    # np.save(os.path.join(out_dir, 'event_kernel_embedding'), event_kernel_emd)
    Print("calc event embedding over")

    Print("calc event score")
    outputs = []
    head = 0
    while "kernel_%d_0" %head in weights_map:
        kernel = weights_map['kernel_%d_0' %head]
        event_kernel_emd = np.matmul(event_ori_emd, kernel)
        event_attn_score = np.matmul(event_kernel_emd, weights_map['att_kernel_%d_1' %head])
        event_attn = np_softmax(event_attn_score)
        np.save(os.path.join(out_dir, 'event_attn_score_%d' %head), event_attn)
        event_attn_embedding = event_kernel_emd * event_attn
        outputs.append(event_attn_embedding)
        head += 1
    

    limit = 400
    event_attn_embedding = np.concatenate(outputs)
    event_scores = calc_event_score_by_rank(event_attn_embedding, limit)
    cnt = 0
    for score in event_scores:
        cnt += score > 0
    print "#not zero event: %d" %cnt 


    outf = file(os.path.join(out_dir, 'event_scores_%d.txt' %limit), 'w')
    for score in event_scores:
        outf.write("%.4f\n" %(score))
    outf.close()
    Print("calc event score over")




if __name__ == "__main__":
    # dataset, model, weights_map = load_sample()
    dataset, model, weights_map = load_death_fixlength16()

    feature_width = 3
    feature_size = weights_map['numeric feature embedding_W'].shape[0] / feature_width
    feature_stat = FeatureStat(feature_size)
    features_ave = feature_stat.stat(dataset.events, dataset.features)
    out_dir = os.path.join(model_dir, 'event_filter')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    calc_event_score(weights_map, features_ave, feature_width, out_dir)




