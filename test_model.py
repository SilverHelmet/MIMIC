from train_rnn import define_simple_seg_rnn, load_argv
from keras.models import Model, load_model
from models.models import np_sigmoid
from models.dataset import Dataset, print_eval, sample_generator
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
from scripts.stat_value_dis import FValueStat
from event_filter.calc_event_score import get_weights
import json
from util import Print

def print_probs(model, data, setting, outpath):
    size = data.size
    results = []
    pred_size = 0
    for inputs, labels in sample_generator(data, setting):
        probs_list = model.predict(inputs)
        results.append(probs_list)
        pred_size += len(labels)
        if pred_size >= size:
            break
        print pred_size, size
    prob_matrix = np.concatenate(results, axis = 0)
    prob_matrix = np.squeeze(prob_matrix)
    np.save(outpath, prob_matrix)

def calc_event_effect(data, prob_path, fv_dict, label):
    prob_mat = np.load(prob_path)
    hours = np.asarray(data.times * 3, dtype = 'int16')
    for i, event_seq in tqdm(enumerate(data.events), total = data.size):
        row = prob_mat[i]
        hour_seq = hours[i]
        for j, e in enumerate(event_seq):
            if j < 10 or e == 0:
                continue
            if not e in fv_dict:
                fv_dict[e] = FValueStat(e, 0)
            diff = prob_mat[i][j] - prob_mat[i][j-1]
            hour = hour_seq[j]
            fv_dict[e].add(hour, label, diff)

def get_death_event_effect(setting):
    # data = Dataset('death_exper/sample/samples.h5')
    data = Dataset('death_exper/death_test_1000.h5')
    models = ['death_t23_notime.model.round8', 'death_t23.model.round5']
    labels = [0, 1]
    models = [os.path.join('RNNmodels', model) for model in models]
    data.load(True, False, True, None, setting)
    fv_dict = {}
    for model_path, label in zip(models, labels):
        model = define_simple_seg_rnn(setting, True)
        model.load_weights(model_path, by_name=True)
        prob_outpath = 'result/death_test_probs_{}.npy'.format(os.path.basename(model_path))
        if not os.path.exists(prob_outpath):
            print_probs(model, data, setting, prob_outpath)
        calc_event_effect(data, prob_outpath, fv_dict, label)

    effect_outpath = 'result/death_event_time_effect.json'
    outf = file(effect_outpath, 'w')
    for e in fv_dict:
        stat = fv_dict[e].to_json()
        outf.write('%d\t%s\n' %(e, json.dumps(stat)))
    outf.close()

def calc_event_attn(weights_map):
    embedding_w = weights_map['embedding_W']
    hid_w = weights_map['helstm_event_hid_w']
    hid_b = weights_map['helstm_event_hid_b']
    hidden = np.tanh(np.dot(embedding_w  , hid_w) + hid_b)
    out_w = weights_map['helstm_event_out_w']
    out_b = weights_map['helstm_event_out_b']
    event_attn = np_sigmoid(np.dot(hidden, out_w) + out_b)
    return event_attn

def get_death_event_view(setting, model):
    event_out_path = 'result/death_event_view.npy'
    if os.path.exists(event_out_path):
        event_attn = np.load(event_out_path)
        return event_attn
    Print('calc event view')
    keys = ['helstm_event_hid_w', 'helstm_event_hid_b', 'helstm_event_out_w', 'helstm_event_out_b', 'embedding_W']
    weights_map = get_weights(model, keys)
    event_attn = calc_event_attn(weights_map)
    np.save(event_out_path, event_attn)
    return event_attn

def get_death_time_view(setting, model, data):
    outpath = 'result/death_{}_time_view.npy'.format(data)
    if os.path.exists(outpath):
        return np.load(outpath)
    Print('calc time view')
    # data = Dataset('death_exper/death_test_1000.h5')
    data = Dataset('death_exper/death_train_1000.h5')
    data.load(True, False, True, None, setting)

    events = data.events
    time = data.times

    keys = ['helstm_period', 'helstm_shift', 'helstm_onend']
    weights_map = get_weights(model, keys)

    time = np.expand_dims(time, -1)
    time = np.repeat(time, 8, -1) 
    shifted_time = time + weights_map['helstm_shift']
    in_period_time = shifted_time % weights_map['helstm_period']
    on_mid = weights_map['helstm_period'] * 0.5 * weights_map['helstm_onend']
    on_end = weights_map['helstm_period'] * weights_map['helstm_onend']
    in_up_phase = in_period_time < on_mid
    in_down_phase = np.asarray(in_period_time <= on_end, dtype = 'int8') - in_up_phase
    other_phase = 1 - (in_up_phase + in_down_phase)
    up_attn = in_period_time / on_mid
    down_attn = (on_end - in_period_time) / on_mid
    other_attn = 1e-3 * in_period_time / weights_map['helstm_period']
    time_attn = in_up_phase * up_attn + in_down_phase * down_attn + other_phase * other_attn

    time_attn_list = []
    for e in range(3418):
        idxs = events == e
        attn = time_attn[idxs]
        if attn.size == 0:
            attn = [0] * 8
        else:
            attn = time_attn[idxs].mean(0)
        time_attn_list.append(attn)
    time_attn_list = np.array(time_attn_list)
    np.save(outpath, time_attn_list)
    return time_attn_list

def topk_event(attn, k):
    view_size = attn.shape[1]
    view_topk = []
    for view_idx in range(view_size):
        view = attn[:, view_idx]
        sorted_idxs = sorted(range(len(view)), key = lambda x: view[x], reverse = True)
        topk = sorted_idxs[:k]
        view_topk.append(topk)
    return view_topk



def get_death_view(ssetting, data):
    assert data in ['test', 'train']
    outpath = 'result/death_{}_view.npy'.format(data)
    # if os.path.exists(outpath):
    #     return np.load(outpath)

    model = None
    models = ['death_t23.model.round5']
    models = [os.path.join('RNNmodels', model) for model in models]

    model = define_simple_seg_rnn(setting, True)
    model.load_weights(models[0], by_name=True)

    event_attn = get_death_event_view(setting, model)
    time_attn = get_death_time_view(setting, model, data)

    attn = event_attn * time_attn
    np.save(outpath, attn)

    event_view_topk = topk_event(event_attn, 10)
    view_topk = topk_event(attn, 10)
    for idx in range(8):
        event_topk = event_view_topk[idx]
        topk = view_topk[idx]
        print 'event view: {}'.format(event_topk)
        print 'view: {}'.format(topk)
        print ""

def sorted_idx_by_view(view):
    scores = view.mean(1)
    sorted_idx = sorted(range(len(scores)), key = lambda x:scores[x],reverse = True)
    return sorted_idx

def test_event_filter():
    view = np.load('result/death_view.npy')
    sorted_idx = sorted_idx_by_view(view)
    print view.mean(1)[sorted_idx][:100]




if __name__ == "__main__":
    args = 'x settings/catAtt_lstm.txt settings/helstm.txt settings/time_feature/time_feature_sum.txt settings/period/period_v14.txt @num_gate_head=8|model_out=RNNmodels/death_t23.model'.split(' ')
    setting = load_argv(args)
    setting['event_dim'] = 3418

    mode = sys.argv[1]
    if mode == 'event_effect':
        get_death_event_effect(setting)
    elif mode == 'view':
        get_death_view(setting, 'train')
        get_death_view(setting, 'test')
    elif mode == 'event_filter':
        test_event_filter()
