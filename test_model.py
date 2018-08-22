from train_rnn import define_simple_seg_rnn, load_argv
from keras.models import Model, load_model
from models.dataset import Dataset, print_eval, sample_generator
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
from scripts.stat_value_dis import FValueStat
import json


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

def calc_event_effect(data, prob_path, outpath):
    prob_mat = np.load(prob_path)
    hours = np.asarray(data.times * 3, dtype = 'int16')
    fv_dict = {}
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
            fv_dict[e].add(hour, 0, diff)
    outf = file(outpath, 'w')
    for e in fv_dict:
        stat = fv_dict[e].to_json()
        outf.write('%d\t%s\n' %(e, json.dumps(stat)))
    outf.close()

if __name__ == "__main__":
    args = 'x settings/catAtt_lstm.txt settings/helstm.txt settings/time_feature/time_feature_sum.txt settings/period/period_v19.txt @time_gate_type=ones|model_out=RNNmodels/death_helstm.model'.split(' ')
    setting = load_argv(args)
    setting['event_dim'] = 3418
    model_path = sys.argv[1]
    model = define_simple_seg_rnn(setting, True)
    model.load_weights(model_path, by_name=True)

    # data = Dataset('death_exper/sample/samples.h5')
    data = Dataset('death_exper/death_test_1000.h5')
    data.load(True, False, True, None, setting)
    outpath = 'result/death_test_probs.npy'
    if not os.path.exists(outpath):
        print_probs(model, data, setting, outpath)
    
    effect_outpath = 'result/death_event_effect.txt'
    calc_event_effect(data, outpath, effect_outpath)

