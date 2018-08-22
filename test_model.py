from train_rnn import define_simple_seg_rnn, load_argv
from keras.models import Model, load_model
from models.dataset import Dataset, print_eval, sample_generator
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os


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
    prob_mat = np.load(outpath)
    e2sum = defaultdict(.0)
    e2cnt = defaultdict(.0)
    for i, event_seq in tqdm(enumerate(data.events), total = data.size):
        row = prob_mat[i]
        for j, e in enumerate(event_seq):
            if j < 10 or e == 0:
                continue
            diff = prob_mat[i][j] - prob_mat[i][j-1]
            e2sum[e] += diff
            e2cnt[e] += 1.0
    
    e2effect = {e:e2sum[e] / e2cnt[e] for e in e2cnt}
    outf = file(outpath, 'w')
    for e in sorted(e2effect.keys(), key = lambda x:e2effect[e], reverse=True):
        outf.write('%d\t%.4f\t%d\n' %(e, e2effect[e], e2cnt[e]))
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

