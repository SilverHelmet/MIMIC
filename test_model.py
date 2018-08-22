from train_rnn import define_simple_seg_rnn, load_argv
from keras.models import Model, load_model
from models.dataset import Dataset, print_eval, sample_generator
import sys
import numpy as np


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

    print_probs(model, data, setting, 'result/death_test_probs.npy')

