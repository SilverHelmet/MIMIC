from util import death_exper_dir, model_dir
import numpy as np
import os
from keras.models import load_model
from models.models import get_custom_objects
from models.dataset import Dataset, print_eval
from train_rnn import load_argv    

def load_event_rank(filepath):
    scores = []
    for line in file(filepath):
        scores.append(float(line))

    seq = range(len(scores))
    sorted_events = sorted(seq, key = lambda x: scores[x], reverse = True)

    return sorted_events


def load_sample():
    settings = ["none"] + "settings/fea_gcn.txt settings/catAtt_lstm.txt settings/timeAggre.txt settings/params/gcn_mode-1.txt settings/out_model.txt".split(' ')
    setting = load_argv(settings)

    model_path = os.path.join(model_dir, 'sample.model')
    exper_dir = os.path.join(death_exper_dir, 'sample')
    dataset_path = os.path.join(exper_dir, 'samples.h5')
    seg_path = os.path.join(exper_dir,  'samples_seg.h5')
    sorted_events = load_event_rank(os.path.join(model_dir, 'event_filter/event_scores.txt'))

    model = load_model(model_path, get_custom_objects())
    dataset = Dataset(dataset_path, seg_path)

    return model, setting, dataset, sorted_events



def load_death_timeAggre():
    settings = ["none"] + "settings/fea_gcn.txt settings/catAtt_lstm.txt settings/timeAggre.txt settings/params/gcn_mode-1.txt settings/out_model.txt".split(' ')
    setting = load_argv(settings)

    model_path = os.path.join(model_dir, 'death_timeAggre_fea_catAtt_gcn.model')
    dataset_path = os.path.join(death_exper_dir, 'death_test_1000.h5')
    seg_path = os.path.join(death_exper_dir, 'segs/death_test_1000_segmode=timeAggre_maxchunk=32.h5')
    sorted_events = load_event_rank(os.path.join(model_dir, 'event_filter/event_scores_300.txt'))
    
    model = load_model(model_path, get_custom_objects())
    dataset = Dataset(dataset_path, seg_path)

    return model, setting, dataset, sorted_events



if __name__ == "__main__":
    # model, setting, dataset, sorted_events = load_sample()
    model, setting, dataset, sorted_events = load_death_timeAggre()

    thresholds = [0.5, 1.0]
    thresholds = [0.05, 0.1, 0.15,0.20,0.3,0.4,0.6,0.8]
    thresholds.reverse()
    
    size = len(sorted_events)
    for threshold in thresholds:
        
        ed = int(size * threshold)
        filtered_events = set(sorted_events[:ed])
        dataset.load(True, False, True, event_set = filtered_events)
        setting['event_dim'] = 3418
        setting['max_segs'] = dataset.segs.shape[1]
        setting['max_seg_length'] = dataset.max_seg_length  

        test_eval = dataset.eval(model, setting)
        print_eval('threshold = %.2f, ' %threshold, test_eval)

    