from util import death_exper_dir, model_dir, Print
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
    settings = ["none"] + "settings/sample_test/sample.txt".split(' ')
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
    
    
    Print('load model from [%s]' %model_path)
    model = load_model(model_path, get_custom_objects())
    Print('load over')
    dataset = Dataset(dataset_path, seg_path)

    return model, setting, dataset, sorted_events

def load_death_fixlength16():
    settings = ["none"] + 'settings/fea_gcn.txt settings/catAtt_lstm.txt settings/fixLength16.txt @gcn_mode=-1|gcn_hidden_dim=8|gcn_num_head=12'.split(' ')
    setting = load_argv(settings)
    model_path = os.path.join(model_dir, 'death_fixLength16_fea_catAtt_gcn_mode-1_12X8.model2')
    dataset_path = os.path.join(death_exper_dir, 'death_test_1000.h5')
    seg_path = os.path.join(death_exper_dir, 'segs/segs/death_test_1000_segmode=fixLength_maxchunk=63_length=16.h5')
    sorted_events = load_event_rank(os.path.join(model_dir, 'event_filter/event_scores_400.txt'))
    
    
    Print('load model from [%s]' %model_path)
    model = load_model(model_path, get_custom_objects())
    Print('load over')
    dataset = Dataset(dataset_path, seg_path)

    return model, setting, dataset, sorted_events


if __name__ == "__main__":
    # model, setting, dataset, sorted_events = load_sample()
    # model, setting, dataset, sorted_events = load_death_timeAggre()
    model, setting, dataset, sorted_events = load_death_fixlength16()

    thresholds = [0.5, 1.0]
    thresholds = [0.05, 0.1, 0.15,0.20,0.3,0.4,0.5, 0.6,0.7,0.8]
    
    thresholds.reverse()
    Print("load dataset")
    dataset.load(True, False, True)
    Print('load over')

    size = len(sorted_events)
    for threshold in thresholds:
        
        ed = int(size * threshold)
        reserved_events = set(sorted_events[:ed])
        
        
        setting['event_dim'] = 3418
        setting['max_segs'] = dataset.segs.shape[1]
        setting['max_seg_length'] = dataset.max_seg_length  

        info = {
            'mask': 0,
            'total': 0,
        }
        Print('eval')
        test_eval = dataset.eval(model, setting,reserved_events, info, verbose = True)
        print('#masked event = %d/%.4f%%' %(info['mask'], info['mask'] * 100.0 / info['total']) )
        print_eval('threshold = %.2f, ' %threshold, test_eval)

    