from util import death_exper_dir, model_dir
import numpy as np
import os
from keras.models import load_model
from models.models import get_custom_objects

def load_event_rank(filepath):
    scores = []
    for line in file(filepath):
        scores.append(float(line))

    seq = range(len(scores))
    sorted_event = sorted(seq, key = lambda x: scores[x], reverse = True)

    return sorted_event


def load_sample():
    model_path = os.path.join(model_dir, 'sample.model')
    exper_dir = os.path.join(death_exper_dir, 'sample')
    dataset_path = os.path.join(exper_dir, 'samples.h5')
    seg_path = os.path.join(exper_dir,  'samples_seg.h5')
    sorted_events = load_event_score(os.path.join(model_dir, 'event_filter/event_scores.txt'))

    model = load_model(model_path, get_custom_objects())
    dataset = Dataset(dataset_path, seg_path)

    return model, dataset, sorted_events



def load_death():
    model_path = os.path.join(model_dir, 'death_timeAggre_fea_catAtt_gcn.model')
    dataset_path = os.path.join(death_exper_dir, 'death_test_1000.h5')
    seg_path = os.path.join(death_exper_dir, 'segs/death_test_1000_segmode=timeAggre_maxchunk=32.h5')
    event_score = load_event_score(os.path.join())



if __name__ == "__main__":
    model, dataset, sorted_events = load_sample()

    
    thresholds = np.arange(0.1, 1.0, 0.1)
    size = len(sorted_event)
    for threshold in thresholds:
        ed = int(size * threshold)
        filtered_events = set(sorted_events[:ed]0
        dataset.load(False, True, True, event_set = filtered_events)

    