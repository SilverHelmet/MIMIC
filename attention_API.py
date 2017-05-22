import sklearn
from get_attention import *
import numpy as np
from matplotlib.colors import ListedColormap
from models.dataset import Dataset
from models.models import np_mask_softmax
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import *
import sys


if __name__ == "__main__":
    
    model = load_model("RNNmodels/death_timeAggre_fea_catAtt.model", custom_objects = get_custom_objects())
    dataset = Dataset('sample_exper/death_sample.h5', 'sample_exper/death_seg.h5')
    print "load over"
    dataset.load()


    data_e = np.array(dataset.event_mat(12))
    data_f = np.array(dataset.feature_mat(12))
    print data_e.shape, data_f.shape
    X = [np.expand_dims(data_e, 0), np.expand_dims(data_f, 0)]
    event_attention = get_event_attention(model, X)[0]
    emds = get_embedding(model, X)[0]
    outputs, states = get_RNN_result(model, X)
    outputs = outputs[0]
    states = states[0]
    print event_attention.shape, emds.shape, outputs.shape, states.shape


