import sklearn
from get_attention import get_event_output
import numpy as np

def plot_event_attention(model, event_mat, time, event_map, plot_event_types,):
    rnn_layer = model.get_layer("rnn")
    if time == 0:
        output = np.zeros(rnn_layer.output_dim)
    else:
        x = np.expand_dims(event_mat[:time, :], 0)
        output = get_event_output(model, x)[0]
        
    
