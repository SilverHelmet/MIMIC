import sklearn
from get_attention import *
import numpy as np
from matplotlib.colors import ListedColormap
from models.dataset import Dataset
from models.models import np_mask_softmax
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import *

def attention_score_at_time(model, event_mat, time):
    if time == 0:
        rnn_layer = model.get_layer('rnn')
        output = np.zeros(rnn_layer.output_dim)
    else:
        x = np.expand_dims(event_mat[:time, :], 0)
        output = get_event_output(model, x)[0]
    event_seq = event_mat[time]
    emds = get_embedding(model, event_seq, name = 'embedding')
    scores = get_event_attention_score_at_seg(model, output, emds)
    return scores, output


def plot_event_attention_at_time(model, event_mat, plot_etype, ax, pca, event_map, output):
    emds = []
    for event_seq in event_mat:
        for event in event_seq:
            if event_map[event] in plot_etype:
                emds.append(event)
    plot_etype_list = sorted(plot_etype)
    labels = [plot_etype_list.index(event_map[event]) for event in emds] 
    emds = get_embedding(model, np.array(emds))
    X = pca.transform(emds)
    x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min, y_max = X[:, 1].min() -  .2, X[:, 1].max() + .2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    

    cm = plt.cm.RdBu
    cm_bright = ListedColormap([
        '#16A085', 
        "#27AE60", 
        "#2980B9", 
        "#8E44AD", 
        "#2C3E50", 
        "#F39C12",
        "#D35400",
        "#C0392B",
        "#BDC3C7",
        "#7F8C8D",
        "#E67E22",
        "#F1C40F"])
    
    emds = pca.inverse_transform(np.array([xx.reshape(-1), yy.reshape(-1)]).T)
    scores = get_event_attention_score_at_seg(model, output, emds)
    scores = scores.reshape(xx.shape)
    ax.contourf(xx, yy, scores, cmap = cm, alpha = .8)
    ax.scatter(X[:, 0], X[:, 1], c = labels, cmap = cm_bright, s= 40)
    

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())



        
    
def get_pca(model):
    emd_layer = model.get_layer("embedding")
    W = K.get_value(emd_layer.W)[2:,:]
    pca = PCA(n_components = 2)
    pca.fit(W)
    return pca


def plot_event_attention(model, event_mat, times, event_map):
    scores = []
    plot_etype = set()
    indices = [-1,-2,0,1]
    outputs = []
    for time in times:
        score, output = attention_score_at_time(model, event_mat, time)
        scores.append(attention_score_at_time(model, event_mat, time))
        outputs.append(output)
        event_seq = event_mat[time]
        sorted_indices = np.argsort(score)
        sorted_indices = [index for index in sorted_indices if event_seq[index] != 0]
        for index in indices:
            plot_etype.add(event_map[event_seq[sorted_indices[index]]])
    
    i = 0
    plt.style.use('ggplot')
    pca = get_pca(model)
    for time, score, output in zip(times, scores, outputs):
        i += 1
        ax = plt.subplot(1, len(times), i)
        event_seq = event_mat[time, :]
        plot_event_attention_at_time(model, event_mat, plot_etype, ax, pca, event_map, output)
    
    plt.show()

def plot_temporal_attention(model, event_mat, times):
    times = np.array(times)
    x = np.expand_dims(event_mat, 0)
    event_att = get_event_attention(model, x)[0]
    temporal_att = get_temporal_attention(model, x)[0]
    event_att = np.expand_dims(temporal_att, 1) + event_att
    temporal_att = temporal_att[times]

    time_width = 2
    bar_width = 0.4
    left = []
    height = []
    labels = []
    indices = [-1,-2,1,0]

    plt.style.use('ggplot')
    colors = []
    plot_etypes = []
    x_pos = 0 + bar_width / 2
    xs = []
    for time in times:
        time_pos = x_pos
        event_seq = event_mat[time]
        att = event_att[time]
        sorted_indices = np.argsort(att)
        sorted_indices = [index for index in sorted_indices if event_seq[index] != 0]
        if len(sorted_indices) <= 4:
            used_indices = reversed(sorted_indices)
        else:
            used_indices = np.array(sorted_indices)[indices]

        for index in used_indices:
            event = event_seq[index]
            if not event in plot_etypes:
                plot_etypes.append(event)
            height.append(att[index])
            labels.append(event)
            left.append(time_pos)
            time_pos += bar_width
            colors.append(plot_etypes.index(event))
            
        
        xs.append((x_pos + time_pos) / 2)
        x_pos += time_width

    colors = np.array(colors, dtype = float)
    colors /= colors.max() 
    height = np.array(height)
    left = np.array(left)
    cmap = plt.cm.get_cmap('RdYlBu')
    colors = cmap(colors)
    plt.bar(left = left, height = height, width = bar_width, 
        color = colors,  align = 'center', alpha = 0.8)
    ax = plt.gca()
    ax.set_xticks(left)
    ax.set_xticklabels(labels, rotation = 25)

    
    plt.plot(xs, temporal_att, color = 'k', marker = 'o')

    ax =plt.gca()
    ax.set_ylim(max(0, temporal_att.min() - 0.1), min(height.max() + 0.1, 2))
    ax.set_xlim((0, left.max() + bar_width + 0.1))
    plt.show()




if __name__ == "__main__":
    
    # event_map = dict([(i,i/2+1) for i in range(10)])
    data1 = np.array([[1,2,3,5,4],[3,1,2,2,0], [7,8,2,4,5]])
    data2 = np.array([[3,1,2,1,1],[1,1,7,8,0], [7,8,7,8,7]])
    event_map = merge_event_map('result/event_des_text.tsv')
    model = load_model("RNNmodels/death_timeAggre_catAtt.model", custom_objects = get_custom_objects())
    # model = load_model("RNNmodels/test.model", custom_objects = get_custom_objects())
    dataset = Dataset('sample_exper/sample_icu.h5py', 'sample_exper/sample_segs.h5py')
    print "load over"
    dataset.load()
    times = [0, 2, 4, 6, 8, 10]
    data = np.array(dataset.event_mat(5))
    # plot_temporal_attention(model, data, times)
    plot_event_attention(model, data, [0,4,10], event_map)
    # plot_temporal_attention(model, data1, [0, 2])
    


    

    
