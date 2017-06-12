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
from event_des import EventDescription

def attention_score_at_time(model, x, time):
    if time == 0:
        rnn_layer = model.get_layer('rnn')
        output = np.zeros(rnn_layer.output_dim)
    else:
        output = get_event_output_at_time(model, x, time)[0]
    emds = get_embedding(model, x)[0][time]
    scores = get_event_attention_score_at_seg(model, output, emds)
    return scores, output


def plot_event_attention_at_time(model, event_mat, feature_mat, plot_etype, ax, pca, event_map, output):
    emds = []
    feas = []
    event_list = []
    x = [np.expand_dims(event_mat, 0), np.expand_dims(feature_mat, 0)]
    emd_out = get_embedding(model, x)[0]

    for i, event_seq in enumerate(event_mat):
        for j, event in enumerate(event_seq):
            if event_map[event] in plot_etype:
                event_list.append(event)
                emds.append(emd_out[i][j])
                feas.append(feature_mat[i][j])
    emds = np.array(emds)
    plot_etype_list = sorted(plot_etype)
    labels = [plot_etype_list.index(event_map[event]) for event in event_list] 
    X = pca.transform(emds)

    scores = get_event_attention_score_at_seg(model, output, emds)
    indices = np.argsort(np.array(scores))
    pos = X[indices[-1]]
    event_des = EventDescription()
    des = event_des.get_des(event_list[indices[-1]], feas[indices[-1]])
    print "\n".join(des)
    print pos
    # ax.text(pos[0], pos[1], "\n".join(des), horizontalalignment = 'center', verticalalignment = "bottom", fontsize = 13)

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 500),
                         np.arange(y_min, y_max, (y_max - y_min) / 500) )
    

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
    print '-' * 50
    for idx, event in enumerate(event_list):
        print X[idx, 0], X[idx, 1], event, event_map[event]
    
    

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


def plot_event_attention(model, event_mat, feature_mat, times, event_map, out_path = None):
    plt.figure(0)
    scores = []
    plot_etype = set()
    indices = [-1,-2,0,1]
    outputs = []
    x = [np.expand_dims(event_mat, 0), np.expand_dims(feature_mat, 0)]
    for time in times:
        score, output = attention_score_at_time(model, x, time)
        scores.append(score)
        outputs.append(output)
        event_seq = event_mat[time]
        sorted_indices = np.argsort(score)
        sorted_indices = [index for index in sorted_indices if event_seq[index] != 0]
        if len(sorted_indices) < 2:
            continue
        for index in indices:
            plot_etype.add(event_map[event_seq[sorted_indices[index]]])
    i = 0
    pca = get_pca(model)
    for time, score, output in zip(times, scores, outputs):
        i += 1
        ax = plt.subplot(1, len(times), i)
        event_seq = event_mat[time, :]
        plot_event_attention_at_time(model, event_mat, feature_mat, plot_etype, ax, pca, event_map, output)
    plt.grid()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close(0)

def plot_temporal_attention(model, event_mat, feature_mat, times):
    fig = plt.figure(0)
    event_des = EventDescription()
    times = np.array(times)
    x = [np.expand_dims(event_mat, 0), np.expand_dims(feature_mat, 0)]
    event_att = get_event_attention(model, x)[0]
    temporal_att = get_temporal_attention(model, x)[0]
    event_att = np.expand_dims(temporal_att, 1) + event_att
    temporal_att = temporal_att[times]

    time_width = 2.5
    bar_width = 0.4
    left = []
    height = []
    labels = []
    indices = [-1,-2,1,0]


    colors = []
    plot_etypes = []
    x_pos = 1 + bar_width / 2
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

        add_text = True
        for index in used_indices:
            event = event_seq[index]
            if not event in plot_etypes:
                plot_etypes.append(event)
            height.append(att[index])
            labels.append(event)
            left.append(time_pos)
            pos_x = time_pos
            pos_y = att[index]
            time_pos += bar_width
            colors.append(plot_etypes.index(event))

            if add_text:
                des = event_des.get_des(event, feature_mat[time][index])
                plt.text(pos_x  - bar_width *2, pos_y + 0.001, "\n".join(des), horizontalalignment = 'left', verticalalignment = "bottom", fontsize = 13)
                add_text = False
            
        
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
    ax.set_xlabel("Time")

    
    plt.plot(xs, temporal_att, color = 'k', marker = 'o')

    ax.set_ylim(max(0, temporal_att.min() - 0.1), height.max() + 0.4)
    ax.set_xlim((0, left.max() + bar_width + 1))
    plt.grid()
    plt.show()
    plt.close(0)



if __name__ == "__main__":
    
    # event_map = dict([(i,i/2+1) for i in range(10)])
    data1 = np.array([[1,2,3,5,4],[3,1,2,2,0], [7,8,2,4,5]])
    data2 = np.array([[3,1,2,1,1],[1,1,7,8,0], [7,8,7,8,7]])
    event_map = merge_event_map('result/event_des_text.tsv')
    model = load_model("RNNmodels/death_timeAggre_fea_catAtt.model", custom_objects = get_custom_objects())
    # model = load_model("RNNmodels/test.model", custom_objects = get_custom_objects())
    dataset = Dataset('sample_exper/death_sample.h5', 'sample_exper/death_seg.h5')
    print "load over"
    dataset.load()

    plt.style.use('ggplot')
    times = range(3, 3 + 3*5, 3)
    data_e = np.array(dataset.event_mat(12))
    data_f = np.array(dataset.feature_mat(12))
    X = [np.expand_dims(data_e, 0), np.expand_dims(data_f, 0)]
    event_attention = get_event_attention(model, X)[0]
    emds = get_embedding(model, X)[0]
    outputs, states = get_RNN_result(model, X)[0]
    print event_attention.shape(), emds.shape(), outputs.shape(), states.shape()


    # plot_temporal_attention(model, data_e, data_f, times)
    # plot_event_attention(model, data_e, data_f, [0, 10, 20], event_map)
    # for idx in range(0, 1000, 5):
    #     data_e = np.array(dataset.event_mat(idx))
    #     data_f = np.array(dataset.feature_mat(idx))
    #     times = [0, 3, 6]
    #     out_path = os.path.join(graph_dir, "event_attention")
    #     out_path = os.path.join(out_path, "%d_%s.png" %(idx, times))
        
    #     plot_event_attention(model, data_e, data_f, times, event_map, out_path)

    # data_e = np.array(dataset.event_mat(600))
    # data_f = np.array(dataset.feature_mat(600))
    # times = [0, 3, 6]
    # plot_event_attention(model, data_e, data_f, times, event_map)
    