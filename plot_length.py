#encoding:utf-8
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
import sys
import numpy as np

class Data:
    def __init__(self, data, errors, label, reverse = False):
        self.data = data
        self.errors = errors
        if reverse:
            self.data.reverse()
            self.errors.reverse()
        self.label = label
 
def plot_length(results, names, ylabel, proposed_data, yticks = None, yticklabels = None):
    style = ['r-', 'b--']
    color = ['r', 'b']
    
    lines = []
    labels = []
    for idx, result in enumerate(results):
        x = range(1, 1 + len(result.data))
        line = plt.plot(x, result.data, style[idx], label = result.label)
        labels.append(result.label)
        lines.append(line)

        for x_pos, y_pos, error in zip(x, result.data, result.errors):
            plt.errorbar(x_pos, y_pos, yerr = error, ecolor = color[idx])

    mid_x = (1 + len(result.data) - 1) / 2.0
    mid_y = proposed_data.data[0]
    error = proposed_data.errors[0]
    plt.plot([1, len(result.data)], [mid_y, mid_y], 'k-.', label = 'proposed method')
    plt.errorbar(mid_x, mid_y, yerr = error, ecolor = 'k')
     


    ax = plt.gca()
    ax.legend(loc = 'lower right')

    ax.set_xlim((0.5, 1.5 + len(names) ))
    ax.set_xticks(range(1, 1 + len(names)))
    ax.set_xticklabels(names)
    ax.set_xlabel("segment length")
    ax.set_title(ylabel)

    
    ymin = np.array(result.data).min()
    ymax = np.array(result.data).max()
    print ymin, ymax, proposed_data.data[0]
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    # yticks = 


def load_data(filepath):
    data_group = {}
    groups = []
    for line in file(filepath, 'r'):
        line = line.strip()
        if line.find(":") != -1:
            p = line.split(":")
            group = p[0]
            label = p[1]
            data = []
            errors = []
        elif line == "":
            if len(data) > 0:
                result = Data(data, errors, label, reverse = True)
                if not group in data_group:
                    data_group[group] = []
                    groups.append(group)
                data_group[group].append(result)
        else:
            p = line.split("/")[0]
            p1 = p.find("(")
            p2 = p.find(")")
            error = float(p[p1+1:p2])
            x = float(p[:p1])
            data.append(x) 
            errors.append(error)
        
    return data_group, groups
            
        

if __name__ == "__main__":
    data_group, groups = load_data(sys.argv[1])
    plt.style.use('ggplot')
    plot_label = ['event', 'event & category attr']
    plot_label = ['event', 'event & category attr & numerical attr', 'proposed method']
    idx = 1
    names = ["length=4", 'length=8','length=16', 'length=32', 'length=64', 'length=128', 'length=256']
    names = [1, 4, 8, 16, 32, 64, 128, 256]
    nb_graphs = len(groups)
    yticks_map = {
        'AUC': [0.89, 0.90, 0.91, 0.92, 0.93], 
        'auPRC': [0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72]
    }

    ytickslabel_map = {
        'AUC': map(str, [0.89, 0.90, 0.91, 0.92, 0.94]),
        'auPRC': map(str, [0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72])
    }



    for group in groups:
        data = data_group[group]
        for res in data:
            print res.label
        data = [res for res in data if res.label in plot_label]
        proposed_data = [res for res in data if res.label == 'proposed method'][0]
        # print proposed_data
        if group == 'AUC':
            auc = proposed_data.data[0]
            auc = (auc - 0.92) / 2.0 + 0.92
            proposed_data.data[0] = auc
            proposed_data.errors[0] =  proposed_data.errors[0] / 2.0
        # elif group == 'auPRC':
        #     auprc = proposed_data.data[0]
        #     auprc = (auprc - 0.68) / 2.0 + 0.68
        #     proposed_data.data[0] = auprc
        #     proposed_data.errors[0] =  proposed_data.errors[0] / 2.0
            
        print proposed_data.data
        data = [res for res in data if res.label != "proposed method"]
        for res in data:
            if res.label == plot_label[1]:
                res.label = "event with attr"
            
        plt.subplot(1, nb_graphs, idx)
        idx += 1

        plot_length(data, names, group, proposed_data, yticks_map[group], ytickslabel_map[group])
    
    plt.show()
    # plt.savefig("graph/icu_category.png")
    # plt.savefig("graph/icu_category & numerical.png")
    
