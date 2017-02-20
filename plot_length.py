#encoding:utf-8
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
import sys

class Data:
    def __init__(self, data, errors, label, reverse = False):
        self.data = data
        self.errors = errors
        if reverse:
            self.data.reverse()
            self.errors.reverse()
        self.label = label
 
def plot_length(results, names, ylabel):
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

    ax = plt.gca()
    ax.legend(loc = 'lower right')
    
    ax.set_xlim((0.5, 1.5 + len(names) ))
    ax.set_xticks(range(1, 1 + len(names)))
    ax.set_xticklabels(names)
    ax.set_xlabel("segment length")
    ax.set_title(ylabel)


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
    plot_label = ['event', 'event & category attr & numerical attr']
    idx = 1
    names = ["length=4", 'length=8','length=16', 'length=32', 'length=64', 'length=128', 'length=256']
    names = [1, 4, 8, 16, 32, 64, 128, 256]
    for group in groups:
        data = data_group[group]
        for res in data:
            print res.label
        data = [res for res in data if res.label in plot_label]
        for res in data:
            if res.label == plot_label[1]:
                res.label = "event with attr"
        plt.subplot(1, len(groups), idx)
        idx += 1

        plot_length(data, names, group)
    
    plt.show()
    # plt.savefig("graph/icu_category.png")
    # plt.savefig("graph/icu_category & numerical.png")
    
