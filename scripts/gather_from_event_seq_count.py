from scripts_util import *
from stat_event_seq import PatientCnt
from util import *
import matplotlib.pyplot as plt
import numpy as np


def load(stat_filepath):
    reader = file(stat_filepath, 'r')
    patient_cnt_map = {}
    while True:
        patient_cnt = PatientCnt.load(reader)
        if patient_cnt is not None:
            pid = patient_cnt.pid
            patient_cnt_map[pid] = patient_cnt
        else:
            break

    return patient_cnt_map
        
def stat(patient_cnt_map):
    for pid in patient_cnt_map:
        patient_cnt_map[pid].stat()

def divide(x, nb_partitions):
    x.sort()
    chunk = len(x) / int(nb_partitions)
    bins = []
    for i in range(nb_partitions):
        value = x[chunk * i]
        if len(bins) == 0 or bins[-1] != value: 
            bins.append(x[chunk * i])
    if bins[-1] != x[-1]:
        bins.append(x[-1])
    return bins
        
        


def plot_nb_hospital(patient_cnt_map, only_emergency, only_not_emergency):
    name = "#hospitals per patient"
    x = []
    for pid in patient_cnt_map:
        value = 0
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            value += 1
        x.append(value)
            
    minx = 1
    maxx = reduce(max, x)
    bins = range(minx, maxx+1, 1)
    plt.figure(1)  
    plt.hist(x, bins = bins,  normed = True)
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_hospital_days(patient_cnt_map, only_emergency, only_not_emergency):
    name = "#days per hospital"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            x.append(hospital_cnt.duration_days)
    # x.sort()
    maxx = reduce(max, x)
    minx = reduce(min, x)
    bins = range(minx, maxx+1, 1)
    # bins = divide(x, 10)
    # print bins
    plt.figure(2)
    plt.hist(x, bins = bins, normed = True)
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_hospital_nb_events(patient_cnt_map, only_emergency, only_not_emergency):
    name = "#events per hospital"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            x.append(hospital_cnt.nb_event)
    bins = [0,300,600,1000,1500,2000,3000,5000,1e4,1e5,reduce(max, x)]
    plt.figure(3)
    ax = plt.gca()
    ax.set_xscale('log')
    weights = np.ones_like(x) / (len(x)+0.0)
    plt.hist(x, bins = bins, weights = weights, normed = False)
    plt.title(name)
    plt.xlabel(str(bins))
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_nb_icu(patient_cnt_map, only_emergency, only_not_emergency):
    name = "#icu per hospital"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            x.append(hospital_cnt.nb_icu)
    plt.figure(4)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = xrange(minx, maxx+1, 1)
    plt.hist(x, bins = bins)
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_icu_hours_to_last(patient_cnt_map, only_emergency, only_not_emergency):
    name = "icu's hours to last icu or admission"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            for icu_cnt in hospital_cnt.icu_cnts:
                x.append(icu_cnt.hours_to_last)
    plt.figure(5)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = [0, 1, 6, 12, 24, 48, 24*4, 24 * 8, 24 * 16, int(maxx)+1]
    
    weights = np.ones_like(x) / (len(x)+0.0)    
    # plt.hist(x, bins = bins, weights = weights)
    plt.hist(x, bins = bins)
    ax = plt.gca()
    ax.set_xscale('symlog', basex = 24)
    plt.xlabel(str(bins))
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_icu_hours_to_last_icu(patient_cnt_map, only_emergency, only_not_emergency):
    name = "icu's hours to last icu"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            for idx, icu_cnt in enumerate(hospital_cnt.icu_cnts):
                if idx == 0:
                    continue
                x.append(icu_cnt.hours_to_last)
    plt.figure(10)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = [0, 1, 6, 12, 24, 48, 24*4, 24 * 8, 24 * 16, int(maxx)+1]
    
    weights = np.ones_like(x) / (len(x)+0.0)    
    # plt.hist(x, bins = bins, weights = weights)
    plt.hist(x, bins = bins)
    ax = plt.gca()
    ax.set_xscale('symlog', basex = 10)
    plt.xlabel(str(bins))
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_icu_nb_events_to_last(patient_cnt_map, only_emergency, only_not_emergency):
    name = "icu's events to last icu or admission"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            for icu_cnt in hospital_cnt.icu_cnts:
                x.append(icu_cnt.nb_events_to_last)
    plt.figure(6)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = [0,1,10,30,100,300,1000,2000,3000,5000,reduce(max, x)]
    weights = np.ones_like(x) / (len(x)+0.0)    
    # plt.hist(x, bins = bins, weights = weights)
    plt.hist(x, bins = bins)
    ax = plt.gca()
    ax.set_xscale('symlog', basex = 10)
    plt.xlabel(str(bins))
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_icu_nb_events_to_last_icu(patient_cnt_map, only_emergency, only_not_emergency):
    name = "icu's events to last icu"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            for idx, icu_cnt in enumerate(hospital_cnt.icu_cnts):
                if idx == 0:
                    continue
                x.append(icu_cnt.nb_events_to_last)
    plt.figure(9)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = [0,1,10,30,100,300,1000,2000,3000,5000,reduce(max, x)]
    weights = np.ones_like(x) / (len(x)+0.0)    
    # plt.hist(x, bins = bins, weights = weights)
    plt.hist(x, bins = bins)
    ax = plt.gca()
    ax.set_xscale('symlog', basex = 10)
    plt.xlabel(str(bins))
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_icu_duration_hours(patient_cnt_map, only_emergency, only_not_emergency):
    name = "#hours per icu"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            for icu_cnt in hospital_cnt.icu_cnts:
                x.append(icu_cnt.duration_hours)
    plt.figure(7)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = [0, 12, 24, 48, 24*4, 24 * 8, 24 * 16, int(maxx)+1]
    
    weights = np.ones_like(x) / (len(x)+0.0)    
    plt.hist(x, bins = bins, weights = weights)
    ax = plt.gca()
    ax.set_xscale('log', basex = 24)
    plt.xlabel(str(bins))
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_icu_nb_events(patient_cnt_map, only_emergency, only_not_emergency):
    name = "#events per icu"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            for icu_cnt in hospital_cnt.icu_cnts:
                x.append(icu_cnt.nb_event)
    plt.figure(8)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = [0,1,10,30,100,300,1000,2000,3000,5000,reduce(max, x)]
    weights = np.ones_like(x) / (len(x)+0.0)    
    plt.hist(x, bins = bins, weights = weights)
    ax = plt.gca()
    ax.set_xscale('symlog', basex = 10)
    plt.xlabel(str(bins))
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_nb_events_after_last_icu(patient_cnt_map, only_emergency, only_not_emergency):
    name = "#events after last icu"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            x.append(hospital_cnt.nb_events_aft_last_icu)
    plt.figure(11)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = [0,1,10,30,100,300,1000,2000,3000,5000,reduce(max, x)]
    weights = np.ones_like(x) / (len(x)+0.0)    
    # plt.hist(x, bins = bins, weights = weights)
    plt.hist(x, bins = bins)
    ax = plt.gca()
    ax.set_xscale('symlog', basex = 10)
    plt.xlabel(str(bins))
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))

def plot_hours_after_last_icu(patient_cnt_map, only_emergency, only_not_emergency):
    name = "#hours after last icu"
    x = []
    for pid in patient_cnt_map:
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            if only_emergency and not hospital_cnt.is_emergency:
                continue
            if only_not_emergency and hospital_cnt.is_emergency:
                continue
            x.append(hospital_cnt.nb_hours_aft_last_icu)
    plt.figure(12)
    minx = reduce(min, x)
    maxx = reduce(max, x)
    bins = [0, 1, 6, 12, 24, 48, 24*4, 24 * 8, 24 * 16, int(maxx)+1]
    
    weights = np.ones_like(x) / (len(x)+0.0)    
    # plt.hist(x, bins = bins, weights = weights)
    plt.hist(x, bins = bins)
    ax = plt.gca()
    ax.set_xscale('symlog', basex = 24)
    plt.xlabel(str(bins))
    plt.title(name)
    plt.savefig(os.path.join(graph_dir, name + ".png"))


    

def plot(patient_cnt_map, only_emergency = False, only_not_emergency = False):
    plt.style.use('ggplot')
    plot_funcs = [plot_nb_hospital, plot_hospital_days, plot_hospital_nb_events, plot_nb_icu,
        plot_icu_hours_to_last, plot_icu_nb_events_to_last,plot_icu_duration_hours, plot_icu_nb_events,
        plot_icu_nb_events_to_last_icu, plot_icu_hours_to_last_icu, plot_nb_events_after_last_icu, plot_hours_after_last_icu]
    for plot_func in plot_funcs[:]:
        plot_func(patient_cnt_map, only_emergency, only_not_emergency)
    
    
    
        


if __name__ == "__main__":
    event_seq_stat_result_path = os.path.join(event_seq_stat_dir, "event_seq_stat.result")
    patient_cnt_map = load(event_seq_stat_result_path)
    stat(patient_cnt_map)
    plot(patient_cnt_map)
    
