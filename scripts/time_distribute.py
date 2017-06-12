import sys
import scripts_util
from util import *
import h5py
import datetime 
import matplotlib.pyplot as plt
import math
import os


def load_dataset(dataset_file):
    d = h5py.File(dataset_file)

    events = d['event'][:]
    times = d['time'][:]
    label = d['label'][:]
    sample_ids = d['sample_id'][:]
    return events, times, label
    
def count(times, st, ed):
    cnt = 0
    for time in times:
        if time >= st and time < ed:
            cnt += 1
    return cnt

def time_distribution(events, times):
    events = [i for i in events if i != 0]
    fi = len(events)
    times = times[:fi]
    times = [parse_time(time) for time in times]
    step_hours = 0.1
    duration_hours = 0.5
    step = datetime.timedelta(hours = step_hours)
    duration = datetime.timedelta(hours = duration_hours)
    
    ddl = times[-1] + duration
    x = []
    y = []
    z = []
    pos = -1
    st = times[0] - datetime.timedelta(hours = -pos) - datetime.timedelta(hours = duration_hours / 2)
    for time in times:
        z.append((time - times[0]).total_seconds()/3600.0)
    while st <= ddl :
        ed = st + duration
        counts = count(times, st, ed)
        x.append(pos)
        y.append(counts)
        st += step
        pos += step_hours
    return x, y, z

def merge(z, sing_size):
    points = set(z)
    new_z = [] 
    new_size = []
    for p in sorted(points):
        size = sing_size * math.sqrt(z.count(p))
        new_z.append(p)
        new_size.append(size)
    return new_z, new_size

def plot(x, y, z, name):
    plt.figure(0)
    plt.plot(x, y, 'b-')
    # plt.plot(z, [0.2] * len(z), 'ro')
    s = [20] * len(z)
    z, s = merge(z, 20)
    plt.scatter(z, [0.1] * len(z), s = s, c = "r")
    plt.xlabel("#time_point = %d" %len(z))
    plt.savefig(os.path.join(time_dis_graph_dir, name))
    
    plt.close(0)
    
        
    
    
    
if __name__ == "__main__":
    dataset = sys.argv[1]
    plt.style.use('ggplot')
    # dataset = "ICU_exper/sample.h5"
    events, times, label = load_dataset(dataset)

    # x, y, z = time_distribution(events[76], times[76])
    rand_permu = np.random.permutation(len(events))[:300]
    for i in rand_permu:
        x, y, z = time_distribution(events[i], times[i])
        plot(x, y, z, 'label=%d_row=%d.png' %(label[i],i))
    
    # print y
    # print z