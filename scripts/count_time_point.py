import h5py
import scripts_util
from util import *
from time_distribute import load_dataset, time_distribution, merge
import matplotlib.pyplot as plt

def count_points(events, times):
    events = [i for i in events if i > 0]
    times = [parse_time(time) for time in times[:len(events)]]
    s = set(times)
    
    return len(s)

def plot(cnts):
    plt.figure(0)
    bins = [0, 5.01, 10.1,15.1,20.1,25.1,30.1,40.1,50.1,75.1,100.1,300.1]
    weights = np.ones_like(cnts) / (len(cnts)+0.0)
    plt.hist(cnts, bins = bins, normed = False, weights = weights)
    plt.xticks([10, 20, 30, 40, 50, 75, 100, 300])
    locs, labels = plt.xticks()
    plt.savefig("graph/#time_points.png")

if __name__ == "__main__":
    dataset = sys.argv[1]
    plt.style.use('ggplot')
    # dataset = "ICU_exper/samples.h5"
    events, times, label = load_dataset(dataset)
    cnts = []
    for i in xrange(len(events)):
        if i % 10000 == 0:
            print "handle %d" %i
        cnt = count_points(events[i], times[i])
        cnts.append(cnt)
    plot(cnts)
    

