from models.dataset import Dataset
from util import parse_time, death_exper_dir, result_dir
import matplotlib.pyplot as plt
import os
import h5py

def stat():
    f = h5py.File(os.path.join(death_exper_dir, 'death_train_1000.h5'), 'r')
    t = f['time'][:]
    f.close()

    outf = file(os.path.join(result_dir, 'time_dis_in_a_day.txt'), 'w')
    for times in t:
        for time_str in times:
            time = parse_time(time_str)
            if time:
                offset = t.hour * 3600 + t.minute * 60 + t.second
                outf.write("%s\n" %offset)

    outf.close()

def plot():
    plt.style.use('ggplot')

if __name__ == "__main__":
    stat()