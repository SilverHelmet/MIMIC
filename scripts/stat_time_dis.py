from models.dataset import Dataset
from util import parse_time, death_exper_dir, result_dir
import os
import h5py

if __name__ == "__main__":
    f = h5py.File(os.path.join(death_exper_dir, 'death_train_1000.h5'), 'r')
    t = f['time'][:]
    f.close()

    outf = file(os.path.join(result_dir, 'time_dis_in_a_day.txt'), 'w')
    for times in t:
        for time_str in times:
            time = parse_time(time)
            if time:
                offset = t.hour * 3600 + t.minute * 60 + t.second
                outf.write("%s\n" %offset)

    outf.close()








