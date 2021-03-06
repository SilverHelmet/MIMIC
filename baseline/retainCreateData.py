import numpy as np
import cPickle
import baseline_util
from util import *
np.set_printoptions(threshold=np.inf)
import h5py
import os


#input is the input string
#time is the number you want to split your data
#we delete all 1s and 0s and then split them into the number you want
#if the num of your string is too short, we will only use every element as input

def treatLongDat(input,time):
    output=[]
    if len(input)<=time:
        for i in input:
            output.append([i])
        return output
    else:
        length=len(input)/time
        for i in range(time):
            if i!=time-1:
                output.append(input[i*length:(i+1)*length])
            else:
                output.append(input[i*length:])
        return output



    for i in test_event:
        all_test_event.append(treatLongDat(stripList(i),20))

    all_test_label=[]
    for i in test_label:
        all_test_label.append(i)

    f=open("./50data/visit.test",'wb')
    cPickle.dump(all_test_event,f,cPickle.HIGHEST_PROTOCOL)
    f.close()
    f=open("./50data/label.test",'wb')
    cPickle.dump(all_test_label,f,cPickle.HIGHEST_PROTOCOL)
    f.close()
    hf_test.close()



def makeFile(dataset, out_dir, tag, prefix):
    f = h5py.File(dataset, 'r')
    events = f['event'][:]
    print "event max index = ", events.max()
    labels = f['label'][:]
    ids = f['sample_id'][:]
    f.close()
    all_event = []
    all_label = []

    for row in events:
        all_event.append(treatLongDat(row,40))

    all_label = []
    for label in labels:
        all_label.append(label)

    all_id = []
    for sample_id in ids:
        all_id.append(sample_id)

    f = open(os.path.join(out_dir, prefix + ".visit." + tag), 'wb')
    cPickle.dump(all_event, f, cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = open(os.path.join(out_dir, prefix + ".label." + tag), 'wb')
    cPickle.dump(all_label,f,cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = open(os.path.join(out_dir, prefix + '.id.' + tag), 'wb')
    cPickle.dump(all_id,f,cPickle.HIGHEST_PROTOCOL)
    f.close()




# dataset_dir = ICU_exper_dir
# prefix = "icu_catAtt"
out_dir = os.path.join(os.path.join(script_dir, "baseline"), 'retainData')

# dataset_dir = ICU_merged_exper_dir
# prefix = "icu"
# files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']

# dataset_dir = death_merged_exper_dir
# prefix = "death"
# dataset_dir = death_exper_dir

# prefix = "death_catAtt"
# files = ["death_train_1000.h5", "death_valid_1000.h5", "death_test_1000.h5"]
# datasets = [os.path.join(dataset_dir, file) for file in files]

# prefix = "zhu"
# files = ['train.h5', 'valid.h5', 'test.h5']
prefix = 'zhu_catAtt'
files = ['train_catAtt.h5', 'valid_catAtt.h5', 'test_catAtt.h5']
dataset_dir = "zhu_exper"
datasets = [os.path.join(dataset_dir, file) for file in files]


makeFile(datasets[0], out_dir, "train", prefix)
makeFile(datasets[1], out_dir, "valid", prefix)
makeFile(datasets[2], out_dir, "test", prefix)
