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
    labels = f['label'][:]
    f.close()
    all_event = []
    all_label = []

    for row in events:
        all_event.append(treatLongDat(row,40))

    all_label = []
    for i in labels:
        all_label.append(i)

    f = open(os.path.join(out_dir, prefix + ".visit." + tag), 'wb')
    cPickle.dump(all_event, f, cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = open(os.path.join(out_dir, prefix + ".label." + tag), 'wb')
    cPickle.dump(all_label,f,cPickle.HIGHEST_PROTOCOL)
    f.close()



out_dir = os.path.join(os.path.join(script_dir, "baseline"), 'retainData')
dataset_dir = ICU_exper_dir
prefix = "icu_catAtt"
files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']
datasets = [os.path.join(dataset_dir, file) for file in files]

makeFile(datasets[0], out_dir, "train", prefix)
makeFile(datasets[1], out_dir, "valid", prefix)
makeFile(datasets[2], out_dir, "test", prefix)
