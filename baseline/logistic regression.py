import baseline_util 
import h5py
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.svm import LinearSVC, SVC
from util import *
import os


        
#f = h5py.File('exper/emerg_urgent_train_100_2000_True.h5', 'r')
# t = h5py.File('exper/emerg_urgent_test_100_2000_True.h5', 'r')
dataset_dir = ICU_exper_dir
files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']
# dataset_dir = death_exper_dir
# files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']
f = h5py.File(os.path.join(dataset_dir, files[0]), 'r')
t = h5py.File(os.path.join(dataset_dir, files[2]), 'r')
labels = f['label'][:]
events = f['event'][:]
event_len = events.shape[1]
sids = f['sample_id'][:]
merged_labels = merge_label(labels, sids)
test_labels = t['label'][:]
test_events = t['event'][:]
test_sids = t['sample_id'][:]
merged_test_labels = merge_label(test_labels, test_sids)



nb_samples = len(labels)
nb_test = len(test_labels)
event_dim = events.max() + 1

count_events = [[0 for col in range(event_dim)] for row in range(nb_samples)]
count_test_events = [[0 for col in range(event_dim)] for row in range(nb_test)]

for i in xrange(0,nb_samples):
    for j in xrange(event_len):
        if events[i][j]<2:
            continue
        count_events[i][events[i][j]]+=1

for i in xrange(0,nb_test):
    for j in xrange(event_len):
        if test_events[i][j]<2:
            continue
        count_test_events[i][test_events[i][j]]+=1
        
        
clf = LogisticRegression()
clf.fit(count_events, labels)
labels_pred = clf.predict(count_test_events)
merged_labels_pred = merge_prob(labels_pred, test_sids, max)
p = np.mean(labels_pred == test_labels)
print("acc =",p)
p = np.mean(merged_labels_pred == merged_test_labels)
print "merged acc =", p

c = clf.predict_proba(count_test_events)
merged_c = merge_prob(c[:, 1], test_sids, max)
auc_value = roc_auc_score(test_labels,c[:,1])
print("auc =", auc_value)
merged_auc = roc_auc_score(merged_test_labels, merged_c)
print "merged auc =", merged_auc

