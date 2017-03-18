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


        
# dataset_dir = ICU_exper_dir
# files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']
# dataset_dir=death_merged_exper_dir
# files = ["death_train_1000.h5", "death_valid_1000.h5", "death_test_1000.h5"]
add_feature = False
# dataset_dir = death_exper_dir
# files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']

dataset_dir = "zhu_exper"
files = ['train.h5', 'valid.h5', 'test.h5']
files = ['train_catAtt.h5', 'valid_catAtt.h5', 'test_catAtt.h5']
f = h5py.File(os.path.join(dataset_dir, files[0]), 'r')
t = h5py.File(os.path.join(dataset_dir, files[2]), 'r')
labels = f['label'][:]
events = f['event'][:]
if "feature" in f:
    features = f['feature'][:]
else:
    features = None
event_len = events.shape[1]
sids = f['sample_id'][:]
merged_labels = merge_label(labels, sids)
test_labels = t['label'][:]
test_events = t['event'][:]
if "feature" in t:
    test_features = t['feature'][:]
else:
    test_features = None
test_sids = t['sample_id'][:]
merged_test_labels = merge_label(test_labels, test_sids)



nb_samples = len(labels)
nb_test = len(test_labels)
event_dim = events.max() + 1
feature_dim = 19

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

count_events = np.array(count_events)
count_test_events = np.array(count_test_events)

def add_feature_cnts(event_cnts, features, feature_dim):
    global event_len
    nb_samples = event_cnts.shape[0]
    feature_cnts = [[0] * feature_dim for i in range(nb_samples)]
    feature_length = features.shape[2]
    for i in xrange(nb_samples):
        feature =features[i]
        for j in xrange(event_len):
            feature_pair = feature[j]
            idx = 0
            while idx < feature_length:
                if feature_pair[idx+1] != 0:
                    index = int(feature_pair[idx])
                    value = feature_pair[idx+1]
                    feature_cnts[i][index] += value
                idx += 2
    feature_cnts = np.array(feature_cnts)
    return np.concatenate([event_cnts, feature_cnts], axis = 1)
                

if add_feature:
    count_events = add_feature_cnts(count_events, features, feature_dim)
    count_test_events = add_feature_cnts(count_test_events, test_features, feature_dim)

print "X1 shape = ", count_events.shape
print "X2 shape = ", count_test_events.shape
        
        
clf = LogisticRegression()
clf.fit(count_events, labels)
labels_pred = clf.predict(count_test_events)
c = clf.predict_proba(count_test_events)

result = []
p = np.mean(labels_pred == test_labels)
result.append(p)
print("acc =",p)
auROC = roc_auc_score(test_labels,c[:,1])
result.append(auROC)
print("auROC =", auROC)
precision, recall,  thresholds = precision_recall_curve(test_labels, c[:, 1])
auPRC = auc(recall, precision)
result.append('auPRC')
print ("auPRC =", auPRC)

merged_labels_pred = merge_prob(labels_pred, test_sids, max)
merged_test_labels = merge_label(test_labels, test_sids)
merged_c = merge_prob(c[:, 1], test_sids, max)

p = np.mean(merged_labels_pred == merged_test_labels)
result.append(p)
print "merged acc =", p
merged_auROC = roc_auc_score(merged_test_labels, merged_c)
result.append(merged_auROC)
print "merged auROC =", merged_auROC
precision, recall, thresholds = precision_recall_curve(merged_test_labels, merged_c)
merged_auPRC = auc(recall, precision)
result.append(merged_auPRC)
print "merged auPRC =", merged_auPRC

from models.dataset import print_eval
print_eval("result", map(result, float))


