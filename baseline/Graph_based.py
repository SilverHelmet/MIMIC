import h5py
import math
import time
import numpy as np
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
        
#f = h5py.File('exper/emerg_urgent_train_100_2000_True.h5', 'r')
# t = h5py.File('exper/emerg_urgent_test_100_2000_True.h5', 'r')
add_feature = False
# dataset_dir = death_exper_dir
# files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']
f = h5py.File('death_exper/death_train_1000.h5', 'r')
t = h5py.File('death_exper/death_test_1000.h5', 'r')
labels = f['label'][:]
events = f['event'][:]
features = f['feature'][:]
times = f['time'][:]
event_len = events.shape[1]
sids = f['sample_id'][:]
#merged_labels = merge_label(labels, sids)
test_labels = t['label'][:]
test_events = t['event'][:]
test_features = t['feature'][:]
test_times = t['time'][:]
test_sids = t['sample_id'][:]
#merged_test_labels = merge_label(test_labels, test_sids)

print labels.shape

nb_samples = len(labels)
nb_test = len(test_labels)
event_dim = events.max() + 1
feature_dim = 649

nb_samples = 500
nb_test = 500
labels = labels[0:nb_samples]
events = events[0:nb_samples]
features = features[0:nb_samples]
times = times[0:nb_samples]
sids = sids[0:nb_samples]

test_labels = test_labels[0:nb_test]
test_events = test_events[0:nb_test]
test_features = test_features[0:nb_test]
test_times = test_times[0:nb_test]
test_sids = test_sids[0:nb_test]

print labels.shape

count_events = [[0 for col in range(event_dim * event_dim)] for row in range(nb_samples)]
count_test_events = [[0 for col in range(event_dim * event_dim)] for row in range(nb_test)]

print event_dim
print 'data transform'

model = PCA(n_components = 500)
r = 2000000
for i in xrange(0,nb_samples):
    print i
    graph = np.zeros((event_dim, event_dim))
    for j in xrange(event_len):
        if events[i][j]<2:
            continue
        for k in xrange(event_len):
            if (j >= k):
                continue
            if events[i][k]<2:
                continue
            t1 = time.mktime(time.strptime(times[i][j], '%Y-%m-%d %H:%M:%S'))
            t2 = time.mktime(time.strptime(times[i][k], '%Y-%m-%d %H:%M:%S'))
            graph[events[i][j]][events[i][k]]+= (math.e ** ((t1 - t2)/r))
            #print (math.e ** ((t1 - t2)/r))/event_dim
            #print t1, t2, math.e ** ((t1 - t2)/r)
            #print events[i][j], events[i][k]
    #print graph[316][325], 'fuck'
    graph = np.reshape(np.transpose(graph), (-1))
    count_events[i] = graph
    #temp = model.fit_transform(graph)

print 'NG'

for i in xrange(0,nb_test):
    print i
    graph = np.zeros((event_dim, event_dim))
    for j in xrange(event_len):
        if events[i][j]<2:
            continue
        for k in xrange(event_len):
            if (j >= k):
                continue
            if events[i][k]<2:
                continue
            t1 = time.mktime(time.strptime(times[i][j], '%Y-%m-%d %H:%M:%S'))
            t2 = time.mktime(time.strptime(times[i][k], '%Y-%m-%d %H:%M:%S'))
            graph[events[i][j]][events[i][k]]+= (math.e ** ((t1 - t2)/r))/event_dim
    graph = np.reshape(np.transpose(graph), (-1))
    count_test_events[i] = graph
    #temp = model.fit_transform(graph)

count_events = np.array(count_events)
count_test_events = np.array(count_test_events)

count_events = model.fit_transform(count_events)
count_test_events = model.fit_transform(count_test_events)

print 'data ready'

print "X1 shape = ", count_events.shape
print "X2 shape = ", count_test_events.shape
        
        
clf = LinearSVC(verbose = True)
clf.fit(count_events, labels)
labels_pred = clf.predict(count_test_events)
c = labels_pred

p = np.mean(labels_pred == test_labels)
print("acc =",p)
auROC = roc_auc_score(test_labels,c)
print("auROC =", auROC)
precision, recall,  thresholds = precision_recall_curve(test_labels, c)
auPRC = auc(recall, precision)
print ("auPRC =", auPRC)

def merge_prob(labels, ids):
    la_len = len(ids)
    matrix = np.zeros((200000))
    for i in range(la_len):
        if labels[i] > matrix[ids[i]]:
            matrix[ids[i]] = labels[i]
    lab = np.zeros((la_len))
    for i in range(la_len):
        lab[i] = matrix[ids[i]]
    return lab

merged_labels_pred = merge_prob(labels_pred, test_sids)
merged_test_labels = merge_prob(test_labels, test_sids)
merged_c = merge_prob(c, test_sids)

p = np.mean(merged_labels_pred == merged_test_labels)
print "merged acc =", p
merged_auROC = roc_auc_score(merged_test_labels, merged_c)
print "merged auROC =", merged_auROC
precision, recall, thresholds = precision_recall_curve(merged_test_labels, merged_c)
merged_auPRC = auc(recall, precision)
print "merged auPRC =", merged_auPRC



