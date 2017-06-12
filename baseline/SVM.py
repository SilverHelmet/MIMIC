import h5py
import numpy as np
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.svm import LinearSVC, SVC

        
#f = h5py.File('exper/emerg_urgent_train_100_2000_True.h5', 'r')
# t = h5py.File('exper/emerg_urgent_test_100_2000_True.h5', 'r')
# dataset_dir = death_exper_dir
# files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']
f = h5py.File('death_exper/death_train_1000.h5', 'r')
t = h5py.File('death_exper/death_test_1000.h5', 'r')
labels = f['label'][:]
events = f['event'][:]
event_len = events.shape[1]
test_labels = t['label'][:]
test_events = t['event'][:]

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

count_events = np.array(count_events)
count_test_events = np.array(count_test_events)

print 'data load'        
clf = LinearSVC(verbose = True)
clf.fit(count_events, labels)
labels_pred = clf.predict(count_test_events)
print '- -'
c = labels_pred

print 'data trained'
p = np.mean(labels_pred == test_labels)
print("acc =",p)
auROC = roc_auc_score(test_labels,c)
print("auROC =", auROC)
precision, recall,  thresholds = precision_recall_curve(test_labels, c)
auPRC = auc(recall, precision)
print ("auPRC =", auPRC)

