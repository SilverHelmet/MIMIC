import h5py
import numpy as np
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import NMF

        
#f = h5py.File('exper/emerg_urgent_train_100_2000_True.h5', 'r')
# t = h5py.File('exper/emerg_urgent_test_100_2000_True.h5', 'r')
add_feature = False
# dataset_dir = death_exper_dir
# files = ['ICUIn_train_1000.h5', 'ICUIn_valid_1000.h5', 'ICUIn_test_1000.h5']
f = h5py.File('ICU_merged_exper/ICUIn_train_1000.h5', 'r')
t = h5py.File('ICU_merged_exper/ICUIn_test_1000.h5', 'r')
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

nb_samples = 100000
nb_test = 20000
#labels = labels[0:nb_samples]
#events = events[0:nb_samples]
#features = features[0:nb_samples]
#times = times[0:nb_samples]
#sids = sids[0:nb_samples]

#test_labels = test_labels[0:nb_test]
#test_events = test_events[0:nb_test]
#test_features = test_features[0:nb_test]
#test_times = test_times[0:nb_test]
#test_sids = test_sids[0:nb_test]

print labels.shape

edim = 10
count_events = [[0 for col in range(edim)] for row in range(nb_samples)]
count_test_events = [[0 for col in range(edim)] for row in range(nb_test)]

count_labels = [0 for row in range(nb_samples)]
count_test_labels = [0 for row in range(nb_test)]

count_sids = [0 for row in range(nb_samples)]
count_test_sids = [0 for row in range(nb_test)]

print event_dim
print 'data transform'
model = NMF(n_components = 10)
i = 0
smp_sum = 0
while smp_sum < nb_samples:
    print i, smp_sum
    dnum = 1
    for j in xrange(event_len):
        if j == 0:
            continue
        if times[i][j] == times[i][j - 1]:
            continue
        dnum += 1
    if (dnum < 10):
        i = i + 1
        continue
    tmp_events = np.zeros((event_dim, dnum))
    cnum = 0
    for j in xrange(event_len):
        if events[i][j]<2:
            continue
        if j != 0 and times[i][j] != times[i][j - 1]:
            cnum += 1
        tmp_events[events[i][j]][cnum]+=1
    tmp_events = np.transpose(tmp_events)
    print tmp_events.shape
    temp = model.fit_transform(tmp_events)
    #print temp.shape
    count_events[smp_sum] = np.mean(temp, axis = 0)
    count_labels[smp_sum] = labels[i]
    count_sids[smp_sum] = sids[i]
    #print temp.shape
    #print tmp_events.shape
    i = i + 1
    smp_sum = smp_sum + 1

print 'NG'

i = 0
smp_sum = 0
while smp_sum < nb_test:
    print i, smp_sum
    dnum = 1
    for j in xrange(event_len):
        if j == 0:
            continue
        if test_times[i][j] == test_times[i][j - 1]:
            continue
        dnum += 1
    if dnum < 10:
        i = i + 1
        continue
    tmp_test_events = np.zeros((event_dim, dnum))
    cnum = 0
    for j in xrange(event_len):
        if test_events[i][j]<2:
            continue
        if j != 0 and test_times[i][j] != test_times[i][j - 1]:
            cnum += 1
        tmp_test_events[test_events[i][j]][cnum]+=1
    tmp_events = np.transpose(tmp_events)
    temp = model.fit_transform(tmp_test_events)
    #print i
    count_test_events[smp_sum] = np.mean(temp, axis = 0)
    count_test_labels[smp_sum] = test_labels[i]
    count_test_sids[smp_sum] = test_sids[i]
    #print temp.shape
    i = i + 1
    smp_sum = smp_sum + 1

count_events = np.array(count_events)
count_test_events = np.array(count_test_events)

labels = count_labels
test_labels = count_test_labels

sids = count_sids
test_sids = count_test_sids

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



