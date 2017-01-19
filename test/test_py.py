from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import numpy as np

class A:
    def __init__(self, pid):
        self.pid = pid

    def ok(self):
        print self.pid
        self.ok2()

    def ok2(self):
        print "A"
    
class B(A):
    def __init__(self, pid):
        self.pid = pid

    def ok2(self):
        print "B"

y_label = np.random.randint(0, 2, (1000000,))
y_pred = np.random.random((1000000,))
auc_value = roc_auc_score(y_label, y_pred)
print auc_value
precision, recall, thredholds = precision_recall_curve(y_label, y_pred)
# print auc(precision, recall, reorder=True)
print auc(recall, precision)