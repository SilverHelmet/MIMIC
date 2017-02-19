from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import sklearn.svm
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.figure(0)
ax = plt.gca()
s = "123"
plt.text(1,2.5 ,"123\n12344", horizontalalignment = 'left', verticalalignment = "bottom", fontsize = 13)
plt.bar([1,2, 3], [2,3,4], width=0.4)
plt.grid()
plt.show()



plt.close(0)