# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:18:25 2016

@author: wenzhang
"""

from keras.models import  Sequential
from keras.layers.recurrent import LSTM
import numpy as np
from  keras.layers.core import RepeatVector, TimeDistributedDense, Activation, Dropout, Dense, Masking, Activation
from keras.layers import Embedding, Input
from keras.optimizers import Adam, SGD
from keras.regularizers import l2, l1l2, l1
from sklearn.metrics import roc_auc_score
import h5py
#np.random.seed(0)
from keras.utils.np_utils import to_categorical


def sample_generator(events, labels, batch_size):
    st = 0
    nb_samples = labels.size
    while st < nb_samples:
        ed = min(st + batch_size, nb_samples)
        nb = ed - st
        output = np.zeros([nb, 1])
        event_em = np.zeros([nb, length, 3391])
        for i in range(st, ed):
            idx = i - st
            if(labels[i][0] == 1):
                output[idx][0] = 1
            for j in range(length):
                for k in range(j*em, j*em + em):
                    if int(events[i][k]) >= 2:
                        event_em[idx][j][int(events[i][k])] = 1
        
        yield(event_em, output)
        st = ed
        if st == nb_samples:
            st = 0

em = 20
# load train data and scaling
# file = h5py.File('C:\\Users\\wenzhang\\Desktop\\haichao\\train_100_1000_True.h5','r')  
file = h5py.File("exper/train_100_1000_True.h5", 'r')
label = file['label'][:]
event = file['event'][:]
train_label = np.array([label], dtype = 'int32').T
train_event = np.array(event, dtype = 'float64')
print (1000/em)+1
length = (1000/em)

#load test data and scaling
# file = h5py.File('C:\\Users\\wenzhang\\Desktop\\haichao\\test_100_1000_True.h5','r')  
file = h5py.File("exper/test_100_1000_True.h5", 'r')
label = file['label'][:]
event = file['event'][:]
test_label = np.array([label], dtype = 'int32').T
test_event = np.array(event, dtype = 'float64')

hidden_size = 128    
embedding_dim = 128
#layers = 1
test_result = np.zeros([8909, 1])
print "finish"
w_reg = None
b_reg = None
w_reg = l2(0.0001)
b_reg = l2(0.0001)
pre = np.zeros([8909,1])
model = Sequential()
model.add(Masking(mask_value=0., input_shape = (1000/em, 3391)))
# model.add(Embedding(input_dim=3391, output_dim=embedding_dim, input_length = length))
# model.add(TimeDistributedDense(input_dim = 3391, output_dim = embedding_dim , name = 'seg_event_embedding', init = "uniform",
        # bias = False))
model.add(LSTM(input_dim = 3391, activation='sigmoid', inner_activation='hard_sigmoid', 
    input_length = None, output_dim = hidden_size,
    W_regularizer = w_reg, b_regularizer = b_reg ))
#return_sequences=True,
# model.add(Dropout(0.5))
#model.add(TimeDistributedDense(1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = Adam(lr=0.001)   # 可以调节的变量
# opt = SGD(lr = 0.03, decay = 0.03)
print "optimizer config"
print opt.get_config()
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print "train size = %d" %train_label.size
print "test size = %d" %test_label.size
a=model.get_config() 
print a
batch_size = 32
for t in range(30):
    print('-' * 50)
    print('Iteration ', t+1)
    nb_samples = train_label.size
    model.fit_generator(sample_generator(train_event, train_label, batch_size), samples_per_epoch = nb_samples, nb_epoch = 1)

    # mini-batch  training
    # st = 0
    # while st < 35638:
    #     ed = min(35638, st + batch_size)
    #     nb = ed - st
    #     train_output = np.zeros([nb, 1])
    #     train_event_em = np.zeros([nb, length, 3391])
    #     for i in range(st, ed):
    #         idx = i - st
    #         if(train_label[i][0] == 1):
    #             train_output[idx][0] = 1
    #         for j in range(length):
    #             for k in range(j*em, j*em + em):
    #                 train_event_em[idx][j][int(train_event[i][k])] = 1
    #     model.fit(train_event_em, train_output, batch_size = batch_size, nb_epoch = 1, verbose = 0)
    #     st = ed
    #     if st / 32 % 100 == 0: 
    #         print "\t iter = %d %d - %d" %(t+1, st, ed)

    # stochasitc training
    # for i in range(35638): #35638
    #     train_output = np.zeros([1, 1])
    #     train_event_em = np.zeros([1, length, 3391])  #35638
    #     if (train_label[i][0] == 1):
    #         train_output[0][0] = 1
    #     for j in range(length):
    #         for k in range(j*em, j*em + em):
    #             if (k < 1000):
    #                 train_event_em[0][j][int(train_event[i][k])] = 1
    #     model.fit(train_event_em, train_output, batch_size=128, nb_epoch=1, verbose = 0)
    #     if i % 1000 == 0:
	# 	print "\t iter = %d, sample = %d", %(t+1, i)

    # for i in range(8909):    #8909
    #     test_event_em = np.zeros([1, length, 3391])   #8909
    #     for j in range(length):
    #         for k in range(j*em, j*em + em):
    #             if (k < 1000):
    #                 test_event_em[0][j][int(test_event[i][k])] = 1
    #     classes = model.predict(test_event_em)
      
    #     pre[i][0]=classes[0][0]
    #     if (i % 1000 == 908):
    #         print i
    # print "old AUC =", roc_auc_score(label, pre)
    # print "old AUC =", roc_auc_score(test_label, pre)
    predictions = model.predict_generator(generator = sample_generator(test_event, test_label, batch_size), 
        val_samples = test_label.size)

    auc = roc_auc_score(test_label, predictions)
    print 'AUC =',auc

    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    acc = np.mean(test_label == predictions)
    print "ACC =", acc



'''
a=model.get_config() 
print("model config is ",a)
#模型和参数的保存和加载

json_string = model.to_json()
open('my_model_architecture.json','w').write(json_string)    
model.save_weights('my_model_weights.h5')    
#加载模型数据和weights  
model = model_from_json(open('my_model_architecture.json').read())    
model.load_weights('my_model_weights.h5')  
'''
