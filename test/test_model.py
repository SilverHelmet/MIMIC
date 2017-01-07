import test_util
import numpy as np
from models.models import *
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Masking, Merge, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Input, merge
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam 


event_dim = 10
embedding_dim = 10
max_segs = 3
max_seg_length = 5
hidden_dim = 7
e_input = Input(shape = (max_segs, max_seg_length))
emd = SegMaskEmbedding(mask_value = 0, input_dim = event_dim, output_dim = embedding_dim, name = "embedding")(e_input)
print "emd"
print emd
emd_model = Model(input = e_input, output = emd)
emd = Embedding(input_dim = event_dim, output_dim = embedding_dim, name = "embedding")(e_input)
rnn = EventAttentionLSTM(att_hidden_dim = 8, output_dim = hidden_dim, inner_activation='hard_sigmoid', activation='sigmoid', 
    input_length = None, return_sequences = False)(emd)
# rnn_model = Model(input = e_input, output = rnn)

# pred = Dense(1, activation = "sigmoid", name = 'prediction')(rnn)
# model = Model(input = e_input, output = pred)
# opt = Adam(lr = 0.001)
# rnn_model.compile(optimizer = opt,
#     loss = 'binary_crossentropy', 
#         metrics=['accuracy'])

data1 = np.array([[1,2,3,5,4],[3,1,0,0,0], [0,0,0,0,0]])
label1 = 0
data = np.array([data1])
# y = rnn_model.predict(np.array([data1]))
# print y
# print y.shape




