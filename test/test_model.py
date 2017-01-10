import test_util
import numpy as np
from models.models import *
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Masking, Merge, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Input, merge
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam 
import theano




event_dim = 10
embedding_dim = 10
max_segs = 3
max_seg_length = 5
hidden_dim = 7
e_input = Input(shape = (max_segs, max_seg_length))
emdd = SegMaskEmbedding(mask_value = 0, input_dim = event_dim, output_dim = embedding_dim, name = "embedding")
emd = emdd(e_input)
emd_model = Model(input = e_input, output = emd)
# emd = Embedding(input_dim = event_dim, output_dim = embedding_dim, name = "embedding")(e_input)
rnnn = EventAttentionLSTM(att_hidden_dim = 8, output_dim = hidden_dim, inner_activation='sigmoid', activation='sigmoid', 
    input_length = None, return_sequences = False)
rnn = rnnn(emd)
rnn_model = Model(input = e_input, output = rnn)

pred = Dense(1, activation = "sigmoid", name = 'prediction')(rnn)
model = Model(input = e_input, output = pred)
opt = Adam(lr = 0.1)
model.compile(optimizer = opt,
    loss = 'binary_crossentropy', 
        metrics=['accuracy'])

print "compile over"
data1 = np.array([[1,2,3,5,4],[3,1,0,0,0], [0,0,0,0,0]])
data2 = np.array([[3,1,2,1,1],[1,1,7,8,0], [7,8,7,8,7]])
label1 = 0
label2 = 1
data = np.array([data1, data2])
print model.predict(data)
for i in range(10):
    model.fit(data, np.array([label1, label2]), nb_epoch=10)
    print model.predict(data)





