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
feature_dim = 3
e_input = Input(shape = (max_segs, max_seg_length))
emdd = SegMaskEmbedding(mask_value = 0, input_dim = event_dim, output_dim = embedding_dim, name = "embedding")
emd = emdd(e_input)
# emd_model = Model(input = e_input, output = emd)

# f_input = Input(shape = (max_segs, max_seg_length, feature_dim))
# feature_layer = TimeDistributed(TimeDistributedDense(output_dim = embedding_dim), name = 'feature_embedding')(f_input)

# f_model = Model(input = f_input, output = feature_layer)
# merged = merge(inputs = [emd, feature_layer], mode = 'sum')
# merged_model = Model(input = [e_input, f_input], output = merged)
# merged_model.compile(optimizer = "adam", loss = 'binary_crossentropy')
# emd = Embedding(input_dim = event_dim, output_dim = embedding_dim, name = "embedding")(e_input)
rnnn = EventAttentionLSTM(att_hidden_dim = 8, output_dim = hidden_dim, inner_activation='sigmoid', activation='sigmoid', 
    input_length = None, return_sequences = False)
rnn = rnnn(emd)
# rnn_model = Model(input = e_input, output = rnn)
# rnn = LSTM(output_dim = hidden_dim, inner_activation = 'hard_sigmoid', activation='sigmoid', consume_less = 'gpu',
            # W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, input_length = None, return_sequences = attention)(embedding)

pred = Dense(1, activation = "sigmoid", name = 'prediction')(rnn)
model = Model(input = e_input, output = pred)
opt = Adam(lr = 0.1)
model.compile(optimizer = opt,
    loss = 'binary_crossentropy', 
        metrics=['accuracy'])
print "compile over"

data1 = np.array([[1,2,3,5,4],[3,1,0,0,0], [0,0,0,0,0]])
data2 = np.array([[3,1,2,1,1],[1,1,7,8,0], [7,8,7,8,7]])
f1 = np.random.randint(0, 2, (max_segs, max_seg_length, feature_dim))
f2 = np.random.randint(0, 2, (max_segs, max_seg_length, feature_dim))
label1 = 0
label2 = 1
data = np.array([data1, data2])
print model.predict(x =data)

for config in model.get_config()['layers']:
    print "\t", config
model.save("test.model")

custom_objects = {
    "SegMaskEmbedding": SegMaskEmbedding,
    "EventAttentionLSTM":EventAttentionLSTM
}
model2 = load_model("test.model", custom_objects = custom_objects)
print model2.predict(x = data)

# print model.predict(data)
# for i in range(10):
#     model.fit(data, np.array([label1, label2]), nb_epoch=10)
#     print model.predict(data)





