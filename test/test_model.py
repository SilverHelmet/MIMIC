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
from get_attention import *
from sklearn.decomposition import PCA




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
    input_length = None, return_sequences = True, name = 'rnn')
rnn = rnnn(emd)
rnn = SimpleAttentionRNN(rnn)
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

# for config in model.get_config()['layers']:
    # print "\t", config


# print model.predict(data)
# for i in range(2):
#     model.fit(data, np.array([label1, label2]), nb_epoch=10, verbose = 0)
#     print model.predict(data)

# model.save("test.model")
model2 = load_model("test.model", custom_objects = get_custom_objects())
name = "embedding"
data = np.array([data2])
att = get_event_attention(model2, data)[0]
print att
print "-----"
embeddings = get_embedding(model2, [6,7,8,7,8,7], name = 'embedding')
output = get_event_output(model2, x = np.array(data[0:2,:]))
att = get_event_attention_at_seg(model2, output, embeddings)
print att
print np_softmax(np.array([att[1:]]))

# print model2.predict(x = data)
# model3 = Model(input = model2.input, output = model2.get_layer("embedding").output)
# emd_out =  model3.predict(x = data)
# mask = np.any(np.not_equal(emd_out, 0.0), axis=-1)
# mask = np.any(np.not_equal(mask, 0.0), axis=-1)
# print mask
# layer = model2.get_layer("rnn")
# x1 = layer.test_call(emd_out, mask)
# model4 = Model(input = model2.input, output = layer.output)
# x2 = model4.predict(x = data)
# print x1
# print "-" * 50
# print x2





