from models.dataset import Dataset
import numpy as np
from graph import build_time_graph
from keras.layers import Input, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from gcn.graph_attention_layer import GraphAttention

s_dataset = Dataset('death_exper/sample/samples.h5', 'death_exper/sample/samples_seg.h5')
s_dataset.load(True, True)
s_dataset.print_shape()
s_dataset.trans_time()

n = 1000
a = build_time_graph(s_dataset.times[1], 0.5)

x = np.random.randint(0, 10, (3, n))
a = np.tile(a, (3,1,1))
print x.shape
print a.shape

X = Input(shape=(n, ))
A = Input(shape=(n, n))

embedding_layer = Embedding(input_dim = 10, output_dim = 20, mask_zero = True, name = 'embedding')
embedding = embedding_layer(X)
graph_attention_1 = GraphAttention(20,
                                   attn_heads=2,
                                   attn_heads_reduction='concat',
                                   activation='elu')([embedding, A])


model = Model(input = [X, A], output = embedding)
emd = model.predict([x, a])
mask = np.not_equal(x, 0)


