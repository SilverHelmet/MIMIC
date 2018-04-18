from __future__ import division

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from gcn import GraphAttention
from gcn.utils import load_data
import numpy as np

# Read data
A_train, X_train, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = load_data('cora')
A_train = A_train.toarray()
X_train = X_train.toarray()
# Y_train = Y_train.toarray()
# Y_val = Y_val.toarray()
# Y_test = Y_test.toarray()

A_train = np.tile(A_train, (10, 1, 1))
X_train = np.tile(X_train, (10, 1, 1))
Y_train = np.tile(Y_train, (10, 1, 1))
Y_val = np.tile(Y_val, (10, 1, 1))
Y_test = np.tile(Y_test, (10, 1, 1))


# Parameters
N = X_train.shape[1]          # Number of nodes in the graph
F = X_train.shape[2]          # Original feature dimesnionality
n_classes = Y_train.shape[2]  # Number of classes
F_ = 8                        # Output dimension of first GraphAttention layer
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate applied to the input of GAT layers
l2_reg = 5e-4                 # Regularization rate for l2
learning_rate = 5e-3          # Learning rate for SGD
epochs = 2000                 # Number of epochs to run for
es_patience = 100             # Patience fot early stopping

# Preprocessing operations
# X_train /= X_train.sum(1).reshape(-1, 1)


# Model definition (as per Section 3.3 of the paper)
X = Input(shape = (N, F))
A = Input(shape = (N, N))

dropout1 = Dropout(dropout_rate)(X)
graph_attention_1 = GraphAttention(F_,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   mask_zero=True)([dropout1, A])
dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes,
                                   attn_heads=1,
                                   attn_heads_reduction='average',
                                   activation='softmax',
                                   kernel_regularizer=l2(l2_reg),
                                   mask_zero=True)([dropout2, A])


# model = Model(input = [X, A], output = graph_attention_1)
# out = model.predict([X_train, A_train])
# print out.shape
# Build model
model = Model(input=[X, A], output=graph_attention_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()

# Callbacks
# tb_callback = TensorBoard(batch_size=N)

# Train model

print X_train.shape
print A_train.shape
print Y_train.shape
model.fit(x = [X_train, A_train],
          y = Y_train,
          nb_epoch=10,
          batch_size=1,
          shuffle=False,  # Shuffling data means shuffling the whole graph
            )
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
