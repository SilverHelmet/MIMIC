from __future__ import absolute_import

from keras import backend as K
from keras import activations, constraints, regularizers, initializations
from keras.layers import Layer, Dropout, LeakyReLU
from models.models import hard_softmax


class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 attn_dropout=0.5,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 attn_kernel_constraint=None,
                 mask_zero=False,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.activation = activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)
        self.kernel_initializer = initializations.get(kernel_initializer)
        self.attn_kernel_initializer = initializations.get(attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.mask_zero = mask_zero
        if self.mask_zero:
            self.supports_masking = True
        else:
            self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]
        # Initialize kernels for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     name='kernel_%s' % head,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            # Attention kernel
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               name='att_kernel_{}'.format(head),
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 name='att_kernel_{}'.format(head),
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint)
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs, mask = None):
        X = inputs[0]  # Node features (batch_size X N x F)
        A = inputs[1]  # Adjacency matrix (batch_size X N x N)

        # Parameters
        N = K.shape(X)[1]  # Number of nodes in the graph

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            linear_transf_X = K.dot(X, kernel)  # (batch_size X N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(linear_transf_X, attention_kernel[0])    # (batch_size X N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(linear_transf_X, attention_kernel[1])  # (batch_size X N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            # dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting
            dense = K.reshape(attn_for_self, (-1, N, 1)) + K.reshape(attn_for_neighs, (-1, 1, N))  #(batch_size X N X N)

            # Add nonlinearty
            # dense = LeakyReLU(alpha=0.2)(dense)``
            dense = K.relu(dense, alpha=0.2)

            # Mask values before activation (Vaswani et al., 2017)
            comparison = K.equal(A, 0.0) # (batch_size X N X N)
            A_mask = K.switch(comparison, K.ones_like(A) * -10e9, K.zeros_like(A))
            masked = dense + A_mask

            # Feed masked values to softmax
            # softmax = K.softmax(masked)  # (batch_size X N x N), attention coefficients
            softmax = hard_softmax(masked)

            # dropout = Dropout(self.attn_dropout)(softmax)  # (batch_size X N x N)
            if 0. < self.attn_dropout < 1.:
                dropout = K.in_train_phase(K.dropout(softmax, self.attn_dropout, None), softmax)
            else:
                dropout = softmax


            # Linear combination with neighbors' features
            node_features = K.batch_dot(dropout, linear_transf_X)  # (batch_size X N x F')

            if self.attn_heads_reduction == 'concat' and self.activation is not None:
                # In case of 'concat', we compute the activation here (Eq 5)
                node_features = self.activation(node_features)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')
            if self.activation is not None:
                # In case of 'average', we compute the activation here (Eq 6)
                output = self.activation(output)

        return output

    def get_output_shape_for(self, input_shape):
        
        output_shape = input_shape[0][0], input_shape[0][1], self.output_dim
        return output_shape

    def compute_mask(self, x, mask=None):
        if self.mask_zero:
            return mask[0]
        else:
            return None