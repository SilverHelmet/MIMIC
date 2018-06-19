from __future__ import absolute_import

from keras import backend as K
from keras import activations, constraints, regularizers, initializations
from keras.layers import Layer, Dropout, LeakyReLU
from models.models import hard_softmax


class GraphAttention(Layer):

    def __init__(self,
                 F1,
                 F2 = 0,
                 input_dim = 0,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 attn_dropout=0.5,
                 activation='tanh',
                 kernel_initializer='glorot_uniform',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 attn_kernel_constraint=None,
                 mask_zero=False,
                 attention_mode = 0,
                 **kwargs):
        '''
        attention_mode
            h: node embedding
            -1: h = sigma ai * ti, ai = (t0, t1) .* w2, ti = hi * w1
            0 : h = sigma ai * hi, ai = (h0, h1) .* w1
            *1: h = sigma ai * hi, ai = (h0, h1) * w1 * w2
            2 : h = (h0, h.0)
            3 : h' = h.2 * w1
            4 : h = (h0, h0.-1)
            5 : h' = h.4 * w3 
        '''
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.base_config = {
            'F1': F1,
            'F2': F2,
            'input_dim': input_dim,
            'attn_heads': attn_heads,
            'attn_heads_reduction': attn_heads_reduction,
            "attn_dropout": attn_dropout,
            "activation": activation,
            "kernel_initializer": kernel_initializer, 
            "attn_kernel_initializer": attn_kernel_initializer,
            "kernel_regularizer": kernel_regularizer,
            "attn_kernel_regularizer": attn_kernel_regularizer,
            "activity_regularizer": activity_regularizer,
            "kernel_constraint": kernel_constraint,
            "attn_kernel_constraint": kernel_constraint,
            "mask_zero": mask_zero,
            "attention_mode": mask_zero,
        }
        self.F1 = F1  # Number of output features (F' in the paper)
        self.F2 = F2
        self.input_dim = input_dim
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
        self.mode = attention_mode

        if self.mask_zero:
            self.supports_masking = True
        else:
            self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if self.mode == -1:
            output_dim = self.F1
        elif self.mode == 0:
            output_dim = self.input_dim
        elif self.mode == 1:
            output_dim = self.F1
        elif self.mode == 2:
            output_dim = self.input_dim * 2
        elif self.mode == 3:
            output_dim = self.F1
        elif self.mode == 4:
            output_dim = self.input_dim + self.F1
        elif self.mode == 5:
            output_dim = self.F2


        if attn_heads_reduction == 'concat':
            
            # Output will have shape (..., K * F')
            self.output_dim = output_dim * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = output_dim
         


        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]
        # Initialize kernels for each attention head
        for head in range(self.attn_heads):
            if self.mode == -1 or self.mode == 5:
            # Layer kernel
                kernel0 = self.add_weight(shape=(F, self.F1),
                                        initializer=self.kernel_initializer,
                                        name='kernel_%s_0' % head,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
                kernel = [kernel0]
                if self.mode == 5:
                    kernel1 = self.add_weight(shape=(F + F1 , self.F1),
                                        initializer=self.kernel_initializer,
                                        name='kernel_%s_1' % head,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
                    kernel.append(kernel1)
                self.kernels.append(kernel)
            elif self.mode == 3:
                kernel0 = self.add_weight(shape=(F + F, self.F1),
                                        initializer=self.kernel_initializer,
                                        name='kernel_%s_0' % head,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
                self.kernels.append([kernel0])

            # Attention kernel
            shapes = []
            if self.mode in [-1, 4, 5]:
                shapes.append((self.F1, 1))
                shapes.append((self.F1, 1))
            elif self.mode == 0:
                shapes.append((F, 1))
                shapes.append((F, 1))
            elif self.mode == 2:
                shapes.append((F, 1))
                shapes.append((F, 1))
            elif self.mode == 3:
                shapes.append((F, 1))
                shapes.append((F, 1))


            attn_kernel = []
            for idx, shape in enumerate(shapes):
                w = self.add_weight(shape=shape,
                                               initializer=self.attn_kernel_initializer,
                                               name='att_kernel_{}_{}'.format(head, idx),
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint)
                attn_kernel.append(w)
            self.attn_kernels.append(attn_kernel)
        self.built = True

    def call_mode0(self, X, A, attn_kernel, kernel, N, use_kernel = False):
        # Compute inputs to attention network
        if use_kernel:
            linear_transf_X = K.dot(X, kernel[0])  # (batch_size X N x F')
        else:
            linear_transf_X = X

        attn_for_self = K.dot(linear_transf_X, attn_kernel[0])    # (batch_size X N x 1), [a_1]^T [Wh_i]
        attn_for_neighs = K.dot(linear_transf_X, attn_kernel[1])  # (batch_size X N x 1), [a_2]^T [Wh_j]
        
        

        # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
        # dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting
        dense = K.reshape(attn_for_self, (-1, N, 1)) + K.reshape(attn_for_neighs, (-1, 1, N))  #(batch_size X N X N)

        # Add nonlinearty
        # dense = LeakyReLU(alpha=0.2)(dense)
        dense = K.relu(dense, alpha=0.2)

        # Mask values before activation (Vaswani et al., 2017)
        comparison = K.equal(A, 0.0) # (batch_size X N X N)
        A_mask = K.switch(comparison, K.ones_like(A) * -10e9, K.zeros_like(A))
        masked = dense + A_mask

        # Feed masked values to softmax
        # softmax = K.softmax(masked)  # (batch_size X N x N), attention coefficients
        softmax = hard_softmax(masked)

        # Linear combination with neighbors' features
        node_features = K.batch_dot(softmax, linear_transf_X)  # (batch_size X N x F')
        if self.attn_heads_reduction == 'concat' and self.activation is not None:
            node_features = self.activation(node_features)
        return node_features

    def call_mode2(self, X, A, attn_kernel, N):
        neigh_features = self.call_mode0(X, A, attn_kernel, None, N, False)
        node_features =  K.concatenate([X, neigh_features])
        return node_features

    def call_mode3(self, X, A, attn_kernel, kernel, N):
        hidden_node_feature = self.call_mode2(X, A, attn_kernel, N)
        node_features = K.dot(hidden_node_feature, kernel[0])
        if self.attn_heads_reduction == 'concat' and self.activation is not None:
            node_features = self.activation(node_features)
        return node_features

    def call_mode4(self, X, A, attn_kernel, kernel, N):
        T = self.call_mode0(X, A, attn_kernel, kernel, N, True)
        node_features = K.concatenate([X, T])
        return node_features

    def call_mode5(self, X, A, attn_kernel, kernel, N):
        node_hidden_features = self.call_mode4(X, A, attn_kernel, kernel, N)
        node_features = K.dot(node_hidden_features, kernel[1])
        if self.attn_heads_reduction == 'concat' and self.activation is not None:
            node_features = self.activation(node_features)
        return node_features



    def call(self, inputs, mask = None):
        X = inputs[0]  # Node features (batch_size X N x F)
        A = inputs[1]  # Adjacency matrix (batch_size X N x N)

        # Parameters
        N = K.shape(X)[1]  # Number of nodes in the graph

        outputs = []
        for head in range(self.attn_heads):
            # kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            if self.mode == -1:
                node_features = self.call_mode0(X, A, attention_kernel, self.kernels[head], N, use_kernel = True)
            elif self.mode == 0:
                node_features = self.call_mode0(X, A, attention_kernel, None, N, use_kernel = False)
            elif self.mode == 2:
                node_features = self.call_mode2(X, A, attention_kernel, N)
            elif self.mode == 3:
                node_features = self.call_mode3(X, A, attention_kernel, self.kernels[head], N)
            elif self.mode == 4:
                node_features = self.call_mode4(X, A, attention_kernel, self.kernels[head], N)
            elif self.mode ==5 :
                node_features = self.call_mode5(X, A, attention_kernel, self.kernels[head], N)


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

    def get_config(self):
        base_config = super(GraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(self.config.items()))