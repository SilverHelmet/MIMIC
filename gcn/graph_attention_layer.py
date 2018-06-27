from __future__ import absolute_import

from keras import backend as K
from keras import activations, constraints, regularizers, initializations
from keras.initializations import zero
from keras.layers import Layer, Dropout, LeakyReLU
from models.models import hard_softmax
import numpy as np 


class GraphAttention(Layer):

    def __init__(self,
                 F1,
                 F2 = 0,
                 input_dim = 0,
                 nb_event = 0,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 attn_dropout=0.5,
                 activation='tanh',
                 kernel_initializer='glorot_uniform',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                #  kernel_constraint=None,
                #  attn_kernel_constraint=None,
                 mask_zero=False,
                 attention_mode = 0,
                 **kwargs):
        '''
        attention_mode
            h: node embedding
            -2: h = sigma ai * ti, ai = (t0, t1) .* w2.h, ti = hi * w1
            -1: h = sigma ai * ti, ai = (t0, t1) .* w2, ti = hi * w1
            0 : h = sigma ai * hi, ai = (h0, h1) .* w1
            *1: h = sigma ai * hi, ai = (h0, h1) * w1 * w2
            2 : h = (h0, h.0)
            3 : h' = h.2 * w1
            4 : h = (h0, h0.-1)
            5 : h' = h.4 * w3
            6 : h = (h0, h0.-1_0, h0.-1_1, ..., h0.-1_#head) 
            7 : h' = h.6 * w3
            8 : h = (h0, h0.-2_0, h0.-2_0,, ..., h0.-2_#head)
            9 : h = h.-1 * w3 + b
        '''
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F1 = F1  # Number of output features (F' in the paper)
        self.F2 = F2
        self.nb_event = nb_event
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
        # self.kernel_constraint = constraints.get(kernel_constraint)
        # self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.mask_zero = mask_zero
        self.mode = attention_mode

        if self.mask_zero:
            self.supports_masking = True
        else:
            self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads
        self.constant_kernels = []

        if self.mode in [-1, -2]:
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
        elif self.mode in [5, 9]:
            output_dim = self.F2
        elif self.mode in [6, 8]:
            output_dim = self.input_dim + self.attn_heads * self.F1


        if self.mode in [6, 8]:
            self.output_dim = output_dim
        elif attn_heads_reduction == 'concat':
            
            # Output will have shape (..., K * F')
            self.output_dim = output_dim * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = output_dim
         
        

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]
        l = input_shape[0][-2]
        # Initialize kernels for each attention head
        for head in range(self.attn_heads):
            if self.mode in [-2, -1, 4, 5, 6, 8]:
            # Layer kernel
                kernel0 = self.add_weight(shape=(F, self.F1),
                                        initializer=self.kernel_initializer,
                                        name='kernel_%s_0' % head,
                                        regularizer=self.kernel_regularizer)
                kernel = [kernel0]
                if self.mode in [5]:
                    kernel1 = self.add_weight(shape=(F + self.F1 , self.F2),
                                        initializer=self.kernel_initializer,
                                        name='kernel_%s_1' % head,
                                        regularizer=self.kernel_regularizer)
                    kernel1_b = self.add_weight(shape=(self.F2,),
                                        initializer='zero',
                                        name='kernel_%s_1_b' % head,
                                        regularizer=self.kernel_regularizer)
                    kernel.append(kernel1)
                    kernel.append(kernel1_b)
                self.kernels.append(kernel)
            elif self.mode == 3:
                kernel0 = self.add_weight(shape=(F + F, self.F1),
                                        initializer=self.kernel_initializer,
                                        name='kernel_%s_0' % head,
                                        regularizer=self.kernel_regularizer)
                self.kernels.append([kernel0])


            # Attention kernel
            shapes = []
            if self.mode in [-2, 8]:
                shapes.append((self.nb_event, self.F1))
            elif self.mode in [-1, 4, 5, 6, 9]:
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
                                               regularizer=self.attn_kernel_regularizer)
                attn_kernel.append(w)


            self.attn_kernels.append(attn_kernel)

        if self.mode in [-2, 8]:
            def my_init(shape, name = None):
                return K.variable(range(shape[0]), dtype = 'int32', name = name)

            w = self.add_weight(shape = (l, ),
                                initializer = my_init,
                                name = 'constant_transform_w',
                                trainable = False)
            # w.set_value(np.array(range(l), dtype='int32'))
            self.constant_kernels.append(w)

        if self.mode == 9:
            kernel1 = self.add_weight(shape=(self.F1 * self.attn_heads, self.F2),
                                initializer=self.kernel_initializer,
                                name='hidden_kernel',
                                regularizer=self.kernel_regularizer)
            kernel1_b = self.add_weight(shape=(self.F2,),
                                        initializer='zero',
                                        name='hidden_kernel_b',
                                        regularizer=self.kernel_regularizer)
            self.constant_kernels.append(kernel1)
            self.constant_kernels.append(kernel1_b)
        self.built = True

    def batch_batch_dot(self, A, B, N, F):

        A = K.reshape(A, (-1, N, F))
        B = K.reshape(B, (-1, F))
        result = K.batch_dot(A, B, (2, 1))
        result = K.reshape(result, (-1, N, N))
        return result

    def call_modep2(self, X, A, E, attn_kernel, kernel, N):
        linear_transf_X = K.dot(X, kernel[0])           # (batch_size X N X F')

        A_index = A * self.constant_kernels[0]          # (batch_size X N X N)
        # A_index = K.cast(A_index, 'int32')

        neighs_emd = K.gather(K.reshape(linear_transf_X, (-1, self.F1)), A_index) # (batch_size X N X N X F')

    

        # E = K.cast(E, 'int32')
        attn_kernel_for_E = K.gather(attn_kernel[0], E) # (batch_size X N X F')

        attn_for_neights = self.batch_batch_dot(neighs_emd, attn_kernel_for_E, N, self.F1) #(batch_size X N X N)

        dense = K.relu(attn_for_neights, alpha=0.2)

        # Mask values before activation (Vaswani et al., 2017)
        comparison = K.equal(A, 0.0) # (batch_size X N X N)
        A_mask = K.switch(comparison, K.ones_like(A) * -10e9, K.zeros_like(A))
        masked = dense + A_mask

        softmax = hard_softmax(masked)

        # Linear combination with neighbors' features
        node_features = K.batch_dot(softmax, linear_transf_X)  # (batch_size X N x F')
        if self.attn_heads_reduction == 'concat' and self.activation is not None:
            node_features = self.activation(node_features)
        
        return K.cast(node_features, 'float32')

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
        node_features = K.dot(node_hidden_features, kernel[1]) + kernel[2]
        if self.attn_heads_reduction == 'concat' and self.activation is not None:
            node_features = self.activation(node_features)
        return node_features

    def call_mode6(self, X, A, attn_kernels, kernels, N):
        outputs = [X]
        for head in range(self.attn_heads):
            kernel = kernels[head]
            attn_kernel = attn_kernels[head]
            feature = self.call_mode0(X, A, attn_kernel, kernel, N, True)
            outputs.append(feature)
        node_features = K.concatenate(outputs)
        return node_features

    def call_mode8(self, X, A, E, attn_kernels, kernels, N):
        outputs = [X]
        for head in range(self.attn_heads):
            kernel = kernels[head]
            attn_kernel = attn_kernels[head]
            feature = self.call_modep2(X, A, E, attn_kernel, kernel, N)
            outputs.append(feature)
        node_features = K.concatenate(outputs)
        return node_features

    def call_mode9(self, X, A, attn_kernels, kernels, N):
        outputs = [X]
        for head in range(self.attn_heads):
            kernel = kernels[head]
            attn_kernel = attn_kernels[head]
            feature = self.call_mode0(X, A, attn_kernel, kernel, N, True)
            outputs.append(feature)
        node_hidden_features = K.concatenate(outputs)

        node_features = K.dot(node_hidden_features, self.constant_kernels[0]) + self.constant_kernels[1]
        node_features = K.relu(node_features, alpha = 0.2)
        return node_features

    


    def call(self, inputs, mask = None):
        X = inputs[0]  # Node features (batch_size X N x F)
        A = inputs[1]  # Adjacency matrix (batch_size X N x N)

        # Parameters
        N = K.shape(X)[1]  # Number of nodes in the graph

        if self.mode == 6:
            node_features = self.call_mode6(X, A, self.attn_kernels, self.kernels, N)
            if node_features.dtype != 'float32':
                node_features = K.cast(node_features, 'float32')
            return node_features
        elif self.mode == 8:
            node_features = self.call_mode8(X, A, inputs[2], self.attn_kernels, self.kernels, N)
            return node_features
        elif self.mode == 9:``
            node_features = self.call_mode9(X, A, self.attn_kernels, self.kernels, N)
            if node_features.dtype != 'float32':
                node_features = K.cast(node_features, 'float32')
            return node_features

        outputs = []
        for head in range(self.attn_heads):
            # kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)
            if self.mode == -2:
                node_features = self.call_modep2(X, A, inputs[2], attention_kernel, self.kernels[head], N)
            elif self.mode == -1:
                node_features = self.call_mode0(X, A, attention_kernel, self.kernels[head], N, use_kernel = True)
            elif self.mode == 0:
                node_features = self.call_mode0(X, A, attention_kernel, None, N, use_kernel = False)
            elif self.mode == 2:
                node_features = self.call_mode2(X, A, attention_kernel, N)
            elif self.mode == 3:
                node_features = self.call_mode3(X, A, attention_kernel, self.kernels[head], N)
            elif self.mode == 4:
                node_features = self.call_mode4(X, A, attention_kernel, self.kernels[head], N)
            elif self.mode == 5:
                node_features = self.call_mode5(X, A, attention_kernel, self.kernels[head], N)

            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')
            if self.activation is not None:
                # In case of 'average', we compute the activation here (Eq 6)
                output = self.activation(output)
        
        if output.dtype != 'float32':
            output = K.cast(output, 'float32')
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
        config = {
            'F1': self.F1,
            'F2': self.F2,
            'nb_event': self.nb_event,
            'input_dim': self.input_dim,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            "attn_dropout": self.attn_dropout,
            "activation": self.activation.__name__,
            "kernel_initializer": self.kernel_initializer.__name__, 
            "attn_kernel_initializer": self.attn_kernel_initializer.__name__,
            "kernel_regularizer": self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
            "attn_kernel_regularizer": self.attn_kernel_regularizer.get_config() if self.attn_kernel_regularizer else None,
            "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
            # "kernel_constraint": kernel_constraint,
            # "attn_kernel_constraint": attn_kernel_constraint,
            "mask_zero": self.mask_zero,
            "attention_mode": self.mode,
        }
        base_config = super(GraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))