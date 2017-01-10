from keras import backend as K
from keras.layers.core import  Flatten, Lambda, Dense
from keras.layers import merge, InputSpec
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
import numpy as np


def SimpleAttentionRNN(rnn):
    score = TimeDistributed(Dense(1))(rnn)
    flatten_score = MaskFlatten(name = 'flatten_score')(score)
    alpha = MaskLambda(function = mask_softmax, name = 'alpha')(flatten_score)
    attention = merge([alpha, rnn], mode = 'dot', dot_axes = 1, name = 'attention')
    return attention

class EventAttentionLSTM(LSTM):
    
    def __init__(self, att_hidden_dim, **kwargs):
        
        super(EventAttentionLSTM, self).__init__(**kwargs)
        self.att_hidden_dim = att_hidden_dim
        self.input_spec = [InputSpec(ndim=4)]
    
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

        self.states = [None, None]

        self.W = self.init((self.input_dim, 4 * self.output_dim),
                            name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
                                    name='{}_U'.format(self.name))
        self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                        K.get_value(self.forget_bias_init((self.output_dim,))),
                                        np.zeros(self.output_dim),
                                        np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))
        # event attention
        self.Wea = self.init((self.input_dim, self.att_hidden_dim),
                            name = '{}_Wea'.format(self.name))
        # output attention
        self.Woa = self.inner_init((self.output_dim, self.att_hidden_dim),
                            name = '{}_Woa'.format(self.name))

        # hidden layer W
        self.Wha = self.init((self.att_hidden_dim, ),
                            name = '{}_Wha'.format(self.name))

        self.trainable_weights = [self.W, self.U, self.b, self.Wea, self.Woa, self.Wha]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x):
        return x

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, length, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2, 3))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def process(self, x, output):
        '''
            args:
                x: (samples, seg_length, input_dim)
                output: (samples, output_dim)
            return:
                seg_emd: (samples, input_dim)
        '''
        mask = K.any(K.not_equal(x, 0.0), axis=-1)  # (samples, seg_length)
        ea = K.dot(x, self.Wea)                     # (samples, seg_length, hidden_dim) 
        oa = K.dot(output, self.Woa)                # (samples, hidden_dim)
        oa = K.expand_dims(oa, 1)                   # (samples, 1, hidden_dim)
        att = ea + oa                               # (samples, seg_length, hidden_dim)
        att = K.tanh(att)                           # (samples, seg_length, hidden_dim)
        att = K.dot(att, self.Wha)                  # (samples, seg_length)
        att = mask_softmax(att, mask)               # (samples, seg_length)
        seg_emd = K.batch_dot(att, x)               # (samples, input_dim)

        return seg_emd

    def step(self, x, states):
        
        
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        x = self.process(x, h_tm1)
        
        
        # LSTM gpu process
        z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b

        z0 = z[:, :self.output_dim]
        z1 = z[:, self.output_dim: 2 * self.output_dim]
        z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
        z3 = z[:, 3 * self.output_dim:]

        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.inner_activation(z3)
     

        h = o * self.activation(c)
        return h, [h, c]

    def test_process(self, x, output):
        '''
            args:
                x: (samples, seg_length, input_dim)
                output: (samples, output_dim)
            return:
                seg_emd: (samples, input_dim)
        '''
        mask = np.any(np.not_equal(x, 0.0), axis=-1)  # (samples, seg_length)
        ea = np.dot(x, K.get_value(self.Wea))        # (samples, seg_length, hidden_dim) 
        oa = np.dot(output, K.get_value(self.Woa))   # (samples, hidden_dim)
        oa = np.expand_dims(oa, 1)                   # (samples, 1, hidden_dim)
        att = oa + ea                                # (samples, seg_length, hidden_dim)
        att = np.tanh(att)                           # (samples, seg_length, hidden_dim)
        att = np.dot(att, K.get_value(self.Wha))     # (samples, seg_length)
        att = np_mask_softmax(att, mask)             # (samples, seg_length)
        seg_emd = np.einsum('ij, ijk -> ik', att, x) # (samples, input_dim)
        return seg_emd
    
    def test_step(self, x,  states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        x = self.test_process(x, h_tm1)
        
        
        # LSTM gpu process
        z = np.dot(x, K.get_value(self.W)) + np.dot(h_tm1, K.get_value(self.U)) + K.get_value(self.b)

        z0 = z[:, :self.output_dim]
        z1 = z[:, self.output_dim: 2 * self.output_dim]
        z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
        z3 = z[:, 3 * self.output_dim:]

        i = np_sigmoid(z0)
        f = np_sigmoid(z1)
        c = f * c_tm1 + i * np_sigmoid(z2)
        o = np_sigmoid(z3)
     

        h = o * np_sigmoid(c)
        return h, [h, c]

    def test_call(self, X, mask = None):
        '''
            printable version call
        '''
        axes = [1,0,2,3]
        print "mask shape =", mask.shape
        X = X.transpose(axes)
        mask = np.expand_dims(mask, -1)
        mask = mask.transpose(axes[:mask.ndim])
        input_length = X.shape[0]
        nb_sample = X.shape[1]
        indices = list(range(input_length))
        
        # unroll rnn
        successive_outputs = []
        successive_states = []
        initial_states = [np.zeros((nb_sample, self.output_dim)) for _ in range(2)]
        states = initial_states
        for i in indices:
            output, new_states = self.test_step(X[i], states)

            if len(successive_outputs) == 0:
                prev_output = np.zeros_like(output)
            else:
                prev_output = successive_outputs[-1]
            
            output = np_switch(mask[i], output, prev_output)
            kept_states = []
            for state, new_state in zip(states, new_states):
                kept_states.append(np_switch(mask[i], new_state, state))
            states = kept_states

            successive_outputs.append(output)
            successive_states.append(states)

        outputs = np.array(successive_outputs)
        return outputs







    
class MaskFlatten(Flatten):
    def __init__(self, **kwargs):
        self.supoprt_mask = True
        super(MaskFlatten, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask = None):
        return input_mask

class MaskLambda(Lambda):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskLambda, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask = None):
        return None

class SegMaskEmbedding(Embedding):
    def __init__(self, mask_value = 0., **kwargs):
        self.mask_value = mask_value
        super(SegMaskEmbedding, self).__init__(**kwargs)

    
    def build(self, input_shape):
        '''
            set W[0] = 0
        '''

        super(SegMaskEmbedding, self).build(input_shape)
        WV = self.W.get_value()
        WV[0] = 0
        K.set_value(self.W, WV)
        
    def compute_mask(self, x, input_mask = None):
        '''
            args:
                x: (samples, max_segs, max_seg_length)
        '''
        return K.any(K.not_equal(x, self.mask_value), axis=-1)      #  (samples, max_segs)
    

    def get_output_shape_for(self, input_shape):
    
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return (input_shape[0], input_length, input_shape[2], self.output_dim)

def np_switch(condition, x1, x2):
    return np.select([condition, np.not_equal(condition, True)], [x1, x2])

def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def np_softmax(x):
    x = x - np.max(x, axis = 1, keepdims = True)
    e_x = np.exp(x)
    y = e_x / np.sum(e_x, axis = 1)
    return y

def np_mask_softmax(x, mask):
    '''
    do np's softmax in `dim` dimension
    args:
        x: (nb_sample, dim)
        mask: (nb_sample, dim)
    return:
        cof: (nb_sample, dim)
    '''
    y = np.ones_like(mask) - mask
    x = y * (-999999999) + x
    return np_softmax(x)


def mask_softmax(x, mask):
    y = K.ones_like(mask) - K.cast(mask, K.floatx())
    z = y * (-999999999) + x
    z = K.softmax(z)
    return K.cast(z, K.floatx())
    