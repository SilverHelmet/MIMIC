from keras import backend as K
from keras.layers.core import  Flatten, Lambda, Dense, Dropout
from keras.layers import merge, InputSpec, Layer, Convolution1D, MaxPooling1D, Merge
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
import numpy as np
from keras_model import my_rnn


def get_custom_objects():
    return {
        "MaskFlatten": MaskFlatten,
        "MaskLambda": MaskLambda,
        "SegMaskEmbedding": SegMaskEmbedding,
        "EventAttentionLSTM": EventAttentionLSTM,
        "mask_softmax": mask_softmax
    }
    
def SimpleAttentionRNN(rnn):
    score = TimeDistributed(Dense(1))(rnn)
    flatten_score = MaskFlatten(name = 'flatten_score')(score)
    alpha = MaskLambda(function = mask_softmax, name = 'alpha')(flatten_score)
    attention = merge([alpha, rnn], mode = 'dot', dot_axes = 1, name = 'attention')
    return attention

def SimpleAttentionRNN2(rnn):
    hidden = TimeDistributed(Dense(128, activation = 'tanh'))(rnn)
    score = TimeDistributed(Dense(1))(hidden)
    flatten_score = MaskFlatten(name = 'flatten_score')(score)
    alpha = MaskLambda(function = mask_softmax, name = 'alpha')(flatten_score)
    attention = merge([alpha, rnn], mode = 'dot', dot_axes = 1, name = 'attention')
    return attention

class EventAttentionGRU(GRU):
    
    def __init__(self, att_hidden_dim, **kwargs):
        super(EventAttentionGRU, self).__init__(**kwargs)
        self.att_hidden_dim = att_hidden_dim
        self.input_spec = [InputSpec(ndim = 4)]

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                            'time dimension is undefined. \n'
                            '- If using a Sequential model, '
                            'specify the time dimension by passing '
                            'an `input_shape` or `batch_input_shape` '
                            'argument to your first layer. If your '
                            'first layer is an Embedding, you can '
                            'also use the `input_length` argument.\n'
                            '- If using the functional API, specify '
                            'the time dimension by passing a `shape` '
                            'or `batch_shape` argument to your Input layer.')
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = my_rnn(self.step, preprocessed_input,
                                            initial_states,
                                            go_backwards=self.go_backwards,
                                            mask=mask,
                                            constants=constants,
                                            unroll=self.unroll,
                                            input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_updates(updates, x)

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]


        self.states = [None]


        self.W = self.init((self.input_dim, 3 * self.output_dim),
                            name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                                    name='{}_U'.format(self.name))

        self.b = K.variable(np.hstack((np.zeros(self.output_dim),
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
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]
        x = self.process(x, h_tm1)



        matrix_x = K.dot(x * B_W[0], self.W) + self.b
        matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

        x_z = matrix_x[:, :self.output_dim]
        x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
        inner_z = matrix_inner[:, :self.output_dim]
        inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

        z = self.inner_activation(x_z + inner_z)
        r = self.inner_activation(x_r + inner_r)

        x_h = matrix_x[:, 2 * self.output_dim:]
        inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
        hh = self.activation(x_h + inner_h)
        
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]
    
    def get_config(self):
        config = {
            "att_hidden_dim": self.att_hidden_dim,
        }
        base_config = super(EventAttentionGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        self.attention.append(att)
        seg_emd = np.einsum('ij, ijk -> ik', att, x) # (samples, input_dim)
        return seg_emd
    
    def test_step(self, x,  states):
        h_tm1 = states[0]
        x = self.test_process(x, h_tm1)

        matrix_x = np.dot(x, K.get_value(self.W)) + K.get_value(self.b)
        matrix_inner = np.dot(h_tm1, K.get_value(self.U)[:, :2 * self.output_dim])

        x_z = matrix_x[:, :self.output_dim]
        x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
        inner_z = matrix_inner[:, :self.output_dim]
        inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

        z = np_sigmoid(x_z + inner_z)
        r = np_sigmoid(x_r + inner_r)

        x_h = matrix_x[:, 2 * self.output_dim:]
        inner_h = np.dot(r * h_tm1, K.get_value(self.U)[:, 2 * self.output_dim:])
        hh = np_sigmoid(x_h + inner_h)
        
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def test_call(self, X, mask = None):
        '''
            printable version call
        '''
        self.attention = []
        axes = [1,0,2,3]
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
        outputs = outputs.transpose([1,0] + range(2, outputs.ndim))

        self.attention = np.array(self.attention)
        self.attention = self.attention.transpose([1,0] + range(2, self.attention.ndim))
        
        return outputs, self.attention


class EventAttentionLSTM(LSTM):
    
    def __init__(self, att_hidden_dim, **kwargs):
        
        super(EventAttentionLSTM, self).__init__(**kwargs)
        self.att_hidden_dim = att_hidden_dim
        self.input_spec = [InputSpec(ndim=4)]

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                            'time dimension is undefined. \n'
                            '- If using a Sequential model, '
                            'specify the time dimension by passing '
                            'an `input_shape` or `batch_input_shape` '
                            'argument to your first layer. If your '
                            'first layer is an Embedding, you can '
                            'also use the `input_length` argument.\n'
                            '- If using the functional API, specify '
                            'the time dimension by passing a `shape` '
                            'or `batch_shape` argument to your Input layer.')
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = my_rnn(self.step, preprocessed_input,
                                            initial_states,
                                            go_backwards=self.go_backwards,
                                            mask=mask,
                                            constants=constants,
                                            unroll=self.unroll,
                                            input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_updates(updates, x)

        if self.return_sequences:
            return outputs
        else:
            return last_output
    
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

    def get_config(self):
        config = {
            "att_hidden_dim": self.att_hidden_dim,
        }
        base_config = super(EventAttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_attention_score(self, output, embeddings):
        '''
            args:
                output: (output_dim, )
                softmax_coeff: scala
                embeddings: (nb_emd, input_dim)
        '''
        mask = np.any(np.not_equal(embeddings, 0.0), axis=-1)   # (nb_emd)
        ea = np.dot(embeddings, K.get_value(self.Wea))          # (nb_emd, hidden_dim)
        oa = np.dot(output, K.get_value(self.Woa))              # (hidden_dim)
        att = oa + ea                                           # (nb_emd, hidden_dim)
        att = np.tanh(att)                                      # (nb_emd, hidden_dim)
        att = np.dot(att, K.get_value(self.Wha))                # (nb_emd, )
        att = att * mask
        return att


        

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
        self.attention.append(att)
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
        self.attention = []
        axes = [1,0,2,3]
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
        outputs = outputs.transpose([1,0] + range(2, outputs.ndim))

        states = np.array(successive_states)
        states = states.transpose([2, 1, 0] + range(3, states.ndim))

        self.attention = np.array(self.attention)
        self.attention = self.attention.transpose([1,0] + range(2, self.attention.ndim))
        
        return outputs, self.attention, states



    
class MaskFlatten(Flatten):
    def __init__(self, **kwargs):
        self.supoprt_mask = True
        super(MaskFlatten, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask = None):
        return input_mask

class MaskLambda(Lambda):
    def __init__(self, **kwargs):
        super(MaskLambda, self).__init__(**kwargs)
        self.supports_masking = True
        

    def compute_mask(self, input, input_mask = None):
        return None

    def get_config(self):
        config = {
            # 'function': self.function.__name__,
            # 'function_type': "function",
            "arguments": {}
        }
        base_config = super(MaskLambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

    def get_config(self):
        config = {
            "mask_value": self.mask_value,
        }
        base_config = super(SegMaskEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaskCNN1D(Convolution1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskCNN1D, self).__init__(**kwargs)

    def compute_mask(self, x, mask = None):
        output_length = x.shape[1] - self.filter_length + 1
        return mask[:, :output_length]


class MaskMaxFilter(Layer):
    def call(self, x, mask = None):
        mask = K.expand_dims(mask, -1)
        y = K.ones_like(mask) - K.cast(mask, K.floatx())
        z = y * (-999999999) + x
        return K.cast(z, K.floatx())


    def compute_mask(self, x, mask):
        return None  

class MaskOutput(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskOutput, self).__init__(**kwargs)

    def call(self, x, mask = None):
        return mask
    
    def compute_mask(self, x, mask = None):
        return mask


def np_switch(condition, x1, x2):
    return np.select([condition, np.not_equal(condition, True)], [x1, x2])

def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def np_softmax(x):
    x = x - np.max(x, axis = 1, keepdims = True)
    e_x = np.exp(x)
    y = e_x / np.sum(e_x, axis = 1, keepdims = True)
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

def mask_max(x, mask):
    
    y = K.ones_like(mask) - K.cast(mask, K.floatx())
    z = y * (-999999) + x
    return z

def make_CNN1D(filter_lengths, feature_maps, emd, max_segs, l2_reg_cof = .0, drop_rate = .0):
    cnn_layers = []
    for nb_filter, filter_length in zip(feature_maps, filter_lengths):
        cnn = MaskCNN1D(nb_filter = nb_filter, filter_length = filter_length, 
             W_regularizer = l2(l2_reg_cof), b_regularizer = l2(l2_reg_cof),
             name = 'cnn_%d*%d' %(nb_filter, filter_length))(emd)
        mask_cnn = MaskMaxFilter()(cnn)
        pooling = MaxPooling1D(pool_length = max_segs - filter_length + 1)(mask_cnn)
        cnn_layers.append(pooling)

    cnn_output = merge(inputs = cnn_layers , mode = 'concat', concat_axis = 2)
    flatten_cnn = Flatten()(cnn_output)
    if drop_rate > .0:
        return Dropout(p = drop_rate)(flatten_cnn)
    else:
        return flatten_cnn
        
