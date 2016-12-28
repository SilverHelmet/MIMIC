from keras import backend as K
from keras.layers.core import  Flatten, Lambda, Dense
from keras.layers import merge
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed


def SimpleAttentionRNN(rnn):
    score = TimeDistributed(Dense(1))(rnn)
    flatten_score = MaskFlatten(name = 'flatten_score')(score)
    alpha = MaskLambda(function = mask_softmax, name = 'alpha')(flatten_score)
    attention = merge([alpha, rnn], mode = 'dot', dot_axes = 1, name = 'attention')
    return attention

class EventAttentionLSTM(LSTM):
    
    def __init__(self, hidden_dim, **kwargs):
        self.att_hidden_dim = att_hidden_dim
        super(EventAttentionLSTM, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

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
                            name = '{}_We'.format(self.name))
        # output attention
        self.Woa = self.inner_init((self.output_dim, self.att_hidden_dim))

        # hidden layer W
        self.Wha = self.inner_init((self.att_hidden_dim, ))

        self.trainable_weights = [self.W, self.U, self.b, self.Wea, self.Wao, self.Wha]

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


    def process(x, output):
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
        x = process(x, h_tm1)
        
        
        # LSTM process
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

def mask_softmax(x, mask):
    y = K.ones_like(mask) - K.cast(mask, K.floatx())
    return K.cast(K.softmax(y * (-999999999) + x), K.floatx())
