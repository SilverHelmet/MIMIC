from keras.layers.recurrent import LSTM
from keras import backend as K
import theano.tensor as T 
import theano
from keras.layers.core import initializations
from keras.layers import InputSpec, merge, Merge, Layer
from keras.backend.theano_backend import expand_dims

class FeatureEmbeddding(Layer):
    def __init__(self, input_dim, output_dim, init='uniform', **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        super(FeatureEmbeddding, self).__init__(**kwargs)
        self.support_zero = True
    
    def build(self, inputs_shape):
        self.W = self.add_weight((self.input_dim, self.output_dim),
                                initializer=self.init,
                                name='{}_W'.format(self.name))

        WV = self.W.get_value()
        WV[0] = 0
        K.set_value(self.W, WV)

        self.feature_trans_w = self.add_weight((self.input_dim, ),
                                initializer=self.init,
                                name='{}_feature_trans_w'.format(self.name))
        self.feature_trans_b = self.add_weight((self.input_dim, ),
                                initializer='zero',
                                name='{}_feature_trans_b'.format(self.name))
        self.build = True

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], self.output_dim)

    def call(self, inputs, mask = None):
        feature_idx = inputs[0]     #(batch, 1000, 3)
        feature_value = inputs[1]   #(batch, 1000, 3)

        if K.dtype(feature_idx) != 'int32':
            feature_idx = K.cast(feature_idx, 'int32')

        feature_trans_w = K.gather(self.feature_trans_w, feature_idx)
        feature_trans_b = K.gather(self.feature_trans_b, feature_idx)
        feature_dist = K.tanh(feature_value * feature_trans_w + feature_trans_b) #(batch, 1000, 3)

        feature_emd = K.gather(self.W, feature_idx) #(batch, 1000, 3, output_dim)

        dist_feature_emd = feature_emd * K.expand_dims(feature_dist)
        
        merged_feature_emd = K.sum(dist_feature_emd, axis = 2, keepdims = False)
        return merged_feature_emd

def period_variable_sampling(setting, length):
    '''
        1 - 2
        3 - 6
        24
    '''
    ratio_1_2 = setting.get('period_1_2', 1.0)
    ratio_3_6 = setting.get('period_3_6', 0.0)
    ratio_24 = setting.get('period_24', 0.0)
    sum_ratio = ratio_1_2 + ratio_3_6 + ratio_24
    ratio_1_2 /= sum_ratio
    ratio_3_6 /= sum_ratio
    ratio_24 /= sum_ratio

    cnt_1_2 = int(length * ratio_1_2)
    cnt_3_6 = int(length * ratio_3_6)
    cnt_24 = length - cnt_1_2 - cnt_3_6
    print 'samping cnt [1, 2]:%d, [3, 6]:%d, [24]:%d' %(cnt_1_2, cnt_3_6, cnt_24)

    lows = [1.0] * cnt_1_2 + [3.0] * cnt_3_6 + [24.0] * cnt_24
    highs = [2.0] * cnt_1_2 + [6.0] * cnt_3_6 + [24.0] * cnt_24
    return lows, highs





class HELSTM(LSTM):
    def __init__(self, event_emd_dim, off_slope = 1e-3, event_hidden_dim = None, setting = {}, **kwargs):
        super(HELSTM, self).__init__(consume_less = 'gpu', **kwargs)
        self.event_hidden_dim = event_hidden_dim
        self.off_slope = off_slope
        self.event_emd_dim = event_emd_dim
        self.setting = setting

    def build(self, input_shape):
        x_input_shape = (input_shape[0], input_shape[1], input_shape[2] - 1 - self.event_emd_dim)
        super(HELSTM, self).build(x_input_shape)
        self.input_spec = [InputSpec(shape = (input_shape))]
        if self.event_hidden_dim is None:
            self.event_hidden_dim = self.event_emd_dim


        self.event_hid_w = self.add_weight((self.event_emd_dim, self.event_hidden_dim),
                                       initializer=self.init,
                                       name='{}_event_hid_w'.format(self.name),
                                       regularizer=self.W_regularizer)

        self.event_hid_b = self.add_weight((self.event_hidden_dim,),
                                        initializer='zero',
                                        name='{}_event_hid_b'.format(self.name),
                                        regularizer=self.b_regularizer)

        self.event_out_w = self.add_weight((self.event_hidden_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_event_out_w'.format(self.name),
                                       regularizer=self.W_regularizer)

        self.event_out_b = self.add_weight((self.output_dim, ),
                                initializer='zero',
                                name='{}_event_out_b'.format(self.name),
                                regularizer=self.b_regularizer)

        def period_init(shape, lows, highs, name = None):
            x = []
            for low, high in zip(lows, highs):
                x.append(np.random.uniform(low, high))
            return K.variable(x, name = name)

        def shift_init(shape, name = None):
            return K.random_uniform_variable(shape, 0.0, 1000.0, name=name)

        def onend_init(shape, name = None):
            return K.variable([0.05] * shape[0], name=name)
        
        lows, highs = period_variable_sampling(self.setting, self.output_dim)
        period_init_lambda = lambda shape, name: period_init(shape = shape, lows = lows, highs = highs, name = name)

        self.period_timegate = self.add_weight((self.output_dim, ),  
                                initializer = period_init_lambda,
                                name = "{}_period".format(self.name))

        self.shift_timegate = self.add_weight((self.output_dim, ), 
                                initializer = shift_init,
                                name = "{}_shift".format(self.name))

        self.on_end_timegate = self.add_weight((self.output_dim, ), 
                                initializer = onend_init,
                                name = "{}_onend".format(self.name))

    def calc_time_gate(self, time_input_n):
        '''
            time_input_n: (batch, )
        '''
        t_broadcast = time_input_n.dimshuffle([0,'x'])                                              # (batch, 1)
        shift_broadcast = self.shift_timegate.dimshuffle(['x',0])                                   # (1, output_dim)
        period_broadcast = K.abs(self.period_timegate.dimshuffle(['x',0]))                          # (1, output_dim)
        on_mid_broadcast = K.abs(self.on_end_timegate.dimshuffle(['x',0])) * 0.5 * period_broadcast # (1, output_dim)
        on_end_broadcast = K.abs(self.on_end_timegate.dimshuffle(['x',0])) * period_broadcast       # (1, output_dim)

        in_cycle_time = T.mod(t_broadcast + shift_broadcast, period_broadcast)                      # (batch, output_dim)
        is_up_phase = K.lesser_equal(in_cycle_time, on_mid_broadcast)                               # ()
        is_down_phase = K.greater(in_cycle_time, on_mid_broadcast)*K.lesser_equal(in_cycle_time, on_end_broadcast)

        # Set the mask
        sleep_wake_mask = K.switch(is_up_phase, in_cycle_time/on_mid_broadcast,
                            K.switch(is_down_phase,
                                (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                    self.off_slope*(in_cycle_time/period_broadcast)))
        return sleep_wake_mask

        

    def step(self, x, states):
        # input_x = x[:, : self.input_dim]
        # event_emd = input_x[:, :self.event_emd_dim]
        event_emd = x[:, : self.event_emd_dim]
        input_x = x[:, self.event_emd_dim: self.event_emd_dim + self.input_dim]
        time_input_n = x[:, -1]
        prev_h = states[0]
        prev_c = states[1]
        h, new_states = super(HELSTM, self).step(input_x, states)
        # return h, new_states

        c = new_states[1]

        event_hidden = K.tanh(K.dot(event_emd, self.event_hid_w) + self.event_hid_b)
        event_attn = K.sigmoid(K.dot(event_hidden, self.event_out_w) + self.event_out_b)

        sleep_wake_mask = self.calc_time_gate(time_input_n)
        # sleep_wake_mask = K.tile(sleep_wake_mask, (1, self.output_dim))
        sleep_wake_mask = sleep_wake_mask * event_attn

        cell = sleep_wake_mask*c + (1.-sleep_wake_mask)*prev_c
        hid = sleep_wake_mask*h + (1.-sleep_wake_mask)*prev_h
        return hid, [hid, cell]


    def get_output_shape_for(self, input_shape):
        input_shape = (input_shape[0], input_shape[1], input_shape[2] - 1)
        return super(HELSTM, self).get_output_shape_for(input_shape)

if __name__ == "__main__":
    from keras.layers import Input, Embedding
    from keras.models import Model
    import numpy as np
    input_length = 5
    emd_dim = 10
    input_dim = 20
    batch = 7
    output_dim = 30
    X = Input(shape = (input_length, ), name = 'x')
    e_emd = Embedding(input_dim = 20, output_dim = emd_dim, mask_zero = True)(X)
    F_idx = Input(shape = (input_length, 3), name = 'f_idx')
    F_value = Input(shape = (input_length, 3), name = 'f_value')
    f_emd = FeatureEmbeddding(input_dim = 10, output_dim = emd_dim, name = 'f_emd')([F_idx, F_value])
    emd = Merge(mode = 'sum', name = 'emd')([e_emd, f_emd])
    
    Time = Input(shape = (input_length, 1), name = 'time')
    merge_layer = Merge(mode = 'concat', concat_axis = 2)
    med_time = merge_layer([e_emd, emd, Time])

    helstm = HELSTM(event_emd_dim = emd_dim, output_dim = output_dim)(med_time)
    model = Model(input = [X, F_idx, F_value, Time], output = helstm)

    # x = np.random.random((batch, input_length, emd_dim))

    x = np.random.randint(0, 3, size = (batch, input_length))
    t = np.random.random((batch, input_length, 1))
    f_idx = np.random.randint(0, 10, size = (batch, input_length, 3))
    f_value = np.random.random((batch, input_length, 3))

    print x.shape
    print t.shape
    model.summary()
    out =  model.predict([x, f_idx, f_value, t])
    print out.shape

    # x = np.random.random((batch, input_length, output_dim))
    # t = np.random.random((batch, input_length, 1))

    # mask1 = np.random.randint(0, 2, (batch, input_length, 1))
    # mask2 = np.ones_like(t)

    # concatenated = np.concatenate([mask1, mask2], axis=2)
    # mask = np.all(concatenated, axis=-1, keepdims=False)
    # mask1 =  np.squeeze(mask)
    # # mask = np.array(mask, dtype = 'int')
    # print mask1
    # print mask

    # print (mask1 != mask).sum()
    # print mask.shape


    


        






        


