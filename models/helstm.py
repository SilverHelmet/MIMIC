from keras.layers.recurrent import LSTM
from keras import backend as K
from theano import tensor as T

class HELSTM(LSTM):
    def __init__(self, off_alpha, event_hidden_dim, **kwargs):
        super(HELSTM, self).__init__(consume_less = 'gpu', **kwargs)
        self.event_hidden_dim = event_hidden_dim
        self.off_alpha = off_alpha

    def build(self, input_shape):
        super(HELSTM, self).build(input_shape)

        self.event_hid_w = self.add_weight((self.input_dim, self.event_hidden_dim),
                                       initializer=self.init,
                                       name='{}_event_hid_w'.format(self.name),
                                       regularizer=self.w_regularizer)

        self.event_hid_b = self.add_weight((self.event_hidden_dim,),
                                        initializer='zero',
                                        name='{}_event_hid_b'.format(self.name),
                                        regularizer=self.b_regularizer)

        self.event_out_w = self.add_weight((self.event_hidden_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_event_out_w'.format(self.name),
                                       regularizer=self.w_regularizer)

        self.event_out_b = self.add_weight((self.output_dim, ),
                                initializer='zero',
                                name='{}_event_out_b'.format(self.name),
                                regularizer=self.b_regularizer)

        def period_init(shape, name = None):
            return K.random_uniform_variable(shape, 10.0, 1000.0, name=name)

        def shift_init(shape, name = None):
            return K.random_uniform_variable(shape, 0.0, 1000.0, name=name)

        def onend_init(shape, name = None):
            return K.variable([0.05], name=name)

        self.period = self.add_weight((1, ),  
                                initializer = period_init,
                                name = "{}_period".format(self.name))

        self.shift = self.add_weight((1, ), 
                                initializer = shift_init,
                                name = "{}_shift".format(self.name))

        self.on_end = self.add_weight((1, ), 
                                initializer = onend_init,
                                name = "{}_onend".format(self.name))

    def calc_time_gate(self, times):
        # times (batch_size, )
        on_mid_broadcast = K.abs(self.on_end * 0.5 * self.period)
        on_end_broadcast = K.abs(self.on_end * period_broadcast)
        in_cycle_time = T.mod(times + self.shift, self.period)

        is_up_phase = K.lesser_equal(in_cycle_time, on_mid_broadcast)
        is_down_phase = K.greater(in_cycle_time, on_mid_broadcast) * K.lesser_equal(in_cycle_time, on_end_broadcast)

        sleep_wake_mask = K.switch(is_up_phase, in_cycle_time/on_mid_broadcast,
                                K.switch(is_down_phase,
                                    (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                        self.off_slope*(in_cycle_time/period)))

        return sleep_wake_mask
        


    def step(self, x, states)
        h, states = super(HELSTM, self).step(x, states)
        c = states[1]

        event_hidden = K.tanh(K.dot(x, self.event_hid_w) + self.event_hid_b)
        event_attn = K.sigmoid(K.dot(event_hidden, self.event_out_w) + self.event_out_b)

        sleep_wake_mask = self.calc_time_gate(time_input_n)






        


