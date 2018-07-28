from keras.layers.recurrent import LSTM
from keras import backend as K
import theano.tensor as T 
import theano
from keras.layers import InputSpec, merge, Merge
from keras.backend.theano_backend import expand_dims


def helstm_rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):

    ndim = inputs[0].ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    if unroll:
        if input_length is None:
            raise ValueError('When specifying `unroll=True`, '
                             'an `input_length` '
                             'must be provided to `rnn`.')

    axes = [1, 0] + list(range(2, ndim))
    # inputs = inputs.dimshuffle(axes)
    # changed by lihaoran
    tmp_inputs = []
    for input_x in inputs:
        input_ndim = input_x.ndim
        input_axes = [1, 0] + list(range(2, input_ndim))
        input_x = input_x.dimshuffle(input_axes)
        tmp_inputs.append(input_x)
    inputs = tmp_inputs

    if constants is None:
        constants = []

    if mask is not None:
        if mask.ndim == ndim-1:
            mask = expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)

        if unroll:
            assert False
        else:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_output = step_function(inputs[0], initial_states + constants)[0] * 0
            # Theano gets confused by broadcasting patterns in the scan op
            initial_output = T.unbroadcast(initial_output, 0, 1)

            def _step(input, mask, output_tm1, *states):
                output, new_states = step_function(input, states)
                # output previous output if masked.
                output = T.switch(mask, output, output_tm1)
                return_states = []
                for state, new_state in zip(states, new_states):
                    return_states.append(T.switch(mask, new_state, state))
                return [output] + return_states

            results, _ = theano.scan(
                _step,
                sequences=[inputs, mask],
                outputs_info=[initial_output] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if isinstance(results, list):
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []
            
    else:
        assert False

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    states = [T.squeeze(state[-1]) for state in states]
    return last_output, outputs, states

class HELSTM(LSTM):
    def __init__(self, off_slope = 1e-3, event_hidden_dim = None, **kwargs):
        super(HELSTM, self).__init__(consume_less = 'gpu', **kwargs)
        self.event_hidden_dim = self.input_dim
        self.off_slope = off_slope

    def build(self, input_shape):
        x_input_shape = (input_shape[0], input_shape[1], input_shape[2] - 1)
        super(HELSTM, self).build(x_input_shape)
        self.input_spec = [InputSpec(shape = (input_shape))]

        if self.event_hidden_dim is None:
            self.event_hidden_dim = self.input_dim


        self.event_hid_w = self.add_weight((self.input_dim, self.event_hidden_dim),
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

        def period_init(shape, name = None):
            return K.random_uniform_variable(shape, 10.0, 1000.0, name=name)

        def shift_init(shape, name = None):
            return K.random_uniform_variable(shape, 0.0, 1000.0, name=name)

        def onend_init(shape, name = None):
            return K.variable([0.05], name=name)

        self.period_timegate = self.add_weight((1, ),  
                                initializer = period_init,
                                name = "{}_period".format(self.name))

        self.shift_timegate = self.add_weight((1, ), 
                                initializer = shift_init,
                                name = "{}_shift".format(self.name))

        self.on_end_timegate = self.add_weight((1, ), 
                                initializer = onend_init,
                                name = "{}_onend".format(self.name))

    def calc_time_gate(time_input_n):
            # Broadcast the time across all units
        t_broadcast = time_input_n.dimshuffle([0,'x'])
        # Get the time within the period
        in_cycle_time = T.mod(t_broadcast + shift_broadcast, period_broadcast)
        # Find the phase
        is_up_phase = T.le(in_cycle_time, on_mid_broadcast)
        is_down_phase = T.gt(in_cycle_time, on_mid_broadcast)*T.le(in_cycle_time, on_end_broadcast)

        # Set the mask
        sleep_wake_mask = T.switch(is_up_phase, in_cycle_time/on_mid_broadcast,
                            T.switch(is_down_phase,
                                (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                    off_slope*(in_cycle_time/period_broadcast)))

        return sleep_wake_mask

    def calc_time_gate(self, time_input_n):
        t_broadcast = time_input_n.dimshuffle([0,'x'])
        shift_broadcast = self.shift_timegate.dimshuffle(['x',0])
        period_broadcast = K.abs(self.period_timegate.dimshuffle(['x',0]))
        on_mid_broadcast = K.abs(self.on_end_timegate.dimshuffle(['x',0])) * 0.5 * period_broadcast
        on_end_broadcast = K.abs(self.on_end_timegate.dimshuffle(['x',0])) * period_broadcast

        in_cycle_time = T.mod(t_broadcast + shift_broadcast, period_broadcast)
        is_up_phase = K.lesser_equal(in_cycle_time, on_mid_broadcast)
        is_down_phase = K.greater(in_cycle_time, on_mid_broadcast)*K.lesser_equal(in_cycle_time, on_end_broadcast)

        # Set the mask
        sleep_wake_mask = K.switch(is_up_phase, in_cycle_time/on_mid_broadcast,
                            K.switch(is_down_phase,
                                (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                    self.off_slope*(in_cycle_time/period_broadcast)))
        return sleep_wake_mask

        # times (batch_size, )
        # on_mid_broadcast = K.abs(self.on_end * 0.5 * self.period)
        # on_end_broadcast = K.abs(self.on_end * self.period)
        # K.tile
        # in_cycle_time = T.mod(times + self.shift, self.period)

        # is_up_phase = K.lesser_equal(in_cycle_time, on_mid_broadcast)
        # is_down_phase = K.greater(in_cycle_time, on_mid_broadcast) * K.lesser_equal(in_cycle_time, on_end_broadcast)

        # sleep_wake_mask = K.switch(is_up_phase, in_cycle_time/on_mid_broadcast,
        #                         K.switch(is_down_phase,
        #                             (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
        #                                 self.off_slope*(in_cycle_time/self.period)))

        # return sleep_wake_mask
        

    def step(self, x, states):
        input_x = x[:, : -1]
        time_input_n = x[:, -1]
        prev_h = states[0]
        prev_c = states[1]
        h, new_states = super(HELSTM, self).step(input_x, states)
        # return h, new_states

        c = new_states[1]

        event_hidden = K.tanh(K.dot(input_x, self.event_hid_w) + self.event_hid_b)
        event_attn = K.sigmoid(K.dot(event_hidden, self.event_out_w) + self.event_out_b)
        sleep_wake_mask = event_attn

        sleep_wake_mask = self.calc_time_gate(time_input_n)
        sleep_wake_mask = K.tile(sleep_wake_mask, (1, self.output_dim))
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
    emd = Embedding(input_dim = 20, output_dim = emd_dim, mask_zero = True)(X)
    Time = Input(shape = (input_length, 1), name = 'time')
    merge_layer = Merge(mode = 'concat', concat_axis = 2)
    med_time = merge_layer([Time, emd])

    helstm = HELSTM(output_dim = output_dim)(med_time)
    model = Model(input = [X, Time], output = helstm)

    # x = np.random.random((batch, input_length, emd_dim))

    x = np.random.randint(0, 3, size = (batch, input_length))
    t = np.random.random((batch, input_length, 1))

    print x.shape
    print t.shape
    model.summary()
    out =  model.predict([x, t])
    print out.shape

    x = np.random.random((batch, input_length, output_dim))
    t = np.random.random((batch, input_length, 1))

    mask1 = np.random.randint(0, 2, (batch, input_length, 1))
    mask2 = np.ones_like(t)

    concatenated = np.concatenate([mask1, mask2], axis=2)
    mask = np.all(concatenated, axis=-1, keepdims=False)
    mask1 =  np.squeeze(mask)
    # mask = np.array(mask, dtype = 'int')
    print mask1
    print mask

    print (mask1 != mask).sum()
    print mask.shape


    


        






        


