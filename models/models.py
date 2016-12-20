from keras import backend as K
from keras.layers.core import  Flatten, Lambda
from keras.layers import merge
from keras.layers.wrappers import TimeDistributed


def SimpleAttentionRNN(rnn):
    score = TimeDistributed(Dense(1))(lstm)
    flatten_score = MaskFlatten(name = 'flatten_score')(score)
    alpha = MaskLambda(function = mask_softmax, name = 'alpha')(flatten_score)
    attention = merge([alpha, rnn], mode = 'dot', dot_axes = 1, name = 'attention')
    return attention

    
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
