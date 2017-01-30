from keras.models import Model, load_model
from models.models import get_custom_objects
from keras import backend as K
import numpy as np

def load_attention_model(filepath):
    model = load_model(filepath, custom_objects = get_custom_objects())
    return model

def get_event_attention(model, x):
    emd_model = Model(input = model.input, output = model.get_layer("embedding").output)
    emd_out =  emd_model.predict(x = x)
    mask = np.any(np.not_equal(emd_out, 0.0), axis=-1)
    mask = np.any(np.not_equal(mask, 0.0), axis=-1)
    rnn =  model.get_layer("rnn")
    _, attention = rnn.test_call(emd_out, mask)
    mask = np.expand_dims(mask, axis = -1)
    return attention * mask

def get_event_output(model, x):
    emd_model = Model(input = model.input, output = model.get_layer("embedding").output)
    emd_out =  emd_model.predict(x = x)
    mask = np.any(np.not_equal(emd_out, 0.0), axis=-1)
    mask = np.any(np.not_equal(mask, 0.0), axis=-1)
    rnn =  model.get_layer("rnn")
    outputs, _ = rnn.test_call(emd_out, mask)
    return outputs[:,-1]

def get_temporal_attention(model, x):
    attention_model = Model(input = model.input, output = model.get_layer("alpha").output)
    return attention_model.predict(x)

def get_event_attention_at_seg(model, output, embedding):
    rnn = model.get_layer('rnn')
    score = rnn.get_attention_score(output, embedding)
    return score

def get_embedding(model, x, name = 'segmaskembedding_1'):
    emd_layer = model.get_layer(name)
    return K.eval(K.gather(emd_layer.W, x))

    
    