from keras.models import Model, load_model
from models.models import get_custom_objects
from keras import backend as K
import numpy as np

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x)))

def load_attention_model(filepath):
    model = load_model(filepath, custom_objects = get_custom_objects())
    return model

def get_RNN_result(model, x):
    emd_out =  get_embedding(model, x)
    mask = np.any(np.not_equal(x[0], 0.0), axis = -1)
    # mask = np.any(np.not_equal(emd_out, 0.0), axis=-1)
    # mask = np.any(np.not_equal(mask, 0.0), axis=-1)
    rnn =  model.get_layer("rnn")
    outputs, attention, states = rnn.test_call(emd_out, mask)
    mask = np.expand_dims(mask, axis = -1)
    return outputs*mask, states*mask

def score_of_outputs(model, outputs):
    layer = model.get_layer(name = 'prediction')
    W, b = layer.get_weights()
    return sigmoid(np.dot(outputs, W) + b)


def get_event_attention(model, x):
    emd_out =  get_embedding(model, x)
    mask = np.any(np.not_equal(x[0], 0.0), axis = -1)
    # mask = np.any(np.not_equal(emd_out, 0.0), axis=-1)
    # mask = np.any(np.not_equal(mask, 0.0), axis=-1)
    rnn =  model.get_layer("rnn")
    _, attention, _ = rnn.test_call(emd_out, mask)
    mask = np.expand_dims(mask, axis = -1)
    return attention * mask


def get_event_output_at_time(model, x, time = -1):
    emd_out =  get_embedding(model, x)
    mask = np.any(np.not_equal(x[0], 0.0), axis = -1)
    # mask = np.any(np.not_equal(emd_out, 0.0), axis=-1)
    # mask = np.any(np.not_equal(mask, 0.0), axis=-1)
    rnn =  model.get_layer("rnn")
    outputs, _ = rnn.test_call(emd_out, mask)
    return outputs[:,time]

def get_temporal_attention(model, x):
    attention_model = Model(input = model.input, output = model.get_layer("alpha").output)
    return attention_model.predict(x)

def get_event_attention_score_at_seg(model, output, embedding):
    rnn = model.get_layer('rnn')
    score = rnn.get_attention_score(output, embedding)
    return score

# def get_embedding(model, x, name = 'embedding'):
#     emd_layer = model.get_layer(name)
#     return K.eval(K.gather(emd_layer.W, x))

def get_embedding(model, x, name = "embedding with feature"):
    emd_model = Model(input = model.input, output = model.get_layer(name).output)
    return emd_model.predict(x)

    
    