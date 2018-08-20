import os
import time
import h5py
import sys
import random
sys.path.insert(0, '.')
from util import Print
from MergeEmbedding import MergeEmbeddingLayer
import numpy as np 
import theano
import theano.tensor as T 
import lasagne
from HELSTM import HELSTMLayer, HELSTMGate
from lasagne.layers.recurrent import Gate
from lasagne.init import Initializer
from lasagne.utils import floatX
from lasagne.random import get_rng
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score

def load_data(path, name, time_off, hour_mul):
    Print('loading %s' %(path + name))
    f = h5py.File(path+name, 'r')
    times = np.asarray(f['time'], dtype='float32')
    hours = np.asarray(times * hour_mul, dtype = 'int8')
    Print("hour_min = %.2f, hour_max = %.2f" %(hours.min(), hours.max()))
    times /= time_off
    Print("time_min = %.2f, time_max = %.2f" %(times.min(), times.max()))
    label = np.asarray(f['label'], dtype='int8')
    event = np.asarray(f['event'], dtype='int16')
    feature_id = np.asarray(f['feature_idx'], dtype='int16')
    feature_value = np.asarray(f['feature_value'], dtype='float32')
    mask = np.asarray(event>0, dtype='int8')
    f.close()
    Print('done.')
    return event, feature_id, feature_value, label, times, mask, hours

def load_data_all(dataset, time_off, hour_mul):
    global ICU_test_data
    global ICU_train_data
    global ICU_valid_data
    Print('dataset is %s' %dataset)
    if dataset == 'labtest':
        data_path = "lab_exper/"
        file_name = "labtest_test_1000.h5"
        ICU_test_data = load_data(data_path, file_name, time_off, hour_mul)
        file_name = "labtest_train_1000.h5"
        ICU_train_data = load_data(data_path, file_name, time_off, hour_mul)
        file_name = "labtest_valid_1000.h5"
        ICU_valid_data = load_data(data_path, file_name, time_off, hour_mul) 
    elif dataset == 'icd9':
        data_path = 'icd_exper/'
        file_name = "icd9_test_5000.h5"
        ICU_test_data = load_data(data_path, file_name, time_off, hour_mul)
        file_name = "icd9_train_5000.h5"
        ICU_train_data = load_data(data_path, file_name, time_off, hour_mul)
        file_name = "icd9_valid_5000.h5"
        ICU_valid_data = load_data(data_path, file_name, time_off, hour_mul)
    elif dataset == 'icd9_rt':
        data_path = 'icd_exper/'
        file_name = "icd9_test_1000_rt.h5"
        ICU_test_data = load_data(data_path, file_name, time_off, hour_mul)
        file_name = "icd9_train_1000_rt.h5"
        ICU_train_data = load_data(data_path, file_name, time_off, hour_mul)
        file_name = "icd9_valid_1000_rt.h5"
        ICU_valid_data = load_data(data_path, file_name, time_off, hour_mul)
    else:
        assert False
    

def get_data(set_name, kind):
    global ICU_test_data
    global ICU_train_data
    global ICU_valid_data
    if(kind=="test"):
        return ICU_test_data
    elif(kind=="train"):
        return ICU_train_data
    else:
        assert kind=="valid"
        return ICU_valid_data

class ExponentialUniformInit(Initializer):
    """
    """
    def __init__(self, range):
        self.range = range

    def sample(self, shape):
        return floatX(np.exp(get_rng().uniform(low=self.range[0],high=self.range[1], size=shape)))

class CustomInit(Initializer):
    def __init__(self, args):
        s = args.period_v3 + args.period_1v3 + args.period_8
        self.p_v3 = args.period_v3 / s
        self.p_1v3 = args.period_1v3 / s
        self.p_8 = args.period_8 / s
        self.vibrate = args.vibrate
    
    def sample(self, shape):
        size = shape[0]
        Print(self.p_v3)
        Print(self.p_1v3)
        Print(self.p_8)
        cnt_v3 = int(size * self.p_v3)
        cnt_1v3 = int(size * self.p_1v3)
        cnt_8 = size - cnt_v3  - cnt_1v3
        period = [0.3333333] * cnt_v3 + [1.333333] * cnt_1v3 + [8] * cnt_8
        vibrate = [0] * size
        for i in range(cnt_v3 + cnt_1v3):
            period[i] += (random.random() * 2 - 1.0) * self.vibrate
        Print("period = %s" %str(period))
        period = floatX(period)
        return period

def get_rnn(event_var, feature_idx, feature_value, mask_var, time_var, arch_size, hour_var, args, num_attention = 0, embed_size=40,
            GRAD_CLIP=100, bn=False, model_type='LSTM', time_feature = False):

    #input layers
    seq_len = args.seq_len
    l_in_event = lasagne.layers.InputLayer(shape=(None, seq_len), input_var = event_var)    
    l_in_feature_idx = lasagne.layers.InputLayer(shape=(None, seq_len, 3), input_var = feature_idx)
    l_in_feature_value = lasagne.layers.InputLayer(shape=(None, seq_len, 3), input_var = feature_value)
    l_mask = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=mask_var)
    l_t = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=time_var)
    if time_feature:
        Print('get_rnn T')
        l_hour =  lasagne.layers.InputLayer(shape = (None, seq_len), input_var = hour_var)

    #embed event
    embed_event = lasagne.layers.EmbeddingLayer(l_in_event, input_size=3418, output_size=embed_size)
    #embed feature_idx
    embed_feature_idx = lasagne.layers.EmbeddingLayer(l_in_feature_idx, input_size=649, output_size=embed_size) 
    #embed feature_value bias
    embed_feature_b = lasagne.layers.EmbeddingLayer(l_in_feature_idx, input_size=649, output_size = 1)
    #embed feature_value trans
    embed_feature_trans = lasagne.layers.EmbeddingLayer(l_in_feature_idx, input_size=649, output_size = 1)

    embed_params = [embed_event.W, embed_feature_idx.W, embed_feature_b.W, embed_feature_trans.W]

    if time_feature:
        Print('add time feature')
        embed_hour = lasagne.layers.EmbeddingLayer(l_hour, input_size = 48, output_size = embed_size)
        embed_params.append(embed_hour.W)
        l_in_merge = MergeEmbeddingLayer(embed_event, embed_feature_idx, embed_feature_b, 
            embed_feature_trans, l_in_feature_value, embed_hour = embed_hour)
    else:
        l_in_merge = MergeEmbeddingLayer(embed_event, embed_feature_idx, embed_feature_b, 
            embed_feature_trans, l_in_feature_value, embed_hour = None)

    #get input_var
    
    
    if model_type=="LSTM":
        l_in_merge = lasagne.layers.ConcatLayer([l_in_merge, lasagne.layers.ReshapeLayer(l_t, [-1, seq_len, 1])], axis=2)

    l_forward = HELSTMLayer(incoming=l_in_merge, time_input=l_t, event_input=embed_event, num_units=arch_size[1],
                            num_attention=num_attention, model=model_type, mask_input=l_mask,
                            ingate=Gate(),
                            forgetgate=Gate(),
                            cell=Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
                            outgate=Gate(),
                            nonlinearity=lasagne.nonlinearities.tanh,
                            grad_clipping=GRAD_CLIP,
                            bn=bn,
                            only_return_final=True,
                            timegate=HELSTMGate(
                                     Period=CustomInit(args),
                                     Shift=lasagne.init.Uniform((0., 1000)),
                                     On_End=lasagne.init.Constant(0.05)))
  
    gate_params = []
    if model_type != 'LSTM':
        gate_params = l_forward.get_gate_params()

    # Softmax
    l_dense = lasagne.layers.DenseLayer(l_forward, num_units=arch_size[2],nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l_out = lasagne.layers.NonlinearityLayer(l_dense, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out, gate_params, embed_params 

def get_train_and_val_fn(inputs, target_var, network, lr):
    # Get network output
    prediction = lasagne.layers.get_output(network)
    # Calculate training accuracy
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
    # Calculate crossentropy between predictions and targets
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Fetch trainable parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Calculate updates for the parameters given the loss
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    # Fetch network output, using deterministic methods
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # Again calculate crossentropy, this time using (test-time) determinstic pass
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    # Also, create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    # Add in the targets to the function inputs
    fn_inputs = inputs + [target_var]
    # Compile a train function with the updates, returning loss and accuracy
    train_fn = theano.function(fn_inputs, [loss, train_acc, prediction], updates=updates)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(fn_inputs, [test_loss, test_acc, test_prediction])

    return train_fn, val_fn

def get_minibatches_idx(n, minibatch_size, shuffle=True):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def valid(train_times, valid_data, test_fn, name):
    # valid the accuracy
    # And a full pass over the validation data:
    valid_err = 0
    valid_acc = 0
    valid_auc = 0
    valid_batches = 0
    y_true_all = []
    y_score_all = []

    valid_event, valid_feature_idx, valid_feature_value, valid_mask, valid_time, valid_hour, valid_label, num_valid, batch_size = valid_data 
    valid_kf = get_minibatches_idx(num_valid, batch_size)
    num_valid_batches = len(valid_kf)
    for _, valid_batch in valid_kf:
        start_time = time.clock()
        b_event = valid_event[valid_batch]
        b_feature_idx = valid_feature_idx[valid_batch]
        b_feature_value = valid_feature_value[valid_batch]
        b_mask = valid_mask[valid_batch]
        b_t = valid_time[valid_batch]
        b_label = valid_label[valid_batch]
        b_hour = valid_hour[valid_batch]
        inputs = [b_event, b_feature_idx, b_feature_value, b_t, b_mask, b_label]
        if args.time_feature:
            inputs.insert(5, b_hour)
        err, acc, pre = test_fn(*inputs)
        y_true = np.asarray(b_label)
        y_score = np.asarray(pre)[:,1]
        y_true_all += list(y_true)
        y_score_all += list(y_score)
        valid_err += err
        valid_acc += acc
        valid_batches += 1

        # Print("name\tBatch {} of {}  : Loss: {} | Accuracy: {} ".format(valid_batches, num_valid_batches, 
        #                                                                                  err, acc*100.))
        # Print("Time:", (time.clock()-start_time))
    valid_err /= valid_batches
    valid_acc = valid_acc * 1.0/valid_batches
    y_true_all = np.asarray(y_true_all)
    y_score_all = np.asarray(y_score_all)
    auc_all = roc_auc_score(y_true_all, y_score_all)
    ap_all = average_precision_score(y_true_all, y_score_all) 
    Print("dataset = {} Train times:{} Loss:{} acc = {}, auROC = {}, auPRC = {}".format(name, train_times, valid_err, valid_acc, auc_all, ap_all))
        
def model(embed, hidden, attention, args, model_type, data_set, name, seed):
    np.random.seed(seed)
    if model_type!="HELSTM":
        attention = 0
    prefix = data_set+"_"
    num_attention = attention
    arch_size = [None, hidden, 2]
    embed_size = embed 
    max_epoch = args.epoch
    batch_size = 128
    valid_freq = args.freq 

    input_event = T.matrix('input_event', dtype='int16')
    input_feature_idx = T.tensor3('input_idx', dtype='int16')
    input_feature_value = T.tensor3('input_value', dtype='float32')
    input_time = T.matrix('input_time', dtype='float32')
    input_mask = T.matrix('input_mask', dtype='int8')
    input_hour = T.matrix('input_hour', dtype='int8')
    input_target = T.ivector('input_target')

    print 'load test data'
    test_event, test_feature_idx, test_feature_value, test_label, test_time, test_mask, test_hours = get_data(data_set, "test")
    num_test = len(test_event)
    #pack them all for further valid use
    test_data = (test_event, test_feature_idx, test_feature_value, test_mask, test_time, test_hours, test_label, num_test, batch_size) 
    
    print 'load train data' 
    train_event, train_feature_idx, train_feature_value, train_label, train_time, train_mask, train_hours = get_data(data_set, "train")
    num_train = len(train_event)
    #pack them all for further valid use
    train_data = (train_event, train_feature_idx, train_feature_value, train_mask, train_time, train_hours, train_label, num_train, batch_size)     

    print 'load valid data'
    valid_event, valid_feature_idx, valid_feature_value, valid_label, valid_time, valid_mask, valid_hours = get_data(data_set, "valid")
    num_valid = len(valid_event)
    #pack them all for further valid use
    valid_data = (valid_event, valid_feature_idx, valid_feature_value, valid_mask, valid_time, valid_hours, valid_label, num_valid, batch_size) 
    
    
    print 'Build network'
    network, gate_params, embed_params = get_rnn(input_event, input_feature_idx, input_feature_value, input_mask, 
                      input_time, arch_size, input_hour, num_attention = num_attention, embed_size = embed_size, 
                      args = args, model_type = model_type, time_feature = args.time_feature)

    print 'Compile'
    inputs = [input_event, input_feature_idx, input_feature_value, input_time, input_mask]
    if args.time_feature:
        inputs.append(input_hour)
    train_fn, test_fn = get_train_and_val_fn(inputs, input_target, network, args.lr)

    print 'Start training'

    train_times = 0
    for epoch in xrange(max_epoch):
        Print("epoch = %d" %epoch)
        train_err = 0
        train_acc = 0
        train_auc = 0
        train_y_true_all = []
        train_y_score_all = []
        train_batches = 0

        kf = get_minibatches_idx(num_train, batch_size)
        num_train_batches = len(kf)
        for _, train_batch in kf:
            train_times += 1
            start_time = time.clock()
            b_event = train_event[train_batch]
            b_feature_idx = train_feature_idx[train_batch]
            b_feature_value = train_feature_value[train_batch]
            b_mask = train_mask[train_batch]
            b_t = train_time[train_batch]
            b_label = train_label[train_batch]
            b_hour = train_hours[train_batch]
            inputs = [b_event, b_feature_idx, b_feature_value, b_t, b_mask, b_label]
            if args.time_feature:
                inputs.insert(5, b_hour)
            err, acc, pre = train_fn(*inputs)
            dat = np.asarray(pre)
            dat_shape = dat.shape
            train_err += err
            train_acc += acc
            train_batches += 1

            print("\tBatch {} of {} in epoch {}: Loss: {} | Accuracy: {} ".format(train_batches, num_train_batches, 
                                                                                            epoch, err, acc*100. ))
            print("Time:", (time.clock()-start_time))
            if(train_times%valid_freq == 0):
                valid(train_times, valid_data, test_fn, 'valid')
                valid(train_times, test_data, test_fn, 'test')
        
    Print('Completed.')

def choose_model(embed, hidden, attention, args, model_type, name, seed):
    name = '{}-{}'.format(name, seed)
    # os.mkdir(name)
    # f = open(name+"/log.txt",'w')
    # f.write("model:{} embed:{} hidden:{} attention:{} period:{} {} seed:{}\n".format(model_type, embed, hidden, attention, period[0], period[1], seed))
    # f.close()
    model(embed, hidden, attention, args, model_type, "ICU",  name, seed) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--emd_dim', type = int, default = 32)
    parser.add_argument('--event_hidden_dim', type = int, default = 64)
    parser.add_argument('--lstm_dim', type = int, default = 32)
    parser.add_argument('--model', type = str, default = "HELSTM") # lstm,helstm,plstm
    parser.add_argument('--time_feature', type = bool, default = False)
    parser.add_argument('--time_off', type = float, default = 3.0)
    parser.add_argument('--name', type = str, default = "exp_HELSTM")
    parser.add_argument('--seed', type = str, default = 1)
    parser.add_argument('--dataset', type = str, default = 'labtest')
    parser.add_argument('--period_v3', type = float, default = 0.25)
    parser.add_argument('--period_1v3', type = float, default = 0.25)
    parser.add_argument('--period_8', type = float, default = 0.5)
    parser.add_argument('--vibrate', type = float, default = .0)
    parser.add_argument('--epoch', type = int, default = 30)
    parser.add_argument('--freq', type = int, default = 500)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--seq_len', type = int, default = 1000)
    parser.add_argument('--hour_mul', type = int, default = 1)
    args = parser.parse_args()
    print args

    load_data_all(args.dataset, args.time_off, args.hour_mul)


    
    choose_model(args.emd_dim, args.event_hidden_dim, args.lstm_dim, args, args.model, args.name, args.seed)
