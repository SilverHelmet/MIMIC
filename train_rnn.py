from keras.models import Sequential
from keras.layers.core import Activation, Dense, Masking, Merge, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Input, merge
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, Adam 
from models.dataset import Dataset, sample_generator
import numpy as np
import h5py
import sys
from util import *
from scripts import gen_fix_segs
from models.models import SimpleAttentionRNN, EventAttentionLSTM, SegMaskEmbedding

def load_data(filepath, seg_filepath = None):
    f = h5py.File(filepath, 'r')
    labels = f['label'][:]
    features = f['feature'][:,:]
    events = f['event'][:, :]
    ids = f['sample_id'][:]
    f.close()
    if seg_filepath is None:
        print "load dataset from [%s]" %filepath
        return labels, features, events, ids
    if seg_filepath == "infer":
        seg_filepath = filepath.split(".h5")[0] + '_segs.h5'
        print "load dataset from [%s], load segment from [%s]" %(filepath, seg_filepath)
    f = h5py.File(seg_filepath, 'r')
    segs = f['segment'][:]
    f.close()
    return labels, features, events, ids, segs


def parse_int(value):
    try:
        return int(value)
    except Exception, e:
        return None

def load_weights(filepath, event_dim, dim):
    print "load weights from [%s]" %filepath
    vocal_size = 0
    file_vocal_size = 0
    weights = np.zeros((event_dim, dim))
    for line in file(filepath):
        parts = line.strip().split(" ")
        if len(parts) == 2:
            assert parse_int(parts[1]) == dim
            file_vocal_size = parse_int(parts[0])
            continue
        if len(parts) != dim + 1:
            continue
        word = parse_int(parts[0])
        if word == None:
            continue
        vocal_size += 1
        weights[word] = map(float, parts[1:])
    
    assert vocal_size == file_vocal_size-2
    return weights


def make_input(setting):
    max_segs = setting['max_segs']
    event_dim = setting['event_dim']
    rnn_model = setting['rnn']
    if rnn_model == 'attlstm':
        max_seg_length = setting.get('max_seg_length', None)
        assert max_seg_length
        return Input(shape = (max_segs, max_seg_length), name = 'seg event input')
    else:
        return Input(shape = (max_segs, event_dim), name = 'seg event input')

def define_simple_seg_rnn(setting):
    hidden_dim = setting['hidden_dim']
    event_len = setting['event_len']
    event_dim = setting['event_dim']
    embedding_dim = setting['embedding_dim']

    print "define simple seg rnn"
    print "embedding_dim = %d" %embedding_dim
    print "hidden_dim = %d" %hidden_dim
    l2_cof = setting["l2_reg_cof"]
    print "l2 regulazation cof = %f" %l2_cof
    w_reg = l2(l2_cof)
    b_reg = l2(l2_cof)
    u_reg = l2(l2_cof)
    
    event_input = make_input(setting)
    attention = setting["attention"]
    rnn_model = setting["rnn"]
    print "rnn = %s" %rnn_model

    if rnn_model == 'gru':
        masked = Masking(mask_value=0)(event_input)
        embedding = TimeDistributed(Dense(embedding_dim, activation='linear', name = 'seg_event_embedding', 
            bias = False), name = "event_embedding")(masked)
        rnn = GRU(output_dim = hidden_dim, inner_activation = 'hard_sigmoid', activation = 'sigmoid', consume_less = 'gpu',
            W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, input_length = None, return_sequences = attention)(embedding)
    elif rnn_model == "lstm":
        masked = Masking(mask_value=0)(event_input)
        # embedding = TimeDistributed(Dense(embedding_dim, activation='linear', name = 'seg_event_embedding', 
        # bias = False), name = "event_embedding")(masked)
        rnn = LSTM(output_dim = hidden_dim, inner_activation = 'hard_sigmoid', activation='sigmoid', consume_less = 'gpu',
            W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, input_length = None, return_sequences = attention)(masked)
    elif rnn_model == "attlstm":
        emd = SegMaskEmbedding(mask_value = 0, input_dim = event_dim, output_dim = embedding_dim, name = "embedding")(event_input)
        rnn = EventAttentionLSTM(att_hidden_dim = 128, output_dim = hidden_dim, inner_activation='hard_sigmoid', activation='sigmoid', consume_less = 'gpu',
            W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, input_length = None, return_sequences = attention)(emd)
    else:
        print "error"
    if attention:
        print "add attention"
        rnn = SimpleAttentionRNN(rnn)
    pred = Dense(1, activation = "sigmoid", name = 'prediction')(rnn)
    model = Model(input = event_input, output = pred)
    opt = Adam(lr = 0.001)
    model.compile(optimizer = opt,
        loss = 'binary_crossentropy', 
         metrics=['accuracy'])
    print "opt config:", opt.get_config()
    for layer in model.get_config()['layers']:
        print "\t", layer   
    return model


def default_setting():
    '''
        return the setting consisting of defualt args
    '''
    setting = {
        'seg_mode': None,
        "batch_size": 32,
        'attention': False, 
        'disturbance': False,   # add feature disturbance
        'segment_flag': False,  # split event seq to event segment
        'aggregation': 'sum',    # only useful when segment_flag is True

        'feature_dim': 648,
        'embedding_dim': 128, 
        'hidden_dim': 128,
        'event_len': 1000,
        'event_dim': 3418,

        'l2_reg_cof': 0.0001,
        
        'rnn': 'lstm',
        'nb_epoch': 100,
    }
    return setting


def load_argv(argv):
    if len(argv) >= 2:
        setting = load_setting(argv[1], default_setting())
    else:
        setting = default_setting()
    return setting

if __name__ == '__main__':
    setting = load_argv(sys.argv)
    train_file = setting["train_dataset"]
    valid_file = setting['valid_dataset']
    test_file = setting['test_dataset']
    print "train dataset = %s" %train_file
    print "valid dataset = %s" %valid_file
    print "test dataset = %s" %test_file
    seg_mode = setting['seg_mode']
    assert seg_mode is not None
    if seg_mode is not None:
        print "seg_mode =", seg_mode
        if seg_mode == 'custom':
            train_seg_file = setting["train_seg_file"]
            valid_seg_file = setting['valid_seg_file']
            test_seg_file = setting["test_seg_file"]
        else:
            train_seg_file = gen_fix_segs.infer_path(train_file, seg_mode)
            valid_seg_file = gen_fix_segs.infer_path(valid_file, seg_mode)
            test_seg_file = gen_fix_segs.infer_path(test_file, seg_mode)

        print "train seg file = [%s]" %train_seg_file
        print "valid seg file = [%s]" %valid_seg_file
        print "test seg file = [%s]" %test_seg_file
        datasets = Dataset.create_datasets(files = [train_file, valid_file, test_file], segs = [train_seg_file, valid_seg_file, test_seg_file])
        for dataset in datasets:
            dataset.load()
        max_segs = datasets[0].segs.shape[1]
        setting['max_segs'] = max_segs
        print "max_segs = %d" %max_segs
    print "train feature shape =", datasets[0].features.shape
    print "train event shape =", datasets[0].events.shape
    
    
    disturbance = setting['disturbance']
    segment_flag = setting['segment']


    model = define_simple_seg_rnn(setting)
    for dataset in datasets:
        print "\tdataset size = %d" %len(dataset.labels)
    print "batch_size =", setting['batch_size']
    print 'start trainning'
    nb_epoch = setting['nb_epoch']
    weights = {}
    for layer in model.layers:
        name = layer.name
        weights[name] = layer.get_weights()
    max_merged_auc = 0
    for epoch_round in range(nb_epoch):
        model.fit_generator(sample_generator(datasets[0], setting), datasets[0].size, nb_epoch = 1, verbose = 1)
        
        val_eval = datasets[1].eval(model, setting)

        print 'Epoch %d/%d, validation acc = %f, auc = %f, merged_acc = %f, merged_auc = %f' \
            %(epoch_round + 1, nb_epoch, val_eval[0], val_eval[1], val_eval[2], val_eval[3])
        if val_eval[3] > max_merged_auc:
            print "new max max_merged_auc"
            test_eval = datasets[2].eval(model, setting)
            print 'round %d test acc = %f, auc = %f, merged_acc = %f, merged_auc = %f'  %(epoch_round + 1, test_eval[0], test_eval[1], test_eval[2], test_eval[3])
            max_merged_auc = val_eval
        new_weights = {}
        
        for layer in model.layers:
            name = layer.name
            new_weights[name] = layer.get_weights()
        if weights is not None:
            diff = {}
            for name in new_weights:
                new_weight = new_weights[name]
                weight = weights[name]
                for i in range(len(weight)):
                    diff[name + "_" + str(weight[i].shape)] = np.abs(weight[i] - new_weight[i]).mean()
            for name in sorted(diff.keys()):
                print '\t', name, "mean diff =", diff[name]
        weights = new_weights
    print "end trainning"
