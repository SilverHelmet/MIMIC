from keras.models import Sequential
from keras.layers.core import Activation, Dense, Masking, Merge, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Input, merge
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, Adam 
import numpy as np
import h5py
import sys
from util import *
from sklearn.metrics import roc_auc_score, accuracy_score
from scripts import gen_fix_segs

def load_data(filepath, seg_filepath = None):
    global event_len
    f = h5py.File(filepath, 'r')
    labels = f['label'][:]
    features = f['feature'][:, -event_len:]
    events = f['event'][:, -event_len:]
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

    
feature_dim = 668
embedding_dim = 128
hiden_dim = 128
event_len = 800
event_dim = 3391


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

def define_simple_sequential_rnn():
    global hiden_dim, event_dim, max_segs

    print "define simple Sequential rnn"
    model = Sequential()
    w_reg = l2(0.0001)
    b_reg = l2(0.0001)
    model.add(Masking(mask_value=0., input_shape = (max_segs, event_dim)))
    model.add(LSTM(input_dim = event_dim, activation='sigmoid', inner_activation='hard_sigmoid', 
        input_length = None, output_dim = hiden_dim,
        W_regularizer = w_reg, b_regularizer = b_reg ))
    model.add(Dense(1, activation = "sigmoid"))
    # model.add(Activation('sigmoid'))
    opt = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])
    print "optimizer config"
    print opt.get_config()
    return model

def define_simple_seg_rnn():
    global hiden_dim
    global event_len, event_dim ,setting, embedding_dim
    print "define simple seg rnn"
    print "embedding_dim = %d" %embedding_dim
    print "hiden_dim = %d" %hiden_dim
    w_reg = l2(0.0001)
    b_reg = l2(0.0001)
    event_input = Input(shape = (max_segs, event_dim), name = "seg_event_input")
    masked = Masking(mask_value=0)(event_input)
    emd = TimeDistributedDense(input_dim = event_dim, output_dim = embedding_dim , name = 'seg_event_embedding', init = "uniform",
        bias = False)(masked)
    lstm = LSTM(input_dim = embedding_dim, output_dim = hiden_dim, inner_activation='hard_sigmoid', activation='sigmoid',
        W_regularizer = w_reg, b_regularizer = b_reg)(emd)
    # lstm = LSTM(input_dim = event_dim, output_dim = hiden_dim, inner_activation='hard_sigmoid', activation='sigmoid',
        # W_regularizer = w_reg, b_regularizer = b_reg, input_length = None)(masked)
    pred = Dense(1, activation = "sigmoid", name = 'prediction')(lstm)
    model = Model(input = event_input, output = pred)
    opt = Adam(lr = 0.001)
    model.compile(optimizer = opt,
        loss = 'binary_crossentropy', 
         metrics=['accuracy'])
    print "opt config:", opt.get_config()
    for layer in model.get_config()['layers']:
        print "\t", layer   
    return model



def define_rnn(disturbance_flag, flat_event_flag, embedding_weights_file = None):
    global feature_dim, embedding_dim, hiden_dim
    global event_len, event_dim, setting
    global lr_hiden, segment_flag, max_segs

    activation = "tanh"
    w_regul_ef = setting.get("w_reg", 0.001)
    b_regul_ef = setting.get("b_reg", 0.001)
    if segment_flag:
        print "max_segs =", max_segs
        event_input = Input(shape = (max_segs, event_dim), name = 'seg_event_input')
    else:
        event_input = Input(shape = (event_len, ), name = "event_input")
    if embedding_weights_file is not None:
        init_weights = load_numpy_array(embedding_weights_file)
        init_weights[0, :] = 0
        init_weights[1, :] = 0
        print "load embedding weights from file [%s]" %embedding_weights_file
        init_weights = [init_weights]
    else:
        init_weights = None
    if segment_flag:
        embedding = TimeDistributed(Dense(embedding_dim, activation='linear', name = 'seg_event_embedding', 
            bias = False, weights = init_weights), name = "event_embedding")(event_input)
    else:
        embedding = Embedding(input_dim = event_dim, output_dim = embedding_dim, mask_zero = True, 
            init_weights = init_weights, name = 'event_embedding')(event_input)
    if segment_flag:
        feature_input = Input(shape = (max_segs, feature_dim), name = 'seg_feature_input')
    else:
        feature_input = Input(shape = (event_len, feature_dim), name = "feature_input")
    disturbance = TimeDistributed(Dense(embedding_dim, W_regularizer=l2(w_regul_ef), activation = activation, 
                    b_regularizer = l2(b_regul_ef), name = 'disturbance'), 
                input_shape = (event_len, feature_dim), name = 'distributed_disturbance')(feature_input)

    if disturbance_flag:
        merged = merge([embedding, disturbance], mode = 'sum', name = 'merged')
    else:
        merged = embedding
    if segment_flag:
        merged = Masking(mask_value=0)(merged)

    lstm = LSTM(output_dim = hiden_dim, name = 'LSTM', 
        W_regularizer=l2(w_regul_ef), activation = "tanh", b_regularizer = l2(b_regul_ef))(merged)

    flat_event_input = Input(shape = (event_dim, ), name = 'flat_event_input')
    if flat_event_flag:
        if lr_hiden:
            print "add lr hiden dim  = %d" %hiden_dim
            hiden_layer = Dense(hiden_dim, bias = True, W_regularizer = l2(w_regul_ef), b_regularizer = l2(b_regul_ef), name = "lr_hiden", activation = activation)(flat_event_input)
            pred = Dense(1, W_regularizer = l2(w_regul_ef), b_regularizer = l2(b_regul_ef), bias = True, name = "flat_event_pred", activation = "sigmoid")(hiden_layer)
        else:
            pred = Dense(1, W_regularizer = l2(w_regul_ef), b_regularizer = l2(b_regul_ef), bias = True, name = "flat_event_pred", activation = "sigmoid")(flat_event_input)
        
    else:
        print "embedding_dim =", embedding_dim
        print "hiden_dim =", hiden_dim
        print "event_len =", event_len
        pred = Dense(1, W_regularizer=l2(w_regul_ef), b_regularizer = l2(b_regul_ef), 
            activation = "sigmoid", name = 'prediction')(lstm)
    if disturbance_flag:
        model = Model(input = [event_input, feature_input], output = pred)
    elif flat_event_flag:
        model = Model(input = flat_event_input, output = pred)
    else:
        model = Model(input = event_input, output = pred)
    # sgd = SGD(lr = 0.3, decay = 0.03, momentum = 0.9, nesterov = True)
    lr = setting.get("lr", 0.3)
    decay = setting.get("decay", 0.03)
    sgd = SGD(lr = lr, decay = decay)
    model.compile(optimizer = sgd, 
        loss = 'binary_crossentropy', 
         metrics=['accuracy'])
    if disturbance_flag:
        print "add disturbance"
    else:
        print "not add disturbance"
    if flat_event_flag:
        print "flag event LR"

    print "w_regularizer_ef =", w_regul_ef
    print 'b_regularizer_ef =', b_regul_ef
    print "activation =", activation
    print "SGD config:", sgd.get_config()
    # (model, to_file='exper/model.graph')
    # for layer in model.get_config()['layers']:
        # print layer   


    return model

def pair_to_vec(feature_pairs):
    global feature_dim
    i = 0
    length = len(feature_pairs)
    vec = np.zeros(feature_dim)
    while i < length:
        if feature_pairs[i+1] != 0:
            index = int(feature_pairs[i])
            value = feature_pairs[i+1]
            vec[index] = value
        i += 2
    return vec


def collect_feature(feature_matrix, st, ed):
    global aggre_mode
    length = len(feature_matrix[0])
    vec = np.zeros(feature_dim)
    for i in range(st, ed):
        feature_pairs = feature_matrix[i]
        idx = 0
        while idx < length:
            if feature_pairs[idx+1] != 0:
                index = int(feature_pairs[idx])
                value = feature_pairs[idx+1]
                vec[index] += value
            idx += 2
    dim = ed - st + 0.0
    if aggre_mode == "ave":
        vec /= dim
    return vec

def merge_fea_by_seg(feature_matrix, split):
    global feature_dim
    seg_fea_matrix = []
    st = 0
    for ed in split:
        if ed == 0:
            seg_fea_matrix.append(np.zeros(feature_dim))
        else:
            seg_fea_matrix.append(collect_feature(feature_matrix, st, ed))
            st = ed
    return seg_fea_matrix


def merge_event_by_seg(event_seq, split):
    global aggre_mode
    st = 0
    event_seqs = []
    for ed in split:
        if ed == 0:
            event_seqs.append([])
        else:
            event_seqs.append(event_seq[st:ed])
            st = ed
    event_cnts = sequence2bow(event_seqs)
    if aggre_mode == "one":
        event_cnts[event_cnts > 1] = 1
    elif aggre_mode == "ave":
        event_cnts = norm_to_prob(event_cnts)
    return event_cnts

def sequence2bow(event):
    global event_dim
    size = len(event)
    ret = np.zeros((size, event_dim))
    for i in range(size):
        seq = event[i]
        for event_idx in seq:
            if event_idx != 0:
                ret[i][event_idx] += 1
    return ret
    # return ret / ret.sum()

def sample_generator(labels, features, events, segs):
    nb_sample = len(labels)
    global batch_size, disturbance, flat_event_flag, segment_flag, add_feature_flag
    while  True:
        i = 0
        while i < nb_sample:
            st = i
            ed = min(i + batch_size, nb_sample)
            # print st, ed
            label = labels[st:ed]
            event = events[st:ed]
            if segment_flag:
                if add_feature_flag:
                    seg_event = []
                    seg_feature_matrixes = [] 
                    for j in range(st, ed):
                        split_seq = segs[j]
                        seg_event.append(merge_event_by_seg(events[j], split_seq))
                        seg_feature_matrixes.append(merge_fea_by_seg(features[j], split_seq))
                    seg_event = np.array(seg_event)
                    seg_feature_matrixes = np.array(seg_feature_matrixes)
                    # print "data size", seg_event.shape, seg_feature_matrixes.shape, label.shape 
                    yield ([seg_event, seg_feature_matrixes] , label)            
                else:
                    seg_event = []
                    for j in range(st, ed):
                        split_seg = segs[j]
                        seg_event.append(merge_event_by_seg(events[j], split_seg))
                    seg_event = np.array(seg_event)
                    yield(seg_event, label)
            elif disturbance:
                feature_matrixes = []
                for j in range(st, ed):
                    feature_matrix = [pair_to_vec(feature) for feature in features[j]]
                    feature_matrixes.append(feature_matrix)
                feature_matrixes = np.array(feature_matrixes) 
                yield ([event, feature_matrixes], label)
            elif flat_event_flag:
                yield(sequence2bow(event), label)
            else:
                yield(event, label)
            i += batch_size 
            if i >= nb_sample:
                i = 0


def load_argv(argv):
    global embedding_dim, hiden_dim
    if len(argv) >= 2:
        embedding_dim = int(argv[1])
    if len(argv) >= 3:
        hiden_dim = int(argv[2])
    if len(argv) >= 4:
        setting = load_setting(argv[3])
    else:
        setting = {}
    return setting

if __name__ == '__main__':
    setting = load_argv(sys.argv)
    seg_mode = setting.get("seg_mode", None)
    train_file = setting["train_dataset"]
    test_file = setting['test_dataset']
    print "train dataset = %s" %train_file
    print "test dataset = %s" %test_file
    if seg_mode is not None:
        print "seg_mode =", seg_mode
        if seg_mode == 'custom':
            train_seg_file = setting["train_seg_file"]
            test_seg_file = setting["test_seg_file"]
        else:
            train_seg_file = gen_fix_segs.infer_path(train_file, seg_mode)
            test_seg_file = gen_fix_segs.infer_path(test_file, seg_mode)
        print "train seg file = [%s]" %train_seg_file
        print "test seg file = [%s]" %test_seg_file
        labels, features, events, ids, segs = load_data(train_file, train_seg_file)
        val_labels, val_feaures, val_events, val_ids, val_segs = load_data(test_file, test_seg_file)
        max_segs = segs.shape[1]
        print "max_segs = %d" %max_segs
    else:
        labels, features, events, ids, = load_data(train_file)
        val_labels, val_feaures, val_events, val_ids = load_data(test_file)
    print "train feature shape =", features.shape
    print "train event shape =", events.shape
    batch_size = int(setting.get("batch_size", 32))
    
    ICU_val_labels = merge_label(val_labels, val_ids)
    # val_generator = sample_generator(val_labels, val_feaures, val_events, val_segs)
    samples_per_epoch = len(labels)
    disturbance = setting.get("disturbance", False)
    flat_event_flag = setting.get("flat_event", False)
    embedding_in = setting.get("embedding_in", None)
    lr_hiden = setting.get('lr_hiden', False)
    segment_flag = setting.get("segment", False)
    aggre_mode = setting.get("aggregation", "sum")
    embedding_out_file = setting.get('embedding_out', None)
    add_feature_flag = setting.get('add_feature', True)


    model = define_simple_seg_rnn()
    # model = define_simple_sequential_rnn()
    # if embedding_in != None:
    #     model = define_rnn(disturbance, flat_event_flag, embedding_in)
    # else:
    #     model = define_rnn(disturbance, flat_event_flag)
    print "validation data size =", len(val_labels)
    print "batch_size =", batch_size
    print 'start trainning'
    early_stopping =EarlyStopping(monitor = "loss", patience = 2)  
    nb_epoch = int(setting.get("nb_epoch", 100))
    weights = {}
    for layer in model.layers:
        name = layer.name
        weights[name] = layer.get_weights()
    for epoch_round in range(nb_epoch):
        model.fit_generator(sample_generator(labels, features, events, segs), samples_per_epoch, 
            nb_epoch = 1, callbacks = [early_stopping])
            # validation_data = val_generator, nb_val_samples = len(val_labels))
        prediction = model.predict_generator(sample_generator(val_labels, val_feaures, val_events, val_segs), val_samples = len(val_labels))
        auc = roc_auc_score(val_labels, prediction)

        ICU_prediction = merge_prob(prediction, val_ids, max)
        ICU_auc = roc_auc_score(ICU_val_labels, ICU_prediction)

        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] =0
        acc = accuracy_score(val_labels, prediction)

        ICU_prediction[ICU_prediction >= 0.5] = 1
        ICU_prediction[ICU_prediction < 0.5] = 0
        ICU_acc = accuracy_score(ICU_val_labels, ICU_prediction)
        print 'Epoch %d/%d, auc = %f, acc = %f, IUC_auc = %f, IUC_acc = %f' %(epoch_round, nb_epoch, auc, acc, ICU_auc, ICU_acc)
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
        # print model.get_weights()
    print "end trainning"
    if embedding_out_file is not None:
        np.save(embedding_out_file, weights["event_embedding"][0])
