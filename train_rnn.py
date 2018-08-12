from keras.models import Sequential
from keras.layers.core import Activation, Dense, Masking, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Input, merge, Merge
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, Adam 
import keras.backend as K
from models.dataset import Dataset, sample_generator, print_eval
import numpy as np
import h5py
import sys
from util import *
from scripts import gen_fix_segs, norm_feature
from models.models import SimpleAttentionRNN, SimpleAttentionRNN2, EventAttentionLSTM, EventAttentionGRU, SegMaskEmbedding, make_CNN1D, GCNMaskedGlobalMaxPooling1D, MaskedGlobalMaxPooling1D
from models.helstm import HELSTM, FeatureEmbeddding
from gcn.graph_attention_layer import GraphAttention


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
    max_segs = setting.get('max_segs', -1)
    event_dim = setting['event_dim']
    rnn_model = setting['rnn']
    use_gcn = setting['GCN']
    if use_gcn or setting['GCN_Seg'] or rnn_model == 'dlstm' or True:
        return Input(shape = (setting['event_len'], ),  dtype = 'int32', name = 'event input')
    if rnn_model == 'attlstm' or rnn_model == "attgru":
        max_seg_length = setting['max_seg_length']
        return Input(shape = (max_segs, max_seg_length), name = 'seg event input')
    else:
        return Input(shape = (max_segs, event_dim), name = 'seg event input')

def define_simple_seg_rnn(setting):
    hidden_dim = setting['hidden_dim']
    event_len = setting['event_len']
    event_dim = setting['event_dim']
    embedding_dim = setting['embedding_dim']
    att_hidden_dim = setting['att_hidden_dim']
    disturbance = setting['disturbance']
    feature_dim = setting.get('feature_dim', 0)
    max_seg_length = setting.get('max_seg_length', 0)
    gcn_numeric_feature = setting['gcn_numeric_feature']
    gcn_numeric_width = setting.get('gcn_numeric_width', 1)
    gcn_numric_feature_hidden_dim = setting.get('gcn_numric_feature_hidden_dim', 0)
    gcn_hidden_dim = setting['gcn_hidden_dim']
    gcn_hidden_dim2 = setting.get('gcn_hidden_dim2', 0)
    gcn_num_head = setting['gcn_num_head']
    

    print "define simple seg rnn"
    print "embedding_dim = %d" %embedding_dim
    print "hidden_dim = %d" %hidden_dim
    if disturbance:
        print "add feature dim"
    l2_cof = setting["l2_reg_cof"]
    print "l2 regulazation cof = %f" %l2_cof
    w_reg = l2(l2_cof)
    b_reg = l2(l2_cof)
    u_reg = l2(l2_cof)
    
    event_input = make_input(setting)
    inputs = [event_input]
    attention = setting["attention"]
    rnn_model = setting["rnn"]
    print "rnn = %s" %rnn_model
    gcn_flag = setting['GCN']
    gcn_seg = setting['GCN_Seg']
    gcn_mode = setting['gcn_mode']
    post_model = setting['post_model']
    time_feature_flag = setting['time_feature']
    emd_post_fc_flag = setting['emd_post_fc']
    emd_post_fc_fc_flag = setting['emd_post_fc_fc']
    if gcn_flag:
        print 'ues graph convolution network'



    if gcn_flag or gcn_numeric_feature or True:
        e_embedding = Embedding(input_dim = event_dim, output_dim = embedding_dim, mask_zero = True, name = 'embedding')(event_input)
        emd_dim = embedding_dim
        if gcn_numeric_feature:
            if setting['numeric_feature_type'] == "HELSTM":
                print 'user helstm event embedding'
                numeric_feature_num = setting['numeric_feature_num']
                num_feature_idx = Input(shape = (event_len, numeric_feature_num), name = 'numeric feature idx')
                num_feature_value = Input(shape = (event_len, numeric_feature_num), name = 'numeric feature value')
                num_feature_emd = FeatureEmbeddding(input_dim = feature_dim, output_dim = embedding_dim, name = 'numeric feture embedding')([num_feature_idx, num_feature_value])
                inputs.append(num_feature_idx)
                inputs.append(num_feature_value)
                embedding = merge(inputs = [e_embedding, num_feature_emd], mode = 'sum', name = 'e_f_embedding')
                emd_dim = embedding_dim
            else:    
                num_feature = Input(shape = (event_len, (gcn_numeric_width * 2 + 1) * feature_dim), name = 'numeric feature')
                inputs.append(num_feature)
                num_emd = TimeDistributedDense(gcn_numric_feature_hidden_dim, activation = 'tanh', name = 'numeric feature embedding')(num_feature)
                embedding = merge(inputs = [e_embedding, num_emd], name = 'merged embedding', mode = 'concat')
                emd_dim = embedding_dim + gcn_numric_feature_hidden_dim
        else:
            embedding = e_embedding
            emd_dim = embedding_dim
        
        if time_feature_flag:
            feature_t = setting['time_feature_type']
            feature_t2gate = setting.get('tfea2gate', False)
            if feature_t == 'concat':
                time_feature_dim = setting['time_feature_dim']
                time_hour_input =  Input(shape = (event_len, ), name = 'hour input')
                inputs.append(time_hour_input)
                time_emd = Embedding(input_dim = 24, output_dim = time_feature_dim, 
                    mask_zero = False, name = 'time feature embedding')(time_hour_input)
                embedding = Merge(mode = 'concat', name = 'event embedding with time_emd')([embedding, time_emd])
                emd_dim = emd_dim + time_feature_dim
            else:
                time_feature_dim = embedding_dim
                time_hour_input = Input(shape = (event_len, ), name = 'hour input')
                inputs.append(time_hour_input)
                time_emd = Embedding(input_dim = 24, output_dim = time_feature_dim, 
                    mask_zero = False, name = 'time feature embedding')(time_hour_input)
                embedding = Merge(mode = 'sum', name = 'event embedding with time_emd')([embedding, time_emd])
                if feature_t2gate:
                    e_embedding = Merge(mode = 'sum', name = 'time feature to time gate')([e_embedding, time_emd])

          
        if gcn_flag:
            print('gcn input dim = %d' %emd_dim)
            edge_mat = Input(shape = (event_len, event_len), dtype = 'float32', name = 'adjacent matrix')
            inputs.append(edge_mat)
            gcn_layer = GraphAttention(F1 = gcn_hidden_dim, F2 = gcn_hidden_dim2, 
                    nb_event = event_dim, mask_zero = not gcn_seg,
                    attention_mode = gcn_mode, input_dim = emd_dim,attn_heads=gcn_num_head, 
                    attn_dropout = 1.0, activation = 'tanh', batch_size = setting['batch_size'],
                    kernel_regularizer=l2(l2_cof), name = 'gcn')
            gcn = gcn_layer([embedding, edge_mat, event_input])
            
        else:
            gcn = embedding

        # if setting.get('gcn_gcn', False):
        #     gcn = GraphAttention(F1 = gcn_hidden_dim, F2 = gcn_hidden_dim2, 
        #             nb_event = event_dim, 
        #             attention_mode = gcn_mode, input_dim = gcn_layer.output_dim, attn_heads=gcn_num_head, 
        #             attn_dropout = 1.0, activation = 'tanh', batch_size = setting['batch_size'],
        #             kernel_regularizer=l2(l2_cof), name = 'gcn_gcn')([gcn, edge_mat, event_input])

        # if setting.get('gcn_dense', False):
        #     gcn = TimeDistributedDense(setting.get('gcn_dense_dim', 64), activation = 'tanh', name = 'gcn_dense')(gcn)

        if emd_post_fc_fc_flag:
            fc_hidden_dim = setting['emd_post_fc_hidden_dim']
            gcn = TimeDistributedDense(output_dim = fc_hidden_dim, name = 'emd post fc',
                            activation = 'tanh', W_regularizer = l2(l2_cof), b_regularizer = l2(l2_cof))(gcn)
            gcn = TimeDistributedDense(output_dim = emd_dim, name = 'emd post fc fc',
                            activation = 'tanh', W_regularizer = l2(l2_cof), b_regularizer = l2(l2_cof))(gcn)
        elif emd_post_fc_flag:
            fc_dim = setting.get('emd_post_fc_dim', emd_dim)
            gcn = TimeDistributedDense(output_dim = fc_dim, name = 'emd post fc',
                            activation = 'tanh', W_regularizer = l2(l2_cof), b_regularizer = l2(l2_cof))(gcn)



        if gcn_seg:
            seg_mat = Input(shape = (max_segs, max_seg_length, event_len), name = 'segment matrix')
            inputs.append(seg_mat)
            gcn = merge(inputs = [seg_mat, gcn], mode = 'dot', name = 'segment', dot_axes = [3, 1])
            gcn = GCNMaskedGlobalMaxPooling1D(name = 'max pooling')(gcn)

        if post_model == 'gcn':
            seg_edge_mat = Input(shape = (max_segs, max_segs), name = 'post edge matrix')
            inputs.append(seg_edge_mat)
            post_gcn_num_head = setting.get("post_gcn_num_head", 8)
            rnn = GraphAttention(F1 = hidden_dim / post_gcn_num_head, activation='tanh', 
                    attention_mode = 11, attn_dropout = 1.0, attn_heads = post_gcn_num_head,
                    kernel_regularizer = l2(l2_cof), name = 'post_gcn',
                    mask_zero = True)([gcn, seg_edge_mat, event_input])
            # rnn = SimpleAttentionRNN(rnn)
            rnn = MaskedGlobalMaxPooling1D(name = 'max pooling after post_gcn')(rnn)
        elif post_model == 'LSTM':
            rnn = LSTM(output_dim = hidden_dim, inner_activation = 'tanh', activation='tanh', consume_less = 'gpu',
                W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, 
                input_length = None, return_sequences = False, name = 'lstm', unroll = False)(gcn)
        elif post_model == "HELSTM":
            time_input = Input(shape = (event_len, 1), name = 'time input')
            inputs.append(time_input)
            event_with_time = Merge(mode = 'concat', concat_axis = 2, name = 'embedding & time')([e_embedding, gcn, time_input])
            event_hidden_dim = setting.get('event_hidden_dim', embedding_dim)
            print 'event_hidden_dim = %d' %(event_hidden_dim)
            rnn = HELSTM(output_dim = hidden_dim, event_emd_dim = embedding_dim, event_hidden_dim = event_hidden_dim, inner_activation = 'tanh', activation = 'tanh', 
                 W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, 
                 input_length = None, return_sequences = False, name = 'helstm', 
                 setting = setting, off_slope = 1e-3)(event_with_time)
    else:
        assert False
    # elif rnn_model == "dlstm":
    #     embedding = Embedding(input_dim = event_dim, output_dim = embedding_dim, mask_zero = True, name = "embedding")(event_input)
    #     rnn = LSTM(output_dim = hidden_dim, inner_activation = 'hard_sigmoid', activation='sigmoid', consume_less = 'gpu',
    #         W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, 
    #         input_length = None, return_sequences = False, name = 'rnn')(embedding)
    # elif rnn_model == "cnn":
    #     masked = Masking(mask_value=0.)(event_input)
    #     embedding = TimeDistributed(Dense(embedding_dim, activation='linear', name = 'embedding', 
    #         bias = False), name = "event_embedding")(masked)
    #     cnn = make_CNN1D(filter_lengths = (2,3,4,5,6,7,8), feature_maps = (100, 100, 100, 100, 100, 100, 100), 
    #                     emd = embedding, max_segs = setting['max_segs'], l2_reg_cof = l2_cof, drop_rate = setting['cnn_drop_rate'])
    #     # lazy 
    #     rnn = cnn
    # elif rnn_model == 'gru':
    #     masked = Masking(mask_value=0)(event_input)
    #     embedding = TimeDistributed(Dense(embedding_dim, activation='linear', name = 'embedding', 
    #         bias = False), name = "event_embedding")(masked)
    #     if disturbance:
    #         feature_input = Input(shape = (max_segs, feature_dim), name = 'feature input')
    #         feature_layer = TimeDistributedDense(output_dim = embedding_dim, name = 'feature_embedding')(feature_input)
    #         embedding = merge(inputs = [embedding, feature_layer], mode = 'sum', name = 'embedding with feature')
    #         inputs = [event_input, feature_input]
    #     rnn = GRU(output_dim = hidden_dim, inner_activation = 'hard_sigmoid', activation = 'sigmoid', consume_less = 'gpu',
    #         W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, 
    #         input_length = None, return_sequences = attention, name = 'rnn')(embedding)
    # elif rnn_model == "lstm":
    #     masked = Masking(mask_value=0)(event_input)
    #     embedding = TimeDistributed(Dense(embedding_dim, activation='linear', name = 'embedding', 
    #     bias = False), name = "event_embedding")(masked)
    #     if disturbance:
    #         feature_input = Input(shape = (max_segs, feature_dim), name = 'feature input')
    #         feature_layer = TimeDistributedDense(output_dim = embedding_dim, name = 'feature_embedding')(feature_input)
    #         embedding = merge(inputs = [embedding, feature_layer], mode = 'sum', name = 'embedding with feature')
    #         inputs = [event_input, feature_input]
    #     rnn = LSTM(output_dim = hidden_dim, inner_activation = 'hard_sigmoid', activation='sigmoid', consume_less = 'gpu',
    #         W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, 
    #         input_length = None, return_sequences = attention, name = 'rnn')(embedding)
    # elif rnn_model == "attgru":
    #     embedding = SegMaskEmbedding(mask_value = 0, input_dim = event_dim, output_dim = embedding_dim, name = "embedding")(event_input)
    #     if disturbance:
            
    #         feature_input = Input(shape = (max_segs, max_seg_length, feature_dim), name = 'feature input')
    #         feature_layer = TimeDistributed(TimeDistributedDense(output_dim = embedding_dim), name = 'feature_embedding')(feature_input)
    #         embedding = merge(inputs = [embedding, feature_layer], mode = 'sum', name = 'embedding with feature')
    #         inputs = [event_input, feature_input]
    #     rnn = EventAttentionGRU(att_hidden_dim = att_hidden_dim, output_dim = hidden_dim, inner_activation='hard_sigmoid', activation='sigmoid', consume_less = 'gpu',
    #         W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, 
    #         input_length = None, return_sequences = attention, name = 'rnn')(embedding)
        
    # elif rnn_model == "attlstm":
    #     embedding = SegMaskEmbedding(mask_value = 0, input_dim = event_dim, output_dim = embedding_dim, name = "embedding")(event_input)
    #     if disturbance:
    #         max_seg_length = setting['max_seg_length']
    #         feature_input = Input(shape = (max_segs, max_seg_length, feature_dim), name = 'feature input')
    #         feature_layer = TimeDistributed(TimeDistributedDense(output_dim = embedding_dim), name = 'feature_embedding')(feature_input)
    #         embedding = merge(inputs = [embedding, feature_layer], mode = 'sum', name = 'embedding with feature')
    #         inputs = [event_input, feature_input]
    #     rnn = EventAttentionLSTM(att_hidden_dim = att_hidden_dim, output_dim = hidden_dim, inner_activation='hard_sigmoid', activation='sigmoid', consume_less = 'gpu',
    #         W_regularizer = w_reg, U_regularizer = u_reg, b_regularizer = b_reg, 
    #         input_length = None, return_sequences = attention, name = 'rnn')(embedding)

    if attention:
        print "add attention"
        rnn = SimpleAttentionRNN(rnn)

    if setting['static_feature']:
        static_feature_input = Input(shape = (setting['static_feature_size'], ), name = 'static feature input')
        # static_feature = Dense(128, activation = "tanh", name = 'W', W_regularizer = l2(l2_cof), b_regularizer = l2(l2_cof))(static_feature_input)
        inputs.append(static_feature_input)
        linear_features = merge(inputs = [rnn, static_feature_input], mode = 'concat', name = 'rnn_and_staticfeature')
        print "add static feature with size = %d" %(setting['static_feature_size'])
    else:
        linear_features = rnn
    
    if len(inputs) == 0:
        inputs = inputs[0]
    pred = Dense(1, activation = "sigmoid", name = 'prediction', W_regularizer = l2(l2_cof), b_regularizer = l2(l2_cof))(linear_features)

    model = Model(input = inputs, output = pred)
    lr = setting['lr']
    opt = Adam(lr = lr)
    model.compile(optimizer = opt,
        loss = 'binary_crossentropy', 
         metrics=['accuracy'])
    print "opt config:", opt.get_config()
    # for layer in model.get_config()['layers']:
    #     print "\t", layer   
    model.summary()
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
        'segment_flag': True,  # split event seq to event segment
        'aggregation': 'sum',    # only useful when segment_flag is True
        'static_feature': False,

        'embedding_dim': 128, 
        'hidden_dim': 128,
        'event_len': 1000,
        'att_hidden_dim': 128, 

        'l2_reg_cof': 0.0001,
        'lr':0.001,
        
        'rnn': 'lstm',
        'nb_epoch': 100,
        'cnn_drop_rate': 0.5,

        'GCN': False,
        'gcn_hidden_dim': 64,
        'gcn_num_head':1,
        'GCN_Seg': False,
        'gcn_numeric_feature': False,
        'numeric_feature_num': 3,
        'numeric_feature_type': "HELSTM",
        "normed_feature": True,
        'gcn_numeric_width': 1,
        'gcn_time_width': 0.5,
        'gcn_mode': -1,

        "post_model": "LSTM",

        "load_time": True,

        'time_feature': False,
        "time_feature_type": 'concat',
        "time_feature_dim": 8,

        "emd_post_fc": False,
        "emd_post_fc_fc": False,
        "emd_post_fc_hidden_dim": 128,

        'sample_generator': True,
        'eventxtime': False,
    }
    return setting

def split_batch_index(size, nb_split):
    idxs = np.random.permutation(size)
    step = size / nb_split
    st_ed_pair = []
    st = 0
    for _ in range(nb_split - 1):
        ed = min(size, st + step)
        st_ed_pair.append((st, ed))
        st += step
    st_ed_pair.append((st, size))
    for st, ed  in st_ed_pair:
        yield idxs[st:ed]




def load_argv(argv):
    setting = default_setting()
    if len(argv) >= 2:
        for arg in argv[1:]:
            setting = load_setting(arg, setting)
    return setting

if __name__ == '__main__':
    setting = load_argv(sys.argv)
    train_file = setting["train_dataset"]
    valid_file = setting['valid_dataset']
    test_file = setting['test_dataset']

    if setting.get('norm_feature', False):
        train_file = norm_feature.infer_outpath(train_file)
        valid_file = norm_feature.infer_outpath(valid_file)
        test_file = norm_feature.infer_outpath(test_file)
    print "train dataset = %s" %train_file
    print "valid dataset = %s" %valid_file
    print "test dataset = %s" %test_file
    seg_mode = setting['seg_mode']
    assert seg_mode is not None
    if seg_mode is not None:
        print "seg_mode =", seg_mode
        if seg_mode == 'custom':
            train_seg_file = setting.get("train_seg_file", None)
            valid_seg_file = setting.get('valid_seg_file', None)
            test_seg_file = setting.get("test_seg_file", None)
        else:
            # train_seg_file = gen_fix_segs.infer_path(train_file, seg_mode)
            # valid_seg_file = gen_fix_segs.infer_path(valid_file, seg_mode)
            # test_seg_file = gen_fix_segs.infer_path(test_file, seg_mode)
            Print('error in seg_mode')
            sys.exit()

        
        print "train seg file = [%s]" %train_seg_file
        print "valid seg file = [%s]" %valid_seg_file
        print "test seg file = [%s]" %test_seg_file
        datasets = Dataset.create_datasets(files = [train_file, valid_file, test_file], segs = [train_seg_file, valid_seg_file, test_seg_file])
        use_GCN = setting['GCN']
        for dataset in datasets:
            dataset.load(load_static_feature = setting['static_feature'], 
            load_time = setting['load_time'], load_transfer_time = setting['load_time'],
            load_normed_feature = setting['normed_feature'], setting = setting)

        setting['event_dim'] = int(datasets[0].events.max() + 1)
        if setting['eventxtime']:
            setting['event_dim'] *= 24
        print "get event_dim from dateset as %d" %setting['event_dim']
        if train_seg_file:
            max_segs = datasets[0].segs.shape[1]
            setting['max_segs'] = max_segs
            setting['max_seg_length'] = datasets[0].max_seg_length
            print "max_segs = %d" %max_segs
            print "max_seg_length = %d" %setting['max_seg_length']


        if not setting['sample_generator']:
            print 'generate model sample all at once'
            for dataset in datasets:
                dataset.generate_model_input(setting)
    print "train feature shape =", datasets[0].features.shape
    print "train event shape =", datasets[0].events.shape
    
    
    

    model = define_simple_seg_rnn(setting)
    for dataset in datasets:
        print "\tdataset size = %d" %len(dataset.labels)
    print "batch_size =", setting['batch_size']
    print 'start trainning'
    early_stop_round = 5
    last_hit_round = 0
    nb_epoch = setting['nb_epoch']
    weights = {}
    for layer in model.layers:
        name = layer.name
        weights[name] = layer.get_weights()
    max_auc = 0
    # K.set_learning_phase(0)
    # for x, y in sample_generator(datasets[0], setting, shuffle = True):
    #     for xs in x:
    #         print xs.shape
    #     print y.shape
    #     model.predict(x, y)
    for epoch_round in range(nb_epoch):
        if epoch_round - last_hit_round -1 >= early_stop_round:
            print "early stop at round %d" %(epoch_round + 1)
            break
        if "model_out" in setting:
            model.save(setting['model_out'] + '.round%d' %(epoch_round + 1))

        if epoch_round >= 2:
            nb_batch = 10
        else:
            nb_batch = 1
        for batch_idx, train_index in enumerate(split_batch_index(datasets[0].size, nb_batch), start = 1):
            if setting['sample_generator']:
                model.fit_generator(sample_generator(datasets[0], setting, shuffle = True, train_index = train_index), len(train_index), nb_epoch = 1, verbose = 1)
            else:
                X_slice = datasets[0].get_inputs(train_index)
                Y_slice = datasets[0].get_labels(train_index)
                model.fit(X_slice, Y_slice, batch_size = setting['batch_size'], nb_epoch=1, verbose=1)
        
            val_eval = datasets[1].eval(model, setting)
            print_eval('Epoch %d/%d Batch %d/%d, validation' %(epoch_round+1, nb_epoch, batch_idx, nb_batch), val_eval)
        
            if val_eval[1] > max_auc or True:
                last_hit_round = epoch_round
                test_eval = datasets[2].eval(model, setting)
                if val_eval[1] > max_auc:
                    max_auc = val_eval[1]
                    print "new max max_auc"
                    print_eval("round %d batch %d" %(epoch_round+1, batch_idx), test_eval)
                else:
                    print_eval("round-%d batch %d" %(epoch_round+1, batch_idx), test_eval)
                
    print "end trainning"

