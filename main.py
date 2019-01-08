import sys
from models.dataset import Dataset
from models.dataset import sample_generator_tf as sample_generator
import numpy as np
from util import load_argv, Print, calc_AUC, calc_APC
import tensorflow as tf
from tqdm import tqdm
import os
from tensorflow.contrib.layers import xavier_initializer
import time


sess = tf.Session()

def add_argv(setting):
    setting['time_base'] = 'abs'
    setting['event_dim'] = 3418

def make_weight(shape, name = None, lower = -1.0, high = 1.0):
    return tf.Variable(tf.random_uniform(shape, lower, high), name = name)

def make_bias(shape, name = None):
    return make_weight(shape, name, 0.0, 0.0)

def make_event_embedding(event_i, time_i, feature_idx_i, feature_value_i, setting):
    event_size = setting['event_size']
    embedding_dim = setting['model_dim']
    feature_size = setting['feature_size']

    event_embeddings = make_weight([event_size, embedding_dim])

    feature_trans_w = make_weight([feature_size])
    feature_trans_b = make_bias([feature_size])
    feature_embeddings = make_weight([feature_size, embedding_dim])

    e_emded = tf.nn.embedding_lookup(event_embeddings, event_i)
    f_emded = tf.nn.embedding_lookup(feature_embeddings, feature_idx_i)
    f_trans_w = tf.nn.embedding_lookup(feature_trans_w, feature_idx_i)
    f_trans_b = tf.nn.embedding_lookup(feature_trans_b, feature_idx_i)
    f_trans_weight = feature_value_i * f_trans_w + f_trans_b

    dist_feature_embed = f_emded * tf.expand_dims(f_trans_weight, -1)

    merged_emd = e_emded + tf.reduce_sum(dist_feature_embed, 2)
    return merged_emd

def graph_conv_layer(input, head, model_dim):
    attn_dim = model_dim / head
    for head_i in range(head):
        qw = make_weight(shape = (model_dim, attn_dim))
    graph_conv_layer
    


def define_model(setting):
    event_len = setting['event_len']
    event_size = setting['event_dim']
    embedding_dim = setting['embedding_dim']
    batch_size = setting['batch_size']
    lr = setting['lr']

    event_input = tf.placeholder(tf.int32, shape=[None, event_len], name = 'event_input')
    sequence_length = tf.count_nonzero(event_input, axis = -1)
    time_input = tf.placeholder(tf.float32, shape = [None, event_len], name = 'time_input')
    feature_idx_input = tf.placeholder(tf.int32, shape = [None, event_len, 3], name = 'feature_idx_input')
    feature_value_input = tf.placeholder(tf.float32, shape = [None, event_len, 3], name = 'feature_value_input')

    e_emded = make_event_embedding(event_input, time_input, feature_idx_input, feature_value_input, setting)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=embedding_dim, name="LSTM_CELL", 
                                    initializer=xavier_initializer())

    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = e_emded, dtype = tf.float32, sequence_length = sequence_length)


    output = states[1]
    pred_w = make_weight([embedding_dim, 1])
    pred_b = make_bias([1])
    pred_score = tf.squeeze(tf.matmul(output, pred_w) + pred_b, axis = [-1])
    pred_prob = tf.sigmoid(pred_score)

    Y = tf.placeholder(tf.float32, shape = [None], name = 'Y')
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_score, labels=Y))
    pred_label =  tf.cast(pred_prob > 0.5, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, pred_label), tf.float32))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # init_op = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init_op)

    # e = np.random.randint(1, 10, (batch_size, event_len))
    # f_idx = np.random.randint(0, 3, (batch_size, event_len, 3))
    # f_value = np.random.random((batch_size, event_len, 3))
    # y = np.random.randint(0, 2, (batch_size, ))
    # for _ in range(10):
    #     pl, l, p, seq_len, acc, _ = sess.run([pred_label, loss, pred_prob, sequence_length, accuracy, train_step], feed_dict={
    #         event_input: e,
    #         feature_idx_input: f_idx,
    #         feature_value_input: f_value,
    #         Y: y,
    #     })
    #     print '---------'
    #     print l
    #     print seq_len
    #     print p, y, pl
    #     print acc
    ops = {
        'train': train_step,
        'loss': loss,
        'pred_prob': pred_prob,
        'acc': accuracy
    }
    inputs = {
        'event': event_input,
        'feature_idx': feature_idx_input,
        'feature_value': feature_value_input,
        'Y': Y,
    }

    return ops, inputs

def eval_model(dataset, ops, inputs, setting):
    global sess
    loss_op = ops['loss']
    acc_op = ops['acc']
    prob_op = ops['pred_prob']
    event_input = inputs['event']
    feature_idx_input = inputs['feature_idx']
    feature_value_input = inputs['feature_value']
    batch_size = setting['batch_size']
    Y = inputs['Y']

    size = dataset.size
    eval_num = 0
    generator = sample_generator(dataset, setting)
    labels = []
    probs = []
    acc, loss = 0.0, 0.0
    total = ((size - 1) / batch_size) + 1
    Print('evaluate model at dataset %s' % os.path.basename(dataset.dataset_file))
    for batch_sample in generator:
        
        event, f_idx, f_value, label = batch_sample
        b_size = len(label)
        
        b_acc, b_loss, b_prob = sess.run([acc_op, loss_op, prob_op], feed_dict = {
            event_input: event,
            feature_idx_input: f_idx,
            feature_value_input: f_value,
            Y: label
        })
        labels.extend(list(label))
        probs.extend(list(b_prob))
        acc += b_acc * b_size
        loss += b_loss * b_size
        eval_num += b_size
        if eval_num >= size:
            assert eval_num == size
            break 
    acc /= float(size)
    loss /= float(size)
    auc = calc_AUC(probs, labels)
    apc = calc_APC(probs, labels)
    return map(float, (loss, acc, auc, apc))

def train(ops, inputs, setting, datasets):
    global sess
    batch_size = setting['batch_size']
    nb_train_steps = setting['nb_train_step']

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    generator = sample_generator(datasets[0], setting)
    train_op = ops['train']
    loss_op = ops['loss']
    acc_op = ops['acc']
    event_input = inputs['event']
    feature_idx_input = inputs['feature_idx']
    feature_value_input = inputs['feature_value']
    Y = inputs['Y']
    eval_step = setting['eval_step']
    cost_time = .0
    for step, batch_sample in enumerate(generator):
        t0 = time.time()
        if step >= nb_train_steps:
            break
        event, f_idx, f_value, label = batch_sample
        _, acc, loss = sess.run([train_op, acc_op, loss_op], feed_dict = {
            event_input: event,
            feature_idx_input: f_idx,
            feature_value_input: f_value,
            Y: label
        })
        t1 = time.time()
        cost_time += t1 - t0
        Print('step %d cost_time = %d, acc = %.4f, loss = %.4f' % (step, int(cost_time), acc, loss))
        if step > 0 and step % eval_step == 0:
            pattern = "%s result, step %d loss = %.4f, acc = %.4f, AUC = %.4f, APC = %.4f"
            valid_res = eval_model(datasets[1], ops, inputs, setting)
            valid_res = tuple(['valid', step] + valid_res)
            Print(pattern %valid_res)
            
            test_res = eval_model(datasets[2], ops, inputs, setting)
            test_res = tuple(['test', step] + test_res)
            Print(pattern %test_res)

if __name__ == "__main__":
    setting = load_argv(sys.argv)
    add_argv(setting)
    train_file = setting["train_dataset"]
    valid_file = setting['valid_dataset']
    test_file = setting['test_dataset']

    datasets = Dataset.create_datasets(files = [train_file, valid_file, test_file])
    for dataset in datasets:
        dataset.load(True, False, True, setting = setting)

    ops, inputs = define_model(setting)
    train(ops, inputs, setting, datasets)
