## Usage
`python train_rnn.py setting_1 setting_2 ... setting_n`

setting_1, setting_2, ..., setting_n are setting files of the model. 
The model will load arguments in these files sequentially， 
An example setting file is as follows.

        # model
        disturbance=False
        attention=True
        rnn=attlstm

        #model args
        embedding_dim=64
        hidden_dim=64
        att_hidden_dim=64


        # model training 
        batch_size=32
        nb_epoch=10
        l2_reg_cof=0.0001
        lr=0.001


        # dataset
        train_dataset=death_exper/death_train_1000.h5
        valid_dataset=death_exper/death_valid_1000.h5
        test_dataset=death_exper/death_test_1000.h5

        # segment
        aggregation=sum
        seg_mode=custom
        train_seg_file=train_segs.h5
        valid_seg_file=valid_segs.h5
        test_seg_file=valid_segs.h5
        

We introduce meaning of some important arguments here.
`train_dataset`, `valid_dataset` and `test_dataset` are training, validation and test datasets seperately. 
Similary, `train_seg_file`, `valid_seg_file` and `test_seg_file` are segmentation of corresponding datasets.

The argument `embedding_dim` corresponds the dimension of embedding layer in the model. `hidden_dim` is the dimension of the hiiden layer of RNN. 
`att_hidden_dim` is the dimension of  hidden layer in event-attention, which is only useful when we use event-attention mechanism.

The valid values of the argument `attention` are ['True' | 'False'], it controls whether use temporal attention. 

The valid values of the argument `rnn` are ['attlstm' | 'rnn' | 'lstm'], it specify the model type of RNN. `rnn=attlstm` specify the LSTM with event-attention.

The argument `disturbance` controls whether use numeric feature.

Arguments in 'model args' section controls the process of model training.












