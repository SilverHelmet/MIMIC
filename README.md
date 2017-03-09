## Usage
To train a model, just run `python train_rnn.py setting_1 setting_2 ... setting_n`.

setting_1, setting_2, ..., setting_n are setting files of the model. 
The model will load arguments in these files sequentially.
An example setting file is as follows:

        # model
        disturbance=False
        attention=True
        rnn=attlstm

        #model args
        embedding_dim=64
        hidden_dim=64
        att_hidden_dim=64
        feature_dim=120

        # model training 
        batch_size=32
        nb_epoch=10
        l2_reg_cof=0.0001
        lr=0.001

        # dataset
        train_dataset=train.h5
        valid_dataset=valid.h5
        test_dataset=test.h5

        # segment
        aggregation=sum
        seg_mode=custom
        train_seg_file=train_segs.h5
        valid_seg_file=valid_segs.h5
        test_seg_file=valid_segs.h5

        # model output
        model_out=out.model
        

We introduce meaning of some important arguments here.

`train_dataset`, `valid_dataset` and `test_dataset` are training, validation and test datasets seperately. 
Similary, `train_seg_file`, `valid_seg_file` and `test_seg_file` are segmentation of corresponding datasets.

The valid values of the argument `attention` are ['True' | 'False'], it controls whether using temporal attention. 
The valid values of the argument `rnn` are ['attlstm' | 'rnn' | 'lstm'], it specify the model type of RNN. `rnn=attlstm` specify the type of RNN is LSTM with event-attention.
The argument `disturbance` controls whether using numeric feature.

The argument `embedding_dim` corresponds the dimension of embedding layer in the model. `hidden_dim` is the dimension of the hidden layer of RNN. 
`att_hidden_dim` is the dimension of  hidden layer in event-attention, which is only useful when you set `rnn` as 'attlstm'.
`feature_dim` is the dimentsion of numeric feature, which is only useful when you set `disturbance` as `True`.



Arguments in `model args` section control the process of model training.

If the value of `model_out` is set, the program will output the final model to local at last.

### Dataset format
The datasets loaded to model should in HDF5 format, we recommand using h5py library to read/write files in HDF5 format.

An complete dataset file should have following attributes:

        label: labels of samples, shape (nb_samples, )
        event: event sequences of samples, shape (nb_samples, max_event_len)
               you can use 0 as event padding
        feature: features of each event, shape (nb_samples, max_event_len, max_feature_len)
                A vector of length max_feature_len discribe numeric features of a event. the vector is in format: index1, value1, index2, value2 ...
                0 is considered as padding.
        sample_id: IDs of samples, shape (nb_samples, )
                If you don't use this attr, just set it to (1 - nb_sample, )

### Segmentation format
The segmentation files are also in HDF5 format.

An complete segmentation file should have following attributes:

        segment: segmentation of event sequences, shape(nb_sample, max_segs)
                0 is considered as padding, so first 0 are omitted
        max_segs: max number of segments of an event sequence
        max_seg_length: max length of an segment

















