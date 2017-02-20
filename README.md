## Usage
`python train_rnn.py setting_1 setting_2 ... setting_n`

setting_1, setting_2, ..., setting_n are setting files of the model. 
The model will load arguments in these files sequentially， 
Here are an example setting file.
        # model
        disturbance=False

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





