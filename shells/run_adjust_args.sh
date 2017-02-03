embedding_dim=64
hidden_dim=64

embedding_args="32 64 128"
hidden_args="32 64 128"
att_hidden_args="64 32 128"
batch_args="32 64 128"
settings="settings/fea_catAtt_lstm.txt settings/timeAggre.txt  settings/event_temporal.txt"
for embedding_dim in $embedding_args
do
    for hidden_dim in $hidden_args
    do
        for att_hidden_dim in $att_hidden_args
        do
            args="embedding_dim=${embedding_dim}|hidden_dim=${hidden_dim}|att_hidden_dim=${att_hidden_dim}"
            if [ "$1" = "norm" ]; then
                args="$args|norm_feature=True"
            fi
            outfile="log/icu_timeAggre_attention_${args//|/_}.log"
            python -u train_rnn.py $settings "#$args" >& $outfile 
        done
    done
done