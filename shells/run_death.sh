THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/event.txt>& log/death_timeAggre_eAtt_lstm.log2
THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/event_temporal.txt>& log/death_timeAggre_eAtt_tAtt_lstm.log2
THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/timeAggre.txt settings/event_temporal.txt >& log/death_timeAggre_fea_catAtt_eAtt_tAtt_lstm.log2