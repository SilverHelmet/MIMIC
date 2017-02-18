# fixlength32 attention
# python -u train_rnn.py settings/lstm_icu.txt settings/fixLength32_icu.txt settings/event_temporal.txt >& log/icu_fixLength32_eAtt_tAtt_lstm.log2 

# timeAggre attention lstm
# python -u train_rnn.py settings/lstm_icu.txt settings/timeAggre_icu.txt  settings/event_temporal.txt>& log/icu_timeAggre_eAtt_tAtt_lstm.log2
# python -u train_rnn.py settings/lstm_icu.txt settings/timeAggre_icu.txt  settings/event.txt>& log/icu_timeAggre_eAtt_lstm.log2
# python -u train_rnn.py settings/lstm_icu.txt settings/timeAggre_icu.txt  settings/temporal.txt>& log/icu_timeAggre_tAtt_lstm.log2

# timeAggre lstm/catAtt/feature
# python -u train_rnn.py settings/lstm_icu.txt settings/timeAggre_icu.txt  >& log/icu_timeAggre_lstm.log2
# python -u train_rnn.py settings/catAtt_lstm_icu.txt settings/timeAggre_icu.txt >& log/icu_timeAggre_catAtt_lstm.log2
# python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/timeAggre_icu.txt  >& log/icu_timeAggre_fea_catAtt_lstm.log2

THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/catAtt_lstm_icu.txt  settings/fixLength32_icu.txt >& log/icu_fixLength32_catAtt_lstm.log2 
THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt  settings/fixLength32_icu.txt >& log/icu_fixLength32_fea_catAtt_lstm.log2 