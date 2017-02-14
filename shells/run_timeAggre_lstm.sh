prefix="THEANO_FLAGS=device=gpu2,floatX=float32"
python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/event_temporal.txt>& log/icu_timeAggre_eAtt_tAtt_lstm.log2 &
wait
python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt settings/event_temporal.txt >& log/icu_timeAggre_catAtt_eAtt_tAtt_lstm.log2 &
wait

# event attention
python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/event.txt>& log/icu_timeAggre_eAtt_lstm.log2 &
wait
python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt settings/event.txt >& log/icu_timeAggre_catAtt_eAtt_lstm.log2 &
wait
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/timeAggre.txt settings/event.txt >& log/icu_timeAggre_fea_catAtt_eAtt_lstm.log2 &
wait

# temporal attention
python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/temporal.txt>& log/icu_timeAggre_tAtt_lstm.log2 &
wait
python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt settings/temporal.txt >& log/icu_timeAggre_catAtt_tAtt_lstm.log2 &
wait
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/timeAggre.txt settings/temporal.txt >& log/icu_timeAggre_fea_catAtt_tAtt_lstm.log2 &
wait

