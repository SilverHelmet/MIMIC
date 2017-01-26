python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt >& log/death_timeAggre_lstm.log1 &
wait
python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt >& log/death_timeAggre_catAtt_lstm.log1 &
wait
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/timeAggre.txt >& log/death_timeAggre_fea_catAtt_lstm.log1 &
wait