python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/event.txt>& log/icu_timeAggre_eAtt_lstm.log1 &
wait
python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt  settings/event.txt>& log/icu_timeAggre_catAtt_eAtt_lstm.log1 &
wait
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/timeAggre.txt  settings/event.txt>& log/icu_timeAggre_fea_catAtt_eAtt_lstm.log1 &
wait