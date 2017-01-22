python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/event_temporal.txt settings/fixLength.txt >& log/icu_fixLength_fea_catAtt_eAtt_tAtt_lstm.log1 &
wait
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/event_temporal.txt >& log/icu_timeAggre_fea_catAtt_eAtt_tAtt_lstm.log1 &
wait