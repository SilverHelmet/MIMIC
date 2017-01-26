# python -u train_rnn.py settings/lstm.txt settings/fixLength32.txt >& log/death_fixLength32_lstm.log1 &
# wait
python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength32.txt >& log/death_fixLength32_catAtt_lstm.log1 &
wait
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/fixLength32.txt >& log/death_fixLength32_fea_catAtt_lstm.log1 &
wait
python -u train_rnn.py settings/lstm.txt settings/fixLength32.txt settings/event_temporal.txt >& log/death_fixLength32_eAtt_tAtt_lstm.log1 &
wait
python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength32.txt settings/event_temporal.txt >& log/death_fixLength32_catAtt_eAtt_tAtt_lstm.log1 &
wait
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/fixLength32.txt settings/event_temporal.txt >& log/death_fixLength32_fea_catAtt_eAtt_tAtt_lstm.log1 &
wait