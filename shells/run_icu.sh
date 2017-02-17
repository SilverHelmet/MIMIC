python -u train_rnn.py settings/lstm_icu.txt settings/fixLength32_icu.txt settings/event_temporal.txt >& log/icu_fixLength32_eAtt_tAtt_lstm.log2 &
wait
python -u train_rnn.py settings/lstm_icu.txt settings/fixLength32_icu.txt >& log/icu_fixLength32_lstm.log2 &
wait
python -u train_rnn.py settings/catAtt_lstm_icu.txt settings/fixLength32_icu.txt >& log/icu_fixLength32_catAtt_lstm.log2 &
wait
python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength32_icu.txt >& log/icu_fixLength32_fea_catAtt_lstm.log2 &
wait