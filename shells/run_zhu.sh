python -u train_rnn.py settings/lstm_zhu.txt settings/timeAggre_zhu.txt  settings/temporal.txt>& log/zhu_timeAggre_tAtt_lstm.log1
python -u train_rnn.py settings/lstm_zhu.txt settings/timeAggre_zhu.txt  settings/event.txt>& log/zhu_timeAggre_eAtt_lstm.log1
python -u train_rnn.py settings/lstm_zhu.txt settings/timeAggre_zhu.txt  settings/event_temporal.txt>& log/zhu_timeAggre_eAtt_tAtt_lstm.log1
python -u train_rnn.py settings/lstm_zhu.txt settings/fixLength32_zhu.txt  settings/event_temporal.txt>& log/zhu_fixLenght32_eAtt_tAtt_lstm.log1