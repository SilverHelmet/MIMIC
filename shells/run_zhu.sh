python -u train_rnn.py settings/lstm_zhu.txt settings/timeAggre_zhu.txt  settings/event.txt>& log/zhu_timeAggre_eAtt_lstm.log2
python -u train_rnn.py settings/lstm_zhu.txt settings/timeAggre_zhu.txt  settings/temporal.txt>& log/zhu_timeAggre_tAtt_lstm.log2
python -u train_rnn.py settings/lstm_zhu.txt settings/timeAggre_zhu.txt  settings/event_temporal.txt>& log/zhu_timeAggre_eAtt_tAtt_lstm.log2
python -u train_rnn.py settings/lstm_zhu.txt settings/fixLength32_zhu.txt  settings/event_temporal.txt>& log/zhu_fixLength32_eAtt_tAtt_lstm.log2

python -u train_rnn.py settings/catAtt_lstm_zhu.txt settings/timeAggre_zhu.txt  settings/event.txt>& log/zhu_timeAggre_eAtt_catAtt_lstm.log2
python -u train_rnn.py settings/catAtt_lstm_zhu.txt settings/timeAggre_zhu.txt  settings/temporal.txt>& log/zhu_timeAggre_tAtt_catAtt_lstm.log2
python -u train_rnn.py settings/catAtt_lstm_zhu.txt settings/timeAggre_zhu.txt  settings/event_temporal.txt>& log/zhu_timeAggre_eAtt_tAtt_catAtt_lstm.log2
python -u train_rnn.py settings/catAtt_lstm_zhu.txt settings/fixLength32_zhu.txt  settings/event_temporal.txt>& log/zhu_fixLength32_eAtt_tAtt_catAtt_lstm.log2

python -u train_rnn.py settings/fea_catAtt_lstm_zhu.txt settings/timeAggre_zhu.txt  settings/event.txt>& log/zhu_timeAggre_eAtt_fea_catAtt_lstm.log2
python -u train_rnn.py settings/fea_catAtt_lstm_zhu.txt settings/timeAggre_zhu.txt  settings/temporal.txt>& log/zhu_timeAggre_tAtt_fea_catAtt_lstm.log2
python -u train_rnn.py settings/fea_catAtt_lstm_zhu.txt settings/timeAggre_zhu.txt  settings/event_temporal.txt>& log/zhu_timeAggre_eAtt_tAtt_fea_catAtt_lstm.log2
python -u train_rnn.py settings/fea_catAtt_lstm_zhu.txt settings/fixLength32_zhu.txt  settings/event_temporal.txt>& log/zhu_fixLength32_eAtt_tAtt_fea_catAtt_lstm.log2