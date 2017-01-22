python -u train_rnn.py settings/lstm.txt settings/temporal.txt>& log/icu_timeAggre_tAtt_lstm.log1 &
python -u train_rnn.py settings/catAtt_lstm.txt settings/temporal.txt>& log/icu_timeAggre_catAtt_tAtt_lstm.log1 &
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/temporal.txt>& log/icu_timeAggre_fea_catAtt_tAtt_lstm.log1 &