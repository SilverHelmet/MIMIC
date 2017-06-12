# event & temporal attention
prefix="THEANO_FLAGS=device=gpu2,floatX=float32"
python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/event_temporal.txt>& log/death_timeAggre_eAtt_tAtt_lstm.log2

python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt settings/event_temporal.txt >& log/death_timeAggre_catAtt_eAtt_tAtt_lstm.log2


# event attention
python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/event.txt>& log/death_timeAggre_eAtt_lstm.log2

python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt settings/event.txt >& log/death_timeAggre_catAtt_eAtt_lstm.log2

python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/timeAggre.txt settings/event.txt >& log/death_timeAggre_fea_catAtt_eAtt_lstm.log2


# temporal attention
python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  settings/temporal.txt>& log/death_timeAggre_tAtt_lstm.log2

python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt settings/temporal.txt >& log/death_timeAggre_catAtt_tAtt_lstm.log2

python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/timeAggre.txt settings/temporal.txt >& log/deathtimeAggre_fea_catAtt_tAtt_lstm.log2

# no attention
python -u train_rnn.py settings/lstm.txt settings/timeAggre.txt  >& log/death_timeAggre_lstm.log2

python -u train_rnn.py settings/catAtt_lstm.txt settings/timeAggre.txt >& log/death_timeAggre_catAtt_lstm.log2

python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/timeAggre.txt  >& log/death_timeAggre_fea_catAtt_lstm.log2


