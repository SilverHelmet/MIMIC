THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength1.txt >& log/death_fixLength1_lstm.log2 &
THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength1.txt >& log/death_catAtt_fixLength1_lstm.log2 &
wait

# THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/fixLength1.txt  >& log/death_fea_catAtt_fixLength1_lstm.log2

# THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength2.txt >& log/death_fixLength2_lstm.log2 &
# python -u train_rnn.py settings/cat_lstm.txt settings/fixLength2.txt >& log/death_catAtt_fixLength2_lstm.log2 &
# wait

# THEANO_FLAGS=device=gpu0,floatX=float32 python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/fixLength2.txt  >& log/death_fea_catAtt_fixLength2_lstm.log2