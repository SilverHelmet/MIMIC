THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength32_icu.txt settings/cnn.txt >& log/icu_fixLength32_cnn.log1 &
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm_icu.txt  settings/fixLength32_icu.txt settings/cnn.txt >& log/icu_fixLength32_catAtt_cnn.log1 &
wait