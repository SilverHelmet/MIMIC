THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength256_icu.txt >& log/icu_fixLength256_lstm.log2
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength128_icu.txt >& log/icu_fixLength128_lstm.log2
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength64_icu.txt >& log/icu_fixLength64_lstm.log2
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength32_icu.txt >& log/icu_fixLength32_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength16_icu.txt >& log/icu_fixLength16_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength8_icu.txt >& log/icu_fixLength8_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength4_icu.txt >& log/icu_fixLength4_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength2_icu.txt >& log/icu_fixLength2_lstm.log2
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm_icu.txt  settings/fixLength1_icu.txt >& log/icu_fixLength1_lstm.log2

