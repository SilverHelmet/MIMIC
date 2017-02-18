
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength256.txt >& log/death_fixLength256_lstm.log2
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength128.txt >& log/death_fixLength128_lstm.log2
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength64.txt >& log/death_fixLength64_lstm.log2
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength32.txt >& log/death_fixLength32_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength16.txt >& log/death_fixLength16_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength8.txt >& log/death_fixLength8_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength4.txt >& log/death_fixLength4_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength2.txt >& log/death_fixLength2_lstm.log2
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/lstm.txt  settings/fixLength1.txt >& log/death_fixLength1_lstm.log2

