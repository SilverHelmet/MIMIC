# THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength1.txt >& log/death_fixLength1_catAtt_lstm.log2 
# THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength2.txt >& log/death_fixLength2_catAtt_lstm.log2 
THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength4.txt >& log/death_fixLength4_catAtt_lstm.log2 
THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength8.txt >& log/death_fixLength8_catAtt_lstm.log2  
THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength16.txt >& log/death_fixLength16_catAtt_lstm.log2  
THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength32.txt >& log/death_fixLength32_catAtt_lstm.log2 
THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength64.txt >& log/death_fixLength64_catAtt_lstm.log2 
THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength128.txt >& log/death_fixLength128_catAtt_lstm.log2 
THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength256.txt >& log/death_fixLength256_catAtt_lstm.log2 
