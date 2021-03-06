dataset_setting="settings/fea_catAtt_lstm.txt"

# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength4.txt >& log/death_fixLength4_fea_catAtt_lstm.log2 

THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength8.txt >& log/death_fixLength8_fea_catAtt_lstm.log2 
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength16.txt >& log/death_fixLength16_fea_catAtt_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength32.txt >& log/death_fixLength32_fea_catAtt_lstm.log2 
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength64.txt >& log/death_fixLength64_fea_catAtt_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength128.txt >& log/death_fixLength128_fea_catAtt_lstm.log2 
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength256.txt >& log/death_fixLength256_fea_catAtt_lstm.log2 

# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength1.txt  >& log/death_fixLength1_fea_catAtt_lstm.log2 
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py $dataset_setting  settings/fixLength2.txt >& log/death_fixLength2_fea_catAtt_lstm.log2 