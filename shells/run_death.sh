
# python -u train_rnn.py settings/lstm.txt  settings/fixLength2.txt >& log/death_fixLength2_lstm.log2 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength16.txt >& log/death_fixLength16_catAtt_lstm.log3 
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength16.txt >& log/death_fixLength16_catAtt_lstm.log4
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength32.txt >& log/death_fixLength32_catAtt_lstm.log3 
# python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/fixLength2.txt  >& log/death_fixLength2_fea_catAtt_lstm.log2

# python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength2_icu.txt  >& log/icu_fixLength2_fea_catAtt_lstm.log1
# python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength2_icu.txt  >& log/icu_fixLength2_fea_catAtt_lstm.log2