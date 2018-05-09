
# python -u train_rnn.py settings/lstm.txt  settings/fixLength2.txt >& log/death_fixLength2_lstm.log2 
THEANO_FLAGS=device=gpu1,floatX=float32 python -u train_rnn.py settings/gcn.txt settings/catAtt_lstm.txt settings/fixLength16.txt settings/params/lr=0.03.txt >& log/death_fixLength16_catAtt_gcn_lr0.03.log1 &
THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/gcn.txt settings/catAtt_lstm.txt settings/fixLength16.txt settings/params/lr=0.003.txt >& log/death_fixLength16_catAtt_gcn_lr0.003.log2 &
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/dlstm.txt settings/static_feature.txt settings/fixLength16.txt >& log/death_catAtt_dlstm.log1 &
# wait
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/gcn.txt settings/catAtt_lstm.txt settings/fixLength16.txt settings/static_feature.txt >& log/death_fixLength16_catAtt_staticfea_gcn.log6 &
# wait
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength16.txt settings/static_feature.txt >& log/death_fixLength16_catAtt_staticfea_lstm.log2
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength32.txt >& log/death_fixLength32_catAtt_lstm.log3 
# python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/fixLength2.txt  >& log/death_fixLength2_fea_catAtt_lstm.log2

# python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength2_icu.txt  >& log/icu_fixLength2_fea_catAtt_lstm.log1
# python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength2_icu.txt  >& log/icu_fixLength2_fea_catAtt_lstm.log2