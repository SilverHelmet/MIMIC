
# python -u train_rnn.py settings/lstm.txt  settings/fixLength2.txt >& log/death_fixLength2_lstm.log2 
wait 
# THEANO_FLAGS=device=cuda0,floatX=float32 python -u train_rnn.py settings/fea_gcn.txt @gcn_time_width=0.5 settings/catAtt_lstm.txt settings/fixLength16.txt >& log/death_fixLength16_fea_catAtt_gcn0.5.log1 &
# THEANO_FLAGS=device=cuda1,floatX=float32 python -u train_rnn.py settings/fea_gcn.txt @gcn_time_width=1.0 settings/catAtt_lstm.txt settings/fixLength16.txt >& log/death_fixLength16_fea_catAtt_gcn1.0.log1 &
# THEANO_FLAGS=device=cuda2,floatX=float32 python -u train_rnn.py settings/fea_gcn.txt @gcn_time_width=1.5 settings/catAtt_lstm.txt settings/fixLength16.txt >& log/death_fixLength16_fea_catAtt_gcn1.5.log1 &

THEANO_FLAGS=device=cuda0,floatX=float32 python -u train_rnn.py settings/fea_gcn.txt @gcn_hidden_dim=16|gcn_num__head=4 settings/catAtt_lstm.txt settings/fixLength16.txt >& log/death_fixLength16_fea_catAtt_gcn0.5_4X16.log1 &
THEANO_FLAGS=device=cuda1,floatX=float32 python -u train_rnn.py settings/fea_gcn.txt @gcn_hidden_dim=8|gcn_num__head=8 settings/catAtt_lstm.txt settings/fixLength16.txt >& log/death_fixLength16_fea_catAtt_gcn0.5_8X8.log1 &
THEANO_FLAGS=device=cuda2,floatX=float32 python -u train_rnn.py settings/fea_gcn.txt @gcn_hidden_dim=4|gcn_num__head=16 settings/catAtt_lstm.txt settings/fixLength16.txt >& log/death_fixLength16_fea_catAtt_gcn0.5_16X4.log1 &

# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/dlstm.txt settings/static_feature.txt settings/fixLength16.txt >& log/death_catAtt_dlstm.log1 &
# wait
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/gcn.txt settings/catAtt_lstm.txt settings/fixLength16.txt settings/static_feature.txt >& log/death_fixLength16_catAtt_staticfea_gcn.log6 &
# wait
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength16.txt settings/static_feature.txt >& log/death_fixLength16_catAtt_staticfea_lstm.log2
# THEANO_FLAGS=device=gpu2,floatX=float32 python -u train_rnn.py settings/catAtt_lstm.txt settings/fixLength32.txt >& log/death_fixLength32_catAtt_lstm.log3 
# python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/fixLength2.txt  >& log/death_fixLength2_fea_catAtt_lstm.log2

# python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength2_icu.txt  >& log/icu_fixLength2_fea_catAtt_lstm.log1
# python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength2_icu.txt  >& log/icu_fixLength2_fea_catAtt_lstm.log2