
python -u train_rnn.py settings/lstm.txt  settings/fixLength2.txt >& log/death_fixLength2_lstm.log2 
python -u train_rnn.py settings/cat_lstm.txt settings/fixLength2.txt >& log/death_fixLength2_catAtt_lstm.log2 
python -u train_rnn.py settings/fea_catAtt_lstm.txt settings/fixLength2.txt  >& log/death_fixLength2_fea_catAtt_lstm.log2

python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength2_icu.txt  >& log/icu_fixLength2_fea_catAtt_lstm.log1
python -u train_rnn.py settings/fea_catAtt_lstm_icu.txt settings/fixLength2_icu.txt  >& log/icu_fixLength2_fea_catAtt_lstm.log2