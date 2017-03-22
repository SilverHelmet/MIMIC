
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength256_zhu.txt >& log/zhu_fixLength256_catAtt_lstm.log1
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength128_zhu.txt >& log/zhu_fixLength128_catAtt_lstm.log1
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength64_zhu.txt >& log/zhu_fixLength64_catAtt_lstm.log1
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength32_zhu.txt >& log/zhu_fixLength32_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength16_zhu.txt >& log/zhu_fixLength16_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength8_zhu.txt >& log/zhu_fixLength8_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength4_zhu.txt >& log/zhu_fixLength4_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength2_zhu.txt >& log/zhu_fixLength2_catAtt_lstm.log1
python -u train_rnn.py settings/catAtt_lstm_zhu.txt  settings/noatt.txt settings/fixLength1_zhu.txt >& log/zhu_fixLength1_catAtt_lstm.log1

