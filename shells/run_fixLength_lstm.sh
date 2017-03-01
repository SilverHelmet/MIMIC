
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength256_zhu.txt >& log/zhu_fixLength256_lstm.log1
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength128_zhu.txt >& log/zhu_fixLength128_lstm.log1
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength64_zhu.txt >& log/zhu_fixLength64_lstm.log1
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength32_zhu.txt >& log/zhu_fixLength32_lstm.log1 
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength16_zhu.txt >& log/zhu_fixLength16_lstm.log1 
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength8_zhu.txt >& log/zhu_fixLength8_lstm.log1 
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength4_zhu.txt >& log/zhu_fixLength4_lstm.log1 
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength2_zhu.txt >& log/zhu_fixLength2_lstm.log1
python -u train_rnn.py settings/lstm_zhu.txt  settings/fixLength1_zhu.txt >& log/zhu_fixLength1_lstm.log1

