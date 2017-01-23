python -u train_rnn.py settings/lstm.txt  settings/fixLength4.txt >& log/icu_fixLength4_lstm.log1 
python -u train_rnn.py settings/lstm.txt  settings/fixLength8.txt >& log/icu_fixLength8_lstm.log1 
python -u train_rnn.py settings/lstm.txt  settings/fixLength64.txt >& log/icu_fixLength64_lstm.log1 
python -u train_rnn.py settings/lstm.txt  settings/fixLength128.txt >& log/icu_fixLength128_lstm.log1 
python -u train_rnn.py settings/lstm.txt  settings/fixLength256.txt >& log/icu_fixLength256_lstm.log1
python -u train_rnn.py settings/lstm.txt  settings/fixLength16.txt >& log/icu_fixLength16_lstm.log1 