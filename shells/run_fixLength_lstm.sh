python -u train_rnn.py settings/lstm.txt  settings/fixLength256.txt >& log/icu_fixLength256_lstm.log1
python -u train_rnn.py settings/lstm.txt  settings/fixLength16.txt >& log/icu_fixLength16_lstm.log1 
python -u train_rnn.py settings/lstm.txt  settings/fixLength2.txt >& log/icu_fixLength2_lstm.log1
python -u train_rnn.py settings/lstm.txt  settings/fixLength1.txt >& log/icu_fixLength1_lstm.log1