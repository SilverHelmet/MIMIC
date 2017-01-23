python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength4.txt >& log/icu_fixLength4_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength8.txt >& log/icu_fixLength8_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength16.txt >& log/icu_fixLength16_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength64.txt >& log/icu_fixLength64_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength128.txt >& log/icu_fixLength128_catAtt_lstm.log1 
python -u train_rnn.py settings/catAtt_lstm.txt  settings/fixLength256.txt >& log/icu_fixLength256_catAtt_lstm.log1 