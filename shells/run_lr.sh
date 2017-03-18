python -u "baseline/logistic regression.py" ori no_feature >& log/lr_zhu.log1
python -u "baseline/logistic regression.py" catAtt no_feature >& log/lr_zhu_catAtt.log1
python -u "baseline/logistic regression.py" catAtt add_feature >& log/lr_zhu_fea_catAtt.log1