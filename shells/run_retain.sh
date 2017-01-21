dir="baseline/retainData/icu_catAtt."
python -u  baseline/retain.py --verbose "${dir}visit" 3418 "${dir}label" "${dir}weight" >& log/retain.log