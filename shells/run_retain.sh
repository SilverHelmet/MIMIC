# dir="baseline/retainData/icu_catAtt."
# THEANO_FLAGS=device=gpu0,floatX=float32 python -u  baseline/retain.py --verbose "${dir}visit" 3418 "${dir}label" "${dir}weight" >& log/retain_icu_catAtt.log2 &
# 
# dir="baseline/retainData/icu."
# THEANO_FLAGS=device=gpu0,floatX=float32 python -u  baseline/retain.py --verbose "${dir}visit" 951 "${dir}label" "${dir}weight" >& log/retain_icu.log2 &
# wait

# dir="baseline/retainData/death_catAtt."
# THEANO_FLAGS=device=gpu1,floatX=float32 python -u  baseline/retain.py --verbose "${dir}visit" 3418 "${dir}label" "${dir}weight" >& log/retain_death_catAtt.log2 &

# dir="baseline/retainData/death."
# THEANO_FLAGS=device=gpu1,floatX=float32 python -u  baseline/retain.py --verbose "${dir}visit" 951 "${dir}label" "${dir}weight" >& log/retain_death.log2 &
# wait