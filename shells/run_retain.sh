# dir="baseline/retainData/icu_catAtt."
# python -u  baseline/retain.py --verbose "${dir}visit" 3418 "${dir}label" "${dir}weight" >& log/retain_icu_catAtt.log &

# dir="baseline/retainData/icu."
# python -u  baseline/retain.py --verbose "${dir}visit" 951 "${dir}label" "${dir}weight" >& log/retain_icu.log &
# wait

dir="baseline/retainData/death_catAtt."
python -u  baseline/retain.py --verbose "${dir}visit" 3418 "${dir}label" "${dir}weight" >& log/retain_death_catAtt.log &

dir="baseline/retainData/death."
python -u  baseline/retain.py --verbose "${dir}visit" 951 "${dir}label" "${dir}weight" >& log/retain_death.log &
wait