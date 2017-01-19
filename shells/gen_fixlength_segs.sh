max_length=256
str=""
length=1
dir=ICU_exper
files="ICUIn_train_1000.h5 ICUIn_valid_1000.h5 ICUIn_test_1000.h5"
seg_dir=ICU_exper/segs

while  [ $length -le $max_length ]
do
    echo "chunk_length = $length"
    for file in $files
    do
        dataset="$dir/$file"
        python scripts/gen_fix_segs.py $dataset fixLength $length $seg_dir
    done
    length=$(($length * 2))
done