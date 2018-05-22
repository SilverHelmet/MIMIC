max_length=256
str=""
length=1
# dir=ICU_exper
# files="ICUIn_train_1000.h5 ICUIn_valid_1000.h5 ICUIn_test_1000.h5"
# seg_dir=ICU_exper/segs
# dir=death_exper
# files="death_test_1000.h5 death_train_1000.h5 death_valid_1000.h5"
# seg_dir=death_exper/segs
# dir=zhu_exper
# files="train.h5 valid.h5 test.h5"
# seg_dir=${dir}/segs
dir=lab_exper
files="labtest_test_1000.h5 labtest_train_1000.h5 labtest_valid_1000.h5"

seg_dir=$dir/{segs}

if [ ! -d "${seg_dir}"]; then
    mkdir "${seg_dir}"
fi


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