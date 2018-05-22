# dir=ICU_exper
# files="ICUIn_train_1000.h5 ICUIn_valid_1000.h5 ICUIn_test_1000.h5"
# seg_dir=ICU_exper/segs
# dir=death_exper
# files="death_test_1000.h5 death_train_1000.h5 death_valid_1000.h5"
# seg_dir=death_exper/segs
dir=zhu_exper
files="test.h5 train.h5 valid.h5"
seg_dir=$dir/segs
dir=lab_exper
files="labtest_test_1000.h5 labtest_train_1000.h5 labtest_valid_1000.h5"

for file in $files
do
    dataset="$dir/$file"
    python scripts/gen_fix_segs.py $dataset timeAggre 64 $seg_dir
done