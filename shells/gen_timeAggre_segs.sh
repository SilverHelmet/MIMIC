dir=ICU_exper
files="ICUIn_train_1000.h5 ICUIn_valid_1000.h5 ICUIn_test_1000.h5"
seg_dir=ICU_exper/segs

for file in $files
do
    dataset="$dir/$file"
    python scripts/gen_fix_segs.py $dataset timeAggre 64 $seg_dir
done