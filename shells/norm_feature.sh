python -u scripts/norm_feature.py \
    ICU_exper/ICUIn_train_1000.h5 \
    ICU_exper/ICUIn_valid_1000.h5 \
    ICU_exper/ICUIn_test_1000.h5 >& log/norm_feature_icu.log &

python -u scripts/norm_feature.py \
    ICU_merged_exper/ICUIn_train_1000.h5 \
    ICU_merged_exper/ICUIn_valid_1000.h5 \
    ICU_merged_exper/ICUIn_test_1000.h5 >& log/norm_feature_icuMerged.log &

wait

python -u scripts/norm_feature.py \
    death_exper/death_train_1000.h5 \
    death_exper/death_valid_1000.h5 \
    death_exper/death_test_1000.h5 >& log/norm_feature_death.log &

python -u scripts/norm_feature.py \
    death_merged_exper/death_train_1000.h5 \
    death_merged_exper/death_valid_1000.h5 \
    death_merged_exper/death_test_1000.h5 >& log/norm_feature_deathMerged.log &

wait