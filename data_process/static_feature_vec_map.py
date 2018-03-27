from util import static_data_dir
from data_process.stat_cat_fea import stat_static_feature
import os
from glob import glob

def build_feature_maps(feature_stat):
    feature_map = {}
    feature_map['patients#dob'] = 0
    for table_feature in feature_stat.dict:
        values = feature_stat.dict[table_feature].keys()
        size = len(values)
        if size > 1000:
            continue
        for value in sorted(values):
            table_feature_value = table_feature + "#" + value
            feature_map[table_feature_value] = len(feature_map)
    return feature_map

def build_feature_map_by_coverage(indir, outpath, coverages):
    feature_map = {}
    feature_map['patients#dob'] = 0
    for filepath in glob(indir + "/*tsv"):
        feature_name = os.path.basename(filepath).split('_stat.tsv')[0]
        threshold = coverages[feature_name]
        for line in file(filepath):
            key, coverage = line.split('\t')        
            coverage = float(coverage)
            if coverage > threshold:
                break
            feature_map[feature_name + "#" +key] = len(feature_map)
    outf = file(outpath, 'w')
    for key in sorted(feature_map.keys(), key = lambda x: feature_map[x]):
        outf.write("%s\t%d\n" %(key, feature_map[key]))
    outf.close()




if __name__ == "__main__":
    # version 1.0
    # feature_stat = stat_static_feature()
    # feature_map = build_feature_maps(feature_stat)
    # outf = file(os.path.join(static_data_dir, 'static_feature_map.tsv'), 'w')
    # for key in sorted(feature_map.keys(), key = lambda x: feature_map[x]):
    #     outf.write("%s\t%d\n" %(key, feature_map[key]))
    # outf.close()

    # version 2.0
    stat_dir = os.path.join(static_data_dir, 'static_feature/stat')
    outpath = os.path.join(static_data_dir, 'static_feature_map.tsv')
    coverages = {
        'admissions#diagnosis': 0.6,
        'admissions#ethnicity': 1.0,
        'admissions#insurance': 1.0,
        'admissions#language': 1.0,
        'admissions#marital_status': 1.0,
        'admissions#religion': 1.0,
        'diagnoses_icd#icd9_code': 0.80,
        'patients#gender': 1.0,
        'procedures_icd#icd9_code': 0.90,

    }
    build_feature_map_by_coverage(stat_dir, outpath, coverages)
    

