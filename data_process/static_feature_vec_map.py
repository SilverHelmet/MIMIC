from util import static_data_dir
from data_process.stat_cat_fea import stat_static_feature
import os

def build_feature_maps(feature_stat):
    feature_map = {}
    feature_map['patients#dob'] = 0
    for table_feature in feature_stat.dict:
        values = feature_stat.dict[table_feature]
        size = len(values)
        if size > 3000:
            continue
        for value in sorted(values):
            table_feature_value = table_feature + "#" + value
            feature_map[table_feature_value] = len(feature_map)
    return feature_map


if __name__ == "__main__":
    feature_stat = stat_static_feature()
    feature_map = build_feature_maps(feature_stat)
    outf = file(os.path.join(static_data_dir, 'static_feature_map.tsv'), 'w')
    for key in sorted(feature_map.keys(), key = lambda x: feature_map[x]):
        outf.write("%s\t%d\n" %(key, feature_map[key]))
    outf.close()
    

