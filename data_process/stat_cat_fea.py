from util import static_data_dir, Print, add_to_cnt_dict
import os
import json
import glob
import sys


class Dict:
    def __init__(self, ignored_names):
        self.ignored_names = ignored_names
        self.dict = {}

    def add_line(self, line, table):
        obj = json.loads(line)
        for name in obj:
            if name in self.ignored_names:
                continue
            value = obj[name]
            if value == "" or "UNKNOWN" in value or "NOT SPECIFIED" in value:
                continue
            whole_name = table + "#" + name
            if not whole_name in self.dict:
                self.dict[whole_name] = {}
            add_to_cnt_dict(self.dict[whole_name], value)



    def print_info(self, outf):
        for name in sorted(self.dict.keys()):
            size = len(self.dict[name])
            outf.write("%s\t%d\n" %(name, size))
            if size <= 100:
                outf.write("\t%s\n" %json.dumps(list(self.dict[name].items())))

    def print_feature_info(self, base_dir):
        for feature in self.dict:
            value_cnt = self.dict[feature]
            values = sorted(value_cnt.keys(), key = lambda x: value_cnt[x], reverse = True)
            total = 0.0;
            for value in values:
                total += value_cnt[value]
            
            outf = file(os.path.join(base_dir, '%s_stat.tsv' %feature), 'w')
            now = 0.0
            for value in values:
                now += value_cnt[value]
                outf.write("%s\t%.4f\n" %(value, now/total))
            outf.close()
                


def stat_static_feature():
    ignored_names = set(['subject_id', 'hadm_id', 'seq_num', 'dob'])
    base_dir = os.path.join(static_data_dir, 'static_feature')
    
    feature_stat = Dict(ignored_names)
    for filepath in glob.glob(base_dir + "/*json"):
        table = os.path.basename(filepath).split('.')[0]
        Print('parse %s' %filepath)
        for line in file(filepath):
            feature_stat.add_line(line, table)
    return feature_stat


if __name__ == "__main__":
    feature_stat = stat_static_feature()
    feature_stat.print_info(sys.stdout)
    static_feature_dir = os.path.join(static_data_dir, 'static_feature')
    stat_dir = os.path.join(static_feature_dir, 'stat')
    if not os.path.exists(stat_dir):
        os.mkdir(stat_dir)
    feature_stat.print_feature_info(stat_dir)
    
    


    