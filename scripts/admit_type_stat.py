import sys
sys.path.append("..")
from util import *
import os

def stat_dis(filepath):
	death_cnt = {}
	for line in file(filepath):
		parts = line.strip().split("\t")
		death_flag = parts[-2] == "True"
		admit_type =parts[-1]
		if not admit_type in death_cnt:
			death_cnt[admit_type] = {"death":0, "total": 0}
		key = "death" if death_flag else "total"
		death_cnt[admit_type][key] += 1
		if key == "death":
			death_cnt[admit_type]['total'] += 1
	for admit_type in death_cnt:
		death = death_cnt[admit_type]['death']
		total = death_cnt[admit_type]['total']
		print "%-10s\tdeath=%d\ttotal=%d" %(admit_type, death, total)

# def stat_admit_day(filepath, admit_type, )




if __name__ == '__main__':
	admit_file = os.path.join(static_data_dir, "admission.tsv")
	stat_dis(admit_file)