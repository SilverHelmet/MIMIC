import sys
import os
sys.path.append("..")
from gather_static_data import get_d_icd_diagnoses
from util import data_dir, result_dir
from extractor import parse_line

diag_icd_map = get_d_icd_diagnoses()
diag_path = os.path.join(data_dir, 'diagnoses_icd.tsv')
code_cnt = {}
number = 0
for line in file(diag_path, 'r'):
	parts = parse_line(line)
	code = parts[3].split("&")[1]
	if not code in code_cnt:
		code_cnt[code] = 0
	code_cnt[code] += 1
	number += 1.0

outf = file(os.path.join(result_dir, 'diagnoses_stat.long.tsv'), 'w')
fmt = '%-7s %5d/%7f%% %s\n' 
for code in sorted(code_cnt.keys(), key = lambda x: code_cnt[x], reverse = True):
	cnt = code_cnt[code]
	rate = round(cnt / number * 100, 3)
	if not code in diag_icd_map:
		exp = "None"
	else:
		exp = diag_icd_map[code][1]
	outf.write(fmt %(code, cnt, rate, exp))
outf.close()



