import sys
import os
import re
sys.path.append("..")
from util import data_dir, parse_time, result_dir
from extractor import parse_line
from stat_data import get_id

death_path = os.path.join(data_dir, 'admissions.deathtime.tsv')




day_cnt = [0] * 366
days = []
for line in file(death_path, 'r'):
	parts = parse_line(line)
	# ID = get_id(parts[0])
	death_time = parse_time(parts[1])
	day = int(death_time.strftime('%j')) - 1
	day_cnt[day] += 1
	days.append(day)

import matplotlib.pyplot as plt

print plt.hist(days, bins = 366, range = [-0.5, 365.5])
outf = file(os.path.join(result_dir, 'death.dayCount.txt'), 'w')
for day in day_cnt:
	outf.write(str(day))
	outf.write("\n")
outf.close()
# plt.show()


