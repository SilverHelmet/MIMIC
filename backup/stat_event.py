from util import *
import glob
from gather_static_data import *
import sys

class EventStat:
    stat_range = None
    def __init__(self, pid, admission):
        self.pid = pid
        self.last = admission.disch
        self.start = self.last
        self.days = admission.range()
        self.cnt = 0
    

    def add_admit(self, time):
        if time + EventStat.stat_range >= self.last:
            self.start = min(self.start, time)

    def adjust(self):
        if self.last == self.start:
            self.start = self.last - EventStat.stat_range


    def add_event(self, time):
        if time >= self.start:
            self.cnt += 1

    def __str__(self):
        out = [self.pid, self.cnt, self.start, self.last]
        return "\t".join(map(str, out))

def load_admission():
    ad_map = {}
    for line in file(static_data_dir + "/admission.tsv"):
        parts = line.strip().split("\t")
        pid = int(parts[0])
        adm = Admission.load_from_line("\t".join(parts[1:]))
        ad_map[pid] = adm
    return ad_map



def init(admit_path, admission_map):
    event_stat_map = {}
    for pid in admission_map:
        event_stat_map[pid] = EventStat(pid, admission_map[pid])

    for line in file(admit_path):
        parts = line.strip().split("\t")
        pid = int(parts[1])
        time = parse_time(parts[3])
        event_stat_map[pid].add_admit(time)
    for pid in event_stat_map:
        event_stat_map[pid].adjust()
    return event_stat_map

def stat_event(filepath, event_stat_map):
    print filepath
    for line in file(filepath):
        parts = line.strip().split("\t")
        time = parse_time(parts[3])
        pid = int(parts[1])
        event_stat_map[pid].add_event(time)

if __name__ == '__main__':
    EventStat.stat_range = datetime.timedelta(days = int(sys.argv[1]))
    admission_map = load_admission()
    admit_path = os.path.join(event_dir, "admissions.admittime.tsv")
    event_stat_map = init(admit_path, admission_map)
    for filepath in glob.glob(event_dir + "/*.tsv"):
        stat_event(filepath, event_stat_map)

    outf = file(os.path.join(event_stat_dir, "days=%d.tsv" %int(sys.argv[1])), 'w')
    for pid in sorted(event_stat_map.keys()):
        outf.write(str(event_stat_map[pid]) + "\n")
    outf.close()




