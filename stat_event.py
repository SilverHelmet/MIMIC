from util import *
from build_event import Event
import glob
import os
import datetime

class SimpleStat:
    def __init__(self):
        self.pid_event_cnt = {}
        self.nb_event = 0
        self.rtype_set = set()

    def add_pid(self, pid):
        if not pid in self.pid_event_cnt:
            self.pid_event_cnt[pid] = 0
            
    
    def add_event(self, line):
        event = Event.load_from_line(line)
        self.nb_event += 1
        pid = event.pid
        rtype =  event.index
        self.add_pid(pid)
        self.rtype_set.add(rtype)
        self.pid_event_cnt[pid] += 1

    def print_info(self):
        out_format = """
        # of patients = {0}
        # of events = {1}
        Avg # of events per patient = {2}
        Max # of events per patient = {3}
        Min # of events per patient = {4}
        # of unique events = {5}
        """
        nb_patients = len(self.pid_event_cnt)
        nb_events = self.nb_event
        ave_events = round((nb_events + 0.0) / nb_patients, 4)
        max_events = reduce(max, self.pid_event_cnt.values())
        min_events = reduce(min, self.pid_event_cnt.values())
        nb_event_type = len(self.rtype_set)
        print out_format.format(
            nb_patients,
            nb_events,
            ave_events,
            max_events,
            min_events,
            nb_event_type
        )

def gather_statistics(filepath, stat):
    print "gather info from %s" %filepath
    for line in file(filepath):
        stat.add_event(line)
        



if __name__ == '__main__':
    # stat event count
    # stat = SimpleStat()
    # for filename in glob.glob(event_dir + "/*tsv"):
    #     gather_statistics(filename, stat)
    # stat.print_info()

    # hospital time duration
    total = datetime.timedelta()
    print total
    cnt = 0
    for line in file(os.path.join(static_data_dir, "single_admission.tsv")):
        cnt += 1
        parts = line.strip().split("\t")
        st = parse_time(parts[2])
        ed = parse_time(parts[3])
        total += ed - st
    # total /= cnt + 0.0
    print total.total_seconds() / (cnt + 0.0) / 3600.0


