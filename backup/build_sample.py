from util import *
import sys
import datetime
from stat_event import load_admission
import glob
from build_event import Event, Feature

class Sample:
    min_interval = datetime.timedelta(seconds = 3600)
    max_interval = datetime.timedelta(days = 10)
    interval_event_idx = 1
    interval_feature_idx = 0
    eps = datetime.timedelta(seconds = 1)

    def __init__(self, pid, start, end, label):
        self.pid = pid
        self.start = start
        self.end = end
        self.label = label
        self.events = []

    def add_event(self, event, max_length):
        if event.index in [2004]:
            return
        if event.time >= self.start and event.time <= self.end:
            self.events.append(event)
            if len(self.events) >= 2 * max_length:
                self.events = sorted(self.events)[-max_length:]

    

    def adjust(self, max_length):
        pid = self.pid
        ad_event = sorted(self.events)[-max_length:]
        self.events = []
        for idx in range(len(ad_event)):
            if idx > 0:
                interval_event = self.build_interval_event(ad_event[idx-1].time, ad_event[idx].time)
                if interval_event is not None:
                    self.events.append(interval_event)
            self.events.append(ad_event[idx])

    def build_interval_event(self, start, end):
        delta = end - start
        if delta < Sample.min_interval:
            return None
        time = start + Sample.eps
        delta = min(delta, Sample.max_interval)
        interval_feature = Feature(Sample.interval_feature_idx, delta.total_seconds()/3600.0)
        return Event(Sample.interval_event_idx, [interval_feature], self.pid, time)

    def valid(self):
        if len(self.events) > 5:
            return True
        else:
            return False

    @staticmethod
    def load_from_line(line):
        parts = line.strip().split("|")

        pid = int(parts[0])
        label = int(parts[1])
        sample = Sample(pid, None, None, label)
        for part in parts[2:]:
            sample.events.append(Event.load_from_line(part))
        return sample

    def __str__(self):
        out = [self.pid, self.label]
        out.extend(self.events)
        out = map(str, out)
        return "|".join(out)

def init_sample(admission_map, pred_range):
    global sample_types 
    sample_map = {}
    cnt = 0
    for pid in admission_map:
        start = admission_map[pid].admit
        end = admission_map[pid].disch
        flag = 1 if admission_map[pid].death == True else 0
        if start + pred_range < end and admission_map[pid].admit_type in sample_types:
            sample_map[pid] = Sample(pid, start, end - pred_range, flag)
        else:
            # print "too short range", pid, start, end
            cnt += 1
    print "error cnt = ", cnt
    return sample_map

def load_event(filepath, event_map):
    for line in file(filepath):
        event = Event.load_from_line(line)
        pid = event.pid
        if pid in event_map:
            event_map[pid].append(line)


def fill_event(filepath, sample_map, max_length, bin_pid):
    for line in file(filepath):
        event = Event.load_from_line(line)
        pid = event.pid

        if not pid in bin_pid:
            continue
        if pid in sample_map:
            sample_map[pid].add_event(event, max_length)

if __name__ == '__main__':
    sample_types = ["emergency", 'urgent']
    max_length = int(sys.argv[1])
    pred_days = float(sys.argv[2])
    pred_range = datetime.timedelta(days = pred_days)
    admission_map = load_admission()
    sample_map = init_sample(admission_map, pred_range)
    event_map = {}
    pids = sorted(sample_map.keys())
    for pid in sample_map:
        event_map[pid] = []

    for filepath in glob.glob(event_dir + '/*.tsv'):
        print "*",
        load_event(filepath, event_map)

    outpath = "%s_samples_%d_%.2f.txt" %(sample_types, max_length, pred_days)
    outf = file(os.path.join(exper_dir, outpath), 'w')
    for pid in sorted(event_map.keys()):
        sample = sample_map[pid]
        for event_line in event_map[pid]:
            sample.add_event(Event.load_from_line(event_line), max_length)
        sample.adjust(max_length)
        outf.write(str(sample))
        outf.write("\n")
        del sample_map[pid]
    outf.close()
