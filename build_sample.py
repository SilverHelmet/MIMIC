from util import *
from scripts.sample_setting import load_ICUIn_sample_setting, PatientSample, Sample, load_death_sample_setting
from build_event import Event
import glob
import json


def init_sample(sample_setting_map, max_event_len):
    sample_map = {}
    for pid in sample_setting_map:
        sample_setting = sample_setting_map[pid]
        sample = PatientSample(sample_setting, max_event_len)
        sample_map[pid] = sample
    return sample_map

def load_event(filepath, sample_map):
    print os.path.basename(filepath)
    cnt = 0
    for line in file(filepath):
        cnt += 1 
        if cnt % 1000000 == 0:
            print "*",
        sys.stdout.flush()
        event = Event.load_from_line(line)
        pid = event.pid
        if pid in sample_map:
            sample_map[pid].add_event(event)
    print ""
        



if __name__ == "__main__":
    max_event_len = 1000
    # sample_setting_path = os.path.join(event_seq_stat_dir, "ICUIn_sample_setting.txt") 
    # sample_setting_map = load_ICUIn_sample_setting(sample_setting_path)
    # out_path = os.path.join(ICU_exper_dir, "ICU_samples_len=%d.txt" %max_event_len)

    sample_setting_path = os.path.join(event_seq_stat_dir, "death_sample_setting.txt")
    sample_setting_map = load_death_sample_setting(sample_setting_path)
    out_path = os.path.join(death_exper_dir, "death_sample_len=%d.txt" %max_event_len)
    
    sample_map = init_sample(sample_setting_map, max_event_len)
    for filepath in glob.glob(event_dir + '/*.tsv'):
        load_event(filepath, sample_map)

    writer = file(out_path, 'w')
    for pid in sorted(sample_map.keys()):
        sample_map[pid].write(writer)
    writer.close()

    # debug read sample & test
    # reader = file(os.path.join(ICU_exper_dir, "ICU_samples_len=300.txt"), 'r')
    # for line in reader:
    #     sample = Sample.load_from_json(json.loads(line))
    #     print sample.events[0]