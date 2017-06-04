from util import *
from scripts.sample_setting import load_ICUIn_sample_setting, PatientSample, Sample, load_death_sample_setting
from build_event import Event
import glob
import json
from tqdm import tqdm
import commands

def init_sample(sample_setting_map, max_event_len):
    sample_map = {}
    for pid in sample_setting_map:
        sample_setting = sample_setting_map[pid]
        sample = PatientSample(sample_setting, max_event_len)
        sample_map[pid] = sample
    return sample_map

def load_event(filepath, sample_map):
    for line in tqdm(file(filepath), total = get_nb_lines(filepath)):
        event = Event.load_from_line(line)
        pid = event.pid
        if pid in sample_map:
            sample_map[pid].add_event(event)

def filter_dict(ori_map, valid_keys):
    new_map = {}
    for key in ori_map:
        if key in valid_keys:
            new_map[key] = ori_map[key]
    return new_map

def build_sample(sample_setting_path, out_path, max_event_len, load_func):
    sample_setting_map = load_func(sample_setting_path) 
    pids = list(sorted(sample_setting_map.keys()))
    cnts = len(pids)
    boundary = [0, int(cnts * 0.2),  int(cnts * 0.4), int(cnts * 0.6), int(cnts * 0.8), cnts]
    starts = boundary[:-1]
    ends = boundary[1:]

    writer = file(out_path, 'w')
    writer.close()
    for st, ed in zip(starts, ends):
        print "-" * 50
        Print("processing %d - %d" %(st, ed))
        valid_pids = set(pids[st:ed])
        sample_setting_map = load_func(sample_setting_path)
        sample_setting_map = filter_dict(sample_setting_map, valid_pids)
        Print("#patients = %d" %(len(sample_setting_map)))

        sample_map = init_sample(sample_setting_map, max_event_len)
        nb_files = int(commands.getoutput('wc -l |ls %s/*.tsv' %event_dir))
        for idx, filepath in enumerate(glob.glob(event_dir + '/*.tsv'), start = 1):
            Print("loading %d/%d %s" %(idx, nb_files, os.path.basename(filepath)))
            load_event(filepath, sample_map)

        writer = file(out_path, 'a')
        for pid in sorted(sample_map.keys()):
            sample_map[pid].write(writer)
        writer.close()


if __name__ == "__main__":
    max_event_len = 5000

    # ICU sample
    Print("generate ICU samples")
    sample_setting_path = os.path.join(event_seq_stat_dir, "ICUIn_sample_setting.txt") 
    out_path = os.path.join(ICU_exper_dir, "ICU_samples_len=%d.txt" %max_event_len)
    build_sample(sample_setting_path, out_path, max_event_len, load_ICUIn_sample_setting)


    # death sample
    Print("generate death samples")
    sample_setting_path = os.path.join(event_seq_stat_dir, "death_sample_setting.txt")
    out_path = os.path.join(death_exper_dir, "death_sample_len=%d.txt" %max_event_len)
    build_sample(sample_setting_path, out_path, max_event_len, load_death_sample_setting)
   


    # debug read sample & test
    # reader = file(os.path.join(ICU_exper_dir, "ICU_samples_len=300.txt"), 'r')
    # for line in reader:
    #     sample = Sample.load_from_json(json.loads(line))
    #     print sample.events[0]