from scripts_util import *
from util import *
from gather_from_event_seq_count import load
import json
from build_event import Event

class PatientSample:
    '''
        a container which contains all samples of a patient
    '''

    def __init__(self, patient_sample_setting, max_event_len):
        self.pid = patient_sample_setting.pid
        self.samples = []
        for setting in patient_sample_setting.settings:
            self.samples.append(Sample(setting, max_event_len, self.pid))
            
    def add_event(self, event):
        for sample in self.samples:
            sample.add_event(event)
    
    def finish(self):
        for sample in self.samples:
            sample.finish()

    def write(self, writer):
        self.finish()
        for sample in self.samples:
            if sample.valid():
                sample.write(writer)

class PatientICUInSampleSetting:
    pred_bias_time = datetime.timedelta(days = 0.5)
    intervel_time = datetime.timedelta(days = 1.0)
    max_sample_per_icu = 5
    nb_icu = 0
    def __init__(self, pid):
        self.pid = pid
        self.settings = []

    def gen_sample_setting(self, patient):
        for hos in patient.hospital_settings:
            hos_st = hos.st
            hos_ed = hos.ed
            last_time = hos_st
            for icu in hos.icu_settings:
                icu_st = icu.st
                icu_ed = icu.ed
                assert icu_st >= last_time
                pred_time = icu_st - PatientICUInSampleSetting.pred_bias_time
                for i in range(PatientICUInSampleSetting.max_sample_per_icu):
                    if pred_time < last_time:
                        break
                    sample_setting = SampleSetting(hos_st, pred_time, 1, icu_st, PatientICUInSampleSetting.nb_icu)
                    self.settings.append(sample_setting)
                    pred_time -= PatientICUInSampleSetting.intervel_time
                PatientICUInSampleSetting.nb_icu += 1
                last_time = icu_ed
            pred_time = hos_ed - PatientICUInSampleSetting.pred_bias_time
            for i in range(PatientICUInSampleSetting.max_sample_per_icu):
                if pred_time < last_time:
                    break
                sample_setting = SampleSetting(hos_st, pred_time, 0, hos_ed, PatientICUInSampleSetting.nb_icu)
                self.settings.append(sample_setting)
                pred_time -= PatientICUInSampleSetting.intervel_time
            PatientICUInSampleSetting.nb_icu += 1

    def to_json(self):
        obj = {
            "pid": self.pid,
            "settings": [setting.to_json() for setting in self.settings]
        }
        return obj

    @staticmethod
    def load_from_json(obj):
        pid = obj['pid']
        patient_setting = PatientICUInSampleSetting(pid)
        patient_setting.settings = [SampleSetting.load_from_json(setting_obj) for setting_obj in obj['settings']]
        return patient_setting


class Sample:
    '''
        init with a sample setting and max event seq length
        add event not in black_list
    '''
    def __init__(self, sample_setting, max_len, pid):
        self.sample_setting = sample_setting
        self.max_len = max_len
        self.pid = pid
        self.events = []
    
    def add_event(self, event):
        if event.index in black_list:
            return
        if event.time > self.sample_setting.st and event.time < self.sample_setting.ed:
            self.events.append(event)
            if len(self.events) >= 1.5 * self.max_len:
                self.events = sorted(self.events)[-self.max_len:]

    def finish(self):
        self.events = sorted(self.events)[-self.max_len:]
    
    
    def valid(self):
        if len(self.events) > 10:
            return True
        else:
            return False
    
    def write(self, writer):
        writer.write(json.dumps(self.to_json()) + "\n")
    
    def to_json(self):
        obj = {
            "pid": self.pid,
            "sample_setting": self.sample_setting.to_json(),
            "max_len": self.max_len,
            "events": [str(event) for event in self.events]
        }
        return obj

    @staticmethod   
    def load_from_json(obj):
        sample_setting = SampleSetting.load_from_json(obj['sample_setting'])
        pid = obj['pid']
        max_len = obj['max_len']
        sample = Sample(sample_setting, max_len, pid)
        for event_str in obj['events']:
            sample.events.append(Event.load_from_line(event_str))
        return sample

    @staticmethod
    def load_label(obj):
        sample_setting = SampleSetting.load_from_json(obj['sample_setting'])
        return sample_setting.label
        
        

class SampleSetting:
    '''
        label: 1 means enter icu, 0 means not 
        st-ed: event sequence slot
        label_time: when is the prediction event
        sample_id: indicate which icu this sample comes from
    '''
    def __init__(self, st, ed, label, label_time, sample_id):
        if type(st) == unicode:
            self.st = parse_time(st)
        else:
            self.st = st
        if type(ed) == unicode:
            self.ed = parse_time(ed)
        else:
            self.ed = ed
        if type(label_time) == unicode:
            self.label_time = parse_time(label_time)
        else:
            self.label_time = label_time   
        self.sample_id = sample_id
        self.label = label
        assert self.st
        assert self.ed
        assert self.label_time

    def to_json(self):
        obj = {
            "st": str(self.st),
            "ed": str(self.ed),
            "label": self.label,
            "label_time": str(self.label_time),
            "sample_id": self.sample_id
        }
        return obj
    
    @staticmethod
    def load_from_json(obj):
        return SampleSetting(**obj)



class PatientSetting:
    def __init__(self, pid):
        self.pid = pid
        self.hospital_settings = []

    def add_hospital_setting(self, hos_setting):
        self.hospital_settings.append(hos_setting)

    def to_json(self):
        obj = {
            "pid": self.pid,
            "hospital_settings": []
        }
        for hos_setting in self.hospital_settings:
            obj['hospital_settings'].append(hos_setting.to_json())
        return obj

    @staticmethod
    def load_from_json(obj):
        pid = obj["pid"]
        patient = PatientSetting(pid)
        for hos_obj in obj['hospital_settings']:
            hos = HospitalSetting.load_from_json(hos_obj)
            patient.add_hospital_setting(hos)
        return patient
                   

class HospitalSetting:
    def __init__(self, st, ed):
        self.st = st
        self.ed = ed
        self.icu_settings = []

    def add_icu_setting(self, icu_setting):
        self.icu_settings.append(icu_setting)

    def to_json(self):
        obj = {
            "st": str(self.st),
            "ed": str(self.ed),
            "icu_settings": [],
        }
        for icu_setting in self.icu_settings:
            obj['icu_settings'].append(icu_setting.to_json())
        return obj

    @staticmethod
    def load_from_json(obj):
        st = parse_time(obj["st"])
        ed = parse_time(obj["ed"])
        hos = HospitalSetting(st, ed)
        for icu_obj in obj['icu_settings']:
            icu = ICUSetting.load_from_json(icu_obj)
            hos.add_icu_setting(icu)
        return hos
        

class ICUSetting:
    def __init__(self, st, ed):
        self.st = st
        self.ed = ed

    def to_json(self):
        obj = {
            "st": str(self.st),
            "ed": str(self.ed),
        }
        return obj

    @staticmethod
    def load_from_json(obj):
        st = parse_time(obj["st"])
        ed = parse_time(obj["ed"])
        icu = ICUSetting(st, ed)
        return icu



def gen_settings(patient_cnt_map):
    setting_map = {}
    for pid in patient_cnt_map:
        patient_setting = PatientSetting(pid)
        setting_map[pid] = patient_setting
        for hospital_cnt in patient_cnt_map[pid].hospital_cnts:
            hospital_setting = HospitalSetting(hospital_cnt.admit_time, hospital_cnt.disch_time)
            patient_setting.add_hospital_setting(hospital_setting)
            for icu_cnt in hospital_cnt.icu_cnts:
                icu_setting = ICUSetting(icu_cnt.st, icu_cnt.ed)
                hospital_setting.add_icu_setting(icu_setting)
    return setting_map

def write(out_path, patient_setting_map):
    writer = file(out_path, 'w')
    for pid in patient_setting_map:
        obj = patient_setting_map[pid].to_json()
        writer.write(json.dumps(obj))
        writer.write("\n")
    writer.close()

def load_patient_setting(setting_path):
    reader = file(setting_path)
    patient_setting_map = {}
    for line in reader:
        p_setting = PatientSetting.load_from_json(json.loads(line))
        patient_setting_map[p_setting.pid] = p_setting
    return patient_setting_map

def gen_sample_setting(setting_out_path, patient_setting_map):
    writer = file(setting_out_path, 'w')
    for pid in sorted(patient_setting_map.keys()):
        sample_setting = PatientICUInSampleSetting(pid)
        sample_setting.gen_sample_setting(patient_setting_map[pid])
        obj = sample_setting.to_json()
        writer.write(json.dumps(obj) + "\n")
    writer.close()

def load_ICUIn_sample_setting(sample_setting_path):
    sample_setting_map = {}
    for line in file(sample_setting_path):
        sample_setting = PatientICUInSampleSetting.load_from_json(json.loads(line))
        sample_setting_map[sample_setting.pid] = sample_setting
    return sample_setting_map

def simple_count(sample_setting_map):
    tot_sample = 0   
    max_sample = 0
    for pid in sample_setting_map:
        sample_setting = sample_setting_map[pid]
        tot_sample += len(sample_setting.settings)
        max_sample = max(max_sample, len(sample_setting.settings))

    print tot_sample / len(sample_setting_map)
    print max_sample

    
    


if __name__ == "__main__":
    # generate patient settings
    # event_seq_stat_result_path = os.path.join(event_seq_stat_dir, "event_seq_stat.result")
    # patient_cnt_map = load(event_seq_stat_result_path)
    # setting_map = gen_settings(patient_cnt_map)
    # result_path = os.path.join(event_seq_stat_dir, "patient_setting.txt")
    # write(result_path, setting_map)

    # load patient setting
    # setting_path = os.path.join(event_seq_stat_dir, "patient_setting.txt")
    # patient_setting_map = load_patient_setting(setting_path)

    # generate icu entrance prediciton sample settings 
    # sample_setting_path = os.path.join(event_seq_stat_dir, "ICUIn_sample_setting.txt")
    # gen_sample_setting(sample_setting_path, patient_setting_map)

    # load icu sample setting & count
    sample_setting_map = load_ICUIn_sample_setting(sample_setting_path)
    simple_count(sample_setting_map)