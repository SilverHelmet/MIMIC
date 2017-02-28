#encoding:utf-8
import datetime
import json
import pandas
import zhu_util
from util import *


def load_grade_map(filepath):
    grade_map = {}
    table = pandas.read_csv(filepath)
    for idx, row in table.iterrows():
        diag = row['行标签']
        grade = row['分级']
        grade_map[diag] = grade
    return grade_map

grade_map = load_grade_map("zhu_data/诊断评分.csv")
miss_diag = 0
class Diagnosis:
    def __init__(self, diag, time_str):
        global grade_map, miss_diag
        self.diag = diag
        self.time = parse_time(time_str)
        if not diag in grade_map:
            miss_diag += 1
            self.grade = 0
        else:
            self.grade = grade_map[diag]
        if self.time is None:
            print time_str
        assert self.time

    def __cmp__(self, other):
        return cmp(self.time, other.time)

class RangeDiagnosis:
    def __init__(self, time, label):
        self.time = time
        self.label = label
    

class PatientDiagnosis:
    def __init__(self, pid):
        self.pid = pid
        self.diags = []


    def add_diag(self, time, diag):
        if time in ["4期","3期","2期"]:
            return
        diag =  Diagnosis(diag, time)
        self.diags.append(diag)

    def finish(self):
        time_range = datetime.timedelta(seconds = 1800)
        self.diags.sort()
        st = 0
        range_diag = []
        while st < len(self.diags):
            ed = st + 1
            max_grade = self.diags[st].grade
            while ed < len(self.diags) and self.diags[ed].time - self.diags[st].time <= time_range:
                max_grade = max(max_grade, self.diags[ed].grade)
                ed += 1
            label = int(max_grade >= 7)
            range_diag.append(RangeDiagnosis(self.diags[ed-1].time, label))
            st = ed
        self.diags = range_diag


        
    

class DiagnosisSampleSetting:
    def __init__(self, pid, ed, label):
        self.pid = pid
        self.ed = ed
        self.label = label

    def to_json(self):
        obj = {
            "pid": self.pid, 
            "ed": self.ed,
            "label": self.label,
        }
        return json.dumps(obj)

    @staticmethod
    def load_from_json(obj_str):
        obj = json.loads(obj_str)
        pid = obj['pid']
        ed = parse_time(obj[ed])
        label = obj['label']
        return DiagnosisSampleSetting(pid, ed, label)

        


def load_diagnosis():
    table = pandas.read_csv("zhu_data/诊断.csv")
    print table.columns
    patient_diagnosis_map = {}

    for idx, row in table.iterrows():
        pid = row['Patient_SN']
        if not pid in patient_diagnosis_map:
            patient_diagnosis_map[pid] = PatientDiagnosis(pid)
        diagnosis = row['诊断']
        time_str = row['诊断时间']
        if time_str in ["1期", "2期", "3期", "4期", "5期"]:
            time_str = row['诊断类型']
        
        patient_diagnosis_map[pid].add_diag(time_str, diagnosis)
    for p_diag in patient_diagnosis_map.values():
        p_diag.finish()
    return patient_diagnosis_map

def buld_patient_sample_setting(self, patient_diag, pid):
    pred_bias_time = datetime.timedelta(days = 0.5)
    interval_time = datetime.timedelta(days = 1.0)
    settings = []

    last_diag_time = None
    for diag in patient_diag.diags:
        for i in range(5):
            ed = diag.time - pred_bias_time - i * interval_time
            label = diag.label
            if last_diag_time is None or last_diag_time < ed:
                sample_setting = DiagnosisSampleSetting(pid, ed, label)
                settings.append(sample_setting)
        last_diag_time = diag.time
    return settings
        

def build_sample_setting(patient_diagnosis_map):
    settings = []
    cnt = 0
    for pid in sorted(patient_diagnosis_map.keys()):
        cnt += 1
        if cnt % 10 == 0:
            print "cnt = %d" %cnt
        patient_diag = patient_diagnosis_map[pid]
        patient_settings = build_patient_sample_setting(patient_diag, pid)
        settings.extend(patient_settings)
    return settings
        
        
            

            



if __name__ == "__main__":
    patient_diagnosis_map = load_diagnosis()
    settings = build_sample_setting(patient_diagnosis_map)
    outf = file("zhu_data/sample_settings.json")
    for setting in settings:
        outf.write(setting.to_json() + "\n")
    outf.close()

    print "miss diagnosis = %d" %miss_diag
