# encoding:utf8
import pandas
import sys
import math

def sort_and_write(outf_path, diagnosis_map):
    diagnoses = set()
    for diagnoses_set in diagnosis_map.itervalues():
        diagnoses = diagnoses.union(diagnoses_set)
    diag_cnt = {}
    for diag in diagnoses:
        diag_cnt[diag] = 0
    for diagnoses_set in diagnosis_map.itervalues():        
        for diag in diagnoses_set:
            diag_cnt[diag] += 1
    
    keys = diag_cnt.keys()
    keys.sort(reverse = True, key = lambda x: diag_cnt[x])
    outf = file(outf_path, 'w')
    outf.write("ID cnt = %d\n" %len(diagnosis_map) )
    for diag in keys:
        ratio = round((diag_cnt[diag] + 0.0) / len(diagnosis_map), 3)
        out = [diag, diag_cnt[diag], ratio]
        out = "\t".join(map(str, out))
        outf.write(out + "\n")
    outf.close()        

if __name__ == "__main__":
    table = pandas.read_csv(sys.argv[1])
    patient_diagnosis_map = {}
    hos_diagnosis_map = {}
    for idx, row in table.iterrows():
        pid = row['Patient_SN']
        hid = row['病历号']
        diagnosis = row["诊断"]
        if not pid in patient_diagnosis_map:
            patient_diagnosis_map[pid] = set()
        patient_diagnosis_map[pid].add(diagnosis)
        
        if not hid in hos_diagnosis_map:
            hos_diagnosis_map[hid] = set()
        hos_diagnosis_map[hid].add(diagnosis)


    sort_and_write('patient_diag.txt', patient_diagnosis_map)
    sort_and_write('hos_diag.txt', hos_diagnosis_map)
    

        
