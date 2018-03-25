from util import connect, static_data_dir, Print, time_format_str
import os
import json
import sys

def download_table(db, out_dir, table, names, time_attrs, limit = 100000):
    outpath = os.path.join(out_dir, table + ".json")
    outf = file(outpath, 'w')
    offset = 0
    while True:
        query = "select * from %s order by row_id limit %d offset %d" %(table, limit, offset)
        Print("\t%s" %query)
        res = db.query(query)
        for row in res.dictresult():
            new_row = {}
            for name in names:
                new_row[name] = row[name]
                if name in time_attrs:
                    new_row[name] = time_format_str(row[name])

            outf.write(json.dumps(new_row) + '\n')

        ntuples = res.ntuples()
        if ntuples < limit:
            break
        offset += ntuples
    outf.close()

if __name__ == "__main__":
    db = connect()
    out_dir = os.path.join(static_data_dir, 'static_feature')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    tables = ['admissions', 'diagnoses_icd', 'patients', 'procedures_icd']
    time_attrs = set(['dob'])
    name_lists = [
        ['subject_id', 'hadm_id', 'insurance', 'language', 'religion', 'marital_status', 'ethnicity', 'diagnosis'],
        ['subject_id', 'hadm_id', 'seq_num', 'icd9_code'], 
        ['subject_id', 'gender', 'dob'],
        ['subject_id', 'hadm_id', 'seq_num', 'icd9_code']]


    for table, names in zip(tables, name_lists):
        download_table(db, out_dir, table, names, time_attrs)

