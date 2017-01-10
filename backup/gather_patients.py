from pg import DB
from pandas import DataFrame
from util import connect, Patient


def gather_basic_info(db):
    print "query patients table"
    res = db.query('select * from patients')
    columns = [field for field in res.listfields() if not field == 'row_id'] 
    print "query replyed"
    Patient.set_attrs(columns)
    patients = {}
    for patient in res.dictresult():
        pid = patient['subject_id']
        patients[pid] = Patient(patient)
    return patients


def gather_admission_info(db, patients):
    pass


def main():
    db = connect()
    patients = gather_basic_info(db)
    Patient.write_to_local(patients, 'data/patients.csv')
    gather_admission_info(db, patients)


if __name__ == '__main__':
    main()


# print res.getresult()

