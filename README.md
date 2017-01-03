## build ICU In samples
`python scripts/patient.py`

    generate event_seq info of each patient/hospital 
    read from `event/*.tsv` write to `result/event_seq.dat`

`python scripts/stat_event_seq.py`

    generate event_seq stat
    read from `result/event_seq.dat` write to `event_seq_stat/event_seq_stat.result`

`python scripts/sample_setting.py`

    generate ICUIn sample setting 
    read from `event_seq_stat/event_seq_stat.result` 
    write to `event_seq_stat/patient_setting.txt` and `event_seq_stat/ICUIn_sample_setting.txt`



