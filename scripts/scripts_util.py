import os
import matplotlib
matplotlib.use('Agg')
scripts_dir = __file__
import sys
sys.path.append(os.path.dirname(os.path.dirname(scripts_dir)))

 

admit_text = ["NEWBORN", "URGENT", "ELECTIVE", "EMERGENCY"]
admit = [3, 4, 5, 6]
emerg_admit = [4, 6]
disch = [2]
icu = [7]
death = [2004]
icu_leave = [3391]
black_list = [2004]

def is_admit(event):
    global admit
    return event.eid in admit

def is_emerg_admit(event):
    global emerg_admit
    return event.eid in emerg_admit

def is_disch(event):
    global disch
    return event.eid in disch

def is_icu_in(event):
    global icu
    return event.eid in icu

def is_icu_leave(event):
    global icu_leave
    return event.eid in icu_leave


