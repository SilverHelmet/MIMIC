import os
import matplotlib
matplotlib.use('Agg')
scripts_dir = __file__
import sys
sys.path.append(os.path.dirname(os.path.dirname(scripts_dir)))

 

admit_text = ["ELECTIVE", "EMERGENCY", "NEWBORN", "URGENT"]
admit = [2, 3, 4, 5]
emerg_admit = [3, 5]
disch = [6]
icu = [7]
death = [2371]
icu_leave = [3418]
black_list = [2371]

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


