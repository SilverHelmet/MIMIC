import os
import glob
import re

def parse_result(line):
    line = line.strip()
    p = line.find("acc")
    line = line[p:]
    parts = line.split(", ")


    return [float(part.split(" =")[1]) for part in parts]

def output(p):
    t = []
    p = map(lambda x: round(x, 4), p)
    for i in range(3):
        t.append(str(p[i]) + '/' + str(p[i+3]))
    print "\t".join(t)
    
if __name__ == "__main__":
    while True:
        line = raw_input("log:")
        p = parse_result(line)
        output(p)
    # max_auc = 0
    # filename = None
    # for filepath in glob.glob("log/icu_timeAggre_attention*"):
    #     for line in file(filepath):
    #         if line.startswith("round"):
    #             res = parse_result(line)
    #             if res[-2] > max_auc:
    #                 max_auc = res[-2]
    #                 filename = os.path.basename(filepath)
    # print max_auc
    # print filename