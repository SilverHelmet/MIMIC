import os
from get_rnn_result import parse_result

def decom(p):
    acc, merged_acc = p[0].split("/")
    auc, merged_auc = p[1].split("/")
    arc, merged_arc = p[2].split("/")
    return (acc, auc, arc, merged_acc, merged_auc, merged_arc)

def new_decom(p):
    return map(float, p)

def merge(p1, p2):
    res = []
    if len(p1) < len(p2):
        tmp = p1
        p1 = p2
        p2 = tmp
    if len(p1) > len(p2):
        p1 = [p1[0], p1[1], p1[2], p1[4],p1[5],p1[6]]
    for term1, term2 in zip(p1, p2):
        term1 = float(term1)
        term2 = float(term2)
        term = (term1 + term2) / 2
        delta = round(abs(term1 - term), 4)
        term = round(term, 4)
        res.append(str(term) + "(" + str(delta) + ")")
    merged_res = []
    bias = min(len(p1), len(p2)) / 2
    for i in range(3):
        merged_res.append(res[i] + "/" + res[i+bias])
    return merged_res

def new_merge(p1, p2):
    res = []
    for t1, t2, in zip(p1, p2):
        t1 = float(t1)
        t2 = float(t2)
        t = (t1 + t2) / 2.0
        delta = round(abs(t1 - t), 4)
        t = round(t, 4)
        res.append(str(t) + "(" + str(delta) + ")")

    return res

# out = file("tmp.txt", "w")
while True:
    res1 = raw_input("log1:")
    res2 = raw_input('log2:')
    p1 = res1.strip().split("\t")
    if len(p1) == 3:
        p1 = new_decom(p1)
    else:
        p1 = parse_result(res1)
    p2 = res2.strip().split('\t')
    if len(p2) == 3:
        p2 = new_decom(p2)
    else:
        p2 = parse_result(res2)
    res = new_merge(p1, p2)
    print "\t".join(res)
    # out.write("\t".join(res) + "\n")
    # out.flush()
# out.close()

    

