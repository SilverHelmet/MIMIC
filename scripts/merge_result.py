import os
from get_rnn_result import parse_result

def decom(p):
    acc, merged_acc = p[0].split("/")
    auc, merged_auc = p[1].split("/")
    arc, merged_arc = p[2].split("/")
    return (acc, auc, arc, merged_acc, merged_auc, merged_arc)

def merge(p1, p2):
    res = []
    for term1, term2 in zip(p1, p2):
        term1 = float(term1)
        term2 = float(term2)
        term = (term1 + term2) / 2
        delta = round(abs(term1 - term), 4)
        term = round(term, 4)
        res.append(str(term) + "(" + str(delta) + ")")
    merged_res = []
    for i in range(3):
        merged_res.append(res[i] + "/" + res[i+3])
    return merged_res


out = file("tmp.txt", "w")
while True:
    res1 = raw_input("log1:")
    res2 = raw_input('log2:')
    p1 = res1.strip().split("\t")
    p1 = decom(p1)
    p2 = parse_result(res2)
    res = merge(p1, p2)
    print "\t".join(res)
    out.write("\t".join(res) + "\n")
    out.flush()
out.close()

    

