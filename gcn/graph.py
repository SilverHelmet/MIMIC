import numpy as np


def build_time_graph(times, width, A):
    n = len(times)
    # A = np.zeros((n, n))
    for i in range(n):
        if times[i] < 0:
            break
        j = i
        while j < n:
            if times[j] - times[i] <= width and times[j] >= 0:
                A[i][j] = 1
                A[j][i] = 1
                j += 1
            else:
                break
    # return A

def build_time_graph_2(times, width, A):
    n = len(times)
    # A = np.zeros((n, n))
    st = 0
    ed = 0
    for i in range(n):
        if times[i] < 0:
            break
        while times[i] - times[st] > width:
            st += 1
        while ed < n and times[ed] - times[i] <= width and times[ed] >= 0:
            ed += 1
        A[i][st:ed] = 1

