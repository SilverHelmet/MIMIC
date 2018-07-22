import numpy as np


def one(x):
    return 1

def invert(x, max_value = 2.0):
    if x == 0:
        return max_value
    else:
        return min(abs(1.0/x), max_value)


time_funcs = {
    'one': one,
    'abs': abs,
    'invert': invert,
}


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


def build_time_graph_2(times, width, A, time_func = one):
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
        print i, st, ed
        for j in range(st, ed):
            A[i][j] = time_func(times[j] - times[i])

def get_seg_time(times, split):
    seg_time = [-1] * len(split)
    st = 0
    for idx, ed in enumerate(split):
        if ed == 0:
            break
        mid = (st + ed) / 2

        seg_time[idx] = times[mid]
        st = ed
    return seg_time

if __name__ == "__main__":
    pass