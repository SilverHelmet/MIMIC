import numpy as np


def build_time_graph(times, width, A):
    n = len(times)
    # A = np.zeros((n, n))
    for i in range(n):
        if times[i] < 0:
            break
        j = i
        while j < n:
            if times[j] - times[i] <= width:
                A[i][j] = 1
                A[j][i] = 1
                j += 1
            else:
                break
    # return A

