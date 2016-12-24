import h5py
import sys


if __name__ == "__main__":
    f = h5py.File(sys.argv[1], 'r')
    segs = f['segment'][:]
    max_seg_length = 0
    for seg in segs:
        st = 0
        for ed in seg:
            if ed == 0:
                break
            max_seg_length = max(max_seg_length, ed -st)
            st = ed
    print max_seg_length
