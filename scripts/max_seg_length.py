import h5py
import sys


if __name__ == "__main__":
    f = h5py.File(sys.argv[1], 'r')
    segs = f['segment'][:]
    max_seg_length = 0
    seg_lengths = []
    for seg in segs:
        st = 0
        for ed in seg:
            if ed == 0:
                break
            max_seg_length = max(max_seg_length, ed -st)
            seg_lengths.append(ed - st)
            st = ed
    print max_seg_length
    print 'size = %d' %(len(seg_lengths))
    seg_lengths.sort(reverse = True)
    print "up 100 = %d" %seg_lengths[100]
    print "up 1000 = %d" %seg_lengths[1000]
    print "up 10000 = %d" %seg_lengths[10000]
    print "up 100000 = %d" %seg_lengths[100000]
    print "up 500000 = %d" %seg_lengths[500000]
