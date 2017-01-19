import h5py
import scripts_util
import util
from util import *
import sys
import math
import numpy as np
import datetime
import os
from numba import jit


def split_to_fix_chunk(event_seq, nb_trunk = 0, mode = "fixChunk#length", time_seq = None):
    if mode == "fixChunk#length":
        event_seq = [i for i in event_seq if i != 0]
        limit = int(math.ceil(len(event_seq)/(nb_trunk + 0.0)))
        seg = []
        st = 0
        while st < len(event_seq):
            ed = min(st + limit, len(event_seq))
            seg.append(ed)
            st = ed
        seg.extend([0] * (nb_trunk-len(seg)))
        return seg 
    elif mode == "fixChunk#time":
        assert time_seq is not None
        event_seq = [i for i in event_seq if i != 0]
        time_seq = [util.parse_time(time_str) for time_str in time_seq]
        fi = len(event_seq)
        delta = time_seq[fi-1] - time_seq[0]
        chunk_delta = delta / nb_trunk
        seg = []
        st = 0
        while st < len(event_seq):
            edtime = time_seq[st] + chunk_delta
            ed = st
            while ed < fi and time_seq[ed] <= edtime:
                ed += 1
            seg.append(ed)
            st = ed
            if len(seg) == nb_trunk - 1:
                seg.append(fi)
                break
        seg.extend([0] * (nb_trunk-len(seg)))
        return seg

def split_by_time(event_seq, time_seq, time_slot, nb_trunk):
    event_seq = [i for i in event_seq if i != 0]
    time_seq = [util.parse_time(time_str) for time_str in time_seq]
    fi = len(event_seq)
    st = 0
    seg = []
    while st < fi:
        edtime = time_seq[st] + time_slot 
        ed = st
        while ed < fi and time_seq[ed] <= edtime:
            ed += 1
        seg.append(ed)
        st = ed
        if len(seg) == nb_trunk:
            break
    seg.extend([0] * (nb_trunk-len(seg)))
    return seg

def split_by_length(event_seq, nb_chunk, chunk_length):
    event_seq = [i for i in event_seq if i != 0]
    fi = len(event_seq)
    st = 0
    seg = []
    while st < fi:
        ed = min(fi, st + chunk_length)
        seg.append(ed)
        st = ed
    assert len(seg) <= nb_chunk
    seg.extend([0] * (nb_chunk-len(seg)))
    return seg

# @jit(cache = True)
def check_valid(times, max_duration, max_chunk):
    # seg = np.zeros(max_chunk, ntype = int)
    seg = [0] * max_chunk
    i = 0
    st = 0
    fi = len(times)
    while st < fi:
        if i == max_chunk:
            return None
        ed = st + 1
        while ed < fi and times[ed] - times[st] <= max_duration:
            ed += 1
        seg[i] = ed
        i += 1
        st = ed
    return seg



eps = 0.001
def get_timeAggre(times, max_chunk):
    global eps
    l = 0
    r = times[-1] + eps * 2
    while l + eps < r:
        mid = (l + r) / 2
        seg = check_valid(times, mid, max_chunk)
        if seg is not None:
            r = mid
        else:
            l = mid
    seg = check_valid(times, r, max_chunk)
    assert seg is not None
    return seg

    


def split_by_timeAggre(event_seq, time_seq, max_chunk):
    event_seq = [i for i in event_seq if i != 0]
    fi = len(event_seq)
    time_seq = [parse_time(time) for time in time_seq[:fi]]
    time_bias = [(time - time_seq[0]).total_seconds() / 3600.0 for time in time_seq]
    seg = get_timeAggre(time_bias, max_chunk)
    return seg


def infer_path(dataset_path, seg_dir, mode, max_chunk, chunk_length):
    filename = os.path.basename(dataset_path).split('.h5')[0]
    if mode == "fixLength":
        return os.path.join(seg_dir, filename + "_segmode=%s_maxchunk=%d_length=%d.h5" %(mode, max_chunk, chunk_length))
    return os.path.join(seg_dir, filename + "_segmode=%s_maxchunk=%d.h5" %(mode, max_chunk))

def infer_and_load(dataset_path, mode):
    path = infer_path(dataset_path, mode)
    print "load segment from [%s]" %path
    f = h5py.File(path, 'r')
    seg = f['segment'][:]
    f.close()
    return seg


if __name__ == "__main__":
    mode = "fixChunk#length"
    mode = "fixChunk#time"
    mode = "fixTime"
    mode = "fixLength"
    mode = "timeAggre"
    time_slot = datetime.timedelta(days = 0.1)
    
    seg_dir = ICU_seg_dir
    max_chunks = 50
    chunk_length = 20

    dataset_path = sys.argv[1]
    mode = sys.argv[2]
    chunk_length = int(sys.argv[3])
    seg_dir = sys.argv[4]

    
    seg_out_path = infer_path(dataset_path, seg_dir, mode, max_chunks, chunk_length)
    print "load data from [%s] write segs to [%s], mode = [%s]" %(dataset_path, seg_out_path, mode)

    f = h5py.File(dataset_path, 'r')
    event = f['event'][:]
    if mode == "fixTime" or mode == "fixChunk#time" or mode == "timeAggre":
        times = f['time'][:]
    f.close()
    segs = []

    print "max_chunks = %d" %max_chunks
    print "chunk_length = %d" %chunk_length
    for idx, event_seq in enumerate(event):
        if idx % 10000 == 0:
            now_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
            print "\t %s collect %d" %(now_time, idx)
        if mode == "fixChunk#length":
            seg = split_to_fix_chunk(event_seq, max_chunks, mode)
        elif mode == "fixChunk#time":
            seg = split_to_fix_chunk(event_seq, max_chunks, mode, times[idx])
        elif mode == "fixTime":
            seg = split_by_time(event_seq, times[idx], time_slot, max_chunks)
        elif mode == "fixLength":
            max_chunks = int(math.ceil(1000/chunk_length))
            seg = split_by_length(event_seq, max_chunks, chunk_length = chunk_length)
        elif mode == "timeAggre":
            seg = split_by_timeAggre(event_seq, times[idx], max_chunks)
        else:
            print "error"
            break
        segs.append(seg)
    out_f = h5py.File(seg_out_path, 'w')
    out_f['segment'] = np.array(segs, dtype=int)
    out_f['max_chunks'] = max_chunks
    out_f.close()
