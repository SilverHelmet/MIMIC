import gmm_util
from util import *
import numpy as np
from scripts import gen_fix_segs
from sklearn.mixture import GMM
import sys
from gen_segs import gen_segs_by_dp, gen_segs_by_dp_mp
import h5py

def merge_one_event(event, seg, aggre_mode, emds, emd_dim, X):
    st = 0  
    for ed in seg:
        if ed == 0:
            continue
        x = np.zeros(emd_dim)
        for idx in event[st:ed]:
            x += emds[idx]
        if aggre_mode == "ave":
            x /= ed - st
        elif aggre_mode == 'sum':
            pass
        else:
            print "error"
        X.append(x)
        st = ed

def merge(events, segs, aggre_mode, emds, emd_dim):
    assert len(events) == len(segs)
    length = len(events)
    X = []
    for i in xrange(length):
        merge_one_event(events[i], segs[i], aggre_mode, emds, emd_dim, X)
        if i % 10000 == 0:
            print "\tmerge %d" %i
            
            
    return np.array(X)
        
def load_dataset(dataset):
    f = h5py.File(dataset)
    return f['event'][:]

def load_segs(dataset):
    f = h5py.File(dataset)
    return f['segment'][:]
    
def write_gmm(gmm):
    weights = gmm.weights_
    covars = gmm.covars_
    means = gmm.means_
    gmm_dir = os.path.join(script_dir, "gmm")
    model_dir = os.path.join(gmm_dir, "model")
    if not os.path.exists(model_dir):
        print "mkdir [%s]" %model_dir
        os.mkdir(model_dir)
    print "save model params to [%s]" %model_dir
    np.save(os.path.join(model_dir, "weights"), weights)
    np.save(os.path.join(model_dir, "means"), means)
    np.save(os.path.join(model_dir, "covars"), covars)

def main(argv):
    setting = load_setting(argv[1])
    emds = load_numpy_array(setting['emd'])
    emd_dim = emds.shape[1]
    train_dataset = setting['train_dataset']
    test_dataset = setting['test_dataset']
    train_events = load_dataset(dataset = train_dataset)
    test_events = load_dataset(dataset = test_dataset)
    train_segs = load_segs(dataset = setting['train_segs'])
    max_segs = train_segs.shape[1]
    aggre_mode = setting['aggre_mode']
    train_segs_out = setting['train_segs_out']
    test_segs_out = setting['test_segs_out']


    print "emd dim = %d" %emd_dim
    print "max seg = %d" %max_segs
    print "aggre_mode = %s" %aggre_mode
    print "merging"
    X = merge(train_events, train_segs, aggre_mode, emds, emd_dim)
    print "merging end"
    print "training GMM"
    gmm = GMM(n_components = 30, covariance_type='diag', n_iter = 100, verbose = 2)
    gmm.fit(X)
    # write_gmm(gmm)
    print "end training"

    gen_segs_by_dp_mp(train_events, gmm, emds, aggre_mode, max_segs, 14, train_segs_out)
    gen_segs_by_dp_mp(test_events, gmm, emds, aggre_mode, max_segs, 10, test_segs_out)

    # new_segs = gen_segs_by_dp(events, gmm, emds, aggre_mode, max_segs)



if __name__ == "__main__":
    #python train_gmm.py emd_file dataset_file seg_mode aggre_mode 
    main(sys.argv)