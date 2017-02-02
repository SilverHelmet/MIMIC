import scripts_util
import sys
from models.dataset import Dataset

dataset = sys.argv[1]
seg = sys.argv[2]
out_dataset = sys.argv[3]
out_seg = sys.argv[4]

d = Dataset(dataset, seg)
d.load()
sample_d = d.sample()
sample_d.save(out_dataset, out_segs)



