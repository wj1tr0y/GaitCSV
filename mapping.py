import os
from tqdm import tqdm
src_path = '/data/Gait3D-sils-64-44-pkl/'
dst_path = '/data/Gait3D-sils-64-44-pkl_view_order'

f = open('gait_3d_cluster_results.txt','r')
for line in tqdm(f.readlines()):
    src, dst = line.strip().split('>')
    src = os.path.join(src_path, src)
    dst = os.path.join(src_path, dst)
    os.makedirs('/'.join(dst.split('/')[:-1]))
    os.symlink(src, dst)