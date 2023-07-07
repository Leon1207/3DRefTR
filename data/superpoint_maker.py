# superpoints fetch
import segmentator
import open3d as o3d
import torch
import os
from six.moves import cPickle
import numpy as np
from torch_scatter import scatter_mean
import wandb

# BRIEF read from pkl
def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def generate_superpoint():
    data_path = r"/userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/"
    split = "val"

    scans = unpickle_data(f'{data_path}/{split}_v3scans.pkl')
    scans = list(scans)[0]

    print("Begin processing...")
    for scan in scans:
        spformer_file = os.path.join("/userhome/lyd/SPFormer/data/scannetv2", split, scan + "_vh_clean_2.ply")
        mesh = o3d.io.read_triangle_mesh(spformer_file)
        vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
        faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
        superpoint = segmentator.segment_mesh(vertices, faces).numpy()
        superpoint = torch.from_numpy(superpoint)
        select_idx = torch.tensor(scans[scan].choices)
        superpoint = torch.index_select(superpoint, 0, select_idx).numpy()
        torch.save(superpoint, os.path.join("/userhome/lyd/RES/superpoint", split, scan + "_superpoint.pth"))
        print("Saving " + scan)

    print("Done.")


if __name__ == '__main__':
    generate_superpoint()



