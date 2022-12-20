from torch.utils.data import Dataset
import os
import h5py    
import numpy as np
import json


class ShapeNetChairs(Dataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__()
        fnames = open(os.path.join(root, f"{split}_files.txt")).readlines()
        fnames = [os.path.join(root, os.path.basename(fname.strip().replace("\n", ""))) for fname in fnames]
        pcs = []
        for fname in fnames:
            data = h5py.File(fname, "r")    
            pc = data["data"][:].astype("float32")
            label = json.load(open(fname[:-3] + "_id2name.json", "r"))
            for pc_, label_ in zip(pc, label):
                # if label_ == "chair":
                pcs.append(pc_)
        self.pcs = np.stack(pcs, axis=0)
        self.transform = transform
    
    def __getitem__(self, indx):
        pc = self.pcs[indx]
        if self.transform is not None:
            pc = self.transform(pc)
        return pc
    
    def __len__(self):
        return len(self.pcs)


if __name__ == "__main__":
    dset = ShapeNetChairs("shapenetcorev2_hdf5_2048")
    print(dset[0].shape)