import numpy as np
from torch.utils.data import Dataset
import torch
import mmap
import ast
import gc


class Ala25Dataset(Dataset):
    def __init__(self, num_samples=10, ifname="alanine_dipeptide_25.npy"):
        self.num_samples = num_samples
        self.ifname = ifname

        # self.get_data_points(batch_size=num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def get_data_point(self, batch_size=1):
        with open(self.ifname, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            
            # Get the offset containing the file header
            header_len = mm[8] + mm[9]
            # 10 is the length of magic string for npy file
            unaligned_offset = header_len + 10
            
            # Offset must be a multiple of dtype size
            # for alignment purposes
            if unaligned_offset % 64 != 0:
                offset = unaligned_offset + unaligned_offset % 64
            else:
                offset = unaligned_offset

            # Get array size from header information
            info_bytes = mm[10:offset]
            info_dict = ast.literal_eval(info_bytes.decode('utf-8'))
            arr_shape = info_dict['shape']

            # mmap np array from buffer using offset
            # not using np.memmap bc it cannot be explicitly closed -
            # risk of memory leak
            angles = np.frombuffer(mm, dtype=float, offset=128)

            angles = angles.reshape(arr_shape)
            num_points = angles.shape[0]
            
            # Select random points from mmapped array 
            point_idx_arr = np.random.randint(low=0, high=num_points, size=batch_size)

            # Normalize dihedral angles and copy training batch subset to new array
            data = angles[point_idx_arr].copy() #/180.
            
            # Close mmap and clear memory
            del angles
            mm.close()
            gc.collect()
            
            return torch.from_numpy(data).reshape(1,5,5)

    def __getitem__(self, idx):     
        return self.get_data_point(), 0

if __name__=='__main__':
    myAla25 = Ala25Dataset()