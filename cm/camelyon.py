"""
# --- CM-Camelyon --- #

DataLoader and load_data function to perform consistency model training on the Camelyon17-WILDS dataset
Based on https://github.com/openai/consistency_models/blob/main/cm/image_datasets.py
"""
import numpy as np
import torchvision.transforms as T
from wilds import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch

from .modified_split import load_split_filenames
from . import logger

hospital_3_4_train_split_size = 84253
hospital_3_test_split_size = 32706

HOSPITAL_4_NUM_PATCHES = 132052
HOSPITAL_3_NUM_PATCHES = 116959

class CamelyonDataLoader:

    def __init__(self, dl):
        self.dl = dl

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        return CamelyonDataLoaderIter(self.dl)

class CamelyonDataLoaderIter:

    def __init__(self, dl):
        self.dli = iter(dl)

    def __next__(self):
        x, y, metadata = next(self.dli)
        x = x / 127.5 - 1 # [-1,1]
        assert x.shape[1:] == (3,96,96), f"{x.shape}"
        return (x, {})

def get_data_loader(target_domain=True, data_dir=None, batch_size=16, split_file=None):
    
    dataset = get_dataset(dataset="camelyon17", root_dir=data_dir)

    train_data = dataset.get_subset("train", transform=T.PILToTensor()) # [0,255] (C,H,W) = (3,96,96)

    hospital = 4
    if not target_domain:
        hospital = 3

    if split_file == None:
        # split_mask is a numpy array, True = part of training split
        split_mask = dataset.split_array == dataset.split_dict["train"]
        
        # center_mask is a Pytorch tensor turned numpy array, True = part of hospital
        center_mask = dataset.metadata_array[:,0] == hospital
        center_mask = center_mask.numpy()

        split_center_mask = np.logical_and(split_mask, center_mask)
        mask_idx = np.where(split_center_mask)[0]

        train_data.indices = mask_idx

        if target_domain:
            assert len(train_data) == HOSPITAL_4_NUM_PATCHES
        else:
            assert len(train_data) == HOSPITAL_3_NUM_PATCHES
    else:
        logger.info(f"using filenames from {split_file} to create a modified split")
        path_list = load_split_filenames(split_file)

        mask_idx = []
        for path in path_list:
            dataset_idx = train_data.dataset._input_array.index(path)
            mask_idx.append(dataset_idx)
        
        train_data.indices = mask_idx

        if split_file != None and target_domain:
            assert len(train_data) == hospital_3_4_train_split_size, f"len(train_data) == {len(train_data)}"
        elif split_file != None and (not target_domain):
            assert len(train_data) == hospital_3_test_split_size, f"len(train_data) == {len(train_data)}"
            assert hospital == 3, f"hospital == {hospital}"

    dl = DataLoader(train_data, shuffle=target_domain, batch_size=batch_size, num_workers=1, drop_last=False) # image_datasets.py has drop_last=True
        
    return dl

def load_data(target_domain=True, data_dir=None, batch_size=16, split_file=None):

    dl = get_data_loader(target_domain, data_dir, batch_size, split_file)
    cdl = CamelyonDataLoader(dl)

    return cdl
