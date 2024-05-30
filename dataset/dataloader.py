import os
import os.path as osp
import numpy as np
from PIL import Image
import glob
import argparse
import json
import copy
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def read_preset(db_root_dir, existing_pids_in_mode):
    presets = {}
    keys_emb = np.load("/data/manho/lc_data/keys.npy")
    for pid in existing_pids_in_mode:
        with open(osp.join(db_root_dir, 'norm_presets', pid + '.json'), 'r') as json_file:
            presets[pid] = json.load(json_file)
        local_keys_emb = sorted(list(presets[pid].keys()))
        assert not False in (local_keys_emb == keys_emb)
        presets[pid] = np.array([presets[pid][k] for k in local_keys_emb])
    return presets

class PhotoSet(Dataset):
    def __init__(self, db_root_dir, mode="train", random_diff=0.5, transform=None):

        print('Initializing dataset ...')
        self.db_root_dir = db_root_dir
        self.p = random_diff
        self.transform = transform
        self.mode = mode
        random.seed(1024)
        
        # Initialize the per sequence images for online training
        self.names, self.dirs, self.presets = self.init_photoset(self.db_root_dir, mode)
        self.max_idx = len(self.names) - 1
        print('Data Root: {}\n# Original Images: {}\n# Images:{}\n# Presets:{}'.format(self.db_root_dir, self.max_idx+1, len(self.dirs), len(self.presets)))
        # print(self.names[:10])

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        _dir = self.dirs[idx]
        samples, gth_preset, pairs = self.read_data(_dir)

        if self.transform is not None:
            samples = self.transform(samples)
        samples['gth_preset'] = torch.from_numpy(gth_preset).float()
        samples['pairs'] = pairs
        return samples

    def read_data(self, _dir):
        reference = Image.open(_dir)
        basename = osp.basename(_dir)
        preset_id = _dir.split('/')[-2]
        if self.mode in ["train"]:

            img_name = self.names[random.randint(0,self.max_idx)] 
        elif self.mode in ["val", "test"]:
            img_name = "0855.jpg" # self.names[0]
        else:
            raise "Unknown mode %s" % self.mode

        img = Image.open(osp.join(self.db_root_dir, self.mode, '0', img_name))
        gth_img = Image.open(osp.join(self.db_root_dir, self.mode, preset_id, img_name))

        base_idx = self.names.index(basename)
        pos_ref_idx = random.randint(0,self.max_idx) 

        if pos_ref_idx == base_idx:
            pos_ref_idx = 0 if pos_ref_idx == self.max_idx else pos_ref_idx + 1

        positive_ref = Image.open(osp.join(self.db_root_dir, self.mode, preset_id, self.names[pos_ref_idx]))

        gth_preset = self.presets[preset_id]
        pairs = [img_name, basename, preset_id] 

        
        return {
            'reference': np.array(reference),
            'positive_reference': np.array(positive_ref),
            'img':  np.array(img),
            'gth_img': np.array(gth_img)
        }, gth_preset, pairs

    @staticmethod
    def init_photoset(db_root_dir, mode):
        names = [osp.basename(k) for k in glob.glob(osp.join(db_root_dir, mode, '0', '*.jpg'))]
        dirs = glob.glob(osp.join(db_root_dir, mode, '*', '*.jpg'))
        existing_pids_in_mode = [osp.basename(k) for k in glob.glob(osp.join(db_root_dir, mode, '*'))]
        presets = read_preset(db_root_dir, existing_pids_in_mode)
        return names, dirs, presets

def test_db():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root_dir", type=str, default="/data/manho/lc_data/lc_dataset", help='path to dataset folder')
    args = parser.parse_args()

    # import dataset.dataloader as db
    import custom_transform as tr
    composed_transforms = transforms.Compose([
        tr.RandomCrop(cropsize=(512,512), basesize=(1280,1920)),
        # tr.RandomHorizontalFlip(),
        # tr.RandomVerticalFlip(),
        # tr.RandomRotation(degrees=[0,90,180,270], size=args.crop_size),
        tr.ToTensor(),
        tr.TensorRandomFlip(),
        tr.TensorRandomRotation(),
        ])
    train_dataset = PhotoSet(args.db_root_dir, transform=composed_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    for i, data_sample in enumerate(train_loader):
        samples, gth_preset = data_sample
        print(gth_preset)
        continue
    pass
