import torch
import torch.nn as nn
from torch.utils.data import Subset 
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Grayscale
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import numpy as np
from skimage import color
from .color_models import rgb2lab, lab2rgb


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
from torchvision.datasets import ImageFolder

LAYER_DIM = {
        0: [1536, 4],
        1: [1536, 8],
        2: [768, 16],
        3: [768, 32],
        4: [384, 64],
        4: [192, 128],
        }


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_inf_batch(loader):
    while True:
        for x in loader:
            yield x


def copy_buff(m_from: nn.Module, m_to: nn.Module):
    for (k1, v1), (k2, v2) in zip(m_from.named_buffers(), m_to.named_buffers()):
        assert k1 == k2
        v2.copy_(v1)


def extract(dataset, target_ids):
    '''
    extract data element based on class index
    '''
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in target_ids:
            indices.append(i)
    return Subset(dataset, indices)


def prepare_dataset(
        path_train,
        path_valid,
        index_target,
        prep=transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])):


    dataset = ImageFolder(path_train, transform=prep)
    dataset = extract(dataset, index_target)

    dataset_val = ImageFolder(path_valid, transform=prep)
    dataset_val = extract(dataset_val, index_target)
    return dataset, dataset_val


def extract_sample(dataset, size_batch, num_iter, is_shuffle, pin_memory=True):
    dataloader = DataLoader(dataset, batch_size=size_batch,
            shuffle=is_shuffle, num_workers=4, pin_memory=pin_memory,
            drop_last=True)
    xs = []
    xgs = []
    cs = []
    for i, (x, c) in enumerate(dataloader):
        if i >= num_iter:
            break
        xg = transforms.Grayscale()(x)
        xs.append(x), cs.append(c), xgs.append(xg)
    return {'xs': xs, 'cs': cs, 'xs_gray': xgs}


def lab_fusion(x_l, x_ab):
    labs = []
    for img_gt, img_hat in zip(x_l, x_ab):

        img_gt = img_gt.permute(1, 2, 0)
        img_hat = img_hat.permute(1, 2, 0)

        img_gt = color.rgb2lab(img_gt)
        img_hat = color.rgb2lab(img_hat)
        
        l = img_gt[:, :, :1]
        ab = img_hat[:, :, 1:]

        img_fusion = np.concatenate((l, ab), axis=-1)
        img_fusion = color.lab2rgb(img_fusion)
        img_fusion = torch.from_numpy(img_fusion)
        img_fusion = img_fusion.permute(2, 0, 1)
        labs.append(img_fusion)
    labs = torch.stack(labs)
     
    return labs


def color_enhacne_blend(x, factor):
    x_g = Grayscale(3)(x)
    out = x_g * (1.0 - factor) + x * factor
    out[out < 0] = 0
    out[out > 1] = 1
    return out


def color_enhacne_abgc(x, factor):
    lab = rgb2lab(x)

    ab = lab[..., 1:3, :, :]
    ab /= 110

    ab[ab > 0] = torch.pow(factor, 1 / ab[ab > 0])
    ab[ab < 0] = -torch.pow(factor, 1 / ab[ab < 0].abs())

    ab *= 110
    lab[..., 1:3, :, :] = ab
    rgb = lab2rgb(lab)

    return rgb


def make_grid_multi(xs, nrow=4):
    return make_grid(torch.cat(xs, dim=0), nrow=nrow)


def read_preset(db_root_dir, existing_pids_in_mode, path_keys):
    presets = {}
    # keys_emb = np.load("/data/manho/lc_data/keys.npy")
    keys_emb = np.load(path_keys)
    for pid in existing_pids_in_mode:
        with open(osp.join(db_root_dir, 'norm_presets', pid + '.json'), 'r') as json_file:
            presets[pid] = json.load(json_file)
        # local_keys_emb = sorted(list(presets[pid].keys()))
        # assert not False in (local_keys_emb == keys_emb)
        presets[pid] = np.array([presets[pid][k] for k in keys_emb])
    return presets

class PhotoSet(Dataset):
    def __init__(self, db_root_dir, mode="train-400", random_diff=0.5, transform=None, path_keys=None):

        print('Initializing dataset ...')
        self.db_root_dir = db_root_dir
        self.p = random_diff
        self.transform = transform
        self.mode = mode
        self.path_keys = path_keys
        random.seed(1024)
        
        # Initialize the per sequence images for online training
        self.names, self.dirs, self.class_idx_dict = self.init_photoset(self.db_root_dir, mode, self.path_keys)
        self.max_idx = len(self.names) - 1
        print('Data Root: {}\n# Original Images: {}\n# Images:{}\n# Presets:{}'.format(self.db_root_dir, self.max_idx+1, len(self.dirs), len(self.dirs) / (self.max_idx+1)))
        # print(self.names[:10])

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        _dir = self.dirs[idx]
        samples, pairs, cls_id = self.read_data(_dir)
        # samples['gth_preset'] = torch.from_numpy(gth_preset).float()
        # samples['pairs'] = pairs
        if self.transform is not None:
            # samples = self.transform(samples)
            samples['reference'] = self.transform(samples['reference'])
            samples['positive_reference'] = self.transform(samples['positive_reference'])
            samples['img'] = self.transform(samples['img'])
            samples['gth_img'] = self.transform(samples['gth_img'])
        
        
        # samples['gth_preset'] = torch.from_numpy(gth_preset).float()
        samples['pairs'] = pairs
        samples['class_idx'] = cls_id
        return samples

    def read_data(self, _dir):
        reference = Image.open(_dir)
        basename = osp.basename(_dir)
        preset_id = _dir.split('/')[-3]
        class_id = _dir.split('/')[-2]

        if self.mode in ["train-400"]:
            
            img_name = self.names[random.randint(0,self.max_idx)]

        elif self.mode in ["val", "test"]:
            img_name = "0855.jpg" # self.names[0]
        else:
            raise "Unknown mode %s" % self.mode

        img = Image.open(osp.join(self.db_root_dir, self.mode, '0', class_id, basename))
        gth_img = reference

        base_idx = self.names.index(basename)
        pos_ref_idx = random.randint(0,self.max_idx)

        if pos_ref_idx == base_idx:
            pos_ref_idx = 0 if pos_ref_idx == self.max_idx else pos_ref_idx + 1

        positive_ref_dir = list(filter(lambda x: x.split('/')[-3] == preset_id
                                  ,list(filter(lambda x: osp.basename(x) == self.names[pos_ref_idx], self.dirs))))[0]
        positive_ref = Image.open(positive_ref_dir)

        # gth_preset = self.presets[preset_id]
        pairs = [img_name, basename, preset_id]

        
        # return {
        #     'reference': np.array(reference),
        #     'positive_reference': np.array(positive_ref),
        #     'img':  np.array(img),
        #     'gth_img': np.array(gth_img)
        # }, gth_preset, pairs, self.class_idx_dict[class_id]
    
        return {
            'reference': np.array(reference),
            'positive_reference': np.array(positive_ref),
            'img':  np.array(img),
            'gth_img': np.array(gth_img)
        }, pairs, self.class_idx_dict[class_id]

    @staticmethod
    def init_photoset(db_root_dir, mode, path_keys):
        names = [osp.basename(k) for k in glob.glob(osp.join(db_root_dir, mode, '0', '*', '*.jpg'))]
        # print(names)
        dirs = glob.glob(osp.join(db_root_dir, mode, '*', '*', '*.jpg'))
        # print(dirs)
        # existing_pids_in_mode = [osp.basename(k) for k in glob.glob(osp.join(db_root_dir, mode, '*'))]
        # print(existing_pids_in_mode)
        
        class_idx_dict = ImageFolder(osp.join(db_root_dir, mode, '0')).class_to_idx
          

        # presets = read_preset(db_root_dir, existing_pids_in_mode, path_keys)
        # presets = None
        return names, dirs, class_idx_dict