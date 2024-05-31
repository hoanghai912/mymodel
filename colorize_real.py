import os
from os import listdir
from os.path import join, exists
# from skimage.color import rgb2lab, lab2rgb
import numpy as np
from train import Colorizer
import torch
import pickle
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage, Grayscale, Resize, Compose
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from utils.common_utils import set_seed, rgb2lab, lab2rgb
from PIL import Image
import timm
from math import ceil
import json

import os.path as osp
import glob
import torch
import torch.nn as nn
from utils import *
from networks.network import get_model
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

class DeepPresetTest(object):
    def __init__(self, ckpt, ckpt_2=""):
        ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
        ckpt_2 = torch.load(ckpt_2, map_location=torch.device('cpu'))

        # Load model
        self.G = get_model(ckpt['opts'].g_net)(ckpt['opts'])

        num_features = self.G.llayer_2[0].in_features
        self.G.llayer_2 = nn.Sequential(
          nn.Linear(num_features, 69),
          nn.Tanh(),
          nn.Linear(69, 4),
          nn.LogSoftmax(dim=1)
        )
        
        self.G.load_state_dict(ckpt_2)

def predict_preset(load_model, path_ref):
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--ckpt", type=str, default="/content/dp_woPPL.pth.tar", help='Checkpoint path')
#   parser.add_argument("--ckpt_2", type=str, default="")
#   parser.add_argument("--image_path", type=str, default="")
#   args = parser.parse_args()
  
    # deep_preset = DeepPresetTest(ckpt, ckpt_2)
    deep_preset = load_model
        
    model = deep_preset.G

    model.eval()

    # print(model)
  

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((352, 352)),
        transforms.CenterCrop((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img_input = Image.open(path_ref)
    img_input = train_transform(img_input)
    img_input = img_input.unsqueeze(0)
    # print("img_input", img_input.shape)

    _, output, _ = model.stylize(img_input, img_input, None, preset_only=True)

    ret, prediction = torch.max(output.data, 1)

    return int(prediction[0])

MODEL2SIZE = {'resnet50d': 224,
              'tf_efficientnet_l2_ns_475': 475}

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)

    # I/O
    parser.add_argument('--path_config', default='./pretrained/config.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt', default='./ckpts/baseline_1000')
    parser.add_argument('--path_output', default='./results_real')
    parser.add_argument('--path_input', default='./resource/real_grays')
    parser.add_argument('--path_ref', default='')
    # parser.add_argument("--ckpt", type=str, default="/content/dp_woPPL.pth.tar", help='Checkpoint path')
    # parser.add_argument("--ckpt_2", type=str, default="")

    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--no_upsample', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dim_f', type=int, default=16)

    # Setting
    parser.add_argument('--type_resize', type=str, default='absolute',
            choices=['absolute', 'original', 'square', 'patch', 'powerof'])
    parser.add_argument('--num_power', type=int, default=4)
    parser.add_argument('--size_target', type=int, default=256)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--cls_model', type=str, default='tf_efficientnet_l2_ns_475')

    return parser.parse_args()


def mapping_class(input_class, origin_json_path, new_class_path):
    f= open(origin_json_path)
    data_origin = json.load(f)
    f= open(new_class_path)
    data_new = json.load(f)
    f.close()
    
    # input_class = str(input_class)

    mapping_1 = {}
    for label, content in data_origin.items():
        mapping_1[label] = content[0]
    mapping_2 = data_new

    for c in input_class:
      c = str(int(c))
      label = mapping_1[c]
      if label in mapping_2:
        return mapping_2[mapping_1[c]]
    
    return mapping_2[mapping_1[input_class]]


def main(args):

    if args.seed >= 0:
        set_seed(args.seed)

    print('Target Epoch is %03d' % args.epoch)

    path_eg = join(args.path_ckpt, 'EG_%03d.ckpt' % args.epoch)
    path_eg_ema = join(args.path_ckpt, 'EG_EMA_%03d.ckpt' % args.epoch)
    path_args = join(args.path_ckpt, 'args.pkl')

    if not exists(path_eg):
        raise FileNotFoundError(path_eg)
    if not exists(path_args):
        raise FileNotFoundError(path_args)

    # Load Configuratuion
    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)
    with open(path_args, 'rb') as f:
        args_loaded = pickle.load(f)

    dev = args.device

    # Load Colorizer
    EG = Colorizer(config, 
                   args.path_ckpt_g,
                   args_loaded.norm_type,
                   id_mid_layer=args_loaded.num_layer,
                   activation=args_loaded.activation, 
                   use_attention=args_loaded.use_attention,
                   dim_f=args.dim_f)
    EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
    EG_ema = ExponentialMovingAverage(EG.parameters(), decay=0.99)
    EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

    EG.eval()
    EG.float()
    EG.to(dev)

    deep_preset = DeepPresetTest(os.path.join(args.path_ckpt, "dp_woPPL.pth.tar"), 
                                  os.path.join(args.path_ckpt, "model_7.pt"))
    
    if args.use_ema:
        print('Use EMA')
        EG_ema.copy_to()

    # Load Classifier
    classifier = timm.create_model(
            args.cls_model,
            pretrained=True,
            num_classes=1000
            ).to(dev)
    classifier.eval()
    # size_cls = MODEL2SIZE[args.cls_model]

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    paths = [join(args.path_input, p) for p in listdir(args.path_input)]

    resizer = None
    if args.type_resize == 'absolute':
        resizer = Resize((args.size_target))
    elif args.type_resize == 'original':
        resizer = Compose([])
    elif args.type_resize == 'square':
        resizer = Resize((args.size_target, args.size_target))
    elif args.type_resize == 'powerof':
        assert args.size_target % (2 ** args.num_power) == 0

        def resizer(x):
            length_long = max(x.shape[-2:])
            length_sort = min(x.shape[-2:])
            unit = ceil((length_long * (args.size_target / length_sort)) / (2 ** args.num_power))
            long = unit * (2 ** args.num_power)

            if x.shape[-1] > x.shape[-2]:
                fn = Resize((args.size_target, long))
            else:
                fn = Resize((long, args.size_target))

            return fn(x)
    elif args.type_resize == 'patch':
        resizer = Resize((args.size_target))
    else:
        raise Exception('Invalid resize type')

    ref = predict_preset(deep_preset, 
                             args.path_ref)
    if (ref == 0): ref = "89"
    elif (ref == 1): ref = "0"
    elif (ref == 2): ref= "34"
    elif (ref == 3): ref = "18"
    # print('ref', ref)

    for path in tqdm(paths):
        
        im = Image.open(path)
        x = ToTensor()(im)
        if x.shape[0] != 1:
            x = Grayscale()(x)

        size = x.shape[1:]

        x = x.unsqueeze(0)
        x = x.to(dev)
        # z = torch.zeros((1, args_loaded.dim_z)).to(dev)
        # z.normal_(mean=0, std=0.8)

        # Classification
        x_cls = x.repeat(1, 3, 1, 1)
        x_cls = Resize((475, 475))(x_cls)
        c = classifier(x_cls)
        # cs = torch.topk(c, args.topk)[1].reshape(-1)
        cs = torch.topk(c, 10)[1].reshape(-1)
        c = mapping_class(cs, 
                            args.path_ckpt + "/original.json", 
                            args.path_ckpt + "/new_class.json")
        c = torch.LongTensor([c])
        c = c.to(dev)

        # ref = predict_preset(deep_preset, 
        #                      args.path_ref)
        # if (ref == 0): ref = "3"
        # elif (ref == 1): ref = "18"
        # elif (ref == 2): ref= "34"
        # elif (ref == 3): ref = "0"
        # print('ref', ref)

        preset_id = [eval(ref)]
        preset_id = torch.LongTensor(preset_id) 
        z = preset_id
        z = z.to(dev)
        # for c in cs:
        # c = torch.LongTensor([c]).to(dev)
        x_resize = resizer(x)

        if args.type_resize == 'patch':
            length = max(x_resize.shape[-2:])
            num_patch = ceil(length / args.size_target)
            direction = 'v' if x.shape[-1] < x.shape[-2] else 'h' 

            patchs = []
            for i in range(num_patch):
                patch =  torch.zeros((args.size_target, args.size_target))
                if i + 1 == num_patch:  # last
                    start = -args.size_target 
                    end = length 
                else:
                    start = i * args.size_target 
                    end = (i + 1) * args.size_target

                if direction == 'v':
                    patch = x_resize[..., start:end, :]
                elif direction == 'h':
                    patch = x_resize[..., :, start:end]
                else:
                    raise Exception('Invalid direction')
                patchs.append(patch)

            outputs = [EG(patch, c, z).add(1).div(2) for patch in patchs]
            cloth = torch.zeros((1, 3, x_resize.shape[-2],
                                        x_resize.shape[-1]))
            for i in range(num_patch):
                output = outputs[i]
                if i + 1 == num_patch:  # last
                    start = -args.size_target 
                    end = length 
                else:
                    start = i * args.size_target 
                    end = (i + 1) * args.size_target

                if direction == 'v':
                    cloth[..., start:end, :] = output
                elif direction == 'h':
                    cloth[..., :, start:end] = output
                else:
                    raise Exception('Invalid direction')

            output = cloth
            im = ToPILImage()(output.squeeze(0))
            im.show()
            raise NotImplementedError()

        with torch.no_grad():
            output = EG(x_resize, c, z)
            output = output.add(1).div(2)

        if args.no_upsample:
            size_output = x_resize.shape[-2:]
            x_rs = x_resize.squeeze(0).cpu()
        else:
            size_output = size
            x_rs = x.squeeze(0).cpu()

        output = transforms.Resize(size_output)(output)
        output = output.squeeze(0)
        output = output.detach().cpu()

        if args.use_rgb:
            x_img = output
        else:
            x_img = fusion(x_rs, output)
        im = ToPILImage()(x_img)

        name = path.split('/')[-1].split('.')[0]
        name = name + '.jpg'

        path_out = join(args.path_output, name)
        im.save(path_out)


def fusion(img_gray, img_rgb):
    img_gray *= 100
    ab = rgb2lab(img_rgb)[..., 1:, :, :]
    lab = torch.cat([img_gray, ab], dim=0)
    rgb = lab2rgb(lab)
    return rgb 

if __name__ == '__main__':
    args = parse()
    main(args)
