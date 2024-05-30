import os
from os.path import join, exists
import numpy as np
from train import Colorizer
import torch
import pickle
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, Resize, Compose, ToTensor
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from utils.common_utils import set_seed, rgb2lab, lab2rgb
from math import ceil

from PIL import Image

import json
import timm


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)

    # I/O
    parser.add_argument('--path_config', default='/kaggle/working/pre_trained/config.pickle')
    parser.add_argument('--path_ckpt_g', default='/kaggle/working/pre_trained/G_ema_256.pth')
    parser.add_argument('--path_ckpt', default='/kaggle/working/ckpts/unknown')
    parser.add_argument('--path_output', default='./results')
    parser.add_argument('--path_imgnet_val', default='/content/sub-train/train/0')
    parser.add_argument('--ref', default='')
    parser.add_argument('--use_ref_image', action='store_true')
    parser.add_argument('--use_classifier', action='store_true')
    parser.add_argument('--classifier_path', default='')

    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--no_upsample', action='store_true')
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dim_f', type=int, default=16)

    parser.add_argument('--type_resize', type=str, default='powerof',
            choices=['absolute', 'original', 'square', 'patch', 'powerof'])
    parser.add_argument('--num_power', type=int, default=4)
    parser.add_argument('--size_target', type=int, default=256)
    parser.add_argument('--iter_max', type=int, default=50000)

    return parser.parse_args()

def mapping(preset_ids):
    res = []
    dict = {0: 0, 18: 1, 34: 2, 89: 3}
    for preset_id in preset_ids:
        res.append(dict[preset_id])
    return res

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
    size_target = 256

    if args.seed >= 0:
        set_seed(args.seed)

    print('Target Epoch is %03d' % args.epoch)

    path_eg = join(args.path_ckpt, 'EG_%03d.ckpt' % args.epoch)
    path_eg_ema = join(args.path_ckpt, 'EG_EMA_%03d.ckpt' % args.epoch)
    path_args = join(args.path_ckpt, 'args.pkl')

    path_ref = args.ref

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


    grays = ImageFolder(args.path_imgnet_val,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((256,256)),
                            transforms.Grayscale()]))
    
    if (args.use_classifier):
        classifier = timm.create_model(
            "tf_efficientnet_l2_ns_475",
            pretrained=True,
            num_classes=1000
            ).to(dev)
        classifier.eval()


    # ref = Image.open(path_ref)
    ref = path_ref
    # custom_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(256)
    # ])
    # ref = custom_transform(ref)

    EG = Colorizer(config, 
                   args.path_ckpt_g,
                   args_loaded.norm_type,
                   id_mid_layer=args_loaded.num_layer,
                   activation=args_loaded.activation, 
                   use_attention=args_loaded.use_attention,
                   # use_res=not args_loaded.no_res,
                   dim_f=args.dim_f)
    EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
    EG_ema = ExponentialMovingAverage(EG.parameters(), decay=0.99)
    EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

    EG.eval()
    EG.float()
    EG.to(dev)

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
            unit = ceil((length_long * (args.size_target / length_sort)) 
                                        / (2 ** args.num_power))
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
    
    if args.use_ema:
        print('Use EMA')
        EG_ema.copy_to()

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    for i, (x, c) in enumerate(tqdm(grays)):
        if i >= args.iter_max:
            break

        size_original = x.shape[1:]
        x = x.unsqueeze(0)
        x = x.to(dev)
        
        if (args.use_classifier):
            x_cls = x.repeat(1, 3, 1, 1)
            x_cls = Resize((475, 475))(x_cls)
            c = classifier(x_cls)
            cs = torch.topk(c, 10)[1].reshape(-1)
            c = mapping_class(cs, 
                              args.classifier_path + "/original.json", 
                              args.classifier_path + "/new_class.json")
            
        c = torch.LongTensor([c])
        c = c.to(dev)

        preset_id = [eval(ref)]
        preset_id = torch.LongTensor(preset_id)

        z = preset_id
        z = z.to(dev)
        # print(z.shape)
        x_down = resizer(x)

        with torch.no_grad():
            output = EG(x_down, c, z)
            output = output.add(1).div(2)

        x = x.squeeze(0).cpu()
        x_down = x_down.squeeze(0).cpu()
        output = output.squeeze(0)
        output = output.detach().cpu()

        if args.no_upsample:
            output = Resize(x_down.shape[-2:])(output)
            lab_fusion = fusion(x_down, output)
        else:
            output = Resize(size_original)(output)
            lab_fusion = fusion(x, output)

        if args.use_rgb:
            im = ToPILImage()(output)
        else:
            im = ToPILImage()(lab_fusion)
        im.save('%s/%05d.jpg' % (args.path_output, i))


def fusion(img_gray, img_rgb):
    img_gray *= 100
    ab = rgb2lab(img_rgb)[..., 1:, :, :]
    lab = torch.cat([img_gray, ab], dim=0)
    rgb = lab2rgb(lab)
    return rgb 


if __name__ == '__main__':
    args = parse()
    main(args)
