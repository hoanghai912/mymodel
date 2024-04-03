import json
import numpy as np
import copy
import torch
# from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import lpips_models

class PresetHandler:
    def __init__(self, base_p_dir='./data/base_presets.json'):
        base_settings = self.json_load(base_p_dir)
        self.p_default = base_settings['p_default']
        self.keys = base_settings['keys']
        self.max_bound_np = np.array([base_settings['max'][k] for k in self.keys])
        self.min_bound_np = np.array([base_settings['min'][k] for k in self.keys])

    @staticmethod
    def json_load(_dir):
        with open(_dir) as json_file:
            content = json.load(json_file)
        return content

    def unnorm_preset(self, p_np):
        out = (p_np + 1)/2*(self.max_bound_np-self.min_bound_np) + self.min_bound_np
        return out.round().astype(np.int16).tolist()

    @staticmethod
    def int2bool(_num):
        return True if _num > 0.5 else False

    def save_numpy_preset(self, out_dir, p_np):
        p_list = self.unnorm_preset(p_np)
        p_base = copy.deepcopy(self.p_default) # be careful of copying address but values
        p_dict = {k:p_list[i] for i,k in enumerate(self.keys)}
        for k in p_dict:
            p_dict[k] = self.int2bool(p_dict[k]) if type(p_base[k]) == bool else p_dict[k]

        p_base.update(p_dict)
        with open(out_dir, 'w') as outfile:
            json.dump(p_base, outfile)

def loss_fn_d(D, c, real, fake):
    real = (real - 0.5) * 2
    critic_real, _ = D(real, c)
    critic_fake, _ = D(fake, c)
    d_loss_real, d_loss_fake = loss_hinge_dis(critic_fake, critic_real)
    loss_d = (d_loss_real + d_loss_fake) / 2  
    return loss_d


def loss_fn_g(D, vgg_per, x, c, args, fake):
    loss_dic = {}
    loss = 0
    if args.loss_adv:
        critic, _ = D(fake, c)
        loss_g = loss_hinge_gen(critic) * args.coef_adv
        loss += loss_g 
        loss_dic['loss_g'] = loss_g 

    fake = fake.add(1).div(2)
    if args.loss_mse:
        loss_mse = args.coef_mse * nn.MSELoss()(x, fake)
        loss += loss_mse
        loss_dic['mse'] = loss_mse
    if args.loss_lpips:
        loss_lpips = args.coef_lpips * vgg_per(x, fake)
        loss += loss_lpips
        loss_dic['lpips'] = loss_lpips

    return loss, loss_dic


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


class PerceptLoss(object):

    def __init__(self):
        pass

    def __call__(self, LossNet, fake_img, real_img):
        with torch.no_grad():
            real_feature = LossNet(real_img.detach())
        fake_feature = LossNet(fake_img)
        perceptual_penalty = F.mse_loss(fake_feature, real_feature)
        return perceptual_penalty

    def set_ftr_num(self, ftr_num):
        pass


class DiscriminatorLoss(object):

    def __init__(self, ftr_num=4, data_parallel=False):
        self.data_parallel = data_parallel
        self.ftr_num = ftr_num

    def __call__(self, D, fake_img, real_img):
        if self.data_parallel:
            with torch.no_grad():
                d, real_feature = nn.parallel.data_parallel(
                    D, real_img.detach())
            d, fake_feature = nn.parallel.data_parallel(D, fake_img)
        else:
            with torch.no_grad():
                d, real_feature = D(real_img.detach())
            d, fake_feature = D(fake_img)
        D_penalty = 0
        for i in range(self.ftr_num):
            f_id = -i - 1
            D_penalty = D_penalty + F.l1_loss(fake_feature[f_id],
                                              real_feature[f_id])
        return D_penalty

    def set_ftr_num(self, ftr_num):
        self.ftr_num = ftr_num


# Preset Loss
def criterionL1():
    return nn.L1Loss().cuda()

def criterionMSE():
    return nn.MSELoss().cuda()

def criterionLPIPS():
    return lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)