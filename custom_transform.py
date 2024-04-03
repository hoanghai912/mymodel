import torch
import collections
import cv2
import random
import numpy as np
from torchvision.transforms import functional as F
from torchvision import transforms

class RandomCrop(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
        cropsize: (h,w)
    """
    def __init__(self, cropsize=(352,352)):
        self.cropsize = cropsize

    def __call__(self, sample):

        start_h, end_h, start_w, end_w, h, w = self.get_crop(sample['img'])
        # sample['reference'] = cv2.resize(sample['reference'], self.cropsize, interpolation=cv2.INTER_CUBIC)

        for key_id in ['img', 'gth_img', 'reference', 'positive_reference']: # ,  'negative_reference']:
            if 'reference' in key_id:
                start_h, end_h, start_w, end_w, h, w = self.get_crop(sample[key_id])
            tmp = sample[key_id]
            tmp = tmp[start_h:end_h, start_w:end_w]
            sample[key_id] = tmp

        return sample
    
    def get_crop(self, img_np):
        h,w,_ = img_np.shape

        start_h = random.randint(0, h - self.cropsize[0])
        start_w = random.randint(0, w - self.cropsize[1])
        end_h = start_h + self.cropsize[0]
        end_w = start_w + self.cropsize[1]
        return start_h, end_h, start_w, end_w, h, w

class ResizeImages(object):
    def __init__(self, size=(512,512)):
        self.size = size

    def __call__(self, sample):
        for key_id in sample.keys():
            # if key_id in ['fname', 'iname']:
            #     continue
            tmp = sample[key_id]
            sample[key_id] = cv2.resize(tmp, self.size, interpolation=cv2.INTER_CUBIC)

        return sample

class CenterCrop(object):
    def __init__(self, cropsize=(1280,1280), basesize=(1280,1920)):
        self.cropsize = cropsize
        self.basesize = basesize

    def __call__(self, sample):

        start_h = self.basesize[0]//2-(self.cropsize[0]//2)
        start_w = self.basesize[1]//2-(self.cropsize[1]//2)
        end_h = start_h + self.cropsize[0]
        end_w = start_w + self.cropsize[1]

        for key_id in sample.keys():
            # if key_id in ['fname', 'iname']:
            #     continue
            tmp = sample[key_id]
            tmp = tmp[start_h:end_h, start_w:end_w]

        return sample

class RandomHorizontalFlip(object):
    """ flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for key_id in sample.keys():
                # if key_id in ['fname', 'iname']:
                #     continue
                tmp = sample[key_id]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[key_id] = tmp

        if random.random() < 0.5:
            for key_id in sample.keys():
                # if key_id in ['fname', 'iname']:
                #     continue
                tmp = sample[key_id]
                tmp = cv2.flip(tmp, flipCode=0)
                sample[key_id] = tmp

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for key_id in sample.keys():
            if sample[key_id] is None:
                continue
            # if key_id in ['fname', 'iname']:
            #     continue
            tmp = sample[key_id]
            # print(np.max(tmp))
            # print(np.min(tmp))

            tmp = tmp / 255.0 # [0,1]
            tmp = (tmp - 0.5)/0.5 # [-1, 1]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            tmp = tmp.transpose((2, 0, 1))
            sample[key_id] = torch.from_numpy(tmp).float()

        return sample

class RandomRotation(object):

    def __init__(self, degrees=[0,90,180,270], size=(120,120)):
        '''
        size : (h, w)
        '''
        self.center = (size[0]/2, size[1]/2)
        self.scale = 1.0
        self.degrees = degrees
        self.size = size

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.choice(degrees)
        return angle

    def __call__(self, sample):
        angle = self.get_params(self.degrees)
        M = cv2.getRotationMatrix2D(self.center, angle, self.scale)
        warp_size = (self.size[1], self.size[0]) if angle in [90, 270] else self.size
        for key_id in sample:
            tmp = sample[key_id]
            _size = (warp_size[0], warp_size[1])
            tmp = cv2.warpAffine(tmp, M, _size) 
            sample[key_id] = tmp
        return sample

class TensorRandomRotation(object):
    def __init__(self, degrees=[0,90,180,270], size=(120,120)):
        '''
        size : (h, w)
        '''
        self.center = (size[0]/2, size[1]/2)
        self.scale = 1.0
        self.degrees = [0,90,180,270]
        self.size = size
        print(self.size)

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.choice(degrees)
        return angle

    def __call__(self, sample):
        angle = self.get_params(self.degrees)
        '''
        x90 = x.transpose(d1, d2).flip(d1)
        x180 = x.flip(d1).flip(d2)
        x270 = x.transpose(d1, d2).flip(d2)
        '''
        if angle == 90:
            for key_id in sample:
                tmp = sample[key_id]
                tmp = tmp.transpose(1, 2).flip(1)
                sample[key_id] = tmp
        elif angle == 180:
            for key_id in sample:
                tmp = sample[key_id]
                tmp = tmp.flip(1).flip(2)
                sample[key_id] = tmp
        else: # angle == 270:
            for key_id in sample:
                tmp = sample[key_id]
                tmp = tmp.transpose(1, 2).flip(2)
                sample[key_id] = tmp
        return sample

class TensorRandomFlip(object):
    def __call__(self, sample):

        if random.random() < 0.5:
            for key_id in sample.keys():
                tmp = sample[key_id]
                tmp = tmp.flip(1)
                sample[key_id] = tmp
        if random.random() < 0.5:
            for key_id in sample.keys():
                tmp = sample[key_id]
                tmp = tmp.flip(2)
                sample[key_id] = tmp

        return sample
