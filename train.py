import os
from os.path import join, exists
import models
from models import Colorizer, VGG16Perceptual

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader 
from torch.utils.data.distributed import DistributedSampler

import pickle
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.losses import loss_fn_d, loss_fn_g, loss_encoder_t
from utils.common_utils import (extract_sample, set_seed,
        make_grid_multi, prepare_dataset, PhotoSet)
from utils.logger import (make_log_scalar, make_log_img, 
                          make_log_ckpt, load_for_retrain,
                          load_for_retrain_EMA)
from utils.common_utils import color_enhacne_blend
import utils

from torch_ema import ExponentialMovingAverage
from functools import partial



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='unknown')
    parser.add_argument('--detail', default='unknown')

    # Mode
    parser.add_argument('--norm_type', default='adabatch', 
            choices=['instance', 'batch', 'layer', 'adain', 'adabatch', 'id'])
    parser.add_argument('--activation', default='relu', 
            choices=['relu', 'lrelu', 'sigmoid'])
    parser.add_argument('--weight_init', default='ortho', 
            choices=['xavier', 'N02', 'ortho', ''])

    # IO
    parser.add_argument('--path_log', default='runs')
    parser.add_argument('--path_ckpts', default='ckpts')
    parser.add_argument('--path_config', default='/kaggle/working/pre_trained/config.pickle')
    parser.add_argument('--path_vgg', default='/kaggle/working/pre_trained/vgg16.pickle')
    parser.add_argument('--path_ckpt_g', default='/kaggle/working/pre_trained/G_ema_256.pth')
    parser.add_argument('--path_ckpt_d', default='/kaggle/working/pre_trained/D_256.pth')
    parser.add_argument('--path_imgnet_train', default='/kaggle/working/sub-train')
    parser.add_argument('--path_imgnet_val', default='./imgnet/val')
    parser.add_argument('--path_keys', default='./keys.npy')

    parser.add_argument('--index_target', type=int, nargs='+', 
            default=list(range(1000)))
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--iter_sample', type=int, default=3)

    # Encoder Traning
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--retrain_epoch', type=int)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--dim_f', type=int, default=16)
    parser.add_argument('--no_res', action='store_true')
    parser.add_argument('--no_cond_e', action='store_true')
    parser.add_argument('--interval_save_loss', default=20)
    parser.add_argument('--interval_save_train', default=150)
    parser.add_argument('--interval_save_test', default=2000)
    parser.add_argument('--interval_save_ckpt', default=4000)

    parser.add_argument('--finetune_g', default=True)
    parser.add_argument('--finetune_d', default=True)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--b1", type=float, default=0.0)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--lr_d", type=float, default=0.00003)
    parser.add_argument("--b1_d", type=float, default=0.0)
    parser.add_argument("--b2_d", type=float, default=0.999)
    parser.add_argument('--use_schedule', default=True)
    parser.add_argument('--schedule_decay', type=float, default=0.90)
    parser.add_argument('--schedule_type', type=str, default='mult',
            choices=['mult', 'linear'])

    # Verbose
    parser.add_argument('--print_config', default=False)

    # loader
    parser.add_argument('--no_pretrained_g', action='store_true')
    parser.add_argument('--no_pretrained_d', action='store_true')

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=True)
    parser.add_argument('--loss_adv', action='store_true', default=True)
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.2)
    parser.add_argument('--coef_adv', type=float, default=0.03)
    parser.add_argument('--vgg_target_layers', type=int, nargs='+',
                            default=[1, 2, 13, 20])

    # EMA
    parser.add_argument('--decay_ema_g', type=float, default=0.999)

    # Others
    parser.add_argument('--dim_z', type=int, default=119)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size_batch', type=int, default=60)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--use_enhance', action='store_true')
    parser.add_argument('--coef_enhance', type=float, default=1.5)
    parser.add_argument('--use_attention', action='store_true')

    # GPU
    parser.add_argument('--multi_gpu', default=True)

    return parser.parse_args()


def setup_dist(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def mapping(preset_ids):
    res = []
    dict = {0: 0, 18: 1, 34: 2, 89: 3}
    for preset_id in preset_ids:
        res.append(dict[preset_id])
    return res

def train(dev, world_size, config, args,
          dataset=None,
          sample_train=None,
          sample_valid=None,
          path_ckpts=None,
          path_log=None,
          ):

    is_main_dev = dev == 0 
    setup_dist(dev, world_size, args.port)
    if is_main_dev:
        # writer = SummaryWriter(path_log)
        pass

    # Setup model
    EG = Colorizer(config, 
                   args.path_ckpt_g, 
                   args.norm_type,
                   id_mid_layer=args.num_layer, 
                   activation=args.activation, 
                   fix_g=(not args.finetune_g),
                   load_g=(not args.no_pretrained_g),
                   init_e=args.weight_init,
                   use_attention=args.use_attention,
                   use_res=(not args.no_res),
                   dim_f=args.dim_f)
    EG.train()
    D = models.Discriminator(**config)
    D.train()


    if not args.no_pretrained_d:
        D.load_state_dict(torch.load(args.path_ckpt_d, map_location='cpu'),
                          strict=False)

    # Optimizer
    optimizer_g = optim.Adam([p for p in EG.parameters() if p.requires_grad],
            lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = optim.Adam(D.parameters(),
            lr=args.lr_d, betas=(args.b1_d, args.b2_d))


    # Schedular
    if args.use_schedule:
        if args.schedule_type == 'mult':
            schedule = lambda epoch: args.schedule_decay ** epoch
        elif args.schedule_type == 'linear':
            schedule = lambda epoch: (args.num_epoch - epoch) / args.num_epoch
        else:
            raise Exception('Invalid shedule type')
        scheduler_g = optim.lr_scheduler.LambdaLR(optimizer=optimizer_g,
                        lr_lambda=schedule)
        scheduler_d = optim.lr_scheduler.LambdaLR(optimizer=optimizer_d,
                        lr_lambda=schedule)


    # Retrain(opt)
    num_iter = 0
    epoch_start = 0
    if args.retrain:
        if args.retrain_epoch is None:
            raise Exception('retrain_epoch is required')
        epoch_start = args.retrain_epoch + 1
        num_iter = load_for_retrain(EG, D, 
                                    optimizer_g, optimizer_d,
                                    scheduler_g, scheduler_d, 
                                    args.retrain_epoch, path_ckpts, 
                                    'cpu')
        dist.barrier()

    # Set Device 
    EG = EG.to(dev)
    D = D.to(dev)
    vgg_per = VGG16Perceptual(args.path_vgg, args.vgg_target_layers).to(dev)
    utils.optimizer_to(optimizer_g, 'cuda:%d' % dev)
    utils.optimizer_to(optimizer_d, 'cuda:%d' % dev)

    # EMA
    ema_g = ExponentialMovingAverage(EG.parameters(), decay=args.decay_ema_g)
    if args.retrain:
        load_for_retrain_EMA(ema_g, args.retrain_epoch, path_ckpts, 'cpu')

    # DDP
    torch.cuda.set_device(dev)
    torch.cuda.empty_cache()

    EG = DDP(EG, device_ids=[dev], 
             find_unused_parameters=True)
    D = DDP(D, device_ids=[dev], 
            find_unused_parameters=False)
    vgg_per = DDP(vgg_per, device_ids=[dev], 
                  find_unused_parameters=True)

    # Datasets
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, 
            shuffle=True if sampler is None else False, 
            sampler=sampler, pin_memory=True,
            num_workers=args.num_worker, drop_last=True)

    color_enhance = partial(color_enhacne_blend, factor=args.coef_enhance)

    # embed_layer = torch.nn.Embedding(100, args.dim_z, _freeze=True)
    # torch.save(embed_layer.weight.data, "embed.pt")

    # AMP
    scaler = GradScaler()

    for epoch in range(epoch_start, args.num_epoch):
        sampler.set_epoch(epoch)
        tbar = tqdm(dataloader)
        tbar.set_description('epoch: %03d' % epoch)
        loss_generator = loss_dis_train = loss_encoder_t_train = None
        test_output = None
        test_preset_id = None
        for i, data_sample in enumerate(tbar):
            EG.train()

            x = data_sample['img']
            c = data_sample['class_idx']
            r = data_sample['reference']
            x, c, r = x.to(dev), c.to(dev), r.to(dev)
            x_gray = transforms.Grayscale()(x)

            real_images = data_sample['gth_img'].to(dev)
            # gth_preset = data_sample['gth_preset'].to(dev)
            preset_id = [eval(x) for x in data_sample['pairs'][-1]]
            preset_id = mapping(preset_id)
            # print("preset_id", preset_id)
            preset_id_embedding = torch.LongTensor(preset_id)
            positive_reference = data_sample['positive_reference'].to(dev)

            #swap reference <-> positive_reference

            # _tmp = r
            # r = positive_reference
            # positive_reference = _tmp

            # Sample z
            # z = torch.zeros((args.size_batch, args.dim_z)).to(dev)
            # z.normal_(mean=0, std=0.8)

            # z = torch.nn.Embedding(400, args.dim_z)(preset_id_embedding)
            z = preset_id_embedding
            z = z.to(dev)

            # Generate fake image
            with autocast():
                fake = EG(x_gray, c, z)
                

            # DISCRIMINATOR 
            x_real = real_images
            if args.use_enhance:
                x_real =  color_enhance(real_images)

            optimizer_d.zero_grad()
            with autocast():
                loss_d = loss_fn_d(D=D,
                                   c=c,
                                   real=x_real,
                                   fake=fake.detach())

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # GENERATOR
            optimizer_g.zero_grad()
            with autocast():
                loss, loss_dic = loss_fn_g(D=D,
                                           vgg_per=vgg_per,
                                           x=x,
                                           c=c,
                                           args=args,
                                           fake=fake)
                
                
                # g_loss = loss

            # scaler.scale(loss).backward(retain_graph=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # EMA
            if is_main_dev:
                ema_g.update()

            loss_dic['loss_d'] = loss_d

            loss_generator = loss.item()
            loss_dis_train = loss_d.item()
            # if (num_iter % 1000 == 0):
            #     # loss_encoder_t_train = loss_encoderT.item()
            #     print("Loss_g =", loss_generator)
            #     print("Loss_discriminator =", loss_dis_train)

            test_output = fake[0].add(1).div(2).detach().cpu()
            test_gt = real_images[0].detach().cpu()
            test_ref = positive_reference[0].detach().cpu()
            test_preset_id = data_sample['pairs'][-1][0]
            test_x_gray = x_gray[0].detach().cpu()

            num_iter += 1
        
        print("Loss_g =", loss_generator)
        print("Loss_discriminator =", loss_dis_train)
        # print("Loss_encoder_t =", loss_encoder_t_train)
        # print("Loss EG =", loss_generator + loss_encoder_t_train)

        # Save Model
        if is_main_dev:
            make_log_ckpt(EG=EG.module,
                          D=D.module,
                          optim_g=optimizer_g,
                          optim_d=optimizer_d,
                          schedule_g=scheduler_g,
                          schedule_d=scheduler_d,
                          ema_g=ema_g,
                          num_iter=num_iter,
                          args=args, epoch=epoch, 
                          path_ckpts=path_ckpts, 
                          test_output=test_output,
                          test_gt=test_gt,
                          test_ref=test_ref,
                          test_preset_id=test_preset_id,
                          x_gray=test_x_gray)

        if args.use_schedule:
            scheduler_d.step()
            scheduler_g.step()


def main():
    args = parse_args()

    # Note Retrain
    if args.retrain:
        print("This is retrain work after EPOCH %03d" % args.retrain_epoch)

    # GPU OPTIONS
    num_gpu = torch.cuda.device_count()
    if num_gpu == 0:
        raise Exception('No available GPU')
    elif num_gpu == 1:
        print('Use single GPU')
    elif num_gpu > 1: 
        print('Use multi GPU: %02d EA' % num_gpu)
    else:
        raise Exception('Invalid GPU setting')

    # Load Configuratuion
    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)

    if args.print_config:
        for i in config:
            print(i, ':', config[i])

    if args.seed >= 0:
        set_seed(args.seed)

    # Make directory for checkpoints    
    if not exists(args.path_ckpts):
        os.mkdir(args.path_ckpts)
    path_ckpts = join(args.path_ckpts, args.task_name)
    if not exists(path_ckpts):
        os.mkdir(path_ckpts)
       
    # Save arguments
    with open(join(path_ckpts, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Logger
    path_log = join(args.path_log, args.task_name)


    composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])


    dataset = PhotoSet(args.path_imgnet_train, transform=composed_transforms, path_keys=args.path_keys)


    is_shuffle = True 
    args.size_batch = int(args.size_batch / num_gpu)

    sample_train = None
    sample_valid = None

    mp.spawn(train,
             args=(num_gpu, config, args, dataset, sample_train, 
                   sample_valid, path_ckpts, path_log),
             nprocs=num_gpu)

if __name__ == '__main__':
    main()
