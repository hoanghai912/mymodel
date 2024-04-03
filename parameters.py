import argparse

def get_params():
    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root_dir", type=str, default="/data/manho/lc_data/lc_large_dataset/", help='path to dataset folder containing train-test-validation folders')
    parser.add_argument("--checkpoint_dir", type=str, default="/data/manho/source-codes/lc_data/models/", help='path to folder for saving checkpoints')
    parser.add_argument("--board_dir", type=str, default="/data/manho/source-codes/lc_data/board_logs/", help='path to folder for saving checkpoints')
    parser.add_argument("--val_out", type=str, default="/data/manho/lc_data/val_out/", help='path to folder for saving validated images')
    parser.add_argument("--model_name", type=str, default='lc_', help='model name for output')
    parser.add_argument("--g_ckpt", type=str, default=None, help='path of checkpoint for G pretrained model')
    parser.add_argument("--d_ckpt", type=str, default=None, help='path of checkpoint for D pretrained model')
    parser.add_argument("--save_interval", type=int, default=50, help='save_interval')
    parser.add_argument("--old_ckpts", type=int, default=[], help='keep track')
    parser.add_argument("--trainer_type", type=str, default='lpips', choices=['lpips', 'gan', 'lpipsv2', 'ganv2', 'lpipsv3'], help='GAN concept or simple way with lpips')

    ## Train details
    parser.add_argument("--epochs", type=int, default=2000, help='number of epochs to train. Default: 200.')
    parser.add_argument("--train_batch_size", type=int, default=8, help='batch size for training. Default: 8.')
    parser.add_argument("--val_batch_size", type=int, default=8, help='batch size for validation. Default: 8.')
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--iters_interval', type=int, default=5000)

    ## Augmentation
    parser.add_argument("--crop_size", type=int, default=352, help='crop size < scale size')
    parser.add_argument("--random_diff", type=float, default=0.0, help='Random to have the original samples')

    ## Main Network
    parser.add_argument("--g_net", type=str, default="netv2", help='net')
    parser.add_argument("--g_depth", type=int, default=5, help='depth of en/decoder')
    parser.add_argument("--g_in_channels", type=list, default=[6,3], help='inchannels')
    parser.add_argument("--g_out_channels", type=list, default=[69,3], help='outchannels')
    parser.add_argument("--g_upsampler", type=str, default="up_blurbilinear", help='')
    parser.add_argument("--g_downsampler", type=str, default="down_blurmax", help='')
    parser.add_argument("--g_norm", type=str, default="evo", help='norm')

    ## Discriminator
    parser.add_argument("--d_net", type=str, default="d_sagan", help='net')
    parser.add_argument("--d_in_channels", type=list, default=9, help='inchannels')
    parser.add_argument("--d_nchannels", type=int, default=64, help='discriminator channels depth')
    parser.add_argument("--d_norm", type=str, default="spectral", help='norm')

    ## Optimizer
    parser.add_argument("--optim", type=str, default='adam', help='radam, adam')
    parser.add_argument("--g_lr", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
    parser.add_argument("--d_lr", type=float, default=0.0004, help='set initial learning rate. Default: 0.0004.')
    parser.add_argument("--beta1", type=float, default=0.9, help='set initial learning rate. Default: 0.9.')
    parser.add_argument("--beta2", type=float, default=0.999, help='set initial learning rate. Default: 0.999.')
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument("--p_weight", type=float, default=0.01, help='learning preset weight')
    parser.add_argument("--s_weight", type=float, default=1, help='simulating lightroom weight')
    parser.add_argument("--g_weight", type=float, default=1, help='generator weight')
    parser.add_argument("--pw_weight", type=float, default=1, help='generator weight')
    parser.add_argument('--gp_weight', type=float, default=10)
    parser.add_argument('--lpips_weight', type=float, default=0.5)
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'resume', 'finetune', 'test'], help='train, resume, finetune, test')

    ## Preset Handler
    parser.add_argument('--base_preset', type=str, default="./data/base_presets.json")

    return parser.parse_args()
