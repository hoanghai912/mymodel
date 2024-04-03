import torch
import torch.nn as nn
from networks.components import get_block, get_layer

def get_model(_name):
    return {
        "netv2": Netv2,
    }[_name]

class Netv2(nn.Module):
    def __init__(self, args):
        super(Netv2, self).__init__()

        print('Init Net')
        self.depth = args.g_depth
        norms = [args.g_norm, args.g_norm]

        ## Declare params
        _tmp = [2**(k)*32 for k in range(self.depth-1)]
        _tmp += [_tmp[-1]]
        pdims = [args.g_in_channels[0]] + _tmp
        sdims = [args.g_in_channels[1]] + _tmp
        pddims = [2048, 1024, args.g_out_channels[0]]
        sddims = [k*2 for k in _tmp[::-1]]
        sddims += [sddims[-1]]
        enc_kernels = [[5,3]] + [[3,3] for k in range(self.depth-1)]

        print("P_Dims: {}\nS_Dims: {}\nPD_Dims: {}\nSD_Dims: {}\nEnc_Kernels: {}".format(pdims, sdims, pddims, sddims, enc_kernels))

        ## Encoder
        for i in range(self.depth):
            setattr(self, 'pconv_{}'.format(i+1), get_layer('basic')(pdims[i],pdims[i+1],kernels=enc_kernels[i], subsampling=args.g_downsampler if i > 0 else 'none', norms=norms))

        for i in range(self.depth):
            setattr(self, 'sconv_{}'.format(i+1), get_layer('basic')(sdims[i],sdims[i+1],kernels=enc_kernels[i], subsampling=args.g_downsampler if i > 0 else 'none', norms=norms))

        ## Linear Decoder
        img_size = args.crop_size[0]//2**(self.depth-1)
        self.linear_in = img_size*img_size*pdims[-1]

        self.pdfc1 = nn.Sequential(
            nn.Linear(self.linear_in, pddims[0]),
            nn.ReLU(),
            nn.Linear(pddims[0], pddims[1]),
            nn.ReLU()
        )

        self.pdfc2 = nn.Sequential(
            nn.Linear(pddims[1], pddims[2]),
            nn.Tanh()
        )

        ## Decoder
        for i in range(self.depth):
            setattr(self, 'sdconv_{}'.format(i+1), get_layer('basic')(sddims[i],sddims[i+1],kernels=[3,3], subsampling=args.g_upsampler  if i < self.depth-1 else 'none', norms=norms))

        self.final_conv = nn.Sequential(
            nn.Conv2d(sddims[-1], args.g_out_channels[1], kernel_size=3, stride=1, padding=1, groups=1, bias=True),
            nn.Tanh()
        )

    def forward(self, X, R):
        sources = [None]
        pout = torch.cat([X, R], 1)
        sout = X

        ## Encoder
        for i in range(self.depth):
            pout = getattr(self, 'pconv_{:d}'.format(i + 1))(pout)
            sout = getattr(self, 'sconv_{:d}'.format(i + 1))(sout)
            sources.append(torch.cat([pout, sout], 1))
        ## Linear Decoder
        p_emb = self.pdfc1(pout.view(-1, self.linear_in))

        p_vec = self.pdfc2(p_emb)

        ## Decoder
        sdout = sources.pop()

        for i in range(self.depth):
            # print(X.size())

            sdout = getattr(self, 'sdconv_{:d}'.format(i + 1))(sdout, _skip_feat=sources[-i-1] if i < self.depth-1 else None)

        return self.final_conv(sdout), p_vec, p_emb

    def estimate_preset(self, _input):
        pout = _input

        ## Encoder
        for i in range(self.depth):
            pout = getattr(self, 'pconv_{:d}'.format(i + 1))(pout)

        ## Linear Decoder
        p_emb = self.pdfc1(pout.view(-1, self.linear_in))
        p_vec = self.pdfc2(p_emb)
        return None, p_vec, p_emb

    def stylize(self, X, R1, preset_out_flag, preset_only=False):
        sources = [None]
        pout = torch.cat([X, R1], 1)
        
        if preset_only:
            return self.estimate_preset(pout)

        sout = X

        ## Encoder
        for i in range(self.depth):
            pout = getattr(self, 'pconv_{:d}'.format(i + 1))(pout)
            sout = getattr(self, 'sconv_{:d}'.format(i + 1))(sout)
            sources.append(torch.cat([pout, sout], 1))

        ## Linear Decoder
        if preset_out_flag:
            p_emb = self.pdfc1(pout.view(-1, self.linear_in))
            p_vec = self.pdfc2(p_emb)
        else:
            p_emb = None
            p_vec = None

        ## Decoder
        sdout = sources.pop()

        for i in range(self.depth):
            # print(X.size())
            sdout = getattr(self, 'sdconv_{:d}'.format(i + 1))(sdout, sources[-i-1])

        return self.final_conv(sdout), p_vec, p_emb

    def get_matched_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}# and not "pdfc" in str(k)}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        return model_dict