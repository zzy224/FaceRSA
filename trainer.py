import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torchvision import transforms, utils

from utils.function import *
from models.stylegan2.model import Generator
from models.mapper.latentmapper import LatentMapper

class Trainer(nn.Module):
    def __init__(self, config, opts):
        super(Trainer, self).__init__()
        # Load Hyperparameters
        self.config = config
        self.device = torch.device(self.config['device'])
        self.scale = 2
        self.scale_mode = 'bilinear'
        self.opts = opts
        self.n_styles = 2 * int(np.log2(config['resolution'])) - 2
        self.opts.start_layer = config['start_layer']
        self.opts.mapping_layers = config['mapping_layers']

        # e4e
        ckpt = torch.load(opts.inversion_model_path, map_location='cpu')
        e4e_opts = ckpt['opts']
        e4e_opts['checkpoint_path'] = opts.inversion_model_path
        e4e_opts['device'] = self.device
        e4e_opts = argparse.Namespace(**e4e_opts)
        self.e4e = pSp(e4e_opts).to(self.device)
        self.e4e.eval()
        # StyleGAN
        self.StyleGAN = self.e4e.decoder
        self.StyleGAN.eval()
        self.mapper = LatentMapper(opts).to(self.device)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        print('loading checkpoint:' + checkpoint_path + '...')
        self.mapper.load_state_dict(state_dict['mapper_state_dict'])
        return 0

