import torch
from torch import nn

from utils.function import *
from models.mapper.passwordmapper import PasswordMapper
from models.mapper.passwordmapper import WMapper
from models.mapper.passwordmapper import ReverseMapper
from models.stylegan2.model import Generator

# mapping: p & w -> w + delta_w, img
class LatentMapper(nn.Module):

    def __init__(self, opts):
        super(LatentMapper, self).__init__()
        # Differ when input variable password
        self.opts = opts
        self.passwordmapper = PasswordMapper(opts).cuda()
        self.wmapper = WMapper(opts).cuda()
        self.remapper = ReverseMapper(opts).cuda()
        self.decoder = Generator(1024, 512, 8)
        self.load_weights()

    def load_weights(self):
        if self.opts.inversion_model_path is not None:
            ckpt = torch.load(self.opts.inversion_model_path, map_location='cpu')
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
        else:
            raise RuntimeError('Missing weight of the inversion model!')	

    def forward(self, x, p, start_layer=5, mapping_layers=4):
        # p_embedding = self.passwordmapper(p)
        p_embedding = self.passwordmapper(p)
        result_w = x + 0.1 * self.wmapper(x, p_embedding, start_layer, mapping_layers)
        result_img, _ = self.decoder([result_w], input_is_latent=True, randomize_noise=False)
        return result_w, result_img