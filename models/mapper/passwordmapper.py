import torch
from torch import nn

from utils.function import *
from models.stylegan2.model import EqualLinear, PixelNorm

# mapping: {0,1}^16 -> R^512
class PasswordMapper(nn.Module):

    def __init__(self, opts):
        super(PasswordMapper, self).__init__()
        # Differ when input variable password
        self.module = nn.Sequential(
            nn.Linear(16, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.module(x)
        return x
    
class ModulationModule(nn.Module):
    def __init__(self, layernum):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = nn.Linear(512, 512)
        self.norm = nn.LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.beta_function = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x) 	
        gamma = self.gamma_function(embedding.float())
        gamma_f = torch.zeros_like(x)
        beta = self.beta_function(embedding.float())
        beta_f = torch.zeros_like(x)
        for i in range(len(gamma)):
            gamma_f[i] = gamma[i].repeat(self.layernum, 1)
            beta_f[i] = beta[i].repeat(self.layernum, 1)
        out = x * gamma_f + beta_f
        out = self.leakyrelu(out)
        return out
    
class SubWMapper(nn.Module):
    def __init__(self, layernum):
        super(SubWMapper, self).__init__()
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum) for i in range(3)])

    def forward(self, x, embedding):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            x = modulation_module(x, embedding)
        return x

class WMapper(nn.Module):

    def __init__(self, opts):
        super(WMapper, self).__init__()
        self.opts = opts
        self.mapping = SubWMapper(self.opts.mapping_layers)

    def forward(self, x, p_embedding, start_layer, mapping_layers):
        # TO DO
        x_fix_1 = x[:, :start_layer, :]
        x_mapping = x[:, start_layer : start_layer + mapping_layers, :]
        x_fix_2 = x[:, start_layer + mapping_layers:, :]
        x_fixed_1 = torch.zeros_like(x_fix_1)
        x_mapped = self.mapping(x_mapping, p_embedding)
        x_fixed_2 = torch.zeros_like(x_fix_2)

        out = torch.cat([x_fixed_1, x_mapped, x_fixed_2], dim=1)
        return out
    
class ReverseMapper(nn.Module):

    def __init__(self, opts):
        super(ReverseMapper, self).__init__()
        self.opts = opts
        self.layer = opts.start_layer
        self.module = nn.Sequential(
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.module(x[:, self.layer, :])
        return x