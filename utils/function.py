import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import argparse

import random

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
	return d_filt

def downscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode=mode)
    return x

# change a single stylegan output to a image
def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

# generate random password with the length user input 
def unsigned_long_to_binary_repr(unsigned_long, passwd_length):
    batch_size = unsigned_long.shape[0]

    binary = np.empty((batch_size, passwd_length), dtype=np.float32)
    for idx in range(batch_size):
        binary[idx, :] = np.array([int(item) for item in bin(unsigned_long[idx])[2:].zfill(passwd_length)])

    return binary

def generate_code(passwd_length, batch_size, device, inv, gen_random_WR):
    p = torch.zeros((batch_size, passwd_length)).to(device)
    inv_p = torch.zeros((batch_size, passwd_length)).to(device)
    rand_p = torch.zeros((batch_size, passwd_length)).to(device)
    rand_inv_p = torch.zeros((batch_size, passwd_length)).to(device)
    rand_inv_2nd_p = torch.zeros((batch_size, passwd_length)).to(device)

    for i in range(batch_size):
        for j in range(passwd_length):
            p[i][j] = random.randint(0,1)
            inv_p[i][j] = 1 - p[i][j]
            rand_p[i][j] = p[i][j]
            rand_inv_p[i][j] = 1 - p[i][j]
            rand_inv_2nd_p[i][j] = random.randint(0,1)
            
    loc = random.randint(0, passwd_length - 1)
    for i in range(batch_size):
        rand_p[i][loc] = 1 - p[i][loc]
        rand_inv_p[i][loc] = 1 - rand_inv_p[i][loc]
    if not inv:
        p = (p - 0.5) * 2
        rand_p = (rand_p - 0.5) * 2
        return p, rand_p
    else:    
        if gen_random_WR:
            return p-0.5, rand_p-0.5, inv_p-0.5, rand_inv_p-0.5, rand_inv_2nd_p-0.5
        return p-0.5, rand_p-0.5, inv_p-0.5, rand_inv_p-0.5
    
    
def get_random_index(total):
    start_idx = random.randint(0, total - 1)
    end_idx = random.randint(0, total - 1)
    while end_idx == start_idx:
        end_idx = random.randint(0, total - 1)
    if start_idx < end_idx:
        return start_idx, end_idx
    else:
        return end_idx, start_idx