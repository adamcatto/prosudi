"""
Much of this file is adapted from https://github.com/nmwsharp/diffusion-net/blob/master/src/layers.py
in order to reproduce the DiffusionNet model
"""

import sys
import os
import random
from collections import Counter

import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import utils
from utils import toNP
import geometry


class LaplacianBlock(nn.Module):
    """
    Applies Laplacian diffusion in the spectral domain like
        f_out = e ^ (lambda_i t) f_in
    with learned per-channel parameters t.
    Inputs:
      - values: (K,C) in the spectral domain
      - evals: (K) eigenvalues
    Outputs:
      - (K,C) transformed values in the spectral domain
    """

    def __init__(self, C_inout):
        super(LaplacianBlock, self).__init__()
        self.C_inout = C_inout

        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        nn.init.constant_(self.diffusion_time, 0.0001)

    def forward(self, x, evals):

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * torch.abs(self.diffusion_time).unsqueeze(0)) 
        return diffusion_coefs * x


class PairwiseDot(nn.Module):
    """
    Compute dot-products between input vectors with a learned complex-linear layer.
    
    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots 
    """

    def __init__(self, C_inout, linear_complex=True):
        super(PairwiseDot, self).__init__()

        self.C_inout = C_inout
        self.linear_complex = linear_complex

        if(self.linear_complex):
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

    def forward(self, vectors):

        vectorsA = vectors # (V,C)

        if self.linear_complex:
            vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
            vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
        else:
            vectorsBreal = self.A(vectors[...,0])
            vectorsBimag = self.A(vectors[...,1])

        dots = vectorsA[...,0] * vectorsBreal + vectorsA[...,1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_inout, C_hidden,
                 dropout=False, pairwise_dot=True, dot_linear_complex=True):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_inout = C_inout
        self.C_hidden = C_hidden

        self.dropout = dropout
        self.pairwise_dot = pairwise_dot
        self.dot_linear_complex = dot_linear_complex

        # Laplacian block
        self.spec0 = LaplacianBlock(self.C_inout)
        
        self.C_mlp = 2*self.C_inout
      
        if self.pairwise_dot:
            self.pairwise_dot = PairwiseDot(self.C_inout, linear_complex=self.dot_linear_complex)
            self.C_mlp += self.C_inout

        # MLPs
        self.mlp0 = MiniMLP([self.C_mlp] + list(self.C_hidden) + [self.C_inout], dropout=self.dropout)


    def forward(self, x0, mass, evals, evecs, grad_from_spectral):

        if x0.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x0.shape, self.C_inout))

        # Transform to spectral
        x0_spec = geometry.to_basis(x0, evecs, mass)
        
        # Laplacian block 
        x0_spec = self.spec0(x0_spec, evals)

        # Transform back to per-vertex 
        x0_lap = geometry.from_basis(x0_spec, evecs)
        x0_comb = torch.cat((x0, x0_lap), dim=-1)

        if self.pairwise_dot:
            # If using the pairwise dot block, add it to the scalar values as well
            x0_grad = utils.cmatvecmul_stacked(grad_from_spectral, x0_spec)
            x0_gradprods = self.pairwise_dot(x0_grad)
            x0_comb = torch.cat((x0_comb, x0_gradprods), dim=-1)
        
        # Apply the mlp
        x0_out = self.mlp0(x0_comb)

        # Skip connection
        x0_out = x0_out + x0

        return x0_out


class DiffusionNetBody(nn.Module):
    """
    defines the main, non-application-specific architecture.
    An application-specific model is to be defined as a DiffusionNetBody model with additional application-specific 
    layers.
    """
    def __init__(self, input_size, dropout=False,
                pairwise_dot=True, dot_linear_complex=True, num_blocks=4, width=32):
        super(DiffusionNetBody, self).__init__()
        
        if isinstance(width, int):
            width_2d = [width, width]
        
        self.input_linear_layer = nn.Linear(input_size, width)
        self.blocks = []

        for i in range(num_blocks):
            self.blocks.append(DiffusionNetBlock(C_inout=width, C_hidden=width_2d))
            self.add_module('block_' + str(i), self.blocks[-1])

    # == we will only be using point clouds in this application
    def forward(self, verts, mass, evals, evecs, grad_from_spectral):
        x0 = geometry.normalize_positions(verts)
        x0 = self.input_linear_layer(x0)

        for _, b in tqdm(enumerate(self.blocks)):
            x0 = b(x0, mass, evals, evecs, grad_from_spectral)

        return x0


class DiffusionNetBindingSite(nn.Module):
    def __init__(self, input_size, dropout=False, pairwise_dot=True, dot_linear_complex=True, num_blocks=4, width=32, num_eigenvecs=128):
        super(DiffusionNetBindingSite, self).__init__()

        self.num_eigenvecs = num_eigenvecs
        self.encoder = DiffusionNetBody(input_size=input_size, dropout=dropout, pairwise_dot=pairwise_dot,
                                        dot_linear_complex=dot_linear_complex, num_blocks=num_blocks, width=width)
        self.fc1 = nn.Linear(width, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 2)
        self.sigmoid = nn.Sigmoid()

        self.blocks = [self.fc1, self.fc2, self.fc3, self.sigmoid]

    def forward(self, protein_point_cloud):
        if isinstance(protein_point_cloud, np.ndarray):
            verts = torch.from_numpy(protein_point_cloud)
        else:
            verts = protein_point_cloud

        # data := (frames, massvec, evals, evecs, grad_from_spectral)
        data = geometry.compute_operators(verts=verts, faces=[], k_eig=self.num_eigenvecs)

        x0 = self.encoder(verts, *data[1:])
        for b in self.blocks:
            x0 = b(x0)
        
        return x0


class DiffusionNetPPI(nn.Module):
    def __init__(self, p1, p2, input_size, dropout=False, pairwise_dot=True, dot_linear_complex=True, num_blocks=4, width=32):
        super(DiffusionNetPPI, self).__init__()
        
        self.encoder1 = DiffusionNetBody(input_size=input_size, dropout=dropout, pairwise_dot=pairwise_dot,
                                        dot_linear_complex=dot_linear_complex, num_blocks=num_blocks, width=width)
        self.encoder2 = DiffusionNetBody(input_size=input_size, dropout=dropout, pairwise_dot=pairwise_dot,
                                        dot_linear_complex=dot_linear_complex, num_blocks=num_blocks, width=width)

        self.fc1 = nn.Linear(1024, 16)
        self.fc2 = nn.Linear(16, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, p1, p2, num_eigenvecs=128):
        # encode surfaces using separate DiffusionNet bodies
        if isinstance(p1, np.ndarray):
            verts1 = torch.from_numpy(p1)
        else:
            verts1 = p1
        
        if isinstance(p2, np.ndarray):
            verts2 = torch.from_numpy(p2)
        else:
            verts2 = p2

        data1 = geometry.compute_operators(verts=verts1, faces=[], k_eig=num_eigenvecs)
        data2 = geometry.compute_operators(verts=verts2, faces=[], k_eig=num_eigenvecs)

        # encode each point cloud with separate DiffusionNetBody encoders
        x1 = self.encoder1(*data1)
        x2 = self.encoder2(*data2)
        
        # TODO: sampling from encoded proteins ==> ??? ==> covariance matrix ==> flatten =: x0
        # need to figure out how best to sample (learnable? random seems out of place here... not sure.)
        
        ## placeholder ##
        x0 = np.random.rand()
        x0 = torch.from_numpy(x0)
        ## / placeholder ##

        x0 = self.fc1(x0)
        print(x0)
        x0 = self.fc2(x0)
        print(x0)
        x0 = self.fc3(x0)
        print(x0)
        return x0


def _do_sanity_check():
    net = DiffusionNetBindingSite(input_size=3)
    verts = torch.from_numpy(np.load('../data/input/raw_pdb_arrays/train/pdb1a0g_atomxyz.npy'))
    #frames, massvec, evals, evecs, grad_from_spectral = geometry.compute_operators(verts=verts, faces=[], k_eig=64)
    #x = net(verts, massvec, evals, evecs,grad_from_spectral)
    x = net(verts)
    print(x)

"""
notes to self:

    +   change dataset to take as input not just xyz coordinate features, but one-hot encoded atomtypes.
        looks like wee will need to change structure of forward passes in the blocks, since they rely on coordinates
        (e.g. `verts` is shape = Nx3)
    
    +   ppi: figure out how to go from encoded point cloud to single feature vector in a spatially meaningful way

    +   some parts of geometry.py need to be modified to be compatible with point clouds – faces as list type is not
        compatible with `numel()`, `compute_operators()` can't be called on empty faces collection
"""
