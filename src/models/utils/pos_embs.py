# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch


def get_eeg_sincos_pos_embed(embed_dim, coords, grid_size, cls_token=False, uniform_power=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [1,grid_size*coords.shape[0], embed_dim] (w/o cls_token)
                or [1,1+grid_size*coords.shape[0], embed_dim] (w/ cls_token)
    """
    grid_x = coords[:, 0].astype(float)
    grid_y = coords[:, 1].astype(float)
    grid_d = torch.arange(grid_size, dtype=torch.float)
    grid_d, grid_x = torch.meshgrid(grid_d, grid_x, indexing='xy')  # order of meshgrid is very important for indexing as [c, t]
    _, grid_y = torch.meshgrid(grid_d, grid_y, indexing='xy')  # order of meshgrid is very important for indexing as [c, t]

    if not uniform_power:
        y_embed_dim = embed_dim // 4
        x_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        y_embed_dim = x_embed_dim = d_embed_dim = int(math.ceil(embed_dim/6)*2)

    emb_y = get_1d_sincos_pos_embed_from_grid(y_embed_dim, grid_y)  # (T*C, D1)
    emb_x = get_1d_sincos_pos_embed_from_grid(x_embed_dim, grid_x)  # (T*C, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*C, D3)
    pos_embed = torch.concat([emb_d, emb_y, emb_x], dim=1)  # (T*C, D)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = torch.concat([pos_embed.new_zeros([1, embed_dim]), pos_embed], dim=0)

    pos_embed = pos_embed.unsqueeze(0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = torch.outer(pos, omega)   # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb
