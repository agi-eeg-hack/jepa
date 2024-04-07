# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class PatchEmbedEEG(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=512,
        in_chans=1,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T = x.shape
        x = x.unsqueeze(2)
        x = self.proj(x).transpose(2, 3).flatten(1, 2).transpose(1, 2)
        return x
