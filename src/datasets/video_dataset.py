# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import glob
import os
import warnings

from logging import getLogger

import numpy as np
import pandas as pd
import scipy.signal

from edfpy import Reader

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=(256*16),
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('VideoDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('VideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """ Video classification dataset. """

    def __init__(
        self,
        data_paths,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_long_videos=int(10**9),
        raw_sample_rate=500,
        model_sample_rate=256,
    ):
        self.data_paths = glob.glob(f"{data_paths}/sub-LTP*/ses-*/eeg/")
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_long_videos = filter_long_videos
        self.raw_sample_rate = raw_sample_rate
        self.model_sample_rate = model_sample_rate

        # Load video paths and labels
        samples, channels, coords_x, coords_y = [], [], [], []
        for data_path in self.data_paths:
            found_edf_file = False
            for file_name in glob.glob(f"{data_path}/*"):
                if not found_edf_file and (
                    file_name.endswith(".bdf") or file_name.endswith(".edf")
                ):
                    samples.append(os.path.realpath(file_name))
                    found_edf_file = True
                elif file_name.endswith("CapTrak_electrodes_normalized.tsv"):
                    data = pd.read_csv(file_name, delimiter="\t")
                    channels.append(list(data.values[:, 1]))
                    coords_x.append(list(data.values[:, 2]))
                    coords_y.append(list(data.values[:, 3]))

        self.samples = samples
        self.channels = channels
        self.coords_x = coords_x
        self.coord_y = coords_y

    def __getitem__(self, index):
        sample = self.samples[index]
        channel_names = self.channels[index]
        coords = np.stack([self.coords_x[index], self.coords_y[index]], axis=1)

        # Keep trying to load videos until you find a valid sample
        loaded_eeg = False
        while not loaded_eeg:
            buffer, clip_indices = self.loadeeg_edf(sample, channel_names) # [N C T]
            loaded_eeg = len(buffer > 0)
            if not loaded_eeg:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]
                channel_names = self.channels[index]
                coords = np.stack([self.coords_x[index], self.coords_y[index]], axis=1)

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, coords, clip_indices

    def loadeeg_edf(self, sample, coords, channel_names):
        """ Load video content using Decord """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            r = Reader(fname)
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fstp = int(self.duration * self.raw_sample_rate / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = int(self.raw_sample_rate * r.duration) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        all_buffers = r.get_physical_samples(all_indices)
        all_buffers = {
            k: scipy.signal.resample(
                all_buffers[k], self.model_sample_rate * clip_len
            ) for k in channel_names
        }

        channel_idxs = np.random.choice(len(channel_names), self.num_model_channels)
        buffers = np.stack([all_buffers[channel_names[idx]] for idx in channel_idxs], axis=0)
        selected_coords = coords[channel_idxs, :]

        return buffers, selected_coords, clip_indices

    def __len__(self):
        return len(self.samples)
