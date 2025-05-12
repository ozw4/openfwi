# %%
# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import sys

sys.path.append('..')

import datetime
import json
import os
import time

import hydra
import network
import numpy as np
import pytorch_ssim
import torch
import torchvision
import transforms as T
import utils
from dataset import FWIDataset
from hydra import compose, initialize
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Compose
from vis import *

with initialize(config_path='configs', version_base='1.3'):
	cfg = compose(config_name='test')


def evaluate(
	model,
	criterions,
	dataloader,
	device,
	k,
	ctx,
	vis_path,
	vis_batch,
	vis_sample,
	missing,
	std,
):
	model.eval()

	label_list, label_pred_list = [], []  # store denormalized predcition & gt in numpy
	label_tensor, label_pred_tensor = (
		[],
		[],
	)  # store normalized prediction & gt in tensor
	if missing or std:
		data_list, data_noise_list = [], []  # store original data and noisy/muted data

	with torch.no_grad():
		batch_idx = 0
		for data, label in dataloader:
			data = data.type(torch.FloatTensor).to(device, non_blocking=True)
			label = label.type(torch.FloatTensor).to(device, non_blocking=True)

			label_np = T.tonumpy_denormalize(
				label, ctx['label_min'], ctx['label_max'], exp=False
			)
			label_list.append(label_np)
			label_tensor.append(label)

			if missing or std:
				# Add gaussian noise
				data_noise = torch.clip(
					data
					+ (std**0.5)
					* torch.randn(data.shape).to(device, non_blocking=True),
					min=-1,
					max=1,
				)

				# Mute some traces
				mute_idx = np.random.choice(data.shape[3], size=missing, replace=False)
				data_noise[:, :, :, mute_idx] = data[0, 0, 0, 0]

				data_np = T.tonumpy_denormalize(
					data, ctx['data_min'], ctx['data_max'], k=k
				)
				data_noise_np = T.tonumpy_denormalize(
					data_noise, ctx['data_min'], ctx['data_max'], k=k
				)
				data_list.append(data_np)
				data_noise_list.append(data_noise_np)
				pred = model(data_noise)
			else:
				pred = model(data)

			label_pred_np = T.tonumpy_denormalize(
				pred, ctx['label_min'], ctx['label_max'], exp=False
			)
			label_pred_list.append(label_pred_np)
			label_pred_tensor.append(pred)

			# Visualization
			if vis_path and batch_idx < vis_batch:
				for i in range(vis_sample):
					plot_velocity(
						label_pred_np[i, 0],
						label_np[i, 0],
						f'{vis_path}/V_{batch_idx}_{i}.png',
					)  # , vmin=ctx['label_min'], vmax=ctx['label_max'])
					if missing or std:
						for ch in [2]:  # range(data.shape[1]):
							plot_seismic(
								data_np[i, ch],
								data_noise_np[i, ch],
								f'{vis_path}/S_{batch_idx}_{i}_{ch}.png',
								vmin=ctx['data_min'] * 0.01,
								vmax=ctx['data_max'] * 0.01,
							)
			batch_idx += 1

	label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
	label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)
	l1 = nn.L1Loss()
	l2 = nn.MSELoss()
	print(f'MAE: {l1(label_t, pred_t)}')
	print(f'MSE: {l2(label_t, pred_t)}')
	ssim_loss = pytorch_ssim.SSIM(window_size=11)
	print(
		f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}'
	)  # (-1, 1) to (0, 1)

	for name, criterion in criterions.items():
		print(f' * Velocity {name}: {criterion(label, label_pred)}')
	#     print(f'   | Velocity 2 layers {name}: {criterion(label[:1000], label_pred[:1000])}')
	#     print(f'   | Velocity 3 layers {name}: {criterion(label[1000:2000], label_pred[1000:2000])}')
	#     print(f'   | Velocity 4 layers {name}: {criterion(label[2000:], label_pred[2000:])}')

	name = f'{cfg.output_path}_to_{cfg.val_anno.split("/")[-1][:-3]}'
	with open(os.path.join(vis_path, f'metrics_{name}.txt'), 'w') as f:
		f.write(f'MAE: {l1(label_t, pred_t)}\n')
		f.write(f'MSE: {l2(label_t, pred_t)}\n')
		f.write(f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}\n')
		for name, criterion in criterions.items():
			f.write(f' * Velocity {name}: {criterion(label, label_pred)}\n')


@hydra.main(config_path='configs', config_name='test', version_base='1.3')
def main(cfg: DictConfig):
	print(cfg)
	print('torch version: ', torch.__version__)
	print('torchvision version: ', torchvision.__version__)

	utils.mkdir(cfg.output_path)
	device = torch.device(cfg.device)
	torch.backends.cudnn.benchmark = True

	with open('../dataset_config.json') as f:
		try:
			ctx = json.load(f)[cfg.dataset]
		except KeyError:
			print('Unsupported dataset.')
			sys.exit()

	if cfg.file_size is not None:
		ctx['file_size'] = cfg.file_size

	print('Loading data')
	print('Loading validation data')
	log_data_min = T.log_transform(ctx['data_min'], k=cfg.k)
	log_data_max = T.log_transform(ctx['data_max'], k=cfg.k)
	transform_valid_data = Compose(
		[
			T.LogTransform(k=cfg.k),
			T.MinMaxNormalize(log_data_min, log_data_max),
		]
	)

	transform_valid_label = Compose(
		[T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])]
	)
	# %%
	if cfg.val_anno[-3:] == 'txt':
		dataset_valid = FWIDataset(
			cfg.val_anno,
			sample_ratio=cfg.sample_temporal,
			file_size=ctx['file_size'],
			transform_data=transform_valid_data,
			transform_label=transform_valid_label,
		)
	else:
		dataset_valid = torch.load(cfg.val_anno)

	print('Creating data loaders')
	valid_sampler = SequentialSampler(dataset_valid)
	dataloader_valid = torch.utils.data.DataLoader(
		dataset_valid,
		batch_size=cfg.batch_size,
		sampler=valid_sampler,
		num_workers=cfg.workers,
		pin_memory=True,
		collate_fn=default_collate,
	)

	print('Creating model')
	if cfg.model not in network.model_dict:
		print('Unsupported model.')
		sys.exit()

	model = network.model_dict[cfg.model](
		upsample_mode=cfg.up_mode,
		sample_spatial=cfg.sample_spatial,
		sample_temporal=cfg.sample_temporal,
		norm=cfg.norm,
	).to(device)

	criterions = {
		'MAE': lambda x, y: np.mean(np.abs(x - y)),
		'MSE': lambda x, y: np.mean((x - y) ** 2),
	}

	if cfg.resume:
		print(cfg.resume)
		checkpoint = torch.load(cfg.resume, map_location='cpu', weights_only=False)
		model.load_state_dict(network.replace_legacy(checkpoint['model']))
		print(
			'Loaded model checkpoint at Epoch {} / Step {}.'.format(
				checkpoint['epoch'], checkpoint['step']
			)
		)

	if cfg.vis:
		# Create folder to store visualization results
		vis_folder = (
			f'visualization_{cfg.vis_suffix}' if cfg.vis_suffix else 'visualization'
		)
		vis_path = os.path.join(cfg.output_path, vis_folder)
		utils.mkdir(vis_path)
	else:
		vis_path = None

	print('Start testing')
	start_time = time.time()
	evaluate(
		model,
		criterions,
		dataloader_valid,
		device,
		cfg.k,
		ctx,
		vis_path,
		cfg.vis_batch,
		cfg.vis_sample,
		cfg.missing,
		cfg.std,
	)
	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print(f'Testing time {total_time_str}')


if __name__ == '__main__':
	main()

# %%
