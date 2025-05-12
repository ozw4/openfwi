# %%

import sys

sys.path.append('/workspace/OpenFWI-main/')

import datetime
import json
import os
import time

import network
import torch
import torchvision
import transforms as T
import utils
from dataset import FWIDataset
from hydra import compose, initialize
from omegaconf import OmegaConf
from scheduler import WarmupMultiStepLR
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

with initialize(config_path='configs', version_base='1.3'):
	cfg = compose(config_name='train')

step = 0


def train_one_epoch(
	model,
	criterion,
	optimizer,
	lr_scheduler,
	dataloader,
	device,
	epoch,
	print_freq,
	writer,
):
	global step
	model.train()

	# Logger setup
	metric_logger = utils.MetricLogger(delimiter='  ')
	metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
	metric_logger.add_meter(
		'samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}')
	)
	header = f'Epoch: [{epoch}]'

	for data, label in metric_logger.log_every(dataloader, print_freq, header):
		start_time = time.time()
		optimizer.zero_grad()
		data, label = data.to(device), label.to(device)
		output = model(data)
		loss, loss_g1v, loss_g2v = criterion(output, label)
		loss.backward()
		optimizer.step()

		loss_val = loss.item()
		loss_g1v_val = loss_g1v.item()
		loss_g2v_val = loss_g2v.item()
		batch_size = data.shape[0]
		metric_logger.update(
			loss=loss_val,
			loss_g1v=loss_g1v_val,
			loss_g2v=loss_g2v_val,
			lr=optimizer.param_groups[0]['lr'],
		)
		metric_logger.meters['samples/s'].update(
			batch_size / (time.time() - start_time)
		)
		if writer:
			writer.add_scalar('loss', loss_val, step)
			writer.add_scalar('loss_g1v', loss_g1v_val, step)
			writer.add_scalar('loss_g2v', loss_g2v_val, step)
		step += 1
		lr_scheduler.step()


def evaluate(model, criterion, dataloader, device, writer):
	model.eval()
	metric_logger = utils.MetricLogger(delimiter='  ')
	header = 'Test:'
	with torch.no_grad():
		for data, label in metric_logger.log_every(dataloader, 20, header):
			data = data.to(device, non_blocking=True)
			label = label.to(device, non_blocking=True)
			output = model(data)
			loss, loss_g1v, loss_g2v = criterion(output, label)
			metric_logger.update(
				loss=loss.item(), loss_g1v=loss_g1v.item(), loss_g2v=loss_g2v.item()
			)

	# Gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print(f' * Loss {metric_logger.loss.global_avg:.8f}\n')
	if writer:
		writer.add_scalar('loss', metric_logger.loss.global_avg, step)
		writer.add_scalar('loss_g1v', metric_logger.loss_g1v.global_avg, step)
		writer.add_scalar('loss_g2v', metric_logger.loss_g2v.global_avg, step)
	return metric_logger.loss.global_avg


OmegaConf.set_struct(cfg, False)
print(OmegaConf.to_yaml(cfg))  # 中身を確認
# 追加パス処理（旧: parse_cfgの後処理相当）

train_anno_name = cfg.train_anno.split('.')[0]
output_path = os.path.join(cfg.model, train_anno_name, cfg.save_name)

cfg.train_anno = os.path.join(cfg.anno_path, cfg.train_anno)
cfg.val_anno = os.path.join(cfg.anno_path, cfg.val_anno)
if cfg.resume:
	cfg.resume = os.path.join(output_path, cfg.resume)
cfg.epochs = cfg.epoch_block * cfg.num_block
print(cfg)

print('torch version: ', torch.__version__)
print('torchvision version: ', torchvision.__version__)

utils.mkdir(output_path)  # create folder to store checkpoints
utils.init_distributed_mode(cfg)  # distributed mode initialization

# Set up tensorboard summary writer
train_writer, val_writer = None, None
if cfg.tensorboard:
	utils.mkdir(output_path)  # create folder to store tensorboard logs
	if not cfg.distributed or ((cfg.rank == 0) and (cfg.local_rank == 0)):
		train_writer = SummaryWriter(os.path.join(output_path, 'logs', 'train'))
		val_writer = SummaryWriter(os.path.join(output_path, 'logs', 'val'))


device = torch.device(cfg.device)
torch.backends.cudnn.benchmark = True

with open('/workspace/OpenFWI-main/dataset_config.json') as f:
	try:
		ctx = json.load(f)[cfg.dataset]
	except KeyError:
		print('Unsupported dataset.')
		sys.exit()

if cfg.file_size is not None:
	ctx['file_size'] = cfg.file_size

# Create dataset and dataloader
print('Loading data')
print('Loading training data')

# Normalize data and label to [-1, 1]
transform_data = Compose(
	[
		T.LogTransform(k=cfg.k),
		T.MinMaxNormalize(
			T.log_transform(ctx['data_min'], k=cfg.k),
			T.log_transform(ctx['data_max'], k=cfg.k),
		),
	]
)
transform_label = Compose([T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])])

if cfg.train_anno[-3:] == 'txt':
	dataset_train = FWIDataset(
		cfg.train_anno,
		preload=False,
		sample_ratio=cfg.sample_temporal,
		file_size=ctx['file_size'],
		transform_data=transform_data,
		transform_label=transform_label,
		flip=cfg.flip,
		shot_drop=cfg.shot_drop,
	)
else:
	dataset_train = torch.load(cfg.train_anno)

print('Loading validation data')
if cfg.val_anno[-3:] == 'txt':
	dataset_valid = FWIDataset(
		cfg.val_anno,
		preload=False,
		sample_ratio=cfg.sample_temporal,
		file_size=ctx['file_size'],
		transform_data=transform_data,
		transform_label=transform_label,
		flip=False,
	)
else:
	dataset_valid = torch.load(cfg.val_anno)

print('Creating data loaders')
if cfg.distributed:
	train_sampler = DistributedSampler(dataset_train, shuffle=True)
	valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
else:
	train_sampler = RandomSampler(dataset_train)
	valid_sampler = RandomSampler(dataset_valid)

dataloader_train = DataLoader(
	dataset_train,
	batch_size=cfg.batch_size,
	sampler=train_sampler,
	num_workers=cfg.workers,
	pin_memory=True,
	drop_last=True,
	collate_fn=default_collate,
)

dataloader_valid = DataLoader(
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
).to(device)

if cfg.distributed and cfg.sync_bn:
	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Define loss function
l1loss = nn.L1Loss()
l2loss = nn.MSELoss()


def criterion(pred, gt):
	loss_g1v = l1loss(pred, gt)
	loss_g2v = l2loss(pred, gt)
	loss = cfg.lambda_g1v * loss_g1v + cfg.lambda_g2v * loss_g2v
	return loss, loss_g1v, loss_g2v


# Scale lr according to effective batch size
lr = cfg.lr * cfg.world_size
optimizer = torch.optim.AdamW(
	model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay
)

# Convert scheduler to be per iteration instead of per epoch
warmup_iters = cfg.lr_warmup_epochs * len(dataloader_train)
lr_milestones = [len(dataloader_train) * m for m in cfg.lr_milestones]
lr_scheduler = WarmupMultiStepLR(
	optimizer,
	milestones=lr_milestones,
	gamma=cfg.lr_gamma,
	warmup_iters=warmup_iters,
	warmup_factor=1e-5,
)

model_without_ddp = model
if cfg.distributed:
	model = DistributedDataParallel(model, device_ids=[cfg.local_rank])
	model_without_ddp = model.module

if cfg.resume:
	checkpoint = torch.load(cfg.resume, map_location='cpu')
	model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
	optimizer.load_state_dict(checkpoint['optimizer'])
	lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
	cfg.start_epoch = checkpoint['epoch'] + 1
	step = checkpoint['step']
	lr_scheduler.milestones = lr_milestones

print('Start training')
start_time = time.time()
best_loss = 10
chp = 1
for epoch in range(cfg.start_epoch, cfg.epochs):
	if cfg.distributed:
		train_sampler.set_epoch(epoch)
	train_one_epoch(
		model,
		criterion,
		optimizer,
		lr_scheduler,
		dataloader_train,
		device,
		epoch,
		cfg.print_freq,
		train_writer,
	)

	loss = evaluate(model, criterion, dataloader_valid, device, val_writer)

	checkpoint = {
		'model': model_without_ddp.state_dict(),
		'optimizer': optimizer.state_dict(),
		'lr_scheduler': lr_scheduler.state_dict(),
		'epoch': epoch,
		'step': step,
		'cfg': cfg,
	}
	# Save checkpoint per epoch
	if loss < best_loss:
		utils.save_on_master(checkpoint, os.path.join(output_path, 'checkpoint.pth'))
		print('saving checkpoint at epoch: ', epoch)
		chp = epoch
		best_loss = loss
	# Save checkpoint every epoch block
	print('current best loss: ', best_loss)
	print('current best epoch: ', chp)
	if output_path and (epoch + 1) % cfg.epoch_block == 0:
		utils.save_on_master(
			checkpoint, os.path.join(output_path, f'model_{epoch + 1}.pth')
		)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f'Training time {total_time_str}')

# %%
