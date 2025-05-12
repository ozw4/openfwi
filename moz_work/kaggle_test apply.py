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
import time
from pathlib import Path

import network
import numpy as np
import torch
import torchvision
import transforms as T
import utils
from hydra import compose, initialize
from torchvision.transforms import Compose
from vis import *

with initialize(config_path='configs', version_base='1.3'):
	cfg = compose(config_name='test')
import pandas as pd
from tqdm import tqdm

print(cfg)
print('torch version: ', torch.__version__)
print('torchvision version: ', torchvision.__version__)

train_anno_name = cfg.train_anno.split('.')[0]
output_path = Path(cfg.model) / train_anno_name / cfg.save_name
weight_path = output_path / 'checkpoint.pth'
utils.mkdir(output_path)
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

test_data_path = Path('/workspace/OpenFWI-main/data/test')
test_data_list = np.sort(list(test_data_path.glob('*npy')))


test_data = []
for path in test_data_list:
	data = np.load(path)
	data = transform_valid_data(data)
	test_data.append(data)

test_data = np.array(test_data)

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


print(weight_path)
checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
model.load_state_dict(network.replace_legacy(checkpoint['model']))
print(
	'Loaded model checkpoint at Epoch {} / Step {}.'.format(
		checkpoint['epoch'], checkpoint['step']
	)
)

print('Start testing')
start_time = time.time()


model.eval()

label_pred_list = []  # store denormalized predcition & gt in numpy

batch_size = 32
with torch.no_grad():
	batch_idx = 0
	for i in tqdm(range(0, len(test_data), batch_size)):
		data = test_data[i : i + batch_size]
		data = torch.from_numpy(data)
		data = data.float().to(device, non_blocking=True)

		pred = model(data)

		label_pred_np = T.tonumpy_denormalize(
			pred, ctx['label_min'], ctx['label_max'], exp=False
		)
		label_pred_list.append(label_pred_np)

label_pred = np.concatenate(label_pred_list, axis=0)
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f'Testing time {total_time_str}')


df_result = []
for i in tqdm(range(len(label_pred))):
	oid = str(test_data_list[0]).split('/')[-1][:-4]
	for y in range(70):
		row = {}
		row['oid_ypos'] = f'{oid}_y_{y}'
		for x in range(1, 70, 2):
			row[f'x_{x}'] = float(label_pred[i, 0, y, x])
		df_result.append(row)

df_submission = pd.DataFrame(df_result)

# CSVとして保存
df_submission.to_csv('submission.csv', index=False)

# %%
