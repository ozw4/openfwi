# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import os

import numpy as np
import transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm


class FWIDataset(Dataset):
	"""FWI dataset
	For convenience, in this class, a batch refers to a npy file
	instead of the batch used during training.

	Args:
	    anno: path to annotation file
	    preload: whether to load the whole dataset into memory
	    sample_ratio: downsample ratio for seismic data
	    file_size: # of samples in each npy file
	    transform_data|label: transformation applied to data or label

	"""

	def __init__(
		self,
		anno,
		preload=True,
		sample_ratio=1,
		file_size=500,
		transform_data=None,
		transform_label=None,
		flip=False,
		shot_drop=False,
	):
		if not os.path.exists(anno):
			print(f'Annotation file {anno} does not exists')
		self.preload = preload
		self.sample_ratio = sample_ratio
		self.file_size = file_size
		self.transform_data = transform_data
		self.transform_label = transform_label
		self.flip = flip
		self.shot_drop = shot_drop
		with open(anno) as f:
			self.batches = f.readlines()
		if preload:
			self.data_list, self.label_list = [], []
			for batch in tqdm(self.batches):
				data, label = self.load_every(batch)
				self.data_list.append(data)
				if label is not None:
					self.label_list.append(label)

	# Load from one line
	def load_every(self, batch):
		batch = batch.split('\t')
		data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
		data = np.load(data_path)[:, :, :: self.sample_ratio, :]
		data = data.astype('float32')
		if len(batch) > 1:
			label_path = batch[1][:-1]
			label = np.load(label_path)
			label = label.astype('float32')
		else:
			label = None

		return data, label

	def __getitem__(self, idx):
		rng = np.random.default_rng()
		batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
		if self.preload:
			data = self.data_list[batch_idx][sample_idx]
			label = (
				self.label_list[batch_idx][sample_idx]
				if len(self.label_list) != 0
				else None
			)
		else:
			batch = self.batches[batch_idx]
			batch = batch.split('\t')
			data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
			data = np.load(data_path, mmap_mode='r')[
				sample_idx, :, :: self.sample_ratio, :
			]
			data = data.astype('float32')
			if len(batch) > 1:
				label_path = batch[1][:-1]
				label = np.load(label_path, mmap_mode='r')[sample_idx]
				label = label.astype('float32')
			else:
				label = None
		if self.transform_data:
			data = self.transform_data(data)
		if self.transform_label and label is not None:
			label = self.transform_label(label)

		if self.flip and rng.random() > 0.5:
			if label is not None:
				label = np.flip(label, axis=2).copy()
			data = np.flip(data, axis=0).copy()
			data = np.flip(data, axis=2).copy()
		if self.shot_drop and rng.random() > 0.66:
			indices = rng.choice(np.arange(5), size=2, replace=False)
			data[indices] = 0
		elif self.shot_drop and rng.random() > 0.33:
			indices = rng.choice(np.arange(5), size=1, replace=False)
			data[indices] = 0

		return data, label if label is not None else np.array([])

	def __len__(self):
		return len(self.batches) * self.file_size


if __name__ == '__main__':
	transform_data = Compose(
		[
			T.LogTransform(k=1),
			T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1)),
		]
	)
	transform_label = Compose([T.MinMaxNormalize(2000, 6000)])
	dataset = FWIDataset(
		'relevant_files/temp.txt',
		transform_data=transform_data,
		transform_label=transform_label,
		file_size=1,
		preload=False,
	)
	data, label = dataset[0]
	print(data.shape)
	print(label is None)
