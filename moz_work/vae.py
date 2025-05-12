# %%

import sys

sys.path.append('..')

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transforms as T
from cuml.manifold import UMAP
from matplotlib import cm
from network import VAEClassifier
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm


def vae_loss(recon_x, x, mu, logvar, logits, target, beta=1.0):
	recon_loss = F.mse_loss(recon_x, x, reduction='mean')
	kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	class_loss = F.cross_entropy(logits, target)
	return recon_loss + beta * kl_loss + class_loss, recon_loss, kl_loss, class_loss


def train_one_epoch(model, dataloader, optimizer, device, beta=1.0):
	model.train()
	total_loss, total_cls, total_recon, total_kl = 0, 0, 0, 0
	for x, y in dataloader:
		x, y = x.to(device), y.to(device)
		optimizer.zero_grad()
		recon_x, logits, mu, logvar = model(x)
		loss, recon, kl, cls = vae_loss(recon_x, x, mu, logvar, logits, y, beta)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
		total_recon += recon.item()
		total_kl += kl.item()
		total_cls += cls.item()
	n = len(dataloader)
	return {
		'loss': total_loss / n,
		'recon': total_recon / n,
		'kl': total_kl / n,
		'class': total_cls / n,
	}


def validate(model, dataloader, device, beta=1.0):
	model.eval()
	total_loss, total_cls, total_recon, total_kl = 0, 0, 0, 0
	correct = 0
	zs, ys = [], []
	with torch.no_grad():
		for x, y in dataloader:
			x, y = x.to(device), y.to(device)
			recon_x, logits, mu, logvar = model(x)
			loss, recon, kl, cls = vae_loss(recon_x, x, mu, logvar, logits, y, beta)
			total_loss += loss.item()
			total_recon += recon.item()
			total_kl += kl.item()
			total_cls += cls.item()
			pred = logits.argmax(dim=1)
			correct += (pred == y).sum().item()
			zs.append(mu.cpu())
			ys.append(y.cpu())
	n = len(dataloader.dataset)
	return {
		'loss': total_loss / len(dataloader),
		'recon': total_recon / len(dataloader),
		'kl': total_kl / len(dataloader),
		'class': total_cls / len(dataloader),
		'acc': correct / n,
		'embeddings': torch.cat(zs, dim=0).numpy(),
		'labels': torch.cat(ys, dim=0).numpy(),
	}


def log_umap(writer, features, labels, epoch, tag='embedding'):
	reducer = UMAP(n_components=2, random_state=42, output_type='numpy')
	embedding_2d = reducer.fit_transform(features)

	plt.figure(figsize=(8, 6))
	unique_labels = np.unique(labels)
	cmap = cm.get_cmap('tab10', len(unique_labels))
	for i, (label, cate) in enumerate(zip(unique_labels, categories, strict=True)):
		mask = labels == label
		plt.scatter(
			embedding_2d[mask, 0],
			embedding_2d[mask, 1],
			label=f'{cate}',
			color=cmap(i),
			s=5,
			alpha=0.7,
		)
	plt.legend(loc='upper right', fontsize='small')
	plt.title(f'UMAP of Latent Space at Epoch {epoch}')
	plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
	plt.tight_layout()

	writer.add_figure(tag, plt.gcf(), epoch)
	plt.close()


with open('/workspace/OpenFWI-main/split_files/onesample_each_train.txt') as f:
	train_lines = [line.strip().split('\t') for line in f]
with open('/workspace/OpenFWI-main/split_files/onesample_each_val.txt') as f:
	val_lines = [line.strip().split('\t') for line in f]


train_data_paths = [line[0] for line in train_lines]
val_data_paths = [line[0] for line in val_lines]
categories = [Path(path).parts[4] for path in train_data_paths]
# %%

train_data = []
train_label = []
val_data = []
val_label = []

for i, data_path in tqdm(enumerate(train_data_paths)):
	da = np.load(data_path)
	train_data.append(da)
	train_label.append([i] * len(da))
for i, data_path in tqdm(enumerate(val_data_paths)):
	da = np.load(data_path)
	val_data.append(da)
	val_label.append([i] * len(da))
train_data = np.concatenate(train_data, axis=0)
train_label = np.concatenate(train_label, axis=0)
val_data = np.concatenate(val_data, axis=0)
val_label = np.concatenate(val_label, axis=0)

k = 1
# Normalize data and label to [-1, 1]
transform_data = Compose(
	[
		T.LogTransform(k=k),
		T.MinMaxNormalize(
			T.log_transform(np.min(train_data), k=k),
			T.log_transform(np.max(train_data), k=k),
		),
	]
)

train_data = transform_data(train_data)
val_data = transform_data(val_data)

train_data = np.transpose(train_data, (0, 1, 3, 2))
val_data = np.transpose(val_data, (0, 1, 3, 2))

train_data = np.pad(train_data, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant')
val_data = np.pad(val_data, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant')

train_data = torch.from_numpy(train_data).float()
train_label = torch.from_numpy(train_label).long()
val_data = torch.from_numpy(val_data).float()
val_label = torch.from_numpy(val_label).long()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAEClassifier(latent_dim=64, num_classes=8)
model.to(device)

epoch = 200
batch_size = 32
train_dataset = TensorDataset(train_data, train_label)
val_dataset = TensorDataset(val_data, val_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Optimizer 設定
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
beta = 1.0  # KL項の重み

writer = SummaryWriter(log_dir='vae_log/vae')
best_acc = 0.0
# 訓練ループ
for ep in range(1, epoch + 1):
	train_log = train_one_epoch(model, train_loader, optimizer, device, beta)
	val_log = validate(model, val_loader, device, beta)
	print(
		f'[Epoch {ep}] '
		f'Train - Loss: {train_log["loss"]:.4f}, Recon: {train_log["recon"]:.4f}, KL: {train_log["kl"]:.4f}, Class: {train_log["class"]:.4f} | '
		f'Val - Loss: {val_log["loss"]:.4f}, Recon: {val_log["recon"]:.4f}, KL: {val_log["kl"]:.4f}, Class: {val_log["class"]:.4f}, Acc: {val_log["acc"]:.4f}'
	)

	writer.add_scalar('Loss/Train', train_log['loss'], ep)
	writer.add_scalar('Loss/Val', val_log['loss'], ep)
	writer.add_scalar('Accuracy/Val', val_log['acc'], ep)
	if epoch % 10 == 0:
		val_log = validate(model, val_loader, device)
		log_umap(writer, val_log['embeddings'], val_log['labels'], ep)
	if val_log['acc'] > best_acc:
		best_acc = val_log['acc']
		torch.save(model.state_dict(), 'vae_log/best_model.pth')
		print(f'Saved new best model with acc {best_acc:.4f} at epoch {epoch}')

# %%
