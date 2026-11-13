#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:17:44 2026

@author: blainebehen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:17:44 2026

@author: blainebehen
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


# Remove duplicate images, then split by game ID so no game appears in both train and test.
csv_path = "/Users/blainebehen/Desktop/Projects/Jersey-Network/archive/train_player_numbers.csv"
img_dir = "/Users/blainebehen/Desktop/Projects/Jersey-Network/archive/train_player_numbers"   # folder with the .png files

df = pd.read_csv(csv_path)
df = df.drop_duplicates("filename")

df["game"] = df["filename"].str.split("_").str[0]


games = df["game"].unique()                             # array of distinct game IDs
rng = np.random.default_rng(0)                          # fixed seed: same split every run
rng.shuffle(games)                                      # random order, so test games aren't just the highest IDs

cut_off = round(len(games) * 0.8)                       # number of games for training
train_games = games[:cut_off]                           # first 80% of games
test_games = games[cut_off:]                            # remaining 20%


df_train = df[df["game"].isin(train_games)]             # True for rows whose game is a training game
df_test = df[df["game"].isin(test_games)]



class JerseyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_labels = df.reset_index(drop=True)    # renumber rows 0..n-1 so idx lookups work
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)                    # number of images

    def __getitem__(self, idx):
        filename = self.img_labels.loc[idx, "filename"]
        img_path = os.path.join(self.img_dir, filename)                 # folder + filename
        image = Image.open(img_path).convert("RGB")                     # load the .png as color
        label = torch.tensor(self.img_labels.loc[idx, "label"], dtype=torch.long)   # jersey number 0-99

        if self.transform:
            image = self.transform(image)              # PIL image -> tensor (and normalized, if included)

        return image, label


# ---------- Channel mean and std (training images only, computed once) ----------
raw_train = JerseyDataset(df_train, img_dir, transform=transforms.ToTensor())   # raw 0-1 values
raw_loader = DataLoader(raw_train, batch_size=1000)

total = torch.zeros(3)        # running sum of pixel values, per channel (R, G, B)
total_sq = torch.zeros(3)     # running sum of squared pixel values, per channel
n_pixels = 0                  # pixels counted per channel

for x, y in raw_loader:                                 # x: B x 3 x 64 x 64
    total += x.sum(dim=(0, 2, 3))                       # sum over images, rows, cols -> 3 numbers
    total_sq += (x ** 2).sum(dim=(0, 2, 3))             # same, for squared values
    n_pixels += x.shape[0] * x.shape[2] * x.shape[3]    # images x 64 x 64

mean = total / n_pixels                                 # mu = sum(x) / N
std = (total_sq / n_pixels - mean ** 2).sqrt()          # sigma = sqrt(sum(x^2)/N - mu^2)


# ---------- Normalized datasets and loaders ----------
norm = transforms.Compose([transforms.ToTensor(),                     # PIL image -> 3 x 64 x 64, values 0-1
                           transforms.Normalize(mean.tolist(), std.tolist())])   # z-score each channel

train_data = JerseyDataset(df_train, img_dir, transform=norm)
test_data = JerseyDataset(df_test, img_dir, transform=norm)

loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)


# ---------- Parameters ----------
# He initialization: weights ~ N(0, 2/n), n = inputs added up per output (fan_in)
n1 = 3 * 3 * 3        # conv 1: 3 color channels x 3 x 3 stencil  = 27
n2 = 16 * 3 * 3       # conv 2: 16 maps x 3 x 3 stencil           = 144
n3 = 32 * 16 * 16     # detectors: all flattened features          = 8192
n4 = 128              # scores: all detectors                      = 128

K1 = (torch.randn(16, 3, 3, 3) * (2 / n1) ** 0.5).requires_grad_(True)    # conv 1 stencils
beta1 = torch.zeros(16).requires_grad_(True)                              # conv 1 intercepts
K2 = (torch.randn(32, 16, 3, 3) * (2 / n2) ** 0.5).requires_grad_(True)   # conv 2 stencils
beta2 = torch.zeros(32).requires_grad_(True)                              # conv 2 intercepts
V = (torch.randn(128, n3) * (2 / n3) ** 0.5).requires_grad_(True)         # detector weights
c = torch.zeros(128).requires_grad_(True)                                 # detector intercepts
W = (torch.randn(100, n4) * (2 / n4) ** 0.5).requires_grad_(True)         # score weights
b = torch.zeros(100).requires_grad_(True)                                 # score intercepts

params = [K1, beta1, K2, beta2, V, c, W, b]                               # everything gradient descent updates


# ---------- Part 1: overfit one batch (sanity check) ----------
x, y = next(iter(loader))     # x = images (64 x 3 x 64 x 64), y = true jersey numbers (64)

for step in range(201):
    z1 = F.conv2d(x, K1, beta1, padding=1)    # 64 x 16 x 64 x 64   conv 1
    a1 = torch.relu(z1)                       #                     ReLU
    p1 = F.max_pool2d(a1, 2)                  # 64 x 16 x 32 x 32   pool

    z2 = F.conv2d(p1, K2, beta2, padding=1)   # 64 x 32 x 32 x 32   conv 2
    a2 = torch.relu(z2)                       #                     ReLU
    p2 = F.max_pool2d(a2, 2)                  # 64 x 32 x 16 x 16   pool

    f = p2.view(-1, 8192)                     # 64 x 8192           flatten
    h = torch.relu(f @ V.T + c)               # 64 x 128            detectors
    s = h @ W.T + b                           # 64 x 100            scores

    loss = F.cross_entropy(s, y)              # loss = mean over batch of -log p_y
    loss.backward()                           # d(loss)/d(theta) for every theta in params

    with torch.no_grad():
        for p in params:
            p -= 0.01 * p.grad                # theta <- theta - 0.01 * d(loss)/d(theta)
            p.grad.zero_()                    # reset gradients

    if step % 20 == 0:
        print("step:", step, "loss:", loss.item())


# ---------- Accuracy ----------
def accuracy(data_loader):
    correct, total = 0, 0
    with torch.no_grad():                         # just predicting, no gradients needed
        for x, y in data_loader:                  # x = images (B x 3 x 64 x 64), y = true jersey numbers (B)
            z1 = F.conv2d(x, K1, beta1, padding=1)
            a1 = torch.relu(z1)
            p1 = F.max_pool2d(a1, 2)

            z2 = F.conv2d(p1, K2, beta2, padding=1)
            a2 = torch.relu(z2)
            p2 = F.max_pool2d(a2, 2)

            f = p2.view(-1, 8192)
            h = torch.relu(f @ V.T + c)
            s = h @ W.T + b                       # scores (B x 100)

            preds = s.argmax(dim=1)               # guess = jersey number with the highest score
            correct += (preds == y).sum().item()  # count guesses that match the truth
            total += len(y)                       # count images seen
    return correct / total                        # fraction correct

print("Test accuracy before full training:", accuracy(test_loader))


# ---------- Part 2: train on all training images ----------
for epoch in range(10):                            # start small: check how long one epoch takes
    for x, y in loader:                           # batches of 64: x is 64 x 3 x 64 x 64, y is 64
        z1 = F.conv2d(x, K1, beta1, padding=1)    # 64 x 16 x 64 x 64   conv 1
        a1 = torch.relu(z1)                       #                     ReLU
        p1 = F.max_pool2d(a1, 2)                  # 64 x 16 x 32 x 32   pool

        z2 = F.conv2d(p1, K2, beta2, padding=1)   # 64 x 32 x 32 x 32   conv 2
        a2 = torch.relu(z2)                       #                     ReLU
        p2 = F.max_pool2d(a2, 2)                  # 64 x 32 x 16 x 16   pool

        f = p2.view(-1, 8192)                     # 64 x 8192           flatten
        h = torch.relu(f @ V.T + c)               # 64 x 128            detectors
        s = h @ W.T + b                           # 64 x 100            scores

        loss = F.cross_entropy(s, y)              # loss = (1/64) * sum_n -log p_{y_n}
        loss.backward()                           # d(loss)/d(theta) for every theta in params

        with torch.no_grad():
            for p in params:
                p -= 0.01 * p.grad                # theta <- theta - 0.01 * d(loss)/d(theta)
                p.grad.zero_()                    # reset gradients

    print(f"Epoch {epoch}: last batch loss {loss.item():.3f}, "
          f"test accuracy {accuracy(test_loader):.3f}, "
          f"train accuracy {accuracy(loader):.3f}")