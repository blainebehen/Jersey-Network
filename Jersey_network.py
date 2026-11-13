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
            image = self.transform(image)              # PIL image -> 3 x 64 x 64 tensor, values 0-1

        return image, label
    
train_data = JerseyDataset(df_train, img_dir, transform=transforms.ToTensor())
test_data = JerseyDataset(df_test, img_dir, transform=transforms.ToTensor())



loader = DataLoader(train_data, batch_size=64, shuffle=True)


test_loader = DataLoader(test_data, batch_size=1000)


# ---------- Parameters ----------
V = (torch.randn(64, 12288) * 0.01).requires_grad_(True)   # v_{j,i}: detector j's weight on pixel i
c = torch.zeros(64).requires_grad_(True)                 # c_j: detector j's intercept
W = (torch.randn(100, 64) * 0.01).requires_grad_(True)    # w_{k,j}: jersey number k's weight on detector j
b = torch.zeros(100).requires_grad_(True)                 # b_k: jersey number k's intercept


# ---------- Part 1: overfit one batch (sanity check) ----------
x, y = next(iter(loader))     # x = images, y = true jersey number 
x = x.view(-1, 12288)           # x_i for each image: 64 x 12288

for step in range(201):
    t = x @ V.T + c           # t_j = sum_i v_{j,i} * x_i + c_j           (64 x 64)
    h = torch.relu(t)         # h_j = max(0, t_j)                          (64 x 64)
    s = h @ W.T + b           # s_k = sum_j w_{k,j} * h_j + b_k            (64 x 100)
    loss = F.cross_entropy(s, y)
                              # p_k = e^{s_k} / sum_m e^{s_m}
                              # loss = mean over batch of -log p_y
    loss.backward()           # computes d(loss)/d(theta) for theta in V, c, W, b

    with torch.no_grad():
        for p in [V, c, W, b]:
            p -= 0.01 * p.grad # theta <- theta - 0.1 * d(loss)/d(theta)
            p.grad.zero_()    # reset gradients (PyTorch adds them up otherwise)

    if step % 20 == 0:
        print("step:", step, "loss:", loss.item())
        

# ---------- Accuracy on unseen images ----------
def accuracy(data_loader):
    correct, total = 0, 0
    with torch.no_grad():                      # just predicting, no gradients needed
        for x, y in data_loader:               # x = images (1000 x 3 x 64 x 64), y = true jersey number (1000)
            x = x.view(-1, 12288)              # x_i for each image                  (1000 x 12288)
            t = x @ V.T + c                    # t_j = sum_i v_{j,i} * x_i + c_j     (1000 x 64)
            h = torch.relu(t)                  # h_j = max(0, t_j)                    (1000 x 64)
            s = h @ W.T + b                    # s_k = sum_j w_{k,j} * h_j + b_k      (1000 x 100)
            preds = s.argmax(dim=1)            # guess = jersey number k with the highest s_k (1000)
            correct += (preds == y).sum().item()   # count guesses that match the truth
            total += len(y)                    # count images seen (+1000 per batch)
    return correct / total                     # fraction correct

print("Test accuracy before full training:", accuracy(test_loader))


# ---------- Part 2: train on all training images ----------
for epoch in range(10):                         # epoch = one full pass through all training images
    for x, y in loader:                        # batches of 64: x is 64 x 3 x 64 x 64, y is 64
        x = x.view(-1, 12288)                  # x_i for each image                  (64 x 12288)
        t = x @ V.T + c                        # t_j = sum_i v_{j,i} * x_i + c_j     (64 x 64)
        h = torch.relu(t)                      # h_j = max(0, t_j)                    (64 x 64)
        s = h @ W.T + b                        # s_k = sum_j w_{k,j} * h_j + b_k      (64 x 100)
        loss = F.cross_entropy(s, y)           # loss = (1/64) * sum_n -log p_{y_n}   (one average)
        loss.backward()                        # computes d(loss)/d(theta) for theta in V, c, W, b

        with torch.no_grad():
            for p in [V, c, W, b]:
                p -= 0.01 * p.grad              # theta <- theta - 0.1 * d(loss)/d(theta)
                p.grad.zero_()                 # reset gradients

    print(f"Epoch {epoch}: last batch loss {loss.item():.3f}, "
          f"test accuracy {accuracy(test_loader):.3f}")