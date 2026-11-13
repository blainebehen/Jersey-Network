#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:37:52 2026

@author: blainebehen
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
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



