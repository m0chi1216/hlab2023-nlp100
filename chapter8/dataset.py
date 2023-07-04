import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class NewsDataset(Dataset):
    def __init__(self, X, y): 
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


class NewsDatasetModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        data_dir = self.data_dir
        self.X_train = torch.load(os.path.join(data_dir, "X_train.pt"))
        self.X_valid = torch.load(os.path.join(data_dir, "X_valid.pt"))
        self.X_test = torch.load(os.path.join(data_dir, "X_test.pt"))
        self.y_train = torch.load(os.path.join(data_dir, "y_train.pt"))
        self.y_valid = torch.load(os.path.join(data_dir, "y_valid.pt"))
        self.y_test = torch.load(os.path.join(data_dir, "y_test.pt"))
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = NewsDataset(self.X_train, self.y_train)
            self.valid_data = NewsDataset(self.X_valid, self.y_valid)
        if stage == "test" or stage is None:
            self.test_data = NewsDataset(self.X_test, self.y_test)
            #本来は不要
            self.train_data = NewsDataset(self.X_train, self.y_train)
            self.valid_data = NewsDataset(self.X_valid, self.y_valid)        

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True, pin_memory=True, num_workers=2)

