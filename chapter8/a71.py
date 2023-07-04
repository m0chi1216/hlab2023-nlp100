import argparse
import json

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import NewsDataset
from model import SLPNet

if __name__ == "__main__":
    
    X_train = torch.load('data/X_train.pt')
    X_valid = torch.load('data/X_valid.pt')
    X_test = torch.load('data/X_test.pt')
    y_train = torch.load('data/y_train.pt')
    y_valid = torch.load('data/y_valid.pt')
    y_test = torch.load('data/y_test.pt')
    
    model = SLPNet(300, 4)

    #71
    y_hat_1 = torch.softmax(model(X_train[0]), dim=-1)
    print(y_hat_1)

    Y_hat = torch.softmax(model(X_train[:4]), dim=-1)
    print(Y_hat)

    #72
    criterion = nn.CrossEntropyLoss()

    out = model(X_train[0])
    l1 = criterion(out, y_train[0])
    model.zero_grad()
    l1.backward()
    
    print(f'損失: {l1:.4f}')
    print(f'勾配:\n{model.fc.weight.grad}')

    l = criterion(model(X_train[:4]), y_train[:4])
    model.zero_grad()
    l.backward()
    print(f'損失: {l:.4f}')
    print(f'勾配:\n{model.fc.weight.grad}')
   