import argparse
import json
import os
from distutils.util import strtobool

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score

from dataset import NewsDatasetModule
from model import SLPNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="exp/tmp")
    parser.add_argument("--train_epochs", default=10, type=int)
    parser.add_argument("--early_stop", type=strtobool)
    parser.add_argument("--dataset_dir", default="./dataset")
    
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    
    model = SLPNet(300, 4)

    # Define scheduler
    scheduler = None
    
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    # Define Loss function.nn.CrossEntropyLoss(out, y_train[0])
    loss_func = nn.CrossEntropyLoss()

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if args.early_stop:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    #Define dataset
    dm = NewsDatasetModule(args.dataset_dir)

    #Train
    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto"
    )
    trainer.fit(model, dm)


    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    model.load_state_dict(state_dict=state_dict["state_dict"])
    
    torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))



if __name__ == "__main__":
    #73
    main()