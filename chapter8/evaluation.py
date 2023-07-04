
import argparse

import torch
import pytorch_lightning as pl

from dataset import NewsDatasetModule
from model import SLPNet

parser = argparse.ArgumentParser()
parser.add_argument("--state_dict", default="./exp/tmp/best_model.pt")
args = parser.parse_args()

def evaluate():
    """Evaluate model"""

    #########################################################
    # Load model
    #########################################################
    
    print("Load Model...")
    print("From state-dict: ", args.state_dict)
    
    model = SLPNet(300, 4)
    sd = torch.load(args.state_dict)
    model.load_state_dict(sd, strict=False)
    
    #Define dataset
    dm = NewsDatasetModule()

    dm.prepare_data()
    dm.setup("test")

    #########################################################
    # Score
    #########################################################
    trainer = pl.Trainer()
    result = trainer.test(model, dataloaders=dm.test_dataloader())

    result = trainer.test(model, dataloaders=dm.train_dataloader())
    
if __name__ == "__main__":
    evaluate()