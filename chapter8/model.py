import pytorch_lightning as pl
import torch
from torch import nn


class SLPNet(pl.LightningModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        self.loss = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("loss", loss, prog_bar=True,)
        return {"loss":loss}
 
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        pred_label = torch.argmax(pred, dim=1)
        acc = torch.sum(y == pred_label) * 1.0/len(y)
        results = {"test_loss":loss, "test_acc": acc}
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return results
    
    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        print("Accuracy: ", avg_acc)
        return results
        