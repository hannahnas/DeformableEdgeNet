from decoder import ResNetDecoder
from encoder import ResNetEncoder
from skip_resnet import SkipResNet
import pytorch_lightning as pl
from criterions import L1_loss, masked_MAE, SSIM_score
from torch import optim
import torch


class InpaintDepthModel(pl.LightningModule):
    def __init__(self, deformable):
        super().__init__()
        
        self.encoder = ResNetEncoder(in_channels=4, deformable=deformable)
        # self.depth_encoder = ResNetEncoder(in_channels=1)
        self.decoder = ResNetDecoder(out_channels=1)


    def forward(self, batch):
        color = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        depth_in = (1 - mask) * depth
        
        x = torch.cat((color, depth_in), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
        

    def _get_reconstruction_loss(self, batch):
        
        depth_gt = batch['depth']
        depth_pred = self.forward(batch)

        l1_depth = L1_loss(depth_pred, depth_gt)

        return l1_depth

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('train_depth_loss', depth_loss)

        return depth_loss

    def validation_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('val_depth_loss', depth_loss)


    def test_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('test_depth_loss', depth_loss)


class SkipResNetModel(pl.LightningModule):
    def __init__(self, deformable, use_edges):
        super().__init__()
        
        self.skipnet = SkipResNet(deformable, use_edges)


    def forward(self, batch):

        return self.skipnet(batch)
        

    def _get_reconstruction_loss(self, batch):
        
        depth_gt = batch['depth']
        depth_pred = self.forward(batch)

        l1_depth = L1_loss(depth_pred, depth_gt)

        return l1_depth

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('train_depth_loss', depth_loss)

        return depth_loss

    def validation_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('val_depth_loss', depth_loss)


    def test_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('test_depth_loss', depth_loss)

        depth_gt = batch['depth']
        mask = batch['mask']
        with torch.no_grad():
            depth_pred = self.forward(batch)

        mae = masked_MAE(depth_pred, depth_gt, mask)
        ssim = SSIM_score(depth_pred, depth_gt, mask)

        self.log('test_masked_MAE', mae)
        self.log('ssim', ssim)