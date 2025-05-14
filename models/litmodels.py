import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from io import BytesIO
from PIL import Image
from .losses import SupervisedSimCLRLoss,MMDLoss
import sys
from utils.plotting import make_corner
from torch.optim.lr_scheduler import CosineAnnealingLR

class SimCLRModel(pl.LightningModule):
    def __init__(self, encoder, projector, temperature=0.1, sup_simclr=False, sup_simclr_datamc=False,
                 classifier=None, shifter=None, lambda_classifier=1.0, lambda_MMD=0.01, MMD=False, pretrain_ckpt=None, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.simclr_criterion = SupervisedSimCLRLoss(temperature=temperature)
        self.sup_simclr = sup_simclr
        self.sup_simclr_datamc = sup_simclr_datamc
        self.classifier = classifier
        self.shifter    = shifter
        self.lambda_classifier = lambda_classifier
        self.val_outputs = []
        self.mmdLoss = MMDLoss()
        self.MMD=MMD
        self.lambda_MMD = lambda_MMD
        #print(self.encoder)

        if pretrain_ckpt is not None:
            self.load_state_dict(torch.load(pretrain_ckpt)['state_dict'])
        self.save_hyperparameters()
        #self.apply(self.init_weights)
        if self.shifter is not None:
            self.shifter.apply(self.init_weights_to_zero)

    def init_weights_to_zero(self,m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            #nn.init.constant_(m.weight, 0.0)
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4,weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
    
    def embed(self,x):
        return self.encoder(x)
    
    def project(self,h):
        return self.projector(h)
    
    def forward(self, x, embed=False):
        z = self.embed(x)
        if embed:
            return z
        h = self.project(z)
        return h
    
    def evaluate_loss(self,batch,validation=False):
        x, labels = batch

        if self.sup_simclr:
            h = self.encoder(x)
            z = self.projector(h)
            z = F.normalize(z,dim=1).unsqueeze(1) # normalize the projection for simclr loss
            loss_simclr = self.simclr_criterion(z, labels=labels)
            if validation:
                self.val_outputs.append((loss_simclr.item(), h.cpu().numpy(), labels.cpu().numpy()))
        elif self.sup_simclr_datamc:
            batch_data,label_class = x
            h = self.encoder(batch_data)
            h = torch.nn.functional.normalize(h,dim=1)
            z = self.projector(h)
            z = F.normalize(z,dim=1).unsqueeze(1) # normalize the projection for simclr loss
            #loss_simclr = self.simclr_criterion(z[labels == 0], labels=label_class[labels == 0])
            loss_simclr = self.simclr_criterion(z, labels=label_class)
            if self.MMD:
                mc   = (h[labels == 0])
                data = (h[labels == 1])
                lMmdloss = self.mmdLoss(data,mc)#*len(labels)
                loss_simclr = loss_simclr + lMmdloss*len(labels)*self.lambda_MMD
            if validation:
                self.val_outputs.append((loss_simclr.item(), h.cpu().numpy(), label_class.cpu().numpy()))            
        else:
            aug0, aug1 = x
            h0 = self.encoder(aug0)
            z0 = self.projector(h0)
            h1 = self.encoder(aug1)
            z1 = self.projector(h1)
            # compute simclr loss with normalized projections
            features = torch.cat([F.normalize(z0,dim=1).unsqueeze(1), F.normalize(z1,dim=1).unsqueeze(1)], dim=1)
            loss_simclr = self.simclr_criterion(features, labels=None)

        # compute supervised classifier loss if using
        if self.classifier is not None:
            if self.sup_simclr or self.sup_simclr_datamc:
                logits = self.classifier(h)
            else:
                logits = self.classifier(h0)
            if self.sup_simclr_datamc:
                _, labelsd = x
                logits = logits[labels == 0]
                labels = labelsd[labels == 0]
            loss_classifier = F.cross_entropy(logits, labels)
            loss = loss_simclr + self.lambda_classifier * loss_classifier
        else:
            loss = loss_simclr
        
        return loss
        
    def training_step(self, batch, batch_idx, log=True):
        loss = self.evaluate_loss(batch, validation=False)
        
        if log:
            self.log("train/loss",
                    loss,
                    on_step=True,
                    on_epoch=True,
                    reduce_fx='mean',
                    logger=True,
                    prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx, log=True):
        loss = self.evaluate_loss(batch, validation=True)

        if log:
            self.log("val/loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx='mean',
                    logger=True,
                    prog_bar=True)

        return loss


    def on_validation_epoch_end(self):
        if self.sup_simclr:
            preds = np.concatenate([o[1] for o in self.val_outputs],axis=0)
            labels = np.concatenate([o[2] for o in self.val_outputs],axis=0)
            fig = make_corner(preds,labels,return_fig=True)
            buf = BytesIO()
            fig.savefig(buf,format='jpg',dpi=200)
            buf.seek(0)
            #self.logger.log_image(
            #    'val/space',
            #    [Image.open(buf)],
            #)
            plt.close(fig)
            self.val_outputs.clear()
            
class JetClassSimCLRModel(SimCLRModel):
    """
    Need special treatment for jetclass because we're repurposing the dataloader/data config structure from weaver,
    and its outputs are particular.
    """
    def evaluate_loss(self,batch,validation=False):
        x, labels, observers = batch
        labels = labels['_label_']

        if self.sup_simclr:
            h = self.encoder(x)
            z = self.projector(h)
            z = F.normalize(z,dim=1).unsqueeze(1) # normalize the projection for simclr loss
            loss_simclr = self.simclr_criterion(z, labels=labels)
            if validation:
                self.val_outputs.append((loss_simclr.item(), h.cpu().numpy(), labels.cpu().numpy()))
        else:
            aug0, aug1 = x
            h0 = self.encoder(aug0)
            z0 = self.projector(h0)
            h1 = self.encoder(aug1)
            z1 = self.projector(h1)
            # compute simclr loss with normalized projections
            features = torch.cat([F.normalize(z0,dim=1).unsqueeze(1), F.normalize(z1,dim=1).unsqueeze(1)], dim=1)
            loss_simclr = self.simclr_criterion(features, labels=None)

        # compute supervised classifier loss if using
        if self.classifier is not None:
            if self.sup_simclr:
                logits = self.classifier(h)
            else:
                logits = self.classifier(h0)
            loss_classifier = F.cross_entropy(logits, labels)
            loss = loss_simclr + self.lambda_classifier * loss_classifier
        else:
            loss = loss_simclr
        
        return loss


class LossCollector(pl.Callback):
    def __init__(self):
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.losses.append(loss.detach().cpu().item())
