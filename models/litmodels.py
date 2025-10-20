import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from io import BytesIO
from PIL import Image
from .losses import SupervisedSimCLRLoss, MMCRLoss
import sys
from utils.plotting import make_corner
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import h5py

class SimCLRModel(pl.LightningModule):
    def __init__(self, encoder, projector, temperature=0.1, sup_simclr=False,
                 classifier=None, lambda_classifier=1.0, pretrain_ckpt=None, visualization_dim=None,
                 sim_metric='cos', label_noise=0.0, label_set=[], save_plots=True, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.simclr_criterion = SupervisedSimCLRLoss(temperature=temperature, sim_metric=sim_metric)
        self.sup_simclr = sup_simclr
        self.classifier = classifier
        self.lambda_classifier = lambda_classifier
        self.val_outputs = []
        self.visualization_dim = visualization_dim
        self.label_noise = label_noise
        self.label_set = label_set
        self.save_plots = save_plots
        if self.label_noise > 0.0:
            assert len(self.label_set) > 0, "Label set must be provided for label noise."
            print("Label noise enabled with set:", self.label_set)
        #print(self.encoder)

        if pretrain_ckpt is not None:
            self.load_state_dict(torch.load(pretrain_ckpt)['state_dict'])
        self.save_hyperparameters()

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
    
    def noise_labels(self, labels):
        N = labels.shape[0]
        num_labels = len(self.label_set)
        p_sel = self.label_noise * (num_labels/(num_labels - 1)) # correct for probability of noising to the same label
        mask = torch.rand(N) < p_sel
        if len(labels.shape) > 1:
            labels[mask] = torch.tensor(np.random.choice(self.label_set, size=mask.sum().item(), replace=True).reshape(-1,1)).to(labels)
        else:
            labels[mask] = torch.tensor(np.random.choice(self.label_set, size=mask.sum().item(), replace=True)).to(labels)
        return labels

    def evaluate_loss(self,batch,validation=False):
        x, labels = batch   

        if self.label_noise > 0.0:
            labels = self.noise_labels(labels)

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
        if self.sup_simclr and self.save_plots:
            preds = np.concatenate([o[1] for o in self.val_outputs],axis=0)
            labels = np.concatenate([o[2] for o in self.val_outputs],axis=0)
            if self.visualization_dim is not None and self.visualization_dim < preds.shape[1]:
                # Compute PCA to reduce dimensionality for visualization
                pca = PCA(n_components=self.visualization_dim)
                preds_viz = pca.fit_transform(preds)
            else:
                preds_viz = preds
            fig = make_corner(preds_viz,labels,return_fig=True)
            buf = BytesIO()
            fig.savefig(buf,format='jpg',dpi=200)
            buf.seek(0)
            self.logger.log_image(
                'val/space',
                [Image.open(buf)],
            )
            plt.close(fig)
            self.val_outputs.clear()
            # Evaluate kNN classifier on embeddings
            """
            # Split data for kNN evaluation
            n_samples = preds.shape[0]
            if n_samples > 1000:  # For efficiency with large datasets
                n_eval = 1000
                indices = np.random.choice(n_samples, n_eval, replace=False)
                eval_preds = preds[indices]
                eval_labels = labels[indices]
            else:
                eval_preds = preds
                eval_labels = labels

            # Train and evaluate kNN classifier
            k_values = [5, 10, 20]
            knn_scores = {}
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                # Use 5-fold cross-validation
                scores = cross_val_score(knn, eval_preds, eval_labels, cv=5)
                knn_scores[k] = scores.mean()
                self.log(f"val/knn_{k}_accuracy", scores.mean(), on_step=False, on_epoch=True)

            # Log best kNN performance
            best_k = max(knn_scores, key=knn_scores.get)
            self.log("val/knn_best_accuracy", knn_scores[best_k], on_step=False, on_epoch=True)
            self.log("val/knn_best_k", best_k, on_step=False, on_epoch=True)
            """

class JetClassSimCLRModel(SimCLRModel):
    """
    Need special treatment for jetclass because we're repurposing the dataloader/data config structure from weaver,
    and its outputs are particular.
    """
    def evaluate_loss(self,batch,validation=False):
        x, labels, observers = batch
        labels = labels['_label_']

        if self.label_noise > 0.0:
            labels = self.noise_labels(labels)

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

class mmcrModel(pl.LightningModule):
    def __init__(self, encoder, projector, num_views, lmbda, pretrain_ckpt=None, visualization_dim=None,
                 label_noise=0.0, label_set=[], save_plots=True, 
                 optimizer=None, scheduler=None, opt_params={}, scheduler_params={},
                 **kwargs):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.num_views = num_views
        self.mmcr_criterion = MMCRLoss(n_views=num_views, lmbda=lmbda)
        self.val_outputs = []
        self.visualization_dim = visualization_dim
        self.label_noise = label_noise
        self.label_set = label_set
        self.save_plots = save_plots
        if self.label_noise > 0.0:
            assert len(self.label_set) > 0, "Label set must be provided for label noise."
            print("Label noise enabled with set:", self.label_set)
        #print(self.encoder)

        # lists to save test embeddings
        self.test_embeddings = []
        self.test_labels = []

        if pretrain_ckpt is not None:
            self.load_state_dict(torch.load(pretrain_ckpt)['state_dict'])
        self.save_hyperparameters()

        print(f"Initialized mmcrModel with num_views={num_views}, lmbda={lmbda}")
    
    def forward(self, x, embed=False):
        z = self.encoder(x)
        if embed:
            return z
        h = self.projector(z)
        return h

    def evaluate_loss(self,batch,validation=False):
        x, labels = batch

        if self.label_noise > 0.0:
            labels = self.noise_labels(labels)

        # assume x has shape (B,N,F) where B is batch size, N is number of views, F is feature dimension
        x = x.view(-1, x.shape[-1])  # Flatten to (batch_size * n_views, feature_dim)
        z = self.encoder(x)
        h = self.projector(z)
        loss = self.mmcr_criterion(h)

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
        if self.save_plots:
            preds = np.concatenate([o[1] for o in self.val_outputs],axis=0)
            labels = np.concatenate([o[2] for o in self.val_outputs],axis=0)
            if self.visualization_dim is not None and self.visualization_dim < preds.shape[1]:
                # Compute PCA to reduce dimensionality for visualization
                pca = PCA(n_components=self.visualization_dim)
                preds_viz = pca.fit_transform(preds)
            else:
                preds_viz = preds
            fig = make_corner(preds_viz,labels,return_fig=True)
            buf = BytesIO()
            fig.savefig(buf,format='jpg',dpi=200)
            buf.seek(0)
            self.logger.log_image(
                'val/space',
                [Image.open(buf)],
            )
            plt.close(fig)
            self.val_outputs.clear()

class JetCLRMMCRModel(mmcrModel):
    """
    Need special treatment for jetclass because we're repurposing the dataloader/data config structure from weaver,
    and its outputs are particular.
    """
    def evaluate_loss(self,batch,validation=False):
        projections = []
        embeddings = None
        for i in range(self.num_views):
            x,labels,observers = batch[i]
            labels = labels['_label_']
            z = self.encoder(x)
            if i==0:
                embeddings = z.detach().cpu().numpy()
            h = self.projector(z)
            projections.append(h.unsqueeze(1))
        projections = torch.cat(projections, dim=1) # (B, N, F) where B is batch size, N is number of views, F is feature dimension
        loss = self.mmcr_criterion(projections)
        
        if validation:
            # Store outputs for validation visualization
            self.val_outputs.append((loss.item(), embeddings, labels.cpu().numpy()))

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(list(self.encoder.parameters())+list(self.projector.parameters()), **self.hparams.opt_params)
        scheduler = self.hparams.scheduler(optimizer, **self.hparams.scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    def test_step(self,batch, batch_idx):
        x, labels, observers = batch
        labels = labels['_label_'].cpu().numpy()
        with torch.no_grad():
            z = self.encoder(x).detach().cpu().numpy()
        self.test_embeddings.append(z)
        self.test_labels.append(labels)
    
    def on_test_epoch_end(self):
        embeddings = np.concatenate(self.test_embeddings, axis=0)
        labels = np.concatenate(self.test_labels, axis=0)
        if len(labels.shape) > 0:
            labels = labels.flatten()
        output_file = self.trainer.default_root_dir + "/test_embeddings.h5"
        with h5py.File(output_file,"a") as fout:
            if "embeddings" not in fout.keys():
                fout.create_dataset("embeddings", shape=embeddings.shape, data=embeddings, maxshape=(None, embeddings.shape[1]))
            else:
                fout["embeddings"].resize((fout["embeddings"].shape[0] + embeddings.shape[0]), axis=0)
                fout["embeddings"][-embeddings.shape[0]:] = embeddings
            
            if "labels" not in fout.keys():
                fout.create_dataset("labels", shape=labels.shape, data=labels, maxshape=(None,))
            else:
                fout["labels"].resize((fout["labels"].shape[0] + labels.shape[0]), axis=0)
                fout["labels"][-labels.shape[0]:] = labels

class JetClassSupMMCRModel(JetCLRMMCRModel):
    def __init__(self,samples_per_class,views_per_class,**kwargs):
        super().__init__(**kwargs)
        self.samples_per_class = samples_per_class
        self.views_per_class = views_per_class

    def evaluate_loss(self,batch,validation=False):
        
        x,labels,observers = batch
        z = self.encoder(x)
        h = self.projector(z)

        labels = labels['_label_']
        uniq_labels, counts = torch.unique(labels, return_counts=True)
        projections = torch.cat([h[labels==l].unsqueeze(1).reshape(self.samples_per_class,self.views_per_class,h.shape[-1]) for l in uniq_labels], dim=0) # (B, N, F) where B is batch size, N is number of classes, F is feature dimension
        loss = self.mmcr_criterion(projections)
        
        if validation:
            # Store outputs for validation visualization
            self.val_outputs.append((loss.item(), h.detach().cpu().numpy(), labels.cpu().numpy()))

        return loss

class ClassifierModel(pl.LightningModule):
    def __init__(self, network, num_classes, name, optimizer=None, scheduler=None, 
                    opt_params={}, scheduler_params={}, **kwargs):
        super().__init__()
        self.network = network
        self.num_classes = num_classes
        self.name = name
        self.criterion = nn.CrossEntropyLoss()
        self.test_evals = []
        self.test_labels = []
        self.test_class_labels = []
        
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, labels, labels_class = batch
        logits = self.forward(x)
        loss = self.criterion(logits, labels_class)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels_class).float().mean()
        
        self.log(f"train_class_{self.name}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train_class_{self.name}/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels, labels_class = batch
        logits = self.forward(x)
        loss = self.criterion(logits, labels_class)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log(f"val_class_{self.name}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val_class_{self.name}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels, labels_class = batch
        logits = self.forward(x)
        loss = self.criterion(logits, labels_class)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels_class).float().mean()

        self.log(f"test_class_{self.name}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"test_class_{self.name}/acc", acc, on_step=False, on_epoch=True)

        self.test_evals.append(logits.cpu().numpy())
        self.test_labels.append(labels.cpu().numpy())
        self.test_class_labels.append(labels_class.cpu().numpy())

        return loss
    
    def on_test_epoch_end(self):
        self.test_evals = np.concatenate(self.test_evals)
        self.test_labels = np.concatenate(self.test_labels)
        self.test_class_labels = np.concatenate(self.test_class_labels)
        output_file = self.trainer.default_root_dir + f"/test_class_{self.name}_results.h5"
        with h5py.File(output_file,"a") as fout:
            if "preds" not in fout.keys():
                fout.create_dataset("preds", shape=self.test_evals.shape, data=self.test_evals, maxshape=(None,self.test_evals.shape[1]))
            else:
                fout["preds"].resize((fout["preds"].shape[0] + self.test_evals.shape[0]), axis=0)
                fout["preds"][-self.test_evals.shape[0]:] = self.test_evals
            
            if "labels" not in fout.keys():
                fout.create_dataset("labels", shape=self.test_labels.shape, data=self.test_labels, maxshape=(None,))
            else:
                fout["labels"].resize((fout["labels"].shape[0] + self.test_labels.shape[0]), axis=0)
                fout["labels"][-self.test_labels.shape[0]:] = self.test_labels
            
            if "class_labels" not in fout.keys():
                fout.create_dataset("class_labels", shape=self.test_class_labels.shape, data=self.test_class_labels, maxshape=(None,))
            else:
                fout["class_labels"].resize((fout["class_labels"].shape[0] + self.test_class_labels.shape[0]), axis=0)
                fout["class_labels"][-self.test_class_labels.shape[0]:] = self.test_class_labels
    
    def configure_optimizers(self):
        if self.hparams.optimizer is not None:
            optimizer = self.hparams.optimizer(self.network.parameters(), **self.hparams.opt_params)
            if self.hparams.scheduler is not None:
                scheduler = self.hparams.scheduler(optimizer, **self.hparams.scheduler_params)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler
                }
            return optimizer
        else:
            return torch.optim.Adam(self.network.parameters(), lr=1e-3)

class NewSimCLRModel(pl.LightningModule):
    def __init__(self, encoder, projector, temperature=0.1, sup_simclr=False,
                 pretrain_ckpt=None, visualization_dim=None,
                 optimizer=None, scheduler=None, opt_params={}, scheduler_params={},
                 sim_metric='cos', label_noise=0.0, label_set=[], save_plots=True, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.simclr_criterion = SupervisedSimCLRLoss(temperature=temperature, sim_metric=sim_metric)
        self.sup_simclr = sup_simclr
        self.val_outputs = []
        self.visualization_dim = visualization_dim
        self.label_noise = label_noise
        self.label_set = label_set
        self.save_plots = save_plots
        if self.label_noise > 0.0:
            assert len(self.label_set) > 0, "Label set must be provided for label noise."
            print("Label noise enabled with set:", self.label_set)

        # lists to save test embeddings
        self.test_embeddings = []
        self.test_labels = []

        if pretrain_ckpt is not None:
            self.load_state_dict(torch.load(pretrain_ckpt)['state_dict'])
        self.save_hyperparameters()

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
    
    def noise_labels(self, labels):
        N = labels.shape[0]
        num_labels = len(self.label_set)
        p_sel = self.label_noise * (num_labels/(num_labels - 1)) # correct for probability of noising to the same label
        mask = torch.rand(N) < p_sel
        if len(labels.shape) > 1:
            labels[mask] = torch.tensor(np.random.choice(self.label_set, size=mask.sum().item(), replace=True).reshape(-1,1)).to(labels)
        else:
            labels[mask] = torch.tensor(np.random.choice(self.label_set, size=mask.sum().item(), replace=True)).to(labels)
        return labels

    def evaluate_loss(self,batch,validation=False):
        x, labels = batch   

        if self.label_noise > 0.0:
            labels = self.noise_labels(labels)

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
        if self.sup_simclr and self.save_plots:
            preds = np.concatenate([o[1] for o in self.val_outputs],axis=0)
            labels = np.concatenate([o[2] for o in self.val_outputs],axis=0)
            if self.visualization_dim is not None and self.visualization_dim < preds.shape[1]:
                # Compute PCA to reduce dimensionality for visualization
                pca = PCA(n_components=self.visualization_dim)
                preds_viz = pca.fit_transform(preds)
            else:
                preds_viz = preds
            fig = make_corner(preds_viz,labels,return_fig=True)
            buf = BytesIO()
            fig.savefig(buf,format='jpg',dpi=200)
            buf.seek(0)
            self.logger.log_image(
                'val/space',
                [Image.open(buf)],
            )
            plt.close(fig)
            self.val_outputs.clear()

class NewJetClassSimCLRModel(NewSimCLRModel):
    """
    Need special treatment for jetclass because we're repurposing the dataloader/data config structure from weaver,
    and its outputs are particular.
    """
    def evaluate_loss(self,batch,validation=False):
        if self.sup_simclr:
            x, labels, observers = batch
            labels = labels['_label_']
            if self.label_noise > 0.0:
                labels = self.noise_labels(labels)
            
            h = self.encoder(x)
            z = self.projector(h)
            z = F.normalize(z,dim=1).unsqueeze(1) # normalize the projection for simclr loss
            loss_simclr = self.simclr_criterion(z, labels=labels)
            if validation:
                self.val_outputs.append((loss_simclr.item(), h.cpu().numpy(), labels.cpu().numpy()))
        else:
            n_views = len(batch)
            projections = []
            for ib in range(n_views):
                x, labels, observers = batch[ib]
                h = self.projector(self.encoder(x))
                h = F.normalize(h,dim=1).unsqueeze(1)
                projections.append(h)
            projections = torch.cat(projections,dim=1) # (B, N, F) where B is batch size, N is number of views, F is feature dimension
            loss_simclr = self.simclr_criterion(projections, labels=None)

        loss = loss_simclr
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(list(self.encoder.parameters())+list(self.projector.parameters()), **self.hparams.opt_params)
        scheduler = self.hparams.scheduler(optimizer, **self.hparams.scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    def test_step(self,batch, batch_idx):
        x, labels, observers = batch
        labels = labels['_label_'].cpu().numpy()
        with torch.no_grad():
            z = self.encoder(x).detach().cpu().numpy()
        self.test_embeddings.append(z)
        self.test_labels.append(labels)
    
    def on_test_epoch_end(self):
        embeddings = np.concatenate(self.test_embeddings, axis=0)
        labels = np.concatenate(self.test_labels, axis=0)
        if len(labels.shape) > 0:
            labels = labels.flatten()
        output_file = self.trainer.default_root_dir + "/test_embeddings.h5"
        with h5py.File(output_file,"a") as fout:
            if "embeddings" not in fout.keys():
                fout.create_dataset("embeddings", shape=embeddings.shape, data=embeddings, maxshape=(None, embeddings.shape[1]))
            else:
                fout["embeddings"].resize((fout["embeddings"].shape[0] + embeddings.shape[0]), axis=0)
                fout["embeddings"][-embeddings.shape[0]:] = embeddings
            
            if "labels" not in fout.keys():
                fout.create_dataset("labels", shape=labels.shape, data=labels, maxshape=(None,))
            else:
                fout["labels"].resize((fout["labels"].shape[0] + labels.shape[0]), axis=0)
                fout["labels"][-labels.shape[0]:] = labels