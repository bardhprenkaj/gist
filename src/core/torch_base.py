from typing import Tuple, Union
import numpy as np
import random
import torch
import tqdm

from src.core.trainable_base import Trainable
from src.utils.cfg_utils import init_dflts_to_of
from src.core.factory_base import get_instance_kvargs
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

class TorchBase(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        self.batch_size = self.local_config['parameters']['batch_size']
        
        self.model: Union[Trainable, torch.nn.Module] = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                                                            self.local_config['parameters']['model']['parameters'])
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=10, factor=0.1)
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device) 
        
        self.patience = 0                 
    
    def real_fit(self):
        instances = self.dataset.get_torch_instances(fold_id=self.fold_id)
        train_loader, val_loader = self.__get_loaders(instances)
        
        best_val_loss =float('inf')
        interrupt: bool = False

        for epoch in tqdm.tqdm(range(self.epochs)):
            self.model.train()
            losses, preds, labels_list = self.fwd(train_loader)
            accuracy = self.accuracy(labels_list, preds)
            log_msg = f'Epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}'
            self.lr_scheduler.step(np.mean(losses))
            
            # check if we need to do early stopping
            if len(val_loader) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_losses, val_preds, val_labels = self.fwd(val_loader, val=True)
                    val_loss = np.mean(val_losses)
                    accuracy = self.accuracy(val_labels, val_preds)
                    log_msg += f'\tvar_loss = {val_loss:.4f}\t var_accuracy = {accuracy:.4f}'
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.patience = 0
                else:
                    self.patience += 1
                    
                if self.early_stopping_threshold and self.patience == self.early_stopping_threshold:
                    log_msg += f"Early stopped training at epoch {epoch}"
                    interrupt = True  # terminate the training loop

            self.context.logger.info(log_msg)
            if interrupt: break

    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 200)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')
        
    def accuracy(self, testy, probs):
        acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc

    def read(self):
        super().read()
        if isinstance(self.model, list):
            for mod in self.model:
                mod.to(self.device)
        else:
            self.model.to(self.device)
            
    def to(self, device):
        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        elif isinstance(self.model, list):
            for model in self.model:
                if isinstance(model, torch.nn.Module):
                    model.to(self.device)


    def fwd(self, loader, val=False):
        losses, preds, labels_list = [], [], []
        for batch in loader:
            batch.batch = batch.batch.to(self.device)
            node_features = batch.x.to(self.device).to(torch.float)
            edge_index = batch.edge_index.to(self.device).to(torch.int64)
            edge_weights = batch.edge_attr.to(self.device).to(torch.float)
            labels = batch.y.to(self.device).long()
            
            if not val:
                self.optimizer.zero_grad()
            
            pred, loss = self.model.fwd(node_features, edge_index, edge_weights, batch.batch, labels, self.loss_fn)
            losses.append(loss.to('cpu').detach().numpy())
            if not val:
                loss.backward()
                self.optimizer.step()
            
            labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
            preds += list(pred.squeeze().detach().to('cpu').numpy())
        
        return losses, preds, labels_list


    def __get_loaders(self, instances) -> Tuple[DataLoader, DataLoader]:
        train_loader, val_loader = None, None
        if self.early_stopping_threshold:
            num_instances = len(instances)
            # get 5% of training instances and reserve them for validation
            indices = list(range(num_instances))
            random.shuffle(indices)
            val_size = int(.05 * len(indices))
            train_size = len(indices) - val_size
            # get the training instances
            train_instances = Subset(instances, indices[:train_size - 1])
            val_instances = Subset(instances, indices[train_size:])
            # get the train and validation loaders
            train_loader = DataLoader(train_instances, batch_size=min(self.batch_size, train_size), shuffle=True, drop_last=True)
            val_loader = DataLoader(val_instances, batch_size=min(self.batch_size, val_size), shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(instances, batch_size=min(self.batch_size, train_size), shuffle=True, drop_last=True)

        return train_loader, val_loader