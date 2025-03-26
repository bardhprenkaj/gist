import numpy as np
import torch
import tqdm

from src.core.oracle_base import Oracle
from src.core.torch_base import TorchBase
from src.dataset.utils.dataset_torch import TorchGeometricDataset

class OracleTorch(TorchBase, Oracle):
                                
            
    def real_fit(self):
        super().real_fit()
        self.evaluate(self.dataset, fold_id=self.fold_id)
            
    @torch.no_grad()
    def evaluate(self, dataset, fold_id=0):            
        loader = dataset.get_torch_loader(fold_id=fold_id, batch_size=self.batch_size, usage='test')
        
        losses = []
        labels_list, preds = [], []
        for batch in tqdm.tqdm(loader):
            batch.batch = batch.batch.to(self.device)
            node_features = batch.x.to(self.device).to(torch.float)
            edge_index = batch.edge_index.to(self.device).to(torch.int64)
            edge_weights = batch.edge_attr.to(self.device).to(torch.float)
            labels = batch.y.to(self.device).long()
            
            self.optimizer.zero_grad()  
            pred, loss = self.model.fwd(node_features, edge_index, edge_weights, batch.batch, labels, self.loss_fn)
            losses.append(loss.to('cpu').detach().numpy())
            
            try:
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())
            except TypeError:
                labels_list += [labels.squeeze().long().detach().to('cpu').numpy()]
                preds += [pred.squeeze().detach().to('cpu').numpy()]
            
        accuracy = self.accuracy(labels_list, preds)
        self.context.logger.info(f'Test accuracy = {np.mean(accuracy):.4f}')


    def _real_predict(self, data_instance):
        return torch.argmax(self._real_predict_proba(data_instance)).item()

    @torch.no_grad()
    def _real_predict_proba(self, data_inst):
        data_inst = TorchGeometricDataset.to_geometric(data_inst)
        node_features = data_inst.x.to(self.device).to(torch.float)
        edge_index = data_inst.edge_index.to(self.device).to(torch.int64)
        edge_weights = data_inst.edge_attr.to(self.device).to(torch.float)
        try:
            if len(node_features.shape) == 1:
                node_features = node_features.unsqueeze(-1)

            return self.model(node_features,edge_index,edge_weights, None).cpu().squeeze()
        except IndexError:
            print(node_features.shape)
            print(edge_index.shape)
            print(edge_weights.shape)
            return self.model(node_features, edge_index, edge_weights, None).cpu().squeeze()
                     
    def check_configuration(self):#TODO: revise configuration
        super().check_configuration()
        local_config = self.local_config

        if 'model' not in local_config['parameters']:
            local_config['parameters']['model'] = {
                'class': "src.oracle.nn.gcn.DownstreamGCN",
                "parameters" : {}
            }

        # set defaults
        local_config['parameters']['model']['parameters']['node_features'] = self.dataset.num_node_features()
        local_config['parameters']['model']['parameters']['n_classes'] = self.dataset.num_classes