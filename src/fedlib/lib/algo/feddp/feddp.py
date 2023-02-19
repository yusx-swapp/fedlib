import torch
from torch import nn
# from torchvision import transforms
from ....utils import get_logger
from ..base import BaseTrainer
import numpy as np
# logger = get_logger()

class Trainer(BaseTrainer):
    def __init__(self,logger=None) -> None:
        super().__init__()
        self.logger = logger

    # pruning_threshold
    def train(self, model:nn.Module, dataloader , criterion, optimizer, scheduler, local_epochs:int, pruning_threshold:float, device):
        """training an autoencoder 

        Args:
            model (nn.Module): _description_
            decoder (nn.Modules): _description_
            dataloader (_type_): _description_
            criterion (_type_): _description_
            optimizer (_type_): _description_
            scheduler (_type_): _description_
            epochs (int): _description_
            device (_type_): _description_
        """
        
        model.to(device)
        model.train()
        epoch_loss = []

        for epoch in range(local_epochs):
            correct = 0
            total = 0
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(dataloader):
                x, labels = x.to(device), labels.to(device)
                
                
  
                
                pred = model(x)
                loss = criterion(pred, labels)

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
                
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            scheduler.step()
            model_sparsity = self.dynamic_pruning(model, threshold=pruning_threshold)

            epoch_loss.append(sum(batch_loss) / len(batch_loss) if batch_loss else 0)
            
            accuracy = correct / total

            # print('Epoch: {}\tLoss: {:.6f}\tAccuracy:{:.6f}'.format(epoch, sum(epoch_loss) / len(epoch_loss), accuracy))
            if self.logger is not None:
                self.logger.info('Epoch: {}\tLoss: {:.6f}\tAccuracy:{:.6f}\tModel sparsity: {:.2f}%'.format(
                    epoch, sum(epoch_loss) / len(epoch_loss), accuracy,model_sparsity*100))
        
        


    def aggregate(self, **kwargs):        
            """fedavg aggregation
            kwargs:
                nets_params: 
                local_datasize:
                global_para: 

            Returns:
                global_para: _description_
            """
            nets_params = kwargs["nets_params"]
            local_datasize = kwargs["local_datasize"]
            global_model_param = kwargs["global_model_param"]

            total_data_points = sum(local_datasize)
            fed_avg_freqs = [size/ total_data_points for size in local_datasize]
            
            
            for idx, net_para in enumerate(nets_params):
                if idx == 0:
                    for key in net_para:
                        global_model_param[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_model_param[key] += net_para[key] * fed_avg_freqs[idx]

            return global_model_param


    def dynamic_pruning(self,model:nn.Module, threshold=1e-3):
        # Prune all the conv layers of the model
        layer_wise_sparsity = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = torch.abs(module.weight) > threshold
                module.weight.data[~mask] = 0
                module_sparsity = 1 - float(torch.sum(module.weight.data != 0)) / float(module.weight.data.nelement())
                layer_wise_sparsity.append(module_sparsity)
                # print('Layer: {}, Sparsity: {:.2f}%'.format(name, module_sparsity*100))

                if self.logger:
                    self.logger.info('Layer: {}, Sparsity: {:.2f}%'.format(name, module_sparsity*100))
        
        model_sparsity = np.mean(layer_wise_sparsity)
        # print('Model sparsity: {:.2f}%'.format(model_sparsity*100))

        if self.logger:
            self.logger.info('Model sparsity: {:.2f}%'.format(model_sparsity*100))
        return model_sparsity

    def test(self, model, test_data, device):

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
            "test_accuracy":0,
        }

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        # if args.dataset == "stackoverflow_lr":
        #     criterion = nn.BCELoss(reduction="sum").to(device)
        # else:
            # criterion = nn.CrossEntropyLoss().to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()
                metrics["test_correct"] += correct.item()
                
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        metrics["test_accuracy"] = metrics["test_correct"] / len(test_data.dataset)
        return metrics



