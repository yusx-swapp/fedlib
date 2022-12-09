import torch
from torch import nn

from ....utils import get_logger

logger = get_logger()

class Trainer:

    def train(self, **kwargs):

        model = kwargs["model"]
        device = kwargs["device"]
        criterion = kwargs["criterion"]
        optimizer = kwargs["optimizer"]
        epochs = kwargs["epochs"]
        dataloader = kwargs["dataloader"]
        
        model.to(device)
        model.train()

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(dataloader):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                optimizer.step()
                if batch_idx % 10 == 0:
                    logger.info('Update Epoch: {} \tLoss: {:.6f}'.format(
                        epoch,  loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logger.info('Epoch: {}\tLoss: {:.6f}'.format(
                epoch, sum(epoch_loss) / len(epoch_loss)))

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

    def test(self, test_data, device, args):
        
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        if args.dataset == "stackoverflow_lr":
            criterion = nn.BCELoss(reduction="sum").to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > 0.5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics["test_precision"] += precision.sum().item()
                    metrics["test_recall"] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        return metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:

        return False