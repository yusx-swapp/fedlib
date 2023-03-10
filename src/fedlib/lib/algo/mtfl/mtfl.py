import torch
from torch import nn
from torchvision import transforms
from ....utils import get_logger
from ..base import BaseTrainer
logger = get_logger()

class Trainer(BaseTrainer):
    def __init__(self,logger=None) -> None:
        super().__init__()
        self.logger = logger

    # 
    def train(self, model:nn.Module, dataloader , criterion, optimizer, scheduler, local_epochs:int, device):
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
        criterion_pred = nn.CrossEntropyLoss()
        criterion_rep = nn.MSELoss()
        epoch_loss = []

        for epoch in range(local_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(dataloader):
                x, labels = x.to(device), labels.to(device)
                
                model.zero_grad()
                
        
                pred_out, decodes_out = model(x)

                loss = criterion_rep(decodes_out, x)

                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())
            
            scheduler.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss) if batch_loss else 0)

            if self.logger is not None:
                logger.info('Encoder Epoch: {}\tLoss: {:.6f}'.format(
                        epoch, sum(epoch_loss) / len(epoch_loss)))
                
                scheduler.step()
        
        for param in model.encoder.parameters():
            param.requires_grad = False
        for epoch in range(local_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(dataloader):
                x, labels = x.to(device), labels.to(device)
                
                model.zero_grad()
                
                pred_out, decodes_out = model(x)

                loss = criterion_pred(pred_out, labels)

                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())
            
            scheduler.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss) if batch_loss else 0)
            


            if self.logger is not None:
                logger.info('Predictor Epoch: {}\tLoss: {:.6f}'.format(
                    epoch, sum(epoch_loss) / len(epoch_loss)))
            # scheduler.step()

        for param in model.encoder.parameters():
            param.requires_grad = True



            #TODO @Sixing: Need add an argument `save_dir` to save the decoder img
            # if epoch % 10 == 0:
            #     pic = self._to_img(output.cpu().data)
            #     from torchvision.utils import save_image
            #     save_image(pic, './dc_img/image_{}.png'.format(epoch))
        


    def aggregate(self, nets_params,local_datasize, global_model_param ):        
            """fedavg aggregation
            kwargs:
                nets_encoders: 
                local_datasize:
                globa_encoder: 

            Returns:
                globa_encoder: _description_
            """
            # nets_params = kwargs["nets_params"]
            # local_datasize = kwargs["local_datasize"]
            # global_model_param = kwargs["global_model_param"]

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
    
    def test_decoder(self, model, test_data, device):

        model.to(device)
        model.eval()

        criterion = torch.nn.MSELoss()
        total_loss = 0
        batches = 0
        with torch.no_grad():
            for _, (x, _) in enumerate(test_data):
                
                x = x.to(device)
                _, x_ = model(x)
                loss = criterion(x_, x)
                total_loss += loss
                batches += 1

        return total_loss

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
            "decoder_loss":0
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
        decoder_criterion = torch.nn.MSELoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                
                x = x.to(device)
                target = target.to(device)
                pred, x_ = model(x)
                
                loss = criterion(pred, target)
                loss_decoder = decoder_criterion(x_,x)
                
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()
                metrics["test_correct"] += correct.item()
                
                metrics['decoder_loss'] +=loss_decoder

                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        metrics["test_accuracy"] = metrics["test_correct"] / len(test_data.dataset)
        return metrics

    def _to_img(self, img, transform = None):
        if transforms is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.nn.functional.pad(
                    torch.autograd.variable.Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

            ])
        return transform(img)

