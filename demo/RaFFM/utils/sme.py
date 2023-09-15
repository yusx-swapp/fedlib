# Implementation for High-performance Sub-Model Extraction

import torch
from torch.nn import Parameter
import random
import numpy as np
import copy
import torch
import time
__all__ = ['salient_submodel_extraction']

def calculate_trainable_params(masked_model):
    """calculate the number of trainable parameters in the model
    Args:
        masked_model: the model to be evaluated
    Returns:
        total_trainable_params: the number of trainable parameters in the model
        total_params: the number of parameters in the model
        percentage: the percentage of trainable parameters in the model
    """
    
    millions = 1000000
    total_trainable_params = 0
    total_params = 0
    for name, module in masked_model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, Parameter):
            original_channels = module.weight.size(0)
            mask = getattr(module, 'weight_mask', torch.ones_like(module.weight).to(module.weight.device))  # Retrieve the mask from the module
            
            # Calculate the number of trainable parameters by summing the mask
            trainable_params_in_module = torch.sum(mask).item()
            total_trainable_params += trainable_params_in_module
            total_params += torch.prod(torch.tensor(module.weight.size())).item()

    return total_trainable_params/millions, total_params/millions, total_trainable_params/total_params

def distilbert_module_handler(model,attn_channel_space,dense_channel_space,zero_fill):
    
    millions = 1000000

    module_identifier = 'MultiHeadSelfAttention' # Model Identifier for DistilBERT
    
    num_attn_head = model.config.num_attention_heads
    total_trainable_params = 0
    total_params = 0
    random.seed(time.time())
    subnetwork = copy.deepcopy(model).cpu()

    for name, module in subnetwork.named_modules():
            
        if module_identifier in str(type(module)):
            

            head_dim = module.q_lin.weight.size(0) // num_attn_head
            
            
            
            sampled_channels = random.choice(attn_channel_space)
            sampled_channels = min(sampled_channels, head_dim)

            mask = torch.cat(
                [torch.ones(sampled_channels), torch.zeros(head_dim - sampled_channels)]*num_attn_head
                                ).cuda()
            mask = mask.view(-1,1)
            

            # q, k 
            module.q_lin.weight.register_hook(lambda grad: grad * mask.to(grad.device) if grad is not None else None)
            module.k_lin.weight.register_hook(lambda grad: grad * mask.to(grad.device) if grad is not None else None)
            

            if zero_fill:
                module.q_lin.weight.data = module.q_lin.weight.data * mask
                module.k_lin.weight.data = module.k_lin.weight.data * mask
                                    
            
            # Calculate #Parameters for q,k
            trainable_params_in_module = torch.prod(torch.tensor((torch.sum(mask).item(),module.q_lin.weight.size(1)))).item()
            total_trainable_params += trainable_params_in_module * 2

            

            
            # # linear
            # linear_dim = module.out_lin.weight.size(0)
            # dense_sampled_channels = random.choice(dense_channel_space)
            # dense_sampled_channels = min(dense_sampled_channels, linear_dim)
            
            # mask = torch.cat([torch.ones(dense_sampled_channels),torch.zeros(linear_dim - dense_sampled_channels)], dim=0)
            # module.out_lin.weight.register_hook(lambda grad: grad * mask.to(grad.device) if grad is not None else None)





            # def mask_grad(grad):
            #     if grad is None:
            #         return None
            #     # Create a mask with the same shape as the gradient
            #     mask = torch.zeros_like(grad).to(grad.device)
            #     # Set the first dense_sampled_channels rows to ones
            #     mask.narrow(0, 0, dense_sampled_channels).fill_(1)
            #     return grad * mask

            # module.out_lin.weight.register_hook(mask_grad)
            # You can still store the mask in the module if needed, using the original weight shape
            # mask_for_module = torch.zeros_like(module.out_lin.weight)
            # mask_for_module[:dense_sampled_channels, :] = 1
            # module.weight_mask = mask_for_module

            # Calculate #Parameters for linear
            # trainable_params_in_module = torch.sum(mask_for_module).item()
            # total_trainable_params += trainable_params_in_module
            #Calculate #Parameters for v
            # total_trainable_params += torch.prod(torch.tensor((module.v_lin.weight.size(0),dense_sampled_channels))).item()

        if hasattr(module, 'weight') and isinstance(module.weight, Parameter) and module.weight.requires_grad:
            total_params += torch.prod(torch.tensor(module.weight.size())).item()
            # print(str(type(module)))
            
            if 'lin1' in name:
                # linear
                linear_dim = module.weight.size(0)
                dense_sampled_channels = random.choice(dense_channel_space)
                dense_sampled_channels = min(dense_sampled_channels, linear_dim)


                # module.weight.register_hook(mask_grad)
                mask_linear = torch.cat([torch.ones(dense_sampled_channels),torch.zeros(linear_dim - dense_sampled_channels)], dim=0)
                # module.weight.register_hook(lambda grad: grad * mask_linear.to(grad.device) if grad is not None else None)
                # You can still store the mask in the module if needed, using the original weight shape
                
                def gradient_mask_hook(grad):
                    if grad is None:
                        return None
                    diag_mask = torch.diag(mask_linear).to(grad.device)
                    return torch.mm(diag_mask, grad)

                module.weight.register_hook(gradient_mask_hook)
                
                mask_for_module = torch.zeros_like(module.weight)
                mask_for_module[:dense_sampled_channels, :] = 1
                # module.weight_mask = mask_for_module

                # Calculate #Parameters for linear
                trainable_params_in_module = torch.sum(mask_for_module).item()
                total_trainable_params += trainable_params_in_module
                
            
            if 'lin1' not in name and  'lin2' not in name :
                total_trainable_params += torch.prod(torch.tensor(module.weight.size())).item()
            
                


                

    
    total_trainable_params = total_trainable_params / millions
    total_params = total_params / millions
    percentage = total_trainable_params / total_params

    return subnetwork, total_trainable_params, total_params, percentage


def bert_module_handler(model,attn_channel_space,dense_channel_space,zero_fill):
    
    millions = 1000000

    module_identifier = 'BertSelfAttention' # Model Identifier for DistilBERT
    
    num_attn_head = model.config.num_attention_heads
    total_trainable_params = 0
    total_params = 0
    random.seed(time.time())
    subnetwork = copy.deepcopy(model).cpu()
    dense_dim = model.config.intermediate_size
    for name, module in subnetwork.named_modules():
            
        if module_identifier in str(type(module)):
            

            head_dim = module.query.weight.size(0) // num_attn_head

            
            sampled_channels = random.choice(attn_channel_space)
            sampled_channels = min(sampled_channels, head_dim)
            
            mask = torch.cat(
                [torch.ones(sampled_channels), torch.zeros(head_dim - sampled_channels)]*num_attn_head
                                )
            mask = mask.view(-1,1)

            def create_attn_mask_hook(mask):
                def gradient_mask_hook(grad):
                    if grad is None:
                        return None
                    return grad * mask.to(grad.device)
                return gradient_mask_hook

            hook = create_attn_mask_hook(mask=mask)
            # q, k 
            module.query.weight.register_hook(hook)
            module.key.weight.register_hook(hook)
            

            if zero_fill:
                module.query.weight.data = module.query.weight.data * mask
                module.key.weight.data = module.key.weight.data * mask
                                    
            
            # Calculate #Parameters for q,k
            trainable_params_in_module = torch.prod(torch.tensor((torch.sum(mask).item(),module.query.weight.size(1)))).item()
            total_trainable_params += trainable_params_in_module * 2

            

            

        if hasattr(module, 'weight') and isinstance(module.weight, Parameter) and module.weight.requires_grad:
            total_params += torch.prod(torch.tensor(module.weight.size())).item()
            if name.endswith('.dense'):
                            # linear
                linear_dim = module.weight.size(0)
                
                dense_sampled_channels = random.choice(dense_channel_space)
                dense_sampled_channels = min(dense_sampled_channels, linear_dim)
                
                # module.weight_mask = torch.tensor(dense_sampled_channels)

                # mask_linear = torch.cat([torch.ones(dense_sampled_channels),torch.zeros(linear_dim - dense_sampled_channels)], dim=0)

                def create_gradient_mask_hook(dense_sampled_channels):
                    def gradient_mask_hook(grad):
                        mask = torch.zeros_like(grad).to(grad.device)
                        mask.narrow(0, 0, dense_sampled_channels).fill_(1)
                        return grad * mask
                    return gradient_mask_hook

                hook = create_gradient_mask_hook(dense_sampled_channels)

                # module.weight.register_hook(lambda grad: gradient_mask_hook(grad,copy.deepcopy(mask_linear)))
                
                module.weight.register_hook(hook)

                
                # You can still store the mask in the module if needed, using the original weight shape
                mask_for_module = torch.zeros_like(module.weight)
                mask_for_module[:dense_sampled_channels, :] = 1
                # module.weight_mask = mask_for_module

                # Calculate #Parameters for linear
                trainable_params_in_module = torch.sum(mask_for_module).item()
                total_trainable_params += trainable_params_in_module
                
                
                
                # total_trainable_params += torch.prod(torch.tensor((dense_dim,dense_sampled_channels))).item()
                # dense_dim = dense_sampled_channels

    
    
    total_trainable_params = total_trainable_params / millions
    total_params = total_params / millions
    percentage = total_trainable_params / total_params

    return subnetwork, total_trainable_params, total_params, percentage

def roberta_module_handler(model,attn_channel_space,dense_channel_space,zero_fill):
    
    millions = 1000000

    module_identifier = 'RobertaSelfAttention' # Model Identifier for DistilBERT
    
    num_attn_head = model.config.num_attention_heads
    total_trainable_params = 0
    total_params = 0
    random.seed(time.time())
    subnetwork = copy.deepcopy(model).cpu()
    dense_dim = model.config.intermediate_size
    for name, module in subnetwork.named_modules():
            
        if module_identifier in str(type(module)):
            

            head_dim = module.query.weight.size(0) // num_attn_head

            
            sampled_channels = random.choice(attn_channel_space)
            sampled_channels = min(sampled_channels, head_dim)
            
            mask = torch.cat(
                [torch.ones(sampled_channels), torch.zeros(head_dim - sampled_channels)]*num_attn_head
                                )
            mask = mask.view(-1,1)

            def create_attn_mask_hook(mask):
                def gradient_mask_hook(grad):
                    if grad is None:
                        return None
                    return grad * mask.to(grad.device)
                return gradient_mask_hook

            hook = create_attn_mask_hook(mask=mask)
            # q, k 
            module.query.weight.register_hook(hook)
            module.key.weight.register_hook(hook)
            

            if zero_fill:
                module.query.weight.data = module.query.weight.data * mask
                module.key.weight.data = module.key.weight.data * mask
                                    
            
            # Calculate #Parameters for q,k
            trainable_params_in_module = torch.prod(torch.tensor((torch.sum(mask).item(),module.query.weight.size(1)))).item()
            total_trainable_params += trainable_params_in_module * 2

            

            

        if hasattr(module, 'weight') and isinstance(module.weight, Parameter) and module.weight.requires_grad:
            total_params += torch.prod(torch.tensor(module.weight.size())).item()
            if name.endswith('.dense'):
                            # linear
                linear_dim = module.weight.size(0)
                
                dense_sampled_channels = random.choice(dense_channel_space)
                dense_sampled_channels = min(dense_sampled_channels, linear_dim)
                
                # module.weight_mask = torch.tensor(dense_sampled_channels)

                # mask_linear = torch.cat([torch.ones(dense_sampled_channels),torch.zeros(linear_dim - dense_sampled_channels)], dim=0)

                def create_gradient_mask_hook(dense_sampled_channels):
                    def gradient_mask_hook(grad):
                        mask = torch.zeros_like(grad).to(grad.device)
                        mask.narrow(0, 0, dense_sampled_channels).fill_(1)
                        return grad * mask
                    return gradient_mask_hook

                hook = create_gradient_mask_hook(dense_sampled_channels)

                # module.weight.register_hook(lambda grad: gradient_mask_hook(grad,copy.deepcopy(mask_linear)))
                
                module.weight.register_hook(hook)

                
                # You can still store the mask in the module if needed, using the original weight shape
                mask_for_module = torch.zeros_like(module.weight)
                mask_for_module[:dense_sampled_channels, :] = 1
                # module.weight_mask = mask_for_module

                # Calculate #Parameters for linear
                trainable_params_in_module = torch.sum(mask_for_module).item()
                total_trainable_params += trainable_params_in_module
                
                
                
                # total_trainable_params += torch.prod(torch.tensor((dense_dim,dense_sampled_channels))).item()
                # dense_dim = dense_sampled_channels

    
    total_trainable_params = total_trainable_params / millions
    total_params = total_params / millions
    percentage = total_trainable_params / total_params

    return subnetwork, total_trainable_params, total_params, percentage

       
        
def salient_submodel_extraction(model, attn_channel_space=None, dense_channel_space=None, zero_fill=False ,target_model_params_size=None):
    
    #set up defualt submodel space
    if attn_channel_space is None:
        attn_channel_space = [256,384,512,640,728,1024,2048]
        attn_channel_space = [32,36,40,44,48,52,56,60,64,68,72,76,80]
    if dense_channel_space is None:
        dense_channel_space = [512,728,1024,2048,3072]
        dense_channel_space = [128,256,256+64,428,512,728,1024,2048,3072]



    while True: # Sample salient submodel that satisfying resource_constraint
        
        
        if 'distilbert' in model.config.model_type.lower():
            subnetwork, total_trainable_params, total_params, percentage = distilbert_module_handler(model,attn_channel_space,dense_channel_space,zero_fill)
        

        elif 'roberta' in model.config.model_type.lower():
            subnetwork, total_trainable_params, total_params, percentage = roberta_module_handler(model,attn_channel_space,dense_channel_space,zero_fill)
        
        elif 'bert' in model.config.model_type.lower():
            subnetwork, total_trainable_params, total_params, percentage = bert_module_handler(model,attn_channel_space,dense_channel_space,zero_fill)
        
        else:
            raise NotImplemented
        
        if target_model_params_size is None:
            return subnetwork, total_trainable_params, total_params, percentage
        

        # Evaluate current network FLOPs
        # trainable_model_params, all_model_params, percentage = calculate_trainable_params(subnetwork)
        
        if target_model_params_size>0 and target_model_params_size<1:
            if percentage > target_model_params_size:
                continue
            else:
                return subnetwork, total_trainable_params, total_params, percentage
        else:
            if total_trainable_params > target_model_params_size:
                continue
            else:
                return subnetwork, total_trainable_params, total_params, percentage

