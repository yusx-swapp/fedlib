import torch
from torch.nn import Parameter
import random
import numpy as np
import copy
import torch
import time
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
def reordering_weights(model):
    """
    Reorder the weights of the model based on the matrix norm of each channel.
    The channel with the largest norm will be placed at the beginning of the weight matrix.
    The channel with the smallest norm will be placed at the end of the weight matrix.
    args:
        model: the model to be reordered
    return:
        model: the reordered model
    """
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, Parameter):
            # Ensure that the weights are trainable
            if module.weight.requires_grad:
                weight_matrix = module.weight.data
                
                if name.endswith('dense'):
                    # print("weight matrix size",weight_matrix.size())
                    norms = torch.norm(weight_matrix, dim=1)
                    # print("norms size",norms.size())
                    # Get the ordering indices
                    order_indices = torch.argsort(norms, descending=True)
                    
                    # Reorder the weight matrix based on the indices
                    ranked_weight_matrix = weight_matrix[order_indices]
                    ranked_bias = module.bias.data[order_indices]
                    # print(weight_matrix)
                    # print(ranked_weight_matrix)
                    # Replace the original weight matrix with the ranked one
                    module.weight = Parameter(ranked_weight_matrix)
                    module.bias = Parameter(ranked_bias)
                    
                else: 
                    continue

                

    return model



def subnetwork_extraction(model, target_model_params_size):
    """
    Extract a subnetwork from the original network based on the target model parameters size
    Args:
        model: the original model
        target_model_params_size: the target model parameters size
    Returns:
        subnetwork: the extracted subnetwork
    """

    possible_channels = [32,64,128,256,512,768,1024,2048,3072]

    while True:
        subnetwork = copy.deepcopy(model)
        previous_channels = 0
        for name, module in subnetwork.named_modules():
            if "encoder" not in name:
                continue
            if hasattr(module, 'weight') and isinstance(module.weight, Parameter) and module.weight.requires_grad:
                original_channels = module.weight.size(0)

                # Generate a random number from the list and get the minimum of the sampled number and original_channels
                sampled_channels = random.choice(possible_channels)
                num_channels = min(sampled_channels, original_channels)
                print("subnet before slicing: ",module.weight.size())

                if previous_channels != 0 and "LayerNorm" not in name:
                    # Slice the weight matrix based on the subnetwork configuration
                    new_weight = module.weight.data[:num_channels]
                    # print("new weight size",new_weight.size())
                    new_weight = new_weight[:, :previous_channels]
                elif previous_channels != 0 and "LayerNorm" in name:
                    new_weight = module.weight.data[:previous_channels]
                else:
                    # Slice the weight matrix based on the subnetwork configuration
                    new_weight = module.weight.data[:num_channels]
                print("subnet after slicing",new_weight.size())
                # Replace the subnetwork's weight in current layer with the new weight
                module.weight = Parameter(new_weight)
                previous_channels = num_channels
        # Evaluate current network FLOPs
        trainable_model_params, all_model_params, percentage = calculate_trainable_params(subnetwork)

        if trainable_model_params > target_model_params_size:
            continue
        else:
            return subnetwork
        
def distillbert_subnetwork(model, target_model_params_size=None):
    millions = 1000000
    possible_channels = [256,384,512,640,728,1024,2048]
    dense_channels = [512,728,1024,2048,3072]
    while True:
        total_trainable_params = 0
        total_params = 0
        random.seed(time.time())
        subnetwork = copy.deepcopy(model)
        previous_channels = 0
        for name, module in subnetwork.named_modules():

                    
            if hasattr(module, 'weight') and isinstance(module.weight, Parameter) and module.weight.requires_grad:
                total_params += torch.prod(torch.tensor(module.weight.size())).item()
                original_channels = module.weight.size(0)
                if "encoder" not in name:
                    total_trainable_params += torch.prod(torch.tensor(module.weight.size())).item()
                    continue
                
                
                elif name.endswith('out_lin'):
                    dense_sampled_channels = random.choice(dense_channels)
                    dense_sampled_channels = min(dense_sampled_channels, original_channels)
                    
                    def mask_grad(grad):
                        if grad is None:
                            return None
                        # Create a mask with the same shape as the gradient
                        mask = torch.zeros_like(grad).to(grad.device)
                        # Set the first dense_sampled_channels rows to ones
                        mask.narrow(0, 0, dense_sampled_channels).fill_(1)
                        return grad * mask

                    module.weight.register_hook(mask_grad)
                    # You can still store the mask in the module if needed, using the original weight shape
                    mask_for_module = torch.zeros_like(module.weight).to('cuda')
                    mask_for_module[:dense_sampled_channels, :] = 1
                    # module.weight_mask = mask_for_module
                    trainable_params_in_module = torch.sum(mask_for_module).item()
                    total_trainable_params += trainable_params_in_module
                
            if 'MultiHeadSelfAttention' in str(type(module)):
                original_channels = module.q_lin.weight.size(0)
                sampled_channels = random.choice(possible_channels)
                sampled_channels = min(sampled_channels, original_channels)
                mask = torch.cat([torch.ones(sampled_channels), torch.zeros(original_channels - sampled_channels)]).to('cuda')
                
                
                module.q_lin.weight.register_hook(lambda grad: grad * mask if grad is not None else None)
                module.k_lin.weight.register_hook(lambda grad: grad * mask if grad is not None else None)
                trainable_params_in_module = torch.sum(mask).item()
                total_trainable_params += trainable_params_in_module


                module.q_lin.weight.data



                    

        
        total_trainable_params = total_trainable_params / millions
        total_params = total_params / millions
        percentage = total_trainable_params / total_params
        
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



def gradient_masking_extraction(model, target_model_params_size=None):
    """
    Extract a subnetwork from the original network based on the target model parameters size
    Args:
        model: the original model
        target_model_params_size: the target model parameters size
    Returns:
        subnetwork: the extracted subnetwork
    """
    if 'distilbert' in model.config.name_or_path.lower():
        return distillbert_subnetwork(model, target_model_params_size)
    millions = 1000000
    possible_channels = [256,384,512,640,728,1024,2048]
    dense_channels = [512,728,1024,2048,3072]
    while True:
        total_trainable_params = 0
        total_params = 0
        random.seed(time.time())
        subnetwork = copy.deepcopy(model)
        previous_channels = 0
        for name, module in subnetwork.named_modules():

                    
            if hasattr(module, 'weight') and isinstance(module.weight, Parameter) and module.weight.requires_grad:
                total_params += torch.prod(torch.tensor(module.weight.size())).item()
                # print(name, module.weight.size())
                if "encoder" not in name:
                    total_trainable_params += torch.prod(torch.tensor(module.weight.size())).item()
                    continue
                original_channels = module.weight.size(0)
                # print(name, "subnet before slicing: ",module.weight.size())
                if name.endswith('.query'):

                    
                    sampled_channels = random.choice(possible_channels)
                    sampled_channels = min(sampled_channels, original_channels)
                    mask = torch.cat([torch.ones(sampled_channels), torch.zeros(original_channels - sampled_channels)]).to('cuda')
                    module.weight.register_hook(lambda grad: grad * mask if grad is not None else None)
                    trainable_params_in_module = torch.sum(mask).item()
                    total_trainable_params += trainable_params_in_module
                elif name.endswith('.key'):
                    # key is always after by query and should match query, so here we dont need to sample again
                    # mask = torch.cat([torch.zeros(sampled_channels), torch.ones(original_channels - sampled_channels)]).to(module.weight.device)
                    mask = torch.cat([torch.zeros(sampled_channels), torch.ones(original_channels - sampled_channels)]).to('cuda')
                    # module.weight_mask = mask  # Store the mask within the module
                    module.weight.register_hook(lambda grad: grad * mask if grad is not None else None)
                    trainable_params_in_module = torch.sum(mask).item()
                    total_trainable_params += trainable_params_in_module
                    
                    
                elif name.endswith('.value'):
                    
                    # mask = torch.cat([torch.zeros(sampled_channels), torch.ones(original_channels - sampled_channels)]).to(module.weight.device)
                    mask = torch.cat([torch.zeros(sampled_channels), torch.ones(original_channels - sampled_channels)]).to('cuda')
                    module.weight_mask = mask  # Store the mask within the module
                    module.weight.register_hook(lambda grad: grad * mask if grad is not None else None)
                    trainable_params_in_module = torch.sum(mask).item()
                    total_trainable_params += trainable_params_in_module
                    

                elif name.endswith('dense'):
                    dense_sampled_channels = random.choice(dense_channels)
                    dense_sampled_channels = min(dense_sampled_channels, original_channels)
                    
                    def mask_grad(grad):
                        if grad is None:
                            return None
                        # Create a mask with the same shape as the gradient
                        mask = torch.zeros_like(grad).to(grad.device)
                        # Set the first dense_sampled_channels rows to ones
                        mask.narrow(0, 0, dense_sampled_channels).fill_(1)
                        return grad * mask

                    module.weight.register_hook(mask_grad)
                    # You can still store the mask in the module if needed, using the original weight shape
                    mask_for_module = torch.zeros_like(module.weight).to('cuda')
                    mask_for_module[:dense_sampled_channels, :] = 1
                    # module.weight_mask = mask_for_module
                    trainable_params_in_module = torch.sum(mask_for_module).item()
                    total_trainable_params += trainable_params_in_module
        
        total_trainable_params = total_trainable_params / millions
        total_params = total_params / millions
        percentage = total_trainable_params / total_params
        
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