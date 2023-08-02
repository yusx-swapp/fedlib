import torch
from torch.nn import Parameter
import random
import numpy as np
import copy

def reordering_weights(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, Parameter):
            # Ensure that the weights are trainable
            if module.weight.requires_grad:
                # Get the weight matrix
                weight_matrix = module.weight.data
                
                if len(weight_matrix.shape) == 4: # Conv2D layer or Multihead Attention
                    # Calculate the matrix norm of each channel
                    norms = torch.norm(weight_matrix, dim=(1,2,3))
                    
                elif len(weight_matrix.shape) == 2: # Linear layer
                    norms = torch.norm(weight_matrix, dim=1)
                    
                else: 
                    continue

                # Get the ordering indices
                order_indices = torch.argsort(norms, descending=True)
                
                # Reorder the weight matrix based on the indices
                ranked_weight_matrix = weight_matrix[order_indices]
                
                # Replace the original weight matrix with the ranked one
                module.weight.data = ranked_weight_matrix

    return model

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def calculate_trainable_model_parameters(model):
    """
    Calculate the number of trainable model parameters
    
    Args:
        model: the model to be evaluated
    
    Returns:
        trainable_model_params: the number of trainable model parameters (in millions)
        all_model_params: the number of all model parameters (in millions)
        percentage: the percentage of trainable model parameters
    """
    millions = 1000000
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return trainable_model_params/millions, all_model_params/millions, 100 * trainable_model_params / all_model_params


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
        trainable_model_params, all_model_params, percentage = calculate_trainable_model_parameters(subnetwork)

        if trainable_model_params > target_model_params_size:
            continue
        else:
            return subnetwork
        
def masking_model(model, target_model_params_size):
    possible_channels = [32,64,128,512,1024,2048]
    subnetwork = copy.deepcopy(model)

    while True:
        for name, module in subnetwork.named_modules():
            if hasattr(module, 'weight') and isinstance(module.weight, Parameter):
                original_channels = module.weight.size(0)

                # Generate a random number from the list and get the minimum of the sampled number and original_channels
                sampled_channels = random.choice(possible_channels)
                num_channels = min(sampled_channels, original_channels)

                # Create a mask that disables gradient for channels outside the subnetwork
                mask = torch.cat([torch.ones(num_channels), torch.zeros(original_channels - num_channels)]).to(module.weight.device)
                
                # Attach a hook to the module that disables gradient during the backward pass
                module.weight.register_hook(lambda grad: grad * mask)

        # Evaluate current network FLOPs
        current_flops = calculate_trainable_model_parameters(subnetwork)

        if current_flops > target_model_params_size:
            continue
        else:
            return subnetwork