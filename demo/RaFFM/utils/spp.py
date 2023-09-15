# Implementation for Salient Parameter Prioritization

import torch
import copy

__all__ = ['l1_norm', 'l2_norm','salient_parameter_prioritization','spp_precision_effect']

def l1_norm(query, key, num_attn_head=12):
    """
    Rank rows of query and key matrices based on the average L1 norm.

    Args:
    - query (torch.Tensor): The query matrix.
    - key (torch.Tensor): The key matrix.

    Returns:
    - torch.Tensor: Ranked row indices based on the average L1 norm.
    """
    
    # Validate input sizes
    if query.size(0) != key.size(0) or query.size(1) != key.size(1):
        raise ValueError("The query and key matrices must have the same dimensions.")

    head_dim = query.size(0) // num_attn_head

    all_ranked_indices = []

    for i in range(num_attn_head):
        start_idx = i * head_dim
        end_idx = (i + 1) * head_dim

        query_head = query[start_idx:end_idx, :]
        key_head = key[start_idx:end_idx, :]
        
        # Calculate L1 norm for each row in both matrices for the current head
        query_norms = query_head.norm(p=1, dim=1)
        key_norms = key_head.norm(p=1, dim=1)

    
        # Compute the average L1 norms for corresponding rows
        avg_norms = (query_norms + key_norms) / 2.0
    
    # Sort the rows based on these average norms in descending order and get the indices
        _, ranked_indices = torch.sort(avg_norms, descending=True)
        
        # Adjust the indices to correspond to their original positions in query and key
        adjusted_indices = ranked_indices + start_idx
        all_ranked_indices.append(adjusted_indices)


    return torch.cat(all_ranked_indices)

def l2_norm(query, key,num_attn_head=12):
    """
    Rank rows of query and key matrices based on the average L1 norm.

    Args:
    - query (torch.Tensor): The query matrix.
    - key (torch.Tensor): The key matrix.

    Returns:
    - torch.Tensor: Ranked row indices based on the average L1 norm.
    """
    
    # Validate input sizes
    if query.size(0) != key.size(0) or query.size(1) != key.size(1):
        raise ValueError("The query and key matrices must have the same dimensions.")

    head_dim = query.size(0) // num_attn_head

    all_ranked_indices = []

    for i in range(num_attn_head):
        start_idx = i * head_dim
        end_idx = (i + 1) * head_dim

        query_head = query[start_idx:end_idx, :]
        key_head = key[start_idx:end_idx, :]
        
        # Calculate L1 norm for each row in both matrices for the current head
        query_norms = query_head.norm(p=2, dim=1)
        key_norms = key_head.norm(p=2, dim=1)

    
        # Compute the average L1 norms for corresponding rows
        avg_norms = (query_norms + key_norms) / 2.0
    
    # Sort the rows based on these average norms in descending order and get the indices
        _, ranked_indices = torch.sort(avg_norms, descending=True)
        
        # Adjust the indices to correspond to their original positions in query and key
        adjusted_indices = ranked_indices + start_idx
        all_ranked_indices.append(adjusted_indices)


    return torch.cat(all_ranked_indices)


def salient_parameter_prioritization(org_model, metric = l1_norm):
    """
    Prioritize the saliant weights of the query and key matrices in all multi-head attention layers of BERT based on a given metric.
    
    Args:
    - org_model (torch.nn.Module): The original BERT model.
    - metric (function): A function that takes in query and key matrices and returns a permutation index.

    Returns:
    - model (torch.nn.Module): The model with permuted weights.
    """
    model = copy.deepcopy(org_model)
    num_attn_head = model.config.num_attention_heads
    
    # Iterate over all modules in the model
    for name, module in model.named_modules():
        # Check if the module is BertSelfAttention
        if "BertSelfAttention" in str(type(module)):
            # Get permutation using the metric function
            perm = metric(module.query.weight.data, module.key.weight.data,num_attn_head=num_attn_head)
            
            # Ensure the permutation is in the correct format
            assert isinstance(perm, torch.Tensor), "The metric function must return a torch.Tensor."
            assert perm.shape[0] == module.query.weight.shape[0], "Invalid permutation size."

            # Permute the query weights
            module.query.weight.data = module.query.weight.data[perm, :]
            if module.query.bias is not None:
                module.query.bias.data = module.query.bias.data[perm]

            # Permute the key weights
            module.key.weight.data = module.key.weight.data[perm, :]
            if module.key.bias is not None:
                module.key.bias.data = module.key.bias.data[perm]

        elif "RobertaSelfAttention" in str(type(module)):
                        # Get permutation using the metric function
            perm = metric(module.query.weight.data, module.key.weight.data,num_attn_head=num_attn_head)
            
            # Ensure the permutation is in the correct format
            assert isinstance(perm, torch.Tensor), "The metric function must return a torch.Tensor."
            assert perm.shape[0] == module.query.weight.shape[0], "Invalid permutation size."

            # Permute the query weights
            module.query.weight.data = module.query.weight.data[perm, :]
            if module.query.bias is not None:
                module.query.bias.data = module.query.bias.data[perm]

            # Permute the key weights
            module.key.weight.data = module.key.weight.data[perm, :]
            if module.key.bias is not None:
                module.key.bias.data = module.key.bias.data[perm]

        elif "MultiHeadSelfAttention" in str(type(module)):
                                    # Get permutation using the metric function
            perm = metric(module.q_lin.weight.data, module.k_lin.weight.data,num_attn_head=num_attn_head)
            
            # Ensure the permutation is in the correct format
            assert isinstance(perm, torch.Tensor), "The metric function must return a torch.Tensor."
            assert perm.shape[0] == module.q_lin.weight.shape[0], "Invalid permutation size."

            # Permute the query weights
            module.q_lin.weight.data = module.q_lin.weight.data[perm, :]
            if module.q_lin.bias is not None:
                module.q_lin.bias.data = module.q_lin.bias.data[perm]

            # Permute the key weights
            module.k_lin.weight.data = module.k_lin.weight.data[perm, :]
            if module.k_lin.bias is not None:
                module.k_lin.bias.data = module.k_lin.bias.data[perm]

    return model


def spp_precision_effect(original_model, permuted_model, tokenized_input):
    """
    Evaluates total difference in final output between the original model and the permuted model.
    The difference is caused by the precision error 
    Args:
    - original_model (torch.nn.Module): The original BERT model.
    - permuted_model (torch.nn.Module): The permuted BERT model.
    - input_text (str): Text to be fed into the models.

    Returns:
    - difference_final_output (torch.Tensor): Total difference in the final output between the two models.
    """


    # Forward pass through both models
    with torch.no_grad():
        original_outputs = original_model(**tokenized_input).last_hidden_state
        permuted_outputs = permuted_model(**tokenized_input).last_hidden_state

    # Compute the total difference
    difference_final_output = torch.abs(original_outputs - permuted_outputs).sum()

    # Compute the error rate
    total_scalars = original_outputs.numel()
    error_rate = difference_final_output.item() / total_scalars

    return error_rate, difference_final_output



def _test_():
    "Test the l1_norm"
    q = torch.randn(5, 4)
    k = torch.randn(5, 4)

    print("Query matrix:\n", q)
    print("\nKey matrix:\n", k)
    print("\nRanked row indices:", l1_norm(q, k))

    #Test the SPP
    from transformers import RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer
   
    # Load a BERT model
    model = BertModel.from_pretrained('bert-base-uncased')

    # Get the permuted model
    permuted_model = salient_parameter_prioritization(model)

     # Tokenize the input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_text = "This is a sample sentence for testing the BERT model."
    inputs = tokenizer(input_text, return_tensors="pt")

    error_rate, diff = spp_precision_effect(model, permuted_model, inputs)
    print("Total difference in final output:", diff.item())
    print("Error rate:", error_rate)


    from transformers import RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer

    # Roberta

    # Load a Roberta model
    roberta_model = RobertaModel.from_pretrained('roberta-base')

    # Tokenize the input using Roberta tokenizer
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_inputs = roberta_tokenizer(input_text, return_tensors="pt")

    # DistilBert

    # Load a DistilBert model
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Tokenize the input using DistilBert tokenizer
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_inputs = distilbert_tokenizer(input_text, return_tensors="pt")
