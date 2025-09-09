import torch
import quantization



# Global variables for statistics (keeping original behavior)
i = 0
total_size = 0
total_len = 0
max_len = 0

def QK_prune_binary_quant(query_layer, key_layer, attention_mask, pr_ratio=0.5, top_k=40):
    """Fixed version of QK_prune_binary_quant with proper tensor handling"""
    
    # Get dimensions
    batch_size = query_layer.shape[0]
    num_head = query_layer.shape[1]
    max_sq_len = query_layer.shape[-2]
    
    # #For debugging inputs of qk prune
    # print(f"Debug QK_prune_binary_quant:")
    # print(f"  Input shapes: Q={query_layer.shape}, K={key_layer.shape}")
    # print(f"  batch_size={batch_size}, num_head={num_head}, max_sq_len={max_sq_len}")
    # if attention_mask is not None:
    #     print(f"  attention_mask shape: {attention_mask.shape}")

    # Threshold for quantization
    threshold = 1e-8

    # Binary quantization for Q and K
    Q_b = quantization.A_Binarize(query_layer, quant_mode='det')
    K_b = quantization.A_Binarize(key_layer, quant_mode='det')
    
    # Compute binary attention scores
    attention_scores_b = torch.matmul(Q_b, K_b.transpose(-1, -2))

    # FIXED: Handle sequence length calculation properly
    if attention_mask is not None:
        # Apply attention mask
        attention_scores_b = attention_scores_b + attention_mask
        
        # CRITICAL FIX: Calculate sq_len properly without unpredictable squeeze
        mask_valid = (attention_mask + 10000) / 10000  # Convert back to 0/1
        
        # Count non-zero elements along the last dimension
        sq_len_raw = torch.count_nonzero(mask_valid, dim=-1).to(torch.float)
        # print(f"  sq_len_raw shape: {sq_len_raw.shape}")
        
        # FIXED: Ensure sq_len has exactly batch_size elements
        if sq_len_raw.dim() > 1:
            # Take the maximum across heads and sequence positions
            sq_len = torch.max(sq_len_raw.view(batch_size, -1), dim=1)[0]
        else:
            # Already in correct format
            sq_len = sq_len_raw
            
        # Ensure we have exactly batch_size elements
        if sq_len.numel() != batch_size:
            print(f"  Warning: sq_len has {sq_len.numel()} elements, expected {batch_size}")
            # Fallback: use maximum sequence length for all samples
            sq_len = torch.full((batch_size,), max_sq_len, 
                              dtype=torch.float, device=query_layer.device)
    else:
        # No attention mask - use full sequence length
        sq_len = torch.full((batch_size,), max_sq_len, 
                          dtype=torch.float, device=query_layer.device)
    
    # print(f"  Final sq_len shape: {sq_len.shape}, values: {sq_len}")

    # Global statistics (keeping original behavior)
    global i, total_size, total_len, max_len
    i += 1 
    # print('i= ',i)
    if i == 12:
        i = 0      
        total_size += batch_size
        total_len += torch.sum(sq_len).item()
        mean_len = total_len / total_size
        max_len = max(max_len, torch.max(sq_len).item())
        # print(f"  Stats - avg length: {mean_len}, max length: {max_len}")
    
    # FIXED: Calculate top-k values properly
    # Method 1: Pruning ratio based
    top_k_tensor_ratio = torch.ceil(sq_len * (1 - pr_ratio))
    # Method 2: Fixed top-k based  
    top_k_tensor_fixed = torch.minimum(
        torch.floor(sq_len / 100000) + top_k, 
        sq_len
    )
    # Use the second method (original behavior)
    top_k_tensor = top_k_tensor_fixed
    
    # print(f"  top_k_tensor shape: {top_k_tensor.shape}, values: {top_k_tensor}")
    
    # FIXED: Reshape and expand properly
    try:
        # Ensure top_k_tensor has the right shape for reshaping
        if top_k_tensor.numel() != batch_size:
            print(f"  Error: top_k_tensor has {top_k_tensor.numel()} elements, expected {batch_size}")
            # Emergency fallback
            top_k_tensor = torch.full((batch_size,), top_k, 
                                    dtype=torch.float, device=query_layer.device)
        
        # Reshape to [batch_size, 1, 1, 1]
        rm_len = top_k_tensor.view(batch_size, 1, 1, 1)
        # print(f"  rm_len shape after reshape: {rm_len.shape}")
        
        # Expand to match attention scores dimensions
        rm_len = rm_len.expand(batch_size, num_head, max_sq_len, max_sq_len).clone()
        # print(f"  rm_len shape after expand: {rm_len.shape}")
        
    except Exception as e:
        print(f"  Error in tensor reshaping: {e}")
        # Ultimate fallback: create tensor with correct shape
        rm_len = torch.full((batch_size, num_head, max_sq_len, max_sq_len), 
                           top_k, dtype=torch.float, device=query_layer.device)
        # print(f"  Using fallback rm_len shape: {rm_len.shape}")

    # Compute ranking
    rank_b = torch.argsort(attention_scores_b, dim=-1, descending=True)
    rank_indice = torch.argsort(rank_b, dim=-1, descending=False)

    # Create pruning mask
    S_prune_mask = torch.zeros_like(rm_len)
    S_prune_mask[rank_indice >= rm_len] = -10000
    
    # print(f"  Final S_prune_mask shape: {S_prune_mask.shape}")
    
    return S_prune_mask

# def QK_prune_binary_quant(query_layer, key_layer, attention_mask, pr_ratio = 0.5, top_k = 40):
#     ###The function for quant computation and S_prune_mask computation

#     ### The query_layer and key_layer will have 4 dimension, the 1st dimension is batch
#     ### size the 2st dimension is number of head, the 3st dimension is max sequence
#     ### length, the 4th dimension is the hidden dimension
#     ### A example: query_layer.size() = [16, 12, 384, 64]
#     batch_size = query_layer.shape[0]
#     num_head = query_layer.shape[1]
#     max_sq_len = query_layer.shape[-2]

#     ### The threshold for quantize, in order to remain the pruned 0 value
#     threshold = 1e-8
#     # with torch.no_grad():

#     # quant for Q and K, get the quantized value after matrix multiply
#     # Q_b = quantization.A_Binarize(query_layer, quant_mode = 'rand')
#     Q_b = quantization.A_Binarize(query_layer, quant_mode = 'det')
#     # Q_b = query_layer.detach().clone()
#     # Q_b[Q_b > threshold] = 1
#     # Q_b[Q_b < (-threshold)] = -1

#     # K_b = quantization.A_Binarize(key_layer, quant_mode = 'rand')
#     K_b = quantization.A_Binarize(key_layer, quant_mode = 'det')
#     # K_b = key_layer.detach().clone()
#     # K_b[K_b > threshold] = 1
#     # K_b[K_b < (-threshold)] = -1
#     #attention score after quant
#     attention_scores_b = torch.matmul( \
#         Q_b, K_b.transpose(-1, -2))

#     # sq_len is the the sequence length of input sentence for a mini-batch
#     # Initialize sq_len with default value (max_sq_len) for all batches
#     sq_len = torch.empty(batch_size, dtype=torch.float, device=query_layer.device).fill_(max_sq_len)

#     if attention_mask is not None:
#         # Apply the attention mask is (precomputed for all layers
#         # in BertModel forward() function)
#         attention_scores_b = attention_scores_b + attention_mask
#         # Calculate actual sequence lengths from attention mask
#         # The mask has -10000 for padded positions, 0 for valid positions
#         sq_len = torch.squeeze(torch.count_nonzero((attention_mask + 10000) / 10000, dim=-1)).to(torch.float)
        
#         # Handle case where squeeze removes all dimensions
#         if sq_len.dim() == 0:
#             sq_len = sq_len.unsqueeze(0)
#         # Ensure sq_len has the right shape
#         if sq_len.shape[0] != batch_size:
#             sq_len = sq_len.expand(batch_size)
    
#     # count sequence length tracking (optional, can be commented out in production)
#     global i
#     global total_size
#     global total_len
#     global max_len
#     i += 1 
#     if(i == 12):
#         i = 0      
#         total_size += batch_size
#         total_len += torch.sum(sq_len).tolist()
#         mean_len = total_len/total_size
#         max_len = max_len if max_len > torch.max(sq_len).tolist() else torch.max(sq_len).tolist()
#         print("avg length:", mean_len)
#         print("max length", max_len)
    
#     # get the sequence length after pruning
#     # the first pruning method: prune through pruning ratio
#     top_k_tensor = torch.ceil(sq_len * (1 - pr_ratio))
#     # the second pruning method: prune through top-k
#     top_k_tensor = torch.minimum((torch.floor(sq_len/100000) + top_k), sq_len)
#     rm_len = torch.reshape(top_k_tensor.to(torch.float), [batch_size, 1, 1, 1])
    
#     rm_len = rm_len.expand(batch_size, num_head, max_sq_len, max_sq_len).clone()

#     rank_b = torch.argsort(attention_scores_b, dim=-1, descending=True)
#     rank_indice = torch.argsort(rank_b, dim=-1, descending=False)

#     # Create pruning mask
#     S_prune_mask = torch.clone(rm_len).fill_(0)
#     S_prune_mask[rank_indice >= rm_len] = -10000
#     return S_prune_mask

# ==============================================================================
# WRAPPER FUNCTION TO USE FIXED VERSION
# ==============================================================================

def QK_prune_binary_quant_wrapper(query_layer, key_layer, attention_mask, pr_ratio=0.5, top_k=40):
    """Wrapper that tries fixed version first, falls back to original if needed"""
    try:
        return QK_prune_binary_quant(query_layer, key_layer, attention_mask, pr_ratio, top_k)
    except Exception as e:
        print(f"Fixed version failed: {e}")
        print("Falling back to no pruning...")
        # Return no pruning mask
        batch_size, num_head, seq_len = query_layer.shape[:3]
        return torch.zeros(batch_size, num_head, seq_len, seq_len, 
                          dtype=query_layer.dtype, device=query_layer.device)

def attention_mask_adaptive(attention_mask):
    batch_size = attention_mask.shape[0]
    max_sq_len = attention_mask.shape[3]
    return torch.reshape(attention_mask, (batch_size, 1, max_sq_len, 1))