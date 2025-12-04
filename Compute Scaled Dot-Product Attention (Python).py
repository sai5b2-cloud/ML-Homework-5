import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix (seq_len_q, d_k)
        K: Key matrix (seq_len_k, d_k)
        V: Value matrix (seq_len_v, d_v)
    
    Returns:
        attention_weights: softmax scores
        context_vector: weighted sum of V
    """
    d_k = Q.shape[1]
    # Compute raw attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Softmax normalization
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Weighted sum of values
    context_vector = np.dot(attention_weights, V)
    
    return attention_weights, context_vector

# Example
Q = np.random.rand(2, 4)
K = np.random.rand(3, 4)
V = np.random.rand(3, 5)
weights, context = scaled_dot_product_attention(Q, K, V)
print("Attention weights:\n", weights)
print("Context vector:\n", context)
