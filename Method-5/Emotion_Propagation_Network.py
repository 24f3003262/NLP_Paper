import jax
import jax.numpy as jnp

def rough_aggregation(neighbor_hs):

    # Logic: High variance = Indiscernibility/Conflict 
    mean_state = jnp.mean(neighbor_hs, axis=0)
    variance = jnp.var(neighbor_hs, axis=0)

    #Split into Granules based on a threshold 
    threshold = jnp.median(variance)
    lower_mask = (variance <= threshold).astype(jnp.float32)
    boundary_mask = (variance > threshold).astype(jnp.float32)

    h_lower = jnp.sum(neighbor_hs * lower_mask, axis=0)
    h_boundary = jnp.sum(neighbor_hs * boundary_mask, axis=0)

    return h_lower, h_boundary


class Graph_Emotion_Network:
    def __init__(self, d=768):
        self.d = d # Hidden dimension from BERT [cite: 159]

    def init_params(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            'W_low': jax.random.normal(k1, (self.d, self.d)) * 0.1,
            'W_bnd': jax.random.normal(k2, (self.d, self.d)) * 0.1,
            'W_self': jax.random.normal(k3, (self.d, self.d)) * 0.1
        }
    
    def forward(self, params, node_h, neighbor_hs):
        # Step 4.5.1: Perform Rough Granulation 
        h_low, h_bnd = rough_aggregation(neighbor_hs)
        
        # Final update: h_v = σ(W_low*h_low + W_bnd*h_bnd + W_self*node_h) 
        term1 = jnp.dot(h_low, params['W_low'])
        term2 = jnp.dot(h_bnd, params['W_bnd'])
        term3 = jnp.dot(node_h, params['W_self'])
        
        return jax.nn.sigmoid(term1 + term2 + term3)