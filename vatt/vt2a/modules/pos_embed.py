import numpy as np

def np_get_1d_sincos_pos_embed(embed_dim, size):
    """
    Compute a 1D sine-cosine positional embedding in a single function.

    Args:
        embed_dim (int): Output dimension for each position. Must be even.
        size (int): Number of positions to encode (e.g., sequence length).

    Returns:
        np.ndarray: Positional embedding of shape (size, embed_dim).
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"

    grid = np.arange(size, dtype=np.float32) 

    half_dim = embed_dim // 2
    omega = np.arange(half_dim, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)                                
    out = np.einsum("m,d->md", grid, omega)
    pos_embed = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return pos_embed