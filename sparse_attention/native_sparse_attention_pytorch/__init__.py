"""
Local copy of Native Sparse Attention for CS441 project.

We expose `SparseAttention` from the in-repo implementation so that other
modules can simply do:

    from sparse_attention.native_sparse_attention_pytorch import SparseAttention
"""

from .native_sparse_attention import SparseAttention

__all__ = ["SparseAttention"]
