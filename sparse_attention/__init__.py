"""
Sparse Attention Module for CS441 Project
Adds trainable sparse attention adapters to frozen Llama models

This module provides:
- Compression modules (MaxPoolCompress, MeanPoolCompress, MLPCompress)
- Sparse attention adapters (SparseAttentionAdapter)
- Llama model with sparse attention (LlamaWithSparseAttention)
- Distillation wrapper (SparseDistillationModel)
"""

from .sparse_attention_model import (
    RMSNorm,
    MaxPoolCompress,
    MeanPoolCompress,
    MLPCompress,
    SparseAttentionAdapter,
    LlamaWithSparseAttention,
)

from .sparse_distillation_model import (
    SparseDistillationModel,
)

__all__ = [
    # Normalization
    'RMSNorm',
    
    # Compression modules
    'MaxPoolCompress',
    'MeanPoolCompress',
    'MLPCompress',
    
    # Sparse attention
    'SparseAttentionAdapter',
    'LlamaWithSparseAttention',
    
    # Distillation
    'SparseDistillationModel',
]

__version__ = '0.1.0'

