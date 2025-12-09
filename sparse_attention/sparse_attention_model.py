"""
Sparse Attention Model for CS441 Project
Adds trainable sparse attention adapters to frozen Llama 3.2 1B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple
from einops import rearrange, repeat, reduce
from fastNLP import logger


# RMSNorm implementation (following Native Sparse Attention)
try:
    RMSNorm = nn.RMSNorm
except AttributeError:
    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization (from Native Sparse Attention)"""
        def __init__(self, dim: int, eps: float = 1e-8):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(dim))
            self.dim = dim
        
        def forward(self, x: torch.Tensor):
            norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return x * norm * self.scale


class MaxPoolCompress(nn.Module):
    """
    Coarse-grained compression using max pooling (following Native Sparse Attention design)
    - Selects most salient feature in each block
    - No positional embeddings (position info already in K/V from RoPE)
    - Includes RMSNorm for stability (like NSA)
    """
    def __init__(self, num_heads, dim_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        
        # RMSNorm for stability (following Native Sparse Attention)
        self.norm = RMSNorm(dim_head)
        
    def forward(self, kv):
        """
        Args:
            kv: [batch, num_heads, seq_len, dim_head]
        Returns:
            compressed: [batch, num_heads, num_blocks, dim_head]
        """
        b, h, n, d = kv.shape
        
        # Truncate to multiple of block_size
        n_blocks = n // self.block_size
        if n_blocks == 0:
            return torch.zeros(b, h, 0, d, device=kv.device, dtype=kv.dtype)
        
        kv_truncated = kv[:, :, :n_blocks * self.block_size, :]
        
        # Reshape to blocks: [b, h, n_blocks, block_size, d]
        kv_blocks = kv_truncated.reshape(b, h, n_blocks, self.block_size, d)
        
        # Max pooling over block dimension
        compressed = kv_blocks.max(dim=3)[0]  # [b, h, n_blocks, d]
        
        # Apply normalization (following Native Sparse Attention)
        compressed = self.norm(compressed)
        
        return compressed


class MeanPoolCompress(nn.Module):
    """
    Coarse-grained compression using mean pooling (inspired by Native Sparse Attention's AvgPoolCompression)
    - Averages all features in each block
    - Good for capturing overall context/semantics
    - Generally more stable than max pooling for distillation
    - No positional embeddings (position info already in K/V from RoPE)
    - Includes RMSNorm for stability (like NSA)
    """
    def __init__(self, num_heads, dim_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        
        # RMSNorm for stability (following Native Sparse Attention)
        self.norm = RMSNorm(dim_head)
        
    def forward(self, kv):
        """
        Args:
            kv: [batch, num_heads, seq_len, dim_head]
        Returns:
            compressed: [batch, num_heads, num_blocks, dim_head]
        """
        b, h, n, d = kv.shape
        
        # Truncate to multiple of block_size
        n_blocks = n // self.block_size
        if n_blocks == 0:
            return torch.zeros(b, h, 0, d, device=kv.device, dtype=kv.dtype)
        
        kv_truncated = kv[:, :, :n_blocks * self.block_size, :]
        
        # Reshape to blocks: [b, h, n_blocks, block_size, d]
        kv_blocks = kv_truncated.reshape(b, h, n_blocks, self.block_size, d)
        
        # Mean pooling over block dimension
        compressed = kv_blocks.mean(dim=3)  # [b, h, n_blocks, d]
        
        # Apply normalization (following Native Sparse Attention)
        compressed = self.norm(compressed)
        
        return compressed


class MLPCompress(nn.Module):
    """
    Learnable MLP-based compression (inspired by Native Sparse Attention's GroupedMLP)
    - Uses per-head MLPs for flexible compression
    - No positional embeddings (position info already in K/V from RoPE)
    - Includes RMSNorm for stability (like NSA)
    """
    def __init__(self, num_heads, dim_head, block_size, expand_factor=1.0):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        
        dim_in = block_size * dim_head
        dim_hidden = max(int(dim_in * expand_factor), dim_head)
        
        # Per-head compression MLPs (following NSA's design)
        self.compress_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in, dim_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_head, bias=False)
            ) for _ in range(num_heads)
        ])
        
        # RMSNorm for stability (following Native Sparse Attention)
        self.norm = RMSNorm(dim_head)
        
    def forward(self, kv):
        """
        Args:
            kv: [batch, num_heads, seq_len, dim_head]
        Returns:
            compressed: [batch, num_heads, num_blocks, dim_head]
        """
        b, h, n, d = kv.shape
        
        # Truncate to multiple of block_size
        n_blocks = n // self.block_size
        if n_blocks == 0:
            return torch.zeros(b, h, 0, d, device=kv.device, dtype=kv.dtype)
        
        kv_truncated = kv[:, :, :n_blocks * self.block_size, :]
        
        # Reshape to blocks
        kv_blocks = kv_truncated.reshape(b, h, n_blocks, self.block_size, d)
        
        # Flatten block dimension: [b, h, n_blocks, block_size * d]
        kv_flat = kv_blocks.reshape(b, h, n_blocks, -1)
        
        # Apply per-head MLPs
        compressed = []
        for head_idx in range(h):
            head_compressed = self.compress_mlps[head_idx](kv_flat[:, head_idx])
            compressed.append(head_compressed)
        
        compressed = torch.stack(compressed, dim=1)  # [b, h, n_blocks, d]
        
        # Apply normalization (following Native Sparse Attention)
        compressed = self.norm(compressed)
        
        return compressed


class SparseAttentionAdapter(nn.Module):
    """
    Trainable sparse attention adapter for each Llama layer
    Implements three-branch architecture:
    1. Sliding window (use base model's)
    2. Compressed coarse-grained attention
    3. Fine-grained selection based on importance
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        compress_block_size=16,
        compress_stride=8,
        selection_block_size=16,
        num_selected_blocks=4,
        sliding_window_size=64,
        k_compress_method='mean_pool',  # 'max_pool', 'mean_pool', or 'mlp'
        v_compress_method='mean_pool',  # 'max_pool', 'mean_pool', or 'mlp'
        teacher_o_proj_weight=None,  # Teacher's o_proj weight for initialization
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = hidden_size // num_heads
        self.num_grouped_queries = num_heads // num_kv_heads
        
        self.compress_block_size = compress_block_size
        self.compress_stride = compress_stride
        self.selection_block_size = selection_block_size
        self.num_selected_blocks = num_selected_blocks
        self.sliding_window_size = sliding_window_size
        
        # K compression (support max_pool, mean_pool, mlp)
        if k_compress_method == 'max_pool':
            self.k_compress = MaxPoolCompress(
                num_kv_heads, self.dim_head, compress_block_size
            )
        elif k_compress_method == 'mean_pool':
            self.k_compress = MeanPoolCompress(
                num_kv_heads, self.dim_head, compress_block_size
            )
        else:  # mlp
            self.k_compress = MLPCompress(
                num_kv_heads, self.dim_head, compress_block_size, expand_factor=1.0
            )
        
        # V compression (support max_pool, mean_pool, mlp)
        if v_compress_method == 'max_pool':
            self.v_compress = MaxPoolCompress(
                num_kv_heads, self.dim_head, compress_block_size
            )
        elif v_compress_method == 'mean_pool':
            self.v_compress = MeanPoolCompress(
                num_kv_heads, self.dim_head, compress_block_size
            )
        else:  # mlp
            self.v_compress = MLPCompress(
                num_kv_heads, self.dim_head, compress_block_size, expand_factor=1.0
            )
        
        # Learnable memory tokens for compressed KV
        self.num_mem_tokens = 1
        self.compress_mem_kv = nn.Parameter(
            torch.zeros(2, num_kv_heads, self.num_mem_tokens, self.dim_head)
        )
        
        # Strategy combiner: learn to weight three branches (following Native Sparse Attention)
        # Use Linear only, apply softmax in forward pass
        self.strategy_combiner = nn.Linear(hidden_size, 3 * num_heads)
        
        # Initialize to STRONGLY favor sliding window initially (critical for stable KL training)
        # After softmax, this gives: compressed≈0.007, fine≈0.007, sliding≈0.986
        # So initially, adapter output ≈ sliding window ≈ original attention
        nn.init.zeros_(self.strategy_combiner.weight)
        self.strategy_combiner.bias.data.copy_(
            torch.tensor([-5., -5., 5.] * num_heads)  # [compressed, fine, sliding]
        )
        
        # Merge and combine heads (like Native Sparse Attention)
        # merge_heads: [b, h, n, d] -> [b, n, h*d]
        # combine_heads: [b, n, h*d] -> [b, n, hidden_size]
        self.combine_heads = nn.Linear(num_heads * self.dim_head, hidden_size, bias=False)
        
        # Initialize combine_heads from teacher's o_proj for better initialization
        # This ensures student output projection matches teacher's at initialization
        if teacher_o_proj_weight is not None:
            with torch.no_grad():
                self.combine_heads.weight.copy_(teacher_o_proj_weight)
            print(f"✅ Initialized combine_heads from teacher's o_proj")
        else:
            # Fallback: small random initialization
            nn.init.normal_(self.combine_heads.weight, mean=0.0, std=0.001)
            print(f"⚠️  Teacher o_proj not provided, using random initialization")
    
    def compute_compressed_attention(
        self,
        q,     # [batch, num_heads, seq_len, dim_head]
        ck,    # compressed k [batch, num_kv_heads, num_blocks, dim_head]
        cv,    # compressed v [batch, num_kv_heads, num_blocks, dim_head]
        scale,
        attention_mask=None  # [batch, seq_len] - for future use if needed
    ):
        """Compute coarse-grained attention over compressed KV"""
        batch_size = q.shape[0]
        
        # Add memory tokens
        mem_ck, mem_cv = repeat(
            self.compress_mem_kv, 
            'kv h m d -> kv b h m d', 
            b=batch_size
        )
        
        ck = torch.cat([mem_ck, ck], dim=2)
        cv = torch.cat([mem_cv, cv], dim=2)
        
        # Handle GQA: repeat compressed KV for each query head group
        ck = repeat(ck, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        cv = repeat(cv, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        
        # Compute attention
        sim = torch.einsum('bhid,bhjd->bhij', q, ck) * scale
        
        # Apply causal mask for compressed attention (following NSA's design)
        # CRITICAL: Block must be STRICTLY in the past (block_end < query_pos)
        # This ensures no future information leakage through compressed blocks
        seq_len = q.shape[2]
        num_compress_blocks = ck.shape[2] - self.num_mem_tokens  # Exclude memory tokens
        
        if num_compress_blocks > 0:
            # Query positions: [0, 1, 2, ..., seq_len-1]
            query_positions = torch.arange(seq_len, device=q.device)  # [seq_len]
            
            # Compressed block end positions (last token in each block)
            # Block i ends at: (i+1) * block_size - 1
            block_end_positions = torch.arange(1, num_compress_blocks + 1, device=q.device) * self.compress_block_size - 1
            
            # Causal mask: block_end < query_position (STRICT <, following NSA!)
            # This means: block must be completely finished before query position
            # query_positions: [seq_len, 1], block_end_positions: [1, num_blocks]
            causal_mask = block_end_positions[None, :] < query_positions[:, None]  # [seq_len, num_blocks]
            
            # Memory tokens are always visible (prepended, with position -1)
            mem_mask = torch.ones(seq_len, self.num_mem_tokens, device=q.device, dtype=torch.bool)
            full_mask = torch.cat([mem_mask, causal_mask], dim=1)  # [seq_len, num_mem + num_blocks]
            
            # Apply mask: [batch, heads, seq_len, num_compressed]
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, num_compressed]
            sim = sim.masked_fill(~full_mask, -1e10)
        
        attn = F.softmax(sim, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, cv)
        
        return out, sim[:, :, :, self.num_mem_tokens:]  # Return importance scores (without mem tokens)
    
    def compute_fine_attention(
        self,
        q,                    # [batch, num_heads, seq_len, dim_head]
        k,                    # [batch, num_kv_heads, seq_len, dim_head]
        v,                    # [batch, num_kv_heads, seq_len, dim_head]
        importance_scores,    # [batch, num_heads, seq_len, num_compress_blocks]
        scale
    ):
        """Select important blocks based on compressed attention and compute fine attention"""
        batch_size, _, seq_len, _ = q.shape
        
        # Average importance across query head groups
        importance = reduce(
            importance_scores,
            'b (h g) ... -> b h ...',
            'mean',
            g=self.num_grouped_queries
        )
        
        num_blocks = importance.shape[-1]
        num_selected = min(self.num_selected_blocks, num_blocks)
        
        if num_selected == 0:
            # Fallback to full attention on current sequence
            k_full = repeat(k, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
            v_full = repeat(v, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
            sim = torch.einsum('bhid,bhjd->bhij', q, k_full) * scale
            attn = F.softmax(sim, dim=-1)
            return torch.einsum('bhij,bhjd->bhid', attn, v_full)
        
        # Select top-k important blocks (per query position)
        # Normalize importance scores
        importance = F.pad(importance, (1, 0), value=-1e10)
        importance = F.softmax(importance, dim=-1)
        importance = importance[..., 1:]  # Remove padding
        
        selected_scores, selected_indices = importance.topk(num_selected, dim=-1)
        
        # Prepare K and V in block format
        fine_divisible_len = (seq_len // self.selection_block_size) * self.selection_block_size
        k_trunc = k[:, :, :fine_divisible_len, :]
        v_trunc = v[:, :, :fine_divisible_len, :]
        
        num_fine_blocks = fine_divisible_len // self.selection_block_size
        
        k_blocks = rearrange(
            k_trunc, 
            'b h (w bs) d -> b h w bs d', 
            bs=self.selection_block_size
        )
        v_blocks = rearrange(
            v_trunc, 
            'b h (w bs) d -> b h w bs d', 
            bs=self.selection_block_size
        )
        
        # Gather selected blocks
        # Expand selected_indices for gathering
        selected_indices_exp = repeat(
            selected_indices,
            'b h q sel -> b h q sel bs d',
            bs=self.selection_block_size,
            d=self.dim_head
        )
        
        # Expand k/v blocks for all query positions
        k_blocks_exp = repeat(k_blocks, 'b h w bs d -> b h q w bs d', q=seq_len)
        v_blocks_exp = repeat(v_blocks, 'b h w bs d -> b h q w bs d', q=seq_len)
        
        # Gather
        selected_k = k_blocks_exp.gather(3, selected_indices_exp)
        selected_v = v_blocks_exp.gather(3, selected_indices_exp)
        
        # Flatten selected blocks
        selected_k = rearrange(selected_k, 'b h q sel bs d -> b h q (sel bs) d')
        selected_v = rearrange(selected_v, 'b h q sel bs d -> b h q (sel bs) d')
        
        # Expand for GQA
        selected_k = repeat(selected_k, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        selected_v = repeat(selected_v, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        
        # Compute attention over selected blocks
        sim = torch.einsum('bhid,bhijd->bhij', q, selected_k) * scale
        
        # Apply causal mask for fine attention (CRITICAL!)
        # Need to ensure selected blocks don't contain future tokens
        # Each selected block has indices [block_idx * block_size, (block_idx+1) * block_size)
        
        # For each query position and each selected block, compute the block's start position
        # selected_indices: [b, h, q, num_selected] - which blocks are selected
        # We need to check if the block tokens are before the query position
        
        # Create causal mask based on block positions
        # For simplicity: a token at position i can attend to a block if the block starts at position < i
        num_selected_tokens = selected_k.shape[3]  # num_selected * block_size
        
        # Importance score mask (keep tokens with non-zero importance)
        importance_mask = repeat(
            selected_scores > 1e-10, 
            'b h q sel -> b (h g) q (sel bs)', 
            g=self.num_grouped_queries, 
            bs=self.selection_block_size
        )
        
        # Causal mask: For each query position, mask out tokens from selected blocks that are in the future
        # This is conservative: we compute per-token causal constraint
        # selected_indices: [b, h, q, num_selected] gives block indices
        # Convert to token positions and create causal mask
        
        # Get block start positions for selected blocks
        block_start_positions = selected_indices * self.selection_block_size  # [b, h, q, num_selected]
        
        # Expand to per-token positions within each block
        # Token j in block i is at position: block_start + j
        token_offsets = torch.arange(self.selection_block_size, device=q.device)  # [block_size]
        token_positions = block_start_positions.unsqueeze(-1) + token_offsets  # [b, h, q, sel, bs]
        token_positions = rearrange(token_positions, 'b h q sel bs -> b h q (sel bs)')  # [b, h, q, num_tokens]
        
        # Query positions
        query_positions = torch.arange(seq_len, device=q.device)  # [seq_len]
        query_positions = query_positions.view(1, 1, -1, 1)  # [1, 1, seq_len, 1]
        
        # Causal constraint: token_position <= query_position
        causal_mask = token_positions <= query_positions  # [b, h, q, num_tokens]
        
        # Expand for GQA
        causal_mask = repeat(causal_mask, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        
        # Combine importance and causal masks
        final_mask = importance_mask & causal_mask
        sim = sim.masked_fill(~final_mask, -1e10)
        
        attn = F.softmax(sim, dim=-1)
        out = torch.einsum('bhij,bhijd->bhid', attn, selected_v)
        
        return out
    
    def forward(
        self,
        hidden_states,
        q, k, v,              # From base model's attention
        attention_mask=None,
        debug_print=False,    # Enable dimension checking
    ):
        """
        Forward pass implementing Native Sparse Attention's three-branch architecture
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            q, k, v: Query, Key, Value from base model
                q: [batch, num_heads, seq_len, dim_head]
                k, v: [batch, num_kv_heads, seq_len, dim_head]
            attention_mask: Optional attention mask
            debug_print: If True, print dimension information for verification
        
        Returns:
            out: [batch, seq_len, hidden_size] - Same shape as Llama's attention output
        """
        batch_size, seq_len, _ = hidden_states.shape
        scale = self.dim_head ** -0.5
        
        # 1. Compressed attention branch
        # K and V are already [batch, num_kv_heads, seq_len, dim_head] from forward
        # No need to rearrange, they come directly from the model
        k_kv = k  # Already [batch, num_kv_heads, seq_len, dim_head]
        v_kv = v  # Already [batch, num_kv_heads, seq_len, dim_head]
        
        if debug_print:
            print(f"\n{'='*60}")
            print(f"Dimension Check - Compression Stage")
            print(f"{'='*60}")
            print(f"Original K shape: {k_kv.shape}")
            print(f"Original V shape: {v_kv.shape}")
            print(f"  -> [batch={k_kv.shape[0]}, kv_heads={k_kv.shape[1]}, seq_len={k_kv.shape[2]}, dim_head={k_kv.shape[3]}]")
        
        ck = self.k_compress(k_kv)
        cv = self.v_compress(v_kv)
        
        if debug_print:
            num_blocks = seq_len // self.compress_block_size
            compression_ratio = seq_len / ck.shape[2] if ck.shape[2] > 0 else float('inf')
            print(f"\nCompressed K shape: {ck.shape}")
            print(f"Compressed V shape: {cv.shape}")
            print(f"  -> [batch={ck.shape[0]}, kv_heads={ck.shape[1]}, num_blocks={ck.shape[2]}, dim_head={ck.shape[3]}]")
            print(f"\n✅ Compression successful!")
            print(f"  - Original sequence length: {seq_len}")
            print(f"  - Compressed to {ck.shape[2]} blocks")
            print(f"  - Compression ratio: {compression_ratio:.2f}x")
            print(f"  - Block size: {self.compress_block_size}")
            print(f"  - Memory saved: {(1 - 1/compression_ratio)*100:.1f}%")
            print(f"{'='*60}\n")
        
        compressed_out, importance_scores = self.compute_compressed_attention(
            q, ck, cv, scale, attention_mask
        )
        
        # 2. Fine attention branch (based on importance from compressed)
        fine_out = self.compute_fine_attention(
            q, k_kv, v_kv, importance_scores, scale
        )
        
        # 3. Sliding window branch
        # Each position attends to itself and the previous window_size-1 tokens
        # This is the standard causal sliding window attention
        
        # Create sliding window mask (vectorized)
        # Position i attends to positions in range [max(0, i-window_size+1), i]
        row_indices = torch.arange(seq_len, device=q.device)[:, None]  # [seq_len, 1]
        col_indices = torch.arange(seq_len, device=q.device)[None, :]  # [1, seq_len]
        
        # Causal: col <= row (only attend to past and present)
        # Window: col > row - window_size (not too far in the past)
        sliding_mask = (col_indices <= row_indices) & (col_indices > row_indices - self.sliding_window_size)
        
        # Compute sliding window attention
        k_full = repeat(k_kv, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        v_full = repeat(v_kv, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        
        sim = torch.einsum('bhid,bhjd->bhij', q, k_full) * scale
        
        # Apply sliding window mask
        sim = sim.masked_fill(~sliding_mask.unsqueeze(0).unsqueeze(0), -1e10)
        
        # Apply attention_mask if provided (mask out padding tokens)
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] with 1 for real tokens, 0 for padding
            # Expand to [batch, 1, 1, seq_len] for broadcasting
            expanded_mask = attention_mask[:, None, None, :].bool()
            sim = sim.masked_fill(~expanded_mask, -1e10)
        
        attn = F.softmax(sim, dim=-1)
        sliding_out = torch.einsum('bhij,bhjd->bhid', attn, v_full)
        
        # 4. Combine three branches with learned weights (following Native Sparse Attention)
        strategy_logits = self.strategy_combiner(hidden_states)  # [batch, seq_len, 3 * num_heads]
        strategy_logits = rearrange(
            strategy_logits,
            'b n (h s) -> b h n s',
            h=self.num_heads,
            s=3
        )
        
        # Apply softmax to ensure weights sum to 1 (following NSA)
        strategy_weights = F.softmax(strategy_logits, dim=-1)  # [batch, num_heads, seq_len, 3]
        
        # Stack outputs: [3, batch, num_heads, seq_len, dim_head]
        stacked_outputs = torch.stack([compressed_out, fine_out, sliding_out], dim=0)
        
        # Weighted combination: [batch, num_heads, seq_len, dim_head]
        combined = torch.einsum('sbhnd,bhns->bhnd', stacked_outputs, strategy_weights)
        
        # 5. Merge heads and combine (like Native Sparse Attention)
        # [batch, num_heads, seq_len, dim_head] -> [batch, seq_len, num_heads * dim_head]
        combined = rearrange(combined, 'b h n d -> b n (h d)')
        
        # [batch, seq_len, num_heads * dim_head] -> [batch, seq_len, hidden_size]
        out = self.combine_heads(combined)
        
        return out  # [batch, seq_len, hidden_size] ✅


class LlamaWithSparseAttention(nn.Module):
    """
    Llama 3.2 1B with trainable sparse attention adapters
    Base model is frozen, only sparse attention modules are trainable
    """
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-1B",
        sparse_attn_config=None,
        device_map='auto',
    ):
        super().__init__()
        
        # Default sparse attention config
        if sparse_attn_config is None:
            sparse_attn_config = {
                'compress_block_size': 16,
                'compress_stride': 8,
                'selection_block_size': 16,
                'num_selected_blocks': 8,
                'sliding_window_size': 64,
                'k_compress_method': 'mean_pool',
                'v_compress_method': 'mean_pool',  # mean_pool is more stable for distillation
            }
        
        # Load frozen base model (use device_map='auto' like softCoT)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map if device_map else 'auto',
        )
        
        # Enable gradient checkpointing to save activation memory
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
            print(f"✅ Enabled gradient checkpointing for memory efficiency")
        
        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        print(f"✅ Loaded and froze base model: {model_id}")
        
        # Get config
        config = self.base_model.config
        
        # Create sparse attention adapters for each layer
        # Pass teacher's o_proj weight for better initialization
        self.sparse_adapters = nn.ModuleList([
            SparseAttentionAdapter(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                teacher_o_proj_weight=self.base_model.model.layers[layer_idx].self_attn.o_proj.weight.data.clone(),
                **sparse_attn_config
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        print(f"✅ Added {len(self.sparse_adapters)} sparse attention adapters (initialized from teacher's o_proj)")
        
        # Move adapters to the same device as base model and save device
        try:
            self.device = next(self.base_model.parameters()).device
        except StopIteration:
            # Fallback if base_model has no parameters (shouldn't happen)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.sparse_adapters = self.sparse_adapters.to(self.device)
        print(f"✅ Moved sparse adapters to {self.device}")
        
        # Store config
        self.config = config
        self.sparse_attn_config = sparse_attn_config
        self.num_grouped_queries = config.num_attention_heads // config.num_key_value_heads
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs
    ):
        """
        Forward pass with sparse attention fully integrated
        
        This replaces the base model's attention with sparse attention adapters.
        Uses the same loss as Native Sparse Attention: standard cross-entropy.
        """
        # Ensure inputs are on the correct device (same as model)
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)
        
        batch_size, seq_len = input_ids.shape
        device = self.device
        
        # 1. Embedding (frozen, but keep gradient flow for adapters)
        hidden = self.base_model.model.embed_tokens(input_ids)
        
        # Collect hidden states if requested (for distillation)
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden,)
        
        # 2. Process through each layer with sparse attention
        for layer_idx in range(len(self.base_model.model.layers)):
            layer = self.base_model.model.layers[layer_idx]
            residual = hidden
            
            # 2.1 Input layer norm (frozen, but keep gradient flow)
            normed = layer.input_layernorm(hidden)
            
            # 2.2 Get Q, K, V from frozen projections (keep gradient flow)
            q_proj = layer.self_attn.q_proj(normed)
            k_proj = layer.self_attn.k_proj(normed)
            v_proj = layer.self_attn.v_proj(normed)
            
            # Reshape to [batch, heads, seq_len, dim_head]
            dim_head = self.config.hidden_size // self.config.num_attention_heads
            
            q = q_proj.view(batch_size, seq_len, self.config.num_attention_heads, dim_head)
            k = k_proj.view(batch_size, seq_len, self.config.num_key_value_heads, dim_head)
            v = v_proj.view(batch_size, seq_len, self.config.num_key_value_heads, dim_head)
            
            q = q.transpose(1, 2)  # [batch, heads, seq_len, dim_head]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # 2.3 Apply trainable sparse adapter
            # This is where the trainable parameters are!
            # Adapter returns [batch, seq_len, hidden_size] directly (like NSA)
            attn_output = self.sparse_adapters[layer_idx](
                normed, q, k, v,
                attention_mask=attention_mask,
                debug_print=False
            )  # [batch, seq_len, hidden_size] ✅
            
            # 2.4 Residual connection (directly, like NSA!)
            hidden = residual + attn_output
            
            # 2.5 Post-attention (frozen, no grad needed for MLP)
            residual = hidden
            with torch.no_grad():
                normed = layer.post_attention_layernorm(hidden)
                mlp_out = layer.mlp(normed)
            
            # 2.6 Residual (detach MLP output to save memory)
            hidden = residual + mlp_out.detach()
            
            # Collect hidden state after this layer (for distillation)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)
        
        # 3. Final norm and LM head (need gradient for loss!)
        # Don't use no_grad here - we need gradient flow to loss
        hidden = self.base_model.model.norm(hidden)
        logits = self.base_model.lm_head(hidden)
        
        # 4. Compute loss (SAME AS NATIVE SPARSE ATTENTION!)
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Standard cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Return in transformers format
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,  # Return collected hidden states
            attentions=None,
        )
    
    def get_trainable_parameters(self):
        """Return only trainable parameters (sparse adapters)"""
        trainable_params = []
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append((name, param))
        
        trainable_count = sum(p.numel() for _, p in trainable_params)
        
        print(f"\n📊 Parameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Percentage trainable: {100 * trainable_count / total_params:.2f}%")
        
        return [p for _, p in trainable_params]
    
    def save_pretrained(self, save_dir, **kwargs):
        """
        Custom save for Trainer compatibility - only save trainable adapters
        This is called by Trainer during checkpointing
        """
        self.save_adapters(save_dir)
    
    def save_adapters(self, save_dir):
        """Save only the trainable sparse attention adapters"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        adapter_state = {
            'sparse_adapters': self.sparse_adapters.state_dict(),
            'config': self.sparse_attn_config,
        }
        
        save_path = os.path.join(save_dir, 'sparse_adapters.pt')
        torch.save(adapter_state, save_path)
        print(f"💾 Saved sparse attention adapters to {save_path} (excluding frozen base model)")
    
    def load_adapters(self, save_dir):
        """Load trained sparse attention adapters"""
        import os
        load_path = os.path.join(save_dir, 'sparse_adapters.pt')
        
        checkpoint = torch.load(load_path, map_location='cpu')
        self.sparse_adapters.load_state_dict(checkpoint['sparse_adapters'])
        
        print(f"📂 Loaded sparse attention adapters from {load_path}")
    
    def verify_gradient_flow(self, input_ids, labels):
        """
        Verify that gradients flow correctly to sparse adapters
        This uses the SAME LOSS as Native Sparse Attention!
        """
        self.train()
        
        logger.info("🔍 Verifying gradient flow (using Native Sparse Attention's loss)...")
        
        # Forward pass - standard cross-entropy
        outputs = self(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        logger.info(f"✅ Loss computed: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients on adapters (should have gradients!)
        adapter_grads = []
        for i, adapter in enumerate(self.sparse_adapters):
            grad_norm = 0.0
            param_count = 0
            
            for name, param in adapter.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    param_count += 1
            
            adapter_grads.append((i, grad_norm, param_count))
            if i < 3 or i >= len(self.sparse_adapters) - 3:  # Show first and last 3
                logger.info(f"  Layer {i}: grad_norm={grad_norm:.6f}, params_with_grad={param_count}")
        
        # Check base model (should NOT have gradients!)
        base_grads = 0
        for param in self.base_model.parameters():
            if param.grad is not None:
                base_grads += param.grad.norm().item()
        
        if base_grads > 0:
            logger.warning(f"⚠️  Base model has gradients ({base_grads:.6f})! This should not happen.")
        else:
            logger.info("✅ Base model correctly frozen (no gradients)")
        
        # Zero gradients
        self.zero_grad()
        
        return adapter_grads
    
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=50,
        temperature=1.0,
        do_sample=False,
        top_k=None,
        top_p=None,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        """
        Generate method using sparse attention
        
        This implements autoregressive generation with sparse attention adapters.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (True) or greedy (False)
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            pad_token_id: Pad token ID
            eos_token_id: EOS token ID
        
        Returns:
            generated_ids: [batch, seq_len + generated_length]
        """
        self.eval()
        
        # Ensure inputs are on correct device
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)
        
        batch_size, cur_len = input_ids.shape
        
        # Initialize attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Set default pad/eos tokens
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id or 0
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # Track which sequences are done
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        # Autoregressive generation loop
        for _ in range(max_new_tokens):
            # Forward pass with current sequence (uses sparse attention!)
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            
            # Get logits for next token (last position)
            next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample or greedy
            if do_sample:
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update sequences
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.device)
            ], dim=-1)
            
            # Check for EOS
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
        
        return input_ids


if __name__ == "__main__":
    # Test model creation
    print("Testing Sparse Attention Model Creation...\n")
    
    model = LlamaWithSparseAttention(
        model_id="meta-llama/Llama-3.2-1B",
        sparse_attn_config={
            'compress_block_size': 16,
            'selection_block_size': 16,
            'num_selected_blocks': 4,
            'k_compress_method': 'max_pool',
            'v_compress_method': 'mlp',
        }
    )
    
    model.get_trainable_parameters()
    
    print("\n✅ Model created successfully!")

