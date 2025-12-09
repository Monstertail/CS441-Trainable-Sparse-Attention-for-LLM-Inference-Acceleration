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
        
        # Strategy combiner: learn to weight three branches
        self.strategy_combiner = nn.Sequential(
            nn.Linear(hidden_size, 3 * num_heads),
            nn.Sigmoid()
        )
        
        # Initialize to favor sliding window initially
        nn.init.zeros_(self.strategy_combiner[0].weight)
        self.strategy_combiner[0].bias.data.copy_(
            torch.tensor([-2., -2., 2.] * num_heads)  # [compressed, fine, sliding]
        )
        
        # Merge and combine heads (like Native Sparse Attention)
        # merge_heads: [b, h, n, d] -> [b, n, h*d]
        # combine_heads: [b, n, h*d] -> [b, n, hidden_size]
        self.combine_heads = nn.Linear(num_heads * self.dim_head, hidden_size, bias=False)
        
        # Initialize combine_heads to near-identity for stable training
        # Start with small random weights so adapter output ≈ 0 initially
        nn.init.normal_(self.combine_heads.weight, mean=0.0, std=0.02)
    
    def compute_compressed_attention(
        self,
        q,     # [batch, num_heads, seq_len, dim_head]
        ck,    # compressed k [batch, num_kv_heads, num_blocks, dim_head]
        cv,    # compressed v [batch, num_kv_heads, num_blocks, dim_head]
        scale
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
        
        # Mask out zero scores
        mask = repeat(selected_scores > 1e-10, 'b h q sel -> b (h g) q (sel bs)', g=self.num_grouped_queries, bs=self.selection_block_size)
        sim = sim.masked_fill(~mask, -1e10)
        
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
            q, ck, cv, scale
        )
        
        # 2. Fine attention branch (based on importance from compressed)
        fine_out = self.compute_fine_attention(
            q, k_kv, v_kv, importance_scores, scale
        )
        
        # 3. Sliding window branch (computed by base model's attention)
        # We'll use the original attention output as sliding window proxy
        # For true sliding window, you'd need to modify base model's attention_mask
        
        # Compute sliding window attention (simplified version)
        # Create sliding window mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        window_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=-self.sliding_window_size
        )
        sliding_mask = ~(causal_mask | window_mask)
        
        # Compute sliding window attention
        k_full = repeat(k_kv, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        v_full = repeat(v_kv, 'b h ... -> b (h g) ...', g=self.num_grouped_queries)
        
        sim = torch.einsum('bhid,bhjd->bhij', q, k_full) * scale
        sim = sim.masked_fill(~sliding_mask.unsqueeze(0).unsqueeze(0), -1e10)
        attn = F.softmax(sim, dim=-1)
        sliding_out = torch.einsum('bhij,bhjd->bhid', attn, v_full)
        
        # 4. Combine three branches with learned weights
        strategy_weights = self.strategy_combiner(hidden_states)  # [batch, seq_len, 3 * num_heads]
        strategy_weights = rearrange(
            strategy_weights,
            'b n (h s) -> b h n s',
            h=self.num_heads,
            s=3
        )
        
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
        self.sparse_adapters = nn.ModuleList([
            SparseAttentionAdapter(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                **sparse_attn_config
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        print(f"✅ Added {len(self.sparse_adapters)} sparse attention adapters")
        
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

