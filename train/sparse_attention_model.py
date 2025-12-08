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


class MaxPoolCompress(nn.Module):
    """
    Coarse-grained compression using max pooling
    """
    def __init__(self, num_heads, dim_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        
        # Learnable intra-block positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(num_heads, block_size, dim_head))
        
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
        
        # Add positional information
        kv_blocks = kv_blocks + self.pos_emb.unsqueeze(0).unsqueeze(2)
        
        # Max pooling over block dimension
        compressed = kv_blocks.max(dim=3)[0]  # [b, h, n_blocks, d]
        
        return compressed


class MLPCompress(nn.Module):
    """
    Learnable MLP-based compression (similar to Native Sparse Attention)
    """
    def __init__(self, num_heads, dim_head, block_size, expand_factor=1.0):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        
        dim_in = block_size * dim_head
        dim_hidden = max(int(dim_in * expand_factor), dim_head)
        
        # Intra-block positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(num_heads, block_size, dim_head))
        
        # Per-head compression MLPs
        self.compress_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in, dim_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_head, bias=False)
            ) for _ in range(num_heads)
        ])
        
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
        
        # Add positional information
        kv_blocks = kv_blocks + self.pos_emb.unsqueeze(0).unsqueeze(2)
        
        # Flatten block dimension: [b, h, n_blocks, block_size * d]
        kv_flat = kv_blocks.reshape(b, h, n_blocks, -1)
        
        # Apply per-head MLPs
        compressed = []
        for head_idx in range(h):
            head_compressed = self.compress_mlps[head_idx](kv_flat[:, head_idx])
            compressed.append(head_compressed)
        
        compressed = torch.stack(compressed, dim=1)  # [b, h, n_blocks, d]
        
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
        k_compress_method='max_pool',  # 'max_pool' or 'mlp'
        v_compress_method='max_pool',  # 'max_pool' or 'mlp' (changed to max_pool for efficiency)
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
        
        # K compression (max pooling by default for efficiency)
        if k_compress_method == 'max_pool':
            self.k_compress = MaxPoolCompress(
                num_kv_heads, self.dim_head, compress_block_size
            )
        else:
            self.k_compress = MLPCompress(
                num_kv_heads, self.dim_head, compress_block_size, expand_factor=1.0
            )
        
        # V compression (max pooling by default for efficiency)
        if v_compress_method == 'max_pool':
            self.v_compress = MaxPoolCompress(
                num_kv_heads, self.dim_head, compress_block_size
            )
        else:
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
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            q, k, v: Query, Key, Value from base model [batch, num_heads, seq_len, dim_head]
            attention_mask: Optional attention mask
            debug_print: If True, print dimension information for verification
        
        Returns:
            Weighted combination of three attention branches
        """
        batch_size, seq_len, _ = hidden_states.shape
        scale = self.dim_head ** -0.5
        
        # 1. Compressed attention branch
        # Compress K and V (only use KV heads)
        k_kv = rearrange(
            k, 
            'b (h g) n d -> b h n d', 
            g=self.num_grouped_queries
        )[:, :self.num_kv_heads]
        
        v_kv = rearrange(
            v,
            'b (h g) n d -> b h n d',
            g=self.num_grouped_queries
        )[:, :self.num_kv_heads]
        
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
        
        # Weighted combination
        combined = torch.einsum('sbhnd,bhns->bhnd', stacked_outputs, strategy_weights)
        
        return combined


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
                'num_selected_blocks': 4,
                'sliding_window_size': 64,
                'k_compress_method': 'max_pool',
                'v_compress_method': 'max_pool',  # Changed to max_pool for efficiency
            }
        
        # Load frozen base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        
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
        
        # Store config
        self.config = config
        self.sparse_attn_config = sparse_attn_config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        """
        Forward pass with sparse attention
        
        Note: This is a simplified version. For full integration, you'd need to:
        1. Hook into each layer's attention computation
        2. Replace/augment with sparse attention
        3. Handle caching for generation
        """
        # For now, just use base model
        # TODO: Implement full sparse attention integration
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
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
        print(f"💾 Saved sparse attention adapters to {save_path}")
    
    def load_adapters(self, save_dir):
        """Load trained sparse attention adapters"""
        import os
        load_path = os.path.join(save_dir, 'sparse_adapters.pt')
        
        checkpoint = torch.load(load_path, map_location='cpu')
        self.sparse_adapters.load_state_dict(checkpoint['sparse_adapters'])
        
        print(f"📂 Loaded sparse attention adapters from {load_path}")


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

