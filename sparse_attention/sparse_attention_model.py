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

# Use the in-repo copy of Native Sparse Attention (under sparse_attention/native_sparse_attention_pytorch)
from .native_sparse_attention_pytorch.native_sparse_attention import SparseAttention


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
            # Preserve input dtype
            input_dtype = x.dtype
            norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            # Ensure scale has same dtype as input
            return (x * norm * self.scale.to(input_dtype)).to(input_dtype)


class SparseAttentionAdapter(nn.Module):
    """
    Thin wrapper that uses the ORIGINAL Native Sparse Attention implementation
    (`native_sparse_attention_pytorch.SparseAttention`) as a per-layer adapter.

    - We feed Llama's pre-attention normalized hidden states into `SparseAttention`
    - We initialize `to_qkv` and `combine_heads` from the frozen Llama teacher layer
    - All sparse patterns, masks, and three-branch logic are delegated to the NSA code
    """

    def __init__(self, config, sparse_attn_config, teacher_layer):
        super().__init__()

        dim = config.hidden_size
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        dim_head = dim // heads

        sliding_window_size = sparse_attn_config.get("sliding_window_size", 64)
        compress_block_size = sparse_attn_config.get("compress_block_size", 16)
        compress_stride = sparse_attn_config.get("compress_stride", compress_block_size)
        selection_block_size = sparse_attn_config.get("selection_block_size", 16)
        num_selected_blocks = sparse_attn_config.get("num_selected_blocks", 4)

        # Use the original NSA module, but disable its internal norm since we
        # already apply Llama's `input_layernorm` before calling this adapter.
        self.sparse_attn = SparseAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            sliding_window_size=sliding_window_size,
            compress_block_size=compress_block_size,
            compress_block_sliding_stride=compress_stride,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks,
            kv_heads=kv_heads,
            num_compressed_mem_kv=1,
            causal=True,
            norm=False,                 # we use Llama's layernorm outside
            use_diff_topk=False,
            use_triton_kernel=False,
            query_heads_share_selected_kv=True,
            compress_mlp=None,          # let NSA construct default MLP
            compress_mlp_expand_factor=1.0,
            strategy_combine_mlp=None,  # let NSA construct default combiner
        )

        # Initialize NSA qkv and combine_heads from the frozen Llama teacher
        with torch.no_grad():
            q_proj = teacher_layer.self_attn.q_proj
            k_proj = teacher_layer.self_attn.k_proj
            v_proj = teacher_layer.self_attn.v_proj
            o_proj = teacher_layer.self_attn.o_proj

            dim_inner = dim_head * heads
            dim_kv_inner = dim_head * kv_heads

            assert q_proj.weight.shape == (dim_inner, dim)
            assert k_proj.weight.shape == (dim_kv_inner, dim)
            assert v_proj.weight.shape == (dim_kv_inner, dim)
            assert self.sparse_attn.to_qkv.weight.shape[0] == dim_inner + 2 * dim_kv_inner

            # Stack teacher q / k / v into NSA's single Linear
            qkv_weight = self.sparse_attn.to_qkv.weight
            qkv_weight[:dim_inner, :] = q_proj.weight.data
            qkv_weight[dim_inner:dim_inner + dim_kv_inner, :] = k_proj.weight.data
            qkv_weight[dim_inner + dim_kv_inner:, :] = v_proj.weight.data

            # Output projection
            assert self.sparse_attn.combine_heads.weight.shape == o_proj.weight.shape
            self.sparse_attn.combine_heads.weight.copy_(o_proj.weight.data)

        print("✅ Initialized NSA SparseAttention adapter from teacher layer weights")

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]  (already input_layernorm-ed)
            attention_mask: currently unused (NSA handles causal masking internally)

        Returns:
            out: [batch, seq_len, hidden_size]
        """
        # Native Sparse Attention does not take a standard attention_mask tensor
        # for padding; for now we rely on training data being trimmed properly.
        # If needed, we can later extend this by using flex attention masks.
        return self.sparse_attn(hidden_states)


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
        
        # Create sparse attention adapters for each layer, using ORIGINAL NSA implementation
        self.sparse_adapters = nn.ModuleList([
            SparseAttentionAdapter(
                config=config,
                sparse_attn_config=sparse_attn_config,
                teacher_layer=self.base_model.model.layers[layer_idx],
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        print(f"✅ Added {len(self.sparse_adapters)} NSA-based sparse attention adapters (initialized from teacher attn)")
        
        # Move adapters to the same device and dtype as base model and save them
        try:
            base_param = next(self.base_model.parameters())
            self.device = base_param.device
            self.dtype = base_param.dtype
        except StopIteration:
            # Fallback if base_model has no parameters (shouldn't happen)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Ensure all adapter parameters match base model's dtype (e.g., bfloat16)
        self.sparse_adapters = self.sparse_adapters.to(device=self.device, dtype=self.dtype)
        print(f"✅ Moved sparse adapters to {self.device} with dtype {self.dtype}")
        
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

            # 2.2 Apply NSA-based trainable sparse adapter
            # This replaces the original self-attention block with Native Sparse Attention
            attn_output = self.sparse_adapters[layer_idx](
                normed,
                attention_mask=attention_mask,
            )  # [batch, seq_len, hidden_size]
            
            # 2.3 Residual connection
            hidden = residual + attn_output
            
            # 2.4 Post-attention (frozen, no grad needed for MLP)
            residual = hidden
            with torch.no_grad():
                normed = layer.post_attention_layernorm(hidden)
                mlp_out = layer.mlp(normed)
            
            # 2.5 Residual (detach MLP output to save memory)
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

