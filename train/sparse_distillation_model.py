"""
Sparse Attention Distillation Model 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from sparse_attention_model import LlamaWithSparseAttention
from fastNLP import logger


class SparseDistillationModel(nn.Module):
    """
    Dual-model distillation architecture for sparse attention training
    
    - teacher: Full attention Llama model (frozen, for reference)
    - student: Llama with sparse attention adapters (trainable)
    """
    
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-1B",
        sparse_attn_config=None,
        teacher_device="auto",  # Kept for compatibility, but ignored (use device_map='auto')
        student_device="auto",  # Kept for compatibility, but ignored (use device_map='auto')
        **kwargs
    ):
        super().__init__()
        
        logger.info(f"🔧 Creating Sparse Distillation Model (optimized - shared base model)")
        logger.info(f"  Using device_map='auto'")
        
        # Load student model with sparse adapters (contains base model)
        logger.info(f"📦 Loading model with sparse attention adapters...")
        
        self.student_model = LlamaWithSparseAttention(
            model_id=model_id,
            sparse_attn_config=sparse_attn_config,
            device_map='auto',
        )
        
        logger.info(f"✅ Student model loaded")
        
        # Teacher = base model without adapters (share the same base model!)
        # No need to load twice - just use student's base model directly
        self.base_model = self.student_model.base_model
        logger.info(f"✅ Teacher uses shared base model (no duplicate loading)")
        
        # Load model config
        self.config = AutoConfig.from_pretrained(model_id)
        
        # Get actual device
        self.device = self.base_model.device
        logger.info(f"📍 Models placed on device: {self.device}")
        
        # Statistical LayerNorm (no learnable params) for stable loss computation
        self.ln_stat = nn.LayerNorm(
            self.config.hidden_size, 
            elementwise_affine=False  # No learnable parameters, just normalization
        ).to(self.device)
        logger.info(f"✅ Added statistical LayerNorm for stable distillation loss")
        
        # Display trainable parameter statistics
        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        """Print trainable parameter statistics"""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        adapter_params = sum(p.numel() for p in self.student_model.sparse_adapters.parameters())
        
        logger.info(f"\n📊 Parameter Statistics:")
        logger.info(f"  Base model (frozen, shared): {base_params:,}")
        logger.info(f"  Sparse adapters (trainable): {adapter_params:,}")
        logger.info(f"  Total: {base_params + adapter_params:,}")
        logger.info(f"  Trainable ratio: {100 * adapter_params / (base_params + adapter_params):.2f}%")
        logger.info(f"  Memory saved: ~{base_params * 2 / 1e9:.2f} GB (by sharing base model)")
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
        **kwargs
    ):
        """
        Forward pass with layer-wise distillation
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Token labels for optional CE loss [batch, seq_len]
        
        Returns:
            CausalLMOutputWithPast with distillation loss
        """
        batch_size, seq_len = input_ids.shape
        
        # No need to move inputs - Trainer handles device placement automatically
        
        # 1. Teacher forward pass (use shared base model, no adapters)
        with torch.no_grad():
            teacher_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,  # Get intermediate layer outputs
                return_dict=True,
            )
            teacher_hidden_states = teacher_outputs.hidden_states  # Tuple of (num_layers+1) tensors
            teacher_logits = teacher_outputs.logits
        
        # 2. Student forward pass (trainable adapters, independent computation)
        # Key: Student processes input independently, like during inference
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Don't compute CE loss inside student
            output_hidden_states=True,  # Get intermediate layer outputs
            return_dict=True,
        )
        student_hidden_states = student_outputs.hidden_states
        student_logits = student_outputs.logits
        
        # Sanity check: Ensure student returned hidden states
        if student_hidden_states is None:
            raise ValueError("Student model did not return hidden_states! Check output_hidden_states=True")
        
        # 3. Compute layer-wise distillation loss (MSE between hidden states)
        distill_loss = 0.0
        num_layers = 0
        
        if student_hidden_states is not None and self.training:
            # Compare output of each layer (teacher vs student)
            # hidden_states[0] = embedding output
            # hidden_states[i+1] = output of layer i (after attention + MLP)
            num_compare_layers = min(len(teacher_hidden_states), len(student_hidden_states))
            
            for layer_idx in range(1, num_compare_layers):  # Skip embedding (idx=0)
                # Both on same device, no need to move
                teacher_h = teacher_hidden_states[layer_idx]
                student_h = student_hidden_states[layer_idx]
                
                # Apply statistical LayerNorm before computing loss
                # This normalizes the scale and makes training more stable
                s_norm = self.ln_stat(student_h)
                t_norm = self.ln_stat(teacher_h.detach())
                
                # MSE loss on normalized hidden states
                layer_loss = F.mse_loss(s_norm, t_norm)
                
                # Alternative: Cosine similarity loss (commented out)
                # cos_sim = F.cosine_similarity(
                #     student_h.view(-1, student_h.size(-1)),
                #     teacher_h.detach().view(-1, teacher_h.size(-1)),
                #     dim=-1
                # )
                # layer_loss = (1 - cos_sim).mean()
                
                distill_loss += layer_loss
                num_layers += 1
        
        # Average distillation loss across all layers
        if num_layers > 0:
            distill_loss = distill_loss / num_layers
        
        # 4. Optional: Add cross-entropy loss if labels are provided
        ce_loss = None
        if labels is not None:
            # Labels already on correct device (handled by Trainer)
            # Shift logits and labels for next-token prediction
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # 5. Combine losses based on training mode
        if self.training:
            if ce_loss is not None:
                # Mixed training: distillation + cross-entropy
                loss = 0.5 * distill_loss + 0.5 * ce_loss
            else:
                # Pure distillation (default)
                loss = distill_loss
        else:
            # Evaluation mode: use CE loss if available
            loss = ce_loss if ce_loss is not None else None
        
        # 6. Prepare output in transformers format
        from transformers.modeling_outputs import CausalLMOutputWithPast
        
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=student_logits,
            past_key_values=None,
            hidden_states=student_hidden_states,
            attentions=None,
        )
        
        # Attach loss components for logging (if training)
        if self.training:
            output.distill_loss = distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
            if ce_loss is not None:
                output.ce_loss = ce_loss.item()
        
        return output
    
    def save_student(self, save_dir):
        """Save only the trainable sparse adapters from student model"""
        self.student_model.save_adapters(save_dir)
        logger.info(f"💾 Saved student adapters to {save_dir}")
    
    def load_student(self, save_dir):
        """Load pre-trained sparse adapters into student model"""
        self.student_model.load_adapters(save_dir)
        logger.info(f"📂 Loaded student adapters from {save_dir}")