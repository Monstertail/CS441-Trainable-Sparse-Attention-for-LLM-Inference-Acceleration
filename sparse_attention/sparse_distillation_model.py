"""
Sparse Attention Distillation Model 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from .sparse_attention_model import LlamaWithSparseAttention
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
        loss_mode="kl_logits",  # 'kl_logits', 'mse_hidden', or 'mixed'
        temperature=2.0,  # Temperature for KL divergence (standard distillation value)
        **kwargs
    ):
        super().__init__()
        
        self.loss_mode = loss_mode
        self.temperature = temperature
        
        logger.info(f"🔧 Creating Sparse Distillation Model (optimized - shared base model)")
        logger.info(f"  Loss mode: {loss_mode}")
        logger.info(f"  Temperature: {temperature}")
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
                use_cache=False,  # Disable KV cache for training (critical!)
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
            use_cache=False,  # Disable KV cache for training (critical!)
            return_dict=True,
        )
        student_hidden_states = student_outputs.hidden_states
        student_logits = student_outputs.logits
        
        # Sanity check: Ensure student returned hidden states
        if student_hidden_states is None:
            raise ValueError("Student model did not return hidden_states! Check output_hidden_states=True")
        
        # 3. Compute distillation loss based on loss_mode
        distill_loss = 0.0
        
        if self.training:
            if self.loss_mode == "kl_output_only":
                # KL Divergence ONLY on output tokens (labels != -100)
                # Benefits:
                # - Focused on the part that matters (output)
                # - Uses soft targets from teacher (better than hard labels)
                # - No wasted computation on prompt tokens
                
                if labels is None:
                    raise ValueError("kl_output_only mode requires labels to be provided!")
                
                # Shift for next-token prediction
                shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
                shift_student_logits = student_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Create mask for output tokens only (where labels != -100)
                mask = (shift_labels != -100)  # [batch, seq_len]
                
                if mask.sum() == 0:
                    distill_loss = torch.tensor(0.0, device=student_logits.device)
                else:
                    # Apply temperature scaling
                    teacher_logits_scaled = shift_teacher_logits / self.temperature
                    student_logits_scaled = shift_student_logits / self.temperature
                    
                    # Compute log probabilities
                    teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)
                    student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
                    
                    # KL divergence per token (no reduction yet)
                    # kl_per_token = sum over vocab: exp(teacher_log) * (teacher_log - student_log)
                    kl_per_token = F.kl_div(
                        student_log_probs.view(-1, student_log_probs.size(-1)),
                        teacher_log_probs.view(-1, teacher_log_probs.size(-1)),
                        log_target=True,
                        reduction='none'
                    ).sum(dim=-1).view(shift_labels.shape)  # [batch, seq_len]
                    
                    # Mask out prompt tokens (only keep output)
                    masked_kl = kl_per_token * mask.float()
                    distill_loss = (masked_kl.sum() / mask.sum()) * (self.temperature ** 2)
                    
                    # Clamp to prevent explosion (use lower threshold for output-only)
                    distill_loss = torch.clamp(distill_loss, max=20.0)
                
            elif self.loss_mode == "kl_logits":
                # KL Divergence loss on final logits (standard knowledge distillation)
                # This is the most common and effective distillation method
                # Only looks at the final output, not intermediate layers
                
                # Apply temperature scaling for soft targets
                teacher_logits_scaled = teacher_logits / self.temperature
                student_logits_scaled = student_logits / self.temperature
                
                # Compute log probabilities
                teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)
                student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
                
                # KL divergence: KL(teacher || student)
                # = sum(teacher_probs * (log(teacher_probs) - log(student_probs)))
                distill_loss = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    log_target=True,
                    reduction='batchmean'
                ) * (self.temperature ** 2)  # Scale by T^2 (standard in distillation)
                
                # Clamp loss to prevent explosion at initialization
                # KL divergence can be very large initially if student output differs significantly
                distill_loss = torch.clamp(distill_loss, max=100.0)
                
            elif self.loss_mode == "mse_hidden":
                # Layer-wise MSE loss on hidden states (previous method)
                num_layers = 0
                
                if student_hidden_states is not None:
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
                        
                        distill_loss += layer_loss
                        num_layers += 1
                
                # Average distillation loss across all layers
                if num_layers > 0:
                    distill_loss = distill_loss / num_layers
                    
            elif self.loss_mode == "mixed":
                # Mixed: KL on logits + MSE on last hidden layer
                # KL divergence on logits (weight 0.7)
                teacher_logits_scaled = teacher_logits / self.temperature
                student_logits_scaled = student_logits / self.temperature
                
                teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)
                student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
                
                kl_loss = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    log_target=True,
                    reduction='batchmean'
                ) * (self.temperature ** 2)
                
                # MSE on last hidden layer (weight 0.3)
                mse_loss = 0.0
                if student_hidden_states is not None and len(student_hidden_states) > 0:
                    teacher_h = teacher_hidden_states[-1]
                    student_h = student_hidden_states[-1]
                    
                    s_norm = self.ln_stat(student_h)
                    t_norm = self.ln_stat(teacher_h.detach())
                    
                    mse_loss = F.mse_loss(s_norm, t_norm)
                
                distill_loss = 0.7 * kl_loss + 0.3 * mse_loss
            
            else:
                raise ValueError(f"Unknown loss_mode: {self.loss_mode}. Choose from: kl_output_only, kl_logits, mse_hidden, mixed")
        
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
            # For distillation modes, use ONLY distillation loss
            # CE loss is computed but not used (kept for potential future use)
            if self.loss_mode in ["kl_output_only", "kl_logits", "mse_hidden"]:
                loss = distill_loss
            elif self.loss_mode == "mixed":
                # Mixed mode: combine distillation + CE
                if ce_loss is not None:
                    loss = 0.7 * distill_loss + 0.3 * ce_loss
                else:
                    loss = distill_loss
            else:
                # Default: pure distillation
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
            output.loss_mode = self.loss_mode
            if ce_loss is not None:
                output.ce_loss = ce_loss.item()
        
        return output
    
    def save_pretrained(self, save_dir, **kwargs):
        """
        Custom save for Trainer compatibility - only save trainable student adapters
        This is called by Trainer during checkpointing (every epoch)
        """
        self.save_student(save_dir)
    
    def save_student(self, save_dir):
        """Save only the trainable sparse adapters from student model"""
        self.student_model.save_adapters(save_dir)
        logger.info(f"💾 Saved student adapters to {save_dir}")
    
    def load_student(self, save_dir):
        """Load pre-trained sparse adapters into student model"""
        self.student_model.load_adapters(save_dir)
        logger.info(f"📂 Loaded student adapters from {save_dir}")